"""
multiseed.py - Closed-world multi-seed evaluation for statistical confidence.

Runs one or more closed-world models over several random seeds, then reports
per-seed metrics, mean +/- std, and paired significance tests. The script can be
placed either at the repository root or inside ReBiAt_code/; it resolves the
training scripts automatically.

Supported models:
  * bigru   -> train_resnet_bigru.py
  * varcnn  -> train_baselines.py --model varcnn
  * netclr  -> train_baselines.py --model netclr

Per (model, seed) it:
  1. trains via the existing train script (skipped if the checkpoint exists),
  2. reproduces the same test split with that seed,
  3. restores the checkpoint and computes accuracy, macro precision/recall/F1,
     and macro one-vs-rest AUC.

Outputs (in --out_dir):
  multiseed_per_seed.csv
  multiseed_summary.json
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def _resolve_code_dir() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve().parent
    if (here / "train_resnet_bigru.py").exists() and (here / "train_baselines.py").exists():
        return here
    code_dir = here / "ReBiAt_code"
    if (code_dir / "train_resnet_bigru.py").exists() and (code_dir / "train_baselines.py").exists():
        return code_dir
    raise FileNotFoundError(
        "Could not locate train_resnet_bigru.py and train_baselines.py. "
        "Place multiseed.py either in the repo root or inside ReBiAt_code/."
    )


CODE_DIR = _resolve_code_dir()
sys.path.insert(0, str(CODE_DIR))

from train_resnet_bigru import load_chronological_key, load_npz, make_loader, set_global_seed, split_chronological, split_stratified  # noqa: E402
import baselines as B  # noqa: E402


METRICS = ["accuracy", "precision", "recall", "f1", "auc"]
TRAIN = {
    "bigru": [str(CODE_DIR / "train_resnet_bigru.py")],
    "varcnn": [str(CODE_DIR / "train_baselines.py"), "--model", "varcnn"],
    "netclr": [str(CODE_DIR / "train_baselines.py"), "--model", "netclr"],
}
DEFAULT_BIGRU_BEST = {
    "lr": 8.3e-4,
    "batch_size": 64,
    "gru_hidden": 128,
    "dropout_enc": 0.24,
    "label_smoothing": 0.02,
}


def train_one(model: str, seed: int, rdir: str, args, extra: list[str]) -> pathlib.Path:
    ckpt = pathlib.Path(rdir) / "best_model.pt"
    if ckpt.exists():
        print(f"  SKIP train {model} seed={seed} (checkpoint exists)", flush=True)
        return ckpt

    cmd = [
        sys.executable,
        "-u",
        *TRAIN[model],
        "--npz",
        args.npz,
        "--results_dir",
        rdir,
        "--seed",
        str(seed),
        "--batch_size",
        str(args.batch_size),
        "--patience",
        str(args.patience),
        "--k_features",
        str(args.k_features),
        "--loss",
        "focal",
        "--early_stop_metric",
        "macro_f1",
        "--use_augment",
        *extra,
    ]
    if model == "bigru":
        cmd.extend(["--epochs", str(args.bigru_epochs)])
    elif model == "varcnn":
        cmd.extend(["--epochs", str(args.varcnn_epochs)])
    else:
        cmd.extend(
            [
                "--epochs",
                str(args.netclr_epochs),
                "--pretrain_epochs",
                str(args.netclr_pretrain_epochs),
                "--pretrain_bs",
                str(args.netclr_pretrain_bs),
                "--pretrain_lr",
                str(args.netclr_pretrain_lr),
                "--temperature",
                str(args.netclr_temperature),
                "--finetune_mode",
                args.netclr_finetune_mode,
            ]
        )
        if args.netclr_no_pretrain:
            cmd.append("--no_pretrain")

    if args.chrono_split:
        cmd.append("--chrono_split")

    print(f"\n  START train {model} seed={seed} ...", flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"  DONE  {model} seed={seed} in {time.time()-t0:.0f}s", flush=True)
    if not ckpt.exists():
        raise FileNotFoundError(f"Training finished but checkpoint is missing: {ckpt}")
    return ckpt


def metrics_one(model: str, seed: int, ckpt_path: pathlib.Path, args, device: torch.device) -> dict:
    cache = pathlib.Path(args.out_dir) / f"metrics_{model}_seed{seed}.json"
    if cache.exists():
        print(f"  SKIP eval  {model} seed={seed} (cached)", flush=True)
        return json.loads(cache.read_text())

    set_global_seed(seed)
    X_seq, X_gl, y, classes = load_npz(args.npz)
    if args.chrono_split:
        te = split_chronological(y, order_key=load_chronological_key(args.npz))[2]
    else:
        te = split_stratified(y)[2]
    n_classes = len(classes)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sel = np.asarray(ckpt["selected_idx"], dtype=np.int64)
    X_gl_te = ckpt["gl_scaler"].transform(X_gl[te][:, sel]).astype(np.float32)
    net = B.rebuild_from_ckpt(ckpt, device, eval_mode=True)
    loader = make_loader(X_seq[te], X_gl_te, y[te], args.batch_size, shuffle=False, num_workers=2)

    logits = []
    with torch.no_grad():
        for xb, gb, _ in loader:
            logits.append(net(xb.to(device), gb.to(device)).cpu())
    logits = torch.cat(logits)
    prob = torch.softmax(logits, dim=1).numpy()
    pred = prob.argmax(1)
    y_true = y[te]

    try:
        auc = float(
            roc_auc_score(
                y_true,
                prob,
                multi_class="ovr",
                average="macro",
                labels=np.arange(n_classes),
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN AUC failed for {model} seed={seed}: {exc}", flush=True)
        auc = float("nan")

    result = {
        "model": model,
        "seed": int(seed),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "auc": auc,
    }
    cache.write_text(json.dumps(result, indent=2))
    print(
        f"  {model:7} seed={seed:<6} acc={result['accuracy']:.4f} "
        f"f1={result['f1']:.4f} auc={result['auc']:.4f}",
        flush=True,
    )
    return result


def summarize(values: list[float]) -> dict:
    v = np.asarray([x for x in values if x == x], dtype=float)
    if len(v) == 0:
        return {"mean": None, "std": None, "n": 0, "min": None, "max": None}
    return {
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
        "n": int(len(v)),
        "min": float(v.min()),
        "max": float(v.max()),
    }


def paired_test(a_vals: list[float], b_vals: list[float]) -> dict:
    a = np.asarray(a_vals, dtype=float)
    b = np.asarray(b_vals, dtype=float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if len(a) < 2:
        return {"n_pairs": int(len(a)), "note": "need >=2 paired seeds"}

    diff = a - b
    out = {"n_pairs": int(len(a)), "mean_diff_a_minus_b": float(diff.mean())}
    t_stat, t_p = stats.ttest_rel(a, b)
    out["ttest_p"] = float(t_p)
    out["t_stat"] = float(t_stat)
    if np.any(diff != 0):
        try:
            out["wilcoxon_p"] = float(stats.wilcoxon(a, b).pvalue)
        except Exception:  # noqa: BLE001
            out["wilcoxon_p"] = None

    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(diff), size=(10000, len(diff)))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    out["ci95_mean_diff"] = [float(lo), float(hi)]
    return out


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--models", default="bigru,varcnn")
    ap.add_argument("--seeds", default="7,113,1009,2027,5051,9001,21013,44017,65537,99991")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--k_features", type=int, default=15)
    ap.add_argument("--chrono_split", action="store_true")
    ap.add_argument("--out_dir", default=str(pathlib.Path(".") / "multiseed"))

    ap.add_argument("--bigru_epochs", type=int, default=80)
    ap.add_argument("--bigru_hp_json", default=None,
                    help="JSON with fixed best params (flat dict or {'best_params': {...}}).")

    ap.add_argument("--varcnn_epochs", type=int, default=80)
    ap.add_argument("--varcnn_lr", type=float, default=1e-3)

    ap.add_argument("--netclr_epochs", type=int, default=50)
    ap.add_argument("--netclr_lr", type=float, default=1e-3)
    ap.add_argument("--netclr_pretrain_epochs", type=int, default=200)
    ap.add_argument("--netclr_pretrain_bs", type=int, default=256)
    ap.add_argument("--netclr_pretrain_lr", type=float, default=3e-4)
    ap.add_argument("--netclr_temperature", type=float, default=0.5)
    ap.add_argument("--netclr_finetune_mode", choices=["full", "linear"], default="full")
    ap.add_argument("--netclr_no_pretrain", action="store_true")
    return ap


def main() -> None:
    args, _ = build_argparser().parse_known_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    for model in models:
        if model not in TRAIN:
            raise ValueError(f"Unsupported model '{model}' (supported: {list(TRAIN)})")

    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Models={models}  Seeds={seeds}  device={device}", flush=True)
    print(f"Code dir={CODE_DIR}", flush=True)

    if args.bigru_hp_json:
        hp_path = pathlib.Path(args.bigru_hp_json)
        if not hp_path.exists():
            raise FileNotFoundError(f"--bigru_hp_json not found: {hp_path}")
    else:
        hp_path = pathlib.Path(args.out_dir) / "bigru_best_params.json"
        hp_path.write_text(json.dumps({"best_params": DEFAULT_BIGRU_BEST}, indent=2))
        print(f"BiGRU fixed best params -> {hp_path}: {DEFAULT_BIGRU_BEST}", flush=True)

    model_extra = {
        "bigru": ["--hp_json", str(hp_path)],
        "varcnn": ["--lr", str(args.varcnn_lr)],
        "netclr": ["--lr", str(args.netclr_lr)],
    }

    rows = {model: [] for model in models}
    for seed in seeds:
        for model in models:
            rdir = str(pathlib.Path(args.out_dir) / f"{model}_seed{seed}")
            ckpt = train_one(model, seed, rdir, args, model_extra[model])
            rows[model].append(metrics_one(model, seed, ckpt, args, device))

    csv_path = pathlib.Path(args.out_dir) / "multiseed_per_seed.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "seed", *METRICS])
        for model in models:
            for row in rows[model]:
                writer.writerow([model, row["seed"], *[row[k] for k in METRICS]])

    summary = {model: {k: summarize([row[k] for row in rows[model]]) for k in METRICS} for model in models}
    paired = {}
    for a_model, b_model in itertools.combinations(models, 2):
        paired[f"{a_model}_vs_{b_model}"] = {
            k: paired_test([r[k] for r in rows[a_model]], [r[k] for r in rows[b_model]])
            for k in METRICS
        }

    out = {
        "config": {
            "npz": args.npz,
            "models": models,
            "seeds": seeds,
            "chrono_split": bool(args.chrono_split),
            "scope": "closed-world only",
        },
        "per_seed": rows,
        "summary": summary,
        "paired_tests": paired,
    }
    json_path = pathlib.Path(args.out_dir) / "multiseed_summary.json"
    json_path.write_text(json.dumps(out, indent=2))

    print("\n=== mean +/- std over seeds (closed-world) ===", flush=True)
    for model in models:
        parts = [
            f"{k}={summary[model][k]['mean']:.4f}+/-{summary[model][k]['std']:.4f}"
            for k in METRICS
            if summary[model][k]["mean"] is not None
        ]
        print(f"  {model:7} (n={summary[model]['accuracy']['n']})  " + "  ".join(parts), flush=True)

    if paired:
        print("\n=== paired significance tests ===", flush=True)
        for pair_name, tests in paired.items():
            print(f"  [{pair_name}]", flush=True)
            for metric in METRICS:
                t = tests[metric]
                if "ttest_p" in t:
                    lo, hi = t["ci95_mean_diff"]
                    print(
                        f"    {metric:9} diff={t['mean_diff_a_minus_b']:+.4f} "
                        f"95%CI=[{lo:+.4f},{hi:+.4f}] p={t['ttest_p']:.4f}",
                        flush=True,
                    )

    print(f"\nWrote {csv_path}\n      {json_path}", flush=True)


if __name__ == "__main__":
    main()
