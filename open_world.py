"""
open_world.py  --  Open-world evaluation for keyword fingerprinting.

How the model produces a per-class confidence score (logit -> probability)
--------------------------------------------------------------------------
Training with cross-entropy loss teaches the final linear layer to assign a
*high logit* to the true class and low logits to all others for each sample.
After training the logit vector z  R^K carries class-specific scores: z[k]
measures how strongly the learned feature embedding matches keyword k.

  softmax probability  p_k = exp(z_k) / _j exp(z_j)

converts the raw logits into a calibrated probability simplex. The *maximum*
softmax value  max_k p_k  is used as the in-distribution confidence score:
a high value means the model assigns most probability mass to a single keyword
(likely in-distribution); a low, spread-out distribution signals an unfamiliar
input (likely out-of-distribution / unknown keyword).

The other OOD scores re-use the same logit vector without retraining:
  energy          -Tlog _k exp(z_k/T)        (Liu et al., NeurIPS 2020)
  mahalanobis     distance to class-conditional Gaussians in embedding space
                  (Lee et al., NeurIPS 2018)

OOD scoring methods (softmax, energy, mahalanobis)
  softmax     Max-softmax probability          (Hendrycks & Gimpel, ICLR 2017)
  energy      Energy score                     (Liu et al., NeurIPS 2020)
  mahalanobis Class-conditional Mahal. dist   (Lee et al., NeurIPS 2018)

Protocols
  1) External unknown dataset (.npz built from truly unseen keywords)  [preferred]
  2) Leave-m-out from known dataset                                     [sanity check]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


# Lazily import symbols from the training module to avoid circular imports at load time.
def _load_training_module():
    from train_resnet_bigru import (
        KeywordClassifier,
        load_chronological_key,
        load_npz,
        make_loader,
        scale_global,
        select_global_features,
        split_chronological,
        split_stratified,
    )
    return {
        "KeywordClassifier": KeywordClassifier,
        "load_chronological_key": load_chronological_key,
        "load_npz": load_npz,
        "make_loader": make_loader,
        "scale_global": scale_global,
        "select_global_features": select_global_features,
        "split_chronological": split_chronological,
        "split_stratified": split_stratified,
    }


def _load_npz_raw(path: str):
    """Load NPZ without strict label-continuity check (used for unknown datasets)."""
    data = np.load(path, allow_pickle=True)
    X_seq = data["X_seq"].astype(np.float32)
    X_global = data["X_global"].astype(np.float32)
    y = data["y"].astype(np.int64)
    classes = [str(x).replace("_", " ").strip() for x in data["classes"].tolist()]
    return X_seq, X_global, y, classes


# ---------------------------------------------------------------------------
# Feature collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_logits(model, loader, device: torch.device):
    model.eval()
    logits_all, y_all = [], []
    for x_seq, x_gl, y in loader:
        x_seq = x_seq.to(device, non_blocking=True)
        x_gl = x_gl.to(device, non_blocking=True)
        logits = model(x_seq, x_gl)
        logits_all.append(logits.cpu())
        y_all.append(y.clone())
    return torch.cat(logits_all, dim=0), torch.cat(y_all, dim=0)


@torch.no_grad()
def collect_embeddings(model, loader, device: torch.device):
    """Collect 288-dim penultimate embeddings via model.get_embedding()."""
    model.eval()
    embs_all, y_all = [], []
    for x_seq, x_gl, y in loader:
        x_seq = x_seq.to(device, non_blocking=True)
        x_gl = x_gl.to(device, non_blocking=True)
        emb = model.get_embedding(x_seq, x_gl)
        embs_all.append(emb.cpu())
        y_all.append(y.clone())
    return torch.cat(embs_all, dim=0), torch.cat(y_all, dim=0)


# ---------------------------------------------------------------------------
# OOD scoring methods
# ---------------------------------------------------------------------------

# Compute max-softmax probability as the in-distribution confidence score.
def score_softmax(logits: torch.Tensor) -> np.ndarray:
    return torch.softmax(logits, dim=1).max(dim=1).values.numpy()


# Compute energy score for OOD detection (Liu et al., NeurIPS 2020).
def score_energy(logits: torch.Tensor, temperature: float = 1.0) -> np.ndarray:
    return (temperature * torch.logsumexp(logits / temperature, dim=1)).numpy()


def fit_mahalanobis(train_embs: torch.Tensor, train_y: np.ndarray,
                    n_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit class-conditional Gaussians on training-split embeddings.

    Lee et al. (NeurIPS 2018) 'A Simple Unified Framework for Detecting
    Out-of-Distribution Samples and Adversarial Attacks'.

    Returns (means, precision) where precision = ^{-1} of the pooled
    within-class covariance with Tikhonov regularisation.
    """
    d = train_embs.size(1)

    means = torch.zeros(n_classes, d)
    for c in range(n_classes):
        mask = train_y == c
        if mask.sum() > 0:
            means[c] = train_embs[mask].mean(dim=0)

    scatter = torch.zeros(d, d)
    n_total = 0
    for c in range(n_classes):
        mask = train_y == c
        nc = int(mask.sum())
        if nc > 1:
            X_c = train_embs[mask] - means[c]
            scatter += X_c.T @ X_c
            n_total += nc

    cov = scatter / max(n_total - n_classes, 1)
    cov += 1e-4 * torch.eye(d)
    precision = torch.linalg.pinv(cov)
    return means, precision


@torch.no_grad()
def score_mahalanobis_loader(model, loader, device: torch.device,
                              means: torch.Tensor, precision: torch.Tensor) -> np.ndarray:
    """Mahalanobis distance OOD score.

    Score = -min_c MD(x, _c, ^{-1}).  Higher -> closer to a known class -> in-distribution.
    Vectorised over all C classes simultaneously.
    """
    means_d = means.to(device)        # (C, d)
    prec_d = precision.to(device)     # (d, d)
    scores = []
    model.eval()
    for x_seq, x_gl, _ in loader:
        x_seq = x_seq.to(device, non_blocking=True)
        x_gl = x_gl.to(device, non_blocking=True)
        emb = model.get_embedding(x_seq, x_gl)          # (B, d)
        diff = emb.unsqueeze(1) - means_d.unsqueeze(0)  # (B, C, d)
        # MD = (diff @ P)  diff summed over d  ->  (B, C)
        md2 = (diff @ prec_d * diff).sum(dim=-1)
        scores.append((-md2.min(dim=1).values).cpu().numpy())
    return np.concatenate(scores)


# ---------------------------------------------------------------------------
# Detection & classification metrics
# ---------------------------------------------------------------------------

# Compute ROC-AUC for binary OOD detection.
def detection_metrics(scores_in: np.ndarray, scores_out: np.ndarray) -> dict[str, float]:
    y = np.concatenate([np.ones(len(scores_in), dtype=np.int64),
                        np.zeros(len(scores_out), dtype=np.int64)])
    scores = np.concatenate([scores_in, scores_out])
    return {
        "roc_auc": float(roc_auc_score(y, scores)),
    }


# Sweep candidate thresholds on validation scores to find the best OOD decision boundary.
def tune_threshold(scores_known_val: np.ndarray, scores_unknown_val: np.ndarray,
                   metric: str = "accuracy") -> tuple[float, dict[str, float]]:
    candidates = np.unique(np.concatenate([scores_known_val, scores_unknown_val]))
    best_thr, best_score, best_stats = float(candidates[0]), -1.0, {}
    y_true = np.concatenate([
        np.ones(len(scores_known_val), dtype=np.int64),
        np.zeros(len(scores_unknown_val), dtype=np.int64),
    ])
    for thr in candidates:
        y_pred = np.concatenate([
            (scores_known_val >= thr).astype(np.int64),
            (scores_unknown_val >= thr).astype(np.int64),
        ])
        score = float(accuracy_score(y_true, y_pred))
        if score > best_score:
            best_score = score
            best_thr   = float(thr)
            best_stats = {"accuracy": score}
    return best_thr, best_stats


# Combine known-class classification and unknown rejection into open-world metrics.
def open_world_classification_metrics(
    known_logits: torch.Tensor,
    known_y: np.ndarray,
    unknown_logits: torch.Tensor,
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    unknown_label = int(known_logits.size(1))
    known_pred    = known_logits.argmax(dim=1).numpy()
    unknown_pred  = unknown_logits.argmax(dim=1).numpy()
    known_open_pred   = np.where(known_scores   >= threshold, known_pred,   unknown_label)
    unknown_open_pred = np.where(unknown_scores >= threshold, unknown_pred, unknown_label)
    y_true = np.concatenate([known_y,
                              np.full(len(unknown_open_pred), unknown_label, dtype=np.int64)])
    y_pred = np.concatenate([known_open_pred, unknown_open_pred])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

# Write results dict to both JSON and CSV under the given path prefix.
def save_results(results: dict, out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    with out_prefix.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    rows = []
    for method, vals in results["methods"].items():
        rows.append({"method": method,
                     **vals["detection"], **vals["open_world"], **vals["threshold_stats"]})
    if rows:
        with out_prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


# ---------------------------------------------------------------------------
# Dataset / model helpers
# ---------------------------------------------------------------------------

def build_known_unknown_sets(args, train_mod):
    """Returns 8-tuple:
       known_X_seq, known_X_gl, known_y, known_classes,
       unk_X_seq, unk_X_gl, unk_y, unk_classes
    unk_y/unk_classes enable per-keyword unknown-recall breakdown.
    """
    load_npz = train_mod["load_npz"]
    X_seq, X_global, y, classes = load_npz(args.known_npz)
    n_classes = len(classes)
    if args.leave_out > 0:
        m = int(args.leave_out)
        if m >= n_classes:
            raise ValueError(f"leave_out must be < n_classes, got {m} vs {n_classes}")
        keep = y < (n_classes - m)
        # Held-out keyword labels remapped to 0-based into unk_classes.
        unk_y       = (y[~keep] - (n_classes - m)).astype(np.int64)
        unk_classes = classes[n_classes - m:]
        return (X_seq[keep], X_global[keep], y[keep], classes[: n_classes - m],
                X_seq[~keep], X_global[~keep], unk_y, unk_classes)
    if not args.unknown_npz:
        raise ValueError("Pass --unknown_npz for external unknown, or --leave_out N for leave-m-out.")
    # Classification labels unused for OOD scoring, but kept for per-keyword breakdown.
    X_seq_u, X_global_u, y_u, classes_u = _load_npz_raw(args.unknown_npz)
    return X_seq, X_global, y, classes, X_seq_u, X_global_u, y_u, classes_u


# Partition known and unknown samples into train/val/test splits for OOD evaluation.
def split_known_unknown(args, y_known, n_unknown, train_mod):
    if args.chrono_split:
        chrono_key = train_mod["load_chronological_key"](args.known_npz)
        if chrono_key is not None and args.leave_out > 0:
            _, _, y_full, classes_full = train_mod["load_npz"](args.known_npz)
            keep = y_full < (len(classes_full) - int(args.leave_out))
            chrono_key = chrono_key[keep]
        tr_idx, val_idx, te_idx = train_mod["split_chronological"](y_known, order_key=chrono_key)
    else:
        tr_idx, val_idx, te_idx = train_mod["split_stratified"](y_known)
    u_idx = np.arange(n_unknown)
    cut   = n_unknown // 2
    return tr_idx, val_idx, te_idx, u_idx[:cut], u_idx[cut:]


def rebuild_model(checkpoint: dict, n_classes: int, global_dim: int,
                  seq_feat: int, train_mod, device):
    # Dispatch on model_type so the SAME OOD machinery scores the thesis model
    # AND the Var-CNN / NetCLR baselines. Legacy checkpoints (no model_type) fall
    # back to the thesis KeywordClassifier - behaviour unchanged.
    arch  = checkpoint.get("model_arch") or {}
    mtype = arch.get("model_type", "resnet_bigru")

    def _clean_state_dict(state: dict) -> dict:
        # torch.compile/DataParallel/wrapper modules can emit prefixed keys.
        # Strip known wrappers so evaluation can rebuild the normal eager model.
        for prefix in ("_orig_mod.", "module.", "model."):
            if any(str(k).startswith(prefix) for k in state):
                state = {
                    str(k)[len(prefix):] if str(k).startswith(prefix) else k: v
                    for k, v in state.items()
                }
        return state

    if mtype != "resnet_bigru":
        import sys
        if str(Path(__file__).parent) not in sys.path:
            sys.path.insert(0, str(Path(__file__).parent))
        import baselines as _B
        # Use arch["global_feat"] so the model is rebuilt with the exact same
        # meta-MLP width as when it was trained (not the current data shape).
        saved_global_feat = int(arch.get("global_feat", global_dim))
        model = _B.build_model(
            mtype, n_classes=n_classes, global_feat=saved_global_feat,
            seq_feat=seq_feat,
            gru_hidden=int(arch.get("gru_hidden", 128)),
            dropout_enc=float(arch.get("dropout_enc", 0.30)),
        ).to(device)
        model.load_state_dict(_clean_state_dict(checkpoint["model_state"]))
        model.eval()
        return model
    KeywordClassifier = train_mod["KeywordClassifier"]
    ckpt_args = checkpoint.get("args", {})
    model = KeywordClassifier(
        n_classes=n_classes,
        global_feat=global_dim,
        seq_feat=seq_feat,
        gru_hidden=int(ckpt_args.get("gru_hidden", 128)),
        gru_layers=int(ckpt_args.get("gru_layers", 2)),
        dropout_enc=float(ckpt_args.get("dropout_enc", 0.30)),
        dropout_fuse=float(ckpt_args.get("dropout_fuse", 0.50)),
    ).to(device)
    model.load_state_dict(_clean_state_dict(checkpoint["model_state"]))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Open-world OOD evaluation for keyword fingerprinting.")
    # Example Kaggle path: /kaggle/working/dataset_kfp_v2_macos_1_50.npz
    parser.add_argument("--known_npz",    default="./dataset_kfp_v2_macos_1_50.npz")
    parser.add_argument("--unknown_npz",  default="")
    parser.add_argument("--checkpoint",   required=True)
    parser.add_argument("--results_dir",  default="./results_open_world")
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--loader_workers", type=int, default=2)
    parser.add_argument("--energy_temperature", type=float, default=1.0)
    parser.add_argument("--threshold_metric", choices=["accuracy"], default="accuracy")
    parser.add_argument("--methods", default="softmax,energy,mahalanobis",
                        help="Comma-separated OOD methods: softmax, energy, mahalanobis.")
    parser.add_argument("--leave_out",    type=int, default=0)
    parser.add_argument("--chrono_split", action="store_true")
    parser.add_argument("--save_logits",  action="store_true",
                        help="Cache raw logits and per-method scores to NPZ for offline analysis.")
    args = parser.parse_args()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_mod = _load_training_module()
    make_loader          = train_mod["make_loader"]
    scale_global         = train_mod["scale_global"]
    select_global_features = train_mod["select_global_features"]

    # weights_only=False: our checkpoint bundles a sklearn StandardScaler
    # (gl_scaler) and selected_idx, not just tensors. The file is produced by
    # our own training script, so it is a trusted source.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    methods    = tuple(x.strip() for x in args.methods.split(",") if x.strip())

    X_seq_k, X_gl_k, y_k, classes_k, X_seq_u, X_gl_u, y_u, classes_u = build_known_unknown_sets(args, train_mod)
    y_u = np.asarray(y_u, dtype=np.int64)
    tr_idx, val_idx, te_idx, u_val_idx, u_te_idx = split_known_unknown(
        args, y_k, len(X_seq_u), train_mod)

    selected_idx = checkpoint.get("selected_idx")
    if selected_idx is not None:
        selected_idx = np.array(selected_idx, dtype=np.int64)
        X_gl_tr     = X_gl_k[tr_idx][:,   selected_idx]
        X_gl_val    = X_gl_k[val_idx][:,  selected_idx]
        X_gl_te     = X_gl_k[te_idx][:,   selected_idx]
        X_gl_u_val  = X_gl_u[u_val_idx][:, selected_idx]
        X_gl_u_te   = X_gl_u[u_te_idx][:,  selected_idx]
    else:
        X_gl_tr, X_gl_val, X_gl_te, selector, selected_idx = select_global_features(
            X_gl_k[tr_idx], y_k[tr_idx], X_gl_k[val_idx], X_gl_k[te_idx],
            k=int(checkpoint.get("k_features", 15)),
        )
        X_gl_u_val = selector.transform(X_gl_u[u_val_idx])
        X_gl_u_te  = selector.transform(X_gl_u[u_te_idx])

    X_gl_tr, X_gl_val, X_gl_te, scaler = scale_global(X_gl_tr, X_gl_val, X_gl_te)
    X_gl_u_val = scaler.transform(X_gl_u_val).astype(np.float32)
    X_gl_u_te  = scaler.transform(X_gl_u_te).astype(np.float32)

    model = rebuild_model(
        checkpoint, len(classes_k), X_gl_tr.shape[1], X_seq_k.shape[2], train_mod, device)

    kw_val  = dict(batch_size=args.batch_size, shuffle=False, num_workers=args.loader_workers)
    known_val_loader   = make_loader(X_seq_k[val_idx], X_gl_val, y_k[val_idx],      **kw_val)
    known_te_loader    = make_loader(X_seq_k[te_idx],  X_gl_te,  y_k[te_idx],       **kw_val)
    unknown_val_loader = make_loader(X_seq_u[u_val_idx], X_gl_u_val,
                                     np.zeros(len(u_val_idx), dtype=np.int64),       **kw_val)
    unknown_te_loader  = make_loader(X_seq_u[u_te_idx],  X_gl_u_te,
                                     np.zeros(len(u_te_idx),  dtype=np.int64),       **kw_val)

    # Collect logits (reused by softmax / energy)
    print("Collecting logits...")
    val_logits_k, _      = collect_logits(model, known_val_loader,   device)
    te_logits_k,  te_y_k = collect_logits(model, known_te_loader,    device)
    val_logits_u, _      = collect_logits(model, unknown_val_loader,  device)
    te_logits_u,  _      = collect_logits(model, unknown_te_loader,   device)

    # Fit Mahalanobis on training split (if requested)
    maha_means = maha_precision = None
    if "mahalanobis" in methods:
        train_loader = make_loader(
            X_seq_k[tr_idx], X_gl_tr, y_k[tr_idx], **kw_val)
        print("Fitting Mahalanobis on training embeddings...", end=" ", flush=True)
        train_embs, train_y_embs = collect_embeddings(model, train_loader, device)
        maha_means, maha_precision = fit_mahalanobis(
            train_embs, train_y_embs.numpy(), len(classes_k))
        print("done.")

    # Optionally save logit tensors for offline analysis
    if args.save_logits:
        out_dir = Path(args.results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / "logits_cache.npz",
            te_logits_k  = te_logits_k.numpy(),
            te_logits_u  = te_logits_u.numpy(),
            te_y_k       = te_y_k.numpy(),
            val_logits_k = val_logits_k.numpy(),
            val_logits_u = val_logits_u.numpy(),
            # Per-keyword unknown-recall breakdown: keyword label of each unknown
            # TEST sample (aligned 1:1 with te_s_u order in scores_*.npz), + names.
            unknown_y_te    = y_u[u_te_idx],
            unknown_classes = np.array(classes_u),
        )
        print(f"Saved: {out_dir / 'logits_cache.npz'}")

    results = {
        "checkpoint":      args.checkpoint,
        "known_npz":       args.known_npz,
        "unknown_npz":     args.unknown_npz,
        "leave_out":       args.leave_out,
        "n_known_classes": len(classes_k),
        "n_known_train":   int(len(tr_idx)),
        "n_known_val":     int(len(val_idx)),
        "n_known_test":    int(len(te_idx)),
        "n_unknown_val":   int(len(u_val_idx)),
        "n_unknown_test":  int(len(u_te_idx)),
        "methods": {},
    }

    hdr = f"{'Method':<13} {'ROC-AUC':>7} {'Accuracy':>8} {'Precision':>9} {'Recall':>7} {'F1':>8}"
    print("\nOpen-world results")
    print(hdr)
    print("-" * len(hdr))

    for method in methods:
        if method == "softmax":
            val_s_k = score_softmax(val_logits_k); val_s_u = score_softmax(val_logits_u)
            te_s_k  = score_softmax(te_logits_k);  te_s_u  = score_softmax(te_logits_u)
        elif method == "energy":
            val_s_k = score_energy(val_logits_k, args.energy_temperature)
            val_s_u = score_energy(val_logits_u, args.energy_temperature)
            te_s_k  = score_energy(te_logits_k,  args.energy_temperature)
            te_s_u  = score_energy(te_logits_u,  args.energy_temperature)
        elif method == "mahalanobis":
            if maha_means is None:
                raise RuntimeError("Mahalanobis params not fitted.")
            val_s_k = score_mahalanobis_loader(model, known_val_loader,   device, maha_means, maha_precision)
            val_s_u = score_mahalanobis_loader(model, unknown_val_loader,  device, maha_means, maha_precision)
            te_s_k  = score_mahalanobis_loader(model, known_te_loader,    device, maha_means, maha_precision)
            te_s_u  = score_mahalanobis_loader(model, unknown_te_loader,  device, maha_means, maha_precision)
        else:
            raise ValueError(f"Unknown method: {method!r}. Choose from: softmax, energy, mahalanobis")

        det         = detection_metrics(te_s_k, te_s_u)
        thr, thr_st = tune_threshold(val_s_k, val_s_u, metric=args.threshold_metric)
        open_m      = open_world_classification_metrics(
            te_logits_k, te_y_k.numpy(), te_logits_u, te_s_k, te_s_u, thr)

        results["methods"][method] = {
            "detection": det, "threshold": thr,
            "threshold_stats": thr_st, "open_world": open_m,
        }
        print(
            f"{method:<13} {det['roc_auc']:>7.4f} {open_m['accuracy']:>8.4f}"
            f" {open_m['precision_macro']:>9.4f} {open_m['recall_macro']:>7.4f}"
            f" {open_m['f1_macro']:>8.4f}"
        )

        if args.save_logits:
            np.savez_compressed(
                Path(args.results_dir) / f"scores_{method}.npz",
                val_s_k=val_s_k, val_s_u=val_s_u,
                te_s_k=te_s_k,   te_s_u=te_s_u,
                threshold=np.array([thr]),
            )

    closed_pred = te_logits_k.argmax(dim=1).numpy()
    results["closed_world_known_test"] = {
        "acc":         float(accuracy_score(te_y_k.numpy(), closed_pred)),
        "precision_macro": float(precision_score(te_y_k.numpy(), closed_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(te_y_k.numpy(), closed_pred, average="macro", zero_division=0)),
        "macro_f1":    float(f1_score(te_y_k.numpy(), closed_pred, average="macro", zero_division=0)),
        "report":      classification_report(
            te_y_k.numpy(), closed_pred, target_names=classes_k,
            digits=4, output_dict=True, zero_division=0),
    }

    out_prefix = Path(args.results_dir) / "open_world_results"
    save_results(results, out_prefix)
    print(f"\nSaved: {out_prefix.with_suffix('.json')} and {out_prefix.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
