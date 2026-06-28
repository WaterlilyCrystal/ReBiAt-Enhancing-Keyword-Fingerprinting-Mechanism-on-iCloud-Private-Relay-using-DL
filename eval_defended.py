"""
eval_defended.py  --  NON-ADAPTIVE attacker evaluation (Methodology 8.3).

Loads a checkpoint trained on UNDEFENDED traffic and evaluates it on the test split of a
DEFENDED dataset, reporting closed-world accuracy and Macro-F1. This is the
"non-adaptive" row of the adversary-adaptation table: the attacker is caught off-guard.

Why this works without re-deriving the split:
  defenses.py preserves sample ORDER and LABELS (only X_seq / X_global change). So
  split_stratified(y) on the defended .npz yields the IDENTICAL test indices used at
  training time (same SEED=42, same y). We therefore evaluate on exactly the held-out
  test traces, now defended. Preprocessing (feature selection + StandardScaler) comes
  from the checkpoint (selected_idx + gl_scaler), i.e. fitted on CLEAN training data --
  precisely a non-adaptive attacker.

The ADAPTIVE counterpart needs no new code: just re-train on the defended set,
  python train_resnet_bigru.py --npz dataset_..._<defense>.npz [--use_augment]
and read its reported Test acc / macro_f1.

Usage:
  python eval_defended.py --checkpoint results/best_model.pt \
                          --defended_npz dataset_kfp_v2_mac_front.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from train_resnet_bigru import (
    KeywordClassifier, load_chronological_key, load_npz, make_loader,
    split_chronological, split_stratified,
)


# Run inference on a DataLoader and return concatenated logit tensor.
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    logits_all = []
    for x_seq, x_gl, _ in loader:
        x_seq = x_seq.to(device, non_blocking=True)
        x_gl = x_gl.to(device, non_blocking=True)
        logits_all.append(model(x_seq, x_gl).cpu())
    return torch.cat(logits_all, dim=0)


# Reconstruct the classifier from a checkpoint, supporting both thesis and baseline model types.
def rebuild_model(ckpt, device):
    arch = ckpt.get("model_arch") or {}
    args = ckpt.get("args", {})

    def _clean_state_dict(state: dict) -> dict:
        prefixes = ("_orig_mod.", "module.", "model.")
        cleaned = dict(state)
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if cleaned and all(str(k).startswith(prefix) for k in cleaned):
                    cleaned = {str(k)[len(prefix):]: v for k, v in cleaned.items()}
                    changed = True
        return cleaned

    # Dispatch on model_type so a Var-CNN / NetCLR baseline checkpoint can be
    # evaluated non-adaptively too. Legacy checkpoints -> KeywordClassifier.
    mtype = arch.get("model_type", "resnet_bigru")
    if mtype != "resnet_bigru":
        import sys
        if str(Path(__file__).parent) not in sys.path:
            sys.path.insert(0, str(Path(__file__).parent))
        import baselines as _B
        model = _B.rebuild_from_ckpt(ckpt, device, eval_mode=True)
        return model
    model = KeywordClassifier(
        n_classes=int(arch.get("n_classes")),
        global_feat=int(arch.get("global_feat")),
        seq_feat=int(arch.get("seq_feat", 3)),
        gru_hidden=int(arch.get("gru_hidden", args.get("gru_hidden", 128))),
        dropout_enc=float(arch.get("dropout_enc", 0.30)),
    ).to(device)
    model.load_state_dict(_clean_state_dict(ckpt["model_state"]))
    model.eval()
    return model


# Parse arguments, load checkpoint and defended dataset, run evaluation and save JSON results.
def main():
    p = argparse.ArgumentParser(description="Non-adaptive attacker on a defended dataset")
    p.add_argument("--checkpoint", required=True, help="Clean-trained .pt (e.g. results/best_model.pt)")
    p.add_argument("--defended_npz", required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--loader_workers", type=int, default=2)
    p.add_argument("--chrono_split", action="store_true",
                   help="Match how the checkpoint was trained (default: stratified)")
    p.add_argument("--leave_out", type=int, default=0,
                   help="Match training: drop the last m keywords (they were Unknown, not trained)")
    p.add_argument("--out_json", default="")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    X_seq, X_global, y, classes = load_npz(args.defended_npz)
    chrono_key = load_chronological_key(args.defended_npz)
    n_classes = len(classes)
    if args.leave_out > 0:
        keep = y < (n_classes - args.leave_out)
        X_seq, X_global, y = X_seq[keep], X_global[keep], y[keep]
        if chrono_key is not None:
            chrono_key = chrono_key[keep]
        classes = classes[: n_classes - args.leave_out]

    # Identical split to training (same seed/y -> same test indices).
    if args.chrono_split:
        _, _, te_idx = split_chronological(y, order_key=chrono_key)
    else:
        _, _, te_idx = split_stratified(y)

    selected_idx = np.array(ckpt["selected_idx"], dtype=np.int64)
    gl_scaler = ckpt["gl_scaler"]
    X_gl_te = gl_scaler.transform(X_global[te_idx][:, selected_idx]).astype(np.float32)

    model = rebuild_model(ckpt, device)
    loader = make_loader(X_seq[te_idx], X_gl_te, y[te_idx],
                         args.batch_size, shuffle=False, num_workers=args.loader_workers)

    logits = predict(model, loader, device)
    y_true = y[te_idx]
    y_pred = logits.argmax(dim=1).numpy()

    res = {
        "checkpoint": args.checkpoint,
        "defended_npz": args.defended_npz,
        "adaptation": "non_adaptive",
        "n_test": int(len(te_idx)),
        "n_classes": len(classes),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    print(f"\nNon-adaptive attacker on {Path(args.defended_npz).name}")
    print(f"  Accuracy : {res['accuracy']:.4f}")
    print(f"  Precision: {res['precision_macro']:.4f}")
    print(f"  Recall   : {res['recall_macro']:.4f}")
    print(f"  F1-score : {res['macro_f1']:.4f}")

    out_json = args.out_json or f"{Path(args.defended_npz).with_suffix('')}_nonadaptive.json"
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved -> {out_json}")


if __name__ == "__main__":
    main()
