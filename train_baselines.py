"""
train_baselines.py
==================
Train a BASELINE keyword-fingerprinting model (Var-CNN or NetCLR) on the SAME
v2 dataset, SAME stratified split, SAME feature selection + scaler, SAME metrics
and SAME checkpoint format as the thesis model (train_resnet_bigru.py).  This is
what makes the comparison fair: only the model differs.

This single script covers three of the four comparison scenarios:
  * CLOSED-WORLD : python train_baselines.py --model varcnn --npz <clean.npz>
  * OPEN-WORLD   : nothing special here - train on the 50-keyword known set, then
                   run open_world_pipeline.py with --checkpoint <this .pt>.
                   (Leave-m-out is also supported via --leave_out for the
                   alternative protocol.)
  * DEFENSE (adaptive attacker) : point --npz at a DEFENDED npz produced by
                   defenses.py - retraining on defended traffic IS the adaptive
                   attacker, identical to how the thesis model is evaluated.
The DEFENSE (non-adaptive) and CONCEPT-DRIFT scenarios consume the checkpoint
written here via eval_defended.py and drift_pipeline.py respectively (both patched
to rebuild baseline models from ckpt['model_arch']['model_type']).

Reuses everything reusable from train_resnet_bigru.py (data loading, split, feature
selection, scaler, augmentation, train/eval loops, focal loss, metrics, result
serialization) so the only methodological difference vs the thesis run is the model.

NetCLR is self-supervised: it first contrastively PRE-TRAINS a DFNet encoder with
NetAugment + NT-Xent on the (unlabeled) training-split sequences, then fine-tunes a
classifier.  Var-CNN is trained supervised end-to-end.

Usage examples:
  # Var-CNN closed-world (mirror the thesis training recipe)
  python train_baselines.py --model varcnn \
      --npz dataset_kfp_v2_macos_1_50.npz \
      --results_dir results_varcnn --epochs 80 --use_augment --loss focal

  # NetCLR closed-world (contrastive pre-train + fine-tune)
  python train_baselines.py --model netclr \
      --npz dataset_kfp_v2_macos_1_50.npz \
      --results_dir results_netclr --epochs 50 --pretrain_epochs 200

  # Var-CNN adaptive attacker against a FRONT-defended set
  python train_baselines.py --model varcnn \
      --npz dataset_kfp_v2_mac_front.npz --results_dir results_varcnn_front
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

# Reuse the thesis pipeline's helpers verbatim - identical preprocessing/eval.
sys.path.insert(0, str(Path(__file__).parent))
from train_resnet_bigru import (  # noqa: E402
    FocalLoss, compute_classification_metrics, eval_epoch, load_chronological_key, load_npz, make_loader,
    save_results, scale_global, select_global_features, set_global_seed,
    split_chronological, split_stratified, summarize_split, train_epoch,
)
import baselines as B  # noqa: E402


# ---------------------------------------------------------------------------
# NetCLR contrastive pre-training (Phase 1)
# ---------------------------------------------------------------------------

def pretrain_netclr(X_seq_tr: np.ndarray, seq_feat: int, device, args) -> dict:
    """Self-supervised NetCLR pre-training on the (unlabeled) training split.

    Returns the DFNet encoder state_dict (to seed the fine-tuning classifier).
    Pre-training uses ONLY the training split, so no test/val information leaks.
    """
    rng     = np.random.default_rng(args.seed)
    augment = B.NetAugment(rng=rng)
    ds      = B.ContrastiveDataset(X_seq_tr, augment)
    loader  = DataLoader(ds, batch_size=args.pretrain_bs, shuffle=True,
                         drop_last=True, num_workers=args.loader_workers,
                         pin_memory=torch.cuda.is_available(),
                         persistent_workers=(args.loader_workers > 0))
    model   = B.DFNetCLR(in_ch=seq_feat, feat_dim=args.feat_dim).to(device)
    if device.type == "cuda" and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception:
            pass
    crit    = B.NTXentLoss(temperature=args.temperature)
    opt     = optim.Adam(model.parameters(), lr=args.pretrain_lr)
    sched   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(len(loader), 1))
    scaler  = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    print(f"\n[NetCLR Phase 1] Contrastive pre-training "
          f"({len(X_seq_tr)} unlabeled traces, {args.pretrain_epochs} epochs)")
    for epoch in range(1, args.pretrain_epochs + 1):
        model.train()
        tot, contrastive_acc, nb = 0.0, 0.0, 0
        for v1, v2 in loader:
            v1, v2 = v1.to(device), v2.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                loss, acc = crit(model(v1), model(v2))
            if scaler is not None:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            tot += loss.item(); contrastive_acc += acc; nb += 1
        sched.step()
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  epoch {epoch:>3d}  ntxent={tot/max(nb,1):.4f}  "
                f"acc={contrastive_acc/max(nb,1):.4f}",
                flush=True,
            )
    return {k: v.cpu().clone() for k, v in model.encoder.state_dict().items()}


# ---------------------------------------------------------------------------
# Main training loop for a single baseline model run
# ---------------------------------------------------------------------------

def run(args):
    set_global_seed(args.seed)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True   # fixed seq length -> cache best algo
    print(f"Model: {args.model} | Device: {device} | Seed: {args.seed}")

    X_seq, X_global, y, classes = load_npz(args.npz)
    chrono_key = load_chronological_key(args.npz)
    n_classes = len(classes)
    print(f"Loaded: {len(y)} samples | {n_classes} classes | "
          f"X_seq {X_seq.shape} | X_global {X_global.shape}")

    # Leave-m-out (optional alt. open-world protocol): hold out the last m keywords.
    if args.leave_out > 0:
        m = int(args.leave_out)
        keep = y < (n_classes - m)
        X_seq, X_global, y = X_seq[keep], X_global[keep], y[keep]
        if chrono_key is not None:
            chrono_key = chrono_key[keep]
        classes = classes[: n_classes - m]
        n_classes -= m
        print(f"Leave-{m}-out: training on {n_classes} known classes "
              f"({len(X_seq)} traces)")

    # Identical split + preprocessing to the thesis pipeline.
    if args.chrono_split:
        tr_idx, val_idx, te_idx = split_chronological(y, order_key=chrono_key)
    else:
        tr_idx, val_idx, te_idx = split_stratified(y)
    print(f"Split: train={len(tr_idx)} val={len(val_idx)} test={len(te_idx)}")
    summarize_split("Train", y[tr_idx], classes)
    summarize_split("Test", y[te_idx], classes)

    X_gl_tr, X_gl_val, X_gl_te, selector, sel_idx = select_global_features(
        X_global[tr_idx], y[tr_idx], X_global[val_idx], X_global[te_idx], k=args.k_features)
    X_gl_tr, X_gl_val, X_gl_te, gl_scaler = scale_global(X_gl_tr, X_gl_val, X_gl_te)

    seq_feat = X_seq.shape[2]

    # ---- Build model ----
    model = B.build_model(
        args.model, n_classes=n_classes, global_feat=X_gl_tr.shape[1],
        seq_feat=seq_feat, dropout_enc=args.dropout_enc, feat_dim=args.feat_dim,
    ).to(device)

    # NetCLR: contrastive pre-train, then seed the encoder.
    if args.model == B.NETCLR and not args.no_pretrain:
        enc_state = pretrain_netclr(X_seq[tr_idx], seq_feat, device, args)
        model.encoder.load_state_dict(enc_state)
        torch.save({"encoder_state": enc_state, "args": vars(args)},
                   f"{args.results_dir}/netclr_pretrained.pt")
        if args.finetune_mode == "linear":
            for p in model.encoder.parameters():
                p.requires_grad = False
            print("[NetCLR Phase 2] Linear probe (encoder frozen)")
        else:
            print("[NetCLR Phase 2] Full fine-tune (encoder trainable)")

    if device.type == "cuda" and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("torch.compile OK (mode=reduce-overhead)")
        except Exception as e:
            print(f"torch.compile skipped: {e}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # ---- Supervised loaders (reuse augmentation + mixup machinery) ----
    aug_kwargs = {"drop_rate": 0.10, "time_sigma": 0.15,
                  "size_sigma": 0.03, "apply_prob": 0.50, "augment": True}
    tr_loader  = make_loader(X_seq[tr_idx], X_gl_tr, y[tr_idx], args.batch_size, True,
                             augment=args.use_augment, aug_kwargs=aug_kwargs,
                             n_classes=n_classes, num_workers=args.loader_workers)
    val_loader = make_loader(X_seq[val_idx], X_gl_val, y[val_idx], args.batch_size, False,
                             num_workers=args.loader_workers)
    te_loader  = make_loader(X_seq[te_idx], X_gl_te, y[te_idx], args.batch_size, False,
                             num_workers=args.loader_workers)

    if args.loss == "focal":
        criterion = FocalLoss(gamma=2.0, label_smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_val_score, best_state, patience_cnt = -1.0, None, 0
    best_val_metrics: dict = {}
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "val_macro_f1": [], "lr": []}

    print(f"\n{'Epoch':>6} {'TrLoss':>9} {'TrAcc':>9} {'ValLoss':>9} {'ValAcc':>9} {'ValF1':>9}")
    print("-" * 60)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, tr_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc, _, _, _, val_metrics = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        history["train_loss"].append(round(tr_loss, 4))
        history["train_acc"].append(round(tr_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["val_macro_f1"].append(round(val_metrics["f1_macro"], 4))
        history["lr"].append(round(scheduler.get_last_lr()[0], 6))
        print(f"{epoch:>6d} {tr_loss:>9.4f} {tr_acc:>9.4f} {val_loss:>9.4f} "
              f"{val_acc:>9.4f} {val_metrics['f1_macro']:>9.4f}  ({time.time()-t0:.1f}s)")

        score = val_metrics["f1_macro"] if args.early_stop_metric == "macro_f1" else val_acc
        if score > best_val_score:
            best_val_score = score
            best_val_metrics = dict(val_metrics); best_val_metrics["loss"] = float(val_loss)
            _base = getattr(model, "_orig_mod", model)  # unwrap torch.compile if active
            best_state = {k: v.cpu().clone() for k, v in _base.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(best {args.early_stop_metric}={best_val_score:.4f})")
                break

    _base = getattr(model, "_orig_mod", model)
    _base.load_state_dict(best_state)

    # ---- Test evaluation (identical metric function as the thesis model) ----
    test_loss, test_acc, y_pred, y_true, y_prob, test_metrics = eval_epoch(
        model, te_loader, criterion, device)
    print(f"\nTest: acc={test_acc:.4f} precision={test_metrics['precision_macro']:.4f} "
          f"recall={test_metrics['recall_macro']:.4f} f1={test_metrics['f1_macro']:.4f}")
    report = classification_report(
        y_true, y_pred, target_names=classes, digits=3, output_dict=True, zero_division=0
    )
    print(classification_report(y_true, y_pred, target_names=classes, digits=3, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)

    # ---- Save checkpoint in the SHARED format (model_arch carries model_type) ----
    model_arch = B.make_model_arch(args.model, n_classes, X_gl_tr.shape[1],
                                   seq_feat, dropout_enc=args.dropout_enc)
    ckpt = {
        "model_state":  best_state,
        "model_arch":   model_arch,
        "classes":      classes,
        "k_features":   args.k_features,
        "selected_idx": sel_idx.tolist(),
        "gl_scaler":    gl_scaler,
        "val_score":    best_val_score,
        "val_metrics":  best_val_metrics,
        "test_acc":     test_acc,
        "test_metrics": test_metrics,
        "args":         vars(args),
    }
    model_path = f"{args.results_dir}/{Path(args.npz).stem}_{args.model}.pt"
    torch.save(ckpt, model_path)
    best_model_path = f"{args.results_dir}/best_model.pt"   # alias for the scenario pipelines
    shutil.copy(model_path, best_model_path)
    np.save(f"{args.results_dir}/confusion_matrix.npy", cm)
    print(f"\nModel saved: {model_path}\nAlias:       {best_model_path}")

    # run_results.json/csv - same schema collect_defense_results.py / compare expect.
    save_results({
        "method":       args.model,
        "test_acc":     test_acc,
        "test_loss":    test_loss,
        "test_metrics": test_metrics,
        "val_score":    best_val_score,
        "val_metrics":  best_val_metrics,
        "n_classes":    n_classes,
        "n_samples":    int(len(y)),
        "model_path":   model_path,
        "history":      history,
        "class_report": report,
        "args":         vars(args),
    }, f"{args.results_dir}/run")
    return model


# Argument parser for CLI entry point; mirrors train_resnet_bigru.py argument names.
def build_argparser():
    p = argparse.ArgumentParser(description="Train Var-CNN / NetCLR baselines (v2 data contract)")
    p.add_argument("--model", choices=[B.VARCNN, B.NETCLR], required=True)
    # Example Kaggle path: /kaggle/working/dataset_kfp_v2_macos_1_50.npz
    p.add_argument("--npz", default="./dataset_kfp_v2_macos_1_50.npz")  # set this to your .npz file path
    p.add_argument("--results_dir", default="./results_baseline")
    p.add_argument("--k_features", type=int, default=15)
    p.add_argument("--epochs", type=int, default=80, help="Supervised (fine-tune) epochs")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--loader_workers", type=int, default=2)
    p.add_argument("--dropout_enc", type=float, default=0.30)
    p.add_argument("--loss", choices=["ce", "focal"], default="focal")
    p.add_argument("--early_stop_metric", choices=["macro_f1", "acc"], default="macro_f1")
    p.add_argument("--use_augment", action="store_true",
                   help="Enable on-the-fly augmentation + mixup for supervised training")
    p.add_argument("--leave_out", type=int, default=0,
                   help="Hold out last m keywords (alternative open-world protocol)")
    p.add_argument("--chrono_split", action="store_true")
    # NetCLR-specific
    p.add_argument("--feat_dim", type=int, default=B.NETCLR_FEAT_DIM,
                   help="NetCLR DFNet encoder output dim (also used as Var-CNN nothing)")
    p.add_argument("--pretrain_epochs", type=int, default=200)
    p.add_argument("--pretrain_bs", type=int, default=256)
    p.add_argument("--pretrain_lr", type=float, default=3e-4)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--finetune_mode", choices=["full", "linear"], default="full",
                   help="NetCLR: full fine-tune (default) or frozen-encoder linear probe")
    p.add_argument("--no_pretrain", action="store_true",
                   help="NetCLR ablation: skip contrastive pre-training (supervised from scratch)")
    return p


if __name__ == "__main__":
    run(build_argparser().parse_args())
