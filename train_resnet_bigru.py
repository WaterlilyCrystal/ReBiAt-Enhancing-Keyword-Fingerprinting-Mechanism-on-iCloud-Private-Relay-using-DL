"""
train_resnet_bigru.py
ResNet-10 (1D) + BiGRU + Temporal Attention + SE blocks for keyword traffic fingerprinting.

Architecture (Methodology Sec. 5):
  X_seq   (N,L,3)  -> ResNet-10(SE) -> (N,L/4,256) -> BiGRU -> TemporalAttention -> h_gru (N,256)
  X_global(N,15)   -> GlobalMLP(64->32)            -> h_global (N,32)
  concat (N,288) -> Dense(256) -> Dropout(0.5) -> Dense(n_classes)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


# Sets the global random seed for reproducibility across numpy and torch.
def set_global_seed(seed: int) -> None:
    global SEED
    SEED = int(seed)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class SEBlock1D(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al., CVPR 2018)."""
    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Sequential(
            nn.Linear(ch, max(ch // r, 4), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(ch // r, 4), ch, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.gap(x).squeeze(-1)
        return x * self.fc(s).unsqueeze(-1)


class BasicBlock1D(nn.Module):
    """ResNet BasicBlock (1-D) with dilation, optional stride/channel change, and SE.

    Two 3x3 convolutions (dilations r1, r2); a 1x1 projection shortcut is used
    when the stride or channel count changes (He et al., 2016).
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 dilations: tuple[int, int] = (1, 1)):
        super().__init__()
        d1, d2 = dilations
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride,
                               padding=d1, dilation=d1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1,
                               padding=d2, dilation=d2, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.se    = SEBlock1D(out_ch)
        self.act   = nn.ELU(inplace=True)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return self.act(out + self.shortcut(x))


class ResNet1D(nn.Module):
    """ResNet-10 (1-D) local extractor: (N, L, 3) -> (N, L/4, 256).

    Stem + 4 BasicBlocks (10 weight layers). Dilation {1,2} in the two
    non-downsampling blocks; stride-2 in blocks 2 & 4 for 4x compression.
    """
    def __init__(self, in_ch: int = 3, dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ELU(inplace=True),
        )
        self.block1 = BasicBlock1D(64,  64,  stride=1, dilations=(1, 2))
        self.block2 = BasicBlock1D(64,  128, stride=2, dilations=(1, 1))
        self.block3 = BasicBlock1D(128, 256, stride=1, dilations=(1, 2))
        self.block4 = BasicBlock1D(256, 256, stride=2, dilations=(1, 1))
        self.drop   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)                  # (N, 3, L)
        x = self.stem(x)
        x = self.block2(self.block1(x))         # -> (N, 128, L/2)
        x = self.block4(self.block3(x))         # -> (N, 256, L/4)
        x = self.drop(x)
        return x.permute(0, 2, 1)               # (N, L/4, 256)


class TemporalAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1, bias=False)

    def forward(self, H: torch.Tensor, return_weights: bool = False):
        w = torch.softmax(self.score(H), dim=1)
        ctx = (H * w).sum(dim=1)
        if return_weights:
            return ctx, w.squeeze(-1)
        return ctx


class KeywordClassifier(nn.Module):
    """Full model: ResNet-10(SE) + BiGRU + TemporalAttention + GlobalMLP -> classifier."""
    def __init__(
        self,
        n_classes: int,
        global_feat: int,
        seq_feat: int = 3,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        dropout_enc: float = 0.30,
        dropout_fuse: float = 0.50,
    ):
        super().__init__()
        self.encoder    = ResNet1D(in_ch=seq_feat, dropout=dropout_enc)
        self.bigru      = nn.GRU(
            input_size=256, hidden_size=gru_hidden, num_layers=gru_layers,
            batch_first=True, bidirectional=True, dropout=dropout_enc,
        )
        self.attn       = TemporalAttention(gru_hidden * 2)
        self.global_mlp = nn.Sequential(
            nn.Linear(global_feat, 64), nn.ReLU(inplace=True),
            nn.Dropout(dropout_enc),
            nn.Linear(64, 32), nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden * 2 + 32, 256), nn.ReLU(inplace=True),
            nn.Dropout(dropout_fuse),
            nn.Linear(256, n_classes),
        )

    def get_embedding(self, x_seq: torch.Tensor, x_global: torch.Tensor) -> torch.Tensor:
        z     = self.encoder(x_seq)
        H, _  = self.bigru(z)
        h_gru = self.attn(H)
        h_gl  = self.global_mlp(x_global)
        return torch.cat([h_gru, h_gl], dim=1)

    def forward(self, x_seq, x_global, return_attention: bool = False):
        z    = self.encoder(x_seq)
        H, _ = self.bigru(z)
        if return_attention:
            h_gru, attn_w = self.attn(H, return_weights=True)
        else:
            h_gru = self.attn(H)
        h_gl   = self.global_mlp(x_global)
        merged = torch.cat([h_gru, h_gl], dim=1)
        logits = self.classifier(merged)
        if return_attention:
            return logits, attn_w
        return logits


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., ICCV 2017) for handling hard / ambiguous examples.
    gamma=2: hard examples get ~4x higher relative weight than easy examples (p_t=0.9).
    Useful for time-sensitive queries like 'did anyone win the powerball'.
    """
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.1):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        n_classes = logits.size(1)

        with torch.no_grad():
            target_dist = torch.full_like(logits, self.label_smoothing / n_classes)
            target_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing + self.label_smoothing / n_classes)

        focal_weight = (1.0 - probs).pow(self.gamma)
        loss = -(target_dist * focal_weight * log_probs).sum(dim=1)
        return loss.mean()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

# Loads a dataset from a .npz file and validates label/class consistency.
def load_npz(path: str):
    data = np.load(path, allow_pickle=True)
    X_seq = data["X_seq"].astype(np.float32)
    X_global = data["X_global"].astype(np.float32)
    y = data["y"].astype(np.int64)
    classes = [str(x).replace("_", " ").strip() for x in data["classes"].tolist()]

    unique_y = np.unique(y)
    expected = np.arange(len(classes), dtype=np.int64)
    if not np.array_equal(unique_y, expected):
        raise ValueError(
            "Label/class mismatch in dataset: "
            f"unique y={unique_y.tolist()} vs expected 0..{len(classes) - 1}. "
            "Re-run feature extraction to rebuild a consistent .npz."
        )

    if X_seq.ndim != 3 or X_global.ndim != 2 or len(X_seq) != len(X_global) or len(y) != len(X_seq):
        raise ValueError(
            "Invalid dataset shapes: expected X_seq=(N,L,C), X_global=(N,F), y=(N,), "
            f"got X_seq={X_seq.shape}, X_global={X_global.shape}, y={y.shape}."
        )
    if not np.isfinite(X_seq).all() or not np.isfinite(X_global).all():
        raise ValueError("Dataset contains NaN or Inf values in X_seq/X_global.")

    return X_seq, X_global, y, classes


def load_chronological_key(path: str) -> np.ndarray | None:
    """Return per-sample ordering metadata for temporal splits, when available."""
    data = np.load(path, allow_pickle=True)
    for key in ("capture_start_time", "sample_order"):
        if key in data.files:
            order = np.asarray(data[key])
            if len(order) == len(data["y"]):
                return order
    if "file_paths" in data.files:
        paths = np.asarray(data["file_paths"]).astype(str)
        if len(paths) == len(data["y"]):
            return paths
    return None


# Splits indices into train/val/test using stratified random sampling.
def split_stratified(y, val_ratio=0.20, test_ratio=0.20):
    idx = np.arange(len(y))
    tr_idx, tmp = train_test_split(idx, test_size=val_ratio + test_ratio,
                                   stratify=y, random_state=SEED)
    val_idx, te_idx = train_test_split(
        tmp, test_size=test_ratio / (val_ratio + test_ratio),
        stratify=y[tmp], random_state=SEED,
    )
    return tr_idx, val_idx, te_idx


# Splits each class into train/val/test by temporal/file order, then concatenates.
def split_chronological(y, val_ratio=0.20, test_ratio=0.20, order_key=None):
    y = np.asarray(y)
    if order_key is None:
        warnings.warn(
            "split_chronological() was called without per-sample ordering metadata; "
            "falling back to current array order within each class. For a defensible "
            "temporal split, rebuild the dataset with the current extract_features_v2.py.",
            RuntimeWarning,
            stacklevel=2,
        )
        order_key = np.arange(len(y))
    order_key = np.asarray(order_key)
    if len(order_key) != len(y):
        raise ValueError(f"order_key length {len(order_key)} does not match y length {len(y)}")

    tr_parts, val_parts, te_parts = [], [], []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        cls_idx = cls_idx[np.argsort(order_key[cls_idx], kind="mergesort")]
        n = len(cls_idx)
        tr_end = int(n * (1.0 - val_ratio - test_ratio))
        val_end = int(n * (1.0 - test_ratio))
        tr_parts.append(cls_idx[:tr_end])
        val_parts.append(cls_idx[tr_end:val_end])
        te_parts.append(cls_idx[val_end:])

    return (
        np.concatenate(tr_parts).astype(np.int64),
        np.concatenate(val_parts).astype(np.int64),
        np.concatenate(te_parts).astype(np.int64),
    )


# Prints a summary of sample counts and class distribution for a data split.
def summarize_split(name: str, y_split: np.ndarray, classes: list[str]) -> None:
    counts = np.bincount(y_split, minlength=len(classes))
    nonzero = counts[counts > 0]
    print(
        f"{name}: samples={len(y_split)} classes_present={(counts > 0).sum()}/{len(classes)} "
        f"min_per_class={int(nonzero.min()) if len(nonzero) else 0} "
        f"max_per_class={int(nonzero.max()) if len(nonzero) else 0}"
    )


def prepare_global_features(X_tr, y_tr, X_val, X_te, k=None):
    """Prepare the global-feature branch inputs.

    No runtime feature *selection* is performed. The fifteen global descriptors
    are curated offline from a larger candidate pool via the ANOVA / mutual-
    information / random-forest importance study in ``feature_selection.py`` and
    are fixed before training. ``SelectKBest(k="all")`` is used here only as an
    identity pass-through, so that the same fitted object can be reapplied
    unchanged to the held-out unknown traces in the open-world evaluation. The
    ``k`` argument is accepted for backward compatibility and ignored.
    """
    sel = SelectKBest(f_classif, k="all")
    X_tr_s  = sel.fit_transform(X_tr, y_tr)
    X_val_s = sel.transform(X_val)
    X_te_s  = sel.transform(X_te)
    return X_tr_s, X_val_s, X_te_s, sel, sel.get_support(indices=True)


# Fits a StandardScaler on training data and transforms train/val/test splits.
def scale_global(X_tr, X_val, X_te):
    sc = StandardScaler()
    X_tr  = sc.fit_transform(X_tr).astype(np.float32)
    X_val = sc.transform(X_val).astype(np.float32)
    X_te  = sc.transform(X_te).astype(np.float32)
    return X_tr, X_val, X_te, sc


# Builds a DataLoader for a split, with optional data augmentation and mixup collation.
def make_loader(X_seq, X_global, y, batch_size: int, shuffle: bool,
                augment: bool = False, aug_kwargs: dict | None = None,
                n_classes: int | None = None,
                num_workers: int = 0) -> DataLoader:
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    if augment:
        from augmentation import AugmentedDataset, mixup_collate_fn
        import functools
        ds      = AugmentedDataset(X_seq, X_global, y, **(aug_kwargs or {}))
        collate = functools.partial(mixup_collate_fn, alpha=0.4, mixup_prob=0.5,
                                    n_classes=n_classes or int(y.max()) + 1)
        return DataLoader(ds, collate_fn=collate, **loader_kwargs)
    ds = TensorDataset(
        torch.from_numpy(X_seq), torch.from_numpy(X_global), torch.from_numpy(y),
    )
    return DataLoader(ds, **loader_kwargs)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def _soft_ce(logits: torch.Tensor, y_soft: torch.Tensor) -> torch.Tensor:
    return -(y_soft * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


# Computes the committee-facing classification metrics: accuracy, precision, recall, and F1.
def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict:
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics = {
        "acc": float((y_true == y_pred).mean()),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
    }
    return metrics


# Runs one training epoch and returns average loss and accuracy.
def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x_seq, x_gl, y_batch in loader:
        x_seq = x_seq.to(device, non_blocking=True)
        x_gl = x_gl.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            logits = model(x_seq, x_gl)
            if y_batch.dim() == 2:
                loss = _soft_ce(logits, y_batch)
                pred_labels = y_batch.argmax(dim=1)
            else:
                loss = criterion(logits, y_batch)
                pred_labels = y_batch

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * x_seq.size(0)
        correct    += (logits.argmax(1) == pred_labels).sum().item()
        n          += x_seq.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_pred, all_true, all_prob = [], [], []
    for x_seq, x_gl, y_batch in loader:
        x_seq = x_seq.to(device, non_blocking=True)
        x_gl = x_gl.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        if y_batch.dim() == 2:
            y_batch = y_batch.argmax(dim=1)
        with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
            logits = model(x_seq, x_gl)
            loss   = criterion(logits, y_batch)
        total_loss += loss.item() * x_seq.size(0)
        probs = torch.softmax(logits, dim=1)
        preds       = logits.argmax(1)
        correct    += (preds == y_batch).sum().item()
        n          += x_seq.size(0)
        all_pred.extend(preds.cpu().numpy())
        all_true.extend(y_batch.cpu().numpy())
        all_prob.extend(probs.cpu().numpy())
    metrics = compute_classification_metrics(np.array(all_true), np.array(all_pred), np.array(all_prob))
    return total_loss / n, correct / n, np.array(all_pred), np.array(all_true), np.array(all_prob), metrics


# ---------------------------------------------------------------------------
# Attention map saving
# ---------------------------------------------------------------------------

# Collects per-class temporal attention weights from the test loader and saves them to a .npz file.
@torch.no_grad()
def save_attention_maps(model, loader, classes, device, out_path, max_per_class=50):
    model.eval()
    weights_by_class: dict[int, list] = {}
    for x_seq, x_gl, y_batch in loader:
        x_seq = x_seq.to(device, non_blocking=True)
        x_gl = x_gl.to(device, non_blocking=True)
        if y_batch.dim() == 2:
            y_batch = y_batch.argmax(dim=1)
        _, attn_w = model(x_seq, x_gl, return_attention=True)
        for i, cls in enumerate(y_batch.numpy()):
            cls = int(cls)
            buf = weights_by_class.setdefault(cls, [])
            if len(buf) < max_per_class:
                buf.append(attn_w[i].cpu().numpy())

    all_weights, all_labels = [], []
    for cls, ws in sorted(weights_by_class.items()):
        all_weights.extend(ws)
        all_labels.extend([cls] * len(ws))

    np.savez_compressed(out_path,
                        weights=np.array(all_weights, dtype=np.float32),
                        labels=np.array(all_labels, dtype=np.int64),
                        class_names=np.array(classes))
    print(f"Attention maps saved: {out_path}  ({len(all_labels)} samples)")


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

# Saves a results dict to both JSON and CSV files under out_prefix.
def save_results(results: dict, out_prefix: str) -> None:
    json_path = f"{out_prefix}_results.json"
    csv_path  = f"{out_prefix}_results.csv"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    flat = {}
    for k, v in results.items():
        if isinstance(v, (int, float, str, bool)):
            flat[k] = v
        elif isinstance(v, list) and v and isinstance(v[0], (int, float)):
            flat[f"{k}_final"] = v[-1]
            flat[f"{k}_best"]  = max(v) if "acc" in k.lower() else min(v)

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in flat.items():
            w.writerow([k, v])

    print(f"Results saved: {json_path}  {csv_path}")


# ---------------------------------------------------------------------------
# K-fold cross-validation
# ---------------------------------------------------------------------------

# Runs k-fold cross-validation and saves per-fold and aggregate results.
def run_kfold(args, X_seq, X_global, y, classes, device, k: int = 5) -> list[dict]:
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    fold_results = []
    criterion    = nn.CrossEntropyLoss(label_smoothing=0.1)

    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X_seq, y), start=1):
        print(f"\n{'='*50}\nFold {fold_idx}/{k}")
        val_size = int(len(tr_idx) * 0.20)
        val_idx  = tr_idx[:val_size]
        tr_idx   = tr_idx[val_size:]

        X_gl_tr, X_gl_val, X_gl_te, sel, _ = prepare_global_features(
            X_global[tr_idx], y[tr_idx],
            X_global[val_idx], X_global[te_idx], k=args.k_features)
        X_gl_tr, X_gl_val, X_gl_te, _ = scale_global(X_gl_tr, X_gl_val, X_gl_te)

        tr_loader  = make_loader(X_seq[tr_idx],  X_gl_tr,  y[tr_idx],  args.batch_size, True,
                                 augment=args.use_augment, n_classes=len(classes),
                                 num_workers=args.loader_workers)
        val_loader = make_loader(X_seq[val_idx], X_gl_val, y[val_idx], args.batch_size, False,
                                 num_workers=args.loader_workers)
        te_loader  = make_loader(X_seq[te_idx],  X_gl_te,  y[te_idx],  args.batch_size, False,
                                 num_workers=args.loader_workers)

        model = KeywordClassifier(
            n_classes=len(classes), global_feat=X_gl_tr.shape[1],
            seq_feat=X_seq.shape[2],
        ).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scaler    = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

        best_val, best_state, patience = 0.0, None, 0
        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc = train_epoch(model, tr_loader, criterion, optimizer, scaler, device)
            val_loss, val_acc, _, _, _, val_metrics = eval_epoch(model, val_loader, criterion, device)
            scheduler.step()
            if val_metrics["f1_macro"] > best_val:
                best_val   = val_metrics["f1_macro"]
                best_state = {k2: v.cpu().clone() for k2, v in model.state_dict().items()}
                patience   = 0
            else:
                patience += 1
                if patience >= args.patience:
                    break

        model.load_state_dict(best_state)
        _, te_acc, y_pred, y_true, _, te_metrics = eval_epoch(model, te_loader, criterion, device)
        print(f"Fold {fold_idx} test acc: {te_acc:.4f} | macro_f1: {te_metrics['f1_macro']:.4f}")
        fold_results.append({
            "fold": fold_idx,
            "val_macro_f1": best_val,
            "test_acc": te_acc,
            "test_macro_f1": te_metrics["f1_macro"],
        })

    accs = [r["test_acc"] for r in fold_results]
    f1s = [r["test_macro_f1"] for r in fold_results]
    print(f"\nK-fold results: acc_mean={np.mean(accs):.4f} acc_std={np.std(accs):.4f} "
          f"f1_mean={np.mean(f1s):.4f} f1_std={np.std(f1s):.4f}")
    save_results({
        "kfold_test_accs": accs,
        "kfold_test_macro_f1s": f1s,
        "kfold_mean_acc": float(np.mean(accs)),
        "kfold_std_acc": float(np.std(accs)),
        "kfold_mean_macro_f1": float(np.mean(f1s)),
        "kfold_std_macro_f1": float(np.std(f1s)),
        "fold_results": fold_results,
        "k": k,
    }, f"{args.results_dir}/kfold")
    return fold_results


# ---------------------------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------------------------

_DEFAULT_SPACE = {
    "lr":              ("loguniform", 1e-4, 1e-2),
    "batch_size":      ("choice", [32, 64, 128]),
    "dropout_enc":     ("uniform",   0.2, 0.5),
    "label_smoothing": ("uniform",   0.0, 0.2),
    "gru_hidden":      ("choice", [64, 128, 256]),
}


# Samples a single hyperparameter configuration from the given search space.
def _sample_params(space, rng):
    params = {}
    for key, spec in space.items():
        if spec[0] == "loguniform":
            lo, hi = np.log(spec[1]), np.log(spec[2])
            params[key] = float(np.exp(rng.uniform(lo, hi)))
        elif spec[0] == "uniform":
            params[key] = float(rng.uniform(spec[1], spec[2]))
        elif spec[0] == "choice":
            params[key] = rng.choice(spec[1])
    return params


# Converts numpy scalars in a params dict to native Python types for JSON serialisation.
def _jsonify_params(params):
    clean = {}
    for key, value in params.items():
        if isinstance(value, np.generic):
            value = value.item()
        clean[key] = value
    return clean


# Runs a short training trial with the given hyperparameters and returns the best validation macro F1.
def _trial_run(params, args, X_seq, X_global, y, classes, device):
    label_smoothing = float(params.get("label_smoothing", 0.1))
    if args.loss == "focal":
        criterion = FocalLoss(gamma=2.0, label_smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    tr_idx, val_idx, _ = split_stratified(y)
    X_gl_tr, X_gl_val, _, _, _ = prepare_global_features(
        X_global[tr_idx], y[tr_idx], X_global[val_idx], X_global, k=args.k_features)
    X_gl_tr, X_gl_val, _, _ = scale_global(X_gl_tr, X_gl_val, X_gl_val)
    bs = int(params.get("batch_size", 64))
    tr_loader  = make_loader(X_seq[tr_idx],  X_gl_tr,  y[tr_idx],  bs, True,
                             augment=args.use_augment, num_workers=args.loader_workers)
    val_loader = make_loader(X_seq[val_idx], X_gl_val, y[val_idx], bs, False,
                             num_workers=args.loader_workers)
    model = KeywordClassifier(
        n_classes=len(classes), global_feat=X_gl_tr.shape[1],
        seq_feat=X_seq.shape[2],
        dropout_enc=float(params.get("dropout_enc", 0.3)),
        gru_hidden=int(params.get("gru_hidden", 128)),
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=float(params.get("lr", 1e-3)), weight_decay=1e-4)
    scaler    = torch.amp.GradScaler('cuda') if device.type == "cuda" else None
    best_val  = 0.0
    for _ in range(args.trial_epochs):
        train_epoch(model, tr_loader, criterion, optimizer, scaler, device)
        _, _, _, _, _, val_metrics = eval_epoch(model, val_loader, criterion, device)
        if val_metrics["f1_macro"] > best_val:
            best_val = val_metrics["f1_macro"]
    return best_val


# Runs random search over the default hyperparameter space and returns the best params found.
def hp_search_random(args, X_seq, X_global, y, classes, device) -> dict:
    rng = np.random.default_rng(args.seed)
    best_p, best_val = {}, 0.0
    trials = []
    print(f"\nReBiAt random hyperparameter search: {args.n_trials} trials, "
          f"{args.trial_epochs} epochs/trial ...")
    for trial in range(1, args.n_trials + 1):
        params  = _jsonify_params(_sample_params(_DEFAULT_SPACE, rng))
        val_f1 = _trial_run(params, args, X_seq, X_global, y, classes, device)
        row = {"trial": trial, "val_macro_f1": float(val_f1), "params": params}
        trials.append(row)
        print(f"  Trial {trial:3d}: val_macro_f1={val_f1:.4f} | {params}")
        if val_f1 > best_val:
            best_val, best_p = val_f1, params
    save_results({
        "metric": "validation_macro_f1",
        "search_space": _DEFAULT_SPACE,
        "trial_epochs": args.trial_epochs,
        "n_trials": args.n_trials,
        "trials": trials,
        "best_params": best_p,
        "best_val_macro_f1": best_val,
    },
                 f"{args.results_dir}/hp_random")
    return best_p


# Runs Optuna-based Bayesian hyperparameter search and returns the best params found.
def hp_search_optuna(args, X_seq, X_global, y, classes, device) -> dict:
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("Install optuna: pip install optuna")

    def objective(trial):
        params = {
            "lr":              trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "batch_size":      trial.suggest_categorical("batch_size", [32, 64, 128]),
            "dropout_enc":     trial.suggest_float("dropout_enc", 0.2, 0.5),
            "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.2),
            "gru_hidden":      trial.suggest_categorical("gru_hidden", [64, 128, 256]),
        }
        return _trial_run(params, args, X_seq, X_global, y, classes, device)

    study = optuna.create_study(
        direction="maximize", study_name="kfp_optuna",
        storage=f"sqlite:///{args.results_dir}/optuna.db", load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    best_p = study.best_params
    save_results({
        "metric": "validation_macro_f1",
        "search_space": _DEFAULT_SPACE,
        "trial_epochs": args.trial_epochs,
        "n_trials": args.n_trials,
        "trials": [
            {
                "trial": t.number,
                "val_macro_f1": None if t.value is None else float(t.value),
                "params": t.params,
                "state": str(t.state),
            }
            for t in study.trials
        ],
        "best_params": best_p,
        "best_val_macro_f1": study.best_value,
    },
                 f"{args.results_dir}/hp_optuna")
    return best_p


# ---------------------------------------------------------------------------
# Main training run
# ---------------------------------------------------------------------------

# Runs the full training pipeline: data loading, optional HP search, train/val/test loop, and result saving.
def run(args):
    set_global_seed(args.seed)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seed: {SEED}")

    X_seq, X_global, y, classes = load_npz(args.npz)
    chrono_key = load_chronological_key(args.npz)
    n_classes = len(classes)
    print(f"Loaded: {len(y)} samples | {n_classes} classes | "
          f"X_seq {X_seq.shape} | X_global {X_global.shape}")

    # ---- Leave-m-out (Methodology 7.3): hold the last m keywords out of training
    #      ENTIRELY; their traces become the Unknown set for open-world eval. The
    #      remaining (K-m) labels stay contiguous 0..K-m-1, so no remap is needed. ----
    X_seq_unk = X_gl_unk_raw = None
    if args.leave_out > 0:
        m = int(args.leave_out)
        keep = y < (n_classes - m)
        X_seq_unk, X_gl_unk_raw = X_seq[~keep], X_global[~keep]
        X_seq, X_global, y = X_seq[keep], X_global[keep], y[keep]
        if chrono_key is not None:
            chrono_key = chrono_key[keep]
        classes = classes[: n_classes - m]
        n_classes -= m
        print(f"Leave-{m}-out: train on {n_classes} known classes; "
              f"{len(X_seq_unk)} traces of {m} held-out keywords reserved as Unknown")

    best_params: dict = {}
    if args.hp_search == "random":
        best_params = hp_search_random(args, X_seq, X_global, y, classes, device)
        args.lr = best_params.get("lr", args.lr)
        args.batch_size = int(best_params.get("batch_size", args.batch_size))
    elif args.hp_search == "optuna":
        best_params = hp_search_optuna(args, X_seq, X_global, y, classes, device)
        args.lr = best_params.get("lr", args.lr)
        args.batch_size = int(best_params.get("batch_size", args.batch_size))
    elif args.hp_json:
        _d = json.loads(Path(args.hp_json).read_text())
        best_params = _d.get("best_params", _d) or {}
        args.lr = float(best_params.get("lr", args.lr))
        args.batch_size = int(best_params.get("batch_size", args.batch_size))
        print(f"Fixed hyperparameters from {args.hp_json}: {best_params}")

    if args.kfold > 1:
        run_kfold(args, X_seq, X_global, y, classes, device, k=args.kfold)
        if not args.train_final:
            return

    if args.chrono_split:
        tr_idx, val_idx, te_idx = split_chronological(y, order_key=chrono_key)
    else:
        tr_idx, val_idx, te_idx = split_stratified(y)
    print(f"Split: train={len(tr_idx)} val={len(val_idx)} test={len(te_idx)}")
    summarize_split("Train", y[tr_idx], classes)
    summarize_split("Val", y[val_idx], classes)
    summarize_split("Test", y[te_idx], classes)

    X_gl_tr, X_gl_val, X_gl_te, selector, sel_idx = prepare_global_features(
        X_global[tr_idx], y[tr_idx], X_global[val_idx], X_global[te_idx], k=args.k_features)
    X_gl_tr, X_gl_val, X_gl_te, gl_scaler = scale_global(X_gl_tr, X_gl_val, X_gl_te)

    aug_kwargs = {"drop_rate": 0.10, "time_sigma": 0.15,
                  "size_sigma": 0.03, "apply_prob": 0.50, "augment": True}
    tr_loader  = make_loader(X_seq[tr_idx],  X_gl_tr,  y[tr_idx],  args.batch_size, True,
                             augment=args.use_augment, aug_kwargs=aug_kwargs, n_classes=n_classes,
                             num_workers=args.loader_workers)
    val_loader = make_loader(X_seq[val_idx], X_gl_val, y[val_idx], args.batch_size, False,
                             num_workers=args.loader_workers)
    te_loader  = make_loader(X_seq[te_idx],  X_gl_te,  y[te_idx],  args.batch_size, False,
                             num_workers=args.loader_workers)

    dropout_enc = float(best_params.get("dropout_enc", 0.30))
    gru_hidden  = int(best_params.get("gru_hidden",   128))
    model = KeywordClassifier(
        n_classes=n_classes, global_feat=X_gl_tr.shape[1],
        seq_feat=X_seq.shape[2], dropout_enc=dropout_enc, gru_hidden=gru_hidden,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    # torch.compile: fuses kernels for 10-30% faster training - zero quality change.
    # Optimizer is created from model.parameters() so gradients still land on the
    # original tensors; best_state is read from model.state_dict() (uncompiled).
    _model_for_train = model
    if device.type == "cuda" and hasattr(torch, "compile"):
        try:
            _model_for_train = torch.compile(model, mode="reduce-overhead")
            print("torch.compile: enabled (reduce-overhead)")
        except Exception as e:
            print(f"torch.compile: skipped - {e}")

    label_smoothing = float(best_params.get("label_smoothing", 0.1))
    if args.loss == "focal":
        criterion = FocalLoss(gamma=2.0, label_smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    optimizer  = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler     = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    best_val_score, best_state, patience_cnt = -1.0, None, 0
    best_val_metrics: dict = {}
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_macro_f1": [],
        "lr": []
    }

    print(f"\n{'Epoch':>6} {'TrainLoss':>10} {'TrainAcc':>10} {'ValLoss':>10} {'ValAcc':>10} {'ValF1':>10}")
    print("-" * 76)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc         = train_epoch(_model_for_train, tr_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc, _, _, _, val_metrics = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        history["train_loss"].append(round(tr_loss, 4))
        history["train_acc"].append(round(tr_acc,   4))
        history["val_loss"].append(round(val_loss,  4))
        history["val_acc"].append(round(val_acc,    4))
        history["val_macro_f1"].append(round(val_metrics["f1_macro"], 4))
        history["lr"].append(round(lr, 6))

        print(f"{epoch:>6d} {tr_loss:>10.4f} {tr_acc:>10.4f} "
              f"{val_loss:>10.4f} {val_acc:>10.4f} {val_metrics['f1_macro']:>10.4f} "
              f"({time.time()-t0:.1f}s)")

        current_score = val_metrics["f1_macro"] if args.early_stop_metric == "macro_f1" else val_acc
        if current_score > best_val_score:
            best_val_score = current_score
            best_val_metrics = dict(val_metrics)
            best_val_metrics["loss"] = float(val_loss)
            best_state   = {k2: v.cpu().clone() for k2, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (best {args.early_stop_metric}: {best_val_score:.4f})")
                break

    model.load_state_dict(best_state)

    test_loss, test_acc, y_pred, y_true, y_prob, test_metrics = eval_epoch(model, te_loader, criterion, device)
    print(f"\nBest val metrics: acc={best_val_metrics.get('acc', 0.0):.4f} "
          f"precision={best_val_metrics.get('precision_macro', 0.0):.4f} "
          f"recall={best_val_metrics.get('recall_macro', 0.0):.4f} "
          f"f1={best_val_metrics.get('f1_macro', 0.0):.4f}")
    print(f"Test metrics: acc={test_acc:.4f} macro_f1={test_metrics['f1_macro']:.4f} "
          f"precision={test_metrics['precision_macro']:.4f} "
          f"recall={test_metrics['recall_macro']:.4f}")
    report = classification_report(
        y_true, y_pred, target_names=classes, digits=3, output_dict=True, zero_division=0
    )
    print(classification_report(y_true, y_pred, target_names=classes, digits=3, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    print("Per-class accuracy:")
    for i, cls in enumerate(classes):
        row = cm[i].sum()
        print(f"  {cls:<30} {cm[i,i]:>4}/{row:<4}  ({cm[i,i]/max(row,1)*100:.1f}%)")

    model_path = f"{args.results_dir}/{Path(args.npz).stem}_resnet_bigru.pt"
    ckpt = {
        "model_state":  best_state,
        "model_arch": {
            "model_type":  "resnet_bigru",
            "n_classes":   n_classes,
            "global_feat": int(X_gl_tr.shape[1]),
            "seq_feat":    int(X_seq.shape[2]),
            "gru_hidden":  gru_hidden,
            "dropout_enc": dropout_enc,
        },
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
    torch.save(ckpt, model_path)
    # Stable alias expected by open_world_pipeline.py
    best_model_path = f"{args.results_dir}/best_model.pt"
    shutil.copy(model_path, best_model_path)
    print(f"\nModel saved: {model_path}")
    print(f"Also saved:  {best_model_path}")

    np.save(f"{args.results_dir}/confusion_matrix.npy", cm)

    if args.save_attn:
        save_attention_maps(model, te_loader, classes, device,
                            out_path=f"{args.results_dir}/attention_maps.npz")

    ood_results: dict = {}
    if args.ood_eval:
        if X_seq_unk is None:
            print("Skipping OOD eval: pass --leave_out N (>0) for a valid "
                  "leave-m-out open-world protocol (Methodology 7.3).")
        else:
            try:
                from open_world import evaluate_open_world
                # Apply the SAME feature selector + scaler (fit on known-train) to
                # the held-out Unknown traces, exactly as for the known test set.
                X_gl_unk = selector.transform(X_gl_unk_raw)
                X_gl_unk = gl_scaler.transform(X_gl_unk).astype(np.float32)
                unk_loader = make_loader(
                    X_seq_unk, X_gl_unk,
                    np.zeros(len(X_seq_unk), dtype=np.int64), args.batch_size, False)
                # In-distribution = held-out TEST split of the known (trained) classes.
                print(f"\nOpen-world (leave-{args.leave_out}-out): "
                      f"{len(te_idx)} known-test vs {len(X_seq_unk)} unknown traces")
                ood_results = evaluate_open_world(
                    model, te_loader, unk_loader, device,
                    methods=("softmax", "energy", "mahalanobis"),
                )
            except ImportError:
                print("open_world.py not found - skipping OOD evaluation.")

    save_results({
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
        "ood_results":  ood_results,
        "best_params":  best_params,
        "args":         vars(args),
    }, f"{args.results_dir}/run")

    return model, ood_results


# Alias: open_world.py, train_baselines.py, and visualize.py import this name.
select_global_features = prepare_global_features

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # On Kaggle: /kaggle/working
    _WORK = "."

    # set this to your .npz file path
    DATASETS = {
        "macos_26_50": f"{_WORK}/dataset_kfp_v2_macos_26_50.npz",
    }
    SELECT = "macos_26_50"

    parser = argparse.ArgumentParser(description="Train ReBiAt keyword fingerprinting model")
    parser.add_argument("--npz",          default=DATASETS[SELECT])  # set this to your .npz file path
    parser.add_argument("--k_features",   type=int,   default=15,
                        help="Deprecated/ignored: the 15 global features are "
                             "curated offline; all are kept at runtime.")
    parser.add_argument("--epochs",       type=int,   default=80)
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--loader_workers", type=int, default=2)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--patience",     type=int,   default=12)
    parser.add_argument("--results_dir",  default="./results")  # On Kaggle: /kaggle/working/results
    parser.add_argument("--chrono_split", action="store_true")
    parser.add_argument("--use_augment",  action="store_true")
    parser.add_argument("--kfold",        type=int,   default=0)
    parser.add_argument("--train_final",  action="store_true")
    parser.add_argument("--hp_search",    choices=["none", "random", "optuna"], default="none")
    parser.add_argument("--n_trials",     type=int,   default=20)
    parser.add_argument("--trial_epochs", type=int,   default=30,
                        help="epochs per hyperparameter-search trial before final retraining")
    parser.add_argument("--hp_json",      default=None,
                        help="Path to a JSON containing fixed best hyperparameters (a flat "
                             "dict or {'best_params': {...}}). When set with "
                             "--hp_search none, the run uses those fixed "
                             "hyperparameters (lr, batch_size, gru_hidden, dropout_enc, "
                             "label_smoothing) instead of the defaults, without "
                             "re-running the search.")
    parser.add_argument("--save_attn",    action="store_true")
    parser.add_argument("--loss",         choices=["ce", "focal"], default="focal",
                        help="ce=CrossEntropy  focal=FocalLoss(gamma=2)")
    parser.add_argument("--early_stop_metric", choices=["macro_f1", "acc"], default="macro_f1")
    parser.add_argument("--ood_eval",     action="store_true")
    parser.add_argument("--leave_out",    type=int, default=0,
                        help="hold out the last N keywords ENTIRELY from training and use "
                             "them as Unknown for true leave-m-out open-world (Methodology 7.3); "
                             "use together with --ood_eval")

    args = parser.parse_args()
    run(args)
