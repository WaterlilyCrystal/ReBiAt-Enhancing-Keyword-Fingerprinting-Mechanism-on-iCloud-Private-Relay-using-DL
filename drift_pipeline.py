"""
drift_pipeline.py - End-to-end concept drift experiment (Methodology Section 9: Concept Drift)

Pipeline:
  Stage 1  Extract S1 features - optional, skipped if --s1_npz is given
  Stage 2  Load / train Session-1 checkpoint - skipped if --s1_checkpoint is given
  Stage 3  Extract S2 features - optional, skipped if --s2_npz is given
  Stage 4  Drift diagnostics - auxiliary embedding-shift summary (saved for inspection)
  Stage 5  Adaptation - F0 (baseline), F1, F3, F3_AUG, F_TEMP, F_TPROTO
  Stage 6  Summary table - pre vs post-adaptation, all metrics side-by-side

References:
  Szkely & Rizzo (2013)            - Energy Distance as auxiliary drift diagnostic
  Jordaney et al. (USENIX 2021)    - CADE concept drift explanation
  Yang et al.     (NDSS 2023)      - DOTS drift-aware traffic classification
  Swallow / Guo   (CCS 2023)       - RobustAugment (fluctuation/aggregation/flatten)

Example usage:
  python drift_pipeline.py \\
      --s1_npz ./features/session1_10kw.npz \\
      --s2_npz ./features/session2_10kw.npz \\
      --s1_checkpoint ./drift_results/session1_10kw_resnet_bigru.pt \\
      --results ./drift_results \\
      --n_shots 20 \\
      --strategies F1,F3,F3_AUG,F_TEMP,F_TPROTO
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

DRIFT_KEYWORDS = [
    "espn",
    "eagles schedule",
    "elon musk",
    "ebay",
    "bengals",
    "airbnb",
    "did anyone win the powerball",
    "amazon",
    "dow jones",
    "cnn",
]


#
# Stage 1 & 3 - Feature extraction
#

# Resolves the directory that directly contains keyword subfolders, handling Kaggle's extra-nesting layout.
def _resolve_data_root(data_dir: str, keywords: list[str]) -> str:
    """
    Find the directory that DIRECTLY contains keyword subfolders.

    Kaggle datasets sometimes wrap files in an extra subfolder
    (e.g. drift-10kw-after/macos-after/ or drift-10kw-after/drift-10kw-after/).
    This function tries data_dir first, then every immediate subdirectory,
    and returns the first one that contains at least one keyword folder.

    Raises RuntimeError with a diagnostic listing if nothing is found.
    """
    import os

    def _normalize(name: str) -> str:
        return " ".join(name.strip().replace("_", " ").split()).lower()

    kw_norm = {_normalize(k) for k in keywords}

    def _has_keyword_folder(directory: str) -> bool:
        try:
            found = {
                _normalize(d)
                for d in os.listdir(directory)
                if os.path.isdir(os.path.join(directory, d))
            }
            return bool(found & kw_norm)
        except OSError:
            return False

    # 1. Try the path given directly
    if _has_keyword_folder(data_dir):
        return data_dir

    # 2. Try each immediate subdirectory (one level deeper)
    try:
        subdirs = sorted(
            os.path.join(data_dir, d)
            for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        )
    except OSError:
        subdirs = []

    for subdir in subdirs:
        if _has_keyword_folder(subdir):
            print(f"  Auto-detected keyword root: {subdir}")
            return subdir

    # 3. Fail with a diagnostic so the user can fix the path manually
    try:
        top_contents = sorted(os.listdir(data_dir))
    except OSError:
        top_contents = ["(cannot list directory)"]

    sub_contents: dict[str, list] = {}
    for subdir in subdirs[:5]:          # show up to 5 subdirs
        try:
            sub_contents[Path(subdir).name] = sorted(os.listdir(subdir))[:20]
        except OSError:
            pass

    raise RuntimeError(
        f"\nNo keyword folders found under: {data_dir}\n"
        f"  Top-level contents ({len(top_contents)} items): {top_contents}\n"
        + "\n".join(
            f"  {k}/  ({len(v)} items): {v}"
            for k, v in sub_contents.items()
        )
        + f"\n  Looking for (normalized): {sorted(kw_norm)}\n"
        f"  Tip: pass the exact path that DIRECTLY contains the keyword subfolders\n"
        f"       via --s2_dir (or --s1_dir), e.g. --s2_dir <path>/<subfolder>"
    )


# Extracts PCAP features for DRIFT_KEYWORDS from data_dir and saves a .npz file for the given session.
def extract_session(
    data_dir: str,
    out_npz:  str,
    session_tag: str,
    n_workers: int = 2,
    min_payload: int = 10,
    min_valid_packets: int = 5,
    max_packets: int = 500,
    skip_if_exists: bool = True,
) -> str:
    """
    Extract PCAP features for DRIFT_KEYWORDS from data_dir (auto-detects subfolder
    layout if keyword folders are one level deeper than data_dir).
    Saves a .npz compatible with train_resnet_bigru.py.
    Returns the path to the .npz file.
    """
    if skip_if_exists and Path(out_npz).exists():
        print(f"  [{session_tag}] {out_npz} already exists - skipping extraction")
        return out_npz

    _ensure_kaggle_path()
    from extract_features_v2 import Config, build_dataset, analyze_features

    # Auto-detect the directory that directly contains keyword folders
    resolved_dir = _resolve_data_root(data_dir, DRIFT_KEYWORDS)

    cfg = Config(
        data_dir=resolved_dir,
        target_port=443,
        min_payload=min_payload,
        min_valid_packets=min_valid_packets,
        max_packets=max_packets,
        n_workers=n_workers,
        output_file=out_npz,
        classes_filter=DRIFT_KEYWORDS,
    )
    print(f"\n  Extracting {session_tag} from: {resolved_dir}")
    print(f"  Keywords: {DRIFT_KEYWORDS}")
    build_dataset(cfg)

    if not Path(out_npz).exists():
        raise RuntimeError(
            f"Feature extraction failed - {out_npz} was not created.\n"
            f"Source dir: {resolved_dir}\n"
            "Verify that .pcap/.pcapng files exist inside the keyword subfolders."
        )
    analyze_features(out_npz)
    return out_npz


# Loads and summarizes an npz dataset file, printing per-class sample counts.
def inspect_npz(npz_path: str) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    classes = [str(c).replace("_", " ").strip() for c in data["classes"].tolist()]
    y = data["y"]
    counts = np.bincount(y, minlength=len(classes))
    print(f"  {Path(npz_path).name}: {len(y)} samples | {len(classes)} classes")
    for i, cls in enumerate(classes):
        print(f"    [{i:2d}] {cls:<35} {counts[i]:>4} traces")
    return {"n_classes": len(classes), "classes": classes, "n_samples": int(len(y))}


#
# Stage 2 - Training
#

# Returns the checkpoint path that train_resnet_bigru.py writes for a given npz file.
def find_checkpoint(results_dir: str, npz_path: str) -> str:
    """Return the path train_resnet_bigru.py writes the model to."""
    stem = Path(npz_path).stem
    return str(Path(results_dir) / f"{stem}_resnet_bigru.pt")


def train_session1(
    s1_npz: str,
    results_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    k_features: int,
    seed: int,
    skip_if_exists: bool = True,
) -> str:
    """
    Train ResNet-10+BiGRU on Session-1 data via subprocess.
    Returns path to saved checkpoint (.pt).
    """
    ckpt = find_checkpoint(results_dir, s1_npz)
    if skip_if_exists and Path(ckpt).exists():
        print(f"  Checkpoint exists - skipping training: {ckpt}")
        return ckpt

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(Path(__file__).parent / "train_resnet_bigru.py"),
        "--npz",          s1_npz,
        "--results_dir",  results_dir,
        "--epochs",       str(epochs),
        "--batch_size",   str(batch_size),
        "--lr",           str(lr),
        "--patience",     str(patience),
        "--k_features",   str(k_features),
        "--seed",         str(seed),
        "--loss",         "focal",
        "--early_stop_metric", "macro_f1",
        "--loader_workers", "2",
    ]
    print("\n  $", " ".join(cmd))
    subprocess.run(cmd, check=True)

    if not Path(ckpt).exists():
        raise RuntimeError(
            f"Training finished but checkpoint not found: {ckpt}\n"
            "Verify that train_resnet_bigru.py is the updated version with "
            "gl_scaler + model_arch in its torch.save() call."
        )
    print(f"  Checkpoint saved: {ckpt}")
    return ckpt


#
# Model loading
#

# Ensures the local script directory is on sys.path so sibling modules can be imported.
def _ensure_kaggle_path() -> None:
    if str(Path(__file__).parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).parent))


# Active model builder for the whole pipeline. Set by load_checkpoint() from the
# checkpoint's model_type so all adaptation strategies build the RIGHT architecture
# (thesis model OR a Var-CNN / NetCLR baseline) from the same arch kwargs. Stays
# None -> KeywordClassifier for legacy checkpoints, so behaviour is unchanged.
_MODEL_BUILDER = None


# Sets the global model builder based on model_type string from a checkpoint.
def _set_model_builder(model_type: str) -> None:
    global _MODEL_BUILDER
    _ensure_kaggle_path()
    if model_type in (None, "resnet_bigru"):
        from train_resnet_bigru import KeywordClassifier
        _MODEL_BUILDER = KeywordClassifier
    else:
        import baselines as _B
        _MODEL_BUILDER = _B.get_builder(model_type)


def _get_model_class():
    """Return the active model builder. Every adaptation strategy instantiates via
    this, so a single dispatch in load_checkpoint() makes them all model-agnostic.
    The builder accepts (n_classes, global_feat, seq_feat, gru_hidden, dropout_enc);
    baselines ignore gru_hidden."""
    global _MODEL_BUILDER
    if _MODEL_BUILDER is not None:
        return _MODEL_BUILDER
    _ensure_kaggle_path()
    from train_resnet_bigru import KeywordClassifier
    return KeywordClassifier


def _build_model(arch: dict, device: torch.device | None = None) -> nn.Module:
    """Instantiate the registered model class from arch kwargs."""
    m = _get_model_class()(
        n_classes   = arch["n_classes"],
        global_feat = arch["global_feat"],
        seq_feat    = arch["seq_feat"],
        gru_hidden  = arch.get("gru_hidden", 128),
        dropout_enc = arch.get("dropout_enc", 0.30),
    )
    return m.to(device) if device is not None else m


def _load_model(arch: dict, state: dict, device: torch.device) -> nn.Module:
    """Build model from arch, load cleaned state dict, move to device."""
    m = _build_model(arch, device)
    m.load_state_dict(_clean_state_dict(state))
    return m


def _clean_state_dict(state: dict) -> dict:
    """Normalize checkpoints saved from wrappers such as torch.compile/DataParallel."""
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


def load_checkpoint(ckpt_path: str, device: torch.device):
    """Load model + preprocessing artifacts from a training checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    required = {"model_state", "model_arch", "classes", "selected_idx", "gl_scaler"}
    missing  = required - set(ckpt.keys())
    if missing:
        raise KeyError(
            f"Checkpoint missing keys: {missing}\n"
            "Regenerate by re-running the updated train_resnet_bigru.py / train_baselines.py."
        )
    arch = ckpt["model_arch"]
    _set_model_builder(arch.get("model_type", "resnet_bigru"))
    model = _build_model(arch, device)
    model.load_state_dict(_clean_state_dict(ckpt["model_state"]))
    return model, ckpt


#
# Data helpers
#

def load_and_remap(npz_path: str, s1_classes: list[str]):
    """
    Load npz, keep only samples whose class name is in s1_classes,
    remap y-labels to S1 indices so both sessions share the same label space.
    """
    data     = np.load(npz_path, allow_pickle=True)
    X_seq    = data["X_seq"].astype(np.float32)
    X_global = data["X_global"].astype(np.float32)
    y        = data["y"].astype(np.int64)
    raw_cls  = [str(c).replace("_", " ").strip() for c in data["classes"].tolist()]

    s1_map = {name: i for i, name in enumerate(s1_classes)}
    keep   = np.zeros(len(y), dtype=bool)
    y_new  = np.full(len(y), -1, dtype=np.int64)
    for s2_i, name in enumerate(raw_cls):
        if name in s1_map:
            m = y == s2_i
            keep |= m
            y_new[m] = s1_map[name]

    X_seq, X_global, y_new = X_seq[keep], X_global[keep], y_new[keep]
    overlap = sorted(s1_classes[i] for i in np.unique(y_new))
    return X_seq, X_global, y_new, overlap


# Applies the S1-fitted feature selector and scaler to new data arrays.
def apply_preprocessors(X_seq: np.ndarray, X_global: np.ndarray, ckpt: dict):
    """Apply S1-fitted feature selector + scaler to any new data."""
    sel_idx = np.array(ckpt["selected_idx"])
    scaler  = ckpt["gl_scaler"]
    X_gl    = scaler.transform(X_global[:, sel_idx]).astype(np.float32)
    return X_seq.astype(np.float32), X_gl


# Wraps arrays into a TensorDataset and returns a DataLoader with optional shuffling.
def make_loader(X_seq, X_global, y, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(X_seq),
        torch.from_numpy(X_global),
        torch.from_numpy(y),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=torch.cuda.is_available())


# Splits indices into train/val/test using stratified sampling with fallback to random.
def stratified_split(y: np.ndarray, val: float = 0.20, test: float = 0.20, seed: int = 42):
    idx = np.arange(len(y))
    try:
        tr, tmp = train_test_split(idx, test_size=val + test, stratify=y, random_state=seed)
        va, te  = train_test_split(tmp, test_size=test / (val + test),
                                   stratify=y[tmp], random_state=seed)
    except ValueError as exc:
        print(f"  [stratified_split] WARNING: stratified split failed ({exc}); "
              "falling back to random split.")
        tr, tmp = train_test_split(idx, test_size=val + test, random_state=seed)
        va, te  = train_test_split(tmp, test_size=test / (val + test), random_state=seed)
    return tr, va, te


#
# RobustAugment (Swallow) - traffic-aware data augmentation
#

_AUG_STRATS = ["fluctuation", "aggregation", "flatten"]


def augment_trace(
    X_seq:     np.ndarray,
    strategy:  str,
    noise_std: float = 0.05,
    rate:      float = 0.30,
    seed:      int   = 0,
) -> np.ndarray:
    """
    RobustAugment (Guo et al. / Swallow) - 3 traffic-aware strategies.
    X_seq: (N, T, F)  F[0] = signed packet size; F[1..] = IPT / other features.

    fluctuation  Gaussian multiplicative noise on all feature dims
                 (simulates measurement jitter / packet-size variation).
    aggregation  Merge consecutive non-zero packets with probability=rate
                 (simulates fast network / kernel batching / GSO).
    flatten      Halve sizes of above-median packets
                 (simulates small MTU / slow path / fragmentation).
    """
    rng = np.random.default_rng(seed)
    X   = X_seq.copy().astype(np.float32)
    N, T, _ = X.shape

    if strategy == "fluctuation":
        noise = rng.normal(1.0, noise_std, X.shape).astype(np.float32)
        signs = np.sign(X[:, :, 0])
        X     = X * noise
        X[:, :, 0] = np.abs(X[:, :, 0]) * signs      # preserve direction sign

    elif strategy == "aggregation":
        for i in range(N):
            j = 0
            while j < T - 1:
                if rng.random() < rate and X[i, j, 0] != 0 and X[i, j + 1, 0] != 0:
                    X[i, j, 0]          += X[i, j + 1, 0]
                    X[i, j + 1: T - 1]   = X[i, j + 2: T]
                    X[i, T - 1]          = 0.0
                else:
                    j += 1

    elif strategy == "flatten":
        abs_s = np.abs(X[:, :, 0])
        nz    = abs_s[abs_s > 0]
        med   = float(np.median(nz)) if len(nz) > 0 else 1.0
        X[:, :, 0] = np.where(abs_s > med, X[:, :, 0] * 0.5, X[:, :, 0])

    return X



#
# Evaluation
#

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             class_names: list[str] | None = None, verbose: bool = False,
             eval_labels: list[int] | np.ndarray | None = None):
    """
    Full evaluation pass.
    Returns (metrics_dict, y_pred, y_true, y_prob, y_logits).
    """
    model.eval()
    all_pred, all_true, all_prob, all_logits = [], [], [], []
    for x_seq, x_gl, y in loader:
        logits = model(x_seq.to(device), x_gl.to(device))
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(1).cpu().numpy()
        all_pred.extend(preds)
        all_true.extend(y.numpy())
        all_prob.extend(probs)
        all_logits.extend(logits.cpu().numpy())

    y_true   = np.array(all_true)
    y_pred   = np.array(all_pred)
    y_prob   = np.array(all_prob)
    y_logits = np.array(all_logits)
    labels = np.array(sorted(np.unique(y_true)) if eval_labels is None else list(eval_labels), dtype=np.int64)
    out_of_scope = int((~np.isin(y_pred, labels)).sum())

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    metrics = {
        "acc":          float((y_true == y_pred).mean()),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
    }
    if verbose and class_names:
        target_names = [class_names[i] for i in labels if i < len(class_names)]
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="y_pred contains classes not in y_true",
                category=UserWarning,
            )
            print(classification_report(y_true, y_pred, labels=labels,
                                        target_names=target_names, digits=3,
                                        zero_division=0))
        if out_of_scope:
            print(f"  out-of-drift-set predictions: {out_of_scope}/{len(y_pred)} "
                  f"({out_of_scope / len(y_pred):.1%})")
    return metrics, y_pred, y_true, y_prob, y_logits


#
# Stage 4 - Drift detection & metric utilities
#


# Computes Energy Distance between two sample sets using subsampled pairwise L2 norms.
def energy_distance(X: np.ndarray, Y: np.ndarray, n_sub: int = 500) -> float:
    """
    Energy Distance D_E(P,Q) = 2E||X-Y|| - E||X-X'|| - E||Y-Y'||
    (Szkely & Rizzo 2013) - proper metric between distributions.
    Value interpretation: 0 = identical; higher = larger distribution gap.
    """
    rng = np.random.default_rng(0)
    if len(X) > n_sub: X = X[rng.choice(len(X), n_sub, replace=False)]
    if len(Y) > n_sub: Y = Y[rng.choice(len(Y), n_sub, replace=False)]

    def _mean_l2(A: np.ndarray, B: np.ndarray) -> float:
        return float(np.sqrt(np.sum((A[:, None] - B[None, :]) ** 2, axis=2)).mean())

    cross  = _mean_l2(X, Y)
    self_x = _mean_l2(X, X) if len(X) > 1 else 0.0
    self_y = _mean_l2(Y, Y) if len(Y) > 1 else 0.0
    return max(0.0, 2.0 * cross - self_x - self_y)


# Collects penultimate embeddings from a model over an entire DataLoader.
@torch.no_grad()
def extract_embeddings(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    embs, ys = [], []
    for x_seq, x_gl, y in loader:
        embs.append(model.get_embedding(x_seq.to(device), x_gl.to(device)).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(embs), np.concatenate(ys)


def compute_drift_signals(
    model: nn.Module,
    s1_loader: DataLoader,
    s2_loader: DataLoader,
    device: torch.device,
) -> dict:
    """Energy Distance D_E per class and globally (Szkely & Rizzo 2013)."""
    embs1, y1 = extract_embeddings(model, s1_loader, device)
    embs2, y2 = extract_embeddings(model, s2_loader, device)

    overlap_cls = np.intersect1d(np.unique(y1), np.unique(y2)).astype(int)
    per_class: dict[int, dict] = {}
    for cls in overlap_cls:
        e1, e2 = embs1[y1 == cls], embs2[y2 == cls]
        if len(e1) < 2 or len(e2) < 2:
            continue
        per_class[int(cls)] = {"de": round(energy_distance(e1, e2), 4)}

    de_vals   = [v["de"] for v in per_class.values()]
    de_global = energy_distance(embs1, embs2)

    return {
        "per_class": per_class,
        "mean_de":   round(float(np.mean(de_vals)), 4) if de_vals else 0.0,
        "global_de": round(de_global, 4),
    }


#
# Stage 5 - Adaptation strategies
#

# Gradient-based strategies dispatched through finetune(). F0 (no adaptation) is
# handled directly as the pre-adaptation baseline and needs no freeze spec; only
# its display label is referenced.
_STRATEGIES = {
    "F0": {
        "label":  "No Adaptation",
    },
    "F1": {
        "label":  "Classifier-only",
        "freeze": ["encoder", "bigru", "attn", "global_mlp"],
        "desc":   "Only the linear classifier head is updated",
    },
    "F3": {
        "label":  "Full Fine-tune",
        "freeze": [],
        "desc":   "All parameters updated with small LR (upper-bound reference)",
    },
}


# Returns a detached CPU clone of the model's state dict.
def _clone(model: nn.Module) -> dict:
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}


# Runs one validation epoch and returns macro-F1 score.
def _val_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """One validation pass -- returns macro-F1 on the loader."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x_seq, x_gl, y in loader:
            logits = model(x_seq.to(device), x_gl.to(device))
            preds.extend(logits.argmax(1).cpu().numpy())
            trues.extend(y.numpy())
    return float(f1_score(trues, preds, labels=np.unique(trues),
                          average="macro", zero_division=0))


# Performs an AMP-aware backward pass with gradient clipping and optimizer step.
def _amp_step(loss: torch.Tensor, optimizer: optim.Optimizer,
              model: nn.Module, amp_scaler) -> None:
    """AMP-aware backward + gradient clip + optimizer step."""
    if amp_scaler:
        amp_scaler.scale(loss).backward()
        amp_scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        amp_scaler.step(optimizer)
        amp_scaler.update()
    else:
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def finetune(
    init_state:   dict,
    strategy:     str,
    arch:         dict,
    tr_loader:    DataLoader,
    val_loader:   DataLoader,
    device:       torch.device,
    n_epochs:     int,
    lr:           float,
    patience:     int,
    aug_strategy: str | None = None,
    aug_prob:     float = 0.5,
) -> tuple[dict, list[float], list[float]]:
    """
    Fine-tune from init_state.  Freeze components per strategy.
    aug_strategy: None | 'fluctuation' | 'aggregation' | 'flatten' | 'random'
      When set, RobustAugment is applied to each training batch with
      probability aug_prob (Swallow-style data augmentation).
    Early-stop on val macro-F1.
    Returns (best_state, train_loss_hist, val_f1_hist).
    """
    model = _load_model(arch, init_state, device)

    for comp_name in _STRATEGIES[strategy]["freeze"]:
        comp = getattr(model, comp_name, None)
        if comp is not None:
            for p in comp.parameters():
                p.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        return _clone(model), [], []

    optimizer  = optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion  = nn.CrossEntropyLoss(label_smoothing=0.05)
    use_amp    = device.type == "cuda"
    amp_scaler = torch.amp.GradScaler("cuda") if use_amp else None
    rng_aug    = np.random.default_rng(42)

    best_f1, best_state, pat_cnt = -1.0, _clone(model), 0
    tr_hist, val_hist = [], []

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        for x_seq, x_gl, y in tr_loader:
            # RobustAugment (Swallow) - applied per batch with probability aug_prob
            if aug_strategy and rng_aug.random() < aug_prob:
                s = (rng_aug.choice(_AUG_STRATS)
                     if aug_strategy == "random" else aug_strategy)
                x_seq = torch.from_numpy(
                    augment_trace(x_seq.numpy(), s,
                                  seed=int(rng_aug.integers(1_000_000)))
                )
            x_seq, x_gl, y = x_seq.to(device), x_gl.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(x_seq, x_gl), y)
            _amp_step(loss, optimizer, model, amp_scaler)
            total_loss += loss.item() * len(y); n += len(y)
        scheduler.step()
        tr_hist.append(round(total_loss / n, 4))

        vf1 = round(_val_epoch(model, val_loader, device), 4)
        val_hist.append(vf1)
        if vf1 > best_f1:
            best_f1 = vf1; best_state = _clone(model); pat_cnt = 0
        else:
            pat_cnt += 1
            if pat_cnt >= patience:
                break

    return best_state, tr_hist, val_hist


def finetune_temporal(
    init_state:   dict,
    arch:         dict,
    tr_loader:    DataLoader,
    val_loader:   DataLoader,
    device:       torch.device,
    n_epochs:     int,
    lr:           float,
    patience:     int,
    l2sp_beta:    float = 0.01,
    aug_strategy: str | None = None,
    aug_prob:     float = 0.3,
) -> tuple[dict, list[float], list[float]]:
    """
    F_TEMP: Temporal-Branch Adaptation - exclusive to the proposed ResNet-BiGRU model.

    Concept drift on iCloud Private Relay concentrates in the macro-temporal
    ordering and spacing of page-loading bursts, which only the BiGRU + temporal
    attention branch models explicitly. The ResNet local encoder captures
    protocol-governed burst primitives that are comparatively drift-stable, and
    the global-statistics branch summarizes flow-level aggregates. F_TEMP
    therefore FREEZES the encoder and the global-MLP and fine-tunes only the
    drift-sensitive temporal reasoning {bigru, attn, classifier}, under three
    measures suited to the few-shot (20 traces/class) regime:

      1. Layer-wise learning rate: attention 1.5x, classifier 1x, BiGRU 0.5x.
         The low-parameter attention module, which directly pools the
         drift-sensitive GRU hidden states, is allowed to adapt fastest.
      2. L2-SP regularization (Li et al., ICML 2018, 'Explicit Inductive Bias
         for Transfer Learning with Convolutional Networks'): the trainable
         parameters are penalized for departing from their Session-1 values,
         which anchors the adaptation and prevents overfitting / catastrophic
         forgetting when only a few labels are available.
      3. Optional RobustAugment on the temporal channels (off by default).

    The method is structurally unavailable to the CNN baselines (Var-CNN,
    NetCLR), which have no temporal recurrent branch to isolate; it raises
    NotImplementedError for them so the caller can skip it cleanly.

    Returns (best_state, train_loss_hist, val_f1_hist).
    """
    model = _load_model(arch, init_state, device)
    if getattr(model, "bigru", None) is None or getattr(model, "attn", None) is None:
        raise NotImplementedError(
            "F_TEMP requires a temporal recurrent branch (bigru + attn); "
            "it is exclusive to the proposed ResNet-BiGRU architecture.")

    # Freeze the drift-stable local encoder and the global-statistics branch.
    for comp_name in ("encoder", "global_mlp"):
        comp = getattr(model, comp_name, None)
        if comp is not None:
            for p in comp.parameters():
                p.requires_grad = False

    # L2-SP anchor: a detached snapshot of the trainable parameters' S1 values.
    named_trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    anchor = {n: p.detach().clone() for n, p in named_trainable}

    def _params(name):
        comp = getattr(model, name, None)
        return [p for p in comp.parameters() if p.requires_grad] if comp is not None else []

    optimizer = optim.AdamW([
        {"params": _params("bigru"),      "lr": lr * 0.5},
        {"params": _params("attn"),       "lr": lr * 1.5},
        {"params": _params("classifier"), "lr": lr},
    ], weight_decay=1e-4)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion  = nn.CrossEntropyLoss(label_smoothing=0.05)
    use_amp    = device.type == "cuda"
    amp_scaler = torch.amp.GradScaler("cuda") if use_amp else None
    rng_aug    = np.random.default_rng(42)

    best_f1, best_state, pat_cnt = -1.0, _clone(model), 0
    tr_hist, val_hist = [], []

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        for x_seq, x_gl, y in tr_loader:
            if aug_strategy and rng_aug.random() < aug_prob:
                s = (rng_aug.choice(_AUG_STRATS)
                     if aug_strategy == "random" else aug_strategy)
                x_seq = torch.from_numpy(
                    augment_trace(x_seq.numpy(), s,
                                  seed=int(rng_aug.integers(1_000_000)))
                )
            x_seq, x_gl, y = x_seq.to(device), x_gl.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(x_seq, x_gl), y)
                if l2sp_beta > 0:
                    reg = sum(((p - anchor[n]) ** 2).sum() for n, p in named_trainable)
                    loss = loss + l2sp_beta * reg
            _amp_step(loss, optimizer, model, amp_scaler)
            total_loss += loss.item() * len(y); n += len(y)
        scheduler.step()
        tr_hist.append(round(total_loss / n, 4))

        vf1 = round(_val_epoch(model, val_loader, device), 4)
        val_hist.append(vf1)
        if vf1 > best_f1:
            best_f1 = vf1; best_state = _clone(model); pat_cnt = 0
        else:
            pat_cnt += 1
            if pat_cnt >= patience:
                break

    return best_state, tr_hist, val_hist


#
# Prototype (nearest-class-mean) adaptation
#
# Motivated by the S1->S2 t-SNE: under drift each keyword's embedding cluster
# stays internally compact but translates to a new location, so the stale linear
# head (fit to S1 cluster positions) fails while the classes remain separable.
# Re-anchoring the decision rule to the Session-2 cluster centroids therefore
# recovers accuracy without re-fitting a linear classifier from few labels.

# Collects penultimate (attention-pooled + global) embeddings and labels from a loader.
@torch.no_grad()
def _embed_all(model: nn.Module, loader: DataLoader, device: torch.device):
    """Collect penultimate (attention-pooled + global) embeddings and labels."""
    model.eval()
    embs, ys = [], []
    for x_seq, x_gl, y in loader:
        e = model.get_embedding(x_seq.to(device), x_gl.to(device))
        embs.append(e.cpu()); ys.append(y)
    return torch.cat(embs), torch.cat(ys).numpy()


def compute_prototypes(model: nn.Module, support_loader: DataLoader,
                       device: torch.device, n_classes: int, l2: bool = True):
    """Per-class mean embedding (prototype) from a labeled support loader.

    Embeddings are L2-normalized before averaging when l2=True, which yields
    cosine-NCM prototypes. Rows for classes absent from the support set are NaN.
    """
    E, Y = _embed_all(model, support_loader, device)
    if l2:
        E = F.normalize(E, dim=1)
    d = E.size(1)
    protos = torch.full((n_classes, d), float("nan"))
    for c in range(n_classes):
        m = (Y == c)
        if m.any():
            protos[c] = E[m].mean(dim=0)
    return protos


@torch.no_grad()
def evaluate_prototype(model: nn.Module, protos: torch.Tensor, loader: DataLoader,
                       device: torch.device, eval_labels, l2: bool = True) -> dict:
    """Nearest-class-mean (cosine) classification using prototypes.

    Returns a metrics dict with the same keys as evaluate() so the result can be
    recorded alongside the gradient-based strategies.
    """
    P = protos.clone()
    if l2:
        P = F.normalize(P, dim=1)
    valid = ~torch.isnan(P).any(dim=1)
    P[~valid] = 0.0
    preds, trues = [], []
    model.eval()
    for x_seq, x_gl, y in loader:
        e = model.get_embedding(x_seq.to(device), x_gl.to(device)).cpu()
        if l2:
            e = F.normalize(e, dim=1)
        sim = e @ P.T                       # (B, C) cosine similarity to prototypes
        sim[:, ~valid] = -1e9               # never predict an unseen class
        preds.extend(sim.argmax(dim=1).numpy())
        trues.extend(y.numpy())
    y_true, y_pred = np.array(trues), np.array(preds)
    labels = np.array(list(eval_labels), dtype=np.int64)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    return {
        "acc":          float((y_true == y_pred).mean()),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
    }


# Returns a failure result dict for strategies that raise an exception.
def _result_failed(label: str, pre_m: dict, elapsed: float, error: str) -> dict:
    return {
        "label":     label,
        "metrics":   pre_m,
        "tr_hist":   [], "val_hist": [],
        "elapsed_s": round(elapsed, 1),
        "n_epochs":  0, "status": "failed", "error": error,
    }


def run_drift_experiment(
    checkpoint: str,
    s1_npz:     str,
    s2_npz:     str,
    results_dir: str,
    batch_size:     int   = 64,
    seed:           int   = 42,
    adapt_epochs:   int   = 20,
    adapt_lr:       float = 1e-4,
    adapt_patience: int   = 5,
    min_s2_samples: int   = 10,
    n_shots: int | None = None,
    strategies: set | None = None,
) -> dict:
    _ALL_STRATS     = {"F1", "F3", "F3_AUG", "F_TEMP", "F_TPROTO"}
    _DEFAULT_STRATS = {"F1", "F3", "F3_AUG", "F_TEMP", "F_TPROTO"}
    run_strats = _DEFAULT_STRATS if not strategies else (set(strategies) & _ALL_STRATS)
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    #  Load S1 checkpoint
    print(f"\n[A] Loading checkpoint: {checkpoint}")
    model, ckpt = load_checkpoint(checkpoint, device)
    s1_classes  = list(ckpt["classes"])
    arch        = ckpt["model_arch"]
    s1_state    = _clone(model)
    print(f"  Classes={arch['n_classes']} | gru_hidden={arch['gru_hidden']} "
          f"| global_feat={arch['global_feat']}")

    #  Load & preprocess S2
    print(f"\n[B] Loading Session-2: {s2_npz}")
    X_seq2_r, X_gl2_r, y2_all, overlap_all = load_and_remap(s2_npz, s1_classes)
    X_seq2_all, X_gl2_all = apply_preprocessors(X_seq2_r, X_gl2_r, ckpt)

    # Per-class counts - report ALL classes before any filtering
    counts2 = np.bincount(y2_all, minlength=arch["n_classes"])
    print(f"  S2 class counts ({len(np.unique(y2_all))} classes):")
    for c in sorted(np.unique(y2_all)):
        print(f"    [{c:2d}] {s1_classes[c]:<38} {int(counts2[c]):>5} traces")

    # Split into adaptation-ready vs sparse classes
    keep_cls = sorted(int(c) for c in np.unique(y2_all) if counts2[c] >= min_s2_samples)
    skip_cls = sorted(int(c) for c in np.unique(y2_all) if counts2[c] <  min_s2_samples)
    if skip_cls:
        print(f"\n  WARNING: {len(skip_cls)} S2 class(es) have <{min_s2_samples} traces - "
              f"EXCLUDED from adaptation (F1/F3/F3_AUG) but INCLUDED in drift detection:")
        for c in skip_cls:
            print(f"    [{c:2d}] {s1_classes[c]}: {int(counts2[c])} trace(s)  "
                  f"  -> re-run Stage 3 with lower --s2_min_payload / --s2_min_valid_packets "
                  f"to try recovering more traces")

    # Adaptation dataset (filtered to classes with enough samples)
    adapt_mask   = np.isin(y2_all, keep_cls)
    X_seq2_adapt = X_seq2_all[adapt_mask]
    X_gl2_adapt  = X_gl2_all[adapt_mask]
    y2_adapt     = y2_all[adapt_mask]
    overlap      = [s1_classes[c] for c in keep_cls]
    print(f"\n  Adaptation classes ({len(overlap)}): {overlap}")

    # Full S2 loader used ONLY for drift detection (all classes, no splitting)
    bs = batch_size
    s2_drift_loader = make_loader(X_seq2_all, X_gl2_all, y2_all, bs, False)

    # Train / val / test split on adaptation-ready data
    tr2, va2, te2 = stratified_split(y2_adapt, seed=seed)

    # Few-shot: subsample train to n_shots per class (val/test unchanged)
    if n_shots is not None:
        rng_fs = np.random.default_rng(seed)
        keep_fs = []
        for cls in np.unique(y2_adapt[tr2]):
            cls_idx = tr2[y2_adapt[tr2] == cls]
            n = min(n_shots, len(cls_idx))
            keep_fs.append(rng_fs.choice(cls_idx, n, replace=False))
        tr2 = np.concatenate(keep_fs)
        print(f"  Few-shot mode: {n_shots} samples/class -> {len(tr2)} train total")

    s2_val = make_loader(X_seq2_adapt[va2], X_gl2_adapt[va2], y2_adapt[va2], bs, False)
    s2_te  = make_loader(X_seq2_adapt[te2], X_gl2_adapt[te2], y2_adapt[te2], bs, False)
    print(f"  S2 adaptation split -> train={len(tr2)} | val={len(va2)} | test={len(te2)}")

    #  Load & preprocess S1 reference (all S2 overlap classes)
    print(f"\n[C] Loading Session-1 reference: {s1_npz}")
    X_seq1_r, X_gl1_r, y1, _ = load_and_remap(s1_npz, s1_classes)
    ov_set_all = set(int(c) for c in np.unique(y2_all))  # all classes (for drift)
    mask1  = np.isin(y1, list(ov_set_all))
    X_seq1_r, X_gl1_r, y1 = X_seq1_r[mask1], X_gl1_r[mask1], y1[mask1]
    X_seq1, X_gl1 = apply_preprocessors(X_seq1_r, X_gl1_r, ckpt)
    s1_ref = make_loader(X_seq1, X_gl1, y1, bs, False)
    print(f"  S1 reference: {len(y1)} traces for {len(ov_set_all)} overlap classes")

    s2_tr = make_loader(X_seq2_adapt[tr2], X_gl2_adapt[tr2], y2_adapt[tr2], bs, True)

    #  Pre-adaptation evaluation (F0 baseline on adaptation test set)
    print(f"\n[D] Pre-adaptation  (S1 model -> S2-test, {len(overlap)} adaptation classes):")
    pre_m, _, _, _, _ = evaluate(model, s2_te, device,
                                  class_names=s1_classes, verbose=True,
                                  eval_labels=keep_cls)
    print(f"  acc={pre_m['acc']:.4f}  precision={pre_m['precision_macro']:.4f}  "
          f"recall={pre_m['recall_macro']:.4f}  f1={pre_m['f1_macro']:.4f}")

    #  Auxiliary drift diagnostic (ALL S2 including sparse classes)
    print(f"\n[E] Drift detection  (S1-ref vs ALL S2, {len(np.unique(y2_all))} classes):")
    drift = compute_drift_signals(model, s1_ref, s2_drift_loader, device)
    print(f"  Auxiliary embedding-shift summary: global_D_E={drift['global_de']:.4f}  "
          f"per_class_mean_D_E={drift['mean_de']:.4f}")

    # Per-class drift table - all classes including sparse (LOW-DATA)
    H = f"  {'Class':<35} {'D_E':>7} {'S2n':>5}"
    print(f"\n{H}")
    print("  " + "" * (len(H) - 2))
    for cls_i in sorted(ov_set_all):
        cls_name = s1_classes[cls_i] if cls_i < len(s1_classes) else str(cls_i)
        n_s2 = int(counts2[cls_i]) if cls_i < len(counts2) else 0
        if cls_i in drift["per_class"]:
            v = drift["per_class"][cls_i]
            print(f"  {cls_name:<35} {v['de']:>7.3f} {n_s2:>5}")
        else:
            print(f"  {cls_name:<35} {'N/A':>7} {n_s2:>5}")

    #  Adaptation strategies
    print("\n[F] Adaptation strategies:")
    results: dict[str, dict] = {
        "F0": {
            "label":     _STRATEGIES["F0"]["label"],
            "metrics":   pre_m,
            "tr_hist":   [],
            "val_hist":  [],
            "elapsed_s": 0.0,
            "n_epochs":  0,
        },
    }

    def _write_report(status: str = "partial", error: str | None = None) -> dict:
        """Persist progress after every completed stage so Kaggle failures do not lose work."""
        report = {
            "status":                    status,
            "error":                     error,
            "checkpoint":                checkpoint,
            "s1_npz":                    s1_npz,
            "s2_npz":                    s2_npz,
            "overlap_classes_all":       overlap_all,
            "overlap_classes_adapt":     overlap,
            "skipped_classes":           [s1_classes[c] for c in skip_cls],
            "min_s2_samples":            min_s2_samples,
            "n_overlap_all":             len(overlap_all),
            "n_overlap_adapt":           len(overlap),
            "completed_strategies":      list(results.keys()),
            "requested_strategies":      sorted(run_strats),
            "s2_split":                  {"train": int(len(tr2)), "val": int(len(va2)), "test": int(len(te2))},
            "drift_summary": {
                "global_de": drift["global_de"],
                "mean_de":   drift["mean_de"],
            },
            "drift":           drift,
            "results": {
                k: {kk: vv for kk, vv in v.items() if kk not in ("tr_hist", "val_hist")}
                for k, v in results.items()
            },
            "history": {
                k: {"train_loss": v.get("tr_hist", []), "val_f1": v.get("val_hist", [])}
                for k, v in results.items() if k != "F0"
            },
        }
        out_name = "drift_report.json" if status == "complete" else "drift_report_partial.json"
        out = Path(results_dir) / out_name
        with open(out, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"  Progress saved: {out}")
        return report

    _write_report("partial")

    def _save_and_eval(tag: str, label: str, best_w: dict,
                       tr_h: list, val_h: list, elapsed: float,
                       extra: dict | None = None) -> dict:
        """Helper: eval on S2 test, save checkpoint, return result dict."""
        m_tmp = _load_model(arch, best_w, device)
        post_m, _, _, _, _ = evaluate(m_tmp, s2_te, device,
                                       class_names=s1_classes, verbose=False,
                                       eval_labels=keep_cls)
        print(f"  acc={post_m['acc']:.4f}  precision={post_m['precision_macro']:.4f}  "
              f"recall={post_m['recall_macro']:.4f}  f1={post_m['f1_macro']:.4f}  "
              f"({elapsed:.0f}s | {len(val_h)} epochs)")
        ck_save = Path(results_dir) / f"model_{tag}.pt"
        torch.save({"model_state": best_w,
                    **{k: ckpt[k] for k in ("classes", "k_features",
                                             "selected_idx", "gl_scaler", "model_arch")}},
                   str(ck_save))
        rec = {"label": label, "metrics": post_m,
               "tr_hist": tr_h, "val_hist": val_h,
               "elapsed_s": round(elapsed, 1), "n_epochs": len(val_h)}
        if extra:
            rec.update(extra)
        return rec

    #  F1 / F3 - supervised fine-tuning
    for strat in [s for s in ("F1", "F3") if s in run_strats]:
        cfg = _STRATEGIES[strat]
        print(f"\n   {strat}: {cfg['label']}  ({cfg['desc']})")
        t0 = time.time()
        try:
            best_w, tr_h, val_h = finetune(
                s1_state, strat, arch, s2_tr, s2_val, device,
                n_epochs=adapt_epochs, lr=adapt_lr, patience=adapt_patience,
            )
            results[strat] = _save_and_eval(strat, cfg["label"], best_w,
                                            tr_h, val_h, time.time() - t0)
            _write_report("partial")
        except Exception as e:
            results[strat] = _result_failed(cfg["label"], pre_m, time.time() - t0, repr(e))
            _write_report("partial", error=f"{strat} failed: {repr(e)}")
            continue

    #  F3_AUG - Swallow-style: F3 + RobustAugment
    if "F3_AUG" in run_strats:
        print(f"\n   F3_AUG: Full Fine-tune + RobustAugment  "
              f"(random fluctuation/aggregation/flatten, p={0.5})")
        t0 = time.time()
        best_w, tr_h, val_h = finetune(
            s1_state, "F3", arch, s2_tr, s2_val, device,
            n_epochs=adapt_epochs, lr=adapt_lr, patience=adapt_patience,
            aug_strategy="random", aug_prob=0.5,
        )
        results["F3_AUG"] = _save_and_eval(
            "F3_AUG", "Full FT + RobustAugment", best_w, tr_h, val_h, time.time() - t0)

    #  F_TEMP - Temporal-branch adaptation (proposed model only)
    if "F_TEMP" in run_strats:
        print("\n   F_TEMP: Temporal-branch adaptation (proposed model only)")
        t0 = time.time()
        try:
            best_w, tr_h, val_h = finetune_temporal(
                s1_state, arch, s2_tr, s2_val, device,
                n_epochs=adapt_epochs, lr=adapt_lr, patience=adapt_patience,
                l2sp_beta=0.01, aug_strategy=None, aug_prob=0.3,
            )
            results["F_TEMP"] = _save_and_eval(
                "F_TEMP", "Temporal branch only", best_w, tr_h, val_h, time.time() - t0)
            _write_report("partial")
        except NotImplementedError as e:
            print(f"   F_TEMP skipped - not applicable to this architecture ({e}).")
        except Exception as e:
            results["F_TEMP"] = _result_failed("Temporal branch only", pre_m, time.time() - t0, repr(e))
            _write_report("partial", error=f"F_TEMP failed: {repr(e)}")

    #  F_TPROTO - Temporal-branch adaptation + prototype head (proposed model only)
    if "F_TPROTO" in run_strats:
        print("\n   F_TPROTO: Temporal-branch adaptation + prototype head (proposed model only)")
        t0 = time.time()
        try:
            best_w, _, _ = finetune_temporal(
                s1_state, arch, s2_tr, s2_val, device,
                n_epochs=adapt_epochs, lr=adapt_lr, patience=adapt_patience, l2sp_beta=0.01)
            pm     = _load_model(arch, best_w, device)
            protos = compute_prototypes(pm, s2_tr, device, arch["n_classes"])
            m      = evaluate_prototype(pm, protos, s2_te, device, keep_cls)
            results["F_TPROTO"] = {
                "label": "Temporal + Prototype", "metrics": m,
                "tr_hist": [], "val_hist": [],
                "elapsed_s": round(time.time() - t0, 1), "n_epochs": 0,
            }
            print(f"  acc={m['acc']:.4f}  precision={m['precision_macro']:.4f}  "
                  f"recall={m['recall_macro']:.4f}  f1={m['f1_macro']:.4f}")
            _write_report("partial")
        except NotImplementedError as e:
            print(f"   F_TPROTO skipped - not applicable to this architecture ({e}).")
        except Exception as e:
            results["F_TPROTO"] = _result_failed("Temporal + Prototype", pre_m, time.time() - t0, repr(e))
            _write_report("partial", error=f"F_TPROTO failed: {repr(e)}")

    #  Summary table
    W = 92
    print("\n" + "=" * W)
    print("CONCEPT DRIFT  -  ADAPTATION ABLATION STUDY  (Session 1 -> Session 2)")
    print("=" * W)
    print(f"{'Strategy':<35} {'Acc':>8} {'Prec':>8} {'Recall':>8} "
          f"{'F1':>8}  {'Acc':>7}  {'F1':>7}  {'Type':<18}")
    print("-" * W)
    base_acc = pre_m["acc"]
    base_f1  = pre_m["f1_macro"]
    _TYPE = {
        "F0":      "Baseline",
        "F1":      "Supervised TL",
        "F3":      "Supervised TL",
        "F3_AUG":  "Swallow (aug)",
        "F_TEMP":  "Temporal-only",
        "F_TPROTO":"Temporal+Proto (ours)",
    }
    for k, r in results.items():
        m      = r["metrics"]
        da     = m["acc"]      - base_acc
        df     = m["f1_macro"] - base_f1
        marker = "  <-" if k == "F0" else ""
        typ    = _TYPE.get(k, "")
        print(f"{k}: {r['label']:<30} {m['acc']:>8.4f} {m['precision_macro']:>8.4f} "
              f"{m['recall_macro']:>8.4f} {m['f1_macro']:>8.4f}  "
              f"{da:>+7.4f}  {df:>+7.4f}  {typ:<18}{marker}")
    print("=" * W)
    print("Acc / F1 are relative to F0 (no-adaptation baseline)")

    #  Save report
    report = _write_report("complete")
    print(f"\nFull report: {Path(results_dir) / 'drift_report.json'}")
    return report


#
# Main orchestrator
#

def main():
    # On Kaggle: /kaggle/working
    _W = "."

    parser = argparse.ArgumentParser(
        description="End-to-end concept drift pipeline: extract -> train -> drift analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data directories
    parser.add_argument(
        "--s1_dir",
        # Example Kaggle path: /kaggle/input/datasets/linhnpcshust/macos-26-50
        default="",
        help="Root dir of Session-1 PCAP data (contains keyword subfolders). "
             "Ignored when --s1_npz is set.",
    )
    parser.add_argument(
        "--s2_dir",
        # Example Kaggle path: /kaggle/input/datasets/linhnpcshust/drift-10kw-after
        default="",
        help="Root dir of Session-2 PCAP data (same keyword folders, new captures)",
    )
    parser.add_argument(
        "--results", default=f"{_W}/drift_results",
        help="Output directory for all stages",
    )

    # Pre-existing file shortcuts (skip extraction / training)
    parser.add_argument(
        "--s1_npz", default=None,
        help="Path to an already-extracted Session-1 .npz file. "
             "When set, Stage 1 (PCAP extraction) is skipped entirely. "
             "Example: /kaggle/input/datasets/linhnpcshust/session1-10kw/session1_10kw.npz",
    )
    parser.add_argument(
        "--s1_checkpoint", default=None,
        help="Path to an already-trained checkpoint (.pt). "
             "When set, Stage 2 (training) is skipped entirely. "
             "Example: /kaggle/input/datasets/linhnpcshust/session1-10kw/session1_10kw_resnet_bigru.pt",
    )
    parser.add_argument(
        "--s2_npz", default=None,
        help="Path to an already-extracted Session-2 .npz file. "
             "When set, Stage 3 (S2 PCAP extraction) is skipped entirely. "
             "Example: /kaggle/input/datasets/linhnpcshust/session2-10kw/session2_10kw.npz",
    )

    # Stage skip flags (for files already in --results dir)
    parser.add_argument(
        "--skip_extract", action="store_true",
        help="Skip feature extraction if .npz files already exist in --results",
    )
    parser.add_argument(
        "--skip_train", action="store_true",
        help="Skip training if checkpoint already exists in --results",
    )

    # Feature extraction
    parser.add_argument("--n_workers",              type=int, default=2)
    parser.add_argument("--min_payload",            type=int, default=10)
    parser.add_argument("--min_valid_packets",      type=int, default=5)
    parser.add_argument("--max_packets",            type=int, default=500)
    # S2-specific extraction thresholds (default to same as S1 thresholds above).
    # Lower these if S2 traces have fewer / smaller packets than S1 (see LOW-DATA warnings).
    parser.add_argument("--s2_min_payload",         type=int, default=None,
                        help="Override --min_payload for S2 extraction only. "
                             "Set to 1-5 if many S2 traces are filtered out.")
    parser.add_argument("--s2_min_valid_packets",   type=int, default=None,
                        help="Override --min_valid_packets for S2 extraction only. "
                             "Set to 1-2 if many S2 traces are filtered out.")

    # Training
    parser.add_argument("--epochs",      type=int,   default=60,
                        help="Training epochs on Session-1")
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--patience",    type=int,   default=12)
    parser.add_argument("--k_features",  type=int,   default=15)
    parser.add_argument("--seed",        type=int,   default=42)

    # Adaptation
    parser.add_argument("--adapt_epochs",   type=int,   default=20)
    parser.add_argument("--adapt_lr",       type=float, default=1e-4)
    parser.add_argument("--adapt_patience", type=int,   default=5)
    parser.add_argument("--s2_min_samples",  type=int,   default=10,
                        help="Min S2 traces per class required for adaptation (F1/F3/F3_AUG/F_TEMP/F_TPROTO). "
                             "Classes with fewer traces are excluded from adaptation "
                             "but still appear in drift detection. Default=10.")
    parser.add_argument("--n_shots", type=int, default=None,
                        help="Few-shot mode: subsample S2 train set to N labeled samples "
                             "per class (val/test unchanged). Use 20 to match the thesis setting.")
    parser.add_argument("--strategies", type=str, default="F1,F3,F3_AUG,F_TEMP,F_TPROTO",
                        help="Comma-separated adaptation strategies to run (F0 always "
                             "included). Available: F1, F3, F3_AUG, F_TEMP, F_TPROTO. "
                             "F_TEMP and F_TPROTO use the temporal branch and apply only to the proposed "
                             "ResNet-BiGRU model (skipped for baselines). "
                             "Pass --strategies all to enable every method.")

    args = parser.parse_args()
    results_dir = args.results
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    out_s1_npz = f"{results_dir}/session1_10kw.npz"
    out_s2_npz = f"{results_dir}/session2_10kw.npz"

    #  Stage 1: Extract S1 features  (or use pre-existing npz)
    if args.s1_npz:
        # User supplied a ready-made npz - skip extraction entirely
        out_s1_npz = args.s1_npz
        print("\n" + "=" * 30)
        print("STAGE 1/4  SKIPPED - using existing S1 npz")
        print("=" * 30)
        print(f"  {out_s1_npz}")
        inspect_npz(out_s1_npz)
    else:
        print("\n" + "=" * 30)
        print("STAGE 1/4  Feature extraction - Session 1")
        print("=" * 30)
        extract_session(
            data_dir=args.s1_dir,
            out_npz=out_s1_npz,
            session_tag="session1",
            n_workers=args.n_workers,
            min_payload=args.min_payload,
            min_valid_packets=args.min_valid_packets,
            max_packets=args.max_packets,
            skip_if_exists=args.skip_extract,
        )
        inspect_npz(out_s1_npz)

    #  Stage 2: Train on S1  (or use pre-existing checkpoint)
    if args.s1_checkpoint:
        # User supplied a ready-made checkpoint - skip training entirely
        checkpoint = args.s1_checkpoint
        print("\n" + "=" * 30)
        print("STAGE 2/4  SKIPPED - using existing checkpoint")
        print("=" * 30)
        print(f"  {checkpoint}")
    else:
        print("\n" + "=" * 30)
        print("STAGE 2/4  Train ResNet-10 + BiGRU - Session 1")
        print("=" * 30)
        checkpoint = train_session1(
            s1_npz=out_s1_npz,
            results_dir=results_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            k_features=args.k_features,
            seed=args.seed,
            skip_if_exists=args.skip_train,
        )

    #  Stage 3: Extract S2 features  (or use pre-existing npz)
    if args.s2_npz:
        out_s2_npz = args.s2_npz
        print("\n" + "=" * 30)
        print("STAGE 3/4  SKIPPED - using existing S2 npz")
        print("=" * 30)
        print(f"  {out_s2_npz}")
        inspect_npz(out_s2_npz)
    else:
        s2_min_payload       = args.s2_min_payload       if args.s2_min_payload       is not None else args.min_payload
        s2_min_valid_packets = args.s2_min_valid_packets if args.s2_min_valid_packets is not None else args.min_valid_packets
        print("\n" + "=" * 30)
        print("STAGE 3/4  Feature extraction - Session 2")
        print("=" * 30)
        if s2_min_payload != args.min_payload or s2_min_valid_packets != args.min_valid_packets:
            print(f"  Using S2-specific thresholds: min_payload={s2_min_payload}  "
                  f"min_valid_packets={s2_min_valid_packets}")
        extract_session(
            data_dir=args.s2_dir,
            out_npz=out_s2_npz,
            session_tag="session2",
            n_workers=args.n_workers,
            min_payload=s2_min_payload,
            min_valid_packets=s2_min_valid_packets,
            max_packets=args.max_packets,
            skip_if_exists=args.skip_extract,
        )
        inspect_npz(out_s2_npz)

    #  Stage 4: Drift detection + adaptation
    print("\n" + "=" * 30)
    print("STAGE 4/4  Drift detection & adaptation")
    print("=" * 30)
    run_drift_experiment(
        checkpoint=checkpoint,
        s1_npz=out_s1_npz,
        s2_npz=out_s2_npz,
        results_dir=results_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        adapt_epochs=args.adapt_epochs,
        adapt_lr=args.adapt_lr,
        adapt_patience=args.adapt_patience,
        min_s2_samples=args.s2_min_samples,
        n_shots=args.n_shots,
        strategies=(None if args.strategies.strip().lower() == "all"
                    else {s.strip() for s in args.strategies.split(",") if s.strip()}),
    )


if __name__ == "__main__":
    main()
