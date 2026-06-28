"""
feature_selection.py
Three-method feature selection for keyword fingerprinting.

Methods applied to (X_global ++ seq_summary_stats):
  1. ANOVA F-score   (SelectKBest + f_classif)
  2. Mutual information  (mutual_info_classif)
  3. Tree-based      (RandomForest + SelectFromModel)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import kurtosis as sp_kurtosis, skew as sp_skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    VarianceThreshold,
    f_classif,
    mutual_info_classif,
)
import joblib


# ---------------------------------------------------------------------------
# Sequence summary statistics
# ---------------------------------------------------------------------------

def seq_summary_stats(X_seq: np.ndarray) -> np.ndarray:
    """
    Extract per-channel summary statistics from sequence data.
    X_seq: (N, L, C)  where C=3 (direction, size_norm, iat_log)
    Returns: (N, 30)
    """
    N, L, C = X_seq.shape
    feats = []

    for c in range(C):
        xc = X_seq[:, :, c]
        feats.append(xc.mean(axis=1))
        feats.append(xc.std(axis=1))
        feats.append(np.median(xc, axis=1))
        feats.append(np.percentile(xc, 25, axis=1))
        feats.append(np.percentile(xc, 75, axis=1))
        feats.append(xc.max(axis=1))
        feats.append(xc.min(axis=1))
        feats.append(sp_kurtosis(xc, axis=1, nan_policy="omit"))
        feats.append(sp_skew(xc,     axis=1, nan_policy="omit"))

    nonzero = (X_seq != 0).sum(axis=1)
    for c in range(C):
        feats.append(nonzero[:, c].astype(float))

    return np.nan_to_num(np.vstack(feats).T, nan=0.0, posinf=0.0, neginf=0.0)


_CH_NAMES   = ["dir", "size", "iat"]
_STAT_NAMES = ["mean", "std", "median", "p25", "p75", "max", "min", "kurtosis", "skewness"]

SEQ_FEATURE_NAMES: list[str] = (
    [f"{stat}_{ch}" for ch in _CH_NAMES for stat in _STAT_NAMES]
    + [f"nonzero_{ch}" for ch in _CH_NAMES]
)

GLOBAL_FEATURE_NAMES: list[str] = [
    # Group A - Packet size distribution
    "avg_size_norm", "std_size_norm",
    "hist_small",    "hist_mid",    "hist_large",   "hist_max",
    # Group B - Timing & throughput
    "avg_iat_log",   "std_iat_log", "throughput_log", "flow_duration_log",
    # Group C - Traffic asymmetry
    "incoming_bytes_ratio", "asymmetry_log",
    # Group D - Server response & burst activity
    "mean_resp_log", "std_resp_log", "burst_count_norm",
]

ALL_FEATURE_NAMES: list[str] = GLOBAL_FEATURE_NAMES + SEQ_FEATURE_NAMES


# ---------------------------------------------------------------------------
# Main selection routine
# ---------------------------------------------------------------------------

# Run all three feature selection methods on the dataset and save selectors to disk.
def run(
    npz_path: str = "dataset_kfp_v2.npz",
    k_select: int = 12,
    out_prefix: str = "selector",
) -> dict:
    data    = np.load(npz_path, allow_pickle=True)
    X_seq   = data["X_seq"]
    X_gl    = data["X_global"]
    y       = data["y"]

    X_seq_stats = seq_summary_stats(X_seq)
    X_all       = np.concatenate([X_gl, X_seq_stats], axis=1)
    all_names   = list(ALL_FEATURE_NAMES)

    print(f"Feature matrix (raw): {X_all.shape}  "
          f"({X_gl.shape[1]} global + {X_seq_stats.shape[1]} seq-stats)")

    vt           = VarianceThreshold(threshold=0.0)
    X_all        = vt.fit_transform(X_all)
    kept_mask    = vt.get_support()
    removed_idx  = np.where(~kept_mask)[0]
    if len(removed_idx):
        removed_names = [all_names[i] if i < len(all_names) else f"feat_{i}"
                         for i in removed_idx]
        print(f"Removed {len(removed_idx)} constant feature(s): {removed_names}")
    all_names = [n for n, keep in zip(all_names, kept_mask) if keep]
    k         = min(k_select, X_all.shape[1])

    print(f"Feature matrix (after constant filter): {X_all.shape}")

    # 1) ANOVA
    anova = SelectKBest(f_classif, k=k).fit(X_all, y)
    joblib.dump(anova, f"{out_prefix}_anova.joblib")

    # 2) Mutual information
    mi_scores = mutual_info_classif(X_all, y, random_state=42)
    mi_idx    = np.argsort(mi_scores)[::-1][:k]
    joblib.dump(mi_idx, f"{out_prefix}_mi_indices.npy")
    np.save(f"{out_prefix}_mi_scores.npy", mi_scores)

    # 3) RandomForest
    rf  = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_all, y)
    sel = SelectFromModel(rf, prefit=True, max_features=k)
    joblib.dump(sel, f"{out_prefix}_rf_sel.joblib")

    def _names(indices):
        return [all_names[i] if i < len(all_names) else f"feat_{i}" for i in indices]

    anova_idx = anova.get_support(indices=True)
    rf_idx    = np.where(sel.get_support())[0]
    print(f"\nTop-{k} features selected:")
    print(f"  ANOVA: {_names(anova_idx)}")
    print(f"  MI:    {_names(mi_idx)}")
    print(f"  RF:    {_names(rf_idx)}")

    return {"anova": anova, "mi_idx": mi_idx, "rf_sel": sel, "X_all": X_all, "y": y}


if __name__ == "__main__":
    _WORK = "."  # On Kaggle: /kaggle/working
    # set this to your .npz file path
    run(npz_path=f"{_WORK}/dataset_kfp_v2_macos_26_50.npz", k_select=12, out_prefix=f"{_WORK}/selector")
