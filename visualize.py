"""
visualize.py
============
Publication-quality figures for the keyword fingerprinting thesis.

Figures produced:
  1. closed_precision_recall.pdf/svg
                                Closed-world P/R/F1-focused metric dot plot
  2. per_class_f1_summary.pdf/svg
                                Top/bottom keyword F1 dot plot + distribution
     per_class_table.csv        Same data as CSV for LaTeX
  3. per_class_table.pdf/svg    Full precision/recall/F1 table (appendix use)
  4. openworld_scores.pdf/svg   Open-world rejection score histograms
  5. tsne_drift.pdf/svg         t-SNE: S1 vs S2 embeddings - distribution shift
  6. drift_comparison.pdf/svg   Macro-F1 adaptation trajectories under drift
  7. feature_importance.pdf/svg ANOVA F-scores for 15 global features
  8. defense_bars.pdf/svg       Adaptive-attacker defense comparison

Usage:
  python visualize.py \\
    --npz      ./dataset_kfp_v2.npz \\
    --ckpt     resnet_bigru:./results_bigru/best_model.pt \\
    --ckpt     varcnn:./results_varcnn/best_model.pt \\
    --ckpt     netclr:./results_netclr/best_model.pt \\
    --drift_tsne_ckpt  resnet_bigru:./drift_bigru/session1_10kw_resnet_bigru.pt \\
    --drift_tsne_s1    resnet_bigru:./session1_10kw.npz \\
    --drift_tsne_s2    resnet_bigru:./session2_10kw.npz \\
    --drift    resnet_bigru:./drift_bigru/drift_report.json \\
    --out_dir  ./figures

All flags except --npz are repeatable and optional.  Missing data -> warning, not crash.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgb

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import traceback

sys.path.insert(0, str(Path(__file__).parent))


# Runs a 2-D t-SNE with a fallback for scikit-learn version differences.
def _fit_tsne(X, perplexity: float, seed: int):
    """Run a 2-D t-SNE in a way that works across scikit-learn versions.

    The iteration argument was renamed from `n_iter` to `max_iter` in
    scikit-learn 1.5. Kaggle images may ship either, so try the new name and
    fall back to the old one instead of crashing the whole script.
    """
    perp = max(5.0, min(perplexity, (len(X) - 1) / 3.0))   # keep perplexity valid
    common = dict(n_components=2, perplexity=perp, random_state=seed,
                  learning_rate="auto", init="pca")
    try:
        return TSNE(max_iter=1000, **common).fit_transform(X)
    except TypeError:
        return TSNE(n_iter=1000, **common).fit_transform(X)


def _safe(label: str, fn, *args, **kwargs):
    """Run one figure stage; on failure print the error and continue so that a
    single broken figure never aborts the rest of the script (and the process
    still exits 0 for the calling notebook)."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:                                  # noqa: BLE001
        print(f"  [warn] {label} failed: {e}")
        traceback.print_exc()
        return None


def _blend_with_white(color: str, amount: float) -> tuple[float, float, float]:
    """Mix a colour with white; larger amount -> lighter tone."""
    rgb = np.array(to_rgb(color), dtype=float)
    return tuple(rgb + (1.0 - rgb) * amount)


def _draw_confidence_ellipse(ax, xy: np.ndarray, color, n_std: float = 1.6) -> None:
    """Draw a covariance ellipse for one cluster; skip degenerate cases."""
    if len(xy) < 4:
        return
    cov = np.cov(xy[:, 0], xy[:, 1])
    if not np.isfinite(cov).all():
        return
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    if np.any(vals <= 0):
        return
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(vals)
    ell = mpatches.Ellipse(
        xy=xy.mean(axis=0),
        width=width,
        height=height,
        angle=angle,
        facecolor=_blend_with_white(color, 0.65),
        edgecolor=color,
        linewidth=1.3,
        alpha=0.28,
        zorder=1,
    )
    ax.add_patch(ell)


TSNE_HIGHLIGHT_CANDIDATES = (
    "door dash", "doordash", "espn", "ebay", "uber eats", "amazon",
)


# Selects which classes to highlight in t-SNE plots based on thesis examples or error rate.
def _pick_tsne_highlights(classes, y_true, y_pred, limit: int = 3) -> list[int]:
    """Prefer named thesis examples, otherwise fall back to most confused classes."""
    norm_classes = [str(c).replace("_", " ").strip().lower() for c in classes]
    picks = []
    for target in TSNE_HIGHLIGHT_CANDIDATES:
        if target in norm_classes:
            picks.append(norm_classes.index(target))
        if len(picks) >= limit:
            return picks[:limit]

    scores = []
    for ci in range(len(classes)):
        mask = y_true == ci
        err = int(np.sum(y_pred[mask] != ci)) if mask.any() else 0
        scores.append((err, ci))
    scores.sort(reverse=True)
    return [ci for err, ci in scores if err > 0][:limit]


# Attempts to resolve a Kaggle-style path to a local equivalent.
def _resolve_existing_path(path_str: str) -> str:
    """Best-effort path resolver for local runs of Kaggle-authored specs."""
    if not path_str:
        return path_str
    raw = str(path_str).strip()
    direct = Path(raw)
    if direct.exists():
        return str(direct)

    candidates = []
    if raw.startswith("/kaggle/working/"):
        candidates.append(Path.cwd() / raw.removeprefix("/kaggle/working/"))
    elif raw.startswith("/kaggle/input/"):
        # Try a basename search when the local export no longer preserves the Kaggle input tree.
        basename = Path(raw).name
        matches = list(Path.cwd().rglob(basename))
        if len(matches) == 1:
            candidates.append(matches[0])

    for cand in candidates:
        if cand.exists():
            return str(cand)
    return raw


# Resolves the drift t-SNE checkpoint with multiple fallback candidate paths.
def _resolve_drift_ckpt_path(method_name: str, ckpt_path: str) -> str:
    """Resolve the checkpoint used for drift t-SNE, with safe fallbacks."""
    resolved = _resolve_existing_path(ckpt_path)
    if Path(resolved).exists():
        return resolved

    aliases = {
        "resnet_bigru": ["resnet_bigru", "bigru"],
        "varcnn": ["varcnn"],
        "netclr": ["netclr"],
    }.get(method_name, [method_name])

    candidates = []
    for alias in aliases:
        candidates.extend([
            Path.cwd() / f"s1_{alias}_10kw" / "best_model.pt",
            Path.cwd() / "Output_ver55" / f"s1_{alias}_10kw" / "best_model.pt",
            Path.cwd() / f"results_{alias}" / "best_model.pt",
            Path.cwd() / "Output_ver55" / f"results_{alias}" / "best_model.pt",
            Path.cwd() / f"drift_{alias}" / "model_F1.pt",
            Path.cwd() / "Output_ver55" / f"drift_{alias}" / "model_F1.pt",
            Path.cwd() / f"drift_{alias}" / "model_F3.pt",
            Path.cwd() / "Output_ver55" / f"drift_{alias}" / "model_F3.pt",
        ])

    for cand in candidates:
        if cand.exists():
            print(f"  [warn] drift t-SNE checkpoint for {method_name} not found at {ckpt_path}; "
                  f"using fallback: {cand}")
            return str(cand)
    return resolved


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

METHOD_COLORS = {
    "resnet_bigru": "#D94A4A",   # stronger red for the thesis model
    "varcnn":       "#90A4AE",   # pastel blue-gray
    "netclr":       "#E69F00",   # colorblind-friendly amber
}
METHOD_LABELS = {
    "resnet_bigru": "ReBiAt",
    "varcnn":       "VarCNN",
    "netclr":       "NetCLR",
}
METHOD_MARKERS = {"resnet_bigru": "o", "varcnn": "s", "netclr": "^"}

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.labelsize":     11,
    "axes.titlesize":     12,
    "legend.fontsize":    10,
    "xtick.labelsize":    10.5,
    "ytick.labelsize":    10.5,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         150,
    "savefig.dpi":        150,
})

GLOBAL_FEATURE_NAMES = [
    "avg_size_norm", "std_size_norm",
    "hist_small", "hist_mid", "hist_large", "hist_max",
    "avg_iat_log", "std_iat_log", "throughput_log", "flow_duration_log",
    "incoming_bytes_ratio", "asymmetry_log",
    "mean_resp_log", "std_resp_log", "burst_count_norm",
]


# ---------------------------------------------------------------------------
# Data-loading helpers
# ---------------------------------------------------------------------------

# Ensures the script directory is on sys.path for sibling module imports.
def _sys_path():
    if str(Path(__file__).parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).parent))


# Lazily imports training utilities from the sibling train_resnet_bigru module.
def _imports():
    _sys_path()
    from train_resnet_bigru import (
        load_npz, split_stratified, make_loader, scale_global, select_global_features,
    )
    return load_npz, split_stratified, make_loader, scale_global, select_global_features


@torch.no_grad()
def _get_embeddings_from_npz(model, X_seq, X_gl, device, batch_size=128):
    """Run model.get_embedding on arbitrary (pre-processed) arrays."""
    _sys_path()
    from train_resnet_bigru import make_loader
    y_dummy = np.zeros(len(X_seq), dtype=np.int64)
    loader  = make_loader(X_seq, X_gl, y_dummy, batch_size=batch_size,
                          shuffle=False, num_workers=2)
    embs = []
    for xb, gb, _ in loader:
        embs.append(model.get_embedding(xb.to(device), gb.to(device)).cpu().numpy())
    return np.concatenate(embs, axis=0)


@torch.no_grad()
def load_model_and_preds(npz_path: str, ckpt_path: str, device_str: str = "cuda"):
    """
    Load a trained checkpoint, apply its preprocessing to the test split,
    run inference, and return predictions + penultimate embeddings.

    Returns: y_true, y_pred, y_prob, embeds, classes
    """
    _sys_path()
    load_npz, split_stratified, make_loader, _, _ = _imports()

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    X_seq, X_global, y, classes = load_npz(npz_path)
    _, _, te_idx = split_stratified(y)

    sel_idx = np.array(ckpt["selected_idx"], dtype=np.int64)
    X_gl_te = ckpt["gl_scaler"].transform(X_global[te_idx][:, sel_idx]).astype(np.float32)

    import baselines as _B
    model  = _B.rebuild_from_ckpt(ckpt, device, eval_mode=True)
    loader = make_loader(X_seq[te_idx], X_gl_te, y[te_idx],
                         batch_size=128, shuffle=False, num_workers=2)

    all_logits, all_embs, all_y = [], [], []
    for xb, gb, yb in loader:
        xb, gb = xb.to(device), gb.to(device)
        all_logits.append(model(xb, gb).cpu())
        all_embs.append(model.get_embedding(xb, gb).cpu())
        all_y.append(yb)

    logits = torch.cat(all_logits)
    embs   = torch.cat(all_embs).numpy()
    y_true = torch.cat(all_y).numpy()
    y_prob = torch.softmax(logits, dim=1).numpy()
    y_pred = logits.argmax(dim=1).numpy()
    return y_true, y_pred, y_prob, embs, classes


# ---------------------------------------------------------------------------
# 1. Closed-world precision / recall summary
# ---------------------------------------------------------------------------

def plot_closed_precision_recall(preds_by_method: dict, out_dir: Path) -> None:
    """Closed-world metric summary with P/R/F1 emphasized over accuracy."""
    methods = list(preds_by_method.keys())
    metric_names = [
        "Macro precision", "Macro recall", "Macro F1",
        "Weighted precision", "Weighted recall", "Weighted F1",
        "Accuracy",
    ]
    values = {}

    for mname, (y_true, y_pred, *_) in preds_by_method.items():
        p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0,
        )
        p_wtd, r_wtd, f_wtd, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0,
        )
        acc = accuracy_score(y_true, y_pred)
        values[mname] = [p_mac, r_mac, f_mac, p_wtd, r_wtd, f_wtd, acc]

    y = np.arange(len(metric_names))
    offsets = np.linspace(-0.16, 0.16, len(methods)) if len(methods) > 1 else [0.0]
    all_scores = np.array([v for vals in values.values() for v in vals], dtype=float)
    xmin = max(0.0, float(np.nanmin(all_scores)) - 0.015)
    if xmin > 0.94:
        xmin = 0.94

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    ax.set_axisbelow(True)
    for yi in y:
        ax.axhline(yi, color="#E1E5EA", lw=0.8, zorder=0)

    for i, mname in enumerate(methods):
        color = METHOD_COLORS.get(mname, "gray")
        ax.scatter(
            values[mname],
            y + offsets[i],
            s=78,
            marker=METHOD_MARKERS.get(mname, "o"),
            color=color,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.96,
            label=METHOD_LABELS.get(mname, mname),
            zorder=3,
        )

    ax.axhline(5.5, color="#7A7A7A", lw=0.9, alpha=0.55)
    ax.set_yticks(y)
    ax.set_yticklabels(metric_names)
    ax.invert_yaxis()
    ax.set_xlim(xmin, 1.005)
    ax.set_xlabel("Score")
    ax.grid(axis="x", alpha=0.24)
    ax.legend(framealpha=0.92, loc="lower left")
    ax.set_title("Closed-world classification metrics")
    fig.tight_layout()
    _savefig(fig, out_dir, "closed_precision_recall")


# ---------------------------------------------------------------------------
# 2. Per-class F1 summary  (PRIMARY closed-world figure)
# ---------------------------------------------------------------------------

def plot_per_class_f1_summary(preds_by_method: dict, out_dir: Path) -> None:
    """
    Per-class F1 summary for comparing methods in closed-world.

    The left panel shows only the top-10 and bottom-10 keywords ranked by the
    proposed model, avoiding the unreadable 50 x 3 bar layout. The right panel
    is a violin plot summarizing the F1 distribution over all 50 keywords.

    Why: When all methods cluster above 97% accuracy the aggregate numbers look
    nearly identical.  Per-class F1 reveals WHICH keywords each model struggles
    with and where the thesis model specifically wins.
    """
    methods = list(preds_by_method.keys())
    ref_method = "resnet_bigru" if "resnet_bigru" in methods else methods[0]
    classes = preds_by_method[ref_method][4]
    n_cls   = len(classes)

    f1_matrix = {}
    for mname, (y_true, y_pred, *_) in preds_by_method.items():
        f1s = []
        for ci in range(n_cls):
            _, _, f1, _ = precision_recall_fscore_support(
                (y_true == ci).astype(int),
                (y_pred == ci).astype(int),
                average="binary", zero_division=0,
            )
            f1s.append(f1)
        f1_matrix[mname] = np.array(f1s, dtype=float)

    ref_scores = f1_matrix[ref_method]
    order_best = np.argsort(ref_scores)[::-1]
    top_n = min(10, n_cls)
    bot_n = min(10, max(n_cls - top_n, 0))
    focus_order = list(order_best[:top_n])
    if bot_n:
        focus_order += list(order_best[-bot_n:])

    cls_names = [classes[i] for i in order_best]
    focus_names = [classes[i] for i in focus_order]
    focus_y = np.arange(len(focus_order))

    fig, (ax_focus, ax_dist) = plt.subplots(
        1, 2, figsize=(13.6, max(7.3, 0.44 * len(focus_order) + 2.4)),
        gridspec_kw={"width_ratios": [3.5, 1.15]},
    )

    ax_focus.set_axisbelow(True)
    for yi in focus_y:
        ax_focus.axhline(yi, color="#D8D8D8", lw=0.7, alpha=0.55, zorder=0)

    for mname in methods:
        vals = f1_matrix[mname][focus_order]
        ax_focus.scatter(
            vals, focus_y,
            s=52,
            marker=METHOD_MARKERS.get(mname, "o"),
            color=METHOD_COLORS.get(mname, "gray"),
            edgecolor="white",
            linewidth=0.6,
            alpha=0.96,
            label=METHOD_LABELS.get(mname, mname),
            zorder=3,
        )

    ax_focus.set_yticks(focus_y)
    ax_focus.set_yticklabels(focus_names, fontsize=11)
    ax_focus.invert_yaxis()
    ax_focus.set_xlim(max(0.0, float(min(ref_scores.min(), 0.85)) - 0.03), 1.01)
    ax_focus.set_xlabel("F1-score")
    ax_focus.set_title(f"Top-{top_n} and bottom-{bot_n} keyword F1-scores")
    ax_focus.grid(axis="x", alpha=0.22)
    ax_focus.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), framealpha=0.92, ncol=len(methods))

    if bot_n:
        ax_focus.axhline(top_n - 0.5, color="#444444", lw=1.2, alpha=0.75)
        x0, x1 = ax_focus.get_xlim()
        ax_focus.text(
            x0 + 0.01 * (x1 - x0), top_n - 0.75,
            "Lowest-ranked keywords",
            fontsize=10.5,
            fontweight="bold",
            color="#444444",
            va="bottom",
        )

    dist_data = [f1_matrix[m] for m in methods]
    vp = ax_dist.violinplot(
        dist_data,
        positions=np.arange(1, len(methods) + 1),
        widths=0.85,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    for body, mname in zip(vp["bodies"], methods):
        body.set_facecolor(METHOD_COLORS.get(mname, "gray"))
        body.set_edgecolor(METHOD_COLORS.get(mname, "gray"))
        body.set_alpha(0.28)
    vp["cmedians"].set_color("#303030")
    vp["cmedians"].set_linewidth(1.4)

    for xpos, mname in enumerate(methods, start=1):
        vals = f1_matrix[mname]
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax_dist.scatter([xpos], [med], color=METHOD_COLORS.get(mname, "gray"), s=35, zorder=3)
        ax_dist.vlines(xpos, q1, q3, color=METHOD_COLORS.get(mname, "gray"), lw=3, alpha=0.75)
    ax_dist.set_xticks(np.arange(1, len(methods) + 1))
    ax_dist.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], rotation=0, ha="center")
    ax_dist.set_ylim(ax_focus.get_xlim())
    ax_dist.set_xlabel("Model")
    ax_dist.set_ylabel("F1-score")
    ax_dist.set_title("F1 distribution\n(50 keywords)")
    ax_dist.grid(axis="y", alpha=0.22)

    fig.subplots_adjust(left=0.18, right=0.985, top=0.88, bottom=0.10, wspace=0.18)
    _savefig(fig, out_dir, "per_class_f1_summary")

    # Also write CSV for LaTeX
    csv_path = out_dir / "per_class_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        cols = ["Keyword"] + [f"{m}_P" for m in methods] + \
               [f"{m}_R" for m in methods] + [f"{m}_F1" for m in methods]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for ci_ord, cls_name in zip(order_best, cls_names):
            row = {"Keyword": cls_name}
            for mname, (y_true, y_pred, *_) in preds_by_method.items():
                p, r, f1, _ = precision_recall_fscore_support(
                    (y_true == ci_ord).astype(int),
                    (y_pred == ci_ord).astype(int),
                    average="binary", zero_division=0,
                )
                row[f"{mname}_P"]  = f"{p:.4f}"
                row[f"{mname}_R"]  = f"{r:.4f}"
                row[f"{mname}_F1"] = f"{f1:.4f}"
            w.writerow(row)
        # Summary rows
        for label, avg in [("Macro-avg", "macro"), ("Weighted-avg", "weighted")]:
            row = {"Keyword": label}
            for mname, (y_true, y_pred, *_) in preds_by_method.items():
                p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred,
                                                               average=avg, zero_division=0)
                row[f"{mname}_P"]  = f"{p:.4f}"
                row[f"{mname}_R"]  = f"{r:.4f}"
                row[f"{mname}_F1"] = f"{f1:.4f}"
            w.writerow(row)
    print(f"  Saved: {csv_path}")


# ---------------------------------------------------------------------------
# 3. Per-class results table  (appendix / supplementary)
# ---------------------------------------------------------------------------

def plot_results_table(preds_by_method: dict, out_dir: Path) -> None:
    """
    Full per-class Precision / Recall / F1 table (all methods x N keywords).
    Method names are embedded in column headers to avoid ax.text/title overlap.
    Better suited to the appendix; use the per-class F1 summary for the main body.
    """
    # Short labels for column headers (long labels caused overlap)
    SHORT = {"resnet_bigru": "ReBiAt", "varcnn": "VarCNN", "netclr": "NetCLR"}

    methods = list(preds_by_method.keys())
    classes = preds_by_method[methods[0]][4]
    n_cls   = len(classes)

    rows_data = []
    for ci, cls_name in enumerate(classes):
        entry = {"name": cls_name}
        for mname, (y_true, y_pred, *_) in preds_by_method.items():
            p, r, f1, _ = precision_recall_fscore_support(
                (y_true == ci).astype(int),
                (y_pred == ci).astype(int),
                average="binary", zero_division=0,
            )
            entry[mname] = (p, r, f1)
        rows_data.append(entry)

    for label, avg in [("Macro-avg", "macro"), ("Weighted-avg", "weighted")]:
        entry = {"name": label}
        for mname, (y_true, y_pred, *_) in preds_by_method.items():
            p, r, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average=avg, zero_division=0)
            entry[mname] = (p, r, f1)
        rows_data.append(entry)

    acc_entry = {"name": "Accuracy"}
    for mname, (y_true, y_pred, *_) in preds_by_method.items():
        acc = accuracy_score(y_true, y_pred)
        acc_entry[mname] = (acc, float("nan"), float("nan"))
    rows_data.append(acc_entry)

    n_rows    = len(rows_data)
    n_methods = len(methods)
    FONT      = 6.5
    ROW_H     = 0.19    # inches per data row; 1.4x font line-height at 6.5pt
    KW_W      = 1.9     # keyword column width (inches)
    VAL_W     = 0.60    # each P / R / F1 column width (inches)
    fig_w     = KW_W + n_methods * 3 * VAL_W + 0.3
    fig_h     = ROW_H * (n_rows + 1) + 0.5   # +1 header row, 0.5 title margin

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    # Embed short method name in every column header -- avoids ax.text overlay
    header = ["Keyword"]
    for m in methods:
        s = SHORT.get(m, m[:7])
        header += [f"{s} P", f"{s} R", f"{s} F1"]

    cell_text  = []
    cell_color = []
    for rd in rows_data:
        is_summary = rd["name"] in ("Macro-avg", "Weighted-avg", "Accuracy")
        f1s    = [rd[m][2] if m in rd else -1 for m in methods]
        best   = int(np.argmax(f1s)) if not is_summary and max(f1s) >= 0 else -1
        vals   = [rd["name"]]
        colors = ["#E8EAF6" if is_summary else "#FAFAFA"]
        for mi, m in enumerate(methods):
            p, r, f1 = rd.get(m, (float("nan"),) * 3)
            vals += [
                f"{p:.3f}" if not np.isnan(p)  else "-",
                f"{r:.3f}" if not np.isnan(r)  else "-",
                f"{f1:.3f}" if not np.isnan(f1) else "-",
            ]
            base = METHOD_COLORS.get(m, "#888888")
            if mi == best:
                # Highlight best-F1 method for this keyword (F1 column stronger)
                colors += [f"{base}22", f"{base}22", f"{base}44"]
            elif is_summary:
                colors += ["#E8EAF6"] * 3
            else:
                colors += ["#FFFFFF", "#FFFFFF", "#F7F7F7"]
        cell_text.append(vals)
        cell_color.append(colors)

    tot_w  = KW_W + n_methods * 3 * VAL_W
    cw     = [KW_W / tot_w] + [VAL_W / tot_w] * (n_methods * 3)

    tbl = ax.table(
        cellText=cell_text, cellColours=cell_color,
        colLabels=header, colWidths=cw,
        loc="upper center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(FONT)
    # Each row should be ROW_H inches; matplotlib default distributes
    # rows to fill the axes, so scale proportionally.
    tbl.scale(1, ROW_H * (n_rows + 1) / max(fig_h - 0.5, 0.01))

    # Dark header row; colour-code by method group
    for j, label in enumerate(header):
        cell = tbl[0, j]
        cell.set_text_props(color="white", fontweight="bold", fontsize=FONT)
        if j == 0:
            cell.set_facecolor("#263238")
        else:
            mi = (j - 1) // 3
            cell.set_facecolor(METHOD_COLORS.get(methods[mi], "#263238"))

    # Bold + tinted summary rows at the bottom
    n_sep = n_cls
    for sep in range(n_sep, n_rows):
        for j in range(len(header)):
            tbl[sep + 1, j].set_facecolor("#E8EAF6")
            tbl[sep + 1, j].set_text_props(fontweight="bold", fontsize=FONT)

    ax.set_title(
        f"Per-class precision, recall, and F1-score for {n_cls} keywords",
        fontsize=10, pad=6, loc="left",
    )
    fig.tight_layout(pad=0.3)
    _savefig(fig, out_dir, "per_class_table")


# ---------------------------------------------------------------------------
# 4. Open-world rejection scores  (softmax / energy / mahalanobis)
# ---------------------------------------------------------------------------

OW_SCORE_METHODS = ("softmax", "energy", "mahalanobis")
OW_SCORE_LABELS  = {
    "softmax":     "Max-softmax probability",
    "energy":      "Energy score",
    "mahalanobis": "Mahalanobis score",
}
OW_SCORE_COLORS  = {
    "softmax":     "#1565C0",   # blue
    "energy":      "#C62828",   # red
    "mahalanobis": "#2E7D32",   # green
}


def plot_openworld_scores(
    results_dir: str,
    out_dir: Path,
    methods: tuple[str, ...] = OW_SCORE_METHODS,
    bins: int = 50,
) -> None:
    """Score-distribution histograms for OOD rejection scores.

    For continuous rejection scores a histogram is the most informative view:
    it shows directly how far apart the monitored (in-distribution) and
    unmonitored (out-of-distribution) score distributions are, where the tuned
    threshold sits, and therefore how cleanly the open-world decision separates
    "which keyword" from "outside the list".  A Venn diagram cannot express this
    because the scores are continuous, not set memberships.

    Reads, from the open_world.py output directory:
      scores_{method}.npz       te_s_k, te_s_u, threshold     (saved with --save_logits)

    Layout: one vertical histogram panel per score. Numeric ROC-AUC and
    thresholded open-world metrics should be reported in the result tables, not
    embedded in this figure.
    """
    rdir = Path(results_dir)

    # Load per-method test scores (in-distribution = known, OOD = unknown)
    loaded = {}
    for m in methods:
        npz_path = rdir / f"scores_{m}.npz"
        if not npz_path.exists():
            print(f"  [warn] OW scores: {npz_path.name} not found - skipping {m}")
            continue
        d = np.load(npz_path)
        loaded[m] = {
            "te_s_k": d["te_s_k"].astype(float),
            "te_s_u": d["te_s_u"].astype(float),
            "threshold": float(d["threshold"][0]) if "threshold" in d else None,
        }

    if not loaded:
        print("  [skip] open-world scores - no scores_*.npz found "
              f"in {results_dir} (run open_world.py with --save_logits)")
        return

    methods_present = [m for m in methods if m in loaded]
    n_hist = len(methods_present)
    fig, axes = plt.subplots(n_hist, 1, figsize=(7.4, 3.25 * n_hist))
    if n_hist == 1:
        axes = [axes]

    for ax, m in zip(axes, methods_present):
        sk = loaded[m]["te_s_k"]
        su = loaded[m]["te_s_u"]
        thr = loaded[m]["threshold"]
        col = OW_SCORE_COLORS.get(m, "gray")

        lo = float(min(sk.min(), su.min()))
        hi = float(max(sk.max(), su.max()))
        edges = np.linspace(lo, hi, bins + 1)

        ax.hist(sk, bins=edges, density=True, color=col, alpha=0.55,
                label="Monitored traffic")
        ax.hist(su, bins=edges, density=True, color="#9E9E9E", alpha=0.55,
                label="Unmonitored traffic")

        if thr is not None:
            ax.axvline(thr, color="black", ls="--", lw=1.3,
                       label="Decision threshold")
            ax.annotate("Reject as\nunmonitored", xy=(0.02, 0.96), xycoords="axes fraction",
                        ha="left", va="top", fontsize=9, color="#555")
            ax.annotate("Accept as\nmonitored", xy=(0.98, 0.96), xycoords="axes fraction",
                        ha="right", va="top", fontsize=9, color=col)

        ax.set_title(OW_SCORE_LABELS.get(m, m))
        ax.set_xlabel("Rejection score")
        ax.set_ylabel("Density")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(framealpha=0.9, loc="upper center", ncol=3)

    fig.suptitle("Open-world rejection score distributions", y=0.995)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.93, bottom=0.07, hspace=0.55)
    _savefig(fig, out_dir, "openworld_scores")


# ---------------------------------------------------------------------------
# 5. t-SNE - Concept drift  (S1 vs S2 session shift)
# ---------------------------------------------------------------------------

def plot_tsne_drift(
    drift_tsne_specs: dict,   # {name: (s1_ckpt_path, s1_npz_path, s2_npz_path)}
    out_dir: Path,
    n_per_class: int = 80,
    perplexity: float = 30.0,
    seed: int = 42,
) -> None:
    """
    t-SNE showing distribution shift from S1 to S2.

    The S1-trained checkpoint's encoder is used to project BOTH sessions into
    the same embedding space.  Same class = same colour; S1 = filled circles,
    S2 = hollow triangles.  Visual drift = cluster displacement between sessions.

    One subplot per method.
    """
    names = list(drift_tsne_specs.keys())
    if not names:
        print("  [skip] drift t-SNE - no --drift_tsne_ckpt / --drift_tsne_s1 / --drift_tsne_s2 specs")
        return

    _sys_path()
    import baselines as _B
    from train_resnet_bigru import make_loader

    n     = len(names)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    rng    = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for ax, name in zip(axes, names):
        s1_ckpt_path, s1_npz_path, s2_npz_path = drift_tsne_specs[name]
        s1_ckpt_path = _resolve_drift_ckpt_path(name, s1_ckpt_path)
        s1_npz_path = _resolve_existing_path(s1_npz_path)
        s2_npz_path = _resolve_existing_path(s2_npz_path)
        missing = [p for p in (s1_ckpt_path, s1_npz_path, s2_npz_path) if not Path(p).exists()]
        if missing:
            ax.text(
                0.5, 0.5,
                "Missing drift t-SNE input\n" + "\n".join(str(Path(p).name) for p in missing),
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color="gray",
            )
            ax.set_title(METHOD_LABELS.get(name, name), fontsize=10)
            ax.axis("off")
            continue

        ckpt    = torch.load(s1_ckpt_path, map_location="cpu", weights_only=False)
        model   = _B.rebuild_from_ckpt(ckpt, device, eval_mode=True)
        sel_idx = np.array(ckpt["selected_idx"], dtype=np.int64)
        s1_classes = list(ckpt["classes"])

        def _embs_from_npz(npz_path):
            data     = np.load(npz_path, allow_pickle=True)
            X_seq    = data["X_seq"].astype(np.float32)
            X_gl     = ckpt["gl_scaler"].transform(
                data["X_global"].astype(np.float32)[:, sel_idx]).astype(np.float32)
            y_raw    = data["y"].astype(np.int64)
            raw_cls  = [str(c).replace("_", " ").strip() for c in data["classes"].tolist()]
            return _get_embeddings_from_npz(model, X_seq, X_gl, device), y_raw, raw_cls

        s1_embs, s1_y, s1_cls_list = _embs_from_npz(s1_npz_path)
        s2_embs, s2_y, s2_cls_list = _embs_from_npz(s2_npz_path)

        # Map class names to S1 indices
        s1_name2idx = {c: i for i, c in enumerate(s1_cls_list)}
        s2_name2idx = {c: i for i, c in enumerate(s2_cls_list)}
        overlap     = [c for c in s1_cls_list if c in s2_name2idx]

        all_E, all_Y, all_sess = [], [], []
        cmap = plt.colormaps.get_cmap("tab10")

        for ci, c in enumerate(overlap):
            s1_idx_c = s1_name2idx[c]
            s2_idx_c = s2_name2idx[c]

            s1_mask = np.where(s1_y == s1_idx_c)[0]
            s2_mask = np.where(s2_y == s2_idx_c)[0]

            n1 = min(len(s1_mask), n_per_class)
            n2 = min(len(s2_mask), n_per_class)
            if n1 == 0 or n2 == 0:
                continue

            s1_chosen = rng.choice(s1_mask, n1, replace=False)
            s2_chosen = rng.choice(s2_mask, n2, replace=False)

            all_E.append(s1_embs[s1_chosen]); all_Y.extend([ci] * n1); all_sess.extend([0] * n1)
            all_E.append(s2_embs[s2_chosen]); all_Y.extend([ci] * n2); all_sess.extend([1] * n2)

        if not all_E:
            ax.text(0.5, 0.5, "No overlapping classes", transform=ax.transAxes,
                    ha="center", fontsize=11, color="gray")
            ax.set_title(METHOD_LABELS.get(name, name), fontsize=10)
            continue

        all_E    = np.concatenate(all_E, axis=0)
        all_Y    = np.array(all_Y)
        all_sess = np.array(all_sess)

        if all_E.shape[1] > 50:
            all_E = PCA(n_components=50, random_state=seed).fit_transform(all_E)
        Z = _fit_tsne(all_E, perplexity, seed)

        n_ov = len(overlap)
        for ci in range(n_ov):
            col = cmap(ci / max(n_ov, 1))
            # S1: filled circles
            mask1 = (all_Y == ci) & (all_sess == 0)
            if mask1.any():
                ax.scatter(Z[mask1, 0], Z[mask1, 1], c=[col], s=12,
                           marker="o", alpha=0.70, linewidths=0)
            # S2: hollow triangles
            mask2 = (all_Y == ci) & (all_sess == 1)
            if mask2.any():
                ax.scatter(Z[mask2, 0], Z[mask2, 1], facecolors="none",
                           edgecolors=[col], s=18, marker="^",
                           alpha=0.70, linewidths=0.8)

        # Legend: session markers only (not 10 class colours - too many)
        s1_patch = plt.Line2D([0], [0], marker="o", color="gray", linestyle="none",
                              markersize=6, label="Session 1", markerfacecolor="gray")
        s2_patch = plt.Line2D([0], [0], marker="^", color="gray", linestyle="none",
                              markersize=6, label="Session 2", markerfacecolor="none",
                              markeredgecolor="gray")
        ax.legend(handles=[s1_patch, s2_patch], fontsize=8, loc="best", framealpha=0.85)
        ax.set_title(METHOD_LABELS.get(name, name), fontsize=10, pad=6)
        ax.set_xlabel("t-SNE 1", fontsize=9)
        ax.set_ylabel("t-SNE 2", fontsize=9)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    fig.suptitle("t-SNE projection of embedding drift between Session 1 and Session 2",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    _savefig(fig, out_dir, "tsne_drift")


# ---------------------------------------------------------------------------
# 6. Concept drift comparison  (before vs after adaptation)
# ---------------------------------------------------------------------------

def plot_drift_comparison(drift_reports: dict, out_dir: Path) -> None:
    """
    Trajectory plot for adaptation strategies under temporal drift.

    Macro-F1 is used as the single summary metric because it best reflects
    balanced per-keyword recovery under the ten-keyword drift setting.
    """
    # The thesis compares the common baselines (F0/F1/F3/F3_AUG) and the
    # proposed-model-specific temporal adaptations when they are present.
    STRATS = ["F0", "F1", "F3", "F3_AUG", "F_TEMP", "F_TPROTO"]
    STRAT_LABELS = {
        "F0":      "No adapt\n(F0)",
        "F1":      "Clf only\n(F1)",
        "F3":      "Full FT\n(F3)",
        "F3_AUG":  "Full FT + aug\n(F3_AUG)",
        "F_TEMP":  "Temporal only\n(F_TEMP)",
        "F_TPROTO":"Temp+Proto\n(F_TPROTO, ours)",
    }

    methods  = list(drift_reports.keys())
    n_s      = len(STRATS)
    x        = np.arange(n_s)

    fig, ax = plt.subplots(figsize=(8.4, 5.4))

    for mname, report in drift_reports.items():
        results = report.get("results", {})
        f1s = np.array([
            results.get(s, {}).get("metrics", {}).get("f1_macro", float("nan"))
            for s in STRATS
        ], dtype=float)
        col  = METHOD_COLORS.get(mname, "gray")
        lbl  = METHOD_LABELS.get(mname, mname)

        ax.plot(
            x, f1s,
            marker=METHOD_MARKERS.get(mname, "o"),
            ms=7,
            lw=2.2,
            color=col,
            label=lbl,
        )

        finite = np.where(np.isfinite(f1s))[0]
        if len(finite):
            last = finite[-1]
            ax.text(
                x[last] + 0.04,
                f1s[last],
                f"{f1s[last]:.2f}",
                fontsize=9,
                color=col,
                va="center",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([STRAT_LABELS.get(s, s) for s in STRATS], fontsize=9)
    ax.set_ylabel("Macro F1-score")
    ax.set_ylim(0, 1.10)
    ax.grid(axis="y", alpha=0.28)
    ax.legend(loc="upper left", framealpha=0.92)
    ax.set_title("Macro-F1 under temporal drift")
    fig.tight_layout()
    _savefig(fig, out_dir, "drift_comparison")


# ---------------------------------------------------------------------------
# 7. Feature importance  (ANOVA F-scores)
# ---------------------------------------------------------------------------

def plot_feature_importance(npz_path: str, ckpt_path: str, out_dir: Path) -> None:
    """
    Bar chart of ANOVA F-scores for the 15 global features.
    Selected features (from checkpoint's selected_idx) highlighted in blue.
    """
    from sklearn.feature_selection import f_classif
    _sys_path()
    from train_resnet_bigru import load_npz, split_stratified

    X_seq, X_global, y, _ = load_npz(npz_path)
    tr_idx, _, _ = split_stratified(y)
    X_tr, y_tr   = X_global[tr_idx], y[tr_idx]

    ckpt        = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sel_idx_set = set(ckpt["selected_idx"])

    f_scores, _ = f_classif(X_tr, y_tr)
    order       = np.argsort(f_scores)[::-1]
    names_ord   = [GLOBAL_FEATURE_NAMES[i] for i in order]
    scores_ord  = f_scores[order]
    colors      = [METHOD_COLORS["resnet_bigru"] if i in sel_idx_set else "#BDBDBD"
                   for i in order]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(names_ord)), scores_ord, color=colors,
                  edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(names_ord)))
    ax.set_xticklabels(names_ord, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("ANOVA F-score", fontsize=11)
    ax.set_title("ANOVA F-statistics of the global feature set", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    sel_patch  = mpatches.Patch(color=METHOD_COLORS["resnet_bigru"],
                                label=f"Selected features (k = {len(sel_idx_set)})")
    drop_patch = mpatches.Patch(color="#BDBDBD", label="Excluded features")
    ax.legend(handles=[sel_patch, drop_patch], fontsize=9)

    for rank, (bar, sc) in enumerate(zip(bars, scores_ord), 1):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + scores_ord.max() * 0.01,
                str(rank), ha="center", va="bottom", fontsize=7, color="#555")

    fig.tight_layout()
    _savefig(fig, out_dir, "feature_importance")


# ---------------------------------------------------------------------------
# 8. Defense comparison  (adaptive attacker only)
# ---------------------------------------------------------------------------

def plot_defense_comparison(defense_summaries: dict, out_dir: Path) -> None:
    """
    Grouped bars for adaptive-attacker defense evaluation.

    Style and semantics follow generate_defense_bars.py so the thesis figure is
    consistent whether it is rendered from comparison CSVs or defense summary
    JSONs.
    """
    defense_labels = {
        "undefended": "Undefended",
        "burstguard": "BurstGuard",
        "wtf_pad": "WTF-PAD",
        "front": "FRONT",
    }
    defense_order = ["undefended", "burstguard", "wtf_pad", "front"]

    def _rows(summary):
        rows = {}
        for row in summary.get("rows", []):
            defense = row.get("defense")
            if defense:
                rows[defense] = row
        return rows

    def _pick_row(summary_rows, defense_key):
        if defense_key == "undefended":
            return summary_rows.get("UNDEFENDED (ceiling)")
        return summary_rows.get(defense_key)

    def _adaptive_values(summary):
        summary_rows = _rows(summary)
        vals = []
        for defense in defense_order:
            row = _pick_row(summary_rows, defense)
            if defense == "undefended":
                val = summary.get("ceiling_accuracy")
                if val is None and row is not None:
                    val = row.get("ceiling_accuracy")
            else:
                val = None if row is None else row.get("adap_accuracy")
            vals.append(float(val) if val is not None else float("nan"))
        return np.array(vals, dtype=float)

    def _xtick_labels(first_summary):
        summary_rows = _rows(first_summary)
        labels = []
        for defense in defense_order:
            row = _pick_row(summary_rows, defense)
            if defense == "undefended":
                labels.append("Undefended\n0.00x")
                continue
            overhead = float(row["O_bw_mean"]) if row and row.get("O_bw_mean") is not None else float("nan")
            labels.append(f"{defense_labels[defense]}\n{overhead:.2f}x")
        return labels

    methods = [m for m in ("resnet_bigru", "varcnn", "netclr") if m in defense_summaries]
    if not methods:
        methods = list(defense_summaries.keys())
    x = np.arange(len(defense_order))
    bar_w = min(0.22, 0.72 / max(len(methods), 1))
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2) * bar_w

    fig, ax = plt.subplots(figsize=(8.6, 5.1))

    for mi, method in enumerate(methods):
        vals = _adaptive_values(defense_summaries[method])
        bars = ax.bar(
            x + offsets[mi],
            vals,
            bar_w * 0.92,
            color=METHOD_COLORS.get(method, "gray"),
            alpha=0.94,
            edgecolor="white",
            linewidth=0.8,
            label=METHOD_LABELS.get(method, method),
        )
        for bar, val in zip(bars, vals):
            if np.isfinite(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    min(val + 0.016, 1.055),
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="#2F2F2F",
                )

    ax.set_title("Adaptive attacker", loc="left", fontweight="bold", pad=10)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Defense and mean bandwidth overhead")
    ax.set_xticks(x)
    ax.set_xticklabels(_xtick_labels(defense_summaries[methods[0]]))
    ax.set_ylim(0, 1.10)
    ax.grid(axis="y", alpha=0.28)
    ax.set_axisbelow(True)

    handles = [
        mpatches.Patch(color=METHOD_COLORS.get(m, "gray"), label=METHOD_LABELS.get(m, m))
        for m in methods
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.18, top=0.82)
    _savefig(fig, out_dir, "defense_bars")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

# Saves a matplotlib figure as both PDF and SVG in the output directory.
def _savefig(fig: plt.Figure, out_dir: Path, stem: str, aliases: tuple[str, ...] = ()) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stems = (stem,) + tuple(a for a in aliases if a != stem)
    for current_stem in stems:
        for ext in ("pdf", "svg"):
            fig.savefig(out_dir / f"{current_stem}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    alias_note = "" if not aliases else f"  (aliases: {', '.join(aliases)})"
    print(f"  Saved: {out_dir}/{stem}.pdf / .svg{alias_note}")


# Parses a list of 'name:path' strings into a dict.
def _parse_specs(specs):
    """Parse 'name:path' list -> {name: path}."""
    result = {}
    for s in specs:
        if ":" not in s:
            print(f"  [warn] bad spec (no ':'): {s!r} - skipped"); continue
        name, path = s.split(":", 1)
        result[name.strip()] = path.strip()
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Generate thesis visualizations")
    p.add_argument("--npz",   required=True, help="Known (50-kw) dataset NPZ")
    p.add_argument("--ckpt",  action="append", default=[],
                   metavar="name:path", help="Closed-world checkpoint (repeat per method)")
    p.add_argument("--drift", action="append", default=[],
                   metavar="name:drift_report.json", help="Drift report (repeat per method)")
    # Deprecated: F_CON was removed; --drift_fcon is accepted and ignored so that
    # older notebook cells keep running instead of failing at argument parsing.
    p.add_argument("--drift_fcon", action="append", default=[],
                   help=argparse.SUPPRESS)
    # Open-world rejection scores (softmax / energy / mahalanobis)
    p.add_argument("--ow_scores_dir", default="",
                   metavar="results_open_world_dir",
                   help="Directory of open_world.py output (scores_*.npz + "
                        "open_world_results.json, written with --save_logits). "
                        "Enables the score-histogram figure.")
    # Drift t-SNE - three parallel flags, matched by name
    p.add_argument("--drift_tsne_ckpt", action="append", default=[],
                   metavar="name:s1_ckpt.pt",
                   help="S1-trained checkpoint for drift t-SNE (repeat per method)")
    p.add_argument("--drift_tsne_s1",   action="append", default=[],
                   metavar="name:session1.npz",
                   help="Session-1 NPZ for drift t-SNE (repeat per method)")
    p.add_argument("--drift_tsne_s2",   action="append", default=[],
                   metavar="name:session2.npz",
                   help="Session-2 NPZ for drift t-SNE (repeat per method)")
    p.add_argument("--defense", action="append", default=[],
                   metavar="name:defense_summary.json",
                   help="Defense summary JSON produced by collect_defense_results.py "
                        "(repeat per method)")
    # Example Kaggle path: /kaggle/working/figures
    p.add_argument("--out_dir", default="./figures")
    p.add_argument("--tsne_samples",    type=int,   default=1500)
    p.add_argument("--tsne_perplexity", type=float, default=40.0)
    p.add_argument("--device", default="cuda")
    # parse_known_args (not parse_args) so a stale/unknown flag from an older
    # notebook cell prints a warning instead of aborting the whole script.
    args, _unknown = p.parse_known_args()
    if _unknown:
        print(f"[warn] ignoring unrecognized arguments: {_unknown}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_map      = _parse_specs(args.ckpt)
    drift_map     = _parse_specs(args.drift)
    dt_ckpt_map   = _parse_specs(args.drift_tsne_ckpt)
    dt_s1_map     = _parse_specs(args.drift_tsne_s1)
    dt_s2_map     = _parse_specs(args.drift_tsne_s2)

    # Load predictions for all available checkpoints
    preds_by_method = {}
    for name, ckpt_path in ckpt_map.items():
        if not Path(ckpt_path).exists():
            print(f"  [warn] checkpoint not found: {ckpt_path} - skipping {name}"); continue
        print(f"Loading {name} ...")
        try:
            preds_by_method[name] = load_model_and_preds(args.npz, ckpt_path, args.device)
            print(f"  {name}: {len(preds_by_method[name][0])} test samples, "
                  f"{len(preds_by_method[name][4])} classes")
        except Exception as e:
            print(f"  [warn] failed to load {name}: {e}")

    classes = preds_by_method[next(iter(preds_by_method))][4] if preds_by_method else []

    # Every figure stage is wrapped in _safe(): a failure in one (e.g. a t-SNE
    # plot on an older sklearn) prints a warning but never aborts the script, so
    # the remaining figures - including drift_comparison - are still produced.

    # 1. Closed-world precision/recall summary
    if preds_by_method:
        print("\nPlotting closed-world precision/recall ...")
        _safe("closed precision/recall", plot_closed_precision_recall, preds_by_method, out_dir)
    else:
        print("[skip] precision/recall - no checkpoints loaded")

    # 2. Per-class F1 summary  (PRIMARY closed-world figure)
    if preds_by_method:
        print("\nPlotting per-class F1 summary ...")
        _safe("per-class F1 summary", plot_per_class_f1_summary, preds_by_method, out_dir)
    else:
        print("[skip] per-class F1 summary - no checkpoints loaded")

    # 3. Full per-class table  (appendix)
    if preds_by_method:
        print("\nPlotting per-class table ...")
        _safe("per-class table", plot_results_table, preds_by_method, out_dir)
    else:
        print("[skip] per-class table - no checkpoints loaded")

    # 4. Open-world rejection scores (softmax / energy / mahalanobis)
    if args.ow_scores_dir:
        print("\nPlotting open-world rejection scores ...")
        _safe("open-world scores", plot_openworld_scores, args.ow_scores_dir, out_dir)
    else:
        print("[skip] open-world scores - pass --ow_scores_dir to enable")

    # 5. Drift t-SNE
    drift_tsne_specs = {}
    for name in dt_ckpt_map:
        if name in dt_s1_map and name in dt_s2_map:
            drift_tsne_specs[name] = (dt_ckpt_map[name], dt_s1_map[name], dt_s2_map[name])
        else:
            print(f"  [warn] drift t-SNE for {name!r}: need --drift_tsne_ckpt, "
                  f"--drift_tsne_s1 and --drift_tsne_s2 all set - skipping")
    if drift_tsne_specs:
        print("\nPlotting drift t-SNE ...")
        _safe("drift t-SNE", plot_tsne_drift, drift_tsne_specs, out_dir, perplexity=args.tsne_perplexity)
    else:
        print("[skip] drift t-SNE - pass --drift_tsne_ckpt / --drift_tsne_s1 / --drift_tsne_s2")

    # 6. Drift comparison chart
    drift_reports = {}
    for name, path in drift_map.items():
        if not Path(path).exists():
            print(f"  [warn] drift report not found: {path}"); continue
        try:
            with open(path, encoding="utf-8") as f:
                drift_reports[name] = json.load(f)
        except Exception as e:                                # noqa: BLE001
            print(f"  [warn] could not read drift report {path}: {e}")

    if drift_reports:
        print("\nPlotting drift comparison ...")
        _safe("drift comparison", plot_drift_comparison, drift_reports, out_dir)
    else:
        print("[skip] drift comparison - no drift reports provided")

    # 7. Feature importance
    if ckpt_map:
        first_name, first_ckpt = next(iter(ckpt_map.items()))
        if Path(first_ckpt).exists():
            print(f"\nPlotting feature importance (from {first_name}) ...")
            _safe("feature importance", plot_feature_importance, args.npz, first_ckpt, out_dir)

    # 8. Defense comparison
    defense_summaries = {}
    for name, path in _parse_specs(args.defense).items():
        p_def = Path(path)
        if not p_def.exists():
            print(f"  [warn] defense summary not found: {path} - skipping {name}")
            continue
        try:
            with p_def.open(encoding="utf-8") as f:
                defense_summaries[name] = json.load(f)
        except Exception as e:                                # noqa: BLE001
            print(f"  [warn] could not read defense summary {path}: {e}")
    if defense_summaries:
        print("\nPlotting defense comparison (adaptive only) ...")
        _safe("defense comparison", plot_defense_comparison, defense_summaries, out_dir)
    else:
        print("[skip] defense comparison - pass --defense name:defense_summary.json to enable")

    print(f"\nAll figures saved to: {out_dir}")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
