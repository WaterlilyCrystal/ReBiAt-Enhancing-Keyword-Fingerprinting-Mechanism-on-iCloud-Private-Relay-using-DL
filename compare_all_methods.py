"""
compare_all_methods.py
======================
Aggregate the results of the THREE methods
  * resnet_bigru  (thesis: ResNet-10 + BiGRU + Attention)
  * varcnn        (Var-CNN, PETS 2019)
  * netclr        (NetCLR, CCS 2023)
across the FOUR scenarios (closed-world, open-world, concept-drift, defense) into
tidy, committee-ready comparison tables (CSV + one master JSON).

Each scenario's result files already exist on disk (produced by the per-method
runs); this script only READS and TABULATES them - no models are run, so it is
fast and fully reproducible.  Every spec is optional and repeatable, so you can
tabulate whatever subset you have finished running.

Spec format (repeat the flag per method):
  --closed     name:/path/run_results.json          # train_*.py output
  --openworld  name:/path/open_world_results.json    # open_world.py output
  --drift      name:/path/drift_report.json          # drift_pipeline.py output
  --defense    name:/path/defense_summary.json       # collect_defense_results.py output

Example:
  python compare_all_methods.py \
    --closed    resnet_bigru:results_bigru/run_results.json \
    --closed    varcnn:results_varcnn/run_results.json \
    --closed    netclr:results_netclr/run_results.json \
    --openworld resnet_bigru:results_ow_bigru/open_world_results.json \
    --openworld varcnn:results_ow_varcnn/open_world_results.json \
    --drift     resnet_bigru:drift_bigru/drift_report.json \
    --drift     varcnn:drift_varcnn/drift_report.json \
    --defense   resnet_bigru:defense_bigru/defense_summary.json \
    --out_dir   results_comparison

Outputs (in --out_dir):
  comparison_closed_world.csv
  comparison_open_world.csv
  comparison_drift.csv
  comparison_defense.csv
  comparison_master.json     (everything, machine-readable)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


# Load a JSON file and return its contents, or None on error.
def _load(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:                      # noqa: BLE001
        print(f"  [warn] could not read {path}: {e}")
        return None


def _parse_specs(specs: list[str]) -> list[tuple[str, str]]:
    """Parse 'name:/path/with:colons.json' -> (name, path). rsplit handles Windows
    drive letters and any ':' inside the path."""
    out = []
    for s in specs or []:
        if ":" not in s:
            print(f"  [warn] bad spec (missing ':'): {s}")
            continue
        name, path = s.split(":", 1)
        out.append((name.strip(), path.strip()))
    return out


# Write a list of row dicts to a CSV file with the given field order.
def _write_csv(rows: list[dict], fields: list[str], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"  wrote {path}  ({len(rows)} rows)")


def _round(x, n=4):
    return round(float(x), n) if isinstance(x, (int, float)) else x


# ---------------------------------------------------------------------------
# Per-scenario extractors
# ---------------------------------------------------------------------------

# Collect closed-world metrics from each method's run_results.json spec.
def collect_closed(specs) -> list[dict]:
    rows = []
    for name, path in _parse_specs(specs):
        d = _load(path)
        if not d:
            continue
        m = d.get("test_metrics", {})
        rows.append({
            "method":        name,
            "test_acc":      _round(d.get("test_acc", m.get("acc"))),
            "f1_macro":      _round(m.get("f1_macro")),
            "precision_macro": _round(m.get("precision_macro")),
            "recall_macro":  _round(m.get("recall_macro")),
            "n_classes":     d.get("n_classes"),
            "source":        path,
        })
    return rows


# Collect open-world detection and classification metrics from each method's result spec.
def collect_openworld(specs) -> list[dict]:
    rows = []
    for name, path in _parse_specs(specs):
        d = _load(path)
        if not d:
            continue
        methods = d.get("methods", {})
        for ood_name, vals in methods.items():
            det = vals.get("detection", {})
            ow  = vals.get("open_world", {})
            rows.append({
                "method":          name,
                "ood_method":      ood_name,
                "roc_auc":         _round(det.get("roc_auc")),
                "accuracy":        _round(ow.get("accuracy")),
                "precision_macro": _round(ow.get("precision_macro")),
                "recall_macro":    _round(ow.get("recall_macro")),
                "f1_macro":        _round(ow.get("f1_macro")),
                "source":          path,
            })
    return rows


# Collect concept-drift metrics per strategy from each method's drift_report.json spec.
def collect_drift(specs) -> list[dict]:
    rows = []
    for name, path in _parse_specs(specs):
        d = _load(path)
        if not d:
            continue
        res = d.get("results", {})
        base = res.get("F0", {}).get("metrics", {})
        base_acc = base.get("acc")
        base_f1  = base.get("f1_macro")
        for strat, rec in res.items():
            m = rec.get("metrics", {})
            rows.append({
                "method":     name,
                "strategy":   strat,
                "label":      rec.get("label", ""),
                "acc":        _round(m.get("acc")),
                "precision_macro": _round(m.get("precision_macro")),
                "recall_macro": _round(m.get("recall_macro")),
                "f1_macro":   _round(m.get("f1_macro")),
                "delta_acc_vs_F0": _round((m.get("acc", 0) - base_acc)
                                          if (base_acc is not None and m.get("acc") is not None) else None),
                "delta_f1_vs_F0":  _round((m.get("f1_macro", 0) - base_f1)
                                          if (base_f1 is not None and m.get("f1_macro") is not None) else None),
                "source":     path,
            })
    return rows


# Collect defense scenario metrics (non-adaptive and adaptive) from each method's defense_summary.json spec.
def collect_defense(specs) -> list[dict]:
    rows = []
    for name, path in _parse_specs(specs):
        d = _load(path)
        if not d:
            continue
        ceiling = d.get("ceiling_accuracy")
        for r in d.get("rows", []):
            rows.append({
                "method":          name,
                "defense":         r.get("defense"),
                "ceiling_accuracy": _round(ceiling),
                "O_bw_mean":       _round(r.get("O_bw_mean")),
                "O_lat_mean":      _round(r.get("O_lat_mean")),
                "nonadap_accuracy": _round(r.get("nonadap_accuracy")),
                "nonadap_f1_macro": _round(r.get("nonadap_macro_f1", r.get("nonadap_f1_macro"))),
                "adap_accuracy":   _round(r.get("adap_accuracy")),
                "adap_f1_macro":   _round(r.get("adap_macro_f1", r.get("adap_f1_macro"))),
                "source":          path,
            })
    return rows


# ---------------------------------------------------------------------------
# Pretty console summary
# ---------------------------------------------------------------------------

# Print a formatted table of rows to stdout with aligned columns.
def _print_table(title: str, rows: list[dict], cols: list[str]) -> None:
    if not rows:
        return
    print(f"\n{'='*78}\n{title}\n{'='*78}")
    widths = {c: max(len(c), *(len(str(r.get(c, ''))) for r in rows)) for c in cols}
    print("  ".join(f"{c:<{widths[c]}}" for c in cols))
    print("-" * (sum(widths.values()) + 2 * (len(cols) - 1)))
    for r in rows:
        print("  ".join(f"{str(r.get(c, '')):<{widths[c]}}" for c in cols))


def main():
    p = argparse.ArgumentParser(description="Tabulate 3-method x 4-scenario comparison")
    p.add_argument("--closed",    action="append", default=[], metavar="name:run_results.json")
    p.add_argument("--openworld", action="append", default=[], metavar="name:open_world_results.json")
    p.add_argument("--drift",     action="append", default=[], metavar="name:drift_report.json")
    p.add_argument("--defense",   action="append", default=[], metavar="name:defense_summary.json")
    p.add_argument("--out_dir",   default="./results_comparison")  # On Kaggle: /kaggle/working/results_comparison
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Collecting closed-world ...")
    closed = collect_closed(args.closed)
    print("Collecting open-world ...")
    ow = collect_openworld(args.openworld)
    print("Collecting concept-drift ...")
    drift = collect_drift(args.drift)
    print("Collecting defense ...")
    defense = collect_defense(args.defense)

    if closed:
        _write_csv(closed, ["method", "test_acc", "precision_macro", "recall_macro",
                            "f1_macro", "n_classes", "source"],
                   out / "comparison_closed_world.csv")
    if ow:
        _write_csv(ow, ["method", "ood_method", "roc_auc", "accuracy",
                        "precision_macro", "recall_macro", "f1_macro",
                        "source"], out / "comparison_open_world.csv")
    if drift:
        _write_csv(drift, ["method", "strategy", "label", "acc",
                           "precision_macro", "recall_macro", "f1_macro",
                           "delta_acc_vs_F0",
                           "delta_f1_vs_F0", "source"],
                   out / "comparison_drift.csv")
    if defense:
        _write_csv(defense, ["method", "defense", "ceiling_accuracy", "O_bw_mean",
                             "O_lat_mean", "nonadap_accuracy", "nonadap_f1_macro",
                             "adap_accuracy", "adap_f1_macro", "source"],
                   out / "comparison_defense.csv")

    master = {"closed_world": closed, "open_world": ow,
              "concept_drift": drift, "defense": defense}
    with (out / "comparison_master.json").open("w", encoding="utf-8") as f:
        json.dump(master, f, indent=2)
    print(f"  wrote {out / 'comparison_master.json'}")

    # Console summaries
    _print_table("CLOSED-WORLD", closed,
                 ["method", "test_acc", "precision_macro", "recall_macro", "f1_macro"])
    _print_table("OPEN-WORLD (per OOD method)", ow,
                 ["method", "ood_method", "roc_auc", "accuracy", "precision_macro", "recall_macro", "f1_macro"])
    _print_table("CONCEPT DRIFT (per strategy)", drift,
                 ["method", "strategy", "acc", "f1_macro", "delta_acc_vs_F0"])
    _print_table("DEFENSE (per defense)", defense,
                 ["method", "defense", "O_bw_mean", "nonadap_accuracy", "adap_accuracy", "adap_f1_macro"])
    print(f"\nComparison tables written to: {out}")


if __name__ == "__main__":
    main()
