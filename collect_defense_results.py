"""
collect_defense_results.py  --  Aggregate every artifact of the defense chapter
(Methodology Sec. 8) into a SINGLE comparison table.

Each script in the defense pipeline already persists its own files:
  defenses.py           -> <stem>_overhead.json         (O_bw, O_lat, params)
  eval_defended.py      -> <stem>_nonadaptive.json      (non-adaptive accuracy/F1)
  train_resnet_bigru.py -> <results_dir>/run_results.json (test_acc / test_metrics = ADAPTIVE)

This script ONLY re-reads those files (it never re-trains or re-evaluates) and writes:
  defense_summary.csv  +  defense_summary.json
with one row per defense (O_bw, O_lat, non-adaptive accuracy/F1, adaptive accuracy/F1), plus the
undefended ceiling, ready for the defense-summary tables and figures (Sec. 8.5).

Usage (after the train / defense / eval cells have produced their files):
  python collect_defense_results.py \
      --clean_results results/run_results.json \
      --defense front:dataset_kfp_v2_mac_front.npz:results_front \
      --defense wtf_pad:dataset_kfp_v2_mac_wtf_pad.npz:results_wtfpad \
      --defense burstguard:dataset_kfp_v2_mac_burstguard.npz:results_bg

Each `--defense` is either:
  name:defended_npz:adaptive_results_dir
or
  name:defended_npz:adaptive_results_dir:nonadaptive_json

The 4-field form is useful when multiple methods evaluate the SAME defended npz:
each method can write its own non-adaptive JSON instead of overwriting the shared
`<defended_npz>_nonadaptive.json`.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


# Load a JSON file from the given path, returning None if missing or unreadable.
def _load(path: str | Path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        with p.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  ! Failed to read {p}: {e}")
        return None


def _metric(d: dict | None, *keys, default=None):
    """Return the first nested value found; each key may be a dotted path 'a.b'."""
    if not d:
        return default
    for key in keys:
        cur = d
        ok = True
        for part in key.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok:
            return cur
    return default


# Collect overhead, non-adaptive, and adaptive metrics for a single defense into one dict.
def collect_one(
    name: str,
    defended_npz: str,
    adaptive_dir: str,
    nonadaptive_json: str | None = None,
) -> dict:
    stem = str(Path(defended_npz).with_suffix(""))
    overhead = _load(f"{stem}_overhead.json")
    nonadap = _load(nonadaptive_json or f"{stem}_nonadaptive.json")
    adap = _load(f"{Path(adaptive_dir) / 'run_results.json'}")

    row = {
        "defense": name,
        "defended_npz": defended_npz,
        "nonadaptive_json": nonadaptive_json or f"{stem}_nonadaptive.json",
        # Overhead (Methodology 8.4) -- new multi-seed schema first, then legacy single-seed
        "O_bw_mean": _metric(overhead, "bandwidth_overhead_across_seeds.mean", "bandwidth_overhead.mean"),
        "O_bw_std": _metric(overhead, "bandwidth_overhead_across_seeds.std", default=0.0),
        "O_bw_median": _metric(overhead, "bandwidth_overhead.median"),
        "O_lat_mean": _metric(overhead, "latency_overhead_across_seeds.mean", "latency_overhead.mean"),
        "params": _metric(overhead, "params", default={}),
        # Non-adaptive attacker (Sec. 8.3)
        "nonadap_accuracy": _metric(nonadap, "accuracy"),
        "nonadap_macro_f1": _metric(nonadap, "macro_f1"),
        # Adaptive attacker = HEADLINE (Sec. 8.3)
        "adap_accuracy": _metric(adap, "test_acc"),
        "adap_macro_f1": _metric(adap, "test_metrics.f1_macro", "test_metrics.macro_f1"),
    }
    missing = [k for k, v in [("overhead", overhead), ("nonadaptive", nonadap),
                              ("adaptive run_results", adap)] if v is None]
    if missing:
        print(f"  [{name}] MISSING: {', '.join(missing)}")
    return row


# Parse CLI arguments, load all result files, and write the summary CSV and JSON.
def main():
    p = argparse.ArgumentParser(description="Aggregate defense-chapter results into one table")
    p.add_argument("--clean_results", default="results/run_results.json",
                   help="run_results.json of the undefended (ceiling) training run")
    p.add_argument("--defense", action="append", default=[],
                   metavar="name:defended_npz:adaptive_results_dir[:nonadaptive_json]",
                   help="Repeat once per defense")
    p.add_argument("--out", default="defense_summary")
    args = p.parse_args()

    clean = _load(args.clean_results)
    ceiling_acc = _metric(clean, "test_acc")
    ceiling_f1 = _metric(clean, "test_metrics.f1_macro", "test_metrics.macro_f1")
    if clean is None:
        print(f"! Ceiling not found at {args.clean_results} -- the ceiling column will be blank.")

    rows = []
    # Reference row: undefended ceiling
    rows.append({"defense": "UNDEFENDED (ceiling)", "defended_npz": args.clean_results,
                 "O_bw_mean": 0.0, "O_bw_median": 0.0, "O_lat_mean": 0.0, "params": {},
                 "nonadap_accuracy": None, "nonadap_macro_f1": None,
                 "adap_accuracy": ceiling_acc, "adap_macro_f1": ceiling_f1})

    for spec in args.defense:
        # Kaggle paths do not contain drive letters, so a 3-field or 4-field split is safe.
        try:
            parts = spec.split(":")
        except ValueError:
            parts = []
        if len(parts) == 3:
            name, defended_npz, adaptive_dir = parts
            nonadaptive_json = None
        elif len(parts) == 4:
            name, defended_npz, adaptive_dir, nonadaptive_json = parts
        else:
            print(
                f"! Skipping malformed spec: {spec!r} "
                "(expected name:npz:dir or name:npz:dir:nonadaptive_json)"
            )
            continue
        rows.append(collect_one(name, defended_npz, adaptive_dir, nonadaptive_json))

    # Compact table to stdout
    print("\n" + "=" * 92)
    hdr = f"{'Defense':<24} {'O_bw':>7} {'O_lat':>7} {'NonAd-Acc':>9} {'Adapt-Acc':>9} {'Adapt-F1':>9}"
    print(hdr); print("-" * 92)
    def fmt(v): return f"{v:.4f}" if isinstance(v, (int, float)) else "  -  "
    for r in rows:
        print(f"{r['defense']:<24} {fmt(r['O_bw_mean']):>7} {fmt(r['O_lat_mean']):>7} "
              f"{fmt(r['nonadap_accuracy']):>9} {fmt(r['adap_accuracy']):>9} {fmt(r['adap_macro_f1']):>9}")
    print("=" * 92)
    if ceiling_acc is not None:
        print(f"Ceiling accuracy = {ceiling_acc:.4f} | Headline metric = adaptive accuracy.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump({"ceiling_accuracy": ceiling_acc, "ceiling_macro_f1": ceiling_f1,
                   "rows": rows}, f, indent=2, default=float)
    fields = ["defense", "O_bw_mean", "O_bw_std", "O_bw_median", "O_lat_mean",
              "nonadap_accuracy", "nonadap_macro_f1", "adap_accuracy", "adap_macro_f1",
              "defended_npz", "nonadaptive_json", "params"]
    with out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            rr = dict(r); rr["params"] = json.dumps(rr.get("params", {}))
            w.writerow(rr)
    print(f"\nSaved summary table -> {out.with_suffix('.csv')}  &  {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
