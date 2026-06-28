"""
open_world_pipeline.py
Complete open-world evaluation pipeline for keyword fingerprinting.

Step 1  Optionally extract features from unknown-keyword PCAPs -> NPZ
        (cached; skipped if --unknown_npz already exists)
Step 2  Run OOD evaluation (softmax / energy / mahalanobis)
Step 3  Save results to JSON, CSV, and optionally raw score/logit caches

Typical usage:
  python open_world_pipeline.py \\
    --known_npz      ./features/known_50kw.npz \\
    --unknown_npz    ./features/openworld_unknown.npz \\
    --checkpoint     ./results_bigru/best_model.pt \\
    --results_dir    ./results_open_world \\
    --save_logits

If the unknown traffic is still raw PCAP rather than an extracted NPZ:
  python open_world_pipeline.py \\
    --known_npz        ./features/known_50kw.npz \\
    --unknown_pcap_dir ./data/openworld_unknown/ \\
    --unknown_npz      ./features/openworld_unknown.npz \\
    --checkpoint       ./results_bigru/best_model.pt \\
    --results_dir      ./results_open_world \\
    --save_logits

Outputs written to --results_dir:
  open_world_results.json    metrics summary
  open_world_results.csv     flat table, one row per OOD method
  logits_cache.npz           raw logits (with --save_logits)
  scores_{method}.npz        per-method known/unknown scores (with --save_logits)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Step 1: Feature extraction for unknown keywords
# ---------------------------------------------------------------------------

def _ensure_module(import_name: str, pip_name: str | None = None) -> None:
    """Import a module, pip-installing it first if it is missing.

    Kaggle base images do not ship scapy; feature extraction from PCAP needs it.
    Installing here (only when extraction is actually required) keeps the notebook
    self-contained - no separate '!pip install' cell to remember.
    """
    import importlib
    pip_name = pip_name or import_name
    try:
        importlib.import_module(import_name)
        return
    except ImportError:
        print(f"[deps] '{import_name}' not found - installing '{pip_name}' ...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", pip_name],
            check=True,
        )
        importlib.invalidate_caches()
        importlib.import_module(import_name)
        print(f"[deps] '{import_name}' installed.")


# Extracts per-packet feature sequences for unknown keywords from raw PCAPs.
def extract_unknown_features(unknown_pcap_dir: str, unknown_npz: str,
                              n_workers: int = 2) -> None:
    """Extract the same 3-channel sequence + 15 global features for unknown keywords.

    Uses the identical Config as the known dataset extraction so that feature
    distributions are directly comparable (same max_packets, min_payload, etc.).
    Classes are auto-discovered from subdirectory names - no need to hard-code them.
    """
    _ensure_module("scapy")  # extract_features_v2 imports scapy at module load
    sys.path.insert(0, str(Path(__file__).parent))
    from extract_features_v2 import Config, build_dataset  # noqa: PLC0415

    cfg = Config(
        data_dir=unknown_pcap_dir,
        output_file=unknown_npz,
        target_port=443,
        min_payload=10,
        min_valid_packets=5,
        max_packets=500,
        n_workers=n_workers,
        classes_filter=[],   # auto-discover all keyword sub-folders
    )
    print(f"\n[Step 1] Extracting unknown keyword features")
    print(f"  source : {unknown_pcap_dir}")
    print(f"  output : {unknown_npz}")
    build_dataset(cfg)


# ---------------------------------------------------------------------------
# Step 2: OOD evaluation (delegates to open_world.py subprocess)
# ---------------------------------------------------------------------------

# Builds and runs the open_world.py subprocess command for OOD evaluation.
def run_ood_evaluation(
    known_npz: str,
    unknown_npz: str,
    checkpoint: str,
    results_dir: str,
    methods: str,
    batch_size: int,
    loader_workers: int,
    energy_temperature: float,
    threshold_metric: str,
    save_logits: bool,
) -> None:
    cmd = [
        sys.executable, str(Path(__file__).parent / "open_world.py"),
        "--known_npz",          known_npz,
        "--unknown_npz",        unknown_npz,
        "--checkpoint",         checkpoint,
        "--results_dir",        results_dir,
        "--methods",            methods,
        "--batch_size",         str(batch_size),
        "--loader_workers",     str(loader_workers),
        "--energy_temperature", str(energy_temperature),
        "--threshold_metric",   threshold_metric,
    ]
    if save_logits:
        cmd.append("--save_logits")

    print(f"\n[Step 2] Running OOD evaluation")
    print("  " + " \\\n    ".join(cmd))
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Entry point: parses CLI arguments, runs feature extraction then OOD evaluation.
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract unknown NPZ + run open-world OOD evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    parser.add_argument("--known_npz",
                        default="./dataset_kfp_v2_macos_1_50.npz",  # Example Kaggle path: /kaggle/working/dataset_kfp_v2_macos_1_50.npz
                        help="Known dataset NPZ (50 keywords, already extracted).")
    # Example Kaggle path: /kaggle/input/datasets/linhnpcshust/openworld_10kw
    parser.add_argument("--unknown_pcap_dir",
                        default="./data/openworld_10kw",
                        help="Root dir containing one sub-folder per unknown keyword.")
    parser.add_argument("--unknown_npz",
                        default="./dataset_kfp_v2_openworld_10kw.npz",  # Example Kaggle path: /kaggle/working/dataset_kfp_v2_openworld_10kw.npz
                        help="Cache path for extracted unknown features (auto-reused).")
    parser.add_argument("--checkpoint",
                        default="./results_bigru/best_model.pt",  # Example Kaggle path: /kaggle/working/results_bigru/best_model.pt
                        help="Trained model checkpoint (.pt).")
    parser.add_argument("--results_dir",
                        default="./results_open_world",  # Example Kaggle path: /kaggle/working/results_open_world
                        help="Output directory for JSON/CSV/NPZ results.")

    # Evaluation settings
    parser.add_argument("--methods",
                        default="softmax,energy,mahalanobis",
                        help="Comma-separated OOD methods: softmax, energy, mahalanobis.")
    parser.add_argument("--batch_size",       type=int,   default=64)
    parser.add_argument("--loader_workers",   type=int,   default=2)
    parser.add_argument("--energy_temperature", type=float, default=1.0)
    parser.add_argument("--threshold_metric", choices=["accuracy"], default="accuracy")

    # Flags
    parser.add_argument("--force_reextract", action="store_true",
                        help="Re-extract unknown NPZ even if the cache file already exists.")
    parser.add_argument("--save_logits", action="store_true",
                        help="Cache raw logits and per-method scores as NPZ for offline analysis.")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Pre-flight checks
    # ------------------------------------------------------------------
    if not Path(args.known_npz).exists():
        raise FileNotFoundError(
            f"Known NPZ not found: {args.known_npz}\n"
            "Run extract_features_v2.py on the 50-keyword dataset first."
        )
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Run train_resnet_bigru.py on the known dataset first:\n"
            "  python train_resnet_bigru.py \\\n"
            f"    --npz {args.known_npz} \\\n"
            "    --results_dir ./results_bigru \\\n"
            "    --epochs 80 --batch_size 64 --lr 1e-3 --patience 12 \\\n"
            "    --k_features 15 --use_augment --loss focal"
        )

    # ------------------------------------------------------------------
    # Step 1: Extract unknown features
    # ------------------------------------------------------------------
    if not Path(args.unknown_npz).exists() or args.force_reextract:
        extract_unknown_features(
            args.unknown_pcap_dir, args.unknown_npz, args.loader_workers)
    else:
        print(f"\n[Step 1] Unknown NPZ already exists - skipping extraction.")
        print(f"  {args.unknown_npz}  (pass --force_reextract to rebuild)")

    # Summarise unknown dataset
    d = np.load(args.unknown_npz, allow_pickle=True)
    classes_u = [str(c).replace("_", " ").strip() for c in d["classes"].tolist()]
    counts_u  = np.bincount(d["y"].astype(np.int64), minlength=len(classes_u))
    print(f"\nUnknown dataset: {len(d['y'])} samples, {len(classes_u)} keywords")
    for kw, n in zip(classes_u, counts_u):
        print(f"  {kw:<25} {n:>4} samples")

    # ------------------------------------------------------------------
    # Step 2: OOD evaluation
    # ------------------------------------------------------------------
    run_ood_evaluation(
        known_npz=args.known_npz,
        unknown_npz=args.unknown_npz,
        checkpoint=args.checkpoint,
        results_dir=args.results_dir,
        methods=args.methods,
        batch_size=args.batch_size,
        loader_workers=args.loader_workers,
        energy_temperature=args.energy_temperature,
        threshold_metric=args.threshold_metric,
        save_logits=args.save_logits,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\nPipeline complete.  Results directory: {args.results_dir}")
    out_dir = Path(args.results_dir)
    if out_dir.exists():
        for f in sorted(out_dir.iterdir()):
            size_kb = f.stat().st_size // 1024
            print(f"  {f.name:<45} {size_kb:>6} KB")


if __name__ == "__main__":
    main()
