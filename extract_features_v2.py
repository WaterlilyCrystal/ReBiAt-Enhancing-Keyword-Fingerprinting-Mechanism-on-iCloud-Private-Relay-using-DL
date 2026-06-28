"""
extract_features_v2.py
Keyword traffic fingerprinting -- feature extraction from raw PCAP files.

Output arrays (saved to .npz):
  X_seq    (N, MAX_PACKETS, 3)  per-packet sequence features
  X_global (N, 15)              flow-level aggregate features
  y        (N,)                 integer class labels
  classes  (K,)                 class name strings

Sequence channels (3) -- Methodology 3.3 (minimal sufficient set, no linear redundancy):
  0  x_signed   dir * payload / 1500                signed size (dir + magnitude)
  1  x_iat      log1p(inter-arrival time ms)        timing, skew-compressed
  2  x_cum      cumsum(payload) / total_payload     cumulative byte fraction

Global features (15) -- fixed curated descriptor set from the offline feature study:
  Group A  Packet size distribution  [0-5]
  Group B  Timing & throughput       [6-9]
  Group C  Traffic asymmetry         [10-11]
  Group D  Server response + bursts  [12-14]
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scapy.all import IP, IPv6, TCP, UDP, PcapReader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

log = logging.getLogger(__name__)


# Config dataclass holding all extraction parameters and default paths.
@dataclass
class Config:
    # Example Kaggle path: /kaggle/input/datasets/linhnpcshust/macos-26-50
    data_dir: str = "./data/macos-26-50"
    target_port: int = 443
    min_payload: int = 10
    min_valid_packets: int = 5
    max_packets: int = 500
    n_workers: int = 4
    output_file: str = "dataset_kfp_v2.npz"
    classes_filter: list[str] = field(default_factory=list)
    shuffle: bool = False


# Parse a single PCAP file and return a list of qualifying packet dicts.
def _read_pcap(path: str, cfg: Config) -> list[dict]:
    raw = []
    try:
        with PcapReader(path) as pcap:
            for pkt in pcap:
                if IP not in pkt and IPv6 not in pkt:
                    continue
                if TCP in pkt:
                    layer = pkt[TCP]
                elif UDP in pkt:
                    layer = pkt[UDP]
                else:
                    continue
                payload = len(layer.payload)
                if payload < cfg.min_payload:
                    continue
                t = float(pkt.time)
                if layer.sport == cfg.target_port:
                    raw.append({"time": t, "dir": 1, "size": payload})
                elif layer.dport == cfg.target_port:
                    raw.append({"time": t, "dir": -1, "size": payload})
    except Exception as e:
        log.debug("PCAP parse error in %s: %s", path, e)

    if len(raw) < cfg.min_valid_packets:
        return []
    raw.sort(key=lambda p: p["time"])
    return raw


# Annotate each packet with burst ID, intra-burst position, burst byte total, and response latency.
def _annotate_bursts(pkts: list[dict]) -> list[dict]:
    burst_bytes: dict[int, int] = {}
    burst_id, cur_dir = 1, pkts[0]["dir"]
    for p in pkts:
        if p["dir"] != cur_dir:
            burst_id += 1
            cur_dir = p["dir"]
        burst_bytes[burst_id] = burst_bytes.get(burst_id, 0) + p["size"]

    burst_id, cur_dir, pos = 1, pkts[0]["dir"], 0
    last_out_t: Optional[float] = None
    out = []
    for p in pkts:
        if p["dir"] != cur_dir:
            resp = (p["time"] - last_out_t) * 1000.0 if (cur_dir == -1 and p["dir"] == 1 and last_out_t) else 0.0
            burst_id += 1
            cur_dir = p["dir"]
            pos = 0
        else:
            resp = 0.0
        if p["dir"] == -1:
            last_out_t = p["time"]
        out.append({**p, "burst_id": burst_id, "pos": pos, "burst_bytes": burst_bytes[burst_id], "resp_ms": resp})
        pos += 1
    return out


def _build_seq(pkts: list[dict], max_pkts: int) -> np.ndarray:
    """
Build per-packet sequential features for a network flow.

Output shape: (max_pkts, 3)

Columns:
    0. Signed packet size
       dir * size / 1500

       Encodes packet direction and size.

    1. Log-scaled inter-arrival time (IAT)
       log10(1 + IAT_ms)

       Captures timing gap between consecutive packets. log10() reduces skew.

    2. Flow progress
       cumulative_bytes / total_bytes

       Indicates the packet's relative position within the flow.

Flows shorter than max_pkts are zero-padded; longer flows are truncated.
"""

    mat = np.zeros((max_pkts, 3), dtype=np.float32)
    total_bytes = float(sum(p["size"] for p in pkts)) or 1.0
    prev_t, cum = pkts[0]["time"], 0.0
    for i, p in enumerate(pkts[:max_pkts]):
        iat_ms = max((p["time"] - prev_t) * 1000.0, 0.0)
        prev_t = p["time"]
        cum += p["size"]
        mat[i, 0] = p["dir"] * (p["size"] / 1500.0)   # signed size (dir x magnitude)
        mat[i, 1] = np.log10(1.0 + iat_ms)            # log10(1 + IAT in ms)
        mat[i, 2] = cum / total_bytes                 # cumulative byte fraction
    return mat


def _build_global(pkts: list[dict]) -> np.ndarray:
    """
    Flow-level 15-dim feature vector
    Features are inspired by prior WF literature; the specific subset and all
    log1p transforms were selected via ANOVA F-score ranking on the collected
    dataset.

    Group A - Packet size distribution [0-5]
      Inspired by: Panchenko et al., "Website Fingerprinting in Onion Routing
      Based Anonymization Networks", ACM WPES 2011.
      Bins are 8 equal-width intervals over [0, 1500] B; indices 0,2,6,7
      retained after ANOVA selection.
      0  avg_size_norm   mean(payload) / 1500
      1  std_size_norm   std(payload)  / 1500
      2  hist_small      fraction of packets in [  0,  187] B
      3  hist_mid        fraction of packets in [375,  562] B
      4  hist_large      fraction of packets in [1125, 1312] B
      5  hist_max        fraction of packets in [1312, 1500] B

    Group B - Timing & throughput [6-9]
      IAT features inspired by: Rahman et al., "Tik-Tok: The Utility of Packet
      Timing in Website Fingerprinting Attacks", PoPETs 2020.
      Throughput and duration are standard NetFlow statistics.
      6  avg_iat_log        mean log1p(inter-arrival time ms)
      7  std_iat_log        std  log1p(inter-arrival time ms)
      8  throughput_log     log1p(total_bytes / flow_duration_s)
      9  flow_duration_log  log1p(flow_duration_s)

    Group C - Traffic asymmetry [10-11]
      Inspired by: Wang et al., "Effective Attacks and Provable Defenses for
      Website Fingerprinting", USENIX Security 2014.
      The log1p ratio form is a custom transformation applied here.
      10 incoming_bytes_ratio  in_bytes / total_bytes
      11 asymmetry_log         log1p(in_bytes / out_bytes)

    Group D - Server response & burst activity [12-14]
      Response latency inspired by: Rahman et al., PoPETs 2020.
      Burst count inspired by: Wang et al., USENIX Security 2014.
      Normalisation by n_packets (burst_count_norm) is a custom adaptation.
      12 mean_resp_log    mean log1p(first-response latency ms)
      13 std_resp_log     std  log1p(first-response latency ms)
      14 burst_count_norm n_bursts / n_packets
    """
    sizes  = np.array([p["size"] for p in pkts], dtype=np.float32)
    dirs   = np.array([p["dir"]  for p in pkts], dtype=np.int8)
    times  = np.array([p["time"] for p in pkts], dtype=np.float64)
    resps  = np.array([p["resp_ms"] for p in pkts if p["resp_ms"] > 0], dtype=np.float32)

    n       = len(pkts)
    in_mask = dirs == 1
    total_b = float(sizes.sum())
    in_b    = float(sizes[in_mask].sum())
    out_b   = total_b - in_b
    dur     = float(times[-1] - times[0])

    iats_log = np.log1p(np.maximum(np.diff(times) * 1000.0, 0.0)) if n > 1 else np.zeros(1)

    burst_ids = [p["burst_id"] for p in pkts]
    n_bursts  = max(burst_ids)

    hist, _ = np.histogram(sizes, bins=8, range=(0, 1500))
    hist_f  = hist / n

    g = np.zeros(15, dtype=np.float32)
    g[0]  = float(sizes.mean()) / 1500.0
    g[1]  = float(sizes.std())  / 1500.0
    g[2]  = hist_f[0]
    g[3]  = hist_f[2]
    g[4]  = hist_f[6]
    g[5]  = hist_f[7]
    g[6]  = float(iats_log.mean())
    g[7]  = float(iats_log.std())
    g[8]  = np.log1p(total_b / max(dur, 1e-6))
    g[9]  = np.log1p(dur)
    g[10] = in_b / max(total_b, 1.0)
    g[11] = np.log1p(in_b / max(out_b, 1.0))
    if len(resps) > 0:
        r     = np.log1p(resps)
        g[12] = float(r.mean())
        g[13] = float(r.std())
    g[14] = n_bursts / n

    return g


# Worker function that reads and processes a single PCAP file into feature arrays.
def process_pcap(args: tuple) -> Optional[tuple]:
    path, cfg = args
    pkts = _read_pcap(path, cfg)
    if not pkts:
        return None
    ann = _annotate_bursts(pkts)
    return _build_seq(ann, cfg.max_packets), _build_global(ann), pkts[0]["time"], pkts[-1]["time"]


def _count_pcap(args: tuple) -> tuple[int, str]:
    """Dry-run worker: count qualifying packets without building feature arrays.
    Returns (n_qualifying_packets, error_message).
    """
    path, cfg = args
    n_match = 0
    try:
        with PcapReader(path) as pcap:
            for pkt in pcap:
                if IP not in pkt and IPv6 not in pkt:
                    continue
                if TCP in pkt:
                    layer = pkt[TCP]
                elif UDP in pkt:
                    layer = pkt[UDP]
                else:
                    continue
                payload = len(layer.payload)
                if payload < cfg.min_payload:
                    continue
                if layer.sport == cfg.target_port or layer.dport == cfg.target_port:
                    n_match += 1
    except Exception as e:
        return n_match, str(e)[:120]
    return n_match, ""


def dry_run_check(cfg: Config) -> dict[str, dict]:
    """Count valid traces per class WITHOUT building feature arrays.

    Prints a per-class table so you can identify sparse/broken classes
    before committing to a full (slow) extraction run.

    Returns {class_name: {"files": int, "valid": int, "skipped": int, "errors": int}}.
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    classes, folder_map, missing = discover_classes(cfg.data_dir, cfg.classes_filter)
    if not classes:
        log.error("No keyword folders found in %s", cfg.data_dir)
        return {}
    if missing:
        log.warning("Missing keyword folders (%d): %s", len(missing), missing)

    paths, labels = collect_files(classes, folder_map)
    log.info("Dry-run: checking %d PCAP files across %d classes (no feature arrays built)\n",
             len(paths), len(classes))

    stats: dict[str, dict] = {cls: {"files": 0, "valid": 0, "skipped": 0, "errors": 0}
                               for cls in classes}
    for lbl in labels:
        stats[classes[lbl]]["files"] += 1

    with ProcessPoolExecutor(max_workers=cfg.n_workers) as ex:
        futs = {ex.submit(_count_pcap, (p, cfg)): labels[i]
                for i, p in enumerate(paths)}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Checking"):
            cls = classes[futs[fut]]
            try:
                n_pkts, err = fut.result()
            except Exception:
                stats[cls]["errors"] += 1
                continue
            if err:
                stats[cls]["errors"] += 1
            elif n_pkts >= cfg.min_valid_packets:
                stats[cls]["valid"] += 1
            else:
                stats[cls]["skipped"] += 1

    total_files = sum(s["files"] for s in stats.values())
    total_valid = sum(s["valid"] for s in stats.values())

    print("\n" + "=" * 72)
    print(f"{'CLASS':<30} {'FILES':>6} {'VALID':>6} {'SKIP':>6} {'ERR':>5} {'RATE':>7}  STATUS")
    print("-" * 72)
    for cls in classes:
        s = stats[cls]
        rate = s["valid"] / max(s["files"], 1)
        flag = "  *** LOW ***" if s["valid"] < 10 else ""
        print(f"{cls:<30} {s['files']:>6} {s['valid']:>6} {s['skipped']:>6} "
              f"{s['errors']:>5} {rate:>6.1%}{flag}")
    print("=" * 72)
    print(f"{'TOTAL':<30} {total_files:>6} {total_valid:>6}")
    print()

    low = [c for c in classes if stats[c]["valid"] < 10]
    if low:
        log.warning("Classes with < 10 valid traces (%d): %s", len(low), low)
        log.warning("Try: --min_payload 1 --min_pkts 2  to loosen filter")
    else:
        log.info("All classes >= 10 valid traces. Ready for full extraction.")

    return stats


_EXCLUDE_DIRS = {".vscode", ".idea", ".git", "__pycache__", "paper",
                 "results_varcnn_paper", "results"}


# Normalize a keyword folder name for case-insensitive, space-insensitive matching.
def _normalize_keyword(name: str) -> str:
    return " ".join(name.strip().replace("_", " ").split()).lower()


# List all non-excluded subdirectories under root_dir and return a normalized-name to folder-name map.
def _list_keyword_dirs(root_dir: str) -> dict[str, str]:
    dirs = sorted(
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and d not in _EXCLUDE_DIRS
    )
    dir_map: dict[str, str] = {}
    for d in dirs:
        norm = _normalize_keyword(d)
        if norm in dir_map:
            log.warning("Duplicate normalized folder name in %s: %r maps to both %r and %r", root_dir, norm, dir_map[norm], d)
            continue
        dir_map[norm] = d
    return dir_map


# Discover available keyword classes in data_dir, optionally filtered to a subset.
def discover_classes(
    data_dir: str,
    classes_filter: list[str],
) -> tuple[list[str], dict[str, str], list[str]]:
    dir_map = _list_keyword_dirs(data_dir)
    if not classes_filter:
        classes = sorted(dir_map.keys())
        return classes, {cls: os.path.join(data_dir, dir_map[cls]) for cls in classes}, []
    classes: list[str] = []
    folder_map: dict[str, str] = {}
    missing: list[str] = []
    for wanted in classes_filter:
        actual = dir_map.get(_normalize_keyword(wanted))
        if actual is None:
            missing.append(wanted)
        else:
            classes.append(wanted)
            folder_map[wanted] = os.path.join(data_dir, actual)
    return classes, folder_map, missing


# Collect all PCAP file paths and corresponding integer labels for the given class list.
def collect_files(classes: list[str], folder_map: dict[str, str]) -> tuple[list[str], list[int]]:
    paths, labels = [], []
    for idx, kw in enumerate(classes):
        folder = folder_map[kw]
        pcaps  = (
            glob.glob(os.path.join(folder, "**", "*.pcap"), recursive=True) +
            glob.glob(os.path.join(folder, "**", "*.pcapng"), recursive=True)
        )
        pcaps = sorted(pcaps)
        paths.extend(pcaps)
        labels.extend([idx] * len(pcaps))
    return paths, labels


# Run the full feature extraction pipeline and save results to a compressed .npz file.
def build_dataset(cfg: Config) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    classes, folder_map, missing = discover_classes(cfg.data_dir, cfg.classes_filter)
    if not classes:
        log.error("No keyword folders found in %s", cfg.data_dir)
        return
    if missing:
        log.warning("Missing keyword folders (%d): %s", len(missing), missing)
    log.info("Classes (%d): %s", len(classes), classes)

    paths, labels = collect_files(classes, folder_map)
    log.info("PCAP files to process: %d", len(paths))
    if not paths:
        log.error("No PCAP files found.")
        return

    file_counts = np.zeros(len(classes), dtype=np.int64)
    for lbl in labels:
        file_counts[lbl] += 1

    args_list = [(p, cfg) for p in paths]
    X_seq_list, X_global_list, y_list = [], [], []
    file_path_list, start_time_list, end_time_list = [], [], []
    n_fail = 0

    with ProcessPoolExecutor(max_workers=cfg.n_workers) as ex:
        futs = {ex.submit(process_pcap, a): (lbl, a[0]) for a, lbl in zip(args_list, labels)}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Extracting"):
            lbl, path = futs[fut]
            try:
                result = fut.result()
            except Exception as e:
                log.debug("Worker error: %s", e)
                result = None
            if result is None:
                n_fail += 1
                continue
            seq, glb, start_t, end_t = result
            X_seq_list.append(seq)
            X_global_list.append(glb)
            y_list.append(lbl)
            file_path_list.append(path)
            start_time_list.append(start_t)
            end_time_list.append(end_t)

    log.info("Valid: %d  |  Skipped: %d", len(y_list), n_fail)
    if not y_list:
        log.error("No valid samples extracted.")
        return

    X_seq    = np.array(X_seq_list,    dtype=np.float32)
    X_global = np.array(X_global_list, dtype=np.float32)
    y        = np.array(y_list,        dtype=np.int64)
    file_paths = np.array(file_path_list, dtype=str)
    start_times = np.array(start_time_list, dtype=np.float64)
    end_times = np.array(end_time_list, dtype=np.float64)
    sample_order = np.arange(len(y), dtype=np.int64)

    if cfg.shuffle:
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(y))
        X_seq, X_global, y = X_seq[idx], X_global[idx], y[idx]
        file_paths, start_times, end_times = file_paths[idx], start_times[idx], end_times[idx]
        sample_order = sample_order[idx]

    np.savez_compressed(cfg.output_file,
                        X_seq=X_seq, X_global=X_global,
                        y=y, classes=np.array(classes),
                        file_paths=file_paths,
                        capture_start_time=start_times,
                        capture_end_time=end_times,
                        sample_order=sample_order)

    with open(cfg.output_file.replace(".npz", "_meta.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    valid_counts = np.bincount(y, minlength=len(classes))
    zero_valid = [classes[i] for i in range(len(classes)) if valid_counts[i] == 0]
    if zero_valid:
        log.warning("Keywords with folder/pcap found but zero valid traces (%d): %s", len(zero_valid), zero_valid)

    log.info("Saved -> %s", cfg.output_file)
    log.info("X_seq    %s  (samples x packets x 3)", X_seq.shape)
    log.info("X_global %s  (samples x 15)",           X_global.shape)
    log.info("y        %s",                           y.shape)
    for i, cls in enumerate(classes):
        log.info("  [%2d] %-30s  files=%d  valid=%d", i, cls, int(file_counts[i]), int(valid_counts[i]))



_FEATURE_NAMES = [
    "avg_size_norm", "std_size_norm",
    "hist_small",    "hist_mid",    "hist_large",   "hist_max",
    "avg_iat_log",   "std_iat_log", "throughput_log", "flow_duration_log",
    "incoming_bytes_ratio", "asymmetry_log",
    "mean_resp_log", "std_resp_log", "burst_count_norm",
]


# Run ANOVA F-score analysis on the global features of a saved .npz dataset.
def analyze_features(npz_path: str, k_select: int = 15) -> None:
    from sklearn.feature_selection import f_classif
    data    = np.load(npz_path, allow_pickle=True)
    X_gl    = data["X_global"]
    y       = data["y"]
    classes = data["classes"].tolist()

    print(f"\nDataset: {len(y)} samples | {len(classes)} classes")
    print(f"X_global shape: {X_gl.shape}")
    print(f"\nANOVA F-score ranking:\n")

    f_scores, p_vals = f_classif(X_gl, y)
    order = np.argsort(f_scores)[::-1]

    print(f"{'Rank':<5} {'Feature':<25} {'F-score':>10} {'p-value':>12}")
    print("-" * 55)
    for rank, i in enumerate(order, 1):
        print(f"{rank:<5} {_FEATURE_NAMES[i]:<25} {f_scores[i]:>10.2f} {p_vals[i]:>12.2e}")


if __name__ == "__main__":
    # On Kaggle: /kaggle/working
    KAGGLE_WORK = "."

    parser = argparse.ArgumentParser(description="Keyword fingerprinting feature extractor v2")
    # Example Kaggle path: /kaggle/input/datasets/linhnpcshust/macos-26-50
    parser.add_argument("--root",         default="./data/macos-26-50",
                        help="PCAP root directory")
    parser.add_argument("--out",          default=f"{KAGGLE_WORK}/dataset_kfp_v2.npz",  # set this to your .npz file path
                        help="Output .npz path (ignored in --dry_run)")
    parser.add_argument("--keywords",      default="",
                        help="Comma-separated keyword list; empty = all folders in --root")
    parser.add_argument("--keywords_file", default="",
                        help="Path to .txt file with one keyword per line (merged with --keywords)")
    parser.add_argument("--port",         type=int, default=443)
    parser.add_argument("--min_payload",  type=int, default=10,
                        help="Minimum UDP/TCP payload bytes to keep a packet")
    parser.add_argument("--min_pkts",     type=int, default=5,
                        help="Minimum qualifying packets for a trace to be valid")
    parser.add_argument("--max_pkts",     type=int, default=500)
    parser.add_argument("--workers",      type=int, default=2)
    parser.add_argument("--dry_run",      action="store_true",
                        help="Count valid traces per class only - do NOT build feature arrays")
    parser.add_argument("--analyze",      action="store_true",
                        help="Run ANOVA feature analysis after full extraction")
    parser.add_argument("--shuffle",      action="store_true",
                        help="Shuffle samples before saving. Leave off for temporal splits.")
    a = parser.parse_args()

    kw_list = [k.strip() for k in a.keywords.split(",") if k.strip()] if a.keywords else []
    if a.keywords_file:
        with open(a.keywords_file, encoding="utf-8") as f:
            kw_list += [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

    cfg = Config(
        data_dir=a.root,
        target_port=a.port,
        min_payload=a.min_payload,
        min_valid_packets=a.min_pkts,
        max_packets=a.max_pkts,
        n_workers=a.workers,
        output_file=a.out,
        classes_filter=kw_list,
        shuffle=a.shuffle,
    )

    if a.dry_run:
        dry_run_check(cfg)
    else:
        build_dataset(cfg)
        if a.analyze:
            analyze_features(cfg.output_file)
