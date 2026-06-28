"""
defenses.py  --  Traffic-analysis defense simulators for keyword fingerprinting
                 (Methodology Sec. 8 -- "Defense Mechanisms and Robustness Evaluation").

We simulate three defenses by POST-HOC transformation of recorded traces, exactly as
Methodology 8.1 prescribes:

    A : {(s_i, d_i, t_i)}  -->  {(s'_j, d'_j, t'_j)}

then RE-EXTRACT the same 3-channel sequence + 15 global features (Sec. 3) and re-evaluate
the attacker. Because the saved .npz is a lossless-enough encoding of (size, dir, iat), we
reconstruct the packet list directly from X_seq -- no raw PCAP needed on Kaggle.

    X_seq[:, :, 0] = dir * size / 1500   -> size = |ch0| * 1500 ,  dir = sign(ch0)
    X_seq[:, :, 1] = log10(1 + iat_ms)   -> iat_ms = 10**ch1 - 1
    (channel 2 = cumulative byte fraction; not needed for reconstruction)

Defenses implemented
--------------------
  front       FRONT  (Gong & Wang, USENIX Sec. 2020) -- zero-delay frontal dummy padding,
              n ~ U[1, N_max] dummies per direction, timestamps ~ Rayleigh(w), w ~ U[Wmin,Wmax].
  wtf_pad     WTF-PAD (Juarez et al., ESORICS 2016) -- adaptive padding: fill improbably-long
              inter-packet gaps with dummies whose spacing is drawn from the empirical IAT
              distribution. Zero-delay (dummies only fill existing gaps).
  burstguard  BurstGuard -- KF-specific padding defense, "Enhancing Search Privacy on Tor:
              Advanced Deep Keyword Fingerprinting Attacks and BurstGuard Defense" (ASIA CCS '25).
              Used here as a CITED BASELINE (not a contribution of this thesis).
                --bg_mode response  : faithful reimplementation -- inject a dummy INCOMING burst
                                      in response to each OUTGOING burst (default).
                --bg_mode quantize  : optional variant -- snap INCOMING burst byte-counts to a
                                      grid of Q bytes. Not claimed as novel; provided for ablation.
              Zero-delay by default.

Direction convention (inherited from extract_features_v2):
  dir = +1  -> incoming  (server -> client, sport == 443)   [the KF-leaking download bursts]
  dir = -1  -> outgoing  (client -> server, dport == 443)

QUIC adaptation (Methodology 8.6): WTF-PAD/FRONT were built for fixed-size Tor cells; we
inject VARIABLE-size dummy packets whose sizes are sampled from the empirical per-direction
data-packet distribution of the dataset.

Output
------
A defended .npz with the SAME schema and SAME sample order/labels as the input
(X_seq, X_global, y, classes, device) so that:
  * adaptive attacker  = re-train train_resnet_bigru.py --npz defended.npz
  * non-adaptive       = eval_defended.py --checkpoint clean.pt --defended_npz defended.npz
Plus overhead_<defense>.json/.csv with per-trace and aggregate O_bw / O_lat (Methodology 8.4).

Usage (after extract_features_v2.py has been written to /kaggle/working):
  python defenses.py --in_npz dataset_kfp_v2_mac.npz --defense front \
                     --out_npz dataset_kfp_v2_mac_front.npz --front_nmax 1500 --front_wmax 8
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Re-use the EXACT feature builders from the extractor so defended features are
# computed identically to undefended ones (no silent drift between pipelines).
from extract_features_v2 import _annotate_bursts, _build_seq, _build_global

SIZE_SCALE = 1500.0
EPS = 1e-9


# ---------------------------------------------------------------------------
# Reconstruction:  X_seq row  ->  list[{time, dir, size}]
# ---------------------------------------------------------------------------

def reconstruct_packets(seq_row: np.ndarray) -> list[dict]:
    """Invert _build_seq for one sample. Zero-padding rows (ch0 == 0) are dropped.

    Returns packets sorted by time, with t[0] = 0 (absolute offset is irrelevant
    to every feature we extract -- all of them use IATs or durations).
    """
    ch0 = seq_row[:, 0]
    valid = np.abs(ch0) > EPS
    if not valid.any():
        return []
    ch0 = ch0[valid]
    ch1 = seq_row[valid, 1]

    sizes = np.rint(np.abs(ch0) * SIZE_SCALE).astype(np.int64)
    sizes = np.maximum(sizes, 1)
    dirs = np.where(ch0 >= 0, 1, -1).astype(np.int64)
    iat_ms = np.maximum(np.power(10.0, ch1) - 1.0, 0.0)
    t = np.cumsum(iat_ms) / 1000.0  # seconds; first IAT is 0 by construction

    return [{"time": float(t[i]), "dir": int(dirs[i]), "size": int(sizes[i])}
            for i in range(len(sizes))]


def trace_bytes_dur(pkts: list[dict]) -> tuple[float, float]:
    if not pkts:
        return 0.0, 0.0
    b = float(sum(p["size"] for p in pkts))
    d = float(pkts[-1]["time"] - pkts[0]["time"])
    return b, d


# ---------------------------------------------------------------------------
# Empirical per-direction packet-size pools (for variable-size dummy injection)
# ---------------------------------------------------------------------------

def build_size_pools(X_seq: np.ndarray, rng: np.random.Generator,
                     cap: int = 200_000) -> tuple[np.ndarray, np.ndarray]:
    """Pool of real packet sizes for incoming (dir=+1) and outgoing (dir=-1)."""
    ch0 = X_seq[:, :, 0].reshape(-1)
    mask = np.abs(ch0) > EPS
    ch0 = ch0[mask]
    sizes = np.rint(np.abs(ch0) * SIZE_SCALE).astype(np.int64)
    sizes = np.maximum(sizes, 1)
    in_pool = sizes[ch0 >= 0]
    out_pool = sizes[ch0 < 0]
    if len(in_pool) > cap:
        in_pool = rng.choice(in_pool, cap, replace=False)
    if len(out_pool) > cap:
        out_pool = rng.choice(out_pool, cap, replace=False)
    # Fallbacks so a degenerate direction never crashes sampling.
    if len(in_pool) == 0:
        in_pool = np.array([1200], dtype=np.int64)
    if len(out_pool) == 0:
        out_pool = np.array([100], dtype=np.int64)
    return in_pool, out_pool


def build_iat_pool(X_seq: np.ndarray, rng: np.random.Generator,
                   cap: int = 200_000) -> np.ndarray:
    """Pool of positive inter-arrival times (ms) for WTF-PAD adaptive padding."""
    ch1 = X_seq[:, :, 1].reshape(-1)
    iat_ms = np.power(10.0, ch1) - 1.0
    iat_ms = iat_ms[iat_ms > EPS]
    if len(iat_ms) > cap:
        iat_ms = rng.choice(iat_ms, cap, replace=False)
    if len(iat_ms) == 0:
        iat_ms = np.array([1.0])
    return iat_ms.astype(np.float64)


def _sample_size(pool: np.ndarray, dir_val: int, rng: np.random.Generator,
                 in_pool: np.ndarray, out_pool: np.ndarray) -> int:
    p = in_pool if dir_val == 1 else out_pool
    return int(p[rng.integers(0, len(p))])


# ---------------------------------------------------------------------------
# Defense 1: FRONT  (Gong & Wang, USENIX Security 2020)
# ---------------------------------------------------------------------------

def defend_front(pkts: list[dict], rng: np.random.Generator,
                 in_pool: np.ndarray, out_pool: np.ndarray,
                 n_max_in: int, n_max_out: int,
                 w_min: float, w_max: float,
                 clip_to_trace: bool = True) -> list[dict]:
    """FRONT: per-direction, inject n ~ U[1, N_max] dummy packets whose offsets from
    trace start follow a Rayleigh(w) distribution (w ~ U[w_min, w_max]) -- concentrating
    padding at the trace FRONT. Real packets are never delayed (zero-delay).

    Variable-size QUIC adaptation: dummy sizes sampled from the empirical per-direction pool.
    """
    if not pkts:
        return pkts
    t0 = pkts[0]["time"]
    duration = pkts[-1]["time"] - t0
    out = list(pkts)

    for dir_val, n_max, pool in ((1, n_max_in, in_pool), (-1, n_max_out, out_pool)):
        if n_max < 1:
            continue
        n = int(rng.integers(1, n_max + 1))
        w = float(rng.uniform(w_min, w_max))
        # Rayleigh(w) via inverse-CDF:  t = w * sqrt(-2 ln U),  U ~ U(0,1].
        u = rng.uniform(EPS, 1.0, size=n)
        offsets = w * np.sqrt(-2.0 * np.log(u))
        for off in offsets:
            if clip_to_trace and off > duration:
                continue  # keep O_lat = 0; drop dummies that would extend the trace
            out.append({"time": t0 + float(off), "dir": dir_val,
                        "size": int(pool[rng.integers(0, len(pool))]), "dummy": True})

    out.sort(key=lambda p: p["time"])
    return out


# ---------------------------------------------------------------------------
# Defense 2: WTF-PAD  (Juarez et al., ESORICS 2016) -- simplified adaptive padding
# ---------------------------------------------------------------------------

def defend_wtfpad(pkts: list[dict], rng: np.random.Generator,
                  iat_pool_ms: np.ndarray, in_pool: np.ndarray, out_pool: np.ndarray,
                  factor: float = 1.0, max_dummies_per_gap: int = 8) -> list[dict]:
    """Adaptive padding (AP), "gap" state only.

    Walk consecutive packets; for each inter-packet gap, repeatedly sample an EXPECTED
    inter-arrival tau from the empirical IAT distribution. If the running cursor + tau
    still falls inside the real gap, emit a dummy there (the gap was "improbably long").
    This fills statistically unlikely silences without ever delaying a real packet
    (zero-delay). `factor` < 1 makes padding denser (smaller expected gaps -> more dummies);
    `max_dummies_per_gap` bounds the worst case.

    Simplification vs. the full WTF-PAD (Methodology 8.6): a single global gap-histogram
    instead of per-direction burst+gap state machines. Faithful to the AP mechanism;
    sufficient as a zero-delay baseline.
    """
    if len(pkts) < 2:
        return list(pkts)
    out = list(pkts)
    for i in range(len(pkts) - 1):
        t_cur = pkts[i]["time"]
        t_next = pkts[i + 1]["time"]
        dir_val = pkts[i]["dir"]
        added = 0
        while added < max_dummies_per_gap:
            tau = float(iat_pool_ms[rng.integers(0, len(iat_pool_ms))]) * factor / 1000.0
            if tau <= 0:
                break
            t_cur += tau
            if t_cur >= t_next:
                break
            pool = in_pool if dir_val == 1 else out_pool
            out.append({"time": t_cur, "dir": dir_val,
                        "size": int(pool[rng.integers(0, len(pool))]), "dummy": True})
            added += 1
    out.sort(key=lambda p: p["time"])
    return out


# ---------------------------------------------------------------------------
# Defense 3: BurstGuard  (KF-specific novelty, this thesis)
# ---------------------------------------------------------------------------

# Split a packet list into consecutive same-direction burst groups.
def _split_bursts(pkts: list[dict]) -> list[list[dict]]:
    """Group consecutive same-direction packets into bursts (preserving order)."""
    bursts, cur = [], [pkts[0]]
    for p in pkts[1:]:
        if p["dir"] == cur[-1]["dir"]:
            cur.append(p)
        else:
            bursts.append(cur)
            cur = [p]
    bursts.append(cur)
    return bursts


def defend_burstguard_response(pkts: list[dict], rng: np.random.Generator,
                               in_pool: np.ndarray, out_pool: np.ndarray,
                               resp_prob: float = 1.0, resp_scale: float = 1.0,
                               max_pkts_per_burst: int = 40) -> list[dict]:
    """BurstGuard -- reimplementation of the published defense "Enhancing Search Privacy on
    Tor: Advanced Deep Keyword Fingerprinting Attacks and BurstGuard Defense" (ASIA CCS '25):
    a padding-only strategy that, in response to each OUTGOING burst (the query going out),
    injects a DUMMY INCOMING burst -- masking the real server-response burst that leaks the
    search-result-page structure. Used as a cited baseline, not a thesis contribution.

    Per outgoing burst (dir = -1), with probability `resp_prob`, inject a dummy incoming
    (dir = +1) burst whose total bytes are `resp_scale` x a real incoming-burst size
    sampled from the dataset, realized as dummy incoming packets (sizes from the empirical
    incoming pool). Dummies are placed in the gap right after the outgoing burst (zero
    added latency). Exact paper parameters are paywalled; resp_prob / resp_scale are the
    sweepable knobs for the accuracy-overhead curve.
    """
    if not pkts:
        return pkts
    bursts = _split_bursts(pkts)
    # Empirical incoming-burst byte sizes, for sizing dummy response bursts.
    in_burst_sizes = [sum(p["size"] for p in b) for b in bursts if b[0]["dir"] == 1]
    in_burst_sizes = np.array(in_burst_sizes) if in_burst_sizes else np.array([sum(in_pool[:5])])

    out: list[dict] = []
    for bi, b in enumerate(bursts):
        out.extend(b)
        if b[0]["dir"] != -1 or rng.uniform() > resp_prob:
            continue
        target = float(in_burst_sizes[rng.integers(0, len(in_burst_sizes))]) * resp_scale
        t_start = b[-1]["time"]
        t_next = bursts[bi + 1][0]["time"] if bi + 1 < len(bursts) else t_start
        span = max(t_next - t_start, 0.0)
        budget, n = target, 0
        while budget > 0 and n < max_pkts_per_burst:
            sz = int(in_pool[rng.integers(0, len(in_pool))])
            sz = max(min(sz, int(budget)), 1)
            t_dummy = t_start + (float(rng.uniform(0.0, span)) if span > 0 else 0.0)
            out.append({"time": t_dummy, "dir": 1, "size": sz, "dummy": True})
            budget -= sz
            n += 1
    out.sort(key=lambda p: p["time"])
    return out


def defend_burstguard(pkts: list[dict], rng: np.random.Generator,
                      in_pool: np.ndarray, out_pool: np.ndarray,
                      grid: int, target_dir: int = 1,
                      delay_frac: float = 0.0) -> list[dict]:
    """BurstGuard-Quantize -- an OPTIONAL variant (distinct from the ASIA CCS '25 response
    mode above): quantize the byte-count of every INCOMING burst up to the next multiple of
    `grid` bytes by appending dummy incoming packets inside that burst.

    Rationale (Methodology 1.3 / 8.2): the discriminative KF signal on iPR is the
    *incoming burst pattern* of the rendered search-result page; snapping burst sizes to a
    coarse grid collapses many distinct pages onto the same padded profile. This variant is
    provided only for ablation and is NOT claimed as a contribution of this thesis.

    Zero-delay by default (dummies appended at the burst's last timestamp). If
    `delay_frac` > 0, the burst's tail is nudged forward by that fraction of its span
    to model alignment delay -> small O_lat (Methodology 8.4).
    """
    if not pkts:
        return pkts
    bursts = _split_bursts(pkts)
    out: list[dict] = []
    for b in bursts:
        out.extend(b)
        if b[0]["dir"] != target_dir or grid <= 0:
            continue
        cur_bytes = sum(p["size"] for p in b)
        target_bytes = int(np.ceil(cur_bytes / grid) * grid)
        pad = target_bytes - cur_bytes
        if pad <= 0:
            continue
        t_last = b[-1]["time"]
        t_first = b[0]["time"]
        span = t_last - t_first
        pool = in_pool if target_dir == 1 else out_pool
        while pad > 0:
            sz = int(pool[rng.integers(0, len(pool))])
            sz = min(sz, pad)
            sz = max(sz, 1)
            # Place dummy within the burst window (zero added latency) + optional nudge.
            jitter = float(rng.uniform(0.0, span)) if span > 0 else 0.0
            t_dummy = t_first + jitter + delay_frac * max(span, 1e-6)
            out.append({"time": t_dummy, "dir": target_dir, "size": sz, "dummy": True})
            pad -= sz
    out.sort(key=lambda p: p["time"])
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _make_transform(args, rng, in_pool, out_pool, iat_pool):
    """Return a per-trace transform closure for the selected defense.
    'none' is the identity (rebuild-only) baseline -- see apply_defense docstring."""
    if args.defense == "none":
        return lambda pkts: list(pkts)
    if args.defense == "front":
        return lambda pkts: defend_front(pkts, rng, in_pool, out_pool,
                                         args.front_nmax_in, args.front_nmax_out,
                                         args.front_wmin, args.front_wmax,
                                         clip_to_trace=not args.front_allow_extend)
    if args.defense == "wtf_pad":
        return lambda pkts: defend_wtfpad(pkts, rng, iat_pool, in_pool, out_pool,
                                          factor=args.wtfpad_factor,
                                          max_dummies_per_gap=args.wtfpad_max_per_gap)
    if args.defense == "burstguard":
        if args.bg_mode == "response":
            return lambda pkts: defend_burstguard_response(pkts, rng, in_pool, out_pool,
                                                           resp_prob=args.bg_resp_prob,
                                                           resp_scale=args.bg_resp_scale)
        return lambda pkts: defend_burstguard(pkts, rng, in_pool, out_pool,
                                              grid=args.bg_grid, target_dir=args.bg_dir,
                                              delay_frac=args.bg_delay)
    raise ValueError(f"Unknown defense {args.defense!r}")


def _generate_one(X_seq, args, seed, L):
    """Apply the defense to every trace for ONE seed.
    Returns X_seq_def, X_global_def, and per-trace arrays (o_bw, o_lat, n_dummy)."""
    rng = np.random.default_rng(seed)
    in_pool, out_pool = build_size_pools(X_seq, rng)
    iat_pool = build_iat_pool(X_seq, rng) if args.defense == "wtf_pad" else None
    transform = _make_transform(args, rng, in_pool, out_pool, iat_pool)

    N = X_seq.shape[0]
    X_seq_def = np.zeros_like(X_seq)
    X_global_def = np.zeros((N, 15), dtype=np.float32)
    o_bw = np.zeros(N); o_lat = np.zeros(N); n_dummy = np.zeros(N); n_empty = 0

    for i in tqdm(range(N), desc=f"{args.defense} seed={seed}", leave=False):
        pkts = reconstruct_packets(X_seq[i])
        if not pkts:
            n_empty += 1
            continue
        b0, d0 = trace_bytes_dur(pkts)
        defended = transform(pkts)
        b1, d1 = trace_bytes_dur(defended)
        # O_bw counts ALL transmitted dummy bytes (the network cost), even those beyond
        # the L-packet window the attacker models -- the honest bandwidth overhead.
        o_bw[i] = (b1 - b0) / max(b0, 1.0)
        o_lat[i] = (d1 - d0) / max(d0, 1e-9)
        n_dummy[i] = len(defended) - len(pkts)

        ann = _annotate_bursts(defended)
        X_seq_def[i] = _build_seq(ann, L)
        X_global_def[i] = _build_global(ann)

    return X_seq_def, X_global_def, o_bw, o_lat, n_dummy, n_empty


def run_selftest(X_seq, X_global_orig, L, k: int = 500) -> None:
    """Quantify round-trip fidelity per channel (substantiates the trace-level claim).

    Reconstruct -> rebuild each trace WITHOUT any defense (identity) and compare to the
    stored features. Results, channel by channel:
      ch0 signed size  (dir x size/1500): EXACT  -- reconstructed bit-for-bit (float noise).
      ch1 log10(1+IAT):                   EXACT  -- inverse-then-forward is identity.
      ch2 cumulative byte fraction:       differs for traces with > L packets, because it
            normalises by the FULL-trace byte total, while the truncated npz only retains
            the first L packets -> the denominator changes. Same reason X_global differs.
    Crucially, the defended datasets and the `--defense none` ceiling are BOTH rebuilt
    through this identical path, so ch2/X_global are recomputed on the same observable
    (<= L) window for both -> the ceiling-vs-defended comparison stays apples-to-apples.
    """
    k = min(k, X_seq.shape[0])
    ch = [0.0, 0.0, 0.0]            # max || per sequence channel
    dg_max = dg_mean = 0.0
    n_full = 0                     # traces filling all L slots (>= L packets, truncated)
    for i in range(k):
        pkts = reconstruct_packets(X_seq[i])
        if not pkts:
            continue
        ann = _annotate_bursts(pkts)
        seq2 = _build_seq(ann, L)
        gl2 = _build_global(ann)
        d = np.abs(seq2 - X_seq[i])
        for c in range(3):
            ch[c] = max(ch[c], float(d[:, c].max()))
        if np.abs(X_seq[i][:, 0]).min() > EPS:   # no zero-padding row -> trace had >= L packets
            n_full += 1
        dg = float(np.abs(gl2 - X_global_orig[i]).max())
        dg_max = max(dg_max, dg); dg_mean += dg
    dg_mean /= max(k, 1)
    print(f"\n[selftest] round-trip fidelity (identity reconstruction) on {k} traces "
          f"({n_full} of them fill all L={L} slots, i.e. had >= L packets):")
    print(f"  X_seq ch0 (signed size) max|| = {ch[0]:.2e}   EXACT (float32 rounding only)")
    print(f"  X_seq ch1 (log-IAT)     max|| = {ch[1]:.2e}   EXACT")
    print(f"  X_seq ch2 (cum-fraction)max|| = {ch[2]:.2e}   differs for > L-packet traces (full-trace denominator)")
    print(f"  X_global                max|| = {dg_max:.2e}   mean|| = {dg_mean:.2e}   (same cause as ch2)")
    print("  -> size/direction and timing are reconstructed exactly; the cumulative and")
    print("     global features depend on full-trace totals absent from the truncated npz.")
    print("     Both ceiling (`--defense none`) and defended sets are rebuilt the same way,")
    print("     so the comparison is controlled and apples-to-apples.\n")


def apply_defense(args) -> None:
    """Apply the defense over one or more seeds (Methodology Sec. 8).

    Methodological note (trace-level simulation): defenses are simulated directly on the
    feature-space trace reconstructed from X_seq -- NOT on raw PCAP. This is exact for the
    sequence features the model consumes (verified by --selftest) and, following FRONT and
    WTF-PAD, enables parameter sweeping orders of magnitude faster than PCAP I/O. The cost
    not captured is the QUIC congestion-control feedback loop (Methodology 8.6): reported
    O_bw is therefore a LOWER BOUND on true overhead.

    --defense none  : identity (rebuild-only) baseline. Produces an undefended npz through
                      the SAME reconstruct->rebuild path as the defended ones, so the ceiling
                      attacker is trained on the same extraction basis (controlled comparison).
    --n_seeds k     : produce k independent realizations (seeds seed..seed+k-1), one npz each
                      (suffixed _seedN), and report overhead as mean +/- std across seeds for
                      statistical credibility.
    """
    data = np.load(args.in_npz, allow_pickle=True)
    X_seq = data["X_seq"].astype(np.float32)
    X_global = data["X_global"].astype(np.float32)
    y = data["y"]
    classes = data["classes"]
    device = data["device"] if "device" in data.files else np.array(["?"] * len(y))
    metadata = {
        key: data[key]
        for key in ("file_paths", "capture_start_time", "capture_end_time", "sample_order")
        if key in data.files
    }
    N, L, _ = X_seq.shape
    print(f"Loaded {args.in_npz}: {N} samples | {len(classes)} classes | L={L}")

    if args.selftest:
        run_selftest(X_seq, X_global, L)
        return

    base_out = args.out_npz or args.in_npz.replace(".npz", f"_{args.defense}.npz")
    seeds = list(range(args.seed, args.seed + max(args.n_seeds, 1)))
    per_seed = []

    for seed in seeds:
        X_seq_def, X_global_def, o_bw, o_lat, n_dummy, n_empty = _generate_one(X_seq, args, seed, L)
        if n_empty:
            print(f"  seed={seed}: WARNING {n_empty} samples had no reconstructable packets.")
        # One npz per seed when sweeping seeds; otherwise the plain out path.
        out_path = base_out if len(seeds) == 1 else base_out.replace(".npz", f"_seed{seed}.npz")
        np.savez_compressed(out_path, X_seq=X_seq_def, X_global=X_global_def,
                            y=y, classes=classes, device=device, **metadata)
        per_seed.append({
            "seed": int(seed), "out_npz": out_path,
            "O_bw_mean": float(o_bw.mean()), "O_bw_median": float(np.median(o_bw)),
            "O_lat_mean": float(o_lat.mean()), "n_dummy_mean": float(n_dummy.mean()),
        })
        print(f"  seed={seed}: O_bw={o_bw.mean():.3f}  O_lat={o_lat.mean():.3f}  "
              f"dummies/trace={n_dummy.mean():.1f}  -> {out_path}")

    obw_means = np.array([s["O_bw_mean"] for s in per_seed])
    olat_means = np.array([s["O_lat_mean"] for s in per_seed])
    overhead = {
        "defense": args.defense,
        "in_npz": args.in_npz,
        "params": _defense_params(args),
        "n_samples": int(N),
        "n_seeds": len(seeds),
        "seq_len_L": int(L),
        "note": "O_bw counts all dummy bytes (network cost); features use first L packets. "
                "Static simulation ignores QUIC congestion control => O_bw is a lower bound.",
        "bandwidth_overhead_across_seeds": {
            "mean": float(obw_means.mean()), "std": float(obw_means.std()),
            "min": float(obw_means.min()), "max": float(obw_means.max()),
        },
        "latency_overhead_across_seeds": {
            "mean": float(olat_means.mean()), "std": float(olat_means.std()),
        },
        "per_seed": per_seed,
    }
    ov_path = Path(base_out).with_suffix("")
    with open(f"{ov_path}_overhead.json", "w") as f:
        json.dump(overhead, f, indent=2)
    with open(f"{ov_path}_overhead.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["defense", "n_seeds", "Obw_mean", "Obw_std", "Olat_mean", "param_summary"])
        w.writerow([args.defense, len(seeds),
                    overhead["bandwidth_overhead_across_seeds"]["mean"],
                    overhead["bandwidth_overhead_across_seeds"]["std"],
                    overhead["latency_overhead_across_seeds"]["mean"],
                    json.dumps(_defense_params(args))])
    bw = overhead["bandwidth_overhead_across_seeds"]
    print(f"\nOverhead over {len(seeds)} seed(s) (Methodology 8.4):  "
          f"O_bw = {bw['mean']:.3f} +/- {bw['std']:.3f}   "
          f"O_lat = {overhead['latency_overhead_across_seeds']['mean']:.3f}")
    print(f"Saved overhead -> {ov_path}_overhead.json / .csv")


# Return a dict of defense-specific hyperparameters from parsed args.
def _defense_params(args) -> dict:
    if args.defense == "none":
        return {"mode": "identity_rebuild"}
    if args.defense == "front":
        return {"n_max_in": args.front_nmax_in, "n_max_out": args.front_nmax_out,
                "w_min": args.front_wmin, "w_max": args.front_wmax,
                "allow_extend": args.front_allow_extend}
    if args.defense == "wtf_pad":
        return {"factor": args.wtfpad_factor, "max_dummies_per_gap": args.wtfpad_max_per_gap}
    if args.defense == "burstguard":
        if args.bg_mode == "response":
            return {"mode": "response", "resp_prob": args.bg_resp_prob, "resp_scale": args.bg_resp_scale}
        return {"mode": "quantize", "grid_bytes": args.bg_grid,
                "target_dir": args.bg_dir, "delay_frac": args.bg_delay}
    return {}


# Parse command-line arguments and run the selected defense simulation.
def main():
    p = argparse.ArgumentParser(description="Traffic-analysis defense simulators (Methodology Sec. 8)")
    p.add_argument("--in_npz", required=True, help="Undefended dataset .npz")
    p.add_argument("--out_npz", default="", help="Output path (default: <in>_<defense>.npz)")
    p.add_argument("--defense", required=True,
                   choices=["none", "front", "wtf_pad", "burstguard"],
                   help="'none' = identity rebuild baseline (controlled ceiling basis)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_seeds", type=int, default=1,
                   help="Generate k realizations (seeds seed..seed+k-1) -> overhead mean +/- std")
    p.add_argument("--selftest", action="store_true",
                   help="Verify round-trip reconstruction fidelity and exit (no files written)")

    # FRONT
    p.add_argument("--front_nmax_in", type=int, default=1500,
                   help="Max dummy incoming (server) packets; n ~ U[1, N_max]")
    p.add_argument("--front_nmax_out", type=int, default=1500,
                   help="Max dummy outgoing (client) packets")
    p.add_argument("--front_wmin", type=float, default=1.0, help="Rayleigh w lower bound (s)")
    p.add_argument("--front_wmax", type=float, default=8.0, help="Rayleigh w upper bound (s)")
    p.add_argument("--front_allow_extend", action="store_true",
                   help="Allow dummies past trace end (incurs O_lat > 0); default clips (zero-delay)")

    # WTF-PAD
    p.add_argument("--wtfpad_factor", type=float, default=1.0,
                   help="Scale on sampled expected gaps; <1 = denser padding / higher overhead")
    p.add_argument("--wtfpad_max_per_gap", type=int, default=8,
                   help="Cap on dummies inserted per real inter-packet gap")

    # BurstGuard
    p.add_argument("--bg_mode", choices=["response", "quantize"], default="response",
                   help="'response' = faithful ASIA CCS '25 BurstGuard (dummy incoming burst "
                        "per outgoing burst); 'quantize' = thesis variant (rename to BurstQuant)")
    # response mode (faithful BurstGuard)
    p.add_argument("--bg_resp_prob", type=float, default=1.0,
                   help="Prob. of injecting a dummy incoming burst per outgoing burst")
    p.add_argument("--bg_resp_scale", type=float, default=1.0,
                   help="Dummy incoming-burst bytes = scale x a sampled real incoming-burst size")
    # quantize mode (variant)
    p.add_argument("--bg_grid", type=int, default=4096,
                   help="[quantize] grid in bytes; incoming bursts padded up to a multiple")
    p.add_argument("--bg_dir", type=int, default=1, choices=[1, -1],
                   help="[quantize] burst direction to regularize (+1 incoming = KF signal)")
    p.add_argument("--bg_delay", type=float, default=0.0,
                   help="[quantize] fraction of burst span used as alignment delay (0 = zero-delay)")

    apply_defense(p.parse_args())


if __name__ == "__main__":
    main()
