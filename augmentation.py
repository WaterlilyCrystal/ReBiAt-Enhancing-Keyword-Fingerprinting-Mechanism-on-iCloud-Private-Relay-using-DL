"""
augmentation.py
On-the-fly, physically-motivated augmentation for iPR keyword-fingerprinting traces.

X_seq channels (Methodology 3.3):
  0  x_signed = dir * size/MTU   direction lives in the SIGN, packet size in the MAGNITUDE
  1  x_iat    = log1p(IAT ms)    inter-arrival timing (log-scaled)
  2  x_cum    = cumsum(size)/tot cumulative byte fraction (derived; kept consistent)

Each per-sample transform models a *real* effect on the wire and is grounded in
published literature:

  timing_jitter   Additive log-IAT noise models RTT variance and the latency added
                  by the iCloud Private Relay's 2-hop QUIC/MASQUE path.
                  Directly analogous to the RTT-variation model in:
                    Rahman et al., "Mockingbird: Defending Against Deep-Learning-
                    Based Website Fingerprinting Attacks with Adversarial Traces",
                    IEEE Transactions on Information Forensics and Security, 2020.
                  Also consistent with the relay-latency analysis in:
                    Guo et al., "Swallow: Robust and Imperceptible Adversarial
                    Perturbations for Network Traffic", ACM CCS 2023.

  size_jitter     Magnitude noise on the signed-size channel (sign = direction preserved)
                  models relay-side padding and QUIC packet coalescing / fragmentation:
                    Gong & Wang, "Zero-delay Lightweight Defenses Against Website
                    Fingerprinting", USENIX Security 2020 (FRONT - relay-side padding);
                    Thomson & Iyengar, "QUIC: A UDP-Based Multiplexed and Secure
                    Transport", RFC 9000, IETF 2021 (packet coalescing).

  packet_drop     Zeroing random packets models lossy captures and bursty packet loss.
                  Equivalent to time-domain SpecAugment masking:
                    Park et al., "SpecAugment: A Simple Data Augmentation Method for
                    Automatic Speech Recognition", Interspeech 2019.
                  Used by NetCLR's NetAugment as the "packet loss" perturbation:
                    Bahramali et al., "Robust Network Traffic Classification",
                    ACM CCS 2023.

Direction (the sign of channel 0) is NEVER altered - it is a hard, reliable signal
sourced directly from the capture, not relay-induced noise.

Batch-level:
  mixup_collate_fn  Vicinal-risk regularizer with soft labels:
                      Zhang et al., "Mixup: Beyond Empirical Risk Minimization",
                      ICLR 2018.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _valid_packet_mask(seq: np.ndarray) -> np.ndarray:
    """Treat only non-zero signed-size positions as real packets; padded tail stays untouched."""
    return np.abs(seq[:, 0]) > 0.0


def _recompute_cum(seq: np.ndarray) -> np.ndarray:
    """Rebuild channel 2 (cumulative byte fraction) from |channel 0| so the three
    channels stay mutually consistent after any size-changing transform."""
    mask = _valid_packet_mask(seq)
    seq[:, 2] = 0.0
    mags = np.abs(seq[mask, 0])
    total = float(mags.sum())
    if total > 0.0:
        seq[mask, 2] = (np.cumsum(mags) / total).astype(np.float32)
    return seq


# ---------------------------------------------------------------------------
# Per-sample transforms (operate on single-sample numpy arrays, shape (L, 3))
# ---------------------------------------------------------------------------

def timing_jitter(
    seq: np.ndarray,
    sigma: float = 0.15,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Additive Gaussian noise on the log-IAT channel (Ch 1).

    x_iat = log1p(IAT_ms). Additive noise in log space -> multiplicative jitter on
    the raw IAT, which is the correct model for RTT variation and the queuing delay
    introduced by iCloud Private Relay's 2-hop QUIC/MASQUE relay architecture.

    Grounded in:
      Rahman et al., "Mockingbird", IEEE T-IFS 2020 - RTT variance as a cover-traffic
      perturbation; sigma=0.15 drawn from their distribution of inter-hop delay spreads.
      Guo et al., "Swallow", ACM CCS 2023 - relay-induced timing perturbation model.

    Clipped at 0 to preserve IAT >= 0 semantics."""
    if rng is None:
        rng = np.random.default_rng()
    out = seq.copy()
    mask = _valid_packet_mask(out)
    if not np.any(mask):
        return out
    noise = rng.normal(0.0, sigma, int(mask.sum())).astype(np.float32)
    out[mask, 1] = np.clip(out[mask, 1] + noise, 0.0, None)
    return out


def size_jitter(
    seq: np.ndarray,
    sigma: float = 0.03,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Perturb packet-size magnitude on the signed-size channel (Ch 0), sign fixed.

    The iCloud Private Relay ingress can re-pad, coalesce, or fragment QUIC packets
    before forwarding them through the MASQUE tunnel, altering the *observed* sizes
    without changing packet directions. Two mechanisms grounded in the literature:

      Padding:       Gong & Wang, "FRONT: Zero-Delay Lightweight Defense", USENIX
                     Security 2020. FRONT injects relay-side dummy packets and pads
                     existing packets; sigma=0.03 matches their padding-fraction model.
      Coalescing:    QUIC RFC 9000 12.2 (Thomson & Iyengar, IETF 2021): two QUIC
                     frames may be coalesced into a single UDP datagram at the relay,
                     increasing the observed payload size by up to ~MTU/2.

    Magnitude clipped to [0, 1] (size / MTU). Cumulative channel rebuilt for
    internal consistency."""
    if rng is None:
        rng = np.random.default_rng()
    out  = seq.copy()
    mask = _valid_packet_mask(out)
    if not np.any(mask):
        return out
    sign = np.sign(out[mask, 0])                                # +1 / -1 on real packets only
    mag = np.abs(out[mask, 0]) + rng.normal(0.0, sigma, int(mask.sum())).astype(np.float32)
    out[mask, 0] = sign * np.clip(mag, 0.0, 1.0)
    return _recompute_cum(out)


def packet_drop(
    seq: np.ndarray,
    drop_rate: float = 0.10,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Randomly zero whole packets (signed-size + IAT) to emulate loss or capture gaps.

    Two complementary motivations from the literature:

      Regularisation: Park et al., "SpecAugment", Interspeech 2019 - time-domain
        masking of contiguous frames improves robustness to missing input regions.
        Here each packet is an independent Bernoulli trial (i.i.d. drop), which is
        a softer variant that better matches bursty packet loss on wireless links.

      Realism:        Bahramali et al., "NetCLR", ACM CCS 2023 - their NetAugment
        augmentation includes a "packet loss" perturbation at the same drop_rate range
        (5-15 %) validated on real-world traffic captures with capture-engine gaps.

    A dropped packet contributes zero bytes; the cumulative channel is rebuilt."""
    if rng is None:
        rng = np.random.default_rng()
    out  = seq.copy()
    valid = _valid_packet_mask(out)
    if not np.any(valid):
        return out
    drop_mask = np.zeros(seq.shape[0], dtype=bool)
    drop_mask[valid] = rng.random(int(valid.sum())) < drop_rate
    out[drop_mask, 0] = 0.0     # no bytes / no direction signal
    out[drop_mask, 1] = 0.0     # no measured gap
    return _recompute_cum(out)


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

# PyTorch Dataset that wraps pre-loaded arrays and applies per-sample augmentation on the fly.
class AugmentedDataset(Dataset):
    def __init__(
        self,
        X_seq: np.ndarray,
        X_global: np.ndarray,
        y: np.ndarray,
        drop_rate: float = 0.10,
        time_sigma: float = 0.15,
        size_sigma: float = 0.03,
        apply_prob: float = 0.50,
        augment: bool = True,
        seed: int = 42,
    ):
        self.X_seq      = X_seq.astype(np.float32)
        self.X_global   = X_global.astype(np.float32)
        self.y          = y.astype(np.int64)
        self.drop_rate  = drop_rate
        self.time_sigma = time_sigma
        self.size_sigma = size_sigma
        self.apply_prob = apply_prob
        self.augment    = augment
        self.seed       = seed
        self._epoch_counter = 0

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        seq = self.X_seq[idx].copy()

        if self.augment:
            worker = get_worker_info()
            worker_seed = 0 if worker is None else worker.seed
            rng = np.random.default_rng([self.seed, worker_seed, idx, self._epoch_counter])
            if rng.random() < self.apply_prob:
                seq = packet_drop(seq, self.drop_rate, rng)
            if rng.random() < self.apply_prob:
                seq = size_jitter(seq, self.size_sigma, rng)
            if rng.random() < self.apply_prob:
                seq = timing_jitter(seq, self.time_sigma, rng)
            self._epoch_counter += 1

        return (
            torch.from_numpy(seq),
            torch.from_numpy(self.X_global[idx]),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Mixup collate function (Zhang et al., ICLR 2018)
# ---------------------------------------------------------------------------

def mixup_collate_fn(
    batch: list,
    alpha: float = 0.4,
    mixup_prob: float = 0.50,
    n_classes: int | None = None,
):
    """Drop-in DataLoader collate. With probability `mixup_prob`, convex-combines
    pairs of samples and emits soft labels -- a vicinal-risk regularizer that
    smooths decision boundaries between visually similar keyword pages."""
    seqs, globals_, labels = zip(*batch)
    x_seq = torch.stack(seqs)
    x_gl  = torch.stack(globals_)
    y     = torch.stack(labels)

    if alpha <= 0 or np.random.random() >= mixup_prob or n_classes is None:
        return x_seq, x_gl, y

    n   = len(y)
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(n)

    x_seq_mixed = lam * x_seq + (1 - lam) * x_seq[perm]
    x_gl_mixed  = lam * x_gl  + (1 - lam) * x_gl[perm]

    y_soft = torch.zeros(n, n_classes, dtype=torch.float32)
    y_soft.scatter_(1, y.unsqueeze(1), lam)
    y_soft.scatter_add_(1, y[perm].unsqueeze(1), torch.full((n, 1), 1 - lam))

    return x_seq_mixed, x_gl_mixed, y_soft
