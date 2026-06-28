"""
baselines.py
============
PyTorch re-implementations of the two comparison baselines for the thesis
"Enhancing Keyword Fingerprinting on iCloud Private Relay using Deep Learning",
plus a small model-dispatcher so that EVERY scenario pipeline (closed-world,
open-world, concept-drift, defense) can drive all three methods identically.

The three methods compared:
  * resnet_bigru  - the thesis contribution (ResNet-10 + BiGRU + Attention),
                    defined in train_resnet_bigru.py (KeywordClassifier).
  * varcnn        - Var-CNN (Bhat et al., PETS 2019).  github.com/sanjit-bhat/Var-CNN
  * netclr        - NetCLR (Bahramali et al., ACM CCS 2023).  arxiv.org/abs/2309.10147

WHY A PYTORCH PORT OF VAR-CNN
-----------------------------
The reference Var-CNN is TensorFlow with 5000-length single-feature sequences and
a 24-dim metadata vector - incompatible with the v2 data contract used everywhere
else in this project (X_seq (N,500,3), X_global (N,15)) and unable to reuse the
PyTorch open-world / drift / defense machinery.  Re-implementing Var-CNN in PyTorch
on the SAME data contract is what makes a *fair, single-protocol* comparison
possible: identical split, identical preprocessing, identical OOD/adaptation/defense
code for all three methods.

FAITHFULNESS / DELIBERATE DEVIATIONS (document these in the methodology):
  * Var-CNN architecture is reproduced faithfully: two ResNet-18 (1-D, dilated
    convolutions) branches - a direction branch and an inter-arrival-time branch -
    plus a metadata MLP, exactly as in the paper.  The paper trains the two ResNets
    SEPARATELY and averages their softmax outputs ("ensemble").  Here both branches
    live in ONE network and are trained jointly (a combined model), so the whole
    thing exposes a single forward()/get_embedding() - required to reuse the shared
    OOD/adaptation/defense pipelines uniformly.  This is the standard combined-model
    rendition of Var-CNN; report it as such.
  * Var-CNN direction input  = sign(X_seq[:,:,0])           (signed-size channel).
    Var-CNN timing input     = X_seq[:,:,1]                 (log inter-arrival time).
    Var-CNN metadata input   = the selected+scaled X_global (the paper's metadata
                               branch); thesis uses the SAME 15 ANOVA features, so
                               the comparison is on the sequence model, not features.
  * NetCLR is reproduced faithfully (DFNet encoder + SimCLR/NT-Xent contrastive
    pre-training with NetAugment, then supervised fine-tuning).  Following the paper
    NetCLR uses ONLY the packet sequence (all 3 channels) and ignores X_global; its
    forward() still ACCEPTS x_global (and ignores it) so the interface is uniform.

UNIFORM MODEL INTERFACE (matches KeywordClassifier in train_resnet_bigru.py)
---------------------------------------------------------------------------
Every model here exposes:
  * forward(x_seq, x_global, return_attention=False) -> logits           (B, n_classes)
  * get_embedding(x_seq, x_global)                   -> penultimate emb   (B, D)
  * .encoder    : nn.Module holding ALL feature-extraction params (so the drift
                  pipeline's "freeze encoder" head-only strategy works generically)
  * .classifier : nn.Sequential(Linear(D,256), ReLU, Dropout, Linear(256,K)) so the
                  drift F_PROTO strategy can index classifier[0] / classifier[3].

x_seq is (B, L, C) channels-LAST (the project-wide convention); models permute
internally as needed.

DISPATCHER
----------
  build_model(model_type, n_classes, global_feat, seq_feat=3, ...) -> nn.Module
  get_builder(model_type) -> callable(n_classes, global_feat, seq_feat, gru_hidden,
                                      dropout_enc) -> nn.Module
  rebuild_from_ckpt(ckpt, device, eval_mode=True) -> nn.Module
The scenario pipelines call these (via a tiny backward-compatible patch) and read
ckpt["model_arch"]["model_type"]; when the key is absent or "resnet_bigru" they fall
straight back to KeywordClassifier, so existing checkpoints behave exactly as before.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# Model-type identifiers (stored in ckpt["model_arch"]["model_type"]).
RESNET_BIGRU = "resnet_bigru"
VARCNN       = "varcnn"
NETCLR       = "netclr"

# Fixed architecture constants so a model can be rebuilt from (n_classes,
# global_feat, seq_feat) alone - no per-run width search.  Keep these stable
# between training and reload, exactly like KeywordClassifier's defaults.
VARCNN_STAGES   = (64, 128, 256, 512)   # ResNet-18 stage widths (paper)
VARCNN_META_DIM = 32                    # metadata MLP output dim
NETCLR_FEAT_DIM = 512                   # DFNet encoder output dim (paper: 512)
HEAD_HIDDEN     = 256                   # penultimate width (matches KeywordClassifier)


# ===========================================================================
# Var-CNN (PyTorch, combined model on the v2 data contract)
# ===========================================================================

class _VCBasicBlock1D(nn.Module):
    """ResNet BasicBlock (1-D) with dilated convolutions (Var-CNN flavour).

    Two k=3 convolutions (dilations d1, d2); a 1x1 projection shortcut is used
    when stride or channel count changes (He et al., 2016).  No SE block -
    Var-CNN does not use squeeze-and-excitation.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 dilations: tuple[int, int] = (1, 2)):
        super().__init__()
        d1, d2 = dilations
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride,
                               padding=d1, dilation=d1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, stride=1,
                               padding=d2, dilation=d2, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.act   = nn.ELU(inplace=True)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + self.shortcut(x))


class _ResNet18Branch1D(nn.Module):
    """One Var-CNN ResNet-18 branch: (B, 1, L) -> (B, 512).

    Stem Conv(k7,->64) + 4 stages of 2 BasicBlocks each (widths 64/128/256/512).
    The first block of stages 2-4 uses stride-2 (3 down-samplings) then global
    average pooling, mirroring the paper's dilated-causal ResNet-18.
    """

    def __init__(self, in_ch: int = 1, stages: tuple[int, ...] = VARCNN_STAGES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, stages[0], kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(stages[0]),
            nn.ELU(inplace=True),
        )
        blocks = []
        prev = stages[0]
        for i, width in enumerate(stages):
            stride = 1 if i == 0 else 2          # down-sample between stages
            blocks.append(_VCBasicBlock1D(prev, width, stride=stride, dilations=(1, 2)))
            blocks.append(_VCBasicBlock1D(width, width, stride=1, dilations=(1, 2)))
            prev = width
        self.blocks = nn.Sequential(*blocks)
        self.gap    = nn.AdaptiveAvgPool1d(1)
        self.out_dim = stages[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, 1, L)
        x = self.stem(x)
        x = self.blocks(x)
        return self.gap(x).squeeze(-1)                     # (B, 512)


class _VarCNNEncoder(nn.Module):
    """All Var-CNN feature extraction: dir branch + time branch + metadata MLP.

    Bundled into a single module named `encoder` on the parent so the drift
    pipeline can freeze the whole backbone with one `getattr(model,'encoder')`.
    Produces the (B, 512+512+32) penultimate embedding.
    """

    def __init__(self, global_feat: int, dropout: float = 0.3):
        super().__init__()
        self.dir_branch  = _ResNet18Branch1D(in_ch=1)
        self.time_branch = _ResNet18Branch1D(in_ch=1)
        self.meta_mlp    = nn.Sequential(
            nn.Linear(global_feat, 64), nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, VARCNN_META_DIM), nn.ELU(inplace=True),
        )
        self.out_dim = self.dir_branch.out_dim + self.time_branch.out_dim + VARCNN_META_DIM

    def forward(self, x_seq: torch.Tensor, x_global: torch.Tensor) -> torch.Tensor:
        # x_seq: (B, L, C) channels-last. Var-CNN inputs:
        #   direction = sign(signed-size channel 0); timing = log-IAT channel 1.
        direction = torch.sign(x_seq[:, :, 0]).unsqueeze(1)   # (B, 1, L)
        timing    = x_seq[:, :, 1].unsqueeze(1)               # (B, 1, L)
        h_dir  = self.dir_branch(direction)
        h_time = self.time_branch(timing)
        h_meta = self.meta_mlp(x_global)
        return torch.cat([h_dir, h_time, h_meta], dim=1)


class VarCNN(nn.Module):
    """Var-CNN (Bhat et al., PETS 2019) - combined PyTorch model, v2 data contract.

    gru_hidden is accepted and ignored (interface compatibility with the
    dispatcher / drift pipeline, which always passes it).
    """

    def __init__(self, n_classes: int, global_feat: int, seq_feat: int = 3,
                 gru_hidden=None, dropout_enc: float = 0.30,
                 dropout_fuse: float = 0.50, **_):
        super().__init__()
        self.encoder = _VarCNNEncoder(global_feat, dropout=dropout_enc)
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.out_dim, HEAD_HIDDEN), nn.ReLU(inplace=True),
            nn.Dropout(dropout_fuse),
            nn.Linear(HEAD_HIDDEN, n_classes),
        )

    def get_embedding(self, x_seq: torch.Tensor, x_global: torch.Tensor) -> torch.Tensor:
        return self.encoder(x_seq, x_global)

    def forward(self, x_seq, x_global, return_attention: bool = False):
        logits = self.classifier(self.encoder(x_seq, x_global))
        if return_attention:                 # Var-CNN has no temporal attention
            return logits, None
        return logits


# ===========================================================================
# NetCLR (PyTorch) - DFNet encoder + contrastive pre-training + classifier
# ===========================================================================

class DFNet(nn.Module):
    """Deep Fingerprinting Network backbone (Sirinam et al. 2018), as used by NetCLR.

    4 conv blocks (in_ch -> 32 -> 64 -> 128 -> 256), each:
      Conv1d(k=8) -> BN -> ELU -> Conv1d(k=8) -> BN -> ELU -> MaxPool(8,4) -> Dropout
    then AdaptiveAvgPool(20) -> flatten -> FC(256*20, feat_dim) -> BN -> ELU.
    Input (B, C, L) channels-first; output (B, feat_dim).
    """

    def __init__(self, in_ch: int = 3, feat_dim: int = NETCLR_FEAT_DIM):
        super().__init__()
        self.feat_dim = feat_dim

        def _block(ci: int, co: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(ci, co, kernel_size=8, padding=4, bias=False),
                nn.BatchNorm1d(co), nn.ELU(inplace=True),
                nn.Conv1d(co, co, kernel_size=8, padding=4, bias=False),
                nn.BatchNorm1d(co), nn.ELU(inplace=True),
                nn.MaxPool1d(kernel_size=8, stride=4),
                nn.Dropout(p=0.1),
            )

        self.block1 = _block(in_ch, 32)
        self.block2 = _block(32, 64)
        self.block3 = _block(64, 128)
        self.block4 = _block(128, 256)
        self.gap    = nn.AdaptiveAvgPool1d(20)
        self.fc     = nn.Linear(256 * 20, feat_dim)
        self.fc_bn  = nn.BatchNorm1d(feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, C, L)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x).flatten(1)
        return F.elu(self.fc_bn(self.fc(x)))


class NetCLRClassifier(nn.Module):
    """NetCLR fine-tuning model: DFNet encoder + classifier head (uniform interface).

    Following the paper, NetCLR uses only the packet sequence (all channels) and
    ignores X_global; forward() still accepts x_global for interface uniformity.
    gru_hidden accepted+ignored.
    """

    def __init__(self, n_classes: int, global_feat: int = 0, seq_feat: int = 3,
                 gru_hidden=None, dropout_enc: float = 0.30,
                 feat_dim: int = NETCLR_FEAT_DIM, **_):
        super().__init__()
        self.encoder = DFNet(in_ch=seq_feat, feat_dim=feat_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, HEAD_HIDDEN), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(HEAD_HIDDEN, n_classes),
        )

    def get_embedding(self, x_seq: torch.Tensor, x_global: torch.Tensor = None) -> torch.Tensor:
        x = x_seq.permute(0, 2, 1)            # (B, L, C) -> (B, C, L)
        return self.encoder(x)

    def forward(self, x_seq, x_global=None, return_attention: bool = False):
        logits = self.classifier(self.get_embedding(x_seq, x_global))
        if return_attention:
            return logits, None
        return logits


class DFNetCLR(nn.Module):
    """DFNet + 2-layer MLP projection head for SimCLR/NetCLR contrastive pre-training.

    Projection head (paper Appendix): feat_dim -> BN -> ReLU -> proj_dim.
    Returns L2-normalised projections for NT-Xent.
    """

    def __init__(self, in_ch: int = 3, feat_dim: int = NETCLR_FEAT_DIM, proj_dim: int = 128):
        super().__init__()
        self.encoder   = DFNet(in_ch=in_ch, feat_dim=feat_dim)
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True), nn.Linear(feat_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projector(self.encoder(x)), dim=1)


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross-Entropy (SimCLR / NetCLR loss, Eq. 1)."""

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.T = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        N   = z1.size(0)
        z   = torch.cat([z1, z2], dim=0)                     # (2N, D)
        sim = (z @ z.T) / self.T
        sim.masked_fill_(torch.eye(2 * N, device=z.device, dtype=torch.bool), float("-inf"))
        pos = torch.cat([torch.arange(N, device=z.device) + N,
                         torch.arange(N, device=z.device)], dim=0)
        loss = F.cross_entropy(sim, pos)
        with torch.no_grad():
            contrastive_acc = float((sim.argmax(1) == pos).float().mean())
        return loss, contrastive_acc


class NetAugment:
    """NetAugment (Bahramali et al., CCS 2023, Section 4) - adapted for 3-channel
    keyword traces.  Operates on channel 0 (signed size); a "burst" is a run of
    consecutive same-sign steps.  Applied in paper order: ChangeContent ->
    MergeIncoming -> AddOutgoing -> Shift.
    """

    def __init__(self, merge_prob=0.10, merge_max=5, add_burst_prob=0.30,
                 upsample_rate=1.0, downsample_rate=0.5, shift_param=10,
                 rng: np.random.Generator | None = None):
        self.merge_prob, self.merge_max = merge_prob, merge_max
        self.add_burst_prob = add_burst_prob
        self.upsample_rate, self.downsample_rate = upsample_rate, downsample_rate
        self.shift_param = shift_param
        self.rng = rng or np.random.default_rng(42)

    # Segment the channel-0 signal into runs of same-sign packets.
    def _segments(self, ch0: np.ndarray):
        if len(ch0) == 0:
            return []
        d = np.sign(ch0); d[d == 0] = 1
        segs, start, cur = [], 0, d[0]
        for i in range(1, len(d)):
            if d[i] != cur:
                segs.append((start, i, int(cur)))
                start, cur = i, d[i]
        segs.append((start, len(d), int(cur)))
        return segs

    # Slightly perturb incoming packet sizes to simulate content changes.
    def change_content(self, x):
        x = x.copy(); L = len(x)
        rate = self.upsample_rate if L < 500 else self.downsample_rate
        for s, e, d in self._segments(x[:, 0]):
            if d < 0:
                std = np.abs(x[s:e, 0]).mean() * rate * 0.3 + 1e-8
                x[s:e, 0] += self.rng.normal(0.0, std, e - s)
        return x

    # Randomly merge consecutive incoming bursts into a single burst.
    def merge_incoming(self, x):
        x = x.copy()
        inc = [(s, e) for s, e, d in self._segments(x[:, 0]) if d < 0]
        i = 0
        while i < len(inc) - 1:
            if self.rng.random() < self.merge_prob:
                s1 = inc[i][0]; j = i + 1
                while j < min(i + self.merge_max, len(inc) - 1) and self.rng.random() < self.merge_prob:
                    j += 1
                en = inc[j][1]
                wm = x[s1:en, 0].mean()
                x[s1:en, 0] = wm + self.rng.normal(0.0, abs(wm) * 0.1 + 1e-8, en - s1)
                i = j + 1
            else:
                i += 1
        return x

    # Insert a synthetic outgoing burst at a random incoming-burst boundary.
    def add_outgoing(self, x):
        if self.rng.random() > self.add_burst_prob:
            return x
        x = x.copy(); L = len(x)
        inc_starts = [s for s, e, d in self._segments(x[:, 0]) if d < 0 and s > 0]
        pos = int(self.rng.choice(inc_starts)) if inc_starts else int(self.rng.integers(1, max(L - 1, 2)))
        bs  = min(int(self.rng.lognormal(3.0, 1.0)), 50)
        end = min(pos + bs, L)
        if end <= pos:
            return x
        region  = x[pos:end, 0]
        ref_mag = (float(np.abs(region).mean()) if region.size else 1.0) + 1.0
        x[pos:end, 0] = self.rng.uniform(0.5, 1.5, end - pos) * ref_mag
        return x

    # Circularly shift the trace and zero-pad the exposed edge.
    def shift(self, x):
        delta = int(self.rng.integers(-self.shift_param, self.shift_param + 1))
        if delta == 0:
            return x
        x = np.roll(x, delta, axis=0)
        if delta > 0:
            x[:delta] = 0.0
        else:
            x[delta:] = 0.0
        return x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = self.change_content(x)
        x = self.merge_incoming(x)
        x = self.add_outgoing(x)
        x = self.shift(x)
        return x.astype(np.float32)


class ContrastiveDataset(Dataset):
    """Two independently NetAugmented views per trace; no labels (for pre-training)."""

    def __init__(self, X: np.ndarray, augment: NetAugment):
        self.X = X                # (N, L, C)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x  = self.X[idx]
        v1 = torch.from_numpy(self.augment(x)).permute(1, 0)   # (C, L)
        v2 = torch.from_numpy(self.augment(x)).permute(1, 0)
        return v1, v2


# ===========================================================================
# Dispatcher
# ===========================================================================

# Lazily import KeywordClassifier to avoid a hard dependency at module load time.
def _resnet_bigru_class():
    """Import KeywordClassifier lazily so baselines.py has no hard dependency on
    train_resnet_bigru at import time (e.g. when only the baselines are used)."""
    import sys
    if str(Path(__file__).parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).parent))
    from train_resnet_bigru import KeywordClassifier
    return KeywordClassifier


def get_builder(model_type: str):
    """Return a callable(n_classes, global_feat, seq_feat, gru_hidden, dropout_enc,
    **kw) -> nn.Module for the given model type.

    For 'resnet_bigru' (or None) this is the thesis KeywordClassifier itself, so
    every scenario pipeline keeps its original behaviour unchanged.
    """
    mt = (model_type or RESNET_BIGRU)
    if mt == RESNET_BIGRU:
        return _resnet_bigru_class()
    if mt == VARCNN:
        return VarCNN
    if mt == NETCLR:
        return NetCLRClassifier
    raise ValueError(f"Unknown model_type {model_type!r}; "
                     f"expected one of {RESNET_BIGRU}, {VARCNN}, {NETCLR}.")


def build_model(model_type: str, n_classes: int, global_feat: int,
                seq_feat: int = 3, gru_hidden=None, dropout_enc: float = 0.30,
                **kw) -> nn.Module:
    """Construct a fresh (untrained) model of the requested type."""
    builder = get_builder(model_type)
    return builder(n_classes=n_classes, global_feat=global_feat, seq_feat=seq_feat,
                   gru_hidden=gru_hidden, dropout_enc=dropout_enc, **kw)


def rebuild_from_ckpt(ckpt: dict, device, eval_mode: bool = True) -> nn.Module:
    """Rebuild a model from a checkpoint dict written by train_resnet_bigru.py or
    train_baselines.py.  Dispatches on ckpt['model_arch']['model_type'] (defaults
    to 'resnet_bigru' for legacy checkpoints).

    Handles checkpoints saved from a torch.compile-wrapped model: in some PyTorch
    versions model.state_dict() on an OptimizedModule returns keys prefixed with
    '_orig_mod.' - these are stripped automatically before loading.
    """
    arch  = ckpt.get("model_arch") or {}
    mtype = arch.get("model_type", RESNET_BIGRU)
    model = build_model(
        mtype,
        n_classes=int(arch["n_classes"]),
        global_feat=int(arch["global_feat"]),
        seq_feat=int(arch.get("seq_feat", 3)),
        gru_hidden=int(arch.get("gru_hidden", 128)),
        dropout_enc=float(arch.get("dropout_enc", 0.30)),
    ).to(device)
    state = ckpt["model_state"]
    for prefix in ("_orig_mod.", "module.", "model."):
        if any(str(k).startswith(prefix) for k in state):
            state = {
                str(k)[len(prefix):] if str(k).startswith(prefix) else k: v
                for k, v in state.items()
            }
    model.load_state_dict(state)
    if eval_mode:
        model.eval()
    return model


# Build the model_arch metadata dict for checkpoint serialization.
def make_model_arch(model_type: str, n_classes: int, global_feat: int,
                    seq_feat: int, dropout_enc: float = 0.30) -> dict:
    """Build the model_arch dict to embed in a checkpoint.  Always includes
    gru_hidden/dropout_enc keys (gru_hidden is a no-op for baselines) because the
    drift pipeline reads arch['gru_hidden'] unconditionally."""
    return {
        "model_type":  model_type,
        "n_classes":   int(n_classes),
        "global_feat": int(global_feat),
        "seq_feat":    int(seq_feat),
        "gru_hidden":  128,            # placeholder; ignored by baselines
        "dropout_enc": float(dropout_enc),
    }


if __name__ == "__main__":
    # Tiny self-test: forward + embedding shapes for both baselines.
    torch.manual_seed(0)
    B, L, C, G, K = 4, 500, 3, 15, 10
    xs = torch.randn(B, L, C)
    xg = torch.randn(B, G)
    for mt in (VARCNN, NETCLR):
        m = build_model(mt, n_classes=K, global_feat=G, seq_feat=C)
        logits = m(xs, xg)
        emb    = m.get_embedding(xs, xg)
        n_par  = sum(p.numel() for p in m.parameters())
        assert logits.shape == (B, K), (mt, logits.shape)
        assert m.classifier[0].in_features == emb.shape[1]
        assert m.classifier[3].out_features == K
        print(f"{mt:8s} OK  logits={tuple(logits.shape)} emb={tuple(emb.shape)} "
              f"params={n_par:,}")
    print("baselines self-test PASS")
