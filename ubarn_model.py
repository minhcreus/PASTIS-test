"""U-BARN: Unet-BERT spAtio-temporal Representation eNcoder.

Reimplementation from Dumeur, Valero & Inglada, IEEE JSTARS 17 (2024) 4350-4367,
"Self-Supervised Spatio-Temporal Representation Learning of Satellite Image
Time Series". The authors' own code is on a CNRS GitLab that requires a Janus
account, so this is written from the paper's description and Appendix A.

Defaults follow the paper: d_model=64, d_hidden=128, 3 transformer layers,
4 heads, mask rate 0.6, permutation masking, single-linear decoder.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# spatio-spectral encoder (Unet, temporal attention removed from bottleneck)
# --------------------------------------------------------------------------

def _conv_block(cin: int, cout: int, groups: int = 4) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.GroupNorm(groups, cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.GroupNorm(groups, cout),
        nn.ReLU(inplace=True),
    )


class SpatioSpectralEncoder(nn.Module):
    """Encodes each date independently: (N, C, H, W) -> (N, d_model, H, W).

    This is the U-TAE Unet with the L-TAE stripped out of the bottleneck, which
    is exactly how the paper describes the SSE (Section III-A1).
    """

    def __init__(self, in_ch: int = 10, widths=(32, 64, 128, 128), d_model: int = 64):
        super().__init__()
        self.inc = _conv_block(in_ch, widths[0])
        self.downs = nn.ModuleList(
            nn.Sequential(nn.MaxPool2d(2), _conv_block(widths[i], widths[i + 1]))
            for i in range(len(widths) - 1)
        )
        rev = list(reversed(widths))
        self.upsamples = nn.ModuleList(
            nn.ConvTranspose2d(rev[i], rev[i + 1], 2, stride=2)
            for i in range(len(widths) - 1)
        )
        self.up_convs = nn.ModuleList(
            _conv_block(rev[i + 1] * 2, rev[i + 1]) for i in range(len(widths) - 1)
        )
        self.out = nn.Conv2d(widths[0], d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.inc(x)]
        for d in self.downs:
            skips.append(d(skips[-1]))
        h = skips.pop()
        for up, conv in zip(self.upsamples, self.up_convs):
            h = up(h)
            s = skips.pop()
            if h.shape[-2:] != s.shape[-2:]:
                h = F.interpolate(h, size=s.shape[-2:], mode="nearest")
            h = conv(torch.cat([h, s], dim=1))
        return self.out(h)


# --------------------------------------------------------------------------
# positional encoding on day-of-year (paper eq. 1, scaling constant 1000)
# --------------------------------------------------------------------------

def doy_encoding(doy: torch.Tensor, d_model: int) -> torch.Tensor:
    """doy: (B, T) float -> (B, T, d_model)."""
    device, dtype = doy.device, torch.float32
    i = torch.arange(d_model // 2, device=device, dtype=dtype)
    denom = torch.pow(torch.tensor(1000.0, device=device), 2 * i / d_model)
    ang = doy.to(dtype).unsqueeze(-1) / denom
    pe = torch.zeros(*doy.shape, d_model, device=device, dtype=dtype)
    pe[..., 0::2] = torch.sin(ang)
    pe[..., 1::2] = torch.cos(ang)
    return pe


# --------------------------------------------------------------------------
# backbone
# --------------------------------------------------------------------------

class UBARN(nn.Module):
    """Patch embedding (SSE + DOY PE) followed by a per-pixel temporal transformer.

    Input  (B, T, C, H, W)  ->  output (B, T, d_model, H, W), i.e. the temporal
    and spatial resolution of the input are preserved, which is the whole point
    of the architecture relative to U-TAE.
    """

    def __init__(
        self,
        in_ch: int = 10,
        d_model: int = 64,
        d_hidden: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        widths=(32, 64, 128, 128),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.sse = SpatioSpectralEncoder(in_ch, widths, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_hidden,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        # nested-tensor fast path is disabled: it warns on padded batches and
        # takes a different code path from the masked one, which is not worth
        # the speedup here
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )

    def embed(self, x: torch.Tensor, doy: torch.Tensor) -> torch.Tensor:
        """(B,T,C,H,W) -> (B,T,d,H,W) patch embeddings with positional encoding."""
        b, t, c, h, w = x.shape
        f = self.sse(x.reshape(b * t, c, h, w)).reshape(b, t, self.d_model, h, w)
        pe = doy_encoding(doy, self.d_model)  # (B,T,d)
        return f + pe[:, :, :, None, None]

    def temporal(self, f: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """(B,T,d,H,W) -> (B,T,d,H,W). `valid` is (B,T) bool; padded dates masked."""
        b, t, d, h, w = f.shape
        seq = f.permute(0, 3, 4, 1, 2).reshape(b * h * w, t, d)
        pad = (~valid)[:, None, None, :].expand(b, h, w, t).reshape(b * h * w, t)
        out = self.transformer(seq, src_key_padding_mask=pad)
        out = torch.nan_to_num(out)  # guard against all-padded rows
        return out.reshape(b, h, w, t, d).permute(0, 3, 4, 1, 2)

    def forward(self, x, doy, valid):
        return self.temporal(self.embed(x, doy), valid)


# --------------------------------------------------------------------------
# pretext task
# --------------------------------------------------------------------------

def permutation_mask(f: torch.Tensor, valid: torch.Tensor, rate: float,
                     generator: torch.Generator | None = None):
    """Corrupt a fraction of dates by permuting embedded values within the batch.

    The paper's key departure from BERT/SITS-Former: rather than substituting a
    constant or Gaussian noise, a masked embedded value is replaced by another
    real embedded value drawn from elsewhere in the batch (another date, another
    pixel, or another feature). This keeps the activation distribution intact and
    reduces the train/inference distribution shift.

    Returns (corrupted_features, date_mask) where date_mask is (B, T) bool.
    """
    b, t, d, h, w = f.shape
    device = f.device
    mask = torch.zeros(b, t, dtype=torch.bool, device=device)
    for i in range(b):
        idx = torch.nonzero(valid[i], as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        k = max(1, int(round(rate * idx.numel())))
        perm = torch.randperm(idx.numel(), device=device, generator=generator)
        mask[i, idx[perm[:k]]] = True

    n_masked = int(mask.sum().item())
    if n_masked == 0:
        return f, mask

    flat = f.reshape(-1)
    draw = torch.randint(
        0, flat.numel(), (n_masked * d * h * w,), device=device, generator=generator
    )
    out = f.clone()
    out[mask] = flat[draw].reshape(n_masked, d, h, w)
    return out, mask


class LinearDecoder(nn.Module):
    """Deliberately shallow: one linear layer on the feature dimension.

    A heavier decoder would let the reconstruction be solved without the encoder
    learning anything useful (paper Section III-B2).
    """

    def __init__(self, d_model: int = 64, out_ch: int = 10):
        super().__init__()
        self.proj = nn.Conv2d(d_model, out_ch, 1)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        b, t, d, h, w = f.shape
        y = self.proj(f.reshape(b * t, d, h, w))
        return y.reshape(b, t, -1, h, w)


def reconstruction_loss(pred, target, date_mask, pixel_valid=None):
    """MSE over masked dates only (paper eq. 3).

    pred/target: (B,T,C,H,W). date_mask: (B,T). pixel_valid: (B,T,H,W) or None.
    PASTIS ships no cloud masks, so pixel_valid is normally None here; pass one
    if you pretrain on the authors' unlabeled Zenodo set, which does have MAJA
    validity masks.
    """
    if date_mask.sum() == 0:
        return pred.sum() * 0.0
    p = pred[date_mask]
    t = target[date_mask]
    if pixel_valid is not None:
        v = pixel_valid[date_mask].unsqueeze(1).float()
        return ((p - t) ** 2 * v).sum() / v.sum().clamp(min=1.0) / p.shape[1]
    return F.mse_loss(p, t)


# --------------------------------------------------------------------------
# downstream head
# --------------------------------------------------------------------------

class ShallowClassifier(nn.Module):
    """Mean-query attention (TAE-style) collapsing time, then a 1x1 conv.

    Needed because U-BARN keeps the temporal axis, and series lengths vary, so a
    plain linear probe cannot be attached. Following the paper, the value
    projection is the identity (V = X).
    """

    def __init__(self, d_model: int = 64, n_classes: int = 20):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.out = nn.Conv2d(d_model, n_classes, 1)
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(self, f: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        b, t, d, h, w = f.shape
        seq = f.permute(0, 3, 4, 1, 2).reshape(b * h * w, t, d)
        pad = (~valid)[:, None, None, :].expand(b, h, w, t).reshape(b * h * w, t)

        q_all = self.q(seq).masked_fill(pad.unsqueeze(-1), 0.0)
        n_valid = (~pad).sum(1, keepdim=True).clamp(min=1).float()
        q = q_all.sum(1) / n_valid                      # master query (N, d)
        k = self.k(seq)                                 # (N, T, d)
        att = (k @ q.unsqueeze(-1)).squeeze(-1) * self.scale
        att = att.masked_fill(pad, float("-inf")).softmax(-1)
        att = torch.nan_to_num(att)
        ctx = (att.unsqueeze(-1) * seq).sum(1)          # (N, d)
        ctx = ctx.reshape(b, h, w, d).permute(0, 3, 1, 2)
        return self.out(ctx)


class SegmentationModel(nn.Module):
    """U-BARN encoder + shallow classifier, in any of the paper's three regimes."""

    def __init__(self, encoder: UBARN, n_classes: int = 20, freeze: bool = False):
        super().__init__()
        self.encoder = encoder
        self.head = ShallowClassifier(encoder.d_model, n_classes)
        self.freeze = freeze
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.encoder.eval()
        return self

    def forward(self, x, doy, valid):
        if self.freeze:
            with torch.no_grad():
                f = self.encoder(x, doy, valid)
        else:
            f = self.encoder(x, doy, valid)
        return self.head(f, valid)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

class ConfusionMeter:
    """Accumulates a confusion matrix and derives OA / Kappa / F1 / mIoU."""

    def __init__(self, n_classes: int, ignore_index: int | None = 19):
        self.n = n_classes
        self.ignore = ignore_index
        self.cm = torch.zeros(n_classes, n_classes, dtype=torch.long)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred = pred.flatten().cpu()
        target = target.flatten().cpu()
        keep = torch.ones_like(target, dtype=torch.bool)
        if self.ignore is not None:
            keep &= target != self.ignore
        pred, target = pred[keep], target[keep]
        idx = target * self.n + pred
        self.cm += torch.bincount(idx, minlength=self.n * self.n).reshape(self.n, self.n)

    def scores(self) -> dict:
        cm = self.cm.float()
        if self.ignore is not None:
            keep = [i for i in range(self.n) if i != self.ignore]
            cm = cm[keep][:, keep]
        total = cm.sum().clamp(min=1)
        tp = cm.diag()
        oa = (tp.sum() / total).item()

        row, col = cm.sum(1), cm.sum(0)
        pe = ((row * col).sum() / (total * total)).item()
        kappa = (oa - pe) / (1 - pe) if pe < 1 else 0.0

        present = row > 0
        prec = tp / col.clamp(min=1)
        rec = tp / row.clamp(min=1)
        f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-9)
        iou = tp / (row + col - tp).clamp(min=1)
        return {
            "OA": oa,
            "Kappa": kappa,
            "F1": f1[present].mean().item(),
            "mIoU": iou[present].mean().item(),
            "per_class_f1": f1.tolist(),
            "per_class_iou": iou.tolist(),
        }
