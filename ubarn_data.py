"""PASTIS data handling, driven by the official split manifest.

Patch IDs, folds and acquisition dates all come from
`pastis_official_splits_v1.json`, so there is no second source of truth and no
dependency on downloading `metadata.geojson`. Raw arrays are read from a local
copy of PASTIS if one is present (Kaggle input, mounted drive) and otherwise
fetched patch-by-patch from the Hugging Face mirror.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from pastis_splits import Manifest, normalize_id

HF_REPO = "IGNF/PASTIS-HD"
N_BANDS = 10
N_CLASSES = 20  # 0 = background, 1..18 = crops, 19 = void
VOID_CLASS = 19

CLASS_NAMES = [
    "Background", "Meadow", "Soft winter wheat", "Corn", "Winter barley",
    "Winter rapeseed", "Spring barley", "Sunflower", "Grapevine", "Beet",
    "Winter triticale", "Winter durum wheat", "Fruits/vegetables/flowers",
    "Potatoes", "Leguminous fodder", "Soybeans", "Orchard", "Mixed cereal",
    "Sorghum", "Void",
]


# --------------------------------------------------------------------------

@dataclass
class DataConfig:
    t_max: int = 40                 # dates kept per series (paper uses up to 100)
    crop: int = 64                  # spatial crop (paper uses 64x64)
    date_start: str = "2019-01-01"  # paper restricts PASTIS to Jan-Nov 2019
    date_end: str = "2019-11-30"
    seed: int = 0

    def tag(self) -> str:
        return f"t{self.t_max}_c{self.crop}_{self.date_start[:4]}"


# --------------------------------------------------------------------------
# locating the raw arrays
# --------------------------------------------------------------------------

LOCAL_HINTS = ["/kaggle/input", "/content/drive/MyDrive", "~/pastis/raw", "./PASTIS"]


def find_local_pastis(extra: str | None = None) -> Path | None:
    """Look for a directory containing both DATA_S2/ and ANNOTATIONS/."""
    cands: list[Path] = []
    if extra:
        cands.append(Path(extra).expanduser())
    for hint in LOCAL_HINTS:
        root = Path(hint).expanduser()
        if not root.exists():
            continue
        cands.append(root)
        try:
            level1 = [p for p in root.iterdir() if p.is_dir()]
            cands.extend(level1)
            for p in level1:
                try:
                    cands.extend(q for q in p.iterdir() if q.is_dir())
                except (PermissionError, OSError):
                    pass
        except (PermissionError, OSError):
            pass
    seen = set()
    for c in cands:
        if c in seen:
            continue
        seen.add(c)
        if (c / "DATA_S2").is_dir() and (c / "ANNOTATIONS").is_dir():
            return c
    return None


def _hf_prefix() -> str:
    from huggingface_hub import list_repo_files

    for f in list_repo_files(HF_REPO, repo_type="dataset"):
        if "DATA_S2/" in f:
            return f.split("DATA_S2/")[0]
    raise FileNotFoundError(f"DATA_S2 not found in {HF_REPO}")


def download_patches(ids, cache_dir: Path, workers: int = 8) -> Path:
    """Fetch only the S2 series and annotations for the given patch IDs."""
    from huggingface_hub import snapshot_download

    prefix = _hf_prefix()
    patterns = []
    for pid in ids:
        patterns.append(f"{prefix}DATA_S2/S2_{pid}.npy")
        patterns.append(f"{prefix}ANNOTATIONS/TARGET_{pid}.npy")
    local = snapshot_download(
        HF_REPO,
        repo_type="dataset",
        allow_patterns=patterns,
        cache_dir=str(cache_dir / "hf"),
        max_workers=workers,
    )
    return Path(local) / prefix if prefix else Path(local)


def resolve_source(ids, cache_dir: Path, local_hint: str | None = None):
    """Return (raw_root, description) — prefer a local copy, else download."""
    local = find_local_pastis(local_hint)
    if local is not None:
        return local, f"local copy at {local}"
    return download_patches(ids, cache_dir), "Hugging Face mirror"


# --------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------

def _doy(yyyymmdd: str) -> int:
    return datetime.strptime(str(yyyymmdd), "%Y%m%d").timetuple().tm_yday


def _select_dates(dates: list[str], cfg: DataConfig) -> list[int]:
    """Positions in the series to keep, after the date window and thinning."""
    lo = int(cfg.date_start.replace("-", ""))
    hi = int(cfg.date_end.replace("-", ""))
    keep = [i for i, d in enumerate(dates) if lo <= int(d) <= hi]
    if not keep:
        keep = list(range(len(dates)))
    if len(keep) > cfg.t_max:
        sel = np.linspace(0, len(keep) - 1, cfg.t_max).round().astype(int)
        keep = [keep[i] for i in sel]
    return keep


def compute_norm_stats(raw_root: Path, ids, cfg: DataConfig, n_sample: int = 40) -> dict:
    """Robust per-band stats: 5th pct, median, 95th pct (paper eq. 4a/4b).

    Pass TRAINING ids only — deriving these from val/test patches is a subtle
    form of leakage.
    """
    rng = np.random.default_rng(cfg.seed)
    ids = [normalize_id(i) for i in ids]
    pick = rng.choice(len(ids), size=min(n_sample, len(ids)), replace=False)
    buf = [[] for _ in range(N_BANDS)]
    for i in pick:
        arr = np.load(raw_root / "DATA_S2" / f"S2_{ids[i]}.npy")
        arr = arr[:: max(1, arr.shape[0] // 8)]
        for b in range(N_BANDS):
            v = arr[:, b].reshape(-1).astype(np.float32)
            buf[b].append(rng.choice(v, size=min(20000, v.size), replace=False))
    stats = {"q05": [], "median": [], "q95": []}
    for b in range(N_BANDS):
        v = np.concatenate(buf[b])
        stats["q05"].append(float(np.quantile(v, 0.05)))
        stats["median"].append(float(np.median(v)))
        stats["q95"].append(float(np.quantile(v, 0.95)))
    return stats


def normalize(arr: np.ndarray, stats: dict) -> np.ndarray:
    q05 = np.asarray(stats["q05"], dtype=np.float32)[None, :, None, None]
    q95 = np.asarray(stats["q95"], dtype=np.float32)[None, :, None, None]
    med = np.asarray(stats["median"], dtype=np.float32)[None, :, None, None]
    x = np.clip(arr.astype(np.float32), q05, q95)
    return (x - med) / np.maximum(q95 - q05, 1e-6)


def build_cache(raw_root: Path, manifest: Manifest, ids, cfg: DataConfig,
                out_dir: Path, norm_ids=None, progress=None) -> Path:
    """Preprocess the given IDs into one memmap + sidecar arrays.

    `norm_ids` should be the training IDs of the fold you intend to run; the
    normalisation statistics are derived from those only.
    """
    ids = [normalize_id(i) for i in ids]
    out_dir = Path(out_dir) / f"{cfg.tag()}_n{len(ids)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "manifest.json").exists():
        return out_dir

    stats = compute_norm_stats(raw_root, list(norm_ids or ids), cfg)
    n, t, c, s = len(ids), cfg.t_max, N_BANDS, cfg.crop

    x_mm = np.lib.format.open_memmap(
        out_dir / "s2.npy", mode="w+", dtype=np.float16, shape=(n, t, c, s, s)
    )
    doy = np.zeros((n, t), dtype=np.int16)
    valid = np.zeros((n, t), dtype=bool)
    target = np.zeros((n, s, s), dtype=np.uint8)
    folds = np.zeros(n, dtype=np.int8)

    off = (128 - s) // 2  # deterministic centre crop
    for i, pid in enumerate(ids):
        dates = manifest.patches[pid]["dates"]
        keep = _select_dates(dates, cfg)
        arr = np.load(raw_root / "DATA_S2" / f"S2_{pid}.npy")[keep]
        arr = normalize(arr[:, :, off:off + s, off:off + s], stats)
        k = min(len(keep), t)
        x_mm[i, :k] = arr[:k].astype(np.float16)
        doy[i, :k] = [_doy(dates[j]) for j in keep[:k]]
        valid[i, :k] = True

        tgt = np.load(raw_root / "ANNOTATIONS" / f"TARGET_{pid}.npy")
        target[i] = tgt[0, off:off + s, off:off + s].astype(np.uint8)
        folds[i] = manifest.fold_of(pid)
        if progress is not None:
            progress()

    x_mm.flush()
    np.save(out_dir / "doy.npy", doy)
    np.save(out_dir / "valid.npy", valid)
    np.save(out_dir / "target.npy", target)
    np.save(out_dir / "folds.npy", folds)
    with open(out_dir / "ids.json", "w") as fh:
        json.dump(ids, fh)
    with open(out_dir / "norm_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    with open(out_dir / "manifest.json", "w") as fh:
        json.dump(
            {"config": asdict(cfg), "n": n, "split_manifest_sha256": manifest.sha256},
            fh, indent=2,
        )
    return out_dir


# --------------------------------------------------------------------------
# cache handle
# --------------------------------------------------------------------------

class PastisCache:
    def __init__(self, cache_dir: str | Path):
        self.dir = Path(cache_dir)
        self.x = np.load(self.dir / "s2.npy", mmap_mode="r")
        self.doy = np.load(self.dir / "doy.npy")
        self.valid = np.load(self.dir / "valid.npy")
        self.target = np.load(self.dir / "target.npy")
        self.folds = np.load(self.dir / "folds.npy")
        with open(self.dir / "ids.json") as fh:
            self.ids = [str(i) for i in json.load(fh)]
        self.pos = {pid: i for i, pid in enumerate(self.ids)}
        with open(self.dir / "norm_stats.json") as fh:
            self.stats = json.load(fh)
        with open(self.dir / "manifest.json") as fh:
            self.meta = json.load(fh)

    def __len__(self):
        return self.x.shape[0]

    def indices_for(self, ids) -> np.ndarray:
        """Map patch IDs to cache rows, dropping any that are not cached."""
        return np.array(
            [self.pos[normalize_id(i)] for i in ids if normalize_id(i) in self.pos],
            dtype=np.int64,
        )

    def coverage(self, ids) -> tuple[int, int]:
        ids = [normalize_id(i) for i in ids]
        return sum(1 for i in ids if i in self.pos), len(ids)

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        q05 = np.asarray(self.stats["q05"], dtype=np.float32)
        q95 = np.asarray(self.stats["q95"], dtype=np.float32)
        med = np.asarray(self.stats["median"], dtype=np.float32)
        shape = [1] * x.ndim
        shape[-3] = len(q05)
        return x * (q95 - q05).reshape(shape) + med.reshape(shape)


def make_dataset(cache: PastisCache, indices, augment: bool = False):
    import torch
    from torch.utils.data import Dataset

    class _DS(Dataset):
        def __init__(self):
            self.idx = np.asarray(indices)

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, i):
            j = int(self.idx[i])
            x = np.asarray(cache.x[j], dtype=np.float32)
            y = np.asarray(cache.target[j], dtype=np.int64)
            if augment:
                if np.random.rand() < 0.5:
                    x, y = x[..., ::-1].copy(), y[..., ::-1].copy()
                if np.random.rand() < 0.5:
                    x, y = x[..., ::-1, :].copy(), y[..., ::-1, :].copy()
            return {
                "x": torch.from_numpy(x),
                "doy": torch.from_numpy(cache.doy[j].astype(np.float32)),
                "valid": torch.from_numpy(cache.valid[j].copy()),
                "y": torch.from_numpy(y),
            }

    return _DS()


def scarce_subset(cache: PastisCache, train_idx, n: int, seed: int = 0) -> np.ndarray:
    """Rare-class-weighted sampling of n training patches (paper Appendix B)."""
    train_idx = np.asarray(train_idx)
    if n >= len(train_idx):
        return train_idx
    counts = np.zeros(N_CLASSES, dtype=np.float64)
    per_patch = np.zeros((len(train_idx), N_CLASSES), dtype=np.float64)
    for r, j in enumerate(train_idx):
        c = np.bincount(cache.target[j].reshape(-1), minlength=N_CLASSES)
        per_patch[r] = c
        counts += c
    s = np.where(counts > 0, 1.0 / np.maximum(counts, 1), 0.0)
    s = s / s.sum()
    p = (per_patch * s[None, :]).sum(1) / np.maximum(per_patch.sum(1), 1)
    p = p / p.sum()
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(train_idx), size=n, replace=False, p=p)
    return train_idx[np.sort(pick)]
