import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # U-BARN on PASTIS

    Masked pretraining of a Unet + transformer on Sentinel-2 time series, evaluated
    on PASTIS crop segmentation. Dumeur, Valero & Inglada, JSTARS 17 (2024).

    Self-contained: no imports, no uploads. Run top to bottom.
    """)
    return


@app.cell
def _():
    import copy, json, math, os, sys, urllib.request
    from dataclasses import dataclass, asdict
    from datetime import datetime
    from pathlib import Path

    import numpy as np

    # --- dependencies ----------------------------------------------------------
    MISSING = [pip for mod, pip in [("torch", "torch"),
                                    ("matplotlib", "matplotlib"),
                                    ("huggingface_hub", "huggingface-hub")]
               if __import__("importlib.util", fromlist=["util"]).find_spec(mod) is None]
    if MISSING:
        print("installing:", " ".join(MISSING))
        os.system(f"{sys.executable} -m pip install -q " + " ".join(MISSING))

    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    # --- where to work ---------------------------------------------------------
    WORK = Path.home() / "ubarn"
    WORK.mkdir(parents=True, exist_ok=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 0 workers: cell-defined classes cannot be pickled for spawn. Raising this
    # passes a fork context instead (Linux only).
    import multiprocessing as _mp

    NUM_WORKERS = 0
    _LOADER_CTX = None
    if NUM_WORKERS > 0:
        try:
            _LOADER_CTX = _mp.get_context("fork")
        except ValueError:
            NUM_WORKERS = 0
            print("fork unavailable on this platform — falling back to 0 workers")


    def loader(dataset, batch_size, shuffle, drop_last=False):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"),
                          drop_last=drop_last,
                          multiprocessing_context=_LOADER_CTX if NUM_WORKERS else None)

    print("torch  :", torch.__version__)
    print("device :", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "(no GPU — attach one from the notebook specs menu)")
    print("work   :", WORK)
    print("workers:", NUM_WORKERS, "(0 = load in the main process)")
    return (
        DEVICE,
        F,
        Path,
        WORK,
        asdict,
        copy,
        dataclass,
        datetime,
        json,
        loader,
        math,
        nn,
        np,
        plt,
        torch,
        urllib,
    )


@app.cell
def _(Path, WORK, urllib):
    # --- the split manifest ----------------------------------------------------
    MANIFEST_URL = ("https://raw.githubusercontent.com/minhcreus/PASTIS-test/"
                    "main/pastis_official_splits_v1.json")
    MANIFEST_PATH = WORK / "pastis_official_splits_v1.json"

    def locate_manifest():
        for cand in [MANIFEST_PATH, Path.cwd() / MANIFEST_PATH.name,
                     Path.home() / MANIFEST_PATH.name]:
            if cand.exists() and cand.stat().st_size > 1_000_000:
                return cand
        urllib.request.urlretrieve(MANIFEST_URL, MANIFEST_PATH)
        return MANIFEST_PATH

    SPLIT_MANIFEST = locate_manifest()
    print(SPLIT_MANIFEST, f"({SPLIT_MANIFEST.stat().st_size:,} bytes)")
    assert SPLIT_MANIFEST.stat().st_size > 1_000_000, "download failed — got an error page?"
    return (SPLIT_MANIFEST,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Library code

    Split handling, data, model. Collapse and move on.
    """)
    return


@app.cell
def _(Path, dataclass, json):
    import hashlib
    SCHEMA = 'pastis_official_splits_v1'

    @dataclass
    class Manifest:
        patches: dict[str, dict]
        splits: dict[str, dict]
        labels: dict
    # --------------------------------------------------------------------------
        expected_hw: tuple[int, int]
        sha256: str

        @property  # id -> {"fold": int, "dates": [YYYYMMDD, ...]}
        def ids(self) -> list[str]:  # "1".."5" -> {"train": [...], "val": [...], "test": [...]}
            return list(self.patches.keys())

        def fold_of(self, pid: str) -> int:
            return self.patches[pid]['fold']

        def fold_sizes(self) -> dict[int, int]:
            out: dict[int, int] = {}
            for p in self.patches.values():
                out[p['fold']] = out.get(p['fold'], 0) + 1
            return dict(sorted(out.items()))

        def split(self, fold: int | str) -> dict[str, list[str]]:
            return self.splits[str(fold)]

    def load_manifest(path: str | Path) -> Manifest:
        path = Path(path)
        raw = path.read_bytes()
        doc = json.loads(raw)
        if doc.get('schema') != SCHEMA:
            raise ValueError(f'expected schema {SCHEMA!r}, got {doc.get('schema')!r}')
        patches = {str(p['id']): {'fold': int(p['fold']), 'dates': [str(d) for d in p['dates']]} for p in doc['patches']}
        return Manifest(patches=patches, splits={str(k): v for k, v in doc['splits'].items()}, labels=doc.get('labels', {'background': 0, 'crop_ids': list(range(1, 19)), 'void': 19}), expected_hw=tuple(doc.get('expected_hw', [128, 128])), sha256=hashlib.sha256(raw).hexdigest())

    def normalize_id(x) -> str:
        """'S2_10000' / 10000 / '10000' -> '10000'."""
        s = str(x)
        for pre in ('S2_', 'TARGET_', 'S1A_', 'S1D_'):
            if s.startswith(pre):
                s = s[len(pre):]
        return s.removesuffix('.npy')

    def validate_split(split: dict, manifest: Manifest | None=None, name: str='split') -> dict:
        """Duplicates, overlap, membership, coverage, fold purity."""
        sets = {}
        report = {'name': name, 'errors': [], 'warnings': [], 'counts': {}}
        for key in ('train', 'val', 'test'):
            raw = [normalize_id(i) for i in split.get(key, [])]
            uniq = set(raw)
            sets[key] = uniq
            report['counts'][key] = len(raw)
            if len(raw) != len(uniq):
    # normalisation of ID spellings
                report['errors'].append(f"{len(raw) - len(uniq)} duplicate id(s) inside '{key}'")
        report['total'] = sum(report['counts'].values())
        for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
            inter = sets[a] & sets[b]
            if inter:
                report['errors'].append(f"{len(inter)} id(s) shared between '{a}' and '{b}': {sorted(inter)[:5]}")
        if manifest is not None:
            known = set(manifest.ids)
            for key in ('train', 'val', 'test'):
                unknown = sets[key] - known
                if unknown:
                    report['errors'].append(f"{len(unknown)} id(s) in '{key}' do not exist in PASTIS: {sorted(unknown)[:5]}")
    # validation
            union = sets['train'] | sets['val'] | sets['test']
            missing = known - union
            if missing:
                report['warnings'].append(f'{len(missing)} PASTIS patch(es) appear in no split')
            report['covers_dataset'] = not missing and (not union - known)
            mix = {}
            for key in ('train', 'val', 'test'):
                folds = {manifest.fold_of(i) for i in sets[key] if i in known}
                mix[key] = sorted(folds)
            report['fold_mix'] = mix
            if any((len(v) > 1 for v in (mix['val'], mix['test']))):
                report['warnings'].append('val/test draw from several official folds — patches adjacent in the field can straddle the boundary, which inflates scores and makes results incomparable to the PASTIS leaderboard')
        report['ok'] = not report['errors']
        report['sets'] = sets
        return report

    def print_report(report: dict) -> None:
        """Integrity report."""
        c, total = (report['counts'], max(report['total'], 1))
        print('=' * 60)
        print(f'SPLIT: {report['name']}')
        print(f'TỔNG SỐ PATCH: {report['total']}')
        for key, label in (('train', 'Train'), ('val', 'Val  '), ('test', 'Test ')):
            print(f'- {label} : {c[key]:>6} files ({c[key] / total * 100:.2f}%)')
        if 'fold_mix' in report:
            print(f'- Fold mix: {report['fold_mix']}')
        print('-' * 60)
        for e in report['errors']:
            print(f'[LỖI] {e}')
        for w in report['warnings']:
            print(f'[CẢNH BÁO] {w}')
        if report['ok'] and (not report['warnings']):
            print('[XÁC NHẬN] Dữ liệu hoàn toàn sạch: không trùng lặp, không rò rỉ, mọi ID đều tồn tại trong PASTIS.')
        elif report['ok']:
            print('[XÁC NHẬN] Không có lỗi chặn — xem cảnh báo ở trên.')
        print('=' * 60)

    def folds_to_run(manifest: Manifest, folds=(1, 2)) -> dict[int, dict[str, list[str]]]:
        """Subset of the five official rotations."""
        out = {}
        for f in folds:
            sp = manifest.split(f)
            out[int(f)] = {k: [normalize_id(i) for i in sp[k]] for k in ('train', 'val', 'test')}
        return out

    def required_ids(runs: dict[int, dict[str, list[str]]]) -> set[str]:
        """Patches touched by the chosen folds."""
        out: set[str] = set()
        for sp in runs.values():
            for key in ('train', 'val', 'test'):
                out.update(sp[key])
    # selecting folds to run
        return out

    return Manifest, load_manifest, normalize_id, print_report, validate_split


@app.cell
def _(Manifest, Path, asdict, dataclass, datetime, json, normalize_id, np):
    HF_REPO = 'IGNF/PASTIS-HD'
    N_BANDS = 10
    N_CLASSES = 20
    VOID_CLASS = 19
    CLASS_NAMES = ['Background', 'Meadow', 'Soft winter wheat', 'Corn', 'Winter barley', 'Winter rapeseed', 'Spring barley', 'Sunflower', 'Grapevine', 'Beet', 'Winter triticale', 'Winter durum wheat', 'Fruits/vegetables/flowers', 'Potatoes', 'Leguminous fodder', 'Soybeans', 'Orchard', 'Mixed cereal', 'Sorghum', 'Void']

    @dataclass
    class DataConfig:
        t_max: int = 40
        crop: int = 64
        date_start: str = '2019-01-01'  # 0 = background, 1..18 = crops, 19 = void
        date_end: str = '2019-11-30'
        seed: int = 0

        def tag(self) -> str:
            return f't{self.t_max}_c{self.crop}_{self.date_start[:4]}'
    LOCAL_HINTS = ['~/pastis/raw', './PASTIS']

    def find_local_pastis(extra: str | None=None) -> Path | None:
        """Find a dir with DATA_S2/ and ANNOTATIONS/."""
        cands: list[Path] = []
        if extra:
    # --------------------------------------------------------------------------
            cands.append(Path(extra).expanduser())
        for hint in LOCAL_HINTS:
            root = Path(hint).expanduser()
            if not root.exists():  # dates kept per series (paper uses up to 100)
                continue  # spatial crop (paper uses 64x64)
            cands.append(root)  # paper restricts PASTIS to Jan-Nov 2019
            try:
                level1 = [p for p in root.iterdir() if p.is_dir()]
                cands.extend(level1)
                for p in level1:
                    try:
                        cands.extend((q for q in p.iterdir() if q.is_dir()))
                    except (PermissionError, OSError):
                        pass
    # locating the raw arrays
            except (PermissionError, OSError):
                pass
        seen = set()
        for c in cands:
            if c in seen:
                continue
            seen.add(c)
            if (c / 'DATA_S2').is_dir() and (c / 'ANNOTATIONS').is_dir():
                return c
        return None

    def _hf_prefix() -> str:
        from huggingface_hub import list_repo_files
        for f in list_repo_files(HF_REPO, repo_type='dataset'):
            if 'DATA_S2/' in f:
                return f.split('DATA_S2/')[0]
        raise FileNotFoundError(f'DATA_S2 not found in {HF_REPO}')

    def download_patches(ids, cache_dir: Path, workers: int=8) -> Path:
        """Fetch S2 series and annotations for the given IDs."""
        from huggingface_hub import snapshot_download
        prefix = _hf_prefix()
        patterns = []
        for pid in ids:
            patterns.append(f'{prefix}DATA_S2/S2_{pid}.npy')
            patterns.append(f'{prefix}ANNOTATIONS/TARGET_{pid}.npy')
        local = snapshot_download(HF_REPO, repo_type='dataset', allow_patterns=patterns, cache_dir=str(cache_dir / 'hf'), max_workers=workers)
        return Path(local) / prefix if prefix else Path(local)

    def resolve_source(ids, cache_dir: Path, local_hint: str | None=None):
        """(raw_root, description): local copy if present, else download."""
        local = find_local_pastis(local_hint)
        if local is not None:
            return (local, f'local copy at {local}')
        return (download_patches(ids, cache_dir), 'Hugging Face mirror')

    def _doy(yyyymmdd: str) -> int:
        return datetime.strptime(str(yyyymmdd), '%Y%m%d').timetuple().tm_yday

    def _select_dates(dates: list[str], cfg: DataConfig) -> list[int]:
        """Series positions to keep after windowing and thinning."""
        lo = int(cfg.date_start.replace('-', ''))
        hi = int(cfg.date_end.replace('-', ''))
        keep = [i for i, d in enumerate(dates) if lo <= int(d) <= hi]
        if not keep:
            keep = list(range(len(dates)))
        if len(keep) > cfg.t_max:
            sel = np.linspace(0, len(keep) - 1, cfg.t_max).round().astype(int)
            keep = [keep[i] for i in sel]
        return keep

    def compute_norm_stats(raw_root: Path, ids, cfg: DataConfig, n_sample: int=40) -> dict:
        """Robust per-band stats (paper eq. 4a/4b). Training ids only."""
        rng = np.random.default_rng(cfg.seed)
        ids = [normalize_id(i) for i in ids]
        pick = rng.choice(len(ids), size=min(n_sample, len(ids)), replace=False)
        buf = [[] for _ in range(N_BANDS)]
        for i in pick:
            arr = np.load(raw_root / 'DATA_S2' / f'S2_{ids[i]}.npy')
            arr = arr[::max(1, arr.shape[0] // 8)]
            for b in range(N_BANDS):
                v = arr[:, b].reshape(-1).astype(np.float32)
                buf[b].append(rng.choice(v, size=min(20000, v.size), replace=False))
        stats = {'q05': [], 'median': [], 'q95': []}
        for b in range(N_BANDS):
            v = np.concatenate(buf[b])
            stats['q05'].append(float(np.quantile(v, 0.05)))
            stats['median'].append(float(np.median(v)))
            stats['q95'].append(float(np.quantile(v, 0.95)))
        return stats

    def normalize(arr: np.ndarray, stats: dict) -> np.ndarray:
    # preprocessing
        q05 = np.asarray(stats['q05'], dtype=np.float32)[None, :, None, None]
        q95 = np.asarray(stats['q95'], dtype=np.float32)[None, :, None, None]
        med = np.asarray(stats['median'], dtype=np.float32)[None, :, None, None]
        x = np.clip(arr.astype(np.float32), q05, q95)
        return (x - med) / np.maximum(q95 - q05, 1e-06)

    def build_cache(raw_root: Path, manifest: Manifest, ids, cfg: DataConfig, out_dir: Path, norm_ids=None, progress=None) -> Path:
        """Preprocess IDs into one memmap + sidecar arrays."""
        ids = [normalize_id(i) for i in ids]
        out_dir = Path(out_dir) / f'{cfg.tag()}_n{len(ids)}'
        out_dir.mkdir(parents=True, exist_ok=True)
        if (out_dir / 'manifest.json').exists():
            return out_dir
        stats = compute_norm_stats(raw_root, list(norm_ids or ids), cfg)
        n, t, c, s = (len(ids), cfg.t_max, N_BANDS, cfg.crop)
        x_mm = np.lib.format.open_memmap(out_dir / 's2.npy', mode='w+', dtype=np.float16, shape=(n, t, c, s, s))
        doy = np.zeros((n, t), dtype=np.int16)
        valid = np.zeros((n, t), dtype=bool)
        target = np.zeros((n, s, s), dtype=np.uint8)
        folds = np.zeros(n, dtype=np.int8)
        off = (128 - s) // 2
        for i, pid in enumerate(ids):
            dates = manifest.patches[pid]['dates']
            keep = _select_dates(dates, cfg)
            arr = np.load(raw_root / 'DATA_S2' / f'S2_{pid}.npy')[keep]
            arr = normalize(arr[:, :, off:off + s, off:off + s], stats)
            k = min(len(keep), t)
            x_mm[i, :k] = arr[:k].astype(np.float16)
            doy[i, :k] = [_doy(dates[j]) for j in keep[:k]]
            valid[i, :k] = True
            tgt = np.load(raw_root / 'ANNOTATIONS' / f'TARGET_{pid}.npy')
            target[i] = tgt[0, off:off + s, off:off + s].astype(np.uint8)
            folds[i] = manifest.fold_of(pid)
            if progress is not None:
                progress()
        x_mm.flush()
        np.save(out_dir / 'doy.npy', doy)
        np.save(out_dir / 'valid.npy', valid)
        np.save(out_dir / 'target.npy', target)
        np.save(out_dir / 'folds.npy', folds)
        with open(out_dir / 'ids.json', 'w') as fh:
            json.dump(ids, fh)
        with open(out_dir / 'norm_stats.json', 'w') as fh:
            json.dump(stats, fh, indent=2)
        with open(out_dir / 'manifest.json', 'w') as fh:
            json.dump({'config': asdict(cfg), 'n': n, 'split_manifest_sha256': manifest.sha256}, fh, indent=2)
        return out_dir

    class PastisCache:

        def __init__(self, cache_dir: str | Path):
            self.dir = Path(cache_dir)
            self.x = np.load(self.dir / 's2.npy', mmap_mode='r')
            self.doy = np.load(self.dir / 'doy.npy')
            self.valid = np.load(self.dir / 'valid.npy')
            self.target = np.load(self.dir / 'target.npy')
            self.folds = np.load(self.dir / 'folds.npy')
            with open(self.dir / 'ids.json') as fh:
                self.ids = [str(i) for i in json.load(fh)]
            self.pos = {pid: i for i, pid in enumerate(self.ids)}
            with open(self.dir / 'norm_stats.json') as fh:
                self.stats = json.load(fh)
            with open(self.dir / 'manifest.json') as fh:
                self.meta = json.load(fh)

        def __len__(self):
            return self.x.shape[0]

        def indices_for(self, ids) -> np.ndarray:
            """Patch IDs -> cache rows, dropping uncached."""
            return np.array([self.pos[normalize_id(i)] for i in ids if normalize_id(i) in self.pos], dtype=np.int64)

        def coverage(self, ids) -> tuple[int, int]:
            ids = [normalize_id(i) for i in ids]
            return (sum((1 for i in ids if i in self.pos)), len(ids))

        def denormalize(self, x: np.ndarray) -> np.ndarray:
            q05 = np.asarray(self.stats['q05'], dtype=np.float32)
            q95 = np.asarray(self.stats['q95'], dtype=np.float32)
            med = np.asarray(self.stats['median'], dtype=np.float32)
            shape = [1] * x.ndim
            shape[-3] = len(q05)
            return x * (q95 - q05).reshape(shape) + med.reshape(shape)

    def make_dataset(cache: PastisCache, indices, augment: bool=False):
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
                        x, y = (x[..., ::-1], y[..., ::-1])
    # cache handle
                    if np.random.rand() < 0.5:
                        x, y = (x[..., ::-1, :], y[..., ::-1, :])
                    k = np.random.randint(4)
                    if k:
                        x, y = (np.rot90(x, k, axes=(-2, -1)), np.rot90(y, k, axes=(-2, -1)))
                    x, y = (np.ascontiguousarray(x), np.ascontiguousarray(y))
                return {'x': torch.from_numpy(x), 'doy': torch.from_numpy(cache.doy[j].astype(np.float32)), 'valid': torch.from_numpy(cache.valid[j].copy()), 'y': torch.from_numpy(y)}
        return _DS()

    def scarce_subset(cache: PastisCache, train_idx, n: int, seed: int=0) -> np.ndarray:
        """Rare-class-weighted sampling (paper Appendix B)."""
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

    return (
        CLASS_NAMES,
        DataConfig,
        N_BANDS,
        N_CLASSES,
        PastisCache,
        VOID_CLASS,
        build_cache,
        make_dataset,
        resolve_source,
        scarce_subset,
    )


@app.cell
def _(F, math, nn, torch):
    def _conv_block(cin: int, cout: int, groups: int=4) -> nn.Sequential:
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(groups, cout), nn.ReLU(inplace=True), nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(groups, cout), nn.ReLU(inplace=True))

    class SpatioSpectralEncoder(nn.Module):
        """Per-date encoder: (N,C,H,W) -> (N,d_model,H,W). U-TAE Unet without L-TAE."""

        def __init__(self, in_ch: int=10, widths=(32, 64, 128, 128), d_model: int=64):
    # --------------------------------------------------------------------------
    # spatio-spectral encoder (Unet, temporal attention removed from bottleneck)
            super().__init__()
            self.inc = _conv_block(in_ch, widths[0])
            self.downs = nn.ModuleList((nn.Sequential(nn.MaxPool2d(2), _conv_block(widths[i], widths[i + 1])) for i in range(len(widths) - 1)))
            rev = list(reversed(widths))
            self.upsamples = nn.ModuleList((nn.ConvTranspose2d(rev[i], rev[i + 1], 2, stride=2) for i in range(len(widths) - 1)))
            self.up_convs = nn.ModuleList((_conv_block(rev[i + 1] * 2, rev[i + 1]) for i in range(len(widths) - 1)))
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
                    h = F.interpolate(h, size=s.shape[-2:], mode='nearest')
                h = conv(torch.cat([h, s], dim=1))
            return self.out(h)

    def doy_encoding(doy: torch.Tensor, d_model: int) -> torch.Tensor:
        """(B,T) day-of-year -> (B,T,d_model)."""
        device, dtype = (doy.device, torch.float32)
        i = torch.arange(d_model // 2, device=device, dtype=dtype)
        denom = torch.pow(torch.tensor(1000.0, device=device), 2 * i / d_model)
        ang = doy.to(dtype).unsqueeze(-1) / denom
        pe = torch.zeros(*doy.shape, d_model, device=device, dtype=dtype)
        pe[..., 0::2] = torch.sin(ang)
        pe[..., 1::2] = torch.cos(ang)
        return pe

    class UBARN(nn.Module):
        """(B,T,C,H,W) -> (B,T,d_model,H,W). Temporal and spatial resolution preserved."""

        def __init__(self, in_ch: int=10, d_model: int=64, d_hidden: int=128, n_layers: int=3, n_heads: int=4, widths=(32, 64, 128, 128), dropout: float=0.1):
            super().__init__()
            self.d_model = d_model
            self.sse = SpatioSpectralEncoder(in_ch, widths, d_model)
            layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_hidden, dropout=dropout, activation='relu', batch_first=True, norm_first=False)
            self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)

        def embed(self, x: torch.Tensor, doy: torch.Tensor) -> torch.Tensor:
            """(B,T,C,H,W) -> (B,T,d,H,W) with positional encoding."""
            b, t, c, h, w = x.shape
            f = self.sse(x.reshape(b * t, c, h, w)).reshape(b, t, self.d_model, h, w)
            pe = doy_encoding(doy, self.d_model)
            return f + pe[:, :, :, None, None]
    # positional encoding on day-of-year (paper eq. 1, scaling constant 1000)

        def temporal(self, f: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
            """(B,T,d,H,W) -> same. `valid` (B,T) bool masks padded dates."""
            b, t, d, h, w = f.shape
            seq = f.permute(0, 3, 4, 1, 2).reshape(b * h * w, t, d)
            pad = (~valid)[:, None, None, :].expand(b, h, w, t).reshape(b * h * w, t)
            out = self.transformer(seq, src_key_padding_mask=pad)
            out = torch.nan_to_num(out)
            return out.reshape(b, h, w, t, d).permute(0, 3, 4, 1, 2)

        def forward(self, x, doy, valid):
            return self.temporal(self.embed(x, doy), valid)

    def permutation_mask(f: torch.Tensor, valid: torch.Tensor, rate: float, generator: torch.Generator | None=None):
        """Corrupt a fraction of dates by permuting embedded values within the batch.
    # backbone

        Returns (corrupted_features, date_mask (B,T) bool).
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
            return (f, mask)
        flat = f.reshape(-1)
        draw = torch.randint(0, flat.numel(), (n_masked * d * h * w,), device=device, generator=generator)
        out = f.clone()
        out[mask] = flat[draw].reshape(n_masked, d, h, w)
        return (out, mask)

    class LinearDecoder(nn.Module):
        """One linear layer on the feature dimension (paper III-B2)."""

        def __init__(self, d_model: int=64, out_ch: int=10):
            super().__init__()
            self.proj = nn.Conv2d(d_model, out_ch, 1)

        def forward(self, f: torch.Tensor) -> torch.Tensor:
            b, t, d, h, w = f.shape
            y = self.proj(f.reshape(b * t, d, h, w))
            return y.reshape(b, t, -1, h, w)

    def reconstruction_loss(pred, target, date_mask, pixel_valid=None):
        """MSE over masked dates only (paper eq. 3). pixel_valid optional."""  # (B,T,d)
        if date_mask.sum() == 0:
            return pred.sum() * 0.0
        p = pred[date_mask]
        t = target[date_mask]
        if pixel_valid is not None:
            v = pixel_valid[date_mask].unsqueeze(1).float()
            return ((p - t) ** 2 * v).sum() / v.sum().clamp(min=1.0) / p.shape[1]
        return F.mse_loss(p, t)
      # guard against all-padded rows
    class ShallowClassifier(nn.Module):
        """Mean-query attention collapsing time, then 1x1 conv. V = X, per the paper."""

        def __init__(self, d_model: int=64, n_classes: int=20, feat_norm: bool=True):
            super().__init__()
            self.norm = nn.LayerNorm(d_model) if feat_norm else nn.Identity()
            self.q = nn.Linear(d_model, d_model)
    # pretext task
            self.k = nn.Linear(d_model, d_model)
            self.out = nn.Conv2d(d_model, n_classes, 1)
            self.scale = 1.0 / math.sqrt(d_model)

        def forward(self, f: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
            b, t, d, h, w = f.shape
            seq = self.norm(f.permute(0, 3, 4, 1, 2).reshape(b * h * w, t, d))
            pad = (~valid)[:, None, None, :].expand(b, h, w, t).reshape(b * h * w, t)
            q_all = self.q(seq).masked_fill(pad.unsqueeze(-1), 0.0)
            n_valid = (~pad).sum(1, keepdim=True).clamp(min=1).float()
            q = q_all.sum(1) / n_valid
            k = self.k(seq)
            att = (k @ q.unsqueeze(-1)).squeeze(-1) * self.scale
            att = att.masked_fill(pad, float('-inf')).softmax(-1)
            att = torch.nan_to_num(att)
            ctx = (att.unsqueeze(-1) * seq).sum(1)
            ctx = ctx.reshape(b, h, w, d).permute(0, 3, 1, 2)
            return self.out(ctx)

    class SegmentationModel(nn.Module):
        """U-BARN encoder + shallow classifier."""

        def __init__(self, encoder: UBARN, n_classes: int=20, freeze: bool=False, feat_norm: bool=True):
            super().__init__()
            self.encoder = encoder
            self.head = ShallowClassifier(encoder.d_model, n_classes, feat_norm)
            self.freeze = freeze
            if freeze:
                for p in self.encoder.parameters():
                    p.requires_grad_(False)

        def train(self, mode: bool=True):
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

    class ConfusionMeter:
        """Confusion matrix -> OA / Kappa / F1 / mIoU."""

        def __init__(self, n_classes: int, ignore_index: int | None=19):
            self.n = n_classes
            self.ignore = ignore_index
            self.cm = torch.zeros(n_classes, n_classes, dtype=torch.long)

        @torch.no_grad()
        def update(self, pred: torch.Tensor, target: torch.Tensor):
            pred = pred.flatten().cpu()
            target = target.flatten().cpu()
            keep = torch.ones_like(target, dtype=torch.bool)
    # downstream head
            if self.ignore is not None:
                keep &= target != self.ignore
            pred, target = (pred[keep], target[keep])
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
            row, col = (cm.sum(1), cm.sum(0))
            pe = ((row * col).sum() / (total * total)).item()
            kappa = (oa - pe) / (1 - pe) if pe < 1 else 0.0
            present = row > 0
            prec = tp / col.clamp(min=1)
            rec = tp / row.clamp(min=1)
            f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-09)  # master query (N, d)
            iou = tp / (row + col - tp).clamp(min=1)  # (N, T, d)
            nan = torch.tensor(float('nan'))
            mf1 = f1[present].mean().item()
            return {'OA': oa, 'mIoU': iou[present].mean().item(), 'mF1': mf1, 'F1': mf1, 'Kappa': kappa, 'per_class_f1': torch.where(present, f1, nan).tolist(), 'per_class_iou': torch.where(present, iou, nan).tolist()}
      # (N, d)
    def spatiotemporal_mask(f, valid, t_rate, s_rate=0.0, block=8, generator=None):
        """Temporal masking (paper) plus optional spatial block masking on the remaining dates.

        Returns (corrupted_features, pixel_mask (B,T,H,W) bool).
        """
        b, t, d, h, w = f.shape
        device = f.device
        date_mask = torch.zeros(b, t, dtype=torch.bool, device=device)
        for i in range(b):
            idx = torch.nonzero(valid[i], as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            k = max(1, int(round(t_rate * idx.numel())))
            perm = torch.randperm(idx.numel(), device=device, generator=generator)
            date_mask[i, idx[perm[:k]]] = True
        px = date_mask[:, :, None, None].expand(b, t, h, w).clone()
        if s_rate > 0:
            gh, gw = (-(-h // block), -(-w // block))
            blocks = torch.rand(b, t, gh, gw, device=device, generator=generator) < s_rate
            blocks &= valid[:, :, None, None] & ~date_mask[:, :, None, None]
            blk = blocks.repeat_interleave(block, 2).repeat_interleave(block, 3)[:, :, :h, :w]
            px |= blk
        n = int(px.sum().item())
        if n == 0:
            return (f, px)
        flat = f.reshape(-1)
        draw = torch.randint(0, flat.numel(), (n, d), device=device, generator=generator)
        out = f.permute(0, 1, 3, 4, 2).clone()
        out[px] = flat[draw]
        return (out.permute(0, 1, 4, 2, 3).contiguous(), px)

    def masked_pixel_loss(pred, target, px):
        """MSE over masked pixels. Equals reconstruction_loss when px is a whole-date mask."""
    # metrics
        if px.sum() == 0:
            return pred.sum() * 0.0
        p = pred.permute(0, 1, 3, 4, 2)[px]
        q = target.permute(0, 1, 3, 4, 2)[px]
        return F.mse_loss(p, q)

    def dice_loss(logits, target, ignore_index, n_classes):
        """Soft multiclass Dice over classes present in the batch."""
        prob = logits.float().softmax(1)
        keep = target != ignore_index
        tgt = torch.where(keep, target, torch.zeros_like(target))
        oh = F.one_hot(tgt, n_classes).permute(0, 3, 1, 2).float() * keep[:, None]
        prob = prob * keep[:, None]
        inter = (prob * oh).sum((0, 2, 3))
        denom = prob.sum((0, 2, 3)) + oh.sum((0, 2, 3))
        present = oh.sum((0, 2, 3)) > 0
        if present.sum() == 0:
            return logits.sum() * 0.0
        return 1.0 - ((2 * inter + 1.0) / (denom + 1.0))[present].mean()

    return (
        ConfusionMeter,
        LinearDecoder,
        SegmentationModel,
        UBARN,
        dice_loss,
        masked_pixel_loss,
        permutation_mask,
        spatiotemporal_mask,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Splits

    `pastis_official_splits_v1.json` defines the partition. The validator checks
    duplicates, overlap, membership, and fold purity.
    """)
    return


@app.cell
def _(SPLIT_MANIFEST, load_manifest, print_report, validate_split):
    MAN = load_manifest(SPLIT_MANIFEST)
    print('sha256  :', MAN.sha256[:16], '...')
    print('patches :', len(MAN.ids))
    print('folds   :', MAN.fold_sizes())
    print()
    for _fold_key in sorted(MAN.splits, key=int):
        print_report(validate_split(MAN.split(_fold_key), MAN, name=f'official fold {_fold_key}'))
    return (MAN,)


@app.cell
def _(MAN, validate_split):
    VAL_FOLD = 4  # fixed early-stopping fold
    FOLDS = [1, 2]  # test folds; must not include VAL_FOLD
    assert VAL_FOLD not in FOLDS, 'the early-stopping fold cannot also be a test fold'

    def make_runs(manifest, test_folds, val_fold):
        out = {}
        for t in test_folds:
            out[t] = {'train': sorted((pid for pid, rec in manifest.patches.items() if rec['fold'] not in (t, val_fold))), 'val': sorted((pid for pid, rec in manifest.patches.items() if rec['fold'] == val_fold)), 'test': sorted((pid for pid, rec in manifest.patches.items() if rec['fold'] == t))}
        return out
    RUNS = make_runs(MAN, FOLDS, VAL_FOLD)
    NEEDED = sorted({i for sp in RUNS.values() for part in sp.values() for i in part})
    for _fold_key, _sp in RUNS.items():
        tr_folds = sorted({MAN.fold_of(i) for i in _sp['train']})
        print(f'test fold {_fold_key}: train={len(_sp['train']):5d} (folds {tr_folds})  val={len(_sp['val']):4d} (fold {VAL_FOLD})  test={len(_sp['test']):4d}')
        rep_custom = validate_split(_sp, MAN, name=f'test fold {_fold_key}')
        assert rep_custom['ok'], rep_custom['errors']
    print(f'\nearly stopping on fold {VAL_FOLD} throughout')
    print(f'unique patches needed: {len(NEEDED)}')
    return FOLDS, NEEDED, RUNS, VAL_FOLD


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## EDA — temporal structure

    From the manifest, before any download.
    """)
    return


@app.cell
def _(MAN, datetime, np, plt):
    eda_dates = {pid: [int(d) for d in rec['dates']] for pid, rec in MAN.patches.items()}
    eda_counts = np.array([len(v) for v in eda_dates.values()])

    def eda_doy(yyyymmdd):
        return datetime.strptime(str(yyyymmdd), '%Y%m%d').timetuple().tm_yday
    eda_all_doy = np.array([eda_doy(d) for v in eda_dates.values() for d in v])
    eda_in_window = np.array([sum((1 for d in v if 20190101 <= d <= 20191130)) for v in eda_dates.values()])
    eda_gaps = []
    for _v in eda_dates.values():
        ds = sorted((datetime.strptime(str(d), '%Y%m%d') for d in _v if 20190101 <= d <= 20191130))
        eda_gaps.extend(((ds[i + 1] - ds[i]).days for i in range(len(ds) - 1)))
    eda_gaps = np.array(eda_gaps)
    fig_eda1, ax_eda1 = plt.subplots(1, 3, figsize=(14, 3.4))
    ax_eda1[0].hist(eda_counts, bins=range(eda_counts.min(), eda_counts.max() + 2), color='#4c72b0', edgecolor='white')
    ax_eda1[0].hist(eda_in_window, bins=range(eda_counts.min(), eda_counts.max() + 2), color='#dd8452', edgecolor='white', alpha=0.85)
    ax_eda1[0].set_title('acquisitions per patch')
    ax_eda1[0].set_xlabel('dates')
    ax_eda1[0].legend(['full series', 'Jan–Nov 2019 window'], fontsize=7)
    ax_eda1[1].hist(eda_all_doy, bins=36, color='#55a868', edgecolor='white')
    ax_eda1[1].set_title('when acquisitions fall')
    ax_eda1[1].set_xlabel('day of year')
    ax_eda1[2].hist(eda_gaps, bins=range(0, min(eda_gaps.max(), 60) + 3, 2), color='#c44e52', edgecolor='white')
    ax_eda1[2].axvline(5, ls='--', c='k', lw=1)
    ax_eda1[2].set_title('gap between consecutive dates')
    ax_eda1[2].set_xlabel('days   (dashed = 5-day nominal revisit)')
    for _a in ax_eda1:
        _a.grid(alpha=0.25)
    fig_eda1.tight_layout()
    fig_eda1
    return eda_counts, eda_gaps, eda_in_window


@app.cell
def _(eda_counts, eda_gaps, eda_in_window, mo, np, ui_tmax):
    eda_lines = [
        f"acquisitions per patch : min {eda_counts.min()}  median {int(np.median(eda_counts))}  max {eda_counts.max()}",
        f"inside Jan-Nov 2019    : min {eda_in_window.min()}  median {int(np.median(eda_in_window))}  max {eda_in_window.max()}",
        f"revisit gap (days)     : median {int(np.median(eda_gaps))}  p90 {int(np.percentile(eda_gaps, 90))}  max {eda_gaps.max()}",
        "",
        f"your setting: dates per series = {ui_tmax.value}",
    ]
    if ui_tmax.value >= int(np.median(eda_in_window)):
        eda_lines.append("-> keeps essentially the whole in-window series; thinning is a no-op for most patches")
    else:
        eda_lines.append(f"-> thins the median patch from {int(np.median(eda_in_window))} to {ui_tmax.value} dates, evenly spaced")
    eda_nl = chr(10)
    mo.md("```" + eda_nl + eda_nl.join(eda_lines) + eda_nl + "```")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data

    `SUBSET` caps how many patches are fetched. Subsetting happens within each
    split, so proportions hold.
    """)
    return


@app.cell
def _(mo):
    ui_subset = mo.ui.slider(100, 2433, value=600, step=100, label="patches")
    ui_tmax = mo.ui.slider(8, 64, value=40, step=4, label="dates per series")
    ui_crop = mo.ui.dropdown(["16", "32", "64", "128"], value="64", label="crop")
    ui_window = mo.ui.dropdown(["Jan-Nov 2019 (paper)", "full series (Sep 2018-Nov 2019)"],
                               value="Jan-Nov 2019 (paper)", label="date window")
    mo.vstack([ui_subset, ui_tmax, ui_crop, ui_window])
    return ui_crop, ui_subset, ui_tmax, ui_window


@app.cell
def _(
    DataConfig,
    NEEDED,
    RUNS,
    mo,
    np,
    ui_crop,
    ui_subset,
    ui_tmax,
    ui_window,
):
    FULL_SERIES = ui_window.value.startswith('full')
    cfg = DataConfig(t_max=ui_tmax.value, crop=int(ui_crop.value), date_start='2018-09-01' if FULL_SERIES else '2019-01-01', date_end='2019-11-30')
    if ui_subset.value < len(NEEDED):
        rng_pick = np.random.default_rng(cfg.seed)
        keep = set()
        for _fold_key, _sp in RUNS.items():
            for part in ('train', 'val', 'test'):
                ids_part = _sp[part]
                k = max(1, round(ui_subset.value * len(ids_part) / len(NEEDED)))
                keep.update(rng_pick.choice(ids_part, size=min(k, len(ids_part)), replace=False).tolist())
        USE_IDS = sorted(keep)
    else:
        USE_IDS = NEEDED
    mem_note = ''
    if cfg.crop == 128:
        mem_note = ' · **128² is 4x the activation memory of 64²** — drop batch size to 1–2'
    mo.md(f'\n**{len(USE_IDS)} patches** · download ≈ {len(USE_IDS) * 14.1 / 1024:.1f} GB ·\ncache ≈ {len(USE_IDS) * cfg.t_max * 10 * cfg.crop ** 2 * 2 / 1000000000.0:.1f} GB{mem_note}\n')
    return USE_IDS, cfg


@app.cell
def _(mo):
    run_cache = mo.ui.run_button(label="① Download + build cache")
    run_cache
    return (run_cache,)


@app.cell
def _(
    FOLDS,
    MAN,
    PastisCache,
    RUNS,
    USE_IDS,
    WORK,
    build_cache,
    cfg,
    mo,
    resolve_source,
    run_cache,
):
    CACHE_TAG = WORK / 'cache' / f'{cfg.tag()}_n{len(USE_IDS)}'
    CACHE_READY = (CACHE_TAG / 'manifest.json').exists()
    mo.stop(not run_cache.value and (not CACHE_READY), mo.md('*Press ① to fetch and preprocess.*'))
    # button-or-on-disk: run_button resets to False, so gating on it alone would
    # wipe `cache` on any upstream change
    RAW_ROOT, source_note = (CACHE_TAG, 'cache already on disk') if CACHE_READY else resolve_source(USE_IDS, WORK)
    norm_ids = [i for i in RUNS[FOLDS[0]]['train'] if i in set(USE_IDS)]
    with mo.status.progress_bar(total=len(USE_IDS), title='preprocessing') as bar_cache:
        CACHE_DIR = build_cache(RAW_ROOT, MAN, USE_IDS, cfg, WORK / 'cache', norm_ids=norm_ids, progress=bar_cache.update)
    cache = PastisCache(CACHE_DIR)
    lines = [f'raw data: {source_note}', f'cache: {CACHE_DIR}', f'{len(cache)} patches · T={cache.x.shape[1]} · {cache.x.shape[-1]}px', '']
    for _fold_key, _sp in RUNS.items():
        cov = '  '.join((f'{p}={cache.coverage(_sp[p])[0]}/{cache.coverage(_sp[p])[1]}' for p in ('train', 'val', 'test')))
        lines.append(f'fold {_fold_key}: {cov}')
    mo.md('```\n' + '\n'.join(lines) + '\n```')
    return CACHE_DIR, cache


@app.cell
def _(VOID_CLASS, cache, np, plt):
    pi, ti = (0, min(10, cache.x.shape[1] - 1))
    xv = cache.denormalize(np.asarray(cache.x[pi, ti], dtype=np.float32))
    rgb_img = np.stack([xv[2], xv[1], xv[0]], -1)
    rgb_img = np.clip(rgb_img / max(np.percentile(rgb_img, 98), 1e-06), 0, 1)
    yv = cache.target[pi].astype(float)
    yv[yv == VOID_CLASS] = np.nan
    fig_prev, ax_prev = plt.subplots(1, 2, figsize=(8, 4))
    ax_prev[0].imshow(rgb_img)
    ax_prev[0].set_title(f'patch {cache.ids[pi]} · DOY {cache.doy[pi, ti]}')
    ax_prev[1].imshow(yv, cmap='tab20', vmin=0, vmax=19, interpolation='nearest')
    ax_prev[1].set_title(f'labels · fold {cache.folds[pi]}')
    for _a in ax_prev:
        _a.axis('off')
    fig_prev.tight_layout()
    fig_prev
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Folds
    """)
    return


@app.cell
def _(CLASS_NAMES, N_CLASSES, cache, np, plt):
    eda_fold_counts = {}
    for fold_id in sorted(set(cache.folds.tolist())):
        rows = np.flatnonzero(cache.folds == fold_id)
        tally = np.zeros(N_CLASSES, dtype=np.int64)
        for _r in rows:
            tally += np.bincount(cache.target[_r].reshape(-1), minlength=N_CLASSES)
        eda_fold_counts[int(fold_id)] = tally
    eda_crop_ids = [c for c in range(1, 19) if any((t[c] > 0 for t in eda_fold_counts.values()))]
    eda_mat = np.array([[eda_fold_counts[f][c] for c in eda_crop_ids] for f in sorted(eda_fold_counts)], dtype=float)
    eda_share = eda_mat / np.maximum(eda_mat.sum(1, keepdims=True), 1)
    fig_eda2, ax_eda2 = plt.subplots(figsize=(11, 3.2))
    im_eda = ax_eda2.imshow(eda_share, cmap='viridis', aspect='auto')
    ax_eda2.set_yticks(range(len(eda_fold_counts)), [f'fold {f}' for f in sorted(eda_fold_counts)])
    ax_eda2.set_xticks(range(len(eda_crop_ids)), [CLASS_NAMES[c] for c in eda_crop_ids], rotation=90, fontsize=7)
    ax_eda2.set_title('share of labelled crop pixels, by fold')
    fig_eda2.colorbar(im_eda, fraction=0.025)
    fig_eda2.tight_layout()
    fig_eda2
    return eda_crop_ids, eda_fold_counts


@app.cell
def _(CLASS_NAMES, cache, eda_crop_ids, eda_fold_counts, mo):
    eda_missing = {f: [CLASS_NAMES[c] for c in eda_crop_ids if eda_fold_counts[f][c] == 0] for f in sorted(eda_fold_counts)}
    eda_rows = ['| fold | patches | labelled px | background | void | classes present | absent |', '|---|---:|---:|---:|---:|---:|---|']
    for _f in sorted(eda_fold_counts):
        _t = eda_fold_counts[_f]
        tot = _t.sum()
        _lab = _t[1:19].sum()
        present = int((_t[1:19] > 0).sum())
        eda_rows.append(f'| {_f} | {int((cache.folds == _f).sum())} | {_lab:,} | {100 * _t[0] / max(tot, 1):.1f}% | {100 * _t[19] / max(tot, 1):.1f}% | {present}/18 | {', '.join(eda_missing[_f]) or '—'} |')
    mo.md(chr(10).join(eda_rows))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Temporal signatures and cloud

    Mean NDVI by day of year per crop, training folds only. Overlapping curves mean
    the signal is not there — check the date window before pretraining.

    PASTIS ships no cloud masks; blue reflectance is the proxy.
    """)
    return


@app.cell
def _(mo):
    EDA_SAMPLE_N = 40   # training patches sampled for the spectral pass
    mo.md(f"Sampling {EDA_SAMPLE_N} training patches for the spectral pass.")
    return (EDA_SAMPLE_N,)


@app.cell
def _(
    CLASS_NAMES,
    EDA_SAMPLE_N,
    FOLDS,
    N_CLASSES,
    RUNS,
    VOID_CLASS,
    cache,
    mo,
    np,
    plt,
):
    eda_train_ids = sorted(set(RUNS[FOLDS[0]]['train']))
    eda_rows_tr = cache.indices_for(eda_train_ids)
    eda_sample = eda_rows_tr[:EDA_SAMPLE_N] if len(eda_rows_tr) > EDA_SAMPLE_N else eda_rows_tr
    EDA_BIN = 10
    eda_nbins = 366 // EDA_BIN + 1
    eda_sum = np.zeros((N_CLASSES, eda_nbins))
    eda_cnt = np.zeros((N_CLASSES, eda_nbins))
    eda_blue, eda_brightness = ([], [])
    with mo.status.progress_bar(total=len(eda_sample), title='sampling') as bar_eda:
        for _r in eda_sample:
            arr = cache.denormalize(np.asarray(cache.x[_r], dtype=np.float32))
            valid_t = cache.valid[_r]
            doys = cache.doy[_r]  # (T,C,H,W)
            red, nir, blue = (arr[:, 2], arr[:, 6], arr[:, 0])
            ndvi = (nir - red) / np.maximum(nir + red, 1e-06)
            tgt = cache.target[_r]
            for _t in range(arr.shape[0]):
                if not valid_t[_t]:
                    continue
                _b = int(doys[_t]) // EDA_BIN
                eda_blue.append(float(blue[_t].mean()))
                eda_brightness.append(float(arr[_t].mean()))
                for _c in np.unique(tgt):
                    if _c == 0 or _c == VOID_CLASS:
                        continue
                    _m = tgt == _c
                    eda_sum[_c, _b] += float(ndvi[_t][_m].mean())
                    eda_cnt[_c, _b] += 1
            bar_eda.update()
    eda_blue = np.array(eda_blue)
    eda_brightness = np.array(eda_brightness)
    eda_prof = np.where(eda_cnt > 0, eda_sum / np.maximum(eda_cnt, 1), np.nan)
    eda_top = sorted([c for c in range(1, 19) if np.isfinite(eda_prof[c]).sum() > 3], key=lambda c: -np.isfinite(eda_prof[c]).sum())[:8]
    fig_eda3, ax_eda3 = plt.subplots(1, 2, figsize=(13, 3.8))
    xs_eda = np.arange(eda_nbins) * EDA_BIN
    for _c in eda_top:
        ax_eda3[0].plot(xs_eda, eda_prof[_c], lw=1.6, label=CLASS_NAMES[_c])
    ax_eda3[0].set_xlabel('day of year')
    ax_eda3[0].set_ylabel('mean NDVI')
    ax_eda3[0].set_title('temporal signature by crop (training folds)')
    ax_eda3[0].legend(fontsize=6, ncol=2)
    ax_eda3[0].grid(alpha=0.25)
    eda_thr = float(np.percentile(eda_blue, 90))
    ax_eda3[1].hist(eda_blue, bins=50, color='#4c72b0', edgecolor='white')
    ax_eda3[1].axvline(eda_thr, ls='--', c='crimson', lw=1.2)
    ax_eda3[1].set_xlabel('mean blue reflectance per date')
    ax_eda3[1].set_title(f'cloud proxy — {100 * (eda_blue > eda_thr).mean():.0f}% of dates above p90')
    ax_eda3[1].grid(alpha=0.25)
    fig_eda3.tight_layout()
    fig_eda3
    return eda_rows_tr, eda_sample


@app.cell
def _(N_BANDS, cache, eda_sample, mo, np):
    eda_norm_sample = np.asarray(cache.x[eda_sample[:12]], dtype=np.float32)
    eda_band_rows = ['| band | p05 | median | p95 | after scaling: median | IQR |', '|---|---:|---:|---:|---:|---:|']
    EDA_BAND_NAMES = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
    for _b in range(N_BANDS):
        _v = eda_norm_sample[:, :, _b].reshape(-1)
        _v = _v[np.isfinite(_v)]
        eda_band_rows.append(f'| {EDA_BAND_NAMES[_b]} | {cache.stats['q05'][_b]:.0f} | {cache.stats['median'][_b]:.0f} | {cache.stats['q95'][_b]:.0f} | {np.median(_v):+.3f} | {np.percentile(_v, 75) - np.percentile(_v, 25):.3f} |')
    eda_nl3 = chr(10)
    mo.md('**Band statistics.** Scaled medians should sit near 0. A band far off means its stats were estimated from too few patches.' + eda_nl3 + eda_nl3 + eda_nl3.join(eda_band_rows))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Patch gallery
    """)
    return


@app.cell
def _(VOID_CLASS, cache, eda_rows_tr, np, plt):
    eda_gal = eda_rows_tr[:6]
    fig_eda4, ax_eda4 = plt.subplots(2, len(eda_gal), figsize=(2.1 * len(eda_gal), 4.6), squeeze=False)
    for _col, _r in enumerate(eda_gal):
        _t = min(10, cache.x.shape[1] - 1)
        _a = cache.denormalize(np.asarray(cache.x[_r, _t], dtype=np.float32))
        img = np.stack([_a[2], _a[1], _a[0]], -1)
        img = np.clip(img / max(np.percentile(img, 98), 1e-06), 0, 1)
        _lab = cache.target[_r].astype(float)
        _lab[_lab == VOID_CLASS] = np.nan
        ax_eda4[0][_col].imshow(img)
        ax_eda4[0][_col].set_title(f'{cache.ids[_r]} · f{cache.folds[_r]}', fontsize=7)
        ax_eda4[1][_col].imshow(_lab, cmap='tab20', vmin=0, vmax=19, interpolation='nearest')
        ax_eda4[0][_col].axis('off')
        ax_eda4[1][_col].axis('off')
    fig_eda4.suptitle('training patches — RGB and labels', fontsize=9)
    fig_eda4.tight_layout()
    fig_eda4
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pretraining

    Corrupt a fraction of dates at the encoder output, reconstruct the
    reflectances. Masked values are permuted from elsewhere in the batch, not
    zeroed.

    One encoder per test fold.
    """)
    return


@app.cell
def _(mo):
    ui_mask = mo.ui.slider(0.1, 0.9, value=0.6, step=0.05, label="mask rate")
    ui_pre_epochs = mo.ui.slider(1, 300, value=40, step=5, label="pretrain epochs")
    ui_pre_lr = mo.ui.dropdown(["3e-4", "1e-3", "3e-3"], value="1e-3", label="pretrain lr")
    ui_smask = mo.ui.slider(0.0, 0.6, value=0.25, step=0.05,
                            label="spatial mask rate (0 = paper)")
    ui_sblock = mo.ui.dropdown(["4", "8", "16"], value="8", label="spatial block (px)")
    ui_bs = mo.ui.slider(1, 16, value=4, step=1, label="batch size")
    mo.vstack([ui_mask, ui_smask, ui_sblock, ui_pre_epochs, ui_pre_lr, ui_bs])
    return ui_bs, ui_mask, ui_pre_epochs, ui_pre_lr, ui_sblock, ui_smask


@app.cell
def _(mo):
    run_pre = mo.ui.run_button(label="② Pretrain (one encoder per fold)")
    run_pre
    return (run_pre,)


@app.cell
def _(
    CACHE_DIR,
    DEVICE,
    FOLDS,
    LinearDecoder,
    MAN,
    N_BANDS,
    RUNS,
    UBARN,
    WORK,
    cache,
    loader,
    make_dataset,
    masked_pixel_loss,
    math,
    mo,
    run_pre,
    spatiotemporal_mask,
    torch,
    ui_bs,
    ui_mask,
    ui_pre_epochs,
    ui_pre_lr,
    ui_sblock,
    ui_smask,
):
    def ckpt_path(fold_key):
        tag = f'pretrain_fold{fold_key}_{CACHE_DIR.name}_m{int(ui_mask.value * 100)}_s{int(ui_smask.value * 100)}b{ui_sblock.value}_e{ui_pre_epochs.value}_lr{ui_pre_lr.value}'
        return WORK / tag / 'best.pt'
    PRE_READY = all((ckpt_path(f).exists() for f in FOLDS))
    mo.stop(not run_pre.value and (not PRE_READY), mo.md('*Press ② to pretrain. ~1.5–2 h per fold on a GPU at 40 epochs.*'))

    def pretrain_fold(fold_key):
        pool = set(RUNS[fold_key]['train'])
        held = set(RUNS[fold_key]['val']) | set(RUNS[fold_key]['test'])
        idx = cache.indices_for(sorted(pool))
        assert not {cache.ids[i] for i in idx} & held, 'leak'
        if len(idx) == 0:
            raise RuntimeError(f'fold {fold_key}: no cached training patches')
        dl = loader(make_dataset(cache, idx, augment=True), ui_bs.value, shuffle=True, drop_last=len(idx) > ui_bs.value)
        torch.manual_seed(0)
        enc = UBARN(in_ch=N_BANDS, d_model=64, d_hidden=128, n_layers=3, n_heads=4).to(DEVICE)
        dec = LinearDecoder(64, N_BANDS).to(DEVICE)
        prm = list(enc.parameters()) + list(dec.parameters())
        opt = torch.optim.AdamW(prm, lr=float(ui_pre_lr.value), weight_decay=0.0001)
        n_ep = ui_pre_epochs.value
        warm = max(1, n_ep // 20)
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: (e + 1) / warm if e < warm else 0.5 * (1 + math.cos(math.pi * (e - warm) / max(1, n_ep - warm))))
        scl = torch.amp.GradScaler('cuda', enabled=DEVICE == 'cuda')
        out = ckpt_path(fold_key).parent
        out.mkdir(parents=True, exist_ok=True)
        hist = []
        for ep in range(ui_pre_epochs.value):
            enc.train()
            dec.train()
            tot, nb = (0.0, 0)
            for b in dl:
                x = b['x'].to(DEVICE, non_blocking=True)
                doy = b['doy'].to(DEVICE, non_blocking=True)
                valid = b['valid'].to(DEVICE, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=DEVICE == 'cuda'):
                    feats = enc.embed(x, doy)
                    corrupt, pxm = spatiotemporal_mask(feats, valid, ui_mask.value, ui_smask.value, int(ui_sblock.value))
                    rec = dec(enc.temporal(corrupt, valid))
                    loss = masked_pixel_loss(rec, x, pxm)
                opt.zero_grad(set_to_none=True)
                scl.scale(loss).backward()
                scl.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(prm, 5.0)
                scl.step(opt)
                scl.update()
                tot += float(loss.detach())
                nb += 1
            ep_loss = tot / max(nb, 1)
            hist.append(ep_loss)
            sch.step()
            if ep_loss <= min(hist):
                torch.save({'encoder': enc.state_dict(), 'decoder': dec.state_dict(), 'loss': ep_loss, 'epoch': ep, 'history': list(hist), 'mask_rate': ui_mask.value, 'spatial_mask': ui_smask.value, 'spatial_block': int(ui_sblock.value), 'fold': fold_key, 'split_manifest_sha256': MAN.sha256, 'pretrain_ids': sorted(pool)}, out / 'best.pt')
            yield (fold_key, ep, ep_loss, out / 'best.pt', hist, enc, dec)
    CKPTS, HISTORIES, VIZ = ({}, {}, {})
    todo = [f for f in FOLDS if not ckpt_path(f).exists()]
    total_steps = max(1, len(todo) * ui_pre_epochs.value)
    with mo.status.progress_bar(total=total_steps, title='pretraining') as bar_pre:
        for fk in FOLDS:
            if fk in todo:
                for _fold_key, ep, ep_loss, ckpt, hist, enc_i, dec_i in pretrain_fold(fk):
                    bar_pre.update()
                CKPTS[_fold_key] = ckpt
                HISTORIES[f'fold {_fold_key}'] = hist
                VIZ[_fold_key] = (enc_i, dec_i)
                print(f'fold {_fold_key}: trained, best masked MSE {min(hist):.5f}')
            else:
                st = torch.load(ckpt_path(fk), map_location='cpu')
                enc_l = UBARN(in_ch=N_BANDS, d_model=64, d_hidden=128, n_layers=3, n_heads=4).to(DEVICE)
                enc_l.load_state_dict(st['encoder'])
                dec_l = LinearDecoder(64, N_BANDS).to(DEVICE)
                if 'decoder' in st:
                    dec_l.load_state_dict(st['decoder'])
                CKPTS[fk] = ckpt_path(fk)
                HISTORIES[f'fold {fk}'] = st.get('history', [st['loss']])
                VIZ[fk] = (enc_l, dec_l)
                print(f'fold {fk}: loaded existing checkpoint, loss {st['loss']:.5f}')
    mo.md('Pretraining done: ' + ', '.join((f'fold {k}' for k in CKPTS)))
    return CKPTS, HISTORIES, VIZ, ckpt_path


@app.cell
def _(HISTORIES, plt):
    fig_loss, ax_loss = plt.subplots(figsize=(6, 3))
    for tag, hist_v in HISTORIES.items():
        ax_loss.plot(hist_v, lw=1.5, label=tag)
    ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("masked MSE")
    ax_loss.set_yscale("log"); ax_loss.legend(fontsize=8); ax_loss.grid(alpha=0.3)
    fig_loss.tight_layout()
    fig_loss
    return


@app.cell
def _(
    DEVICE,
    FOLDS,
    RUNS,
    VIZ,
    cache,
    np,
    permutation_mask,
    plt,
    torch,
    ui_mask,
):
    # held-out patch
    viz_fold = FOLDS[-1]
    enc_v, dec_v = VIZ[viz_fold]
    enc_v.eval()
    dec_v.eval()
    held_ids = sorted(set(RUNS[viz_fold]['val']) | set(RUNS[viz_fold]['test']))
    held_rows = cache.indices_for(held_ids)
    vi = int(held_rows[0]) if len(held_rows) else 0
    xv1 = torch.from_numpy(np.asarray(cache.x[vi], dtype=np.float32))[None].to(DEVICE)
    dv1 = torch.from_numpy(cache.doy[vi].astype(np.float32))[None].to(DEVICE)
    vv1 = torch.from_numpy(cache.valid[vi].copy())[None].to(DEVICE)
    with torch.no_grad():
        fv = enc_v.embed(xv1, dv1)
        cv, mv = permutation_mask(fv, vv1, ui_mask.value)
        rv = dec_v(enc_v.temporal(cv, vv1))

    def to_rgb(arr4):
        a = cache.denormalize(arr4)
        img = np.stack([a[2], a[1], a[0]], -1)
        return np.clip(img / max(np.percentile(img, 98), 1e-06), 0, 1)
    picks = torch.nonzero(mv[0]).flatten().cpu().numpy()[:6]
    fig_rec, ax_rec = plt.subplots(2, len(picks), figsize=(2.1 * len(picks), 4.4), squeeze=False)
    for _col, tt in enumerate(picks):
        ax_rec[0][_col].imshow(to_rgb(xv1[0, tt].cpu().numpy()))
        ax_rec[0][_col].set_title(f'DOY {cache.doy[vi, tt]}', fontsize=8)
        ax_rec[1][_col].imshow(to_rgb(rv[0, tt].float().cpu().numpy()))
        ax_rec[0][_col].axis('off')
        ax_rec[1][_col].axis('off')
    fig_rec.suptitle(f'held-out patch {cache.ids[vi]} (fold {viz_fold}) — top: masked input · bottom: reconstruction', fontsize=9)
    fig_rec.tight_layout()
    fig_rec
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Downstream

    | | |
    |---|---|
    | LP | frozen encoder, classifier only |
    | FT | pretrained init, everything trains |
    | SL | random init, same architecture |

    Early stop on fold 4. Metrics: OA, per-class IoU and F1, mIoU, mF1.
    Each `(fold, fraction, seed)` maps to one subset shared by all three regimes.
    """)
    return


@app.cell
def _(mo):
    ui_pcts = mo.ui.multiselect(["1", "5", "10", "20", "50", "100"],
                                value=["1", "5", "10", "20", "50", "100"],
                                label="label fractions (%)")
    ui_nseeds = mo.ui.dropdown(["3", "5"], value="3", label="seeds")
    ui_max_epochs = mo.ui.slider(10, 200, value=80, step=10, label="max epochs (FT / SL)")
    ui_lp_epochs = mo.ui.slider(10, 300, value=150, step=10, label="max epochs (LP)")
    ui_patience = mo.ui.slider(3, 30, value=10, step=1, label="early-stop patience")
    ui_val_cap = mo.ui.slider(0, 400, value=150, step=50,
                              label="val patches (0 = all of fold 4)")
    ui_lr = mo.ui.dropdown(["3e-4", "1e-3", "3e-3"], value="1e-3", label="lr (FT / SL)")
    ui_lp_lr = mo.ui.dropdown(["1e-3", "3e-3", "1e-2"], value="1e-2", label="lr (LP)")
    ui_save_w = mo.ui.dropdown(["off", "best seed per (fold, regime, fraction)", "all runs"],
                               value="best seed per (fold, regime, fraction)",
                               label="save downstream weights")
    ui_cw = mo.ui.checkbox(True, label="class-balanced loss")
    ui_fnorm = mo.ui.checkbox(True, label="normalise features before head")
    ui_ft_mode = mo.ui.dropdown(["plain", "LP-FT"], value="LP-FT", label="FT strategy")
    ui_lpft_epochs = mo.ui.slider(0, 40, value=10, step=5, label="LP-FT head-only epochs")
    ui_enc_mult = mo.ui.dropdown(["1", "0.3", "0.1"], value="0.1", label="FT encoder lr x")
    ui_warm = mo.ui.slider(0, 10, value=3, step=1, label="warmup epochs")
    ui_loss = mo.ui.dropdown(["CE", "CE + Dice"], value="CE + Dice", label="loss")
    ui_ls = mo.ui.dropdown(["0", "0.05", "0.1"], value="0.05", label="label smoothing")
    ui_ema = mo.ui.checkbox(True, label="EMA weights")
    ui_tta = mo.ui.checkbox(True, label="test-time augmentation (8x)")
    mo.vstack([
        mo.md("**Protocol**"),
        ui_pcts, ui_nseeds, ui_patience, ui_val_cap,
        mo.md("**Optimisation**"),
        ui_max_epochs, ui_lp_epochs, ui_lr, ui_lp_lr, ui_warm,
        mo.md("**Fine-tuning (FT only)**"),
        ui_ft_mode, ui_lpft_epochs, ui_enc_mult,
        mo.md("**Shared by all regimes**"),
        ui_loss, ui_cw, ui_ls, ui_ema, ui_tta, ui_fnorm,
        mo.md("**Output**"),
        ui_save_w,
    ])
    return (
        ui_cw,
        ui_ema,
        ui_enc_mult,
        ui_fnorm,
        ui_ft_mode,
        ui_loss,
        ui_lp_epochs,
        ui_lp_lr,
        ui_lpft_epochs,
        ui_lr,
        ui_ls,
        ui_max_epochs,
        ui_nseeds,
        ui_patience,
        ui_pcts,
        ui_save_w,
        ui_tta,
        ui_val_cap,
        ui_warm,
    )


@app.cell
def _(CKPTS, FOLDS, cfg, mo, ui_nseeds, ui_pcts, ui_save_w):
    PCTS = sorted(int(v) for v in ui_pcts.value)
    SEEDS = list(range(int(ui_nseeds.value)))
    REGIMES = ["LP", "FT", "SL"] if CKPTS else ["SL"]
    N_RUNS = len(FOLDS) * len(PCTS) * len(SEEDS) * len(REGIMES)

    est_nl = chr(10)
    est_unit = sum(PCTS) / 100.0 * len(SEEDS) * len(REGIMES) * len(FOLDS)
    est_lines = [
        f"regimes {', '.join(REGIMES)} | fractions {PCTS} | seeds {SEEDS} | folds {FOLDS}",
        f"runs {N_RUNS} | cost ~{est_unit:.1f} full-label runs | ~{est_unit * 10 / 60:.1f} h at 600 patches / 64²",
    ]
    if cfg.crop == 128:
        est_lines.append("crop 128: multiply by ~4")
    if ui_save_w.value != "off":
        n_saved = (len(FOLDS) * len(PCTS) * len(REGIMES)
                   * (len(SEEDS) if ui_save_w.value.startswith("all") else 1))
        est_lines.append(f"weights: {n_saved} files x ~5.6 MB = ~{n_saved * 5.6 / 1024:.2f} GB")
    if est_unit * 10 / 60 * (4 if cfg.crop == 128 else 1) > 7:
        est_lines.append("!! will not fit in one session alongside pretraining")
    mo.md("```" + est_nl + est_nl.join(est_lines) + est_nl + "```")
    return N_RUNS, PCTS, REGIMES, SEEDS


@app.cell
def _(mo):
    run_dn = mo.ui.run_button(label="3. Run downstream")
    run_dn
    return (run_dn,)


@app.cell
def _(
    CACHE_DIR,
    CKPTS,
    FOLDS,
    PCTS,
    REGIMES,
    RUNS,
    SEEDS,
    VAL_FOLD,
    WORK,
    cache,
    json,
    mo,
    np,
    run_dn,
    scarce_subset,
    ui_cw,
    ui_ema,
    ui_enc_mult,
    ui_fnorm,
    ui_ft_mode,
    ui_loss,
    ui_lp_epochs,
    ui_lp_lr,
    ui_lpft_epochs,
    ui_lr,
    ui_ls,
    ui_max_epochs,
    ui_patience,
    ui_save_w,
    ui_tta,
    ui_val_cap,
    ui_warm,
):
    RESULTS_FILE = WORK / 'results_downstream.json'
    RUN_SIG = {'folds': FOLDS, 'val_fold': VAL_FOLD, 'pcts': PCTS, 'seeds': SEEDS, 'regimes': REGIMES, 'max_epochs': ui_max_epochs.value, 'lp_epochs': ui_lp_epochs.value, 'patience': ui_patience.value, 'val_cap': ui_val_cap.value, 'lr': ui_lr.value, 'lp_lr': ui_lp_lr.value, 'class_balanced': ui_cw.value, 'feat_norm': ui_fnorm.value, 'save_weights': ui_save_w.value, 'ft_mode': ui_ft_mode.value, 'lpft_epochs': ui_lpft_epochs.value, 'enc_mult': ui_enc_mult.value, 'warmup': ui_warm.value, 'loss': ui_loss.value, 'label_smoothing': ui_ls.value, 'ema': ui_ema.value, 'tta': ui_tta.value, 'cache': str(CACHE_DIR), 'ckpts': {str(k): str(v) for k, v in CKPTS.items()}}
    DN_READY = False
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as fh_prev:
            prev = json.load(fh_prev)
        DN_READY = prev.get('signature') == RUN_SIG
    mo.stop(not run_dn.value and (not DN_READY), mo.md('*Press 3. Check the cost estimate above first.*'))

    def subset_key(fold_key, pct, seed):
        return f'f{fold_key}_p{pct}_s{seed}'

    def make_label_subset(fold_key, pct, seed):
        """Patch IDs for one (fold, fraction, seed). Rare-class weighted."""
        train_ids = [i for i in RUNS[fold_key]['train'] if i in cache.pos]
        if pct >= 100:
            return list(train_ids)
        n = max(1, int(round(len(train_ids) * pct / 100)))
        rows = cache.indices_for(train_ids)
        picked = scarce_subset(cache, rows, min(n, len(rows)), seed=seed)
        return [cache.ids[r] for r in picked]
    SUBSETS = {subset_key(f, p, sd): make_label_subset(f, p, sd) for f in FOLDS for p in PCTS for sd in SEEDS}
    VAL_IDS = {}
    for _f in FOLDS:
        va_all = cache.indices_for(RUNS[_f]['val'])
        if ui_val_cap.value and len(va_all) > ui_val_cap.value:
            va_pick = np.random.default_rng(12345).choice(len(va_all), size=ui_val_cap.value, replace=False)
            va_all = va_all[np.sort(va_pick)]
        VAL_IDS[_f] = va_all
    sub_nl = chr(10)
    sub_lines = [f'{len(SUBSETS)} subsets, shared across {len(REGIMES)} regimes']
    for _f in FOLDS:
        sub_lines.append(f'fold {_f}: val {len(VAL_IDS[_f])} (fold {VAL_FOLD}), test {len(cache.indices_for(RUNS[_f]['test']))}')
        for _p in PCTS:
            sizes = sorted({len(SUBSETS[subset_key(_f, _p, sd)]) for sd in SEEDS})
            sub_lines.append(f'  {_p:>3}% -> {sizes}')
    mo.md('```' + sub_nl + sub_nl.join(sub_lines) + sub_nl + '```')
    return DN_READY, RESULTS_FILE, RUN_SIG, SUBSETS, VAL_IDS, prev, subset_key


@app.cell
def _(
    CKPTS,
    ConfusionMeter,
    DEVICE,
    DN_READY,
    FOLDS,
    MAN,
    N_BANDS,
    N_CLASSES,
    N_RUNS,
    PCTS,
    REGIMES,
    RESULTS_FILE,
    RUNS,
    RUN_SIG,
    SEEDS,
    SUBSETS,
    SegmentationModel,
    UBARN,
    VAL_IDS,
    VOID_CLASS,
    WORK,
    cache,
    copy,
    dice_loss,
    json,
    loader,
    make_dataset,
    mo,
    np,
    prev,
    subset_key,
    torch,
    ui_bs,
    ui_cw,
    ui_ema,
    ui_enc_mult,
    ui_fnorm,
    ui_ft_mode,
    ui_loss,
    ui_lp_epochs,
    ui_lp_lr,
    ui_lpft_epochs,
    ui_lr,
    ui_ls,
    ui_max_epochs,
    ui_patience,
    ui_save_w,
    ui_tta,
    ui_warm,
):
    def build_model(mode, fold_key):
        enc = UBARN(in_ch=N_BANDS, d_model=64, d_hidden=128, n_layers=3, n_heads=4)
        if mode in ('LP', 'FT'):
            enc.load_state_dict(torch.load(CKPTS[fold_key], map_location='cpu')['encoder'])
        return SegmentationModel(enc, n_classes=N_CLASSES, freeze=mode == 'LP', feat_norm=ui_fnorm.value).to(DEVICE)

    def class_weights(rows):
        """Inverse-sqrt frequency over the labelled subset; void excluded."""
        counts = np.zeros(N_CLASSES, dtype=np.float64)
        for r in rows:
            counts += np.bincount(cache.target[r].reshape(-1), minlength=N_CLASSES)
        counts[VOID_CLASS] = 0
        w = np.where(counts > 0, 1.0 / np.sqrt(np.maximum(counts, 1)), 0.0)
        w = w / w[w > 0].mean()
        return torch.tensor(w, dtype=torch.float32)

    @torch.no_grad()
    def predict(model, x, doy, valid, tta=False):
        if not tta:
            return model(x, doy, valid).float().softmax(1)
        acc = 0
        for flip in (False, True):
            xf = x.flip(-1) if flip else x
            for k in range(4):
                pr = model(torch.rot90(xf, k, dims=(-2, -1)), doy, valid).float().softmax(1)
                pr = torch.rot90(pr, -k, dims=(-2, -1))
                acc = acc + (pr.flip(-1) if flip else pr)
        return acc / 8

    @torch.no_grad()
    def evaluate(model, dl, tta=False):
        model.eval()
        meter = ConfusionMeter(N_CLASSES, ignore_index=VOID_CLASS)
        for b in dl:
            with torch.amp.autocast('cuda', enabled=DEVICE == 'cuda'):
                pr = predict(model, b['x'].to(DEVICE), b['doy'].to(DEVICE), b['valid'].to(DEVICE), tta)
            meter.update(pr.argmax(1), b['y'])
        return meter

    class EMA:
        """Averages trainable parameters only; frozen weights and buffers are copied exactly."""

        def __init__(self, model, decay):
            self.decay, self.n = (decay, 0)
            self.track = {n for n, p in model.named_parameters() if p.requires_grad}
            self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

        @torch.no_grad()
        def update(self, model):
            self.n += 1
            d = min(self.decay, (1 + self.n) / (10 + self.n))
            for k, v in model.state_dict().items():
                if k in self.track and v.dtype.is_floating_point:
                    self.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)
                else:
                    self.shadow[k].copy_(v)

    def fit(model, tr_dl, va_dl, epochs, patience, head_lr, enc_mult, weight, early_stop=True):
        head_p = [p for p in model.head.parameters() if p.requires_grad]
        enc_p = [p for p in model.encoder.parameters() if p.requires_grad]
        groups = [{'params': head_p, 'lr': head_lr}]
        if enc_p:
            groups.append({'params': enc_p, 'lr': head_lr * enc_mult})
        opt = torch.optim.AdamW(groups, weight_decay=0.0001)
        base_lr = [g['lr'] for g in opt.param_groups]
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', patience=max(2, patience // 2), factor=0.5)
        scl = torch.amp.GradScaler('cuda', enabled=DEVICE == 'cuda')
        ce = torch.nn.CrossEntropyLoss(weight=weight.to(DEVICE) if weight is not None else None, ignore_index=VOID_CLASS, label_smoothing=float(ui_ls.value))
        use_dice = ui_loss.value == 'CE + Dice'
        ema = EMA(model, 0.99) if ui_ema.value else None
        ema_model = copy.deepcopy(model) if ema else None
        warm = min(ui_warm.value, max(epochs - 1, 0))
        prm = head_p + enc_p
        best, best_state, stale, ran = (-1.0, None, 0, 0)
        for ep in range(epochs):
            if ep < warm:
                for g, b0 in zip(opt.param_groups, base_lr):
                    g['lr'] = b0 * (ep + 1) / warm
            model.train()
            for b in tr_dl:
                yb = b['y'].to(DEVICE)
                with torch.amp.autocast('cuda', enabled=DEVICE == 'cuda'):
                    lg = model(b['x'].to(DEVICE), b['doy'].to(DEVICE), b['valid'].to(DEVICE))
                loss = ce(lg.float(), yb)
                if use_dice:
                    loss = loss + dice_loss(lg, yb, VOID_CLASS, N_CLASSES)
                opt.zero_grad(set_to_none=True)
                scl.scale(loss).backward()
                scl.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(prm, 5.0)
                scl.step(opt)
                scl.update()
                if ema:
                    ema.update(model)
            ran = ep + 1
            target = model
            if ema:
                ema_model.load_state_dict(ema.shadow)
                target = ema_model
            val_miou = evaluate(target, va_dl).scores()['mIoU']
            if ep >= warm:
                sch.step(val_miou)
            if val_miou > best + 1e-05:
                best, stale = (val_miou, 0)
                best_state = {k: v.detach().cpu().clone() for k, v in target.state_dict().items()}
            elif early_stop and ep >= warm:
                stale += 1
                if stale >= patience:
                    break
        if best_state:
            model.load_state_dict(best_state)
        return (best, ran)

    def train_run(mode, fold_key, tr_dl, va_dl, weight):
        """LP / FT / SL. Shared: warmup, EMA, label smoothing, loss. FT only: LP-FT, encoder lr multiplier."""
        m = build_model(mode, fold_key)
        head_lr = float(ui_lp_lr.value if mode == 'LP' else ui_lr.value)
        ran_total = 0
        if mode == 'FT' and ui_ft_mode.value == 'LP-FT' and (ui_lpft_epochs.value > 0):
            for prm_e in m.encoder.parameters():
                prm_e.requires_grad_(False)
            m.freeze = True
            _, r1 = fit(m, tr_dl, va_dl, ui_lpft_epochs.value, ui_patience.value, float(ui_lp_lr.value), 1.0, weight, early_stop=False)
            for prm_e in m.encoder.parameters():
                prm_e.requires_grad_(True)
            m.freeze = False
            ran_total += r1
        best, r2 = fit(m, tr_dl, va_dl, ui_lp_epochs.value if mode == 'LP' else ui_max_epochs.value, ui_patience.value, head_lr, float(ui_enc_mult.value) if mode == 'FT' else 1.0, weight)
        return (m, best, ran_total + r2)
    WEIGHT_DIR = WORK / 'weights'
    BEST_SO_FAR = {}

    def save_run_weights(model, fold_key, pct, seed, mode, score, meta):
        if ui_save_w.value == 'off':
            return None
        key = (fold_key, mode, pct)
        if ui_save_w.value.startswith('best'):
            if score <= BEST_SO_FAR.get(key, -1.0):
                return None
            BEST_SO_FAR[key] = score
            name = f'{mode}_f{fold_key}_p{pct}_best.pt'
        else:
            name = f'{mode}_f{fold_key}_p{pct}_s{seed}.pt'
        WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
        path = WEIGHT_DIR / name
        torch.save({'model': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'regime': mode, 'fold': fold_key, 'pct': pct, 'seed': seed, 'n_classes': N_CLASSES, 'd_model': 64, 'feat_norm': ui_fnorm.value, 'frozen_encoder': mode == 'LP', 'split_manifest_sha256': MAN.sha256, 'pretrain_ckpt': str(CKPTS.get(fold_key, '')), 'metrics': meta}, path)
        return str(path)
    results = []
    with mo.status.progress_bar(total=0 if DN_READY else N_RUNS, title='downstream runs') as bar_dn:
        if DN_READY:
            results = prev['runs']
        else:
            for _fold_key in FOLDS:
                va_dl = loader(make_dataset(cache, VAL_IDS[_fold_key]), ui_bs.value, shuffle=False)
                te_dl = loader(make_dataset(cache, cache.indices_for(RUNS[_fold_key]['test'])), ui_bs.value, shuffle=False)
                for pct in PCTS:
                    for seed in SEEDS:
                        ids_used = SUBSETS[subset_key(_fold_key, pct, seed)]
                        tr_rows = cache.indices_for(ids_used)
                        weight = class_weights(tr_rows) if ui_cw.value else None
                        tr_dl = loader(make_dataset(cache, tr_rows, augment=True), ui_bs.value, shuffle=True)
                        for _mode in REGIMES:
                            torch.manual_seed(seed)
                            np.random.seed(seed)
                            _m, val_best, epochs_run = train_run(_mode, _fold_key, tr_dl, va_dl, weight)
                            sc = evaluate(_m, te_dl, tta=ui_tta.value).scores()
                            wpath = save_run_weights(_m, _fold_key, pct, seed, _mode, sc['mIoU'], {k: sc[k] for k in ('OA', 'mIoU', 'mF1', 'Kappa')})
                            results.append({'weights': wpath, 'fold': _fold_key, 'pct': pct, 'seed': seed, 'regime': _mode, 'n_train': len(tr_rows), 'epochs_run': epochs_run, 'val_mIoU': val_best, 'subset_sig': ','.join(sorted(ids_used))[:64], 'OA': sc['OA'], 'mIoU': sc['mIoU'], 'mF1': sc['mF1'], 'Kappa': sc['Kappa'], 'per_class_iou': sc['per_class_iou'], 'per_class_f1': sc['per_class_f1']})
                            del _m
                            if DEVICE == 'cuda':
                                torch.cuda.empty_cache()
                            bar_dn.update()
            with open(RESULTS_FILE, 'w') as fh:
                json.dump({'signature': RUN_SIG, 'runs': results}, fh, indent=2)
            with open(WORK / 'label_subsets.json', 'w') as fh:
                json.dump(SUBSETS, fh, indent=2)
    mo.md(f'{len(results)} runs ' + ('loaded from disk' if DN_READY else 'finished'))
    return WEIGHT_DIR, results


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Subset check
    """)
    return


@app.cell
def _(FOLDS, PCTS, SEEDS, mo, results):
    chk_bad = []
    for _f in FOLDS:
        for _p in PCTS:
            for _sd in SEEDS:
                sigs = {r['regime']: r['subset_sig'] for r in results if r['fold'] == _f and r['pct'] == _p and (r['seed'] == _sd)}
                if len(set(sigs.values())) > 1:
                    chk_bad.append((_f, _p, _sd))
    chk_nl = chr(10)
    if chk_bad:
        mo.md('**MISMATCH:**' + chk_nl + chk_nl.join((f'- fold {a}, {b}%, seed {c}' for a, b, c in chk_bad)))
    else:
        mo.md(f'All {len(FOLDS) * len(PCTS) * len(SEEDS)} groups: identical subsets across regimes.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Summary
    """)
    return


@app.cell
def _(FOLDS, PCTS, REGIMES, SEEDS, mo, np, results):
    def ms(vals):
        v = np.array(vals, dtype=float)
        return f'{100 * v.mean():.2f} ± {100 * v.std():.2f}'
    res_nl = chr(10)
    res_rows = ['| fraction | n | regime | OA | mIoU | mF1 | Kappa | epochs |', '|---:|---:|---|---:|---:|---:|---:|---:|']
    for _p in PCTS:
        for _mode in REGIMES:
            sel = [r for r in results if r['pct'] == _p and r['regime'] == _mode]
            if not sel:
                continue
            res_rows.append(f'| {_p}% | {int(np.median([r['n_train'] for r in sel]))} | {_mode} | {ms([r['OA'] for r in sel])} | {ms([r['mIoU'] for r in sel])} | {ms([r['mF1'] for r in sel])} | {ms([r['Kappa'] for r in sel])} | {int(np.median([r['epochs_run'] for r in sel]))} |')
    res_rows.append('| — | — | *U-TAE (leaderboard)* | *83.2* | *63.1* | — | — | — |')
    mo.md(f'Mean ± std (%) over {len(SEEDS)} seeds × {len(FOLDS)} folds. Leaderboard row is 5-fold, 128², full series.' + res_nl + res_nl + res_nl.join(res_rows))
    return


@app.cell
def _(PCTS, REGIMES, np, plt, results):
    fig_dn, axs_dn = plt.subplots(1, 4, figsize=(15, 3.4))
    dn_styles = {'LP': ('-o', '#4c72b0'), 'FT': ('-s', '#dd8452'), 'SL': ('--^', '#55a868')}
    for ax_d, metric in zip(axs_dn, ['OA', 'mIoU', 'mF1', 'Kappa']):
        for _mode in REGIMES:
            xs = [p for p in PCTS if any((r['pct'] == p and r['regime'] == _mode for r in results))]
            mu = [float(np.mean([r[metric] for r in results if r['pct'] == p and r['regime'] == _mode])) for p in xs]
            _sd = [float(np.std([r[metric] for r in results if r['pct'] == p and r['regime'] == _mode])) for p in xs]
            fmt, _col = dn_styles.get(_mode, ('-o', None))
            ax_d.errorbar(xs, mu, yerr=_sd, fmt=fmt, color=_col, capsize=3, ms=4, label=_mode)
        ax_d.set_xscale('log')
        ax_d.set_xticks(PCTS, [f'{p}%' for p in PCTS], fontsize=7)
        ax_d.set_xlabel('label fraction')
        ax_d.set_title(metric)
        ax_d.grid(alpha=0.3)
    axs_dn[0].legend(fontsize=8)
    fig_dn.tight_layout()
    fig_dn
    return (dn_styles,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Per class
    """)
    return


@app.cell
def _(PCTS, mo):
    ui_pc_frac = mo.ui.dropdown([str(p) for p in PCTS], value=str(PCTS[-1]),
                                label="fraction")
    ui_pc_frac
    return (ui_pc_frac,)


@app.cell
def _(CLASS_NAMES, REGIMES, mo, np, results, ui_pc_frac):
    pc_frac = int(ui_pc_frac.value)
    PC_CLASSES = list(range(0, 19))

    def pc_stats(mode, key):
        sel = [r for r in results if r['pct'] == pc_frac and r['regime'] == mode]
        arr = np.array([r[key] for r in sel], dtype=float)
        return (np.nanmean(arr, 0), np.nanstd(arr, 0))
    pc_nl = chr(10)
    pc_head = '| class | ' + ' | '.join((f'{m} IoU | {m} F1' for m in REGIMES)) + ' |'
    pc_rows = [pc_head, '|---|' + '---:|' * (2 * len(REGIMES))]
    pc_cache = {m: (pc_stats(m, 'per_class_iou'), pc_stats(m, 'per_class_f1')) for m in REGIMES}
    for _c in PC_CLASSES:
        cells_pc = []
        for _m in REGIMES:
            (_iou_mu, _iou_sd), (f1_mu, f1_sd) = pc_cache[_m]
            for mu_v, sd_v in ((_iou_mu[_c], _iou_sd[_c]), (f1_mu[_c], f1_sd[_c])):
                cells_pc.append('—' if np.isnan(mu_v) else f'{100 * mu_v:.1f} ± {100 * sd_v:.1f}')
        pc_rows.append(f'| {CLASS_NAMES[_c]} | ' + ' | '.join(cells_pc) + ' |')
    mo.md(f'Per-class IoU and F1 (%) at {pc_frac}% labels' + pc_nl + pc_nl + pc_nl.join(pc_rows))
    return PC_CLASSES, pc_cache, pc_frac


@app.cell
def _(CLASS_NAMES, PC_CLASSES, REGIMES, dn_styles, np, pc_cache, pc_frac, plt):
    fig_pc, ax_pc = plt.subplots(figsize=(13, 3.6))
    pc_w = 0.8 / len(REGIMES)
    pc_x = np.arange(len(PC_CLASSES))
    for i, _m in enumerate(REGIMES):
        (_iou_mu, _iou_sd), _ = pc_cache[_m]
        ax_pc.bar(pc_x + i * pc_w, np.nan_to_num(_iou_mu[PC_CLASSES]), pc_w, yerr=np.nan_to_num(_iou_sd[PC_CLASSES]), capsize=2, label=_m, color=dn_styles.get(_m, (None, None))[1])
    ax_pc.set_xticks(pc_x + pc_w * (len(REGIMES) - 1) / 2, [CLASS_NAMES[c] for c in PC_CLASSES], rotation=75, fontsize=7)
    ax_pc.set_ylabel('IoU')
    ax_pc.set_title(f'per-class IoU at {pc_frac}% labels')
    ax_pc.legend(fontsize=8)
    ax_pc.grid(axis='y', alpha=0.3)
    fig_pc.tight_layout()
    fig_pc
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Download

    Session disk is wiped after 12 h, or 90 min idle.
    """)
    return


@app.cell
def _(FOLDS, Path, WEIGHT_DIR, WORK, ckpt_path, mo):
    def size_of(paths):
        return sum(Path(x).stat().st_size for x in paths if Path(x).exists())

    dl_items, dl_notes = [], []

    for dl_name in ["results_downstream.json", "label_subsets.json"]:
        dl_p = WORK / dl_name
        if dl_p.exists():
            dl_items.append(mo.download(data=dl_p.read_bytes(), filename=dl_name,
                                        mimetype="application/json",
                                        label=f"{dl_name} ({dl_p.stat().st_size / 1e3:.0f} kB)"))

    dl_weights = sorted(WEIGHT_DIR.glob("*.pt")) if WEIGHT_DIR.exists() else []
    dl_pre = [ckpt_path(f) for f in FOLDS if ckpt_path(f).exists()]

    if dl_weights or dl_pre:
        import io, zipfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for q in dl_pre:
                zf.write(q, f"pretrain/{q.parent.name}.pt")
            for q in dl_weights:
                zf.write(q, f"downstream/{q.name}")
        dl_items.append(mo.download(data=buf.getvalue(), filename="ubarn_weights.zip",
                                    mimetype="application/zip",
                                    label=f"weights.zip ({len(dl_pre)} pretrain + "
                                          f"{len(dl_weights)} downstream, "
                                          f"{buf.tell() / 1e6:.0f} MB)"))
        dl_notes.append(f"pretrain {size_of(dl_pre) / 1e6:.0f} MB, "
                        f"downstream {size_of(dl_weights) / 1e6:.0f} MB")

    mo.vstack(dl_items + ([mo.md(" · ".join(dl_notes))] if dl_notes else [])
              or [mo.md("*Nothing saved yet.*")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reloading weights

    ```python
    st = torch.load("LP_f1_p100_best.pt", map_location="cpu")
    enc = UBARN(in_ch=N_BANDS, d_model=st["d_model"], d_hidden=128, n_layers=3, n_heads=4)
    m = SegmentationModel(enc, n_classes=st["n_classes"],
                          freeze=st["frozen_encoder"], feat_norm=st["feat_norm"])
    m.load_state_dict(st["model"])
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
