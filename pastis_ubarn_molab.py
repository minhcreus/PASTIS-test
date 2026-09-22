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
    # U-BARN on PASTIS — single notebook, 2 folds

    Self-supervised masked pretraining of a Unet + transformer on Sentinel-2 image
    time series, evaluated on PASTIS crop segmentation over the official folds.
    Method from Dumeur, Valero & Inglada, IEEE JSTARS 17 (2024) 4350–4367.

    Everything is in this one file: no local imports, no second notebook, no files
    to upload. The split manifest is fetched from GitHub on first run.

    Run top to bottom. Cells evaluate as you open them; the four expensive steps
    wait for a button.
    """)
    return


@app.cell
def _():
    import json, math, os, sys, urllib.request
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
    NUM_WORKERS = max(0, min(2, (os.cpu_count() or 1) - 1))

    print("torch  :", torch.__version__)
    print("device :", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "(no GPU — attach one from the notebook specs menu)")
    print("work   :", WORK)
    print("workers:", NUM_WORKERS)
    return (
        DEVICE,
        DataLoader,
        F,
        NUM_WORKERS,
        Path,
        WORK,
        asdict,
        dataclass,
        datetime,
        json,
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
    # Downloaded from GitHub on first run, so this notebook needs no uploaded
    # files at all. Point MANIFEST_URL at your own fork if you change the splits.

    MANIFEST_URL = ("https://raw.githubusercontent.com/minhcreus/PASTIS-test/"
                    "main/pastis_official_splits_v1.json")
    MANIFEST_PATH = WORK / "pastis_official_splits_v1.json"

    def locate_manifest():
        """Prefer a local copy anywhere obvious, else download."""
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

    Three cells: split handling, data handling, model. Inlined rather than
    imported, because a molab notebook cannot count on sibling files being there.
    Collapse them and move on — nothing here needs editing.
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
        """Return a report dict. Checks duplicates, cross-set overlap, membership,
        coverage, and — when a manifest is given — whether the split respects the
        official geographic folds."""
        sets = {}
        report = {'name': name, 'errors': [], 'warnings': [], 'counts': {}}
        for key in ('train', 'val', 'test'):
            raw = [normalize_id(i) for i in split.get(key, [])]
            uniq = set(raw)
            sets[key] = uniq
    # normalisation of ID spellings
            report['counts'][key] = len(raw)
            if len(raw) != len(uniq):
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
    # validation
                if unknown:
                    report['errors'].append(f"{len(unknown)} id(s) in '{key}' do not exist in PASTIS: {sorted(unknown)[:5]}")
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
        """Integrity report, in the format used in the project's existing checks."""
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
        """Pick a subset of the five official rotations, e.g. the first two."""
        out = {}
        for f in folds:
            sp = manifest.split(f)
            out[int(f)] = {k: [normalize_id(i) for i in sp[k]] for k in ('train', 'val', 'test')}
        return out

    def required_ids(runs: dict[int, dict[str, list[str]]]) -> set[str]:  # fold purity: a split that mixes official folds leaks spatial context
        """Every patch touched by the chosen folds — what you actually need on disk."""
        out: set[str] = set()
        for sp in runs.values():
            for key in ('train', 'val', 'test'):
                out.update(sp[key])
    # selecting folds to run
        return out

    return (
        Manifest,
        folds_to_run,
        load_manifest,
        normalize_id,
        print_report,
        required_ids,
        validate_split,
    )


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
        """Look for a directory containing both DATA_S2/ and ANNOTATIONS/."""
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
        """Fetch only the S2 series and annotations for the given patch IDs."""
        from huggingface_hub import snapshot_download
        prefix = _hf_prefix()
        patterns = []
        for pid in ids:
            patterns.append(f'{prefix}DATA_S2/S2_{pid}.npy')
            patterns.append(f'{prefix}ANNOTATIONS/TARGET_{pid}.npy')
        local = snapshot_download(HF_REPO, repo_type='dataset', allow_patterns=patterns, cache_dir=str(cache_dir / 'hf'), max_workers=workers)
        return Path(local) / prefix if prefix else Path(local)

    def resolve_source(ids, cache_dir: Path, local_hint: str | None=None):
        """Return (raw_root, description) — prefer a local copy, else download."""
        local = find_local_pastis(local_hint)
        if local is not None:
            return (local, f'local copy at {local}')
        return (download_patches(ids, cache_dir), 'Hugging Face mirror')

    def _doy(yyyymmdd: str) -> int:
        return datetime.strptime(str(yyyymmdd), '%Y%m%d').timetuple().tm_yday

    def _select_dates(dates: list[str], cfg: DataConfig) -> list[int]:
        """Positions in the series to keep, after the date window and thinning."""
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
        """Robust per-band stats: 5th pct, median, 95th pct (paper eq. 4a/4b).

        Pass TRAINING ids only — deriving these from val/test patches is a subtle
        form of leakage.
        """
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
    # preprocessing
            stats['q95'].append(float(np.quantile(v, 0.95)))
        return stats

    def normalize(arr: np.ndarray, stats: dict) -> np.ndarray:
        q05 = np.asarray(stats['q05'], dtype=np.float32)[None, :, None, None]
        q95 = np.asarray(stats['q95'], dtype=np.float32)[None, :, None, None]
        med = np.asarray(stats['median'], dtype=np.float32)[None, :, None, None]
        x = np.clip(arr.astype(np.float32), q05, q95)
        return (x - med) / np.maximum(q95 - q05, 1e-06)

    def build_cache(raw_root: Path, manifest: Manifest, ids, cfg: DataConfig, out_dir: Path, norm_ids=None, progress=None) -> Path:
        """Preprocess the given IDs into one memmap + sidecar arrays.

        `norm_ids` should be the training IDs of the fold you intend to run; the
        normalisation statistics are derived from those only.
        """
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

        def indices_for(self, ids) -> np.ndarray:  # deterministic centre crop
            """Map patch IDs to cache rows, dropping any that are not cached."""
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
                        x, y = (x[..., ::-1].copy(), y[..., ::-1].copy())
    # cache handle
                    if np.random.rand() < 0.5:
                        x, y = (x[..., ::-1, :].copy(), y[..., ::-1, :].copy())
                return {'x': torch.from_numpy(x), 'doy': torch.from_numpy(cache.doy[j].astype(np.float32)), 'valid': torch.from_numpy(cache.valid[j].copy()), 'y': torch.from_numpy(y)}
        return _DS()

    def scarce_subset(cache: PastisCache, train_idx, n: int, seed: int=0) -> np.ndarray:
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

    return (
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
        """Encodes each date independently: (N, C, H, W) -> (N, d_model, H, W).

        This is the U-TAE Unet with the L-TAE stripped out of the bottleneck, which
    # --------------------------------------------------------------------------
    # spatio-spectral encoder (Unet, temporal attention removed from bottleneck)
        is exactly how the paper describes the SSE (Section III-A1).
        """

        def __init__(self, in_ch: int=10, widths=(32, 64, 128, 128), d_model: int=64):
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
        """doy: (B, T) float -> (B, T, d_model)."""
        device, dtype = (doy.device, torch.float32)
        i = torch.arange(d_model // 2, device=device, dtype=dtype)
        denom = torch.pow(torch.tensor(1000.0, device=device), 2 * i / d_model)
        ang = doy.to(dtype).unsqueeze(-1) / denom
        pe = torch.zeros(*doy.shape, d_model, device=device, dtype=dtype)
        pe[..., 0::2] = torch.sin(ang)
        pe[..., 1::2] = torch.cos(ang)
        return pe

    class UBARN(nn.Module):
        """Patch embedding (SSE + DOY PE) followed by a per-pixel temporal transformer.

        Input  (B, T, C, H, W)  ->  output (B, T, d_model, H, W), i.e. the temporal
        and spatial resolution of the input are preserved, which is the whole point
        of the architecture relative to U-TAE.
        """

        def __init__(self, in_ch: int=10, d_model: int=64, d_hidden: int=128, n_layers: int=3, n_heads: int=4, widths=(32, 64, 128, 128), dropout: float=0.1):
            super().__init__()
            self.d_model = d_model
            self.sse = SpatioSpectralEncoder(in_ch, widths, d_model)
            layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_hidden, dropout=dropout, activation='relu', batch_first=True, norm_first=False)
            self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)

        def embed(self, x: torch.Tensor, doy: torch.Tensor) -> torch.Tensor:
    # positional encoding on day-of-year (paper eq. 1, scaling constant 1000)
            """(B,T,C,H,W) -> (B,T,d,H,W) patch embeddings with positional encoding."""
            b, t, c, h, w = x.shape
            f = self.sse(x.reshape(b * t, c, h, w)).reshape(b, t, self.d_model, h, w)
            pe = doy_encoding(doy, self.d_model)
            return f + pe[:, :, :, None, None]

        def temporal(self, f: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
            """(B,T,d,H,W) -> (B,T,d,H,W). `valid` is (B,T) bool; padded dates masked."""
            b, t, d, h, w = f.shape
            seq = f.permute(0, 3, 4, 1, 2).reshape(b * h * w, t, d)
            pad = (~valid)[:, None, None, :].expand(b, h, w, t).reshape(b * h * w, t)
            out = self.transformer(seq, src_key_padding_mask=pad)
            out = torch.nan_to_num(out)
            return out.reshape(b, h, w, t, d).permute(0, 3, 4, 1, 2)

    # backbone
        def forward(self, x, doy, valid):
            return self.temporal(self.embed(x, doy), valid)

    def permutation_mask(f: torch.Tensor, valid: torch.Tensor, rate: float, generator: torch.Generator | None=None):
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
            return (f, mask)
        flat = f.reshape(-1)
        draw = torch.randint(0, flat.numel(), (n_masked * d * h * w,), device=device, generator=generator)
        out = f.clone()
        out[mask] = flat[draw].reshape(n_masked, d, h, w)
        return (out, mask)
      # nested-tensor fast path is disabled: it warns on padded batches and
    class LinearDecoder(nn.Module):  # takes a different code path from the masked one, which is not worth
        """Deliberately shallow: one linear layer on the feature dimension.  # the speedup here

        A heavier decoder would let the reconstruction be solved without the encoder
        learning anything useful (paper Section III-B2).
        """

        def __init__(self, d_model: int=64, out_ch: int=10):
            super().__init__()
            self.proj = nn.Conv2d(d_model, out_ch, 1)
      # (B,T,d)
        def forward(self, f: torch.Tensor) -> torch.Tensor:
            b, t, d, h, w = f.shape
            y = self.proj(f.reshape(b * t, d, h, w))
            return y.reshape(b, t, -1, h, w)

    def reconstruction_loss(pred, target, date_mask, pixel_valid=None):
        """MSE over masked dates only (paper eq. 3).

        pred/target: (B,T,C,H,W). date_mask: (B,T). pixel_valid: (B,T,H,W) or None.  # guard against all-padded rows
        PASTIS ships no cloud masks, so pixel_valid is normally None here; pass one
        if you pretrain on the authors' unlabeled Zenodo set, which does have MAJA
        validity masks.
        """
        if date_mask.sum() == 0:
            return pred.sum() * 0.0
        p = pred[date_mask]
    # pretext task
        t = target[date_mask]
        if pixel_valid is not None:
            v = pixel_valid[date_mask].unsqueeze(1).float()
            return ((p - t) ** 2 * v).sum() / v.sum().clamp(min=1.0) / p.shape[1]
        return F.mse_loss(p, t)

    class ShallowClassifier(nn.Module):
        """Mean-query attention (TAE-style) collapsing time, then a 1x1 conv.

        Needed because U-BARN keeps the temporal axis, and series lengths vary, so a
        plain linear probe cannot be attached. Following the paper, the value
        projection is the identity (V = X).
        """

        def __init__(self, d_model: int=64, n_classes: int=20):
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
            q = q_all.sum(1) / n_valid
            k = self.k(seq)
            att = (k @ q.unsqueeze(-1)).squeeze(-1) * self.scale
            att = att.masked_fill(pad, float('-inf')).softmax(-1)
            att = torch.nan_to_num(att)
            ctx = (att.unsqueeze(-1) * seq).sum(1)
            ctx = ctx.reshape(b, h, w, d).permute(0, 3, 1, 2)
            return self.out(ctx)

    class SegmentationModel(nn.Module):
        """U-BARN encoder + shallow classifier, in any of the paper's three regimes."""

        def __init__(self, encoder: UBARN, n_classes: int=20, freeze: bool=False):
            super().__init__()
            self.encoder = encoder
            self.head = ShallowClassifier(encoder.d_model, n_classes)
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
        """Accumulates a confusion matrix and derives OA / Kappa / F1 / mIoU."""

        def __init__(self, n_classes: int, ignore_index: int | None=19):
            self.n = n_classes
            self.ignore = ignore_index
            self.cm = torch.zeros(n_classes, n_classes, dtype=torch.long)

        @torch.no_grad()
        def update(self, pred: torch.Tensor, target: torch.Tensor):
            pred = pred.flatten().cpu()
            target = target.flatten().cpu()
    # downstream head
            keep = torch.ones_like(target, dtype=torch.bool)
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
            f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-09)
            iou = tp / (row + col - tp).clamp(min=1)
    # metrics
            return {'OA': oa, 'Kappa': kappa, 'F1': f1[present].mean().item(), 'mIoU': iou[present].mean().item(), 'per_class_f1': f1.tolist(), 'per_class_iou': iou.tolist()}  # master query (N, d)  # (N, T, d)  # (N, d)

    return (
        ConfusionMeter,
        LinearDecoder,
        SegmentationModel,
        UBARN,
        permutation_mask,
        reconstruction_loss,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Splits

    `pastis_official_splits_v1.json` is the only thing that defines the partition.
    The validator checks duplicates, cross-set overlap, that every ID actually
    exists in PASTIS, and whether val/test come from a single official fold.
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
def _(MAN, folds_to_run, required_ids):
    FOLDS = [1, 2]
    RUNS = folds_to_run(MAN, FOLDS)
    NEEDED = sorted(required_ids(RUNS))
    for _fold_key, _sp in RUNS.items():
        print(f'fold {_fold_key}: train={len(_sp['train']):5d}  val={len(_sp['val']):4d} (fold {sorted({MAN.fold_of(i) for i in _sp['val']})})  test={len(_sp['test']):4d} (fold {sorted({MAN.fold_of(i) for i in _sp['test']})})')
    print(f'\nunique patches needed: {len(NEEDED)}')
    return FOLDS, NEEDED, RUNS


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data

    `SUBSET` caps how many patches are fetched — the full set is ~34 GB and molab's
    disk is wiped at the end of the session. Subsetting happens **within** each
    split, so proportions hold and no patch changes side.

    First run: 200 patches, 16 dates. That proves the chain in about ten minutes.
    """)
    return


@app.cell
def _(mo):
    ui_subset = mo.ui.slider(100, 2433, value=600, step=100, label="patches")
    ui_tmax = mo.ui.slider(8, 100, value=40, step=4, label="dates per series")
    ui_crop = mo.ui.dropdown(["16", "32", "64"], value="64", label="crop")
    mo.vstack([ui_subset, ui_tmax, ui_crop])
    return ui_crop, ui_subset, ui_tmax


@app.cell
def _(DataConfig, NEEDED, RUNS, mo, np, ui_crop, ui_subset, ui_tmax):
    cfg = DataConfig(t_max=ui_tmax.value, crop=int(ui_crop.value))
    if ui_subset.value < len(NEEDED):
        rng_pick = np.random.default_rng(cfg.seed)
        keep = set()
        for _fold_key, _sp in RUNS.items():
            for part in ('train', 'val', 'test'):
                ids_part = _sp[part]
                _k = max(1, round(ui_subset.value * len(ids_part) / len(NEEDED)))
                keep.update(rng_pick.choice(ids_part, size=min(_k, len(ids_part)), replace=False).tolist())
        USE_IDS = sorted(keep)
    else:
        USE_IDS = NEEDED
    mo.md(f'\n**{len(USE_IDS)} patches** · download ≈ {len(USE_IDS) * 14.1 / 1024:.1f} GB ·\ncache ≈ {len(USE_IDS) * cfg.t_max * 10 * cfg.crop ** 2 * 2 / 1000000000.0:.1f} GB\n')
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
    mo.stop(not run_cache.value, mo.md('*Press ① to fetch and preprocess.*'))
    RAW_ROOT, source_note = resolve_source(USE_IDS, WORK)
    norm_ids = [i for i in RUNS[FOLDS[0]]['train'] if i in set(USE_IDS)]
    with mo.status.progress_bar(total=len(USE_IDS), title='preprocessing') as bar_cache:
        CACHE_DIR = build_cache(RAW_ROOT, MAN, USE_IDS, cfg, WORK / 'cache', norm_ids=norm_ids, progress=bar_cache.update)
    cache = PastisCache(CACHE_DIR)
    lines = [f'raw data: {source_note}', f'cache: {CACHE_DIR}', f'{len(cache)} patches · T={cache.x.shape[1]} · {cache.x.shape[-1]}px', '']
    for _fold_key, _sp in RUNS.items():
        cov = '  '.join((f'{p}={cache.coverage(_sp[p])[0]}/{cache.coverage(_sp[p])[1]}' for p in ('train', 'val', 'test')))
        lines.append(f'fold {_fold_key}: {cov}')
    mo.md('```\n' + '\n'.join(lines) + '\n```')
    return (cache,)


@app.cell
def _(VOID_CLASS, cache, mo, np, plt, run_cache):
    mo.stop(not run_cache.value)

    pi, ti = 0, min(10, cache.x.shape[1] - 1)
    xv = cache.denormalize(np.asarray(cache.x[pi, ti], dtype=np.float32))
    rgb_img = np.stack([xv[2], xv[1], xv[0]], -1)
    rgb_img = np.clip(rgb_img / max(np.percentile(rgb_img, 98), 1e-6), 0, 1)
    yv = cache.target[pi].astype(float)
    yv[yv == VOID_CLASS] = np.nan

    fig_prev, ax_prev = plt.subplots(1, 2, figsize=(8, 4))
    ax_prev[0].imshow(rgb_img)
    ax_prev[0].set_title(f"patch {cache.ids[pi]} · DOY {cache.doy[pi, ti]}")
    ax_prev[1].imshow(yv, cmap="tab20", vmin=0, vmax=19, interpolation="nearest")
    ax_prev[1].set_title(f"labels · fold {cache.folds[pi]}")
    for a in ax_prev:
        a.axis("off")
    fig_prev.tight_layout()
    fig_prev
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pretraining

    Corrupt a fraction of dates at the encoder output, reconstruct the original
    reflectances. Masked values are **permuted** from elsewhere in the batch rather
    than zeroed — the paper's main departure from SITS-Former.

    One encoder per fold. A fold's own training list already excludes its val and
    test, so per-fold pretraining gets ~1455 patches; a single shared encoder would
    be limited to the intersection across both folds, 968 patches.
    """)
    return


@app.cell
def _(mo):
    ui_mask = mo.ui.slider(0.1, 0.9, value=0.6, step=0.05, label="mask rate")
    ui_pre_epochs = mo.ui.slider(1, 200, value=40, step=1, label="pretrain epochs")
    ui_bs = mo.ui.slider(1, 16, value=4, step=1, label="batch size")
    mo.vstack([ui_mask, ui_pre_epochs, ui_bs])
    return ui_bs, ui_mask, ui_pre_epochs


@app.cell
def _(mo):
    run_pre = mo.ui.run_button(label="② Pretrain (one encoder per fold)")
    run_pre
    return (run_pre,)


@app.cell
def _(
    DEVICE,
    DataLoader,
    FOLDS,
    LinearDecoder,
    MAN,
    NUM_WORKERS,
    N_BANDS,
    RUNS,
    UBARN,
    WORK,
    cache,
    make_dataset,
    mo,
    permutation_mask,
    reconstruction_loss,
    run_pre,
    torch,
    ui_bs,
    ui_mask,
    ui_pre_epochs,
):
    mo.stop(not run_pre.value, mo.md('*Press ② to pretrain. ~1.5–2 h per fold on a GPU at 40 epochs.*'))

    def pretrain_fold(fold_key):
        pool = set(RUNS[fold_key]['train'])
        held = set(RUNS[fold_key]['val']) | set(RUNS[fold_key]['test'])
        idx = cache.indices_for(sorted(pool))
        assert not {cache.ids[i] for i in idx} & held, 'leak'
        if len(idx) == 0:
            raise RuntimeError(f'fold {fold_key}: no cached training patches')
        dl = DataLoader(make_dataset(cache, idx, augment=True), batch_size=ui_bs.value, shuffle=True, num_workers=NUM_WORKERS, pin_memory=DEVICE == 'cuda', drop_last=len(idx) > ui_bs.value)
        torch.manual_seed(0)
        enc = UBARN(in_ch=N_BANDS, d_model=64, d_hidden=128, n_layers=3, n_heads=4).to(DEVICE)
        dec = LinearDecoder(64, N_BANDS).to(DEVICE)
        prm = list(enc.parameters()) + list(dec.parameters())
        opt = torch.optim.Adam(prm, lr=0.001)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
        scl = torch.amp.GradScaler('cuda', enabled=DEVICE == 'cuda')
        out = WORK / f'pretrain_fold{fold_key}_m{int(ui_mask.value * 100)}'
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
                    corrupt, dmask = permutation_mask(feats, valid, ui_mask.value)
                    rec = dec(enc.temporal(corrupt, valid))
                    loss = reconstruction_loss(rec, x, dmask)
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
            sch.step(ep_loss)
            if ep_loss <= min(hist):
                torch.save({'encoder': enc.state_dict(), 'loss': ep_loss, 'epoch': ep, 'mask_rate': ui_mask.value, 'fold': fold_key, 'split_manifest_sha256': MAN.sha256, 'pretrain_ids': sorted(pool)}, out / 'best.pt')
            yield (fold_key, ep, ep_loss, out / 'best.pt', hist, enc, dec)
    CKPTS, HISTORIES = ({}, {})
    VIZ = {}
    total_steps = len(FOLDS) * ui_pre_epochs.value
    with mo.status.progress_bar(total=total_steps, title='pretraining') as bar_pre:
        for fk in FOLDS:
            for _fold_key, ep, ep_loss, ckpt, hist, enc_i, dec_i in pretrain_fold(fk):
                bar_pre.update()
            CKPTS[_fold_key] = ckpt
            HISTORIES[f'fold {_fold_key}'] = hist
            VIZ[_fold_key] = (enc_i, dec_i)
            print(f'fold {_fold_key}: best masked MSE {min(hist):.5f} -> {ckpt}')
    mo.md('Pretraining done: ' + ', '.join((f'fold {k}' for k in CKPTS)))
    return CKPTS, HISTORIES, VIZ


@app.cell
def _(HISTORIES, mo, plt, run_pre):
    mo.stop(not run_pre.value)

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
    mo,
    np,
    permutation_mask,
    plt,
    run_pre,
    torch,
    ui_mask,
):
    mo.stop(not run_pre.value)

    # reconstruction on a HELD-OUT patch, so this is a real check not memorisation
    viz_fold = FOLDS[-1]
    enc_v, dec_v = VIZ[viz_fold]
    enc_v.eval(); dec_v.eval()
    held_ids = sorted(set(RUNS[viz_fold]["val"]) | set(RUNS[viz_fold]["test"]))
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
        return np.clip(img / max(np.percentile(img, 98), 1e-6), 0, 1)

    picks = torch.nonzero(mv[0]).flatten().cpu().numpy()[:6]
    fig_rec, ax_rec = plt.subplots(2, len(picks), figsize=(2.1 * len(picks), 4.4),
                                   squeeze=False)
    for col, tt in enumerate(picks):
        ax_rec[0][col].imshow(to_rgb(xv1[0, tt].cpu().numpy()))
        ax_rec[0][col].set_title(f"DOY {cache.doy[vi, tt]}", fontsize=8)
        ax_rec[1][col].imshow(to_rgb(rv[0, tt].float().cpu().numpy()))
        ax_rec[0][col].axis("off"); ax_rec[1][col].axis("off")
    fig_rec.suptitle(f"held-out patch {cache.ids[vi]} (fold {viz_fold}) — "
                     f"top: masked input · bottom: reconstruction", fontsize=9)
    fig_rec.tight_layout()
    fig_rec
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Downstream

    **FR** frozen encoder · **FT** fine-tuned · **e2e** random init, same
    architecture. Run both folds and average.
    """)
    return


@app.cell
def _(mo):
    ui_dn_epochs = mo.ui.slider(5, 150, value=40, step=5, label="downstream epochs")
    ui_ntrain = mo.ui.dropdown(["all", "10", "30", "100", "200"], value="all",
                               label="training patches per fold")
    mo.vstack([ui_dn_epochs, ui_ntrain])
    return ui_dn_epochs, ui_ntrain


@app.cell
def _(mo):
    run_dn = mo.ui.run_button(label="③ Run both folds (FR / FT / e2e)")
    run_dn
    return (run_dn,)


@app.cell
def _(
    CKPTS,
    ConfusionMeter,
    DEVICE,
    DataLoader,
    FOLDS,
    NUM_WORKERS,
    N_BANDS,
    N_CLASSES,
    RUNS,
    SegmentationModel,
    UBARN,
    VOID_CLASS,
    WORK,
    cache,
    json,
    make_dataset,
    mo,
    np,
    run_dn,
    scarce_subset,
    torch,
    ui_bs,
    ui_dn_epochs,
    ui_ntrain,
):
    mo.stop(not run_dn.value, mo.md('*Press ③ to train and evaluate.*'))

    def build_model(mode, fold_key):
        enc = UBARN(in_ch=N_BANDS, d_model=64, d_hidden=128, n_layers=3, n_heads=4)
        if mode in ('FR', 'FT'):
            enc.load_state_dict(torch.load(CKPTS[fold_key], map_location='cpu')['encoder'])
        return SegmentationModel(enc, n_classes=N_CLASSES, freeze=mode == 'FR').to(DEVICE)

    def fold_loaders(fold_key, n_train, batch_size):
        sp = RUNS[fold_key]
        tr = cache.indices_for(sp['train'])
        va = cache.indices_for(sp['val'])
        te = cache.indices_for(sp['test'])
        if n_train is not None:
            tr = scarce_subset(cache, tr, n_train)
        mk = lambda idx, shuf, aug: DataLoader(make_dataset(cache, idx, augment=aug), batch_size=batch_size, shuffle=shuf, num_workers=NUM_WORKERS, pin_memory=DEVICE == 'cuda')
        return (mk(tr, True, True), mk(va, False, False), mk(te, False, False), len(tr), len(te))

    @torch.no_grad()
    def evaluate(model, dl):
        model.eval()
        meter = ConfusionMeter(N_CLASSES, ignore_index=VOID_CLASS)
        for b in dl:
            with torch.amp.autocast('cuda', enabled=DEVICE == 'cuda'):
                lg = model(b['x'].to(DEVICE), b['doy'].to(DEVICE), b['valid'].to(DEVICE))
            meter.update(lg.argmax(1), b['y'])
        return meter

    def train_head(model, tr_dl, va_dl, epochs, bar=None):
        prm = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(prm, lr=0.001)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
        scl = torch.amp.GradScaler('cuda', enabled=DEVICE == 'cuda')
        lossf = torch.nn.CrossEntropyLoss(ignore_index=VOID_CLASS)
        best, best_state = (-1.0, None)
        for _ in range(epochs):
            model.train()
            tot = 0.0
            for b in tr_dl:
                with torch.amp.autocast('cuda', enabled=DEVICE == 'cuda'):
                    lg = model(b['x'].to(DEVICE), b['doy'].to(DEVICE), b['valid'].to(DEVICE))
                    loss = lossf(lg, b['y'].to(DEVICE))
                opt.zero_grad(set_to_none=True)
                scl.scale(loss).backward()
                scl.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(prm, 5.0)
                scl.step(opt)
                scl.update()
                tot += float(loss.detach())
            sch.step(tot / max(len(tr_dl), 1))
            miou = evaluate(model, va_dl).scores()['mIoU']
            if miou > best:
                best = miou
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if bar is not None:
                bar.update()
        if best_state:
            model.load_state_dict(best_state)
        return best
    N_TRAIN = None if ui_ntrain.value == 'all' else int(ui_ntrain.value)
    REGIMES = ['FR', 'FT', 'e2e'] if CKPTS else ['e2e']
    results = []
    with mo.status.progress_bar(total=len(FOLDS) * len(REGIMES) * ui_dn_epochs.value, title='downstream') as bar_dn:
        for _fold_key in FOLDS:
            _tr_dl, _va_dl, _te_dl, _n_tr, n_te = fold_loaders(_fold_key, N_TRAIN, ui_bs.value)
            for _mode in REGIMES:
                torch.manual_seed(_fold_key)
                _m = build_model(_mode, _fold_key)
                train_head(_m, _tr_dl, _va_dl, ui_dn_epochs.value, bar=bar_dn)
                _sc = evaluate(_m, _te_dl).scores()
                results.append({'fold': _fold_key, 'regime': _mode, 'n_train': _n_tr, 'n_test': n_te, **{k: _sc[k] for k in ('Kappa', 'OA', 'F1', 'mIoU')}})
                del _m
                if DEVICE == 'cuda':
                    torch.cuda.empty_cache()
    with open(WORK / 'results_2fold.json', 'w') as _fh:
        json.dump(results, _fh, indent=2)
    rows = ['| fold | regime | train | test | Kappa | OA | F1 | mIoU |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        rows.append(f'| {r['fold']} | {r['regime']} | {r['n_train']} | {r['n_test']} | {r['Kappa']:.4f} | {r['OA']:.4f} | {r['F1']:.4f} | {r['mIoU']:.4f} |')
    rows.append('')
    rows.append('**Mean across folds**')
    rows.append('')
    rows.append('| regime | Kappa | OA | F1 | mIoU |')
    rows.append('|---|---:|---:|---:|---:|')
    for _mode in REGIMES:
        sel = [r for r in results if r['regime'] == _mode]
        cellsm = []
        for _k in ('Kappa', 'OA', 'F1', 'mIoU'):
            v = np.array([r[_k] for r in sel])
            cellsm.append(f'{v.mean():.4f} ±{v.std():.4f}')
        rows.append(f'| {_mode} | ' + ' | '.join(cellsm) + ' |')
    mo.md('\n'.join(rows))
    return build_model, evaluate, fold_loaders, train_head


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Label-scarcity sweep

    The experiment that decides whether the pretraining did anything. At full label
    count FT and e2e land on top of each other — that is the paper's own result.
    The separation appears at small training sizes.
    """)
    return


@app.cell
def _(mo):
    ui_sizes = mo.ui.multiselect(["10", "20", "30", "50", "100"],
                                 value=["10", "30", "100"], label="training sizes")
    ui_sweep_epochs = mo.ui.slider(5, 100, value=30, step=5, label="epochs per run")
    run_sweep = mo.ui.run_button(label="④ Run scarcity sweep")
    mo.vstack([ui_sizes, ui_sweep_epochs, run_sweep])
    return run_sweep, ui_sizes, ui_sweep_epochs


@app.cell
def _(
    CKPTS,
    DEVICE,
    FOLDS,
    WORK,
    build_model,
    evaluate,
    fold_loaders,
    json,
    mo,
    np,
    plt,
    run_sweep,
    torch,
    train_head,
    ui_bs,
    ui_sizes,
    ui_sweep_epochs,
):
    mo.stop(not run_sweep.value, mo.md('*Press ④. This is sizes × 2 regimes × 2 folds runs.*'))
    mo.stop(not CKPTS, mo.md('**Pretrain first** — the sweep compares FT against e2e.'))
    sizes = sorted((int(s) for s in ui_sizes.value))
    sweep = []
    with mo.status.progress_bar(total=len(sizes) * len(FOLDS) * 2 * ui_sweep_epochs.value, title='scarcity sweep') as bar_sw:
        for size in sizes:
            for _fold_key in FOLDS:
                _tr_dl, _va_dl, _te_dl, _n_tr, _ = fold_loaders(_fold_key, size, ui_bs.value)
                for _mode in ['FT', 'e2e']:
                    torch.manual_seed(_fold_key)
                    _m = build_model(_mode, _fold_key)
                    train_head(_m, _tr_dl, _va_dl, ui_sweep_epochs.value, bar=bar_sw)
                    _sc = evaluate(_m, _te_dl).scores()
                    sweep.append({'n': _n_tr, 'fold': _fold_key, 'regime': _mode, **{k: _sc[k] for k in ('Kappa', 'OA', 'F1', 'mIoU')}})
                    del _m
                    if DEVICE == 'cuda':
                        torch.cuda.empty_cache()
    with open(WORK / 'sweep_2fold.json', 'w') as _fh:
        json.dump(sweep, _fh, indent=2)
    fig_sw, axs_sw = plt.subplots(1, 4, figsize=(14, 3.2))
    for ax_s, metric in zip(axs_sw, ['Kappa', 'OA', 'F1', 'mIoU']):
        for _mode, style in [('FT', '-o'), ('e2e', '--s')]:
            xs = sorted({r['n'] for r in sweep if r['regime'] == _mode})
            mu = [float(np.mean([r[metric] for r in sweep if r['regime'] == _mode and r['n'] == x])) for x in xs]
            sd = [float(np.std([r[metric] for r in sweep if r['regime'] == _mode and r['n'] == x])) for x in xs]
            if xs:
                ax_s.errorbar(xs, mu, yerr=sd, fmt=style, capsize=3, ms=4, label=_mode)
        ax_s.set_xscale('log')
        ax_s.set_xlabel('training patches')
        ax_s.set_title(metric)
        ax_s.grid(alpha=0.3)
    axs_sw[0].legend(fontsize=8)
    fig_sw.tight_layout()
    fig_sw
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Save before the session ends

    molab sessions stop after 12 hours, or 90 minutes idle, and the disk goes with
    them. Nothing under `WORK` survives.

    ```python
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_file(path_or_fileobj=str(WORK / "results_2fold.json"),
                    path_in_repo="results_2fold.json",
                    repo_id="<you>/ubarn-pastis", repo_type="model", token="hf_...")
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
