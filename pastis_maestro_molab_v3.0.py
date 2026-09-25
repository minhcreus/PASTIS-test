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
    # MAESTRO on PASTIS · v3.0.2

    Masked-autoencoder pretraining (Labatie et al. 2025, arXiv 2508.10894) on Sentinel-2 time series; label efficiency under a fixed protocol: fold 4 validation, LP / FT / SL on identical nested subsets, mean ± std over seeds.

    Run ① → ② → ③.
    """)
    return


@app.cell
def _():
    VERSION = "3.0.2"
    import time as time_mod
    SESSION_T0 = time_mod.time()
    RESULTS_VERSION = ".".join(VERSION.split(".")[:2])

    import copy, json, math, os, sys, urllib.request
    from dataclasses import dataclass, asdict
    from datetime import datetime
    from pathlib import Path

    import numpy as np

    MISSING = [pip for mod, pip in [("torch", "torch"), ("matplotlib", "matplotlib"),
                                    ("huggingface_hub", "huggingface-hub")]
               if __import__("importlib.util", fromlist=["util"]).find_spec(mod) is None]
    if MISSING:
        os.system(f"{sys.executable} -m pip install -q " + " ".join(MISSING))

    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    WORK = Path.home() / "pastis_ssl"
    WORK.mkdir(parents=True, exist_ok=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    AMP_DTYPE = (torch.bfloat16 if DEVICE == "cuda" and torch.cuda.is_bf16_supported()
                 else torch.float16)
    AMP_ON = DEVICE == "cuda"
    FUSED = DEVICE == "cuda"

    print("version:", VERSION)
    print("torch  :", torch.__version__)
    print("device :", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "(no GPU)")
    print("work   :", WORK)
    return (
        AMP_DTYPE,
        AMP_ON,
        DEVICE,
        F,
        FUSED,
        Path,
        RESULTS_VERSION,
        SESSION_T0,
        VERSION,
        WORK,
        asdict,
        copy,
        dataclass,
        datetime,
        json,
        math,
        nn,
        np,
        os,
        plt,
        time_mod,
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

    def _is_rate_limit(err) -> bool:
        msg = str(err)
        return '429' in msg or 'Too Many Requests' in msg or 'rate limit' in msg.lower()

    def _with_backoff(fn, retries=8, base_wait=30, log=print, what='request'):
        """Retry fn() on HTTP 429 with exponential backoff; re-raise anything else."""
        import time
        for attempt in range(retries):
            try:
                return fn()
            except Exception as err:
                if not _is_rate_limit(err) or attempt == retries - 1:
                    raise
                wait = min(600, base_wait * 2 ** attempt)
                log(f'rate limited on {what}; waiting {wait}s (attempt {attempt + 1}/{retries})')
                time.sleep(wait)

    def _hf_prefix(token=None, log=print) -> str:
        from huggingface_hub import list_repo_files
        files = _with_backoff(lambda: list_repo_files(HF_REPO, repo_type='dataset', token=token), log=log, what='file listing')
        for f in files:
            if 'DATA_S2/' in f:
                return f.split('DATA_S2/')[0]
        raise FileNotFoundError(f'DATA_S2 not found in {HF_REPO}')

    def download_patches(ids, cache_dir: Path, workers: int=4, token=None, batch: int=400, base_wait: int=30, log=print, progress=None) -> Path:
        """Fetch S2 series and annotations for the given IDs.

        In batches, with backoff on HTTP 429. Files already in the cache are not
        refetched, so an interrupted download resumes where it stopped. Xet
        transfers are disabled: each file otherwise costs an extra API call for a
        read token, which is what exhausts the anonymous rate limit.
        """
        import os
        os.environ['HF_HUB_DISABLE_XET'] = '1'
        try:
            import huggingface_hub.constants as hf_const
            hf_const.HF_HUB_DISABLE_XET = True
        except Exception:
            pass
        from huggingface_hub import snapshot_download
        ids = list(ids)
        prefix = _hf_prefix(token, log)
        local = None
        for i in range(0, len(ids), batch):
            chunk = ids[i:i + batch]
            patterns = []
            for pid in chunk:
                patterns.append(f'{prefix}DATA_S2/S2_{pid}.npy')
                patterns.append(f'{prefix}ANNOTATIONS/TARGET_{pid}.npy')
            local = _with_backoff(lambda: snapshot_download(HF_REPO, repo_type='dataset', allow_patterns=patterns, cache_dir=str(cache_dir / 'hf'), max_workers=workers, token=token), base_wait=base_wait, log=log, what=f'batch {i // batch + 1}/{-(-len(ids) // batch)}')
            if progress is not None:
                progress(len(chunk))
        return Path(local) / prefix if prefix else Path(local)

    def resolve_source(ids, cache_dir: Path, local_hint: str | None=None, token=None, log=print, progress=None):
        """(raw_root, description): local copy if present, else download."""
        local = find_local_pastis(local_hint)
        if local is not None:
            return (local, f'local copy at {local}')
        return (download_patches(ids, cache_dir, token=token, log=log, progress=progress), 'Hugging Face mirror')

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
        q05 = np.asarray(stats['q05'], dtype=np.float32)[None, :, None, None]
        q95 = np.asarray(stats['q95'], dtype=np.float32)[None, :, None, None]
        med = np.asarray(stats['median'], dtype=np.float32)[None, :, None, None]
        x = np.clip(arr.astype(np.float32), q05, q95)
        return (x - med) / np.maximum(q95 - q05, 1e-06)
    # preprocessing

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

    def make_dataset(cache: PastisCache, indices, augment: bool=False, patch=None, random_crop: bool=False, flips: bool=True):
        """patch: network input size; cropped randomly (training) or centrally (eval)
        from the cached patch. flips: dihedral augmentation when augment=True."""
        import torch
        from torch.utils.data import Dataset
        full = cache.x.shape[-1]
        ps = patch or full

        class _DS(Dataset):

            def __init__(self):
                self.idx = np.asarray(indices)

            def __len__(self):
                return len(self.idx)

            def __getitem__(self, i):
                j = int(self.idx[i])
                if ps < full:
                    if random_crop:
                        r0, c0 = np.random.randint(0, full - ps + 1, size=2)
                    else:
                        r0 = c0 = (full - ps) // 2
                    x = np.asarray(cache.x[j, ..., r0:r0 + ps, c0:c0 + ps], dtype=np.float32)
    # cache handle
                    y = np.asarray(cache.target[j, r0:r0 + ps, c0:c0 + ps], dtype=np.int64)
                else:
                    x = np.asarray(cache.x[j], dtype=np.float32)
                    y = np.asarray(cache.target[j], dtype=np.int64)
                if augment and flips:
                    if np.random.rand() < 0.5:
                        x, y = (x[..., ::-1], y[..., ::-1])
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
        resolve_source,
    )


@app.cell
def _(math, nn, torch):
    S2_GROUPS = ((0, 4), (4, 8), (8, 10))
    PRESETS = {'tiny': dict(dim=192, depth=12, heads=3, dec_dim=128, dec_depth=4, dec_heads=4), 'small': dict(dim=384, depth=12, heads=6, dec_dim=256, dec_depth=4, dec_heads=8), 'base': dict(dim=768, depth=12, heads=12, dec_dim=512, dec_depth=8, dec_heads=16)}

    def select_bins(x, D, train, gen=None):
        """Indices of D time steps: truncate to D*k consecutive steps, one step per bin.
        Train: random step per bin. Eval: step with the lowest mean absolute deviation
        to the bin's pixel-wise median (paper Sec. 6.2.1). x: (T, ...)."""  # B02-B05, B06-B8A, B11-B12 (paper Tab. 2)
        T = x.shape[0]
        k = T // D
        if k == 0:
            return torch.linspace(0, T - 1, D).round().long()
        L = k * D
        start = int(torch.randint(0, T - L + 1, (1,), generator=gen)) if train else (T - L) // 2
        if train:
            pick = torch.randint(0, k, (D,), generator=gen)
        else:
            xb = x[start:start + L].reshape(D, k, -1).float()
            mad = (xb - xb.median(dim=1, keepdim=True).values).abs().mean(-1)
            pick = mad.argmin(1).cpu()
        return start + torch.arange(D) * k + pick

    def temporal_features(doy, days_since_ref):
        """8 features per step: two DOY harmonics and the offset from a reference date (x4)."""
        d = doy / 365.25
        r = (days_since_ref / 365.25).unsqueeze(-1).expand(*doy.shape, 4)
        return torch.cat([torch.stack([torch.sin(2 * math.pi * d), torch.cos(2 * math.pi * d), torch.sin(4 * math.pi * d), torch.cos(4 * math.pi * d)], -1), r], -1)

    def sincos_2d(dim, g):
        assert dim % 4 == 0
        y, x = torch.meshgrid(torch.arange(g, dtype=torch.float32), torch.arange(g, dtype=torch.float32), indexing='ij')
        om = 1.0 / 10000 ** (torch.arange(dim // 4, dtype=torch.float32) / (dim // 4))
        out = [f(c.flatten()[:, None] * om[None]) for c in (y, x) for f in (torch.sin, torch.cos)]
        return torch.cat(out, 1)

    def patchify(x, P):
        """(B, D, C, H, W) -> (B, D, S, P*P*C), S = (H/P)*(W/P)."""
        B, D, C, H, W = x.shape
        g = H // P
        x = x.reshape(B, D, C, g, P, g, P).permute(0, 1, 3, 5, 4, 6, 2)
        return x.reshape(B, D, g * g, P * P * C)

    def build_mask(B, D, S, ratio=0.75, p_space=0.25, p_time=0.25, p_block=0.0, block_frac=(0.25, 0.5), device='cpu', gen=None):
        """Structured masking (spatial tubes, whole time steps, optional contiguous time
        block) adjusted to an exact overall ratio (paper Sec. 6.3.1). Returns masked
        (B, D*S) bool and keep indices (B, n_keep)."""
        rnd = lambda *s: torch.rand(*s, device=device, generator=gen)
        m = (rnd(B, 1, S) < p_space) | (rnd(B, D, 1) < p_time)
        if p_block > 0:
            lo, hi = block_frac
            span = (lo + (hi - lo) * rnd(B)) * D
            span = span.round().clamp(1, D).long()
            start = (rnd(B) * (D - span + 1).float()).floor().long()
            t = torch.arange(D, device=device)[None]
            blk = (t >= start[:, None]) & (t < (start + span)[:, None]) & (rnd(B, 1) < p_block)
            m = m | blk[:, :, None]
        m = m.expand(B, D, S).reshape(B, D * S)
        score = m.float() + 0.5 * rnd(B, D * S)
        n_mask = int(round(ratio * D * S))
        order = score.argsort(1, descending=True)
        masked = torch.zeros(B, D * S, dtype=torch.bool, device=device)
        masked.scatter_(1, order[:, :n_mask], True)
        return (masked, order[:, n_mask:])

    def group_norm_targets(t, C, groups=S2_GROUPS, eps=1e-06):
        """Patch-group-wise normalisation of reconstruction targets (paper Sec. 3.2).
        t: (..., P*P*C) laid out as (P*P, C)."""
        s = t.shape
        t = t.reshape(*s[:-1], -1, C)
        out = torch.empty_like(t)
        for a, b in groups:
            g = t[..., a:b]
            mu = g.mean(dim=(-2, -1), keepdim=True)
            sd = g.std(dim=(-2, -1), keepdim=True)
            out[..., a:b] = (g - mu) / (sd + eps)
        return out.reshape(s)

    class Block(nn.Module):

        def __init__(self, dim, heads, mlp=4.0):
            super().__init__()
            self.n1, self.n2 = (nn.LayerNorm(dim), nn.LayerNorm(dim))
            self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
            self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp)), nn.GELU(), nn.Linear(int(dim * mlp), dim))

        def forward(self, x):
            h = self.n1(x)
            x = x + self.attn(h, h, h, need_weights=False)[0]
            return x + self.mlp(self.n2(x))

    class Encoder(nn.Module):
        """Joint-token multispectral tokeniser + early temporal fusion ViT encoder."""

        def __init__(self, in_ch=10, patch=2, crop=16, dim=384, depth=12, heads=6, **_):
            super().__init__()
            self.P, self.C, self.g, self.dim = (patch, in_ch, crop // patch, dim)
            self.embed = nn.Linear(patch * patch * in_ch, dim)
            self.register_buffer('pos', sincos_2d(dim - 8, self.g), persistent=False)
            self.te = nn.Linear(8, 8, bias=False)
            nn.init.eye_(self.te.weight)
            self.blocks = nn.ModuleList((Block(dim, heads) for _ in range(depth)))
            self.norm = nn.LayerNorm(dim)

        def tokens(self, x, tfeat):
            B, D = x.shape[:2]
            S = self.g * self.g
            tok = self.embed(patchify(x, self.P))
            pe = torch.cat([self.pos[None, None].expand(B, D, S, -1), self.te(tfeat)[:, :, None].expand(B, D, S, 8)], -1)
            return (tok + pe).reshape(B, D * S, self.dim)

        def forward(self, x, tfeat, keep=None):
            z = self.tokens(x, tfeat)
            if keep is not None:
                z = torch.gather(z, 1, keep[..., None].expand(-1, -1, self.dim))
            for b in self.blocks:
                z = b(z)
            return self.norm(z)

    class MAE(nn.Module):

        def __init__(self, in_ch=10, patch=2, crop=16, preset='small', **overrides):
            super().__init__()
            cfg = {**PRESETS[preset], **overrides}
            self.encoder = Encoder(in_ch, patch, crop, cfg['dim'], cfg['depth'], cfg['heads'])
            dd = cfg['dec_dim']
            self.dec_embed = nn.Linear(cfg['dim'], dd)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, dd))
            nn.init.normal_(self.mask_token, std=0.02)
            self.register_buffer('dpos', sincos_2d(dd - 8, crop // patch), persistent=False)
            self.dec_te = nn.Linear(8, 8, bias=False)
            nn.init.eye_(self.dec_te.weight)
            self.dec = nn.ModuleList((Block(dd, cfg['dec_heads']) for _ in range(cfg['dec_depth'])))
            self.dec_norm = nn.LayerNorm(dd)
            self.pred = nn.Linear(dd, patch * patch * in_ch)

        def forward(self, x, tfeat, masked, keep):
            B, D = x.shape[:2]
            S = self.encoder.g ** 2
            z = self.dec_embed(self.encoder(x, tfeat, keep))
            full = self.mask_token.to(z.dtype).expand(B, D * S, -1).clone()
            full.scatter_(1, keep[..., None].expand(-1, -1, z.shape[-1]), z)
            pe = torch.cat([self.dpos[None, None].expand(B, D, S, -1), self.dec_te(tfeat)[:, :, None].expand(B, D, S, 8)], -1)
            h = full + pe.reshape(B, D * S, -1)
            for b in self.dec:
                h = b(h)
            return self.pred(self.dec_norm(h))

        def loss(self, x, tfeat, masked, keep, groups=S2_GROUPS):
            pred = self(x, tfeat, masked, keep)
            tgt = group_norm_targets(patchify(x, self.encoder.P).flatten(1, 2), self.encoder.C, groups)
            err = (pred.float() - tgt.float()).abs().mean(-1)
            return (err * masked).sum() / masked.sum().clamp(min=1)

    class Segmenter(nn.Module):
        """Per spatial token: attentive pooling over time, then a dense layer to
        class logits for each of its P*P pixels (paper Sec. 3.3)."""

        def __init__(self, encoder, n_classes, heads=None, freeze=False):
            super().__init__()
            self.encoder, self.freeze, self.K = (encoder, freeze, n_classes)
            d = encoder.dim
            self.query = nn.Parameter(torch.zeros(1, 1, d))
            nn.init.normal_(self.query, std=0.02)
            self.pn = nn.LayerNorm(d)
            self.pool = nn.MultiheadAttention(d, heads or max(1, d // 64), batch_first=True)
            self.out = nn.Linear(d, n_classes * encoder.P ** 2)
            if freeze:
                for p in self.encoder.parameters():
                    p.requires_grad_(False)

        def train(self, mode=True):
            super().train(mode)
            if self.freeze:
                self.encoder.eval()
            return self

        def forward(self, x, tfeat):
            B, D = x.shape[:2]
            g, P, d = (self.encoder.g, self.encoder.P, self.encoder.dim)
            if self.freeze:
                with torch.no_grad():
                    z = self.encoder(x, tfeat)
            else:
                z = self.encoder(x, tfeat)
            z = self.pn(z.reshape(B, D, g * g, d).transpose(1, 2).reshape(B * g * g, D, d))
            pooled = self.pool(self.query.expand(B * g * g, 1, d), z, z, need_weights=False)[0]
            lg = self.out(pooled.squeeze(1)).reshape(B, g, g, P, P, self.K)
            return lg.permute(0, 5, 1, 3, 2, 4).reshape(B, self.K, g * P, g * P)

    def tile_to_crops(x, crop):
        """(..., H, W) -> (n, ..., crop, crop) non-overlapping, row-major."""
        H, W = x.shape[-2:]
        gh, gw = (H // crop, W // crop)
        lead = x.shape[:-2]
        x = x.reshape(*lead, gh, crop, gw, crop)
        nd = len(lead)
        perm = (nd, nd + 2) + tuple(range(nd)) + (nd + 1, nd + 3)
        return x.permute(*perm).reshape(gh * gw, *lead, crop, crop)

    def crops_to_tile(c, H, W):
        """Inverse of tile_to_crops: (n, ..., crop, crop) -> (..., H, W)."""
        crop = c.shape[-1]
        gh, gw = (H // crop, W // crop)
        lead = c.shape[1:-2]
        nd = len(lead)
        c = c.reshape(gh, gw, *lead, crop, crop)
        perm = tuple(range(2, 2 + nd)) + (0, 2 + nd, 1, 3 + nd)
        return c.permute(*perm).reshape(*lead, H, W)

    class EpochEMA:
        """Per-epoch weight EMA, alpha = 1 - 1/(0.2 * n_epochs) (paper Sec. 6.2.4)."""

        def __init__(self, model, n_epochs):
            self.alpha = 1.0 - 1.0 / max(1.0, 0.2 * n_epochs)
            self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

        @torch.no_grad()
        def update(self, model):
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    self.shadow[k].mul_(self.alpha).add_(v.detach(), alpha=1 - self.alpha)
                else:
                    self.shadow[k].copy_(v)

    class ConfusionMeter:
        """Confusion matrix -> OA / Kappa / F1 / mIoU."""

        def __init__(self, n_classes: int, ignore_index: int | None=19):
            self.n = n_classes
            self.ignore = ignore_index
            self.cm = torch.zeros(n_classes, n_classes, dtype=torch.long)

        @torch.no_grad()
        def update(self, pred: torch.Tensor, target: torch.Tensor):
            pred = pred.flatten()
            target = target.flatten().to(pred.device)
            nn2 = self.n * self.n
            idx = target * self.n + pred
            if self.ignore is not None:
                idx = torch.where(target == self.ignore, torch.full_like(idx, nn2), idx)
            cm = torch.zeros(nn2 + 1, dtype=torch.long, device=pred.device)
            cm.scatter_add_(0, idx, torch.ones_like(idx))
            cm = cm[:nn2].reshape(self.n, self.n)
            if self.cm.device != cm.device:
                self.cm = self.cm.to(cm.device)
            self.cm += cm

        @staticmethod
        def _metrics(cm, cls):
            """Macro metrics over classes `cls`; rows outside `cls` are dropped entirely,
            so pixels of other classes neither count nor penalise."""
            cm = cm.clone()
            keep_rows = torch.zeros(cm.shape[0], dtype=torch.bool)
            keep_rows[cls] = True
            cm[~keep_rows] = 0
            total = cm.sum().clamp(min=1)
            tp = cm.diag()
            oa = (tp[cls].sum() / total).item()
            row, col = (cm.sum(1), cm.sum(0))
            pe = ((row * col).sum() / (total * total)).item()
            kappa = (oa - pe) / (1 - pe) if pe < 1 else 0.0
            present = torch.zeros_like(keep_rows)
            present[cls] = row[cls] > 0
            prec = tp / col.clamp(min=1)
            rec = tp / row.clamp(min=1)
            f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-09)
            iou = tp / (row + col - tp).clamp(min=1)
            nan = torch.tensor(float('nan'))
            return {'OA': oa, 'Kappa': kappa, 'mIoU': iou[present].mean().item(), 'mF1': f1[present].mean().item(), 'per_class_iou': torch.where(present, iou, nan).tolist(), 'per_class_f1': torch.where(present, f1, nan).tolist(), 'per_class_precision': torch.where(present, prec, nan).tolist(), 'per_class_recall': torch.where(present, rec, nan).tolist()}

        def scores(self) -> dict:
            """Two conventions.

            Plain keys: U-TAE / leaderboard — void ignored, background is a class.
            `*_crop` keys: Dumeur et al. Table IV — only pixels labelled with one of the
            18 crop classes count; predicting background on a crop pixel is an error.
            """
            cm = self.cm.float().cpu()
            if self.ignore is not None:
                cm = cm.clone()
                cm[self.ignore] = 0
                cm[:, self.ignore] = 0
            n = cm.shape[0]
            all_cls = [c for c in range(n) if c != self.ignore]
            crop_cls = [c for c in all_cls if c != 0]
            a = self._metrics(cm, all_cls)
            c = self._metrics(cm, crop_cls)
            out = dict(a)
            out['F1'] = a['mF1']
            out.update({f'{k}_crop': v for k, v in c.items() if not k.startswith('per_class')})
            for k in ('iou', 'f1', 'precision', 'recall'):
                out[f'per_class_{k}_crop'] = c[f'per_class_{k}']
            out['confusion'] = self.cm.cpu().tolist()
            return out

    return (
        ConfusionMeter,
        Encoder,
        EpochEMA,
        MAE,
        PRESETS,
        Segmenter,
        build_mask,
        crops_to_tile,
        select_bins,
        temporal_features,
        tile_to_crops,
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
    FOLDS = [5]  # test fold 5, val 4, train 1-3 = MAESTRO fold I
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
    """)
    return


@app.cell
def _(mo):
    MAESTRO_EP = {"ft": {5: 200, 20: 100, 100: 50}, "lp": {5: 40, 20: 20, 100: 10}}
    CFG_PRESETS = {
        "MAESTRO · Tiny encoder (full grid in ~1 session/fold)": dict(model="tiny"),
        "MAESTRO · Small encoder": dict(model="small"),
        "MAESTRO · Base encoder (paper; single-seed anchor runs)": dict(model="base"),
    }
    COMMON = dict(crop=16, patch=2, bins=16, mask=0.75, p_space=0.25, p_time=0.25, p_block=0.5,
                  block_lo=0.25, block_hi=0.5, pre_epochs=100, pre_bs=72, pre_lr=3e-5,
                  dn_bs=48, ft_lr=1e-5, lp_lr=1e-5, ft_final_div=2.0, pre_final_div=1e4,
                  wd=0.01, warm_frac=0.2, val_every=5, patience_frac=0.3, min_frac=0.5,
                  min_delta=0.002, val_cap=150, t_max=61)
    ui_preset = mo.ui.dropdown(list(CFG_PRESETS), value=list(CFG_PRESETS)[0], label="configuration")
    ui_preset
    return CFG_PRESETS, COMMON, MAESTRO_EP, ui_preset


@app.cell
def _(CFG_PRESETS, COMMON, mo, ui_preset):
    P = {**COMMON, **CFG_PRESETS[ui_preset.value]}
    ui_subset = mo.ui.slider(100, 2433, value=2433, step=1, label="patches")
    ui_tmax = mo.ui.slider(16, 100, value=P["t_max"], step=1, label="max dates cached per series")
    ui_window = mo.ui.dropdown(["full series (Sep 2018-Nov 2019)", "Jan-Nov 2019"],
                               value="full series (Sep 2018-Nov 2019)", label="date window")
    ui_hf_token = mo.ui.text(kind="password", label="Hugging Face token (optional)", full_width=True)
    mo.vstack([ui_subset, ui_tmax, ui_window, ui_hf_token])
    return P, ui_hf_token, ui_subset, ui_tmax, ui_window


@app.cell
def _(
    DataConfig,
    NEEDED,
    P,
    RUNS,
    WORK,
    mo,
    np,
    ui_subset,
    ui_tmax,
    ui_window,
):
    FULL_SERIES = ui_window.value.startswith('full')
    cfg = DataConfig(t_max=ui_tmax.value, crop=128, date_start='2018-09-01' if FULL_SERIES else '2019-01-01', date_end='2019-11-30')
    CROP, PATCH_SZ, BINS = (P['crop'], P['patch'], P['bins'])
    REP = (128 // CROP) ** 2
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
    import shutil as _sh
    cache_gb = len(USE_IDS) * cfg.t_max * 10 * 128 ** 2 * 2 / 1000000000.0
    free_gb = _sh.disk_usage(WORK).free / 1000000000.0
    mo.md(f'**{len(USE_IDS)} patches** · download ≈ {len(USE_IDS) * 14.1 / 1024:.1f} GB · cache ≈ {cache_gb:.1f} GB · crops {CROP}² · {BINS} temporal bins' + (f' · **disk: need ~{cache_gb:.0f} GB, {free_gb:.0f} GB free**' if cache_gb > 0.9 * free_gb else ''))
    return BINS, CROP, PATCH_SZ, REP, USE_IDS, cfg


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
    os,
    resolve_source,
    run_cache,
    ui_hf_token,
):
    CACHE_TAG = WORK / 'cache' / f'{cfg.tag()}_n{len(USE_IDS)}'
    CACHE_READY = (CACHE_TAG / 'manifest.json').exists()
    mo.stop(not run_cache.value and (not CACHE_READY), mo.md('*Press ① to fetch and preprocess.*'))
    # button-or-on-disk: run_button resets to False, so gating on it alone would
    # wipe `cache` on any upstream change
    HF_TOKEN = ui_hf_token.value.strip() or os.environ.get('HF_TOKEN') or None
    if CACHE_READY:
        RAW_ROOT, source_note = (CACHE_TAG, 'cache already on disk')
    else:
        with mo.status.progress_bar(total=len(USE_IDS), title='downloading patches', subtitle='authenticated' if HF_TOKEN else 'anonymous') as bar_dl:
            RAW_ROOT, source_note = resolve_source(USE_IDS, WORK, token=HF_TOKEN, log=lambda m: bar_dl.update(increment=0, subtitle=m), progress=lambda k: bar_dl.update(increment=k))
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
def _(
    AMP_DTYPE,
    AMP_ON,
    BINS,
    CROP,
    ConfusionMeter,
    DEVICE,
    N_CLASSES,
    REP,
    VOID_CLASS,
    cache,
    crops_to_tile,
    math,
    mo,
    np,
    select_bins,
    temporal_features,
    tile_to_crops,
    torch,
):
    REF_DOY = 244
    def abs_days(doy_row, valid_row):
        d = doy_row.astype(np.int64).copy()
        out = np.zeros_like(d)
        year = 0
        prev = None
        for t in range(len(d)):
            if not valid_row[t]:
                continue
            if prev is not None and d[t] < prev:
                year += 1
            prev = d[t]
            out[t] = year * 365 + d[t] - REF_DOY
        return out

    T_LEN = cache.valid.sum(1).astype(np.int64)
    DAYS = np.stack([abs_days(cache.doy[i], cache.valid[i]) for i in range(len(cache))])
    TFEAT_ALL = temporal_features(torch.from_numpy(cache.doy.astype(np.float32)),
                                  torch.from_numpy(DAYS.astype(np.float32)))

    X_GPU = None
    store_note = "data path: memory-mapped cache (CPU)"
    if DEVICE == "cuda":
        free_b, total_b = torch.cuda.mem_get_info()
        if cache.x.nbytes < 0.7 * total_b:
            X_GPU = torch.empty(cache.x.shape, dtype=torch.float16, device=DEVICE)
            for i0 in range(0, len(cache), 32):
                X_GPU[i0:i0 + 32] = torch.from_numpy(np.asarray(cache.x[i0:i0 + 32])).to(DEVICE)
            store_note = f"data path: GPU-resident ({cache.x.nbytes / 1e9:.1f} of {total_b / 1e9:.0f} GB)"
    Y_ALL = torch.from_numpy(cache.target.astype(np.int64))
    if X_GPU is not None:
        Y_ALL = Y_ALL.to(DEVICE)
    TFEAT_DEV = TFEAT_ALL.to(DEVICE)


    def tile_x(r):
        t = int(T_LEN[r])
        if X_GPU is not None:
            return X_GPU[r, :t]
        return torch.from_numpy(np.asarray(cache.x[r, :t])).to(DEVICE)


    def batch_draws(rows_b, gen, augment=True):
        """All random choices for one batch, drawn on the CPU in one go."""
        rng = np.random.default_rng(int(torch.randint(0, 2 ** 62, (1,), generator=gen)))
        r = np.asarray(rows_b, dtype=np.int64)
        B = len(r)
        T = T_LEN[r]
        k = np.maximum(T // BINS, 1)
        L = k * BINS
        start = rng.integers(0, np.maximum(T - L, 0) + 1)
        pick = np.floor(rng.random((B, BINS)) * k[:, None]).astype(np.int64)
        idx = start[:, None] + np.arange(BINS)[None] * k[:, None] + pick
        short = T < BINS
        if short.any():
            idx[short] = np.floor(np.arange(BINS)[None] * T[short, None] / BINS).astype(np.int64)
        idx = np.minimum(idx, T[:, None] - 1)
        ij = rng.integers(0, 128 - CROP + 1, size=(B, 2))
        rot = rng.integers(0, 4, size=B) if augment else np.zeros(B, np.int64)
        flip = (rng.random(B) < 0.5).astype(np.int64) if augment else np.zeros(B, np.int64)
        return np.concatenate([r[:, None], ij, rot[:, None], flip[:, None], idx], 1)


    PIX = torch.arange(CROP)
    T_MAX = cache.x.shape[1]


    def assemble(draws, xsrc, ysrc, tsrc, dev):
        """Build a batch from packed draws with a few tensor ops (no per-sample loop)."""
        B = draws.shape[0]
        r, i, j, rot, flip, idx = (draws[:, 0], draws[:, 1], draws[:, 2], draws[:, 3],
                                   draws[:, 4], draws[:, 5:])
        pix = PIX.to(dev)
        lin = (i[:, None, None] + pix[None, :, None]) * 128 + (j[:, None, None] + pix[None, None, :])
        lin = lin.reshape(B, 1, 1, CROP * CROP)
        C = xsrc.shape[2]
        frame = (r[:, None] * T_MAX + idx)[:, :, None, None]
        chan = torch.arange(C, device=dev)[None, None, :, None]
        x = torch.take(xsrc, (frame * C + chan) * (128 * 128) + lin).reshape(B, BINS, C, CROP, CROP).float()
        y = torch.take(ysrc, r[:, None] * (128 * 128) + lin.reshape(B, -1)).reshape(B, CROP, CROP)
        tf = tsrc[r[:, None], idx]
        xr = torch.stack([torch.rot90(x, q, (-2, -1)) for q in range(4)])
        yr = torch.stack([torch.rot90(y, q, (-2, -1)) for q in range(4)])
        ar = torch.arange(B, device=dev)
        x, y = xr[rot, ar], yr[rot, ar]
        f = flip.bool()
        x = torch.where(f[:, None, None, None, None], x.flip(-1), x)
        y = torch.where(f[:, None, None], y.flip(-1), y)
        return x, tf, y


    def train_batch(rows_b, gen, augment=True):
        if X_GPU is not None:
            d = torch.from_numpy(batch_draws(rows_b, gen, augment))
            if DEVICE == "cuda":
                d = d.pin_memory().to(DEVICE, non_blocking=True)
            return assemble(d, X_GPU, Y_ALL, TFEAT_DEV, X_GPU.device)
        return train_batch_slow(rows_b, gen, augment)


    def train_batch_slow(rows_b, gen, augment=True):
        xs, tfs, ys = [], [], []
        for r in rows_b:
            r = int(r)
            i, j = torch.randint(0, 128 - CROP + 1, (2,), generator=gen).tolist()
            xt = tile_x(r)
            idx = select_bins(xt, BINS, True, gen)
            x = xt[idx.to(xt.device), :, i:i + CROP, j:j + CROP].float()
            y = Y_ALL[r, i:i + CROP, j:j + CROP].to(DEVICE)
            if augment:
                k = int(torch.randint(0, 4, (1,), generator=gen))
                if k:
                    x, y = torch.rot90(x, k, (-2, -1)), torch.rot90(y, k, (-2, -1))
                if torch.rand(1, generator=gen).item() < 0.5:
                    x, y = x.flip(-1), y.flip(-1)
            xs.append(x); ys.append(y); tfs.append(TFEAT_DEV[r, idx.to(TFEAT_DEV.device)])
        return torch.stack(xs), torch.stack(tfs), torch.stack(ys)


    def epoch_rows(rows, gen):
        rep = np.repeat(np.asarray(rows), REP)
        return rep[torch.randperm(len(rep), generator=gen).numpy()]


    def tile_crops(r):
        """Non-overlapping crops of one tile with eval bin selection per crop."""
        xt = tile_x(r)
        T = xt.shape[0]
        if T < BINS:
            s0 = 0
            xc = tile_to_crops(xt, CROP)
            n = xc.shape[0]
            idx = (torch.arange(BINS, device=xc.device) * T // BINS)[None].expand(n, -1)
        else:
            k = T // BINS
            L = k * BINS
            s0 = (T - L) // 2
            xc = tile_to_crops(xt[s0:s0 + L], CROP)
            n = xc.shape[0]
            xb = xc.reshape(n, BINS, k, -1).float()
            mad = (xb - xb.median(dim=2, keepdim=True).values).abs().mean(-1)
            idx = s0 + torch.arange(BINS, device=xc.device)[None] * k + mad.argmin(2)
        sel = torch.gather(xc, 1, (idx - s0)[:, :, None, None, None].expand(-1, -1, *xc.shape[2:]))
        tf = TFEAT_DEV[r][idx.to(TFEAT_DEV.device)]
        return sel.float(), tf


    @torch.no_grad()
    def predict_tiles(model, rows_g, bs=512):
        parts = [tile_crops(int(r)) for r in rows_g]
        xc = torch.cat([p_[0] for p_ in parts]); tf = torch.cat([p_[1] for p_ in parts])
        outs = []
        for a in range(0, xc.shape[0], bs):
            with torch.autocast(DEVICE, dtype=AMP_DTYPE, enabled=AMP_ON):
                lg = model(xc[a:a + bs], tf[a:a + bs]).float()
            lg[:, VOID_CLASS] = float("-inf")
            outs.append(lg.argmax(1))
        pred = torch.cat(outs)
        n = pred.shape[0] // len(rows_g)
        return [crops_to_tile(pred[q * n:(q + 1) * n], 128, 128) for q in range(len(rows_g))]


    def predict_tile(model, r):
        return predict_tiles(model, [r])[0]


    def evaluate_rows(model, rows, group=8):
        model.eval()
        meter = ConfusionMeter(N_CLASSES, ignore_index=VOID_CLASS)
        rows = [int(r) for r in rows]
        for a in range(0, len(rows), group):
            g = rows[a:a + group]
            for r, pr in zip(g, predict_tiles(model, g)):
                meter.update(pr, Y_ALL[r].to(DEVICE))
        return meter


    def lr_lambda(total, warm_frac, final_div, start_div=25.0):
        w = max(1, int(total * warm_frac))
        def f(step):
            if step < w:
                return 1.0 / start_div + (1 - 1.0 / start_div) * step / w
            p = min(1.0, (step - w) / max(1, total - w))
            return 1.0 / final_div + (1 - 1.0 / final_div) * 0.5 * (1 + math.cos(math.pi * p))
        return f

    mo.md(store_note)
    return epoch_rows, evaluate_rows, lr_lambda, train_batch


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
    """)
    return


@app.cell
def _(P, mo):
    ui_mask = mo.ui.slider(0.5, 0.9, value=P["mask"], step=0.05, label="mask ratio (paper 0.75)")
    ui_pspace = mo.ui.slider(0.0, 0.5, value=P["p_space"], step=0.05, label="spatial tube masking prob.")
    ui_ptime = mo.ui.slider(0.0, 0.5, value=P["p_time"], step=0.05, label="random time-step masking prob.")
    ui_pblock = mo.ui.slider(0.0, 1.0, value=P["p_block"], step=0.1, label="contiguous time-block prob.")
    ui_pre_epochs = mo.ui.slider(10, 200, value=P["pre_epochs"], step=10, label="pretraining epochs")
    ui_pre_bs = mo.ui.slider(8, 128, value=P["pre_bs"], step=8, label="pretraining batch")
    ui_pre_val = mo.ui.checkbox(False, label="also pretrain on fold-4 images (as MAESTRO; labels never used)")
    mo.vstack([ui_mask, ui_pspace, ui_ptime, ui_pblock, ui_pre_epochs, ui_pre_bs, ui_pre_val])
    return (
        ui_mask,
        ui_pblock,
        ui_pre_bs,
        ui_pre_epochs,
        ui_pre_val,
        ui_pspace,
        ui_ptime,
    )


@app.cell
def _(mo):
    run_pre = mo.ui.run_button(label="② Pretrain (one encoder per fold)")
    run_pre
    return (run_pre,)


@app.cell
def _(
    BINS,
    CACHE_DIR,
    CROP,
    P,
    PATCH_SZ,
    WORK,
    ui_mask,
    ui_pblock,
    ui_pre_bs,
    ui_pre_epochs,
    ui_pre_val,
    ui_pspace,
    ui_ptime,
):
    def ckpt_path(fold_key):
        tag = (f"mae_fold{fold_key}_{CACHE_DIR.name}_{P['model']}_D{BINS}_c{CROP}p{PATCH_SZ}"
               f"_m{int(ui_mask.value * 100)}_s{int(ui_pspace.value * 100)}"
               f"_t{int(ui_ptime.value * 100)}_b{int(ui_pblock.value * 100)}"
               f"_e{ui_pre_epochs.value}_bs{ui_pre_bs.value}{'_v4' if ui_pre_val.value else ''}")
        return WORK / tag / "best.pt"

    return (ckpt_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Continue from a previous session

    A test fold takes longer than one molab session. Before a session ends, download
    the **session bundle** (last cell). In the next session, with the same settings,
    drop it here before pressing ② and ③: the pretrained encoder and finished runs
    are restored, and ③ carries on from the next run.
    """)
    return


@app.cell
def _(mo):
    ui_restore = mo.ui.file(filetypes=[".zip", ".json", ".pt"], multiple=True, kind="area",
                            max_size=2_000_000_000,
                            label="drop a session bundle, results_downstream.json, or weights zip")
    ui_restore
    return (ui_restore,)


@app.cell
def _(FOLDS, MAN, Path, WORK, ckpt_path, json, mo, torch, ui_restore):
    import io as _io
    import zipfile as _zf
    RESTORED_RUNS = []
    restore_notes = []
    want = {ckpt_path(f).parent.name: f for f in FOLDS}

    def _restore_ckpt(name, data):
        tag = Path(name).stem if name.endswith('.pt') and (not name.endswith('best.pt')) else Path(name).parent.name
        if tag not in want:
            restore_notes.append(f'- `{name}`: skipped, pretraining settings differ from this session')
            return
        st = torch.load(_io.BytesIO(data), map_location='cpu', weights_only=False)
        if st.get('split_manifest_sha256') != MAN.sha256:
            restore_notes.append(f'- `{name}`: skipped, made with a different split manifest')
            return
        dst = ckpt_path(want[tag])
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        restore_notes.append(f'- pretrained encoder for test fold {want[tag]} restored')

    def _restore_results(name, data):
        doc = json.loads(data)
        RESTORED_RUNS.append(doc)
        restore_notes.append(f'- `{name}`: {len(doc.get('runs', []))} runs read (used by ③ if its settings match)')
    for _up in ui_restore.value:
        try:
            if _up.name.endswith('.zip'):
                with _zf.ZipFile(_io.BytesIO(_up.contents)) as z:
                    for zn in z.namelist():
                        if zn.startswith('pretrain/') and zn.endswith('.pt'):
                            _restore_ckpt(zn, z.read(zn))
                        elif zn.startswith('downstream/') and zn.endswith('.pt'):
                            (WORK / 'weights').mkdir(parents=True, exist_ok=True)
                            (WORK / 'weights' / Path(zn).name).write_bytes(z.read(zn))
                        elif zn.endswith('results_downstream.json'):
                            _restore_results(f'{_up.name}:{zn}', z.read(zn))
            elif _up.name.endswith('.json'):
                _restore_results(_up.name, _up.contents)
            elif _up.name.endswith('.pt'):
                _restore_ckpt(_up.name, _up.contents)
        except Exception as err_r:
            restore_notes.append(f'- `{_up.name}`: could not read ({type(err_r).__name__})')
    mo.md(chr(10).join(restore_notes) if restore_notes else '*Nothing restored — starting fresh.*')
    return RESTORED_RUNS, restore_notes


@app.cell
def _(
    AMP_DTYPE,
    AMP_ON,
    BINS,
    CROP,
    DEVICE,
    FOLDS,
    FUSED,
    MAE,
    MAN,
    N_BANDS,
    P,
    PATCH_SZ,
    REP,
    RUNS,
    build_mask,
    cache,
    ckpt_path,
    epoch_rows,
    lr_lambda,
    mo,
    restore_notes,
    run_pre,
    torch,
    train_batch,
    ui_mask,
    ui_pblock,
    ui_pre_bs,
    ui_pre_epochs,
    ui_pre_val,
    ui_pspace,
    ui_ptime,
):
    _ = restore_notes
    PRE_READY = all((ckpt_path(f).exists() for f in FOLDS))
    mo.stop(not run_pre.value and (not PRE_READY), mo.md('*Press ② to pretrain.*'))

    def pretrain_fold(fold_key):
        ids = list(RUNS[fold_key]['train']) + (list(RUNS[fold_key]['val']) if ui_pre_val.value else [])
        rows = cache.indices_for(ids)
        assert not set(cache.indices_for(RUNS[fold_key]['test']).tolist()) & set(rows.tolist())
        torch.manual_seed(0)
        gen = torch.Generator().manual_seed(0)
        mae = MAE(in_ch=N_BANDS, patch=PATCH_SZ, crop=CROP, preset=P['model']).to(DEVICE)
        bs = ui_pre_bs.value
        opt = torch.optim.AdamW(mae.parameters(), lr=P['pre_lr'] * bs ** 0.5, betas=(0.9, 0.99), weight_decay=P['wd'], fused=FUSED)
        spe = -(-len(rows) * REP // bs)
        total = spe * ui_pre_epochs.value
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(total, P['warm_frac'], P['pre_final_div']))
        S = (CROP // PATCH_SZ) ** 2
        hist = []
        for ep in range(ui_pre_epochs.value):
            mae.train()
            er = epoch_rows(rows, gen)
            tot = torch.zeros((), device=DEVICE)
            for a in range(0, len(er), bs):
                x, tf, _y = train_batch(er[a:a + bs], gen)
                masked, keep = build_mask(x.shape[0], BINS, S, ui_mask.value, ui_pspace.value, ui_ptime.value, ui_pblock.value, (P['block_lo'], P['block_hi']), device=DEVICE)
                with torch.autocast(DEVICE, dtype=AMP_DTYPE, enabled=AMP_ON):
                    loss = mae.loss(x, tf, masked, keep)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sch.step()
                tot += loss.detach()
            hist.append(float(tot) / spe)
            yield (fold_key, ep, hist)
        out = ckpt_path(fold_key)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'encoder': mae.encoder.state_dict(), 'model': P['model'], 'fold': fold_key, 'history': hist, 'epoch': ui_pre_epochs.value, 'bins': BINS, 'crop': CROP, 'patch': PATCH_SZ, 'split_manifest_sha256': MAN.sha256, 'pretrain_ids': sorted(ids)}, out)
    CKPTS, HISTORIES = ({}, {})
    for _fk in FOLDS:
        cp = ckpt_path(_fk)
        if cp.exists():
            st = torch.load(cp, map_location='cpu', weights_only=False)
            HISTORIES[_fk] = st['history']
            print(f'fold {_fk}: loaded existing encoder ({cp.parent.name})')
        else:
            with mo.status.progress_bar(total=ui_pre_epochs.value, title=f'pretraining fold {_fk}') as bar_pre:
                for _fk, ep, hist in pretrain_fold(_fk):
                    bar_pre.update(subtitle=f'epoch {ep + 1} · masked L1 {hist[-1]:.4f}')
            HISTORIES[_fk] = hist
            print(f'fold {_fk}: pretrained, final masked L1 {hist[-1]:.4f}')
        CKPTS[_fk] = cp
    return CKPTS, HISTORIES


@app.cell
def _(HISTORIES, plt):
    fig_loss, ax_loss = plt.subplots(figsize=(6, 3))
    for _fk, h in HISTORIES.items():
        ax_loss.plot(range(1, len(h) + 1), h, label=f'fold {_fk}')
    ax_loss.set_xlabel('epoch')
    ax_loss.set_ylabel('masked L1 (group-normalised)')
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)
    fig_loss.tight_layout()
    fig_loss
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Downstream
    """)
    return


@app.cell
def _(P, mo):
    ui_pcts = mo.ui.multiselect(["1", "5", "10", "20", "50", "100"], value=["5", "10", "20", "50", "100"],
                                label="label fractions (%)")
    ui_nseeds = mo.ui.dropdown(["3", "5"], value="3", label="seeds")
    ui_regimes = mo.ui.multiselect(["LP", "FT", "SL"], value=["LP", "FT", "SL"], label="regimes")
    ui_ep_scale = mo.ui.slider(0.1, 1.0, value=1.0, step=0.05, label="epoch scale (1 = MAESTRO schedule)")
    ui_val_every = mo.ui.slider(1, 10, value=P["val_every"], step=1, label="validate on fold 4 every N epochs")
    ui_patience = mo.ui.slider(0.1, 1.0, value=P["patience_frac"], step=0.05,
                               label="early-stop patience (fraction of planned epochs)")
    ui_val_cap = mo.ui.slider(0, 482, value=P["val_cap"], step=1, label="fold-4 tiles for validation (0 = all)")
    ui_dn_bs = mo.ui.slider(8, 128, value=P["dn_bs"], step=8, label="downstream batch")
    ui_ema = mo.ui.checkbox(True, label="per-epoch EMA of weights (FT / SL)")
    ui_budget = mo.ui.slider(1.0, 12.0, value=10.5, step=0.5,
                             label="stop ③ before this many hours of session time (molab ends at 12)")
    ui_save_w = mo.ui.dropdown(["off", "best seed per (fold, regime, fraction)", "all runs"],
                               value="best seed per (fold, regime, fraction)", label="save downstream weights")
    mo.vstack([ui_pcts, ui_nseeds, ui_regimes, ui_ep_scale, ui_val_every, ui_patience, ui_val_cap,
               ui_dn_bs, ui_ema, ui_budget, ui_save_w])
    return (
        ui_budget,
        ui_dn_bs,
        ui_ema,
        ui_ep_scale,
        ui_nseeds,
        ui_patience,
        ui_pcts,
        ui_regimes,
        ui_save_w,
        ui_val_cap,
        ui_val_every,
    )


@app.cell
def _(
    CKPTS,
    FOLDS,
    MAESTRO_EP,
    P,
    REP,
    RUNS,
    cache,
    mo,
    np,
    ui_ep_scale,
    ui_nseeds,
    ui_pcts,
    ui_regimes,
    ui_val_cap,
    ui_val_every,
):
    PCTS = sorted((int(v) for v in ui_pcts.value))
    SEEDS = list(range(int(ui_nseeds.value)))
    REGIMES = [m for m in ('LP', 'FT', 'SL') if m in ui_regimes.value and (m == 'SL' or CKPTS)]
    N_RUNS = len(FOLDS) * len(PCTS) * len(SEEDS) * len(REGIMES)

    def planned_epochs(mode, pct):
        tbl = MAESTRO_EP['lp' if mode == 'LP' else 'ft']
        xs = sorted(tbl)
        lx = np.log([float(x) for x in xs])
        ly = np.log([float(tbl[x]) for x in xs])
        v = np.exp(np.interp(np.log(pct), lx, ly, left=None, right=None))
        if pct < xs[0]:
            v = tbl[xs[0]] * (tbl[xs[0]] / tbl[xs[1]]) ** (np.log(xs[0] / pct) / np.log(xs[1] / xs[0]))
        return max(1, int(round(v * ui_ep_scale.value)))
    GF = {'tiny': (32.7, 10.9), 'small': (87.1, 29.0), 'base': (348.1, 116.0)}[P['model']]
    n_tr = {f: len([i for i in RUNS[f]['train'] if i in cache.pos]) for f in FOLDS}
    flops = 0.0
    for _f in FOLDS:
        for p_ in PCTS:
            n_ = max(1, round(n_tr[_f] * p_ / 100)) * REP
            for m_ in REGIMES:
                e_ = planned_epochs(m_, p_)
                flops += len(SEEDS) * n_ * e_ * (GF[1] if m_ == 'LP' else GF[0]) * 1000000000.0
                flops += len(SEEDS) * (e_ / ui_val_every.value) * (ui_val_cap.value or 482) * REP * GF[1] * 1000000000.0
    ep_lines = '  '.join((f'{p}%: FT/SL {planned_epochs('FT', p)} · LP {planned_epochs('LP', p)}' for p in PCTS))
    mo.md(f'```\nregimes {', '.join(REGIMES)} | fractions {PCTS} | seeds {SEEDS} | test folds {FOLDS} | runs {N_RUNS}\nplanned epochs ({REP} crops per tile per epoch): {ep_lines}\ncompute ≈ {flops / 1e+18:.1f} EFLOP -> ~{flops / 150000000000000.0 / 3600:.0f} h at 150 TFLOP/s, ~{flops / 50000000000000.0 / 3600:.0f} h at 50\nearly stopping may end runs sooner; the progress bar gives the real rate\n```')
    return N_RUNS, PCTS, REGIMES, SEEDS, planned_epochs


@app.cell
def _(mo):
    run_dn = mo.ui.run_button(label="3. Run downstream")
    run_dn
    return (run_dn,)


@app.cell
def _(
    AMP_DTYPE,
    BINS,
    CACHE_DIR,
    CKPTS,
    CROP,
    FOLDS,
    N_RUNS,
    P,
    PATCH_SZ,
    PCTS,
    REGIMES,
    RESTORED_RUNS,
    RESULTS_VERSION,
    RUNS,
    SEEDS,
    VAL_FOLD,
    WORK,
    cache,
    json,
    mo,
    np,
    run_dn,
    ui_dn_bs,
    ui_ema,
    ui_ep_scale,
    ui_mask,
    ui_patience,
    ui_pblock,
    ui_pre_bs,
    ui_pre_epochs,
    ui_pre_val,
    ui_preset,
    ui_pspace,
    ui_ptime,
    ui_save_w,
    ui_val_cap,
    ui_val_every,
):
    RESULTS_FILE = WORK / 'results_downstream.json'
    RUN_SIG = {'version': RESULTS_VERSION, 'method': 'MAESTRO', 'folds': FOLDS, 'val_fold': VAL_FOLD, 'pcts': PCTS, 'seeds': SEEDS, 'regimes': REGIMES, 'model': P['model'], 'bins': BINS, 'crop': CROP, 'patch': PATCH_SZ, 'ep_scale': ui_ep_scale.value, 'val_every': ui_val_every.value, 'patience_frac': ui_patience.value, 'min_frac': P['min_frac'], 'min_delta': P['min_delta'], 'val_cap': ui_val_cap.value, 'dn_bs': ui_dn_bs.value, 'ema': ui_ema.value, 'ft_lr': P['ft_lr'], 'lp_lr': P['lp_lr'], 'ft_final_div': P['ft_final_div'], 'wd': P['wd'], 'save_weights': ui_save_w.value, 'pretrain': {'mask': ui_mask.value, 'p_space': ui_pspace.value, 'p_time': ui_ptime.value, 'p_block': ui_pblock.value, 'epochs': ui_pre_epochs.value, 'bs': ui_pre_bs.value, 'with_val_images': ui_pre_val.value}, 'preset': ui_preset.value, 'amp': str(AMP_DTYPE), 'cache': str(CACHE_DIR), 'ckpts': {str(k): str(v) for k, v in CKPTS.items()}}

    def sig_core(sig):
        """Settings that define a run. Which folds a session lists does not change a
        run's result (runs are keyed by fold), so fold 1's runs stay valid when a
        later session lists folds 1 and 2."""
        return {k: v for k, v in sig.items() if k not in ('folds', 'ckpts')}
    PRIOR_RUNS = []
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as fh_prev:
            prev = json.load(fh_prev)
        if sig_core(prev.get('signature', {})) == sig_core(RUN_SIG):
            PRIOR_RUNS = [r for r in prev['runs'] if r['fold'] in FOLDS]
    restored_used = 0
    prior_keys = {(r['fold'], r['pct'], r['seed'], r['regime']) for r in PRIOR_RUNS}
    for doc_r in RESTORED_RUNS:
        if sig_core(doc_r.get('signature', {})) != sig_core(RUN_SIG):
            continue
        for _r in doc_r.get('runs', []):
            k_r = (_r['fold'], _r['pct'], _r['seed'], _r['regime'])
            if _r['fold'] in FOLDS and k_r not in prior_keys:
                PRIOR_RUNS.append(_r)
                prior_keys.add(k_r)
                restored_used += 1
    if restored_used:
        with open(RESULTS_FILE, 'w') as fh_prev:
            json.dump({'signature': RUN_SIG, 'runs': PRIOR_RUNS}, fh_prev)
    restore_mismatch = [d for d in RESTORED_RUNS if sig_core(d.get('signature', {})) != sig_core(RUN_SIG)]
    DN_READY = len(PRIOR_RUNS) >= N_RUNS
    if restore_mismatch:
        sig_m0 = sig_core(restore_mismatch[0].get('signature', {}))
        diff_r = sorted((k for k in set(sig_core(RUN_SIG)) | set(sig_m0) if sig_core(RUN_SIG).get(k) != sig_m0.get(k)))
        print('restored results not used; settings differ in:', ', '.join(diff_r))
    mo.stop(not run_dn.value and (not DN_READY), mo.md(f'*{len(PRIOR_RUNS)}/{N_RUNS} runs on disk' + (f' ({restored_used} restored)' if restored_used else '') + '. Press 3 to ' + ('resume.*' if PRIOR_RUNS else 'start. Check the cost estimate first.*')))

    def subset_key(fold_key, pct, seed):
        return f'f{fold_key}_p{pct}_s{seed}'

    def make_label_subset(fold_key, pct, seed):
        """Nested per (fold, seed): fraction p takes the first p% of one fixed permutation."""
        train_ids = sorted((i for i in RUNS[fold_key]['train'] if i in cache.pos))
        order = np.random.default_rng(1000 * int(fold_key) + seed).permutation(len(train_ids))
        n = len(train_ids) if pct >= 100 else max(1, int(round(len(train_ids) * pct / 100)))
        return [train_ids[i] for i in sorted(order[:n])]
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
    return (
        DN_READY,
        PRIOR_RUNS,
        RESULTS_FILE,
        RUN_SIG,
        SUBSETS,
        VAL_IDS,
        subset_key,
    )


@app.cell
def _(
    AMP_DTYPE,
    AMP_ON,
    BINS,
    CKPTS,
    CROP,
    DEVICE,
    DN_READY,
    Encoder,
    EpochEMA,
    F,
    FOLDS,
    FUSED,
    MAN,
    N_BANDS,
    N_CLASSES,
    N_RUNS,
    P,
    PATCH_SZ,
    PCTS,
    PRESETS,
    PRIOR_RUNS,
    REGIMES,
    REP,
    RESULTS_FILE,
    RUNS,
    RUN_SIG,
    SEEDS,
    SESSION_T0,
    SUBSETS,
    Segmenter,
    VAL_IDS,
    VOID_CLASS,
    WORK,
    cache,
    copy,
    epoch_rows,
    evaluate_rows,
    json,
    lr_lambda,
    math,
    mo,
    planned_epochs,
    subset_key,
    time_mod,
    torch,
    train_batch,
    ui_budget,
    ui_dn_bs,
    ui_ema,
    ui_patience,
    ui_save_w,
    ui_val_every,
):
    WEIGHT_DIR = WORK / 'weights'
    BEST_SO_FAR = {}

    def build_model(mode, fold_key):
        enc = Encoder(in_ch=N_BANDS, patch=PATCH_SZ, crop=CROP, **PRESETS[P['model']])
        if mode in ('LP', 'FT'):
            enc.load_state_dict(torch.load(CKPTS[fold_key], map_location='cpu', weights_only=False)['encoder'])
        return Segmenter(enc, N_CLASSES, freeze=mode == 'LP').to(DEVICE)

    def train_run(mode, fold_key, tr_rows, va_rows, pct, seed):
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(10000 + seed)
        model = build_model(mode, fold_key)
        n_ep = planned_epochs(mode, pct)
        bs = min(ui_dn_bs.value, len(tr_rows) * REP)
        base = P['lp_lr'] if mode == 'LP' else P['ft_lr']
        final_div = P['pre_final_div'] if mode == 'LP' else P['ft_final_div']
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=base * bs ** 0.5, betas=(0.9, 0.99), weight_decay=P['wd'], fused=FUSED)
        spe = -(-len(tr_rows) * REP // bs)
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(spe * n_ep, P['warm_frac'], final_div))
        ema = EpochEMA(model, n_ep) if ui_ema.value and mode != 'LP' else None
        shadow = copy.deepcopy(model) if ema else None
        patience = max(ui_val_every.value, int(math.ceil(ui_patience.value * n_ep)))
        min_ep = int(P['min_frac'] * n_ep)
        best, best_ref, best_state, last_imp, ran = (-1.0, -1.0, None, 0, 0)
        for ep in range(n_ep):
            model.train()
            er = epoch_rows(tr_rows, gen)
            for a in range(0, len(er), bs):
                x, tf, y = train_batch(er[a:a + bs], gen)
                with torch.autocast(DEVICE, dtype=AMP_DTYPE, enabled=AMP_ON):
                    loss = F.cross_entropy(model(x, tf).float(), y, ignore_index=VOID_CLASS)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sch.step()
            ran = ep + 1
            if ema:
                ema.update(model)
            if ran % ui_val_every.value == 0 or ran == n_ep or ran >= min_ep:
                target = model
                if ema:
                    shadow.load_state_dict(ema.shadow)
                    target = shadow
                v = evaluate_rows(target, va_rows).scores()['mIoU']
                if v > best:
                    best = v
                    best_state = {k: t.detach().cpu().clone() for k, t in target.state_dict().items()}
                if v > best_ref + P['min_delta']:
                    best_ref, last_imp = (v, ran)
                if ran >= min_ep and ran - last_imp >= patience:
                    break
        model.load_state_dict(best_state)
        return (model, best, ran, n_ep)

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
        torch.save({'model': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'regime': mode, 'fold': fold_key, 'pct': pct, 'seed': seed, 'encoder_size': P['model'], 'n_classes': N_CLASSES, 'bins': BINS, 'crop': CROP, 'patch': PATCH_SZ, 'split_manifest_sha256': MAN.sha256, 'pretrain_ckpt': str(CKPTS.get(fold_key, '')), 'metrics': meta}, path)
        return str(path)

    def run_key(r):
        return (r['fold'], r['pct'], r['seed'], r['regime'])

    def save_results(rows):
        tmp = RESULTS_FILE.with_suffix('.tmp')
        with open(tmp, 'w') as fh:
            json.dump({'signature': RUN_SIG, 'runs': rows}, fh)
        tmp.replace(RESULTS_FILE)
    results = list(PRIOR_RUNS)
    DONE = {run_key(r) for r in results}
    for _r in results:
        if _r.get('weights'):
            k_best = (_r['fold'], _r['regime'], _r['pct'])
            BEST_SO_FAR[k_best] = max(BEST_SO_FAR.get(k_best, -1.0), _r['mIoU'])
    todo_runs = [(f, p, sd, m_) for f in FOLDS for p in PCTS for sd in SEEDS for m_ in REGIMES if (f, p, sd, m_) not in DONE]
    cost = lambda p_, f_, m_: len(SUBSETS[subset_key(f_, p_, 0)]) * planned_epochs(m_, p_) * (1 if m_ == 'LP' else 3) + len(VAL_IDS[f_]) * planned_epochs(m_, p_) // ui_val_every.value
    total_cost = sum((cost(p_, f_, m_) for f_, p_, _, m_ in todo_runs))
    budget_s = ui_budget.value * 3600
    units_done, secs_done, STOPPED_EARLY = (0, 0.0, False)
    with mo.status.progress_bar(total=max(total_cost, 1), title='downstream runs', subtitle=f'{len(DONE)}/{N_RUNS} already on disk') as bar_dn:
        for _fold_key, pct, seed, _mode in todo_runs:
            used = time_mod.time() - SESSION_T0
            est = cost(pct, _fold_key, _mode) * (secs_done / units_done) if units_done else 0.0
            if used + est > budget_s:
                STOPPED_EARLY = True
                break
            t_run = time_mod.time()
            ids_used = SUBSETS[subset_key(_fold_key, pct, seed)]
            tr_rows = cache.indices_for(ids_used)
            _m, val_best, epochs_run, n_ep = train_run(_mode, _fold_key, tr_rows, VAL_IDS[_fold_key], pct, seed)
            sc = evaluate_rows(_m, cache.indices_for(RUNS[_fold_key]['test'])).scores()
            wpath = save_run_weights(_m, _fold_key, pct, seed, _mode, sc['mIoU'], {k: sc[k] for k in ('OA', 'mIoU', 'mF1', 'Kappa')})
            results.append({'weights': wpath, 'fold': _fold_key, 'pct': pct, 'seed': seed, 'regime': _mode, 'n_train': len(tr_rows), 'epochs_run': epochs_run, 'epochs_planned': n_ep, 'val_mIoU': val_best, 'subset_sig': ','.join(sorted(ids_used))[:64], **{k: v for k, v in sc.items()}})
            save_results(results)
            del _m
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()
            units_done += cost(pct, _fold_key, _mode)
            secs_done += time_mod.time() - t_run
            bar_dn.update(increment=cost(pct, _fold_key, _mode), subtitle=f'{len(results)}/{N_RUNS} · last: fold {_fold_key}, {pct}%, seed {seed}, {_mode} · {epochs_run}/{n_ep} epochs · mIoU {100 * sc['mIoU']:.1f}')
    with open(WORK / 'label_subsets.json', 'w') as fh:
        json.dump(SUBSETS, fh, indent=2)
    mo.md(f'{len(results)}/{N_RUNS} runs' + (' (loaded from disk)' if DN_READY else '') + (f' — **stopped to stay inside the {ui_budget.value:g} h budget.** Download the session bundle below and continue in the next session.' if STOPPED_EARLY else ''))
    return results, run_key


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Results from other sessions

    Run one test fold per session, download `results_downstream.json` each time,
    then add the earlier files here. Everything below uses the combined set.
    """)
    return


@app.cell
def _(mo):
    ui_merge = mo.ui.file(filetypes=[".json"], multiple=True, kind="area",
                          label="drop results_downstream.json files from other sessions")
    ui_merge
    return (ui_merge,)


@app.cell
def _(RUN_SIG, json, mo, results, run_key, ui_merge):
    MERGE_IGNORE = {'folds', 'cache', 'ckpts', 'data_path', 'amp'}
    ALL_RESULTS = list(results)
    merge_seen = {run_key(r) for r in ALL_RESULTS}
    merge_notes = []
    for _up in ui_merge.value:
        try:
            doc_m = json.loads(_up.contents)
        except Exception as err_m:
            merge_notes.append(f'- {_up.name}: not readable ({err_m})')
            continue
        sig_m = doc_m.get('signature', {})
        diff_m = sorted((k for k in set(sig_m) | set(RUN_SIG) if k not in MERGE_IGNORE and sig_m.get(k) != RUN_SIG.get(k)))
        if diff_m:
            merge_notes.append(f'- {_up.name}: **skipped**, settings differ in {', '.join(diff_m)}')
            continue
        added_m = 0
        for _r in doc_m.get('runs', []):
            if run_key(_r) not in merge_seen:
                ALL_RESULTS.append(_r)
                merge_seen.add(run_key(_r))
                added_m += 1
        merge_notes.append(f'- {_up.name}: added {added_m} runs, folds {sorted({r['fold'] for r in doc_m.get('runs', [])})}')
    ALL_FOLDS = sorted({r['fold'] for r in ALL_RESULTS})
    mo.md(f'**{len(ALL_RESULTS)} runs** across test folds {ALL_FOLDS}' + (chr(10) + chr(10) + chr(10).join(merge_notes) if merge_notes else ''))
    return ALL_FOLDS, ALL_RESULTS


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Subset check
    """)
    return


@app.cell
def _(ALL_FOLDS, ALL_RESULTS, PCTS, SEEDS, mo):
    chk_bad = []
    for _f in ALL_FOLDS:
        for _p in PCTS:
            for _sd in SEEDS:
                sigs = {r['regime']: r['subset_sig'] for r in ALL_RESULTS if r['fold'] == _f and r['pct'] == _p and (r['seed'] == _sd)}
                if len(set(sigs.values())) > 1:
                    chk_bad.append((_f, _p, _sd))
    chk_nl = chr(10)
    if chk_bad:
        mo.md('**MISMATCH:**' + chk_nl + chk_nl.join((f'- fold {a}, {b}%, seed {c}' for a, b, c in chk_bad)))
    else:
        mo.md(f'All {len(ALL_FOLDS) * len(PCTS) * len(SEEDS)} groups: identical subsets across regimes.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Summary
    """)
    return


@app.cell
def _(ALL_FOLDS, ALL_RESULTS, PCTS, REGIMES, SEEDS, mo, np):
    def ms(vals):
        v = np.array(vals, dtype=float)
        return f"{100 * v.mean():.1f} ± {100 * v.std():.1f}"

    MAESTRO_REF = {"FT": {5: 52.5, 20: 59.2, 100: 68.8}, "SL": {5: 38.8, 20: 52.2, 100: 64.6},
                   "LP": {100: 61.2}}

    def table(sfx, with_ref):
        head = "| fraction | n | regime | OA | macro-F1 | mIoU | Kappa | epochs |"
        if with_ref:
            head += " MAESTRO-B fold I mIoU |"
        rows = [head, "|---:|---:|---|---:|---:|---:|---:|---:|" + ("---:|" if with_ref else "")]
        for p_ in PCTS:
            for mode in REGIMES:
                sel = [r for r in ALL_RESULTS if r["pct"] == p_ and r["regime"] == mode]
                if not sel:
                    continue
                row = (f"| {p_}% | {int(np.median([r['n_train'] for r in sel]))} | {mode} "
                       f"| {ms([r['OA' + sfx] for r in sel])} | {ms([r['mF1' + sfx] for r in sel])} "
                       f"| {ms([r['mIoU' + sfx] for r in sel])} | {ms([r['Kappa' + sfx] for r in sel])} "
                       f"| {int(np.median([r['epochs_run'] for r in sel]))} |")
                if with_ref:
                    ref = MAESTRO_REF.get(mode, {}).get(p_)
                    row += f" {ref:.1f} |" if ref is not None else " — |"
                rows.append(row)
        return chr(10).join(rows)

    nl = chr(10)
    mo.md(f"Mean ± std (%) over {len(SEEDS)} seeds × test folds {ALL_FOLDS}." + nl + nl
          + "**All 19 classes** (void ignored, background counted) — the convention MAESTRO reports. "
            "Reference column: MAESTRO-B, fold I (test fold 5), Sentinel-1 + Sentinel-2 + SPOT, one run, "
            "geographically sampled subsets; FT = fine-tuned after pretraining, SL = ViT from scratch, "
            "LP = probing." + nl + nl + table("", True)
          + nl + nl + "**18 crop classes** (background and void excluded)." + nl + nl + table("_crop", False))
    return (MAESTRO_REF,)


@app.cell
def _(ALL_FOLDS, ALL_RESULTS, MAESTRO_REF, PCTS, REGIMES, np, plt):
    fig_dn, axs_dn = plt.subplots(1, 3, figsize=(14, 3.6))
    dn_styles = {'LP': ('-o', '#4c72b0'), 'FT': ('-s', '#dd8452'), 'SL': ('--^', '#55a868')}
    for ax_d, metric in zip(axs_dn, ['mIoU', 'mF1', 'OA']):
        for _mode in REGIMES:
            xs = [p for p in PCTS if any((r['pct'] == p and r['regime'] == _mode for r in ALL_RESULTS))]
            _mu = [float(np.mean([r[metric] for r in ALL_RESULTS if r['pct'] == p and r['regime'] == _mode])) for p in xs]
            _sd = [float(np.std([r[metric] for r in ALL_RESULTS if r['pct'] == p and r['regime'] == _mode])) for p in xs]
            fmt, _col = dn_styles.get(_mode, ('-o', None))
            ax_d.errorbar(xs, 100 * np.array(_mu), yerr=100 * np.array(_sd), fmt=fmt, color=_col, capsize=3, ms=4, label=_mode)
            if metric == 'mIoU' and 5 in ALL_FOLDS and (_mode in MAESTRO_REF):
                ref = MAESTRO_REF[_mode]
                ax_d.plot(list(ref), list(ref.values()), 'x', color=_col, ms=8, mew=2)
        ax_d.set_xscale('log')
        ax_d.set_xticks(PCTS, [f'{p}%' for p in PCTS], fontsize=7)
        ax_d.set_xlabel('label fraction')
        ax_d.set_title({'mIoU': 'mIoU (19 classes)', 'mF1': 'macro-F1', 'OA': 'OA'}[metric])
        ax_d.grid(alpha=0.3)
    axs_dn[0].legend(fontsize=8, title='x = MAESTRO-B', title_fontsize=7)
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
def _(ALL_RESULTS, CLASS_NAMES, REGIMES, mo, np, ui_pc_frac):
    pc_frac = int(ui_pc_frac.value)
    PC_CLASSES = list(range(0, 19))

    def pc_stats(mode, key):
        sel = [r for r in ALL_RESULTS if r['pct'] == pc_frac and r['regime'] == mode]
        arr = np.array([r[key] for r in sel], dtype=float)
        return (np.nanmean(arr, 0), np.nanstd(arr, 0))
    pc_nl = chr(10)
    pc_rows = ['| class | ' + ' | '.join((f'{m} P | {m} R | {m} F1' for m in REGIMES)) + ' |', '|---|' + '---:|' * (3 * len(REGIMES))]
    pc_cache = {m: [pc_stats(m, f'per_class_{k}') for k in ('precision', 'recall', 'f1')] for m in REGIMES}
    for _c in PC_CLASSES:
        vals = []
        for _m in REGIMES:
            for _mu, _sd in pc_cache[_m]:
                vals.append('—' if np.isnan(_mu[_c]) else f'{100 * _mu[_c]:.1f} ± {100 * _sd[_c]:.1f}')
        pc_rows.append(f'| {CLASS_NAMES[_c]} | ' + ' | '.join(vals) + ' |')
    mo.md(f'Per-class precision, recall and F1 (%) at {pc_frac}% labels' + pc_nl + pc_nl + pc_nl.join(pc_rows))
    return PC_CLASSES, pc_frac, pc_stats


@app.cell
def _(CLASS_NAMES, PC_CLASSES, REGIMES, dn_styles, np, pc_frac, pc_stats, plt):
    fig_pc, ax_pc = plt.subplots(figsize=(13, 3.6))
    w_pc = 0.8 / max(len(REGIMES), 1)
    for k_pc, _m in enumerate(REGIMES):
        _mu, _sd = pc_stats(_m, 'per_class_f1')
        ax_pc.bar(np.arange(len(PC_CLASSES)) + k_pc * w_pc, 100 * np.nan_to_num(_mu[PC_CLASSES]), w_pc, yerr=100 * np.nan_to_num(_sd[PC_CLASSES]), label=_m, color=dn_styles[_m][1], capsize=2)
    ax_pc.set_xticks(np.arange(len(PC_CLASSES)) + w_pc * (len(REGIMES) - 1) / 2, [CLASS_NAMES[c] for c in PC_CLASSES], rotation=60, ha='right', fontsize=7)
    ax_pc.set_ylabel('F1 (%)')
    ax_pc.set_title(f'per-class F1 at {pc_frac}% labels')
    ax_pc.legend(fontsize=8)
    ax_pc.grid(axis='y', alpha=0.3)
    fig_pc.tight_layout()
    fig_pc
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Confusion matrix
    """)
    return


@app.cell
def _(REGIMES, mo):
    ui_cm_mode = mo.ui.dropdown(REGIMES, value=REGIMES[0], label="regime")
    ui_cm_mode
    return (ui_cm_mode,)


@app.cell
def _(ALL_RESULTS, CLASS_NAMES, np, pc_frac, plt, ui_cm_mode):
    cm_sel = [r for r in ALL_RESULTS if r["pct"] == pc_frac and r["regime"] == ui_cm_mode.value and "confusion" in r]
    cm_sum = np.sum([np.array(r["confusion"], dtype=np.float64) for r in cm_sel], 0) if cm_sel else np.zeros((20, 20))
    cm_k = cm_sum[:19, :19]
    cm_n = cm_k / np.maximum(cm_k.sum(1, keepdims=True), 1)
    fig_cm, ax_cm = plt.subplots(figsize=(7.5, 6.5))
    im_cm = ax_cm.imshow(cm_n, cmap="Blues", vmin=0, vmax=1)
    ax_cm.set_xticks(range(19), [CLASS_NAMES[c] for c in range(19)], rotation=75, fontsize=6)
    ax_cm.set_yticks(range(19), [CLASS_NAMES[c] for c in range(19)], fontsize=6)
    ax_cm.set_xlabel("predicted"); ax_cm.set_ylabel("true")
    ax_cm.set_title(f"{ui_cm_mode.value} · {pc_frac}% labels · row-normalised, summed over {len(cm_sel)} runs", fontsize=9)
    fig_cm.colorbar(im_cm, fraction=0.046)
    fig_cm.tight_layout()
    fig_cm
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Download

    Session disk is wiped after 12 h, or 90 min idle.
    """)
    return


@app.cell
def _(mo):
    run_dl = mo.ui.run_button(label="prepare downloads")
    mo.vstack([mo.md("Builds the files from what is on disk now. If ③ is running, "
                     "interrupt it first (finished runs are already saved)."), run_dl])
    return (run_dl,)


@app.cell
def _(FOLDS, VERSION, WORK, ckpt_path, json, mo, run_dl):
    mo.stop(not run_dl.value, mo.md("*Press **prepare downloads**.*"))
    import io as _io3, zipfile as _zf3
    DL_RESULTS = WORK / "results_downstream.json"
    DL_SUBSETS = WORK / "label_subsets.json"
    dl_pre = [ckpt_path(f) for f in FOLDS if ckpt_path(f).exists()]
    dl_weights = sorted((WORK / "weights").glob("*.pt")) if (WORK / "weights").exists() else []
    dl_items = []

    if DL_RESULTS.exists() or dl_pre:
        buf_b = _io3.BytesIO()
        with _zf3.ZipFile(buf_b, "w", _zf3.ZIP_STORED, allowZip64=True) as zb:
            for q in dl_pre:
                zb.write(q, f"pretrain/{q.parent.name}.pt")
            for q in (DL_RESULTS, DL_SUBSETS):
                if q.exists():
                    zb.write(q, q.name)
        n_runs_b = len(json.loads(DL_RESULTS.read_text())["runs"]) if DL_RESULTS.exists() else 0
        dl_items.append(mo.download(
            data=buf_b.getvalue(),
            filename=f"session_bundle_v{VERSION}_folds{''.join(map(str, FOLDS))}.zip",
            mimetype="application/zip",
            label=f"session bundle: {len(dl_pre)} encoder(s) + {n_runs_b} runs ({buf_b.tell() / 1e6:.0f} MB)"))

    for q in (DL_RESULTS, DL_SUBSETS):
        if q.exists():
            dl_items.append(mo.download(data=q.read_bytes(), filename=q.name,
                                        mimetype="application/json",
                                        label=f"{q.name} ({q.stat().st_size / 1e3:.0f} kB)"))

    if dl_weights or dl_pre:
        buf_w = _io3.BytesIO()
        with _zf3.ZipFile(buf_w, "w", _zf3.ZIP_STORED, allowZip64=True) as zw:
            for q in dl_pre:
                zw.write(q, f"pretrain/{q.parent.name}.pt")
            for q in dl_weights:
                zw.write(q, f"downstream/{q.name}")
        dl_items.append(mo.download(data=buf_w.getvalue(), filename="maestro_weights.zip",
                                    mimetype="application/zip",
                                    label=f"all weights ({len(dl_pre)} pretrain + {len(dl_weights)} "
                                          f"downstream, {buf_w.tell() / 1e6:.0f} MB)"))

    mo.vstack(dl_items or [mo.md("*Nothing saved yet.*")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reloading weights

    ```python
    st = torch.load("FT_f5_p100_best.pt", map_location="cpu")
    enc = Encoder(in_ch=10, patch=st["patch"], crop=st["crop"], **PRESETS[st["encoder_size"]])
    m = Segmenter(enc, st["n_classes"], freeze=st["regime"] == "LP")
    m.load_state_dict(st["model"])
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
