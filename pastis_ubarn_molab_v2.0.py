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
    # U-BARN on PASTIS · v2.4.1

    Masked pretraining of a Unet + transformer on Sentinel-2 time series, evaluated
    on PASTIS crop segmentation. Dumeur, Valero & Inglada, JSTARS 17 (2024).

    Self-contained: no imports, no uploads. Run top to bottom.
    """)
    return


@app.cell
def _():
    VERSION = "2.4.1"
    RESULTS_VERSION = ".".join(VERSION.split(".")[:2])   # patch releases stay mergeable

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
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # bf16 needs no GradScaler, which removes a host sync on every optimizer step
    AMP_DTYPE = (torch.bfloat16 if DEVICE == "cuda" and torch.cuda.is_bf16_supported()
                 else torch.float16)
    USE_SCALER = DEVICE == "cuda" and AMP_DTYPE == torch.float16
    FUSED = DEVICE == "cuda"

    # every run builds and compiles a fresh model; past the cache limit dynamo would
    # quietly run eager
    import torch._dynamo as _dynamo
    for _k in ("cache_size_limit", "recompile_limit", "accumulated_cache_size_limit",
               "accumulated_recompile_limit"):
        if hasattr(_dynamo.config, _k):
            setattr(_dynamo.config, _k, max(getattr(_dynamo.config, _k), 256))

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


    def maybe_compile(fn, enabled):
        """torch.compile with a one-time fallback to eager if compilation fails."""
        if not enabled:
            return fn
        # the transformer's fused inference fast path bakes input strides into the
        # compiled graph; chunked batches then fail its layout check. The compiler
        # fuses those ops itself, so the fast path buys nothing here.
        if hasattr(torch.backends, "mha"):
            torch.backends.mha.set_fastpath_enabled(False)
        # automatic dynamic: shapes start fixed; only a dimension that actually varies
        # (the number of valid frames) becomes symbolic. dynamic=True also makes H and
        # W symbolic, which breaks compiling the training graph.
        cfn = torch.compile(fn)
        state = {"ok": None}

        def wrapper(*a, **k):
            if state["ok"] is False:
                return fn(*a, **k)
            try:
                out = cfn(*a, **k)
                state["ok"] = True
                return out
            except Exception as err:
                if state["ok"]:
                    raise
                print(f"torch.compile unavailable ({type(err).__name__}); running eager")
                state["ok"] = False
                return fn(*a, **k)
        return wrapper


    def loader(dataset, batch_size, shuffle, drop_last=False, min_steps=0):
        sampler = None
        if shuffle and min_steps and len(dataset) < min_steps * batch_size:
            sampler = torch.utils.data.RandomSampler(dataset, num_samples=min_steps * batch_size)
        return DataLoader(dataset, batch_size=batch_size,
                          shuffle=shuffle if sampler is None else False, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"),
                          drop_last=drop_last,
                          multiprocessing_context=_LOADER_CTX if NUM_WORKERS else None)

    print("version:", VERSION)
    print("torch  :", torch.__version__)
    print("device :", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "(no GPU — attach one from the notebook specs menu)")
    print("work   :", WORK)
    print("workers:", NUM_WORKERS, "(0 = load in the main process)")
    print("amp    :", str(AMP_DTYPE).split(".")[-1], "| grad scaler:", USE_SCALER)
    return (
        AMP_DTYPE,
        DEVICE,
        F,
        FUSED,
        Path,
        RESULTS_VERSION,
        USE_SCALER,
        WORK,
        asdict,
        copy,
        dataclass,
        datetime,
        json,
        loader,
        math,
        maybe_compile,
        nn,
        np,
        os,
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
        make_dataset,
        resolve_source,
        scarce_subset,
    )


@app.cell
def _(F, math, nn, torch):
    def _conv_layer(cin, cout, norm, k=3, s=1, p=1):
        nrm = nn.GroupNorm(4, cout) if norm == 'group' else nn.BatchNorm2d(cout)
        return nn.Sequential(nn.Conv2d(cin, cout, k, s, p, padding_mode='reflect'), nrm, nn.ReLU())

    class _DownBlock(nn.Module):
        """Fig. 13: strided conv (k4 s2 p1) + GN + ReLU, then conv block with residual."""

    # --------------------------------------------------------------------------
    # spatio-spectral encoder (Unet, temporal attention removed from bottleneck)
        def __init__(self, d_in, d_out):
            super().__init__()
            self.down = _conv_layer(d_in, d_in, 'group', 4, 2, 1)
            self.conv1 = _conv_layer(d_in, d_out, 'group')
            self.conv2 = _conv_layer(d_out, d_out, 'group')

        def forward(self, x):
            x = self.conv1(self.down(x))
            return x + self.conv2(x)

    class _UpBlock(nn.Module):
        """Fig. 14: transposed conv (k4 s2 p1) + BN + ReLU, concat skip, conv block with residual."""

        def __init__(self, d_in, d_out, d_skip):
            super().__init__()
            self.skip = nn.Sequential(nn.Conv2d(d_skip, d_skip, 1), nn.BatchNorm2d(d_skip), nn.ReLU())
            self.up = nn.Sequential(nn.ConvTranspose2d(d_in, d_out, 4, 2, 1), nn.BatchNorm2d(d_out), nn.ReLU())
            self.conv1 = _conv_layer(d_out + d_skip, d_out, 'batch')
            self.conv2 = _conv_layer(d_out, d_out, 'batch')

        def forward(self, x, skip):
            x = self.conv1(torch.cat([self.up(x), self.skip(skip)], dim=1))
            return x + self.conv2(x)

    class SpatioSpectralEncoder(nn.Module):
        """Per-date Unet: (N,C,H,W) -> (N,d_model,H,W).

        Encoder widths from Table X (64, 64, 64, 128), blocks from Figs. 13-14.
        Decoder widths are U-TAE's (32, 32, 64, 128); the paper does not state them,
        but they reproduce its parameter count (Table XII) to within 0.4%.
        """

        def __init__(self, in_ch=10, enc=(64, 64, 64, 128), dec=(32, 32, 64, 128), d_model=64):
            super().__init__()
            self.inc = nn.Sequential(_conv_layer(in_ch, enc[0], 'group'), _conv_layer(enc[0], enc[0], 'group'))
            self.downs = nn.ModuleList((_DownBlock(enc[i], enc[i + 1]) for i in range(len(enc) - 1)))
            self.ups = nn.ModuleList((_UpBlock(dec[i], dec[i - 1], enc[i - 1]) for i in range(len(enc) - 1, 0, -1)))
            self.out = _conv_layer(dec[0], d_model, 'batch')

        def forward(self, x):
            skips = [self.inc(x)]
            for d in self.downs:
                skips.append(d(skips[-1]))
            h = skips.pop()
            for up in self.ups:
                h = up(h, skips.pop())
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

        def __init__(self, in_ch: int=10, d_model: int=64, d_hidden: int=128, n_layers: int=3, n_heads: int=4, enc=(64, 64, 64, 128), dec=(32, 32, 64, 128), dropout: float=0.1):
            super().__init__()
            self.d_model = d_model
    # positional encoding on day-of-year (paper eq. 1, scaling constant 1000)
            self.max_seqs = 32768
            self.sse = SpatioSpectralEncoder(in_ch, enc, dec, d_model)
            layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_hidden, dropout=dropout, activation='relu', batch_first=True, norm_first=False)
            self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)

        def embed(self, x: torch.Tensor, doy: torch.Tensor, valid=None, vidx=None) -> torch.Tensor:
            """(B,T,C,H,W) -> (B,T,d,H,W) with positional encoding.

            Padded dates never pass through the SSE. Pass `vidx` (flat indices of
            valid frames, built on the host) to do this without a device sync.
            """
            b, t, c, h, w = x.shape
            flat = x.reshape(b * t, c, h, w)
            if vidx is None and valid is not None:
                vidx = torch.nonzero(valid.reshape(-1)).squeeze(1)
    # backbone
            if vidx is None:
                f = self.sse(flat)
            else:
                enc_v = self.sse(flat.index_select(0, vidx))
                f = enc_v.new_zeros(b * t, self.d_model, h, w).index_copy(0, vidx, enc_v)
            f = f.reshape(b, t, self.d_model, h, w)
            pe = doy_encoding(doy, self.d_model)
            return f + pe[:, :, :, None, None]

        def temporal(self, f: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
            """(B,T,d,H,W) -> same. `valid` (B,T) bool masks padded dates."""
            b, t, d, h, w = f.shape
            seq = f.permute(0, 3, 4, 1, 2).reshape(b * h * w, t, d)
            pad = (~valid)[:, None, None, :].expand(b, h, w, t).reshape(b * h * w, t)
            n = seq.shape[0]
            if n <= self.max_seqs:
                out = self.transformer(seq, src_key_padding_mask=pad)
            else:
                out = torch.cat([self.transformer(seq[i:i + self.max_seqs], src_key_padding_mask=pad[i:i + self.max_seqs]) for i in range(0, n, self.max_seqs)])
            out = torch.nan_to_num(out)
            return out.reshape(b, h, w, t, d).permute(0, 3, 4, 1, 2)

        def forward(self, x, doy, valid, vidx=None):
            return self.temporal(self.embed(x, doy, valid, vidx), valid)

    def permutation_mask(f: torch.Tensor, valid: torch.Tensor, rate: float, generator: torch.Generator | None=None):
        """Corrupt a fraction of dates by permuting embedded values within the batch.

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

    class LinearDecoder(nn.Module):  # (B,T,d)
        """One linear layer on the feature dimension (paper III-B2)."""

        def __init__(self, d_model: int=64, out_ch: int=10):
            super().__init__()
            self.proj = nn.Conv2d(d_model, out_ch, 1)

        def forward(self, f: torch.Tensor) -> torch.Tensor:
            b, t, d, h, w = f.shape  # one sequence per pixel, so batch 16 at 64x64 is 65,536 sequences, one past
            y = self.proj(f.reshape(b * t, d, h, w))  # the limit of PyTorch's fused attention kernels (65,535). Sequences are
            return y.reshape(b, t, -1, h, w)  # independent, so chunking is exact.

    def reconstruction_loss(pred, target, date_mask, pixel_valid=None):
        """MSE over masked dates only (paper eq. 3). pixel_valid optional."""
        if date_mask.sum() == 0:
            return pred.sum() * 0.0
        p = pred[date_mask]
        t = target[date_mask]
        if pixel_valid is not None:  # guard against all-padded rows
            v = pixel_valid[date_mask].unsqueeze(1).float()
            return ((p - t) ** 2 * v).sum() / v.sum().clamp(min=1.0) / p.shape[1]
        return F.mse_loss(p, t)

    class ShallowClassifier(nn.Module):
        """Mean-query attention collapsing time, then 1x1 conv. V = X, per the paper."""

    # pretext task
        def __init__(self, d_model: int=64, n_classes: int=20, feat_norm: bool=True):
            super().__init__()
            self.norm = nn.LayerNorm(d_model) if feat_norm else nn.Identity()
            self.q = nn.Linear(d_model, d_model)
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

        def forward(self, x, doy, valid, vidx=None):
            if self.freeze:
                with torch.no_grad():
                    f = self.encoder(x, doy, valid, vidx)
            else:
                f = self.encoder(x, doy, valid, vidx)
            return self.head(f, valid)

    class ConfusionMeter:
        """Confusion matrix -> OA / Kappa / F1 / mIoU."""

        def __init__(self, n_classes: int, ignore_index: int | None=19):
            self.n = n_classes
            self.ignore = ignore_index
            self.cm = torch.zeros(n_classes, n_classes, dtype=torch.long)

        @torch.no_grad()
    # downstream head
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
            keep_rows[cls] = True  # master query (N, d)
            cm[~keep_rows] = 0  # (N, T, d)
            total = cm.sum().clamp(min=1)
            tp = cm.diag()
            oa = (tp[cls].sum() / total).item()
            row, col = (cm.sum(1), cm.sum(0))  # (N, d)
            pe = ((row * col).sum() / (total * total)).item()
            kappa = (oa - pe) / (1 - pe) if pe < 1 else 0.0
            present = torch.zeros_like(keep_rows)
            present[cls] = row[cls] > 0
            prec = tp / col.clamp(min=1)
            rec = tp / row.clamp(min=1)
            f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-09)
            iou = tp / (row + col - tp).clamp(min=1)
            nan = torch.tensor(float('nan'))
            return {'OA': oa, 'Kappa': kappa, 'mIoU': iou[present].mean().item(), 'mF1': f1[present].mean().item(), 'per_class_iou': torch.where(present, iou, nan).tolist(), 'per_class_f1': torch.where(present, f1, nan).tolist()}

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
            out['per_class_iou_crop'] = c['per_class_iou']
            out['per_class_f1_crop'] = c['per_class_f1']
    # metrics
            return out

    def spatiotemporal_mask(f, valid, t_rate, s_rate=0.0, block=8, generator=None):
        """Temporal masking (paper) plus optional spatial block masking on the remaining dates.

        Each sample gets round(t_rate * n_valid) masked dates (at least one), drawn
        uniformly among its valid dates. Masked positions are overwritten with
        values drawn from anywhere in the batch's feature tensor. No host syncs.

        Returns (corrupted_features, pixel_mask (B,T,H,W) bool).
        """
        b, t, d, h, w = f.shape
        device = f.device
        n_valid = valid.sum(1)
        k = torch.clamp(torch.round(t_rate * n_valid.float()), min=1).long()
        k = torch.minimum(k, n_valid)
        score = torch.rand(b, t, device=device, generator=generator)
        score = score.masked_fill(~valid, 2.0)
        rank = score.argsort(1).argsort(1)
        date_mask = (rank < k[:, None]) & valid
        px = date_mask[:, :, None, None].expand(b, t, h, w)
        if s_rate > 0:
            gh, gw = (-(-h // block), -(-w // block))
            blocks = torch.rand(b, t, gh, gw, device=device, generator=generator) < s_rate
            blocks &= valid[:, :, None, None] & ~date_mask[:, :, None, None]
            blk = blocks.repeat_interleave(block, 2).repeat_interleave(block, 3)[:, :, :h, :w]
            px = px | blk
        vec = f.permute(0, 1, 3, 4, 2).reshape(-1, d)
        src = torch.randint(0, vec.shape[0], (b * t * h * w,), device=device, generator=generator)
        shift = int(torch.randint(0, d, (1,), generator=None).item())
        repl = vec.index_select(0, src).roll(shift, dims=1)
        repl = repl.reshape(b, t, h, w, d).permute(0, 1, 4, 2, 3)
        out = torch.where(px[:, :, None], repl, f)
        return (out, px)

    def masked_pixel_loss(pred, target, px):
        """MSE over masked pixels. Equals reconstruction_loss when px is a whole-date mask."""
        m = px[:, :, None].to(pred.dtype)
        se = ((pred - target) ** 2 * m).sum()
        return se / (m.sum() * pred.shape[2]).clamp(min=1.0)

    def dice_loss(logits, target, ignore_index, n_classes):
        """Soft multiclass Dice over classes present in the batch."""
        prob = logits.float().softmax(1)
        keep = target != ignore_index
        tgt = torch.where(keep, target, torch.zeros_like(target))
        oh = F.one_hot(tgt, n_classes).permute(0, 3, 1, 2).float() * keep[:, None]
        prob = prob * keep[:, None]
        inter = (prob * oh).sum((0, 2, 3))
        denom = prob.sum((0, 2, 3)) + oh.sum((0, 2, 3))
        present = (oh.sum((0, 2, 3)) > 0).to(prob.dtype)
        dice = (2 * inter + 1.0) / (denom + 1.0)
        return 1.0 - (dice * present).sum() / present.sum().clamp(min=1.0)  # padded dates rank last  # source value for each position: a random pixel of a random date anywhere in  # the batch, with a random cyclic shift along the feature axis (paper III-B1:  # another date, another pixel, or another feature)  # CPU draw, no device sync

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
    PRESETS = {
        "paper (Dumeur et al. 2024)": dict(
            patches=2433, t_max=48, cache_crop="128", patch="64",
            window="Jan-Nov 2019 (paper)", random_crop=True, flips=False,
            mask=0.6, smask=0.0, pre_epochs=100, pre_lr="1e-3", pre_bs=2, pre_opt="adam+plateau",
            max_epochs=150, lp_epochs=150, min_epochs=100, patience=20, val_cap=0,
            lr="1e-3", lp_lr="1e-3", warm=0, dn_bs=2, min_steps=0, lr_scale=False,
            ft_mode="plain", lpft=0, enc_mult="1", loss="CE", cw=False, ls="0",
            ema=False, tta=False, fnorm=False, opt="adam", val_every=5, lr_ref_bs=2,
            pre_lr_scale=False, min_delta=0.002),
        "paper · batch 16": dict(
            patches=2433, t_max=48, cache_crop="128", patch="64",
            window="Jan-Nov 2019 (paper)", random_crop=True, flips=False,
            mask=0.6, smask=0.0, pre_epochs=100, pre_lr="1e-3", pre_bs=16, pre_opt="adam+plateau",
            max_epochs=150, lp_epochs=150, min_epochs=100, patience=20, val_cap=150,
            lr="1e-3", lp_lr="1e-3", warm=0, dn_bs=16, min_steps=0, lr_scale=True,
            ft_mode="plain", lpft=0, enc_mult="1", loss="CE", cw=False, ls="0",
            ema=False, tta=False, fnorm=False, opt="adam", val_every=5, lr_ref_bs=2,
            pre_lr_scale=True, min_delta=0.002),
        "improved": dict(
            patches=2433, t_max=48, cache_crop="128", patch="64",
            window="full series (Sep 2018-Nov 2019)", random_crop=True, flips=True,
            mask=0.6, smask=0.25, pre_epochs=100, pre_lr="1e-3", pre_bs=16, pre_opt="adamw+cosine",
            max_epochs=150, lp_epochs=150, min_epochs=0, patience=10, val_cap=150,
            lr="1e-3", lp_lr="1e-2", warm=3, dn_bs=16, min_steps=20, lr_scale=True,
            ft_mode="LP-FT", lpft=10, enc_mult="0.1", loss="CE + Dice", cw=True, ls="0.05",
            ema=True, tta=True, fnorm=True, opt="adamw", val_every=1, lr_ref_bs=4,
            pre_lr_scale=True, min_delta=0.002),
    }
    ui_preset = mo.ui.dropdown(list(PRESETS), value="paper (Dumeur et al. 2024)",
                               label="configuration")
    ui_preset
    return PRESETS, ui_preset


@app.cell
def _(PRESETS, mo, ui_preset):
    P = PRESETS[ui_preset.value]
    ui_subset = mo.ui.slider(100, 2433, value=P["patches"], step=1, label="patches")
    ui_tmax = mo.ui.slider(8, 100, value=P["t_max"], step=1, label="dates per series")
    ui_crop = mo.ui.dropdown(["64", "128"], value=P["cache_crop"], label="cached patch size")
    ui_patch = mo.ui.dropdown(["32", "64", "128"], value=P["patch"], label="network input size")
    ui_window = mo.ui.dropdown(["Jan-Nov 2019 (paper)", "full series (Sep 2018-Nov 2019)"],
                               value=P["window"], label="date window")
    ui_rcrop = mo.ui.checkbox(P["random_crop"], label="random crop in training (paper)")
    ui_flips = mo.ui.checkbox(P["flips"], label="flips + rot90 augmentation")
    ui_hf_token = mo.ui.text(kind="password", label="Hugging Face token (optional; avoids rate limits)",
                             full_width=True)
    mo.vstack([ui_subset, ui_tmax, ui_crop, ui_patch, ui_window, ui_rcrop, ui_flips, ui_hf_token])
    return (
        P,
        ui_crop,
        ui_flips,
        ui_hf_token,
        ui_patch,
        ui_rcrop,
        ui_subset,
        ui_tmax,
        ui_window,
    )


@app.cell
def _(
    DataConfig,
    NEEDED,
    RUNS,
    WORK,
    mo,
    np,
    ui_crop,
    ui_flips,
    ui_patch,
    ui_rcrop,
    ui_subset,
    ui_tmax,
    ui_window,
):
    FULL_SERIES = ui_window.value.startswith('full')
    cfg = DataConfig(t_max=ui_tmax.value, crop=int(ui_crop.value), date_start='2018-09-01' if FULL_SERIES else '2019-01-01', date_end='2019-11-30')
    PATCH = min(int(ui_patch.value), cfg.crop)
    RANDOM_CROP = ui_rcrop.value and PATCH < cfg.crop
    FLIPS = ui_flips.value
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
    mem_note = f' · network sees {PATCH}²' + (' (random crops)' if RANDOM_CROP else '')
    free_gb = _sh.disk_usage(WORK).free / 1000000000.0
    need_gb = len(USE_IDS) * 14.1 / 1024 + len(USE_IDS) * cfg.t_max * 10 * cfg.crop ** 2 * 2 / 1000000000.0
    if need_gb > 0.9 * free_gb:
        mem_note += f' · **disk: need ~{need_gb:.0f} GB, {free_gb:.0f} GB free**'
    mo.md(f'\n**{len(USE_IDS)} patches** · download ≈ {len(USE_IDS) * 14.1 / 1024:.1f} GB ·\ncache ≈ {len(USE_IDS) * cfg.t_max * 10 * cfg.crop ** 2 * 2 / 1000000000.0:.1f} GB{mem_note}\n')
    return FLIPS, PATCH, RANDOM_CROP, USE_IDS, cfg


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
    DEVICE,
    FLIPS,
    PATCH,
    RANDOM_CROP,
    cache,
    loader,
    make_dataset,
    mo,
    np,
    torch,
):
    class GpuLoader:
        """Batches from a GPU-resident cache with no host syncs: indices and crop
        offsets stay on the CPU, and valid-frame indices are built from a CPU copy."""

        def __init__(self, store, rows, batch_size, shuffle, augment, drop_last=False, min_steps=0, patch=None, random_crop=False, flips=True):
            self.s, self.bs, self.shuffle, self.augment = (store, batch_size, shuffle, augment)
            self.full = store['x'].shape[-1]
            self.ps = patch or self.full
            self.random_crop, self.flips = (random_crop, flips)
            self.rows = np.asarray(rows, dtype=np.int64)
            self.drop_last = drop_last and len(self.rows) > batch_size
            n = len(self.rows)
            base = n // batch_size if self.drop_last else -(-n // batch_size)
            self.repeat = bool(shuffle and min_steps and (base < min_steps))
            self.steps = min_steps if self.repeat else base

        def __len__(self):
            return self.steps

        def __iter__(self):
            n, dev = (len(self.rows), self.s['dev'])
            if self.repeat:
                need = self.steps * self.bs
                perm = torch.cat([torch.randperm(n) for _ in range(-(-need // n))])[:need].numpy()
                rows = self.rows[perm]
            elif self.shuffle:
                rows = self.rows[torch.randperm(n).numpy()]
            else:
                rows = self.rows
            for i in range(len(self)):
                j = rows[i * self.bs:(i + 1) * self.bs].tolist()
                if self.ps < self.full:
                    m = self.full - self.ps
                    if self.random_crop:
                        offs = torch.randint(0, m + 1, (len(j), 2)).tolist()
                    else:
                        offs = [(m // 2, m // 2)] * len(j)
                    x = torch.stack([self.s['x'][jj, ..., r:r + self.ps, c:c + self.ps] for jj, (r, c) in zip(j, offs)]).float()
                    y = torch.stack([self.s['y'][jj, r:r + self.ps, c:c + self.ps] for jj, (r, c) in zip(j, offs)])
                else:
                    jt = torch.as_tensor(j).pin_memory().to(dev, non_blocking=True) if dev == 'cuda' else torch.as_tensor(j)
                    x = self.s['x'][jt].float()
                    y = self.s['y'][jt].clone()
                if self.augment and self.flips:
                    for k in range(x.shape[0]):
                        if torch.rand(1).item() < 0.5:
                            x[k], y[k] = (x[k].flip(-1), y[k].flip(-1))
                        if torch.rand(1).item() < 0.5:
                            x[k], y[k] = (x[k].flip(-2), y[k].flip(-2))
                        r = int(torch.randint(4, (1,)))
                        if r:
                            x[k] = torch.rot90(x[k], r, (-2, -1))
                            y[k] = torch.rot90(y[k], r, (-2, -1))
                v_cpu = self.s['valid_cpu'][j]
                vidx = torch.nonzero(v_cpu.reshape(-1)).squeeze(1).pin_memory().to(dev, non_blocking=True) if dev == 'cuda' else torch.nonzero(v_cpu.reshape(-1)).squeeze(1)
                jt = torch.as_tensor(j).pin_memory().to(dev, non_blocking=True) if dev == 'cuda' else torch.as_tensor(j)
                yield {'x': x, 'doy': self.s['doy'][jt], 'valid': self.s['valid'][jt], 'y': y, 'vidx': vidx}
    GPU_STORE = None
    gpu_note = 'data path: CPU DataLoader'
    if DEVICE == 'cuda':
        free_b, total_b = torch.cuda.mem_get_info()
        need_b = cache.x.nbytes + cache.target.size * 8
        if need_b < 0.5 * total_b:
            _xs = torch.empty(cache.x.shape, dtype=torch.float16, device=DEVICE)
            for i0 in range(0, len(cache), 64):
                _xs[i0:i0 + 64] = torch.from_numpy(np.asarray(cache.x[i0:i0 + 64])).to(DEVICE)
            GPU_STORE = {'dev': DEVICE, 'x': _xs, 'doy': torch.from_numpy(cache.doy.astype(np.float32)).to(DEVICE), 'valid': torch.from_numpy(cache.valid.copy()).to(DEVICE), 'valid_cpu': torch.from_numpy(cache.valid.copy()), 'y': torch.from_numpy(cache.target.astype(np.int64)).to(DEVICE)}
            gpu_note = f'data path: GPU-resident ({need_b / 1000000000.0:.1f} GB of {total_b / 1000000000.0:.0f} GB)'
        else:
            gpu_note = f'data path: CPU DataLoader (cache {need_b / 1000000000.0:.1f} GB exceeds 50% of {total_b / 1000000000.0:.0f} GB)'

    def make_loader(rows, batch_size, shuffle, augment=False, drop_last=False, min_steps=0):
        """Training loaders random-crop to PATCH; eval loaders centre-crop."""
        rc = bool(augment and RANDOM_CROP)
        if GPU_STORE is not None:
            return GpuLoader(GPU_STORE, rows, batch_size, shuffle, augment, drop_last, min_steps, patch=PATCH, random_crop=rc, flips=FLIPS)
        return loader(make_dataset(cache, rows, augment=augment, patch=PATCH, random_crop=rc, flips=FLIPS), batch_size, shuffle=shuffle, drop_last=drop_last, min_steps=min_steps)
    EVAL_BS_MULT = 4 if GPU_STORE is not None else 1
    mo.md(gpu_note)
    return EVAL_BS_MULT, GPU_STORE, make_loader


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
def _(P, mo):
    ui_mask = mo.ui.slider(0.1, 0.9, value=P["mask"], step=0.05, label="mask rate")
    ui_smask = mo.ui.slider(0.0, 0.6, value=P["smask"], step=0.05,
                            label="spatial mask rate (0 = paper)")
    ui_sblock = mo.ui.dropdown(["4", "8", "16"], value="8", label="spatial block (px)")
    ui_pre_epochs = mo.ui.slider(1, 400, value=P["pre_epochs"], step=1, label="pretrain epochs")
    ui_pre_lr = mo.ui.dropdown(["3e-4", "1e-3", "3e-3"], value=P["pre_lr"], label="pretrain lr")
    ui_pre_opt = mo.ui.dropdown(["adam+plateau", "adamw+cosine"], value=P["pre_opt"],
                                label="pretrain optimiser")
    ui_bs = mo.ui.slider(1, 64, value=P["pre_bs"], step=1, label="pretrain batch size")
    ui_compile = mo.ui.checkbox(False,
                                label="torch.compile (experimental: crashed on Blackwell with a CUDA "
                                      "misaligned-address error; restart the kernel if it does)")
    mo.vstack([ui_mask, ui_smask, ui_sblock, ui_pre_epochs, ui_pre_lr, ui_pre_opt, ui_bs, ui_compile])
    return (
        ui_bs,
        ui_compile,
        ui_mask,
        ui_pre_epochs,
        ui_pre_lr,
        ui_pre_opt,
        ui_sblock,
        ui_smask,
    )


@app.cell
def _(mo):
    run_pre = mo.ui.run_button(label="② Pretrain (one encoder per fold)")
    run_pre
    return (run_pre,)


@app.cell
def _(
    AMP_DTYPE,
    CACHE_DIR,
    DEVICE,
    EVAL_BS_MULT,
    FLIPS,
    FOLDS,
    FUSED,
    LinearDecoder,
    MAN,
    N_BANDS,
    P,
    PATCH,
    RANDOM_CROP,
    RUNS,
    UBARN,
    USE_SCALER,
    WORK,
    cache,
    make_loader,
    masked_pixel_loss,
    math,
    maybe_compile,
    mo,
    run_pre,
    spatiotemporal_mask,
    torch,
    ui_bs,
    ui_compile,
    ui_mask,
    ui_pre_epochs,
    ui_pre_lr,
    ui_pre_opt,
    ui_sblock,
    ui_smask,
):
    def ckpt_path(fold_key):
        tag = f'pretrain_fold{fold_key}_{CACHE_DIR.name}_m{int(ui_mask.value * 100)}_s{int(ui_smask.value * 100)}b{ui_sblock.value}_e{ui_pre_epochs.value}_lr{ui_pre_lr.value}_bs{ui_bs.value}_{ui_pre_opt.value.replace('+', '-')}_p{PATCH}{('_rc' if RANDOM_CROP else '')}{('_fl' if FLIPS else '')}_v2'
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
        dl = make_loader(idx, ui_bs.value, shuffle=True, augment=True, drop_last=len(idx) > ui_bs.value)
        va_idx = cache.indices_for(RUNS[fold_key]['val'])
        va_dl = make_loader(va_idx, max(ui_bs.value, 2) * EVAL_BS_MULT, shuffle=False)
        torch.manual_seed(0)
        enc = UBARN(in_ch=N_BANDS, d_model=64, d_hidden=128, n_layers=3, n_heads=4).to(DEVICE)
        dec = LinearDecoder(64, N_BANDS).to(DEVICE)
        prm = list(enc.parameters()) + list(dec.parameters())
        enc.embed = maybe_compile(enc.embed, ui_compile.value)
        enc.temporal = maybe_compile(enc.temporal, ui_compile.value)
        n_ep = ui_pre_epochs.value
        paper_opt = ui_pre_opt.value == 'adam+plateau'  # fold 4 images, labels unused: selects the checkpoint, as the paper's
        if paper_opt:  # held-out unlabelled validation set does
            pre_scale = (ui_bs.value / P['lr_ref_bs']) ** 0.5 if P['pre_lr_scale'] else 1.0
            opt = torch.optim.Adam(prm, lr=float(ui_pre_lr.value) * pre_scale, fused=FUSED)
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
        else:
            opt = torch.optim.AdamW(prm, lr=float(ui_pre_lr.value) * (ui_bs.value / 4) ** 0.5, weight_decay=0.0001, fused=FUSED)
            warm = max(1, n_ep // 20)
            sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: (e + 1) / warm if e < warm else 0.5 * (1 + math.cos(math.pi * (e - warm) / max(1, n_ep - warm))))
        scl = torch.amp.GradScaler('cuda', enabled=USE_SCALER)
        out = ckpt_path(fold_key).parent
        out.mkdir(parents=True, exist_ok=True)
        hist, vhist = ([], [])
        last_v, best_v = (None, None)

        @torch.no_grad()
        def val_loss():
            enc.eval()
            dec.eval()
            g = torch.Generator(device=DEVICE).manual_seed(1234)
            tot_v, nb_v = (0.0, 0)
            for b in va_dl:
                x = b['x'].to(DEVICE)
                doy = b['doy'].to(DEVICE)
                valid = b['valid'].to(DEVICE)
                with torch.amp.autocast('cuda', dtype=AMP_DTYPE, enabled=DEVICE == 'cuda'):
                    feats = enc.embed(x, doy, valid, b.get('vidx'))
                    corrupt, pxm = spatiotemporal_mask(feats, valid, ui_mask.value, ui_smask.value, int(ui_sblock.value), generator=g)
                    rec = dec(enc.temporal(corrupt, valid))
                    tot_v = tot_v + masked_pixel_loss(rec.float(), x, pxm)
                nb_v += 1
            return float(tot_v) / max(nb_v, 1)
        for ep in range(n_ep):
            enc.train()
            dec.train()
            tot, nb = (0.0, 0)
            for b in dl:
                x = b['x'].to(DEVICE, non_blocking=True)
                doy = b['doy'].to(DEVICE, non_blocking=True)
                valid = b['valid'].to(DEVICE, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=AMP_DTYPE, enabled=DEVICE == 'cuda'):
                    feats = enc.embed(x, doy, valid, b.get('vidx'))
                    corrupt, pxm = spatiotemporal_mask(feats, valid, ui_mask.value, ui_smask.value, int(ui_sblock.value))
                    rec = dec(enc.temporal(corrupt, valid))
                    loss = masked_pixel_loss(rec.float(), x, pxm)
                opt.zero_grad(set_to_none=True)
                scl.scale(loss).backward()
                if not paper_opt:
                    scl.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(prm, 5.0)
                scl.step(opt)
                scl.update()
                tot = tot + loss.detach()
                nb += 1
            ep_loss = float(tot) / max(nb, 1)
            if len(va_idx):
                pre_eval = (ep + 1) % P['val_every'] == 0 or ep + 1 == n_ep
                if pre_eval:
                    last_v = val_loss()
            else:
                pre_eval, last_v = (True, ep_loss)
            hist.append(ep_loss)
            vhist.append(last_v if last_v is not None else float('nan'))
            if paper_opt:
                if last_v is not None:
                    sch.step(last_v)
            else:
                sch.step()
            if pre_eval and (best_v is None or last_v <= best_v):
                best_v = last_v
                torch.save({'encoder': enc.state_dict(), 'decoder': dec.state_dict(), 'loss': best_v, 'train_loss': ep_loss, 'epoch': ep, 'history': list(hist), 'val_history': list(vhist), 'mask_rate': ui_mask.value, 'spatial_mask': ui_smask.value, 'spatial_block': int(ui_sblock.value), 'fold': fold_key, 'optimizer': ui_pre_opt.value, 'split_manifest_sha256': MAN.sha256, 'pretrain_ids': sorted(pool)}, out / 'best.pt')
            yield (fold_key, ep, last_v, out / 'best.pt', hist, enc, dec)
        enc.__dict__.pop('embed', None)
        enc.__dict__.pop('temporal', None)  # no val set: select on train loss
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
        fv = enc_v.embed(xv1, dv1, vv1)
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
def _(P, mo, ui_preset):
    ui_pcts = mo.ui.multiselect(["1", "5", "10", "20", "50", "100"],
                                value=["1", "5", "10", "20", "50", "100"],
                                label="label fractions (%)")
    ui_nseeds = mo.ui.dropdown(["3", "5"], value="3", label="seeds")
    ui_max_epochs = mo.ui.slider(10, 500, value=P["max_epochs"], step=10, label="max epochs (FT / SL)")
    ui_lp_epochs = mo.ui.slider(10, 500, value=P["lp_epochs"], step=10, label="max epochs (LP)")
    ui_min_epochs = mo.ui.slider(0, 200, value=P["min_epochs"], step=10,
                                 label="min epochs before early stopping (paper: 100)")
    ui_patience = mo.ui.slider(3, 50, value=P["patience"], step=1, label="early-stop patience")
    ui_val_every = mo.ui.slider(1, 10, value=P["val_every"], step=1,
                                label="validate every N epochs until the minimum")
    ui_val_cap = mo.ui.slider(0, 500, value=P["val_cap"], step=50,
                              label="val patches (0 = all of fold 4)")
    ui_lr = mo.ui.dropdown(["3e-4", "1e-3", "3e-3"], value=P["lr"], label="lr (FT / SL)")
    ui_lp_lr = mo.ui.dropdown(["1e-3", "3e-3", "1e-2"], value=P["lp_lr"], label="lr (LP)")
    ui_opt = mo.ui.dropdown(["adam", "adamw"], value=P["opt"], label="optimiser")
    ui_sync_debug = mo.ui.checkbox(False, label="warn on GPU host syncs (diagnostic, one run)")
    ui_save_w = mo.ui.dropdown(["off", "best seed per (fold, regime, fraction)", "all runs"],
                               value="best seed per (fold, regime, fraction)",
                               label="save downstream weights")
    ui_cw = mo.ui.checkbox(P["cw"], label="class-balanced loss")
    ui_fnorm = mo.ui.checkbox(P["fnorm"], label="normalise features before head")
    ui_ft_mode = mo.ui.dropdown(["plain", "LP-FT"], value=P["ft_mode"], label="FT strategy")
    ui_lpft_epochs = mo.ui.slider(0, 40, value=P["lpft"], step=5, label="LP-FT head-only epochs")
    ui_enc_mult = mo.ui.dropdown(["1", "0.3", "0.1"], value=P["enc_mult"], label="FT encoder lr x")
    ui_warm = mo.ui.slider(0, 10, value=P["warm"], step=1, label="warmup epochs")
    ui_dn_bs = mo.ui.slider(1, 64, value=P["dn_bs"], step=1, label="downstream batch size")
    ui_lr_scale = mo.ui.checkbox(P["lr_scale"],
                                 label=f"scale lr by sqrt(batch / {P['lr_ref_bs']})")
    ui_min_steps = mo.ui.slider(0, 50, value=P["min_steps"], step=5,
                                label="min steps per epoch (0 = one pass)")
    ui_loss = mo.ui.dropdown(["CE", "CE + Dice"], value=P["loss"], label="loss")
    ui_ls = mo.ui.dropdown(["0", "0.05", "0.1"], value=P["ls"], label="label smoothing")
    ui_ema = mo.ui.checkbox(P["ema"], label="EMA weights")
    ui_tta = mo.ui.checkbox(P["tta"], label="test-time augmentation (8x)")
    mo.vstack([
        mo.md(f"**Preset: {ui_preset.value}**"),
        mo.md("**Protocol**"),
        ui_pcts, ui_nseeds, ui_min_epochs, ui_patience, ui_val_every, ui_val_cap,
        mo.md("**Optimisation**"),
        ui_max_epochs, ui_lp_epochs, ui_lr, ui_lp_lr, ui_opt, ui_warm,
        ui_dn_bs, ui_lr_scale, ui_min_steps,
        mo.md("**Fine-tuning (FT only)**"),
        ui_ft_mode, ui_lpft_epochs, ui_enc_mult,
        mo.md("**Shared by all regimes**"),
        ui_loss, ui_cw, ui_ls, ui_ema, ui_tta, ui_fnorm,
        mo.md("**Output**"),
        ui_save_w, ui_sync_debug,
    ])
    return (
        ui_cw,
        ui_dn_bs,
        ui_ema,
        ui_enc_mult,
        ui_fnorm,
        ui_ft_mode,
        ui_loss,
        ui_lp_epochs,
        ui_lp_lr,
        ui_lpft_epochs,
        ui_lr,
        ui_lr_scale,
        ui_ls,
        ui_max_epochs,
        ui_min_epochs,
        ui_min_steps,
        ui_nseeds,
        ui_opt,
        ui_patience,
        ui_pcts,
        ui_save_w,
        ui_sync_debug,
        ui_tta,
        ui_val_cap,
        ui_val_every,
        ui_warm,
    )


@app.cell
def _(
    CKPTS,
    FOLDS,
    RUNS,
    cache,
    mo,
    ui_dn_bs,
    ui_ft_mode,
    ui_lpft_epochs,
    ui_min_epochs,
    ui_min_steps,
    ui_nseeds,
    ui_patience,
    ui_pcts,
):
    PCTS = sorted((int(v) for v in ui_pcts.value))
    SEEDS = list(range(int(ui_nseeds.value)))
    REGIMES = ['LP', 'FT', 'SL'] if CKPTS else ['SL']
    N_RUNS = len(FOLDS) * len(PCTS) * len(SEEDS) * len(REGIMES)
    est_nl = chr(10)
    n_train_full = {f: len([i for i in RUNS[f]['train'] if i in cache.pos]) for f in FOLDS}
    ep_guess = max(ui_min_epochs.value, 40) + ui_patience.value
    steps = 0
    for _f in FOLDS:
        for p_ in PCTS:
            n_ = max(1, round(n_train_full[_f] * p_ / 100))
            bs_ = min(ui_dn_bs.value, n_)
            per_ep = max(-(-n_ // bs_), ui_min_steps.value)
            for m_ in REGIMES:
                e_ = ep_guess + (ui_lpft_epochs.value if m_ == 'FT' and ui_ft_mode.value == 'LP-FT' else 0)
                steps += per_ep * e_ * len(SEEDS)
    est_lines = [f'regimes {', '.join(REGIMES)} | fractions {PCTS} | seeds {SEEDS} | folds {FOLDS}', f'runs {N_RUNS} | ~{steps / 1000000.0:.2f} M optimiser steps at batch {ui_dn_bs.value}, assuming ~{ep_guess} epochs per run', f'at 20 ms/step: ~{steps * 0.02 / 3600:.0f} h  |  at 40 ms/step: ~{steps * 0.04 / 3600:.0f} h', "measure your real ms/step from the first run's progress bar"]
    if steps * 0.02 / 3600 > 10:
        est_lines += ['', '!! exceeds one 12 h session. Results save after every run and resume', '   within a session; across sessions the disk is wiped, so either run one', '   test fold per session (FOLDS in the splits cell) and download each', '   results file, or trim fractions / seeds.']
    mo.md('```' + est_nl + est_nl.join(est_lines) + est_nl + '```')
    return N_RUNS, PCTS, REGIMES, SEEDS


@app.cell
def _(mo):
    run_dn = mo.ui.run_button(label="3. Run downstream")
    run_dn
    return (run_dn,)


@app.cell
def _(
    AMP_DTYPE,
    CACHE_DIR,
    CKPTS,
    FLIPS,
    FOLDS,
    GPU_STORE,
    N_RUNS,
    P,
    PATCH,
    PCTS,
    RANDOM_CROP,
    REGIMES,
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
    scarce_subset,
    ui_cw,
    ui_dn_bs,
    ui_ema,
    ui_enc_mult,
    ui_fnorm,
    ui_ft_mode,
    ui_loss,
    ui_lp_epochs,
    ui_lp_lr,
    ui_lpft_epochs,
    ui_lr,
    ui_lr_scale,
    ui_ls,
    ui_max_epochs,
    ui_min_epochs,
    ui_min_steps,
    ui_opt,
    ui_patience,
    ui_preset,
    ui_save_w,
    ui_tta,
    ui_val_cap,
    ui_val_every,
    ui_warm,
):
    RESULTS_FILE = WORK / 'results_downstream.json'
    RUN_SIG = {'version': RESULTS_VERSION, 'folds': FOLDS, 'val_fold': VAL_FOLD, 'pcts': PCTS, 'seeds': SEEDS, 'regimes': REGIMES, 'max_epochs': ui_max_epochs.value, 'lp_epochs': ui_lp_epochs.value, 'patience': ui_patience.value, 'val_cap': ui_val_cap.value, 'lr': ui_lr.value, 'lp_lr': ui_lp_lr.value, 'class_balanced': ui_cw.value, 'feat_norm': ui_fnorm.value, 'save_weights': ui_save_w.value, 'ft_mode': ui_ft_mode.value, 'lpft_epochs': ui_lpft_epochs.value, 'enc_mult': ui_enc_mult.value, 'warmup': ui_warm.value, 'loss': ui_loss.value, 'label_smoothing': ui_ls.value, 'ema': ui_ema.value, 'tta': ui_tta.value, 'data_path': 'gpu' if GPU_STORE is not None else 'cpu', 'dn_bs': ui_dn_bs.value, 'min_steps': ui_min_steps.value, 'min_epochs': ui_min_epochs.value, 'optimizer': ui_opt.value, 'val_every': ui_val_every.value, 'lr_ref_bs': P['lr_ref_bs'], 'min_delta': P['min_delta'], 'lr_scale': ui_lr_scale.value, 'preset': ui_preset.value, 'patch': PATCH, 'random_crop': RANDOM_CROP, 'flips': FLIPS, 'amp': str(AMP_DTYPE), 'cache': str(CACHE_DIR), 'ckpts': {str(k): str(v) for k, v in CKPTS.items()}}
    PRIOR_RUNS = []
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as fh_prev:
            prev = json.load(fh_prev)
        if prev.get('signature') == RUN_SIG:
            PRIOR_RUNS = prev['runs']
    DN_READY = len(PRIOR_RUNS) >= N_RUNS
    mo.stop(not run_dn.value and (not DN_READY), mo.md(f'*{len(PRIOR_RUNS)}/{N_RUNS} runs on disk. Press 3 to ' + ('resume.*' if PRIOR_RUNS else 'start. Check the cost estimate first.*')))

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
    CKPTS,
    ConfusionMeter,
    DEVICE,
    DN_READY,
    EVAL_BS_MULT,
    FOLDS,
    FUSED,
    MAN,
    N_BANDS,
    N_CLASSES,
    N_RUNS,
    P,
    PCTS,
    PRIOR_RUNS,
    REGIMES,
    RESULTS_FILE,
    RUNS,
    RUN_SIG,
    SEEDS,
    SUBSETS,
    SegmentationModel,
    UBARN,
    USE_SCALER,
    VAL_IDS,
    VOID_CLASS,
    WORK,
    cache,
    copy,
    dice_loss,
    json,
    make_loader,
    maybe_compile,
    mo,
    np,
    subset_key,
    torch,
    ui_compile,
    ui_cw,
    ui_dn_bs,
    ui_ema,
    ui_enc_mult,
    ui_fnorm,
    ui_ft_mode,
    ui_loss,
    ui_lp_epochs,
    ui_lp_lr,
    ui_lpft_epochs,
    ui_lr,
    ui_lr_scale,
    ui_ls,
    ui_max_epochs,
    ui_min_epochs,
    ui_min_steps,
    ui_opt,
    ui_patience,
    ui_save_w,
    ui_sync_debug,
    ui_tta,
    ui_val_every,
    ui_warm,
):
    def build_model(mode, fold_key):
        enc = UBARN(in_ch=N_BANDS, d_model=64, d_hidden=128, n_layers=3, n_heads=4)
        if mode in ('LP', 'FT'):
            enc.load_state_dict(torch.load(CKPTS[fold_key], map_location='cpu')['encoder'])
        m = SegmentationModel(enc, n_classes=N_CLASSES, freeze=mode == 'LP', feat_norm=ui_fnorm.value).to(DEVICE)
        m.forward = maybe_compile(m.forward, ui_compile.value)
        return m

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
    def predict(model, x, doy, valid, tta=False, vidx=None):
        if not tta:
            return model(x, doy, valid, vidx).float().softmax(1)
        acc = 0
        for flip in (False, True):
            xf = x.flip(-1) if flip else x
            for k in range(4):
                pr = model(torch.rot90(xf, k, dims=(-2, -1)), doy, valid, vidx).float().softmax(1)
                pr = torch.rot90(pr, -k, dims=(-2, -1))
                acc = acc + (pr.flip(-1) if flip else pr)
        return acc / 8

    @torch.no_grad()
    def evaluate(model, dl, tta=False):
        model.eval()
        meter = ConfusionMeter(N_CLASSES, ignore_index=VOID_CLASS)
        for b in dl:
            with torch.amp.autocast('cuda', dtype=AMP_DTYPE, enabled=DEVICE == 'cuda'):
                pr = predict(model, b['x'].to(DEVICE), b['doy'].to(DEVICE), b['valid'].to(DEVICE), tta, b.get('vidx'))
            meter.update(pr.argmax(1), b['y'])
        return meter

    class EMA:
        """Averages trainable parameters only, in one fused op per step."""

        def __init__(self, model, decay):
            self.decay, self.n = (decay, 0)
            self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
            named = dict(model.named_parameters())
            keys = [k for k in self.shadow if k in named and named[k].requires_grad and named[k].dtype.is_floating_point]
            self.src = [named[k].data for k in keys]
            self.dst = [self.shadow[k] for k in keys]

        @torch.no_grad()
        def update(self, model):
            self.n += 1
            w = 1.0 - min(self.decay, (1 + self.n) / (10 + self.n))
            if self.src:
                torch._foreach_lerp_(self.dst, self.src, w)

    def fit(model, tr_dl, va_dl, epochs, patience, head_lr, enc_mult, weight, early_stop=True):
        head_p = [p for p in model.head.parameters() if p.requires_grad]
        enc_p = [p for p in model.encoder.parameters() if p.requires_grad]
        groups = [{'params': head_p, 'lr': head_lr}]
        if enc_p:
            groups.append({'params': enc_p, 'lr': head_lr * enc_mult})
        if ui_opt.value == 'adam':
            opt = torch.optim.Adam(groups, fused=FUSED)
        else:
            opt = torch.optim.AdamW(groups, weight_decay=0.0001, fused=FUSED)
        base_lr = [g['lr'] for g in opt.param_groups]
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', patience=max(2, patience // 2), factor=0.5)
        scl = torch.amp.GradScaler('cuda', enabled=USE_SCALER)
        ce = torch.nn.CrossEntropyLoss(weight=weight.to(DEVICE) if weight is not None else None, ignore_index=VOID_CLASS, label_smoothing=float(ui_ls.value))
        use_dice = ui_loss.value == 'CE + Dice'
        ema = EMA(model, 0.99) if ui_ema.value else None
        ema_model = None
        if ema:
            ema_model = copy.deepcopy(model)
            ema_model.__dict__.pop('forward', None)
        warm = min(ui_warm.value, max(epochs - 1, 0))
        prm = head_p + enc_p
        best, best_state, ran = (-1.0, None, 0)
        last_val, last_improve, best_ref = (None, 0, -1.0)
        for ep in range(epochs):
            if ep < warm:
                for g, b0 in zip(opt.param_groups, base_lr):
                    g['lr'] = b0 * (ep + 1) / warm
            model.train()
            for b in tr_dl:
                yb = b['y'].to(DEVICE)
                with torch.amp.autocast('cuda', dtype=AMP_DTYPE, enabled=DEVICE == 'cuda'):
                    lg = model(b['x'].to(DEVICE), b['doy'].to(DEVICE), b['valid'].to(DEVICE), b.get('vidx'))
                loss = ce(lg.float(), yb)
                if use_dice:
                    loss = loss + dice_loss(lg, yb, VOID_CLASS, N_CLASSES)
                opt.zero_grad(set_to_none=True)
                scl.scale(loss).backward()
                if ui_opt.value != 'adam':
                    scl.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(prm, 5.0)
                scl.step(opt)
                scl.update()
                if ema:
                    ema.update(model)
            ran = ep + 1
            do_eval = ran % ui_val_every.value == 0 or ran >= ui_min_epochs.value or ran == epochs
            if do_eval:
                target = model
                if ema:
                    ema_model.load_state_dict(ema.shadow)
                    target = ema_model
                last_val = evaluate(target, va_dl).scores()['mIoU']
                if last_val > best:
                    best_state = {k: v.detach().cpu().clone() for k, v in target.state_dict().items()}
                    if last_val > best_ref + P['min_delta']:
                        best_ref, last_improve = (last_val, ran)
                    best = last_val
            if ep >= warm and last_val is not None:
                sch.step(last_val)  # before the minimum epoch count early stopping cannot fire, so
            if early_stop and do_eval and (ep >= warm) and (ran >= ui_min_epochs.value) and (ran - last_improve >= patience):  # validation there only picks the best checkpoint: every val_every epochs
                break
        if best_state:
            model.load_state_dict(best_state)
        return (best, ran)

    def train_run(mode, fold_key, tr_dl, va_dl, weight, lr_scale=1.0):
        """LP / FT / SL. Shared: warmup, EMA, label smoothing, loss. FT only: LP-FT, encoder lr multiplier."""
        m = build_model(mode, fold_key)
        head_lr = float(ui_lp_lr.value if mode == 'LP' else ui_lr.value) * lr_scale
        ran_total = 0
        if mode == 'FT' and ui_ft_mode.value == 'LP-FT' and (ui_lpft_epochs.value > 0):
            for prm_e in m.encoder.parameters():
                prm_e.requires_grad_(False)
            m.freeze = True
            _, r1 = fit(m, tr_dl, va_dl, ui_lpft_epochs.value, ui_patience.value, float(ui_lp_lr.value) * lr_scale, 1.0, weight, early_stop=False)  # stepping every epoch with the latest value keeps plateau patience in epochs
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
    cost = lambda p_, f_: len(SUBSETS[subset_key(f_, p_, 0)]) + len(VAL_IDS[f_]) // 4
    total_cost = sum((cost(p_, f_) for f_, p_, _, _ in todo_runs))
    with mo.status.progress_bar(total=max(total_cost, 1), title='downstream runs', subtitle=f'{len(DONE)}/{N_RUNS} already on disk') as bar_dn:
        for _fold_key in FOLDS:
            if not any((t[0] == _fold_key for t in todo_runs)):
                continue
            va_dl = make_loader(VAL_IDS[_fold_key], ui_dn_bs.value * EVAL_BS_MULT, shuffle=False)
            te_dl = make_loader(cache.indices_for(RUNS[_fold_key]['test']), ui_dn_bs.value * EVAL_BS_MULT, shuffle=False)
            for pct in PCTS:
                for seed in SEEDS:
                    pending = [m_ for m_ in REGIMES if (_fold_key, pct, seed, m_) not in DONE]
                    if not pending:
                        continue
                    ids_used = SUBSETS[subset_key(_fold_key, pct, seed)]
                    tr_rows = cache.indices_for(ids_used)
                    weight = class_weights(tr_rows) if ui_cw.value else None
                    bs_eff = min(ui_dn_bs.value, len(tr_rows))
                    lr_scale = (bs_eff / P['lr_ref_bs']) ** 0.5 if ui_lr_scale.value else 1.0
                    tr_dl = make_loader(tr_rows, bs_eff, shuffle=True, augment=True, min_steps=ui_min_steps.value)
                    for _mode in pending:
                        torch.manual_seed(seed)
                        np.random.seed(seed)
                        if DEVICE == 'cuda' and ui_sync_debug.value:
                            torch.cuda.set_sync_debug_mode('warn')
                        try:
                            _m, val_best, epochs_run = train_run(_mode, _fold_key, tr_dl, va_dl, weight, lr_scale)
                        finally:
                            if DEVICE == 'cuda':
                                torch.cuda.set_sync_debug_mode('default')
                        sc = evaluate(_m, te_dl, tta=ui_tta.value).scores()
                        wpath = save_run_weights(_m, _fold_key, pct, seed, _mode, sc['mIoU'], {k: sc[k] for k in ('OA', 'mIoU', 'mF1', 'Kappa')})
                        results.append({'weights': wpath, 'fold': _fold_key, 'pct': pct, 'seed': seed, 'regime': _mode, 'n_train': len(tr_rows), 'epochs_run': epochs_run, 'val_mIoU': val_best, 'subset_sig': ','.join(sorted(ids_used))[:64], 'OA': sc['OA'], 'mIoU': sc['mIoU'], 'mF1': sc['mF1'], 'Kappa': sc['Kappa'], 'OA_crop': sc['OA_crop'], 'mIoU_crop': sc['mIoU_crop'], 'mF1_crop': sc['mF1_crop'], 'Kappa_crop': sc['Kappa_crop'], 'per_class_iou': sc['per_class_iou'], 'per_class_f1': sc['per_class_f1'], 'per_class_iou_crop': sc['per_class_iou_crop'], 'per_class_f1_crop': sc['per_class_f1_crop']})
                        save_results(results)
                        del _m
                        if DEVICE == 'cuda':
                            torch.cuda.empty_cache()
                        bar_dn.update(increment=cost(pct, _fold_key), subtitle=f'{len(results)}/{N_RUNS} · last: fold {_fold_key}, {pct}%, seed {seed}, {_mode} · {epochs_run} epochs · mIoU(crop) {100 * sc['mIoU_crop']:.1f}')
    with open(WORK / 'label_subsets.json', 'w') as fh:
        json.dump(SUBSETS, fh, indent=2)
    mo.md(f'{len(results)}/{N_RUNS} runs' + (' (loaded from disk)' if DN_READY else ''))
    return WEIGHT_DIR, results, run_key


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
    for up in ui_merge.value:
        try:
            doc_m = json.loads(up.contents)
        except Exception as err_m:
            merge_notes.append(f'- {up.name}: not readable ({err_m})')
            continue
        sig_m = doc_m.get('signature', {})
        diff_m = sorted((k for k in set(sig_m) | set(RUN_SIG) if k not in MERGE_IGNORE and sig_m.get(k) != RUN_SIG.get(k)))
        if diff_m:
            merge_notes.append(f'- {up.name}: **skipped**, settings differ in {', '.join(diff_m)}')
            continue
        added_m = 0
        for _r in doc_m.get('runs', []):
            if run_key(_r) not in merge_seen:
                ALL_RESULTS.append(_r)
                merge_seen.add(run_key(_r))
                added_m += 1
        merge_notes.append(f'- {up.name}: added {added_m} runs, folds {sorted({r['fold'] for r in doc_m.get('runs', [])})}')
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

    def table(sfx, refs):
        rows = ["| fraction | n | regime | Kappa | OA | mF1 | mIoU | epochs |",
                "|---:|---:|---|---:|---:|---:|---:|---:|"]
        for p_ in PCTS:
            for mode in REGIMES:
                sel = [r for r in ALL_RESULTS if r["pct"] == p_ and r["regime"] == mode]
                if not sel or f"OA{sfx}" not in sel[0]:
                    continue
                rows.append(
                    f"| {p_}% | {int(np.median([r['n_train'] for r in sel]))} | {mode} "
                    f"| {ms([r['Kappa' + sfx] for r in sel])} | {ms([r['OA' + sfx] for r in sel])} "
                    f"| {ms([r['mF1' + sfx] for r in sel])} | {ms([r['mIoU' + sfx] for r in sel])} "
                    f"| {int(np.median([r['epochs_run'] for r in sel]))} |")
        rows += [f"| 100% | — | *{n}* | {k} | {o} | {f} | {m} | — |" for n, k, o, f, m in refs]
        return chr(10).join(rows)

    PAPER_T4 = [("FR (paper)", "79.0 ± 1.1", "83.2 ± 1.0", "61.8 ± 1.7", "50.1 ± 1.5"),
                ("FT (paper)", "89.2 ± 1.1", "91.2 ± 0.9", "81.6 ± 1.8", "71.3 ± 2.2"),
                ("e2e (paper)", "89.3 ± 1.0", "91.3 ± 0.8", "82.0 ± 1.3", "71.6 ± 1.7"),
                ("U-TAE (paper)", "88.3 ± 1.2", "90.6 ± 0.9", "80.3 ± 2.3", "69.6 ± 2.7")]
    LEADERBOARD = [("U-TAE (leaderboard, 128²)", "—", "83.2", "—", "63.1")]
    nl = chr(10)
    mo.md(f"Mean ± std (%) over {len(SEEDS)} seeds × {len(ALL_FOLDS)} test folds {ALL_FOLDS}. "
          "LP / FT / SL correspond to the paper's FR / FT / e2e." + nl + nl
          + "**Paper convention** (Dumeur et al., Table IV): 18 crop classes, "
            "background and void pixels excluded." + nl + nl + table("_crop", PAPER_T4)
          + nl + nl + "**Leaderboard convention** (U-TAE): void excluded, background is a class."
          + nl + nl + table("", LEADERBOARD))
    return


@app.cell
def _(ALL_RESULTS, PCTS, REGIMES, np, plt):
    fig_dn, axs_dn = plt.subplots(1, 4, figsize=(15, 3.4))
    dn_styles = {'LP': ('-o', '#4c72b0'), 'FT': ('-s', '#dd8452'), 'SL': ('--^', '#55a868')}
    for ax_d, metric in zip(axs_dn, ['OA_crop', 'mIoU_crop', 'mF1_crop', 'Kappa_crop']):
        for _mode in REGIMES:
            _xs = [p for p in PCTS if any((r['pct'] == p and r['regime'] == _mode for r in ALL_RESULTS))]
            mu = [float(np.mean([r[metric] for r in ALL_RESULTS if r['pct'] == p and r['regime'] == _mode])) for p in _xs]
            _sd = [float(np.std([r[metric] for r in ALL_RESULTS if r['pct'] == p and r['regime'] == _mode])) for p in _xs]
            fmt, _col = dn_styles.get(_mode, ('-o', None))
            ax_d.errorbar(_xs, mu, yerr=_sd, fmt=fmt, color=_col, capsize=3, ms=4, label=_mode)
        ax_d.set_xscale('log')
        ax_d.set_xticks(PCTS, [f'{p}%' for p in PCTS], fontsize=7)
        ax_d.set_xlabel('label fraction')
        ax_d.set_title(metric.replace('_crop', '') + ' (18 crops)')
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
def _(ALL_RESULTS, CLASS_NAMES, REGIMES, mo, np, ui_pc_frac):
    pc_frac = int(ui_pc_frac.value)
    PC_CLASSES = list(range(1, 19))

    def pc_stats(mode, key):
        sel = [r for r in ALL_RESULTS if r['pct'] == pc_frac and r['regime'] == mode]
        arr = np.array([r[key] for r in sel], dtype=float)
        return (np.nanmean(arr, 0), np.nanstd(arr, 0))
    pc_nl = chr(10)
    pc_head = '| class | ' + ' | '.join((f'{m} IoU | {m} F1' for m in REGIMES)) + ' |'
    pc_rows = [pc_head, '|---|' + '---:|' * (2 * len(REGIMES))]
    pc_cache = {m: (pc_stats(m, 'per_class_iou_crop'), pc_stats(m, 'per_class_f1_crop')) for m in REGIMES}
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
