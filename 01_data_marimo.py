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
    # 1 · Splits and data preparation

    Establishes **one** source of truth for the partition — `pastis_official_splits_v1.json` —
    and builds the preprocessed cache that notebooks 02 and 03 read.

    Everything keyed off patch ID, so the same partition holds across pretraining,
    fine-tuning and evaluation. Runs unchanged on molab, Kaggle, Colab or locally.
    """)
    return


@app.cell
def _():
    # --- environment bootstrap -------------------------------------------------
    import os, sys, json
    from pathlib import Path

    def find_file(*names, extra=()):
        """Locate a file across molab / Kaggle / Colab / local layouts."""
        roots = [Path.cwd(), Path.cwd().parent, Path.home(),
                 Path("/kaggle/working"), Path("/kaggle/input"),
                 Path("/content"), *[Path(e) for e in extra]]
        for r in roots:
            if not r.exists():
                continue
            for n in names:
                p = r / n
                if p.exists():
                    return p
            for n in names:
                hits = sorted(r.glob(f"*/{n}")) + sorted(r.glob(f"*/*/{n}"))
                if hits:
                    return hits[0]
        return None

    # put the shared modules on the path
    _mod = find_file("ubarn_model.py")
    MODULE_DIR = _mod.parent if _mod else Path.cwd()
    if str(MODULE_DIR) not in sys.path:
        sys.path.insert(0, str(MODULE_DIR))

    # where caches and checkpoints live (Kaggle only allows writes under /kaggle/working)
    WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.home() / "ubarn"
    WORK.mkdir(parents=True, exist_ok=True)

    IN_KAGGLE = Path("/kaggle/input").exists()

    try:
        from tqdm.auto import tqdm
    except ImportError:                      # keep the notebooks runnable without tqdm
        class tqdm:
            def __init__(self, iterable=None, total=None, desc=None, leave=True, **kw):
                self.iterable, self.n, self.total, self.desc = iterable, 0, total, desc
            def __iter__(self):
                for x in self.iterable or ():
                    yield x
            def update(self, n=1):
                self.n += n
            def close(self):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False

    print("modules :", MODULE_DIR)
    print("work dir:", WORK)
    return WORK, find_file, json, tqdm


@app.cell
def _(find_file):
    import pastis_splits as ps

    SPLIT_MANIFEST = find_file("pastis_official_splits_v1.json")
    if SPLIT_MANIFEST is None:
        raise FileNotFoundError(
            "pastis_official_splits_v1.json not found. Upload it next to this "
            "notebook (or to /kaggle/working) before running."
        )

    MAN = ps.load_manifest(SPLIT_MANIFEST)
    print("manifest :", SPLIT_MANIFEST)
    print("sha256   :", MAN.sha256[:16], "...")
    print("patches  :", len(MAN.ids))
    print("folds    :", MAN.fold_sizes())
    print("labels   : background=%s, crops=%d, void=%s" % (
        MAN.labels["background"], len(MAN.labels["crop_ids"]), MAN.labels["void"]))
    return MAN, SPLIT_MANIFEST, ps


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Integrity check

    Stronger than a duplicate/overlap check. It also verifies that every ID in a
    split **exists in PASTIS**, and reports whether val/test are drawn from a
    single official fold — a split that mixes folds puts spatially adjacent
    parcels on both sides of the boundary.
    """)
    return


@app.cell
def _(MAN, ps):
    for _f in sorted(MAN.splits, key=int):
        rep = ps.validate_split(MAN.split(_f), MAN, name=f"official fold {_f}")
        ps.print_report(rep)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Comparing against an existing split file

    If you already have a split lying around, run it through the same validator
    before trusting it. The membership check is the one that matters: an ID that
    does not exist in PASTIS will either crash the loader or, worse, be skipped
    silently and quietly shrink your test set.
    """)
    return


@app.cell
def _(MAN, find_file, json, ps):
    _legacy = find_file('pastis_splits.json')
    if _legacy is not None:
        with open(_legacy) as _fh:
            legacy = json.load(_fh)
        ps.print_report(ps.validate_split(legacy, MAN, name=f'legacy · {_legacy.name}'))
    else:
        print('no legacy split file found — skipping')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Which folds to run

    Two folds, per the plan. Folds 1 and 2 are the first two official rotations:
    their test sets are folds 5 and 1 respectively, so between them you get two
    disjoint test partitions and a fair average.
    """)
    return


@app.cell
def _(MAN, ps):
    FOLDS = [1, 2]
    RUNS = ps.folds_to_run(MAN, FOLDS)
    NEEDED = sorted(ps.required_ids(RUNS))
    for _f, _sp in RUNS.items():
        te_folds = sorted({MAN.fold_of(i) for i in _sp['test']})
        va_folds = sorted({MAN.fold_of(i) for i in _sp['val']})
        print(f'fold {_f}: train={len(_sp['train']):5d}  val={len(_sp['val']):4d} (fold {va_folds})  test={len(_sp['test']):4d} (fold {te_folds})')
    print(f'\nunique patches needed for {FOLDS}: {len(NEEDED)}')
    return FOLDS, NEEDED, RUNS


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Optional: work on a subset first

    The full 2433-patch download is ~34 GB. On an ephemeral runtime, prove the
    pipeline works on a few hundred patches, then scale. Subsetting is done
    **within each split** so the train/val/test proportions are preserved and no
    patch ever changes side.
    """)
    return


@app.cell
def _(NEEDED, RUNS):
    SUBSET = 600  # None = use everything the folds need
    CFG_T_MAX = 40  # dates per series
    CFG_CROP = 64  # spatial crop
    import ubarn_data as ud
    cfg = ud.DataConfig(t_max=CFG_T_MAX, crop=CFG_CROP)
    if SUBSET is not None and SUBSET < len(NEEDED):
        import numpy as _np
        _rng = _np.random.default_rng(cfg.seed)
        _keep = set()
        for _f, _sp in RUNS.items():
            for _part in ('train', 'val', 'test'):
                ids = _sp[_part]
                k = max(1, round(SUBSET * len(ids) / len(NEEDED)))
                _keep.update(_rng.choice(ids, size=min(k, len(ids)), replace=False).tolist())
        USE_IDS = sorted(_keep)
    else:
        USE_IDS = NEEDED
    print(f'caching {len(USE_IDS)} patches')
    print(f'download  ~{len(USE_IDS) * 14.1 / 1024:.1f} GB')
    print(f'cache     ~{len(USE_IDS) * cfg.t_max * 10 * cfg.crop ** 2 * 2 / 1000000000.0:.1f} GB')
    return USE_IDS, cfg, ud


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fetch and preprocess

    Uses a local copy of PASTIS if it finds one (a Kaggle dataset attached under
    `/kaggle/input`, a mounted drive); otherwise pulls the needed patches from the
    Hugging Face mirror.

    Normalisation statistics come from **fold 1's training IDs only**. Computing
    them over the whole dataset would leak test-set statistics into every run.
    """)
    return


@app.cell
def _(mo):
    run_build = mo.ui.run_button(label="Download + build cache")
    run_build
    return (run_build,)


@app.cell
def _(FOLDS, MAN, RUNS, USE_IDS, WORK, cfg, mo, run_build, tqdm, ud):
    mo.stop(not run_build.value, mo.md("Press **Download + build cache** above. This fetches the patches and preprocesses them — minutes to tens of minutes."))

    RAW_ROOT, how = ud.resolve_source(USE_IDS, WORK, local_hint=None)
    print("raw data from:", how)

    _norm_ids = [i for i in RUNS[FOLDS[0]]["train"] if i in set(USE_IDS)]
    print(f"normalisation stats from {len(_norm_ids)} fold-{FOLDS[0]} training patches")

    _bar = tqdm(total=len(USE_IDS), desc="preprocessing")
    CACHE_DIR = ud.build_cache(RAW_ROOT, MAN, USE_IDS, cfg, WORK / "cache",
                               norm_ids=_norm_ids, progress=_bar.update)
    _bar.close()

    cache = ud.PastisCache(CACHE_DIR)
    print("\ncache:", CACHE_DIR)
    print(f"{len(cache)} patches · T={cache.x.shape[1]} · {cache.x.shape[-1]}px")
    return CACHE_DIR, cache


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Coverage of each split after caching

    If you subset, some IDs in a split will not be on disk. This reports exactly
    how much of each split survived — check it before reading any metric, because
    a test set that quietly lost half its patches will still produce a number.
    """)
    return


@app.cell
def _(RUNS, cache):
    for _f, _sp in RUNS.items():
        row = []
        for _part in ('train', 'val', 'test'):
            have, want = cache.coverage(_sp[_part])
            row.append(f'{_part}={have}/{want}')
        print(f'fold {_f}: ' + '  '.join(row))
    return


@app.cell
def _(CACHE_DIR, FOLDS, MAN, SPLIT_MANIFEST, WORK, cache, cfg, json):
    # persist the exact run configuration alongside the cache
    RUN_CONFIG = {'split_manifest': str(SPLIT_MANIFEST), 'split_manifest_sha256': MAN.sha256, 'folds': FOLDS, 'cache_dir': str(CACHE_DIR), 'cached_ids': len(cache), 't_max': cfg.t_max, 'crop': cfg.crop, 'date_window': [cfg.date_start, cfg.date_end]}
    with open(WORK / 'run_config.json', 'w') as _fh:
        json.dump(RUN_CONFIG, _fh, indent=2)
    print(json.dumps(RUN_CONFIG, indent=2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sanity check
    """)
    return


@app.cell
def _(cache, ud):
    import matplotlib.pyplot as plt
    import numpy as np

    _i, _t = 0, min(10, cache.x.shape[1] - 1)
    _x = cache.denormalize(np.asarray(cache.x[_i, _t], dtype=np.float32))
    _rgb = np.stack([_x[2], _x[1], _x[0]], -1)
    _rgb = np.clip(_rgb / max(np.percentile(_rgb, 98), 1e-6), 0, 1)
    _y = cache.target[_i].astype(float)
    _y[_y == ud.VOID_CLASS] = np.nan

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(_rgb); ax[0].set_title(f"patch {cache.ids[_i]} · DOY {cache.doy[_i, _t]}")
    ax[1].imshow(_y, cmap="tab20", vmin=0, vmax=19, interpolation="nearest")
    ax[1].set_title(f"labels · fold {cache.folds[_i]}")
    for a in ax:
        a.axis("off")
    plt.tight_layout(); plt.show()
    return (np,)


@app.cell
def _(cache, np, ud):
    _counts = np.bincount(cache.target.reshape(-1), minlength=ud.N_CLASSES)
    print(f"{'class':<28}{'pixels':>12}{'share':>9}")
    for _c in range(ud.N_CLASSES):
        if _counts[_c]:
            print(f"{ud.CLASS_NAMES[_c]:<28}{_counts[_c]:>12,}{100*_counts[_c]/_counts.sum():>8.2f}%")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    Next: **02_pretrain.ipynb**. It reads `run_config.json`, so no paths to copy.
    """)
    return


if __name__ == "__main__":
    app.run()
