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
    # 3 · Downstream crop segmentation — 2-fold

    Runs the official rotations in `run_config.json` (folds 1 and 2) and reports
    mean ± spread across them.

    Three regimes per fold:

    - **FR** — frozen encoder, only the shallow classifier trains. Tests whether the
      representation is any good on its own.
    - **FT** — fine-tuned from the pretrained weights.
    - **e2e** — identical architecture, random init. The honest baseline.

    Expect FT ≈ e2e at full label count; the gap opens under label scarcity. Run
    the sweep at the bottom before concluding anything about the pretraining.
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
    return Path, WORK, json, tqdm


@app.cell
def _(Path, WORK, json):
    import torch
    import numpy as np
    import pastis_splits as ps
    import ubarn_data as ud
    import ubarn_model as um
    from torch.utils.data import DataLoader
    with open(WORK / 'run_config.json') as _fh:
        RUN_CONFIG = json.load(_fh)
    MAN = ps.load_manifest(RUN_CONFIG['split_manifest'])
    assert MAN.sha256 == RUN_CONFIG['split_manifest_sha256'], 'split manifest changed'
    cache = ud.PastisCache(RUN_CONFIG['cache_dir'])
    FOLDS = RUN_CONFIG['folds']
    RUNS = ps.folds_to_run(MAN, FOLDS)
    CKPTS = RUN_CONFIG.get('pretrain_ckpts', {})

    def ckpt_for(fold):
        """Per-fold checkpoint if notebook 02 made one, else the shared encoder."""
        for key in (str(fold), 'None'):
            path = CKPTS.get(key)
            if path and Path(path).exists():
                return Path(path)
        return None
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', DEVICE)
    print('folds :', FOLDS)
    for _f in FOLDS:
        print(f'  fold {_f} ckpt:', ckpt_for(_f) or '(none — FR/FT skipped)')
    return (
        DEVICE,
        DataLoader,
        FOLDS,
        MAN,
        RUNS,
        cache,
        ckpt_for,
        np,
        torch,
        ud,
        um,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Verify the pretraining never saw these test sets

    Cheap, and catches the single most common way a reported number turns out to
    be wrong.
    """)
    return


@app.cell
def _(FOLDS, MAN, RUNS, ckpt_for, torch):
    any_ckpt = False
    for _f in FOLDS:
        path = ckpt_for(_f)
        if path is None:
            print(f'fold {_f}: no checkpoint')
            continue
        any_ckpt = True
        st = torch.load(path, map_location='cpu')
        assert st.get('split_manifest_sha256') in (None, MAN.sha256), f'fold {_f} checkpoint was pretrained against a different split manifest'
        pre = set(st.get('pretrain_ids', []))
        bad_te = pre & set(RUNS[_f]['test'])
        bad_va = pre & set(RUNS[_f]['val'])
        print(f'fold {_f}: pretrained on {len(pre)} ids | ∩test={len(bad_te)} ∩val={len(bad_va)}')
        assert not bad_te and (not bad_va), f'LEAK in fold {_f}'
        del st
    print('\nclean ✓' if any_ckpt else '\nno checkpoints — only e2e is meaningful')
    return


@app.cell
def _(DEVICE, DataLoader, RUNS, cache, ckpt_for, torch, tqdm, ud, um):
    def build_model(mode, fold):
        enc = um.UBARN(in_ch=ud.N_BANDS, d_model=64, d_hidden=128, n_layers=3, n_heads=4)
        if mode in ("FR", "FT"):
            path = ckpt_for(fold)
            if path is None:
                raise RuntimeError(f"{mode} requested but no checkpoint for fold {fold}")
            enc.load_state_dict(torch.load(path, map_location="cpu")["encoder"])
        model = um.SegmentationModel(enc, n_classes=ud.N_CLASSES, freeze=(mode == "FR"))
        return model.to(DEVICE)


    def fold_loaders(fold, n_train=None, batch_size=4, seed=0):
        sp = RUNS[fold]
        tr = cache.indices_for(sp["train"])
        va = cache.indices_for(sp["val"])
        te = cache.indices_for(sp["test"])
        if n_train is not None:
            tr = ud.scarce_subset(cache, tr, n_train, seed=seed)
        mk = lambda idx, shuf, aug: DataLoader(
            ud.make_dataset(cache, idx, augment=aug), batch_size=batch_size,
            shuffle=shuf, num_workers=2, pin_memory=(DEVICE == "cuda"))
        return mk(tr, True, True), mk(va, False, False), mk(te, False, False), len(tr), len(te)


    @torch.no_grad()
    def evaluate(model, dl):
        model.eval()
        meter = um.ConfusionMeter(ud.N_CLASSES, ignore_index=ud.VOID_CLASS)
        for b in dl:
            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                logits = model(b["x"].to(DEVICE), b["doy"].to(DEVICE), b["valid"].to(DEVICE))
            meter.update(logits.argmax(1), b["y"])
        return meter


    def train_one(model, tr_dl, va_dl, epochs, lr=1e-3, quiet=False):
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
        scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))
        lossf = torch.nn.CrossEntropyLoss(ignore_index=ud.VOID_CLASS)

        best, best_state = -1.0, None
        it = range(epochs) if quiet else tqdm(range(epochs), desc="train", leave=False)
        for _ in it:
            model.train()
            tot = 0.0
            for b in tr_dl:
                with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                    logits = model(b["x"].to(DEVICE), b["doy"].to(DEVICE), b["valid"].to(DEVICE))
                    loss = lossf(logits, b["y"].to(DEVICE))
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                scaler.step(opt); scaler.update()
                tot += loss.item()
            sched.step(tot / max(len(tr_dl), 1))
            miou = evaluate(model, va_dl).scores()["mIoU"]
            if miou > best:
                best = miou
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if best_state:
            model.load_state_dict(best_state)
        return best

    return build_model, evaluate, fold_loaders, train_one


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run the two folds
    """)
    return


@app.cell
def _(mo):
    run_folds = mo.ui.run_button(label="Run both folds")
    run_folds
    return (run_folds,)


@app.cell
def _(
    DEVICE,
    FOLDS,
    build_model,
    ckpt_for,
    evaluate,
    fold_loaders,
    mo,
    run_folds,
    torch,
    train_one,
):
    mo.stop(not run_folds.value, mo.md('Press **Run both folds** to train FR / FT / e2e on each fold.'))
    EPOCHS = 40
    BATCH_SIZE = 4
    N_TRAIN = None
    REGIMES = ['FR', 'FT', 'e2e']  # None = all training patches; set e.g. 30 for the scarce regime
    results = []
    for _fold in FOLDS:
        _tr, _va, _te, _n_tr, n_te = fold_loaders(_fold, N_TRAIN, BATCH_SIZE)
        for _mode in REGIMES:
            if _mode in ('FR', 'FT') and ckpt_for(_fold) is None:
                continue
            torch.manual_seed(_fold)
            model = build_model(_mode, _fold)
            train_one(model, _tr, _va, EPOCHS)
            _sc = evaluate(model, _te).scores()
            results.append({'fold': _fold, 'regime': _mode, 'n_train': _n_tr, 'n_test': n_te, **{k: _sc[k] for k in ('Kappa', 'OA', 'F1', 'mIoU')}})
            print(f'fold {_fold} · {_mode:3s} · train {_n_tr:4d} · test {n_te:4d} | Kappa {_sc['Kappa']:.4f}  OA {_sc['OA']:.4f}  F1 {_sc['F1']:.4f}  mIoU {_sc['mIoU']:.4f}')
            del model
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()
    return BATCH_SIZE, REGIMES, results


@app.cell
def _(REGIMES, WORK, json, np, results):
    # mean +/- spread across the two folds
    print(f'{'regime':<8}' + ''.join((f'{m:>18}' for m in ('Kappa', 'OA', 'F1', 'mIoU'))))
    for _mode in REGIMES:
        rows = [r for r in results if r['regime'] == _mode]
        if not rows:
            continue
        line = f'{_mode:<8}'
        for _m in ('Kappa', 'OA', 'F1', 'mIoU'):
            v = np.array([r[_m] for r in rows])
            line += f'{v.mean():>11.4f} ±{v.std():.4f}'
        print(line)
    with open(WORK / 'results_2fold.json', 'w') as _fh:
        json.dump(results, _fh, indent=2)
    print('\nsaved ->', WORK / 'results_2fold.json')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Two folds give you a mean and a crude spread, not a confidence interval. If FR,
    FT and e2e land within one standard deviation of each other, the honest
    reading is that this setup cannot separate them — report it that way rather
    than picking the winning fold.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Label-scarcity sweep

    The experiment that actually decides whether the pretraining worked. FT and e2e
    at each training-set size, both folds. Budget: `len(sizes) × 2 regimes × 2 folds`
    training runs.
    """)
    return


@app.cell
def _(mo):
    run_sweep = mo.ui.run_button(label="Run scarcity sweep")
    run_sweep
    return (run_sweep,)


@app.cell
def _(
    BATCH_SIZE,
    DEVICE,
    FOLDS,
    WORK,
    build_model,
    ckpt_for,
    evaluate,
    fold_loaders,
    json,
    mo,
    run_sweep,
    torch,
    train_one,
):
    mo.stop(not run_sweep.value, mo.md('Press **Run scarcity sweep**. This is `len(SIZES) x 2 regimes x 2 folds` training runs.'))
    SIZES = [10, 30, 100]
    SWEEP_EPOCHS = 30
    sweep = []
    for size in SIZES:
        for _fold in FOLDS:
            _tr, _va, _te, _n_tr, _ = fold_loaders(_fold, size, BATCH_SIZE)
            for _mode in ['FT', 'e2e']:
                if _mode == 'FT' and ckpt_for(_fold) is None:
                    continue
                torch.manual_seed(_fold)
                _m = build_model(_mode, _fold)
                train_one(_m, _tr, _va, SWEEP_EPOCHS, quiet=True)
                _sc = evaluate(_m, _te).scores()
                sweep.append({'n': _n_tr, 'fold': _fold, 'regime': _mode, **{k: _sc[k] for k in ('Kappa', 'OA', 'F1', 'mIoU')}})
                print(f'n={_n_tr:4d} fold={_fold} {_mode:3s} mIoU={_sc['mIoU']:.4f}')
                del _m
                if DEVICE == 'cuda':
                    torch.cuda.empty_cache()
    with open(WORK / 'sweep_2fold.json', 'w') as _fh:
        json.dump(sweep, _fh, indent=2)
    return (sweep,)


@app.cell
def _(np, sweep):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.2))
    for ax, metric in zip(axs, ['Kappa', 'OA', 'F1', 'mIoU']):
        for _mode, style in [('FT', '-o'), ('e2e', '--s')]:
            xs = sorted({r['n'] for r in sweep if r['regime'] == _mode})
            mu = [np.mean([r[metric] for r in sweep if r['regime'] == _mode and r['n'] == x]) for x in xs]
            sd = [np.std([r[metric] for r in sweep if r['regime'] == _mode and r['n'] == x]) for x in xs]
            if xs:
                ax.errorbar(xs, mu, yerr=sd, fmt=style, capsize=3, ms=4, label=_mode)
        ax.set_xscale('log')
        ax.set_xlabel('training patches')
        ax.set_title(metric)
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reading the result

    FT above e2e at small `n`, converging as `n` grows, reproduces the paper's
    central claim. If FT never separates: too few pretraining epochs, a mask rate
    that is off, or leakage between the pretraining pool and the test folds. The
    assertions in notebook 02 and at the top of this one rule out the third.
    """)
    return


if __name__ == "__main__":
    app.run()
