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
    # 2 · Self-supervised pretraining

    BERT-style pretext task: corrupt a fraction of dates at the output of the
    spatio-spectral encoder, then reconstruct the original reflectances from the
    transformer's representation. Masked values are **permuted** from elsewhere in
    the batch rather than zeroed — the paper's main departure from SITS-Former,
    and what keeps the activation distribution close to inference conditions.

    **Split discipline.** Pretraining uses only the union of the *training* IDs of
    the folds being run. Val and test patches are never seen, not even unlabelled.
    That is what makes the downstream comparison in notebook 03 meaningful.
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
    return WORK, json, tqdm


@app.cell
def _(WORK, json):
    import torch
    import numpy as np
    import pastis_splits as ps
    import ubarn_data as ud
    import ubarn_model as um
    with open(WORK / 'run_config.json') as _fh:
        RUN_CONFIG = json.load(_fh)
    MAN = ps.load_manifest(RUN_CONFIG['split_manifest'])
    assert MAN.sha256 == RUN_CONFIG['split_manifest_sha256'], 'split manifest changed since notebook 01 — rebuild the cache'
    cache = ud.PastisCache(RUN_CONFIG['cache_dir'])
    FOLDS = RUN_CONFIG['folds']
    RUNS = ps.folds_to_run(MAN, FOLDS)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', DEVICE, torch.cuda.get_device_name(0) if DEVICE == 'cuda' else '')
    print('folds :', FOLDS, '| cached patches:', len(cache))
    return DEVICE, FOLDS, MAN, RUNS, RUN_CONFIG, cache, np, torch, ud, um


@app.cell
def _(FOLDS, RUNS, cache):
    PRETRAIN_PER_FOLD = True  # one encoder per fold; False = one shared encoder

    def pretrain_pool(fold=None):
        """IDs that may be pretrained on without touching any held-out patch."""
        if fold is not None:
            return set(RUNS[fold]['train'])  # a fold's own training list already excludes its val and test
        tr, ho = (set(), set())
        for f in FOLDS:
            tr |= set(RUNS[f]['train'])
            ho |= set(RUNS[f]['val']) | set(RUNS[f]['test'])
        return tr - ho
    if PRETRAIN_PER_FOLD:
        POOLS = {f: pretrain_pool(f) for f in FOLDS}
    else:
        POOLS = {None: pretrain_pool()}
    for _key, _pool in POOLS.items():
        idx = cache.indices_for(sorted(_pool))
        ho = set()
        for _f in [_key] if _key is not None else FOLDS:
            ho |= set(RUNS[_f]['val']) | set(RUNS[_f]['test'])
        leak = {cache.ids[i] for i in idx} & ho
        assert not leak, f'LEAK: {len(leak)} held-out patches in pool {_key}'
        label = f'fold {_key}' if _key is not None else 'shared'
        print(f'{label:>10}: {len(_pool):5d} ids -> {len(idx):5d} cached, no leakage')
    return POOLS, PRETRAIN_PER_FOLD


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Per-fold vs shared pretraining

    With two folds, fold 1's test set is fold 5 and fold 2's test set is fold 1 — a
    patch can be *training* data in one rotation and *test* data in the other. A
    single shared encoder must therefore be pretrained on the intersection,
    subtracting every held-out ID across both folds. For folds 1 and 2 that leaves
    **968** patches out of 2433.

    Pretraining one encoder per fold instead uses that fold's own training list,
    which by construction excludes its val and test: **~1455** patches, 50% more
    data. The cost is two pretraining runs instead of one. `PRETRAIN_PER_FOLD=True`
    is the default for that reason; flip it if compute is the binding constraint.
    """)
    return


@app.cell
def _(DEVICE, MAN, WORK, cache, torch, tqdm, ud, um):
    MASK_RATE  = 0.60     # paper: optima at 0.30 and 0.60, collapse past 0.80
    EPOCHS     = 40
    BATCH_SIZE = 4
    LR         = 1e-3

    from torch.utils.data import DataLoader


    def pretrain(pool_ids, tag):
        idx = cache.indices_for(sorted(pool_ids))
        dl = DataLoader(ud.make_dataset(cache, idx, augment=True),
                        batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
                        pin_memory=(DEVICE == "cuda"), drop_last=True)

        torch.manual_seed(0)
        enc = um.UBARN(in_ch=ud.N_BANDS, d_model=64, d_hidden=128,
                       n_layers=3, n_heads=4).to(DEVICE)
        dec = um.LinearDecoder(64, ud.N_BANDS).to(DEVICE)
        params = list(enc.parameters()) + list(dec.parameters())
        opt = torch.optim.Adam(params, lr=LR)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
        scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

        out = WORK / f"pretrain_{tag}_m{int(MASK_RATE*100)}"
        out.mkdir(parents=True, exist_ok=True)
        hist = []

        for ep in range(EPOCHS):
            enc.train(); dec.train()
            tot, nb = 0.0, 0
            for b in tqdm(dl, desc=f"{tag} ep {ep+1}/{EPOCHS}", leave=False):
                x = b["x"].to(DEVICE, non_blocking=True)
                doy = b["doy"].to(DEVICE, non_blocking=True)
                valid = b["valid"].to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                    f = enc.embed(x, doy)
                    corrupt, dmask = um.permutation_mask(f, valid, MASK_RATE)
                    rec = dec(enc.temporal(corrupt, valid))
                    loss = um.reconstruction_loss(rec, x, dmask)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                scaler.step(opt); scaler.update()
                tot += loss.item(); nb += 1

            ep_loss = tot / max(nb, 1)
            hist.append(ep_loss)
            sched.step(ep_loss)
            if ep_loss <= min(hist):
                torch.save({"encoder": enc.state_dict(), "decoder": dec.state_dict(),
                            "mask_rate": MASK_RATE, "epoch": ep, "loss": ep_loss,
                            "tag": tag, "split_manifest_sha256": MAN.sha256,
                            "pretrain_ids": sorted(pool_ids)}, out / "best.pt")
            print(f"[{tag}] epoch {ep+1:3d}  masked MSE {ep_loss:.5f}")

        return out / "best.pt", hist, enc, dec

    return EPOCHS, MASK_RATE, pretrain


@app.cell
def _(mo):
    run_pretrain = mo.ui.run_button(label="Start pretraining")
    run_pretrain
    return (run_pretrain,)


@app.cell
def _(POOLS, mo, pretrain, run_pretrain):
    mo.stop(not run_pretrain.value, mo.md('Press **Start pretraining**. One run per fold, roughly 1.5–2 h each on a GPU at the default settings.'))
    CKPTS, HISTORIES = ({}, {})
    for _key, _pool in POOLS.items():
        _tag = f'fold{_key}' if _key is not None else 'shared'
        ckpt, _hist, encoder, decoder = pretrain(_pool, _tag)
        CKPTS[str(_key)] = str(ckpt)
        HISTORIES[_tag] = _hist
        print(f'{_tag}: best {min(_hist):.5f} -> {ckpt}\n')
    return CKPTS, HISTORIES, decoder, encoder


@app.cell
def _(HISTORIES):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 3))
    for _tag, _hist in HISTORIES.items():
        plt.plot(_hist, lw=1.5, label=_tag)
    plt.xlabel('epoch')
    plt.ylabel('masked MSE')
    plt.yscale('log')
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reconstruction check

    Masked dates should come back with plausible temporal dynamics. Cloudy dates
    should be *smoothed away* rather than reproduced — the model learns to treat
    cloud as an outlier in the temporal profile.

    This is run on a **held-out** patch, so it is a genuine check rather than
    memorisation.
    """)
    return


@app.cell
def _(
    DEVICE,
    FOLDS,
    MASK_RATE,
    RUNS,
    cache,
    decoder,
    encoder,
    np,
    plt,
    torch,
    um,
):
    encoder.eval()
    decoder.eval()
    _ho = set()
    for _f in FOLDS:
        _ho |= set(RUNS[_f]['val']) | set(RUNS[_f]['test'])
    _rows = cache.indices_for(sorted(_ho))
    _i = int(_rows[0]) if len(_rows) else 0
    x1 = torch.from_numpy(np.asarray(cache.x[_i], dtype=np.float32))[None].to(DEVICE)
    d1 = torch.from_numpy(cache.doy[_i].astype(np.float32))[None].to(DEVICE)
    v1 = torch.from_numpy(cache.valid[_i].copy())[None].to(DEVICE)
    with torch.no_grad():
        f1 = encoder.embed(x1, d1)
        c1, m1 = um.permutation_mask(f1, v1, MASK_RATE)
        r1 = decoder(encoder.temporal(c1, v1))

    def rgb(t4):
        a = cache.denormalize(t4)
        img = np.stack([a[2], a[1], a[0]], -1)
        return np.clip(img / max(np.percentile(img, 98), 1e-06), 0, 1)
    sel = torch.nonzero(m1[0]).flatten().cpu().numpy()[:6]
    fig, axes = plt.subplots(2, len(sel), figsize=(2.1 * len(sel), 4.4), squeeze=False)
    for c, t in enumerate(sel):
        axes[0][c].imshow(rgb(x1[0, t].cpu().numpy()))
        axes[0][c].set_title(f'DOY {cache.doy[_i, t]}', fontsize=8)
        axes[1][c].imshow(rgb(r1[0, t].float().cpu().numpy()))
        axes[0][c].axis('off')
        axes[1][c].axis('off')
    fig.suptitle(f'held-out patch {cache.ids[_i]} — top: masked input · bottom: reconstruction', fontsize=9)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Persist the checkpoint

    molab and Colab wipe the filesystem at the end of the session. On Kaggle,
    anything under `/kaggle/working` survives as notebook output. Otherwise push it
    to the Hub:

    ```python
    from huggingface_hub import HfApi
    HfApi().upload_file(path_or_fileobj=CKPTS["1"],
                        path_in_repo="ubarn_pastis_m60.pt",
                        repo_id="<user>/ubarn-pastis", repo_type="model",
                        token="hf_...")
    ```
    """)
    return


@app.cell
def _(CKPTS, EPOCHS, MASK_RATE, PRETRAIN_PER_FOLD, RUN_CONFIG, WORK, json):
    RUN_CONFIG['pretrain_ckpts'] = CKPTS  # {"1": path, "2": path} or {"None": path}
    RUN_CONFIG['pretrain_per_fold'] = PRETRAIN_PER_FOLD
    RUN_CONFIG['mask_rate'] = MASK_RATE
    RUN_CONFIG['pretrain_epochs'] = EPOCHS
    with open(WORK / 'run_config.json', 'w') as _fh:
        json.dump(RUN_CONFIG, _fh, indent=2)
    print(json.dumps(CKPTS, indent=2))
    print('run_config.json updated — notebook 03 will pick this up')
    return


if __name__ == "__main__":
    app.run()
