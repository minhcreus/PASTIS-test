# U-BARN on PASTIS — 2-fold, .ipynb

Self-supervised masked pretraining of a Unet + transformer on Sentinel-2 image
time series, evaluated on PASTIS crop segmentation over the official folds.
Reimplemented from:

> I. Dumeur, S. Valero, J. Inglada. *Self-Supervised Spatio-Temporal
> Representation Learning of Satellite Image Time Series.* IEEE JSTARS 17
> (2024) 4350–4367.

The authors' code sits on a CNRS GitLab behind a Janus login, so this is
written from the paper text and Appendix A. Everything depends only on openly
downloadable data.

## Files

| file | what it is |
|---|---|
| `pastis_official_splits_v1.json` | the split manifest — single source of truth |
| `pastis_splits.py` | manifest loader + integrity validator |
| `ubarn_data.py` | data resolution, preprocessing, memmap cache, dataset |
| `ubarn_model.py` | SSE, DOY encoding, transformer, permutation masking, heads, metrics |
| `01_data.ipynb` | validate splits → fetch → cache → inspect |
| `02_pretrain.ipynb` | SSL pretraining, one encoder per fold |
| `03_downstream_2fold.ipynb` | FR / FT / e2e across both folds + scarcity sweep |
| `*_marimo.py` | the same three notebooks in marimo format, with run-button gates |
| `SETUP_MARIMO.md` | step-by-step for molab and marimo |

Pick one format — Jupyter or marimo — and run them in order. They hand off through `run_config.json` in the work
directory, so there are no paths to copy between notebooks.

## Split discipline

`pastis_official_splits_v1.json` carries all 2433 patch records (id, fold,
acquisition dates) plus the five official rotations. Nothing else defines the
partition — the notebooks never re-derive folds, never sample their own
train/test, and assert on the manifest's SHA-256 at every stage so a changed
manifest cannot silently invalidate a checkpoint.

The validator in `pastis_splits.py` checks four things:

1. duplicate IDs within a set,
2. overlap between train / val / test,
3. **membership** — every ID actually exists in PASTIS,
4. **fold purity** — whether val and test come from a single official fold.

Point 3 is the one that usually bites. An ID that does not exist either crashes
the loader or gets skipped silently, quietly shrinking the test set while still
producing a number. Point 4 matters because PASTIS's folds are geographic: a
random patch-level split puts spatially adjacent parcels on both sides of the
boundary, inflating scores and making results incomparable to the leaderboard.

### On `pastis_splits.json`

That file does not pass. It claims 2468 patches against PASTIS's real 2433, and
**35 of its IDs do not exist** — 30 in train, 5 in test (`10095`, `10236`,
`20425`, `30314`, …). It is also a random 80/10/10 that cuts across all five
official folds. Notebook 01 runs it through the validator so you can see the
report, but the pipeline uses the official manifest.

## Two folds

`FOLDS = [1, 2]`. Their test sets are folds 5 and 1 respectively, so the two
test partitions are disjoint and averaging them is meaningful.

**Per-fold pretraining is the default.** With two rotations a patch can be
training data in one and test data in the other, so a single shared encoder
would have to be pretrained on the intersection — 968 of 2433 patches for folds
1 and 2. Pretraining one encoder per fold uses that fold's own training list,
which by construction excludes its val and test: ~1455 patches, 50% more data,
at the cost of two pretraining runs. Set `PRETRAIN_PER_FOLD = False` in
notebook 02 if compute is the binding constraint.

Notebook 03 re-verifies, from the IDs stored inside each checkpoint, that the
encoder never saw that fold's val or test patches, and asserts if it did.

## Where it runs

Environment detection is automatic — molab, Kaggle, Colab or local.

- **molab**: push to GitHub, open via `molab.marimo.io/github` (it opens
  Jupyter notebooks as well as marimo ones), turn the GPU on from the notebook
  specs button. 4 CPUs / 32 GB RAM, optional RTX Pro 6000 (96 GB), 12-hour
  sessions, ephemeral disk.
- **Kaggle**: attach a PASTIS dataset under `/kaggle/input` and it is used
  directly, no download. Output goes to `/kaggle/working`, which persists.
- Anywhere else: patches are fetched from the `IGNF/PASTIS-HD` Hugging Face
  mirror, by ID, so you only pull what the chosen folds need.

Upload `pastis_official_splits_v1.json` next to the notebooks (or to
`/kaggle/working`); `_find()` locates it.

## Runtimes and memory

With `SUBSET=600`, `t_max=40`, 64² crops, on a GPU: caching 10–20 min,
pretraining 40 epochs ≈ 1.5–2 h per fold, one downstream run ≈ 10 min, the
scarcity sweep ≈ 1 h.

The transformer runs one sequence per pixel, so activation memory scales with
`batch × crop² × dates`, not batch alone. Measured: a pretraining step at
`crop=32, T=16` peaks ~0.6 GB above baseline and scales close to linearly in
that product. The defaults (`batch=4, T=40, crop=64`) land near 20 GB in fp32,
10–12 GB under the mixed precision the notebooks use. Comfortable on 96 GB; you
can push batch to 8 or 16. It will **not** fit in 32 GB of CPU RAM — drop to
`crop=32, t_max=16, batch=1` for a smoke test.

Start with `SUBSET=200`, `CFG_T_MAX=16`, `EPOCHS=3` to confirm the chain runs
before spending a session on it.

## Faithful to the paper

Permutation masking rather than constant/Gaussian substitution; 60% mask rate;
single-linear decoder; d_model 64, d_hidden 128, 3 layers, 4 heads; DOY
sinusoidal encoding with the 1000 constant; mean-query attention classifier;
Jan–Nov 2019 date window; robust 5th/95th percentile normalisation computed on
training patches only; 64² centre crops; rare-class-weighted sampling for small
training sets (Appendix B).

## Deliberately different

- **Pretraining corpus.** The paper pretrains on a separate 9-tile unlabelled
  Sentinel-2 set, geographically disjoint from PASTIS. These notebooks pretrain
  on PASTIS training folds so a result fits in one session — self-pretraining,
  not transfer. The proper corpus is open at Zenodo `10.5281/zenodo.7891924`;
  swapping it in also gets you the MAJA validity masks that
  `reconstruction_loss` already accepts.
- **No cloud masks on PASTIS**, so the loss weights all pixels equally where
  the paper down-weights invalid ones.
- Series thinned to `t_max` dates, against up to 100 in the paper.
- Single-head mean-query attention in the classifier.
- Mixed precision and gradient clipping, neither of which the paper mentions.

Expect absolute numbers below the published ones. What is reproducible at this
scale is the *relative* comparison between FR, FT and e2e as label count
varies — which is the paper's actual claim. Two folds give a mean and a crude
spread, not a confidence interval: if the three regimes land within one
standard deviation of each other, report that rather than picking the better
fold.

## Data

Sentinel-2 series from `IGNF/PASTIS-HD` on Hugging Face. Original benchmark:
`VSainteuf/pastis-benchmark`, Zenodo record 5012942. Licence: Open Licence /
etalab-2.0. Labels: 0 = background, 1–18 = crop types, 19 = void (ignored in
the loss and in every metric).
