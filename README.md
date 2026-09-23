# U-BARN on PASTIS

Masked pretraining of a Unet + transformer on Sentinel-2 time series, evaluated
on PASTIS crop segmentation. Method from Dumeur, Valero & Inglada, JSTARS 17
(2024) 4350–4367.

## Run

```
https://molab.marimo.io/github/minhcreus/PASTIS-test/blob/main/pastis_ubarn_molab.py
```

Attach the GPU from the notebook specs button before ②. Locally:
`uvx marimo edit pastis_ubarn_molab.py`

Two files, nothing else needed — the manifest is downloaded on first run if
absent.

| | |
|---|---|
| ① | download + build cache |
| ② | pretrain, one encoder per test fold |
| ③ | downstream: LP / FT / SL × fractions × seeds |

First pass: 200 patches, 16 dates, fractions `["10","100"]`, max epochs 10.

## Protocol

Fold 4 is the fixed early-stopping set. Each test fold trains on the three
folds that are neither test nor val: with `FOLDS = [1, 2]`, test 1 trains on
{2,3,5} and test 2 on {1,3,5}.

| | |
|---|---|
| LP | frozen encoder, classifier only |
| FT | pretrained init, everything trains |
| SL | random init, same architecture |

Fractions 1, 5, 10, 20, 50, 100%. Seeds 3 or 5, reported mean ± std over seeds
and folds.

Metrics follow Dumeur et al. (2024): OA, per-class IoU and F1, mIoU, mF1
(Kappa kept for reference). Void is ignored; background counts as a class,
matching the U-TAE convention. Absent classes are excluded from the means.

Each `(fold, fraction, seed)` maps to one subset, built before training and
handed to all three regimes; each run records a signature and a check cell
compares them. At 100% the seed only affects init and batch order. Sampling is
rare-class weighted (Appendix B) — same seed reproduces exactly, different
seeds give 0% overlap at 1%, ~13% at 10%.

One encoder per test fold: a fold's training list already excludes its test
fold and fold 4, asserted in the notebook.

## Controls that matter

**Data**

| control | default | why |
|---|---|---|
| crop | 64 | leaderboard uses 128; ~4x memory |
| date window | Jan–Nov 2019 | full series adds autumn 2018 sowing, key for winter crops |
| dates per series | 40 | PASTIS has 38–61 |
| patches | 600 | 2433 = whole dataset |

**Pretraining**

| control | default | why |
|---|---|---|
| spatial mask rate | 0.25 | block masking on top of temporal; 0 reproduces the paper exactly. The paper's conclusion names temporal-only masking as the reason its frozen maps lack spatial consistency |
| spatial block | 8 px | |
| epochs / lr | 40 / 1e-3 | AdamW, cosine with warmup |

**Downstream — FT only**

| control | default | why |
|---|---|---|
| FT strategy | LP-FT | train the head on a frozen encoder first, then unfreeze. A random head's early gradients otherwise distort pretrained features (Kumar et al., ICLR 2022). Stage 2 still trains every parameter, so this is full fine-tuning |
| LP-FT head-only epochs | 10 | |
| FT encoder lr x | 0.1 | encoder learns at a tenth of the head's rate |

**Downstream — shared by all regimes**

| control | default | why |
|---|---|---|
| loss | CE + Dice | Dice targets the macro metrics directly |
| class-balanced loss | on | inverse-sqrt frequency, from the labelled subset only |
| label smoothing | 0.05 | |
| EMA weights | on | frozen weights excluded, so LP stays bit-exact |
| test-time augmentation | on | 8 dihedral transforms, test set only |
| warmup | 3 epochs | |
| LP lr / epochs | 1e-2 / 150 | a frozen encoder needs a higher lr |
| normalise features | on | LayerNorm before the head |

FT-only levers exist to protect pretrained features; SL has none to protect, so
giving it a reduced encoder lr would only handicap the baseline. Everything
else applies identically to LP, FT and SL.

The pretraining checkpoint path encodes cache tag, mask rate, epochs and lr,
so changing any of them triggers a fresh pretrain.

## Outputs

Everything under `~/ubarn`:

```
results_downstream.json     one row per run, incl. per-class IoU/F1 and weights path
label_subsets.json          patch IDs per (fold, fraction, seed)
cache/<tag>/                preprocessed memmap
pretrain_fold<N>_<tag>/     encoder + decoder + loss history
weights/<REGIME>_f<N>_p<PCT>[_s<SEED>].pt
```

Each downstream checkpoint carries its regime, fold, fraction, seed, test
metrics, the manifest hash and the pretrain checkpoint it came from. Reload:

```python
st = torch.load(path, map_location="cpu")
enc = UBARN(in_ch=N_BANDS, d_model=st["d_model"], d_hidden=128, n_layers=3, n_heads=4)
m = SegmentationModel(enc, n_classes=st["n_classes"],
                      freeze=st["frozen_encoder"], feat_norm=st["feat_norm"])
m.load_state_dict(st["model"])
```

The download cell gives buttons for the two JSONs and a zip of all checkpoints.
Budget ~5.6 MB per downstream file: best-seed at the default grid is 36 files
(~200 MB), all-runs is 108 (~600 MB).

## Resumability

`run_button.value` resets to False once its dependent cells finish, so gating
on the button alone would destroy `cache` / `CKPTS` / `results` on any upstream
change. Each step is gated on button **or** artefact on disk:

| step | skipped when |
|---|---|
| ① | `cache/<tag>/manifest.json` exists |
| ② | `pretrain_fold<N>_m<rate>/best.pt` exists |
| ③ | `results_downstream.json` signature matches current settings |

View cells depend on `cache` / `HISTORIES` / `results`, not on buttons.
*Ancestor stopped* means the producing cell above hasn't run.

## DataLoader workers

`NUM_WORKERS = 0`. Cell-defined classes cannot be pickled, and molab starts
workers with spawn, so anything above 0 fails with `Can't get local object`.
Only fork survives. Loading is not the bottleneck — a sample is a ~3 MB memmap
read. Raising `NUM_WORKERS` passes a fork context (Linux only).

## Memory and cost

Memory scales with `batch × crop² × dates`, not batch alone. Defaults
(batch 4, 40 dates, 64²) sit near 10–12 GB under mixed precision. Won't fit in
32 GB CPU RAM; use crop 16 / 8 dates / batch 1 for a smoke test.

Downstream runs = folds × fractions × seeds × regimes, default 108, but cost
scales with the fraction. The notebook prints an hours estimate before ③ and
warns if it won't fit alongside pretraining. Sessions end at 12 h or 90 min
idle; the last cell has the upload snippet.

## Differences from the paper

Faithful: permutation masking, 60% rate, single-linear decoder, d_model 64 /
d_hidden 128 / 3 layers / 4 heads, DOY encoding with the 1000 constant,
mean-query classifier, Jan–Nov 2019 window, robust 5/95 normalisation from
training patches only, 64² centre crops, Appendix B sampling.

Different: pretrained on PASTIS training folds rather than the separate 9-tile
unlabelled corpus (Zenodo `10.5281/zenodo.7891924`), so this is
self-pretraining, not transfer; no cloud masks, so the loss weights all pixels
equally; series thinned to `t_max`; single-head mean-query attention; mixed
precision and gradient clipping.

Absolute numbers will sit below published ones. The relative comparison across
label fractions is what survives at this scale.

## Data

Sentinel-2 from `IGNF/PASTIS-HD` on Hugging Face, fetched by patch ID.
Benchmark: `VSainteuf/pastis-benchmark`, Zenodo 5012942, etalab-2.0. Labels:
0 background, 1–18 crops, 19 void (ignored in loss and metrics).
