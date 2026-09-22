# U-BARN on PASTIS — one marimo notebook

Self-supervised masked pretraining of a Unet + transformer on Sentinel-2 image
time series, evaluated on PASTIS crop segmentation over the official folds.
Method from Dumeur, Valero & Inglada, IEEE JSTARS 17 (2024) 4350–4367. The
authors' code is behind a CNRS Janus login, so this is written from the paper
and Appendix A.

## Run it

```
https://molab.marimo.io/github/minhcreus/PASTIS-test/blob/main/pastis_ubarn_molab.py
```

Attach the GPU from the **notebook specs** button before pressing ②.

Locally: `uvx marimo edit pastis_ubarn_molab.py`

## Files

| file | what it is |
|---|---|
| `pastis_ubarn_molab.py` | the whole thing — every cell, no imports to satisfy |
| `pastis_official_splits_v1.json` | split manifest; also downloaded automatically if absent |

Nothing else is needed. The notebook fetches the manifest from this repo's raw
URL on first run, so even a bare molab session with just the one file works.

## Why one file

Earlier versions split this into three notebooks importing three modules. Both
choices fail in molab:

- **Separate notebooks cannot hand off.** Each molab notebook is its own
  container, so a cache built in one is invisible to the next. There is no
  `run_config.json` bridge that survives.
- **Sibling `.py` files are not reliably materialised.** `import ubarn_model`
  works locally and fails in molab.

So: one file, one session, everything inlined.

## What you press

| | |
|---|---|
| ① | Download + build cache |
| ② | Pretrain — one encoder per fold |
| ③ | Run both folds, FR / FT / e2e |
| ④ | Label-scarcity sweep |

Everything else evaluates as you scroll. Nothing expensive starts on its own —
marimo re-runs dependent cells on any edit, so the four heavy steps sit behind
buttons and a slider nudge cannot restart a two-hour job.

First pass, set patches to 200 and dates to 16. Ten minutes, and it proves the
chain before you spend a session on it.

## Splits

`pastis_official_splits_v1.json` is the only thing defining the partition: all
2433 patch records (id, fold, dates) plus the five official rotations. The
validator checks duplicates, cross-set overlap, that every ID actually exists
in PASTIS, and whether val/test come from one official fold.

The membership check is the one that bites. A non-existent ID either crashes
the loader or is skipped silently, shrinking your test set while still
producing a number. Fold purity matters because PASTIS's folds are geographic —
a random patch-level split puts adjacent parcels on both sides of the boundary.

## Two folds

`FOLDS = [1, 2]`. Their test sets are folds 5 and 1, so the two test partitions
are disjoint and averaging them means something.

**One encoder per fold.** With two rotations a patch can be training data in
one and test data in the other, so a shared encoder would have to be pretrained
on the intersection — 968 of 2433 patches. A fold's own training list already
excludes its val and test: ~1455 patches, 50% more, for two pretraining runs.
The notebook asserts no held-out ID reaches the pretraining pool.

## Runtimes and memory

With 600 patches, 40 dates, 64² crops, on the GPU: caching 10–20 min,
pretraining ≈ 1.5–2 h per fold, one downstream run ≈ 10 min, sweep ≈ 1 h.

The transformer runs one sequence per pixel, so memory scales with
`batch × crop² × dates`, not batch alone. Measured: a step at `crop=32, T=16`
peaks ~0.6 GB above baseline, close to linear in that product. Defaults
(`batch=4, T=40, crop=64`) land near 10–12 GB under mixed precision —
comfortable on 96 GB, and batch 8 or 16 is fine. Without a GPU it will not fit
in 32 GB of RAM; drop to `crop=16, dates=8, batch=1` for a smoke test.

Sessions end at 12 hours or 90 minutes idle and the disk goes with them. Save
checkpoints and result JSONs before you close the tab — the last cell has the
snippet.

## Faithful to the paper

Permutation masking rather than constant/Gaussian substitution; 60% mask rate;
single-linear decoder; d_model 64, d_hidden 128, 3 layers, 4 heads; DOY
sinusoidal encoding with the 1000 constant; mean-query attention classifier;
Jan–Nov 2019 date window; robust 5th/95th percentile normalisation from
training patches only; 64² centre crops; rare-class-weighted sampling for small
training sets (Appendix B).

## Deliberately different

- **Pretraining corpus.** The paper pretrains on a separate 9-tile unlabelled
  Sentinel-2 set, disjoint from PASTIS. This pretrains on PASTIS training folds
  so a result fits in one session — self-pretraining, not transfer. The real
  corpus is open at Zenodo `10.5281/zenodo.7891924`, and swapping it in also
  gets you the MAJA validity masks that `reconstruction_loss` already accepts.
- No cloud masks on PASTIS, so the loss weights all pixels equally.
- Series thinned to `t_max` dates against up to 100 in the paper.
- Single-head mean-query attention in the classifier.
- Mixed precision and gradient clipping, neither of which the paper mentions.

Expect absolute numbers below the published ones. What survives at this scale
is the relative comparison between FR, FT and e2e as label count varies — the
paper's actual claim. At full labels the three land on top of each other; that
is the paper's result too, not a bug. Run ④.

Two folds give a mean and a crude spread, not a confidence interval. If the
regimes fall within one standard deviation of each other, report that rather
than picking the better fold.

## Data

Sentinel-2 series from `IGNF/PASTIS-HD` on Hugging Face, fetched by patch ID so
you only pull what the chosen folds need. Original benchmark:
`VSainteuf/pastis-benchmark`, Zenodo 5012942, Open Licence / etalab-2.0.
Labels: 0 background, 1–18 crops, 19 void (ignored in loss and metrics).
