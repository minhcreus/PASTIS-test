# Setting this up in marimo

Two formats ship side by side. Same code, same results:

- `01_data.ipynb`, `02_pretrain.ipynb`, `03_downstream_2fold.ipynb` — Jupyter
- `01_data_marimo.py`, `02_pretrain_marimo.py`, `03_downstream_2fold_marimo.py` — marimo

Use the marimo ones if you want reactivity and the run-button gates. molab opens
either, so the `.ipynb` files also work there unchanged.

---

## Step 1 — Put the files in a GitHub repo

The notebooks import `ubarn_model.py`, `ubarn_data.py` and `pastis_splits.py`,
and read `pastis_official_splits_v1.json`. molab mirrors a whole repo, so this
is the only route where those four files come along automatically.

Flat layout, no subdirectories — the bootstrap cell finds modules beside the
notebook:

```
your-repo/
├── 01_data_marimo.py
├── 02_pretrain_marimo.py
├── 03_downstream_2fold_marimo.py
├── pastis_splits.py
├── ubarn_data.py
├── ubarn_model.py
└── pastis_official_splits_v1.json
```

```bash
git init
git add .
git commit -m "U-BARN on PASTIS, 2-fold"
git remote add origin git@github.com:<you>/<repo>.git
git push -u origin main
```

The manifest is ~4 MB — well under GitHub's limits, no LFS needed.

## Step 2 — Open it in molab

Go to:

```
https://molab.marimo.io/github/<you>/<repo>/blob/main/01_data_marimo.py
```

Or open `molab.marimo.io`, then swap `/notebooks` for `/github` in the URL and
paste your repo path. GitHub is the source of truth: edit locally, push, and
reopen to pick up changes.

## Step 3 — Turn the GPU on before notebook 02

Click the **notebook specs** button in the app header and attach the GPU.
Default is 4 CPUs / 32 GB RAM with no accelerator, which is fine for notebook
01 but will not run pretraining at the default settings.

Do this *before* pressing the pretraining button — attaching a GPU restarts the
kernel, and you would lose the cache built in notebook 01.

## Step 4 — Run notebook 01

Cells run automatically on open, except the expensive one. The validation,
split listing and fold selection all evaluate immediately; read the integrity
reports before going further.

Then set your three knobs and press **Download + build cache**:

| variable | default | note |
|---|---|---|
| `SUBSET` | 600 | `None` uses every patch the folds need (~34 GB) |
| `CFG_T_MAX` | 40 | dates kept per series |
| `CFG_CROP` | 64 | spatial crop |

First time through, use `SUBSET=200`, `CFG_T_MAX=16` to confirm the chain runs.

Check the coverage table at the end. If a split shows `test=3/496` you have
subsetted too hard for a meaningful number.

## Step 5 — Run notebook 02

Open `02_pretrain_marimo.py` in the same session so it sees the cache. It reads
`run_config.json`, so nothing needs copying.

Check the pool sizes it prints (one per fold, ~1455 patches each), then press
**Start pretraining**. Roughly 1.5–2 h per fold on the GPU at 40 epochs.

## Step 6 — Run notebook 03

Two gates. **Run both folds** does the FR / FT / e2e comparison at full label
count; **Run scarcity sweep** does the experiment that actually tells you
whether the pretraining helped. Run the sweep.

## Step 7 — Get your results out before the session dies

molab sessions end after 12 hours, or 90 minutes idle, and the filesystem goes
with them. Push checkpoints and results somewhere:

```python
from huggingface_hub import HfApi
api = HfApi()
for name in ["results_2fold.json", "sweep_2fold.json"]:
    api.upload_file(path_or_fileobj=str(WORK / name),
                    path_in_repo=name,
                    repo_id="<you>/ubarn-pastis", repo_type="model",
                    token="hf_...")
api.upload_file(path_or_fileobj=CKPTS["1"], path_in_repo="ubarn_fold1.pt",
                repo_id="<you>/ubarn-pastis", repo_type="model", token="hf_...")
```

---

## Running locally instead

```bash
uvx marimo edit 01_data_marimo.py
```

Or against a pinned environment:

```bash
uv venv && source .venv/bin/activate
uv pip install marimo torch numpy matplotlib huggingface-hub tqdm
marimo edit 01_data_marimo.py
```

`uvx marimo edit <molab-url>` pulls a molab notebook down to your machine.

---

## If you edit the .ipynb files and want to reconvert

```bash
marimo convert 01_data.ipynb -o 01_data_marimo.py
```

Two things break silently in the conversion, both already handled in the
shipped files. Watch for them if you write new cells.

**Underscore names are cell-local.** A helper called `_find` defined in one
cell is invisible to every other cell — you get a `NameError` at runtime, not a
conversion error. That is why the bootstrap helper is `find_file` and not
`_find`. Underscore prefixes are still correct for throwaway temporaries used
inside a single cell, and the notebooks use them that way.

**Every cell runs on open, and re-runs when a dependency changes.** Without a
gate, nudging `SUBSET` would restart the download; nudging `MASK_RATE` would
restart training. Wrap anything expensive:

```python
# one cell
run_thing = mo.ui.run_button(label="Start")
run_thing

# the next cell
mo.stop(not run_thing.value, mo.md("Press **Start** above."))
...expensive work...
```

Also note that marimo forbids defining the same name in two cells, and
mutating an object across cells does not retrigger dependents. Notebook 02
mutates `RUN_CONFIG` in its final cell to record the checkpoint paths — that is
deliberate, and harmless because the cell writes the file itself.

To check a converted notebook before trusting it, `marimo edit` it: the editor
flags multiply-defined names and cycles up front.
