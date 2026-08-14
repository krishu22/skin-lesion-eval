# HAM10000 Skin Lesion Classification

Trains and evaluates a `timm` Swin Transformer classifier on the HAM10000 dataset
(seven lesion types: `nv`, `mel`, `bkl`, `bcc`, `akiec`, `vasc`, `df`).

The pipeline is split into three independent axes that compose freely:

1. **Dataset** — raw HAM10000 vs. lesion-segmented HAM10000. Chosen at download time and
   independent of everything below it.
2. **Preprocessing / augmentation** — DullRazor hair removal (applied before standard
   augmentations, if enabled), then the usual crop/flip/rotation/color-jitter pipeline, then
   optional batch-level MixUp/CutMix applied dynamically during training.
3. **Loss function** — cross-entropy, focal, class-balanced focal, confusion-aware CE, or
   logit-adjusted CE.

Any combination of the above is a single command with no code or YAML edits required — see
"Running ablations" below.

## Setup

```bash
python3 -m pip install -r requirements.txt
# or, on a fresh RunPod instance:
export WANDB_API_KEY=...      # required, used for non-interactive `wandb login`
bash runpod_setup.sh
```

Kaggle credentials (needed once, to download the dataset):
```bash
mkdir -p ~/.kaggle
# paste your kaggle.json content into ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

## Step 1 — download the dataset

Segmentation is chosen here and doesn't affect anything downstream — every later config/CLI
option works the same on either variant.

```bash
bash scripts/download_dataset.sh ham10000              # raw HAM10000
bash scripts/download_dataset.sh ham10000_segmented     # lesion-segmented HAM10000
```
Downloads to `ham10000/` or `ham10000_segmented/` respectively; skips re-downloading if the
data is already present.

## Step 2 — train + evaluate

```bash
python3 scripts/train.py --config configs/experiment.yaml [overrides...]
```

This single command trains, saves the best checkpoint to `<output_dir>/checkpoints/best.pt`,
then runs final evaluation on the test set and writes metrics CSVs to `<output_dir>/metrics/`.
`configs/experiment.yaml` is the one base config; everything variable is passed as a trailing
`key=value` (or `key.subkey=value`) override — no need to edit YAML, commit, or push to try a
different combination on a RunPod terminal.

### Overridable keys

| Override | Values | Effect |
|---|---|---|
| `data=` | `ham10000` \| `ham10000_segmented` | raw vs. segmented dataset (must match what you downloaded) |
| `data.dullrazor=` | `true` \| `false` | hair removal, applied before standard augmentations |
| `loss=` | `ce` \| `focal` \| `cb_focal` \| `confusion_aware_ce` \| `logit_adjusted_ce` | loss function |
| `train=` | `default` \| `cosine_warm_restarts` \| `dryrun` | optimizer/scheduler preset (see `configs/train/`) |
| `train.mixup.enabled=` | `true` \| `false` | batch-level MixUp during training |
| `train.mixup.alpha=` | float | MixUp Beta-distribution alpha |
| `train.mixup.batch_prob=` | float | fraction of batches MixUp is applied to |
| `train.cutmix.enabled=` | `true` \| `false` | batch-level CutMix during training |
| `train.cutmix.alpha=` | float | CutMix Beta-distribution alpha |
| `train.cutmix.batch_prob=` | float | fraction of batches CutMix is applied to |
| `train.metric_for_best=` | `balanced_accuracy` \| `macro_f1` | metric used for early stopping / checkpoint selection |
| `run_name=` | string | **set this every run** — used for W&B run naming |
| `output_dir=` | path | **set this every run** — where checkpoints/metrics are written |
| `splits_dir=` | path | shared lesion-level train/val/test split cache (see below); leave at its default unless you deliberately want a different split |

At most one of MixUp/CutMix is ever applied to a given batch even if both are enabled;
`batch_prob` is each one's share of batches.

> **Always pass a unique `run_name=` and `output_dir=` per ablation.** They default to
> `baseline` / `outputs/baseline` — reusing them across runs will overwrite the previous run's
> checkpoint and metrics.

### The shared train/val/test split

`splits_dir` (default `outputs/splits`) is intentionally **separate** from `output_dir`. The
lesion-level split is computed once on the first run and cached there as
`train.csv`/`val.csv`/`test.csv` + `manifest.json`; every subsequent run — regardless of
`run_name`/`output_dir`, and regardless of `data=ham10000` vs. `data=ham10000_segmented` —
loads that same cached split instead of recomputing a new random one. This is what makes
ablations comparable: every combination of dullrazor/mixup/cutmix/loss trains and evaluates on
the exact same images. `outputs/splits/` is checked into git for this reason (everything else
under `outputs/` is per-run and gitignored). Only pass `splits_dir=` to something else if you
deliberately want a different split (e.g. a different `test_size`/seed) — and note that changing
`configs/data/*.yaml`'s `split:` block has no effect once a cached split already exists at
`splits_dir`; delete the cache directory first to force a recompute.

### Example ablation commands

Smoke-test the pipeline first (2 epochs, tiny patience):
```bash
python3 scripts/train.py --config configs/experiment.yaml \
    train=dryrun run_name=smoke output_dir=outputs/smoke
```

Raw data, plain cross-entropy, no extras:
```bash
python3 scripts/train.py --config configs/experiment.yaml \
    run_name=raw_ce output_dir=outputs/raw_ce
```

Segmented data + DullRazor + focal loss:
```bash
python3 scripts/train.py --config configs/experiment.yaml \
    data=ham10000_segmented data.dullrazor=true loss=focal \
    run_name=seg_dullrazor_focal output_dir=outputs/seg_dullrazor_focal
```

Raw data + MixUp + logit-adjusted CE:
```bash
python3 scripts/train.py --config configs/experiment.yaml \
    loss=logit_adjusted_ce train.mixup.enabled=true train.mixup.alpha=0.4 \
    run_name=raw_mixup_la output_dir=outputs/raw_mixup_la
```

Segmented data + DullRazor + CutMix + class-balanced focal loss:
```bash
python3 scripts/train.py --config configs/experiment.yaml \
    data=ham10000_segmented data.dullrazor=true loss=cb_focal \
    train.cutmix.enabled=true \
    run_name=seg_dr_cutmix_cbfocal output_dir=outputs/seg_dr_cutmix_cbfocal
```

## Pipeline details

1. **Config**: `configs/experiment.yaml` composes `data` / `model` / `loss` / `train` sub-configs
   (see `src/config.py`); CLI overrides are merged in on top via OmegaConf.
2. **Splits**: `src/data/splits.py` loads `<data_dir>/HAM10000_metadata.csv`, maps `dx` to class
   indices, and performs lesion-level `GroupShuffleSplit` (no lesion leakage across
   train/val/test). Cached under `splits_dir` (shared across every run — see above).
3. **Transforms**: `src/data/transforms.py` — optional DullRazor, then RandomResizedCrop /
   flips / rotation / color jitter / random erasing (train only), then normalize.
4. **Model**: `src/models/build.py` builds a `timm` model per `configs/model/swin.yaml`.
5. **Loss**: `src/losses/build.py` dispatches on `loss.name`; class counts for
   `cb_focal`/`confusion_aware_ce`/`logit_adjusted_ce` are computed from the train split.
6. **Training**: `scripts/train.py` → `src/engine/trainer.py` — AdamW, cosine or cosine-warm-restart
   scheduler with linear warmup, optional MixUp/CutMix per batch, early stopping and
   checkpointing on `train.metric_for_best`.
7. **Evaluation**: `src/engine/evaluator.py` — balanced accuracy, macro F1, per-class metrics,
   macro-AUC, NLL, Brier score, ECE; writes CSVs to `<output_dir>/metrics/`.

Logging is via Weights & Biases (`src/utils/logger.py`); there is no separate local log file.

## Manual smoke-test scripts

`scripts/checks/*.py` are standalone ad hoc scripts (not pytest) for exercising individual
pieces of the pipeline against `configs/experiment.yaml` — e.g. `check_config.py`,
`check_dataset.py`, `check_trainer.py`. Run any of them directly, e.g.:
```bash
python3 scripts/checks/check_config.py
```

`scripts/sanity_check_mixup_cutmix.py` unit-tests the MixUp/CutMix math directly.
