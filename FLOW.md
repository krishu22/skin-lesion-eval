Project Flow — HAM10000 Skin Lesion Classification
===============================================

Purpose
-------
- Goal: train and evaluate an image classification pipeline on the HAM10000 dataset to classify seven lesion types (nv, mel, bkl, bcc, akiec, vasc, df).

High-level flow
---------------
1. Configuration: top-level config (e.g. `configs/stage1_baseline.yaml`) composes data, model, loss, and train settings. The `data` default selects `configs/data/ham10000.yaml` (raw images, `use_segmented: false`) or `configs/data/ham10000_segmented.yaml` (lesion-segmented images from https://www.kaggle.com/datasets/krishu22/ham10000-segmented-224, `use_segmented: true`, downloaded to `ham10000_segmented/`; that dataset bundles its own `HAM10000_metadata.csv` alongside `HAM10000_images_part_1`/`part_2`, so no cross-dataset metadata fetch is needed). Everything downstream (splits, dataset, transforms) is unaffected by which one is selected — only `data_dir` differs, image folder names / filenames and the metadata CSV are identical.
2. Data loading & splits: `src/data/splits.py` loads `<data_dir>/HAM10000_metadata.csv`, maps `dx` to class indices, and performs lesion-level GroupShuffleSplit to produce train/val/test (no lesion leakage).
3. Transforms: `src/data/transforms.py`
   - Train: Resize -> RandomCrop (with pad) -> optional RandomHorizontalFlip -> RandomRotation -> ColorJitter -> ToTensor -> Normalize
   - Eval: Resize -> ToTensor -> Normalize
4. Dataset: `src/data/dataset.py` opens images with PIL, applies transforms, returns `(image_tensor, label)`.
5. Model: `src/models/build.py` builds a `timm` model from `configs/model/swin.yaml` (default `swin_tiny_patch4_window7_224`, pretrained, `num_classes=7`). Model moved to device.
6. Loss: `src/losses/build.py` chooses `nn.CrossEntropyLoss` by config (simple CE used for train and val).
7. Training loop: `scripts/train.py` -> `src/engine/trainer.py`
   - optimizer: AdamW, lr and weight_decay from config
   - scheduler: CosineAnnealingWarmRestarts
   - `run_epoch` handles both train and eval modes, computes avg loss and balanced accuracy (sklearn's `balanced_accuracy_score`), logs to W&B and prints
   - early stopping based on validation balanced accuracy and `patience` in config; best model `state_dict` saved to `outputs/.../checkpoints/best.pt`
8. Evaluation: `src/engine/evaluator.py` loads the saved checkpoint, runs inference, computes:
   - balanced accuracy, macro F1, per-class precision/recall/F1, specificity, confusion matrix
   - macro-AUC (safe-handled for missing classes), NLL (`log_loss`), Brier score, and ECE (implemented in file)
   - writes CSVs to `outputs/.../metrics/` and logs summary to W&B

Key files
---------
- `scripts/train.py` — main entrypoint
- `runpod_setup.sh` — installs deps and logs into W&B (expects `WANDB_API_KEY` env var)
- `scripts/download_dataset.sh <config.yaml>` — downloads the dataset variant (raw or segmented) the given config selects, from Kaggle (requires `~/.kaggle/kaggle.json`)
- `configs/` — configuration compositions (data, model, loss, train)
- `src/` — code: data, models, engine, utils

Results achieved (as-run)
-------------------------
- Best validation balanced accuracy observed: ~0.7568 (training stopped by early stopping at epoch ~19)
- Training exhibited strong overfitting (train bal-acc ~0.96 while val bacc ~0.65–0.73)
- Final per-class metrics and confusion matrix are saved under `outputs/stage1_baseline/metrics/` (CSV files produced by evaluator)

Reproducibility & logging
-------------------------
- Seed: `src/utils/seed.py` sets RNG seeds and supports deterministic mode via config flag `train.deterministic` (config example sets `true`). Deterministic CUDA behavior depends on PyTorch/CUDA versions and may not guarantee bitwise-equal runs on different hardware.
- Metrics/logging: training logs `train_loss`, `train_bal_acc`, `val_loss`, `val_bal_acc`, and learning rate to W&B via `src/utils/logger.py`. Evaluator writes CSVs locally and logs summary to W&B.

Run instructions (pod / local)
-----------------------------
1. Install deps (repo contains `requirements.txt`) or run `bash runpod_setup.sh` (expects `WANDB_API_KEY` env var).
2. Prepare Kaggle credentials: create `~/.kaggle/kaggle.json` with your Kaggle username/key and `chmod 600`.
3. Download the dataset variant selected by your config (pass the same config you'll train with — the script reads its `defaults.data` to decide raw vs. segmented and the target directory):
   - `bash scripts/download_dataset.sh configs/stage1_baseline.yaml`
4. Train & evaluate (single command), using the SAME config path used in step 3:
   - `python3 scripts/train.py --config configs/stage1_baseline.yaml`
   - This trains, saves best checkpoint to `outputs/.../checkpoints/best.pt`, then runs final evaluation and saves metrics CSVs.
   - To use segmented images with logit-adjusted loss instead: `bash scripts/download_dataset.sh configs/stage6_segmented_logit_adjusted.yaml` then `python3 scripts/train.py --config configs/stage6_segmented_logit_adjusted.yaml`.

Notes & recommendations
-----------------------
- Augmentations: current transforms are simple and suitable for baseline. Consider stronger augmentations (RandomResizedCrop, MixUp/CutMix) to improve generalization.
- Imbalance: HAM10000 is class-imbalanced. Try class-weighted CE, `WeightedRandomSampler`, or focal loss.
- Regularization: reduce lr (e.g., 5e-5), add dropout on classifier head, or use staged finetuning (freeze backbone then unfreeze).
- Scheduler: consider `ReduceLROnPlateau` when validation metric plateaus.
- Persistence: use Network volume on Runpod to persist `outputs/` across pod terminations.

Contact points in the repo
--------------------------
- Training loop: `src/engine/trainer.py`
- Evaluation: `src/engine/evaluator.py`
- Data split: `src/data/splits.py`
- Transforms: `src/data/transforms.py`
- Model builder: `src/models/build.py`
- Loss builder: `src/losses/build.py`

Summary
-------
This repository provides a complete baseline pipeline for HAM10000 multi-class skin lesion classification using a `timm` Swin transformer backbone, simple augmentations, cross-entropy loss, lesion-level grouped splits, and a robust evaluation suite. The pipeline is Runpod-ready after supplying `WANDB_API_KEY` and Kaggle credentials; results are logged to W&B and saved locally. The baseline achieves ~0.757 balanced accuracy on validation (best run) and needs targeted regularization and imbalance handling to improve generalization.
