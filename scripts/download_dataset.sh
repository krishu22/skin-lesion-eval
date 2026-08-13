#!/bin/bash
set -e

DATA_NAME="${1:-ham10000}"
DATA_CONFIG="configs/data/${DATA_NAME}.yaml"

if [ ! -f "$DATA_CONFIG" ]; then
    echo "ERROR: no such data config '$DATA_CONFIG'."
    echo "Usage: bash scripts/download_dataset.sh [ham10000|ham10000_segmented]"
    exit 1
fi

# ---- STEP 0: read data_dir / use_segmented from the data config ----
CONFIG_OUT=$(python3 -c "
import yaml
data_cfg = yaml.safe_load(open('$DATA_CONFIG'))
print(data_cfg.get('use_segmented', False), data_cfg['data_dir'])
")
read -r USE_SEGMENTED DATA_DIR <<< "$CONFIG_OUT"

if [ "$USE_SEGMENTED" = "True" ]; then
    KAGGLE_DATASET="krishu22/ham10000-segmented-224"
    echo "===== Segmented HAM10000 dataset download (data: $DATA_NAME) ====="
else
    KAGGLE_DATASET="kmader/skin-cancer-mnist-ham10000"
    echo "===== HAM10000 dataset download (data: $DATA_NAME) ====="
fi

# ---- STEP 1: check Kaggle credentials exist ----
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "ERROR: ~/.kaggle/kaggle.json not found."
    echo "Upload your Kaggle API token to this pod first:"
    echo "  mkdir -p ~/.kaggle"
    echo "  (paste your kaggle.json content into ~/.kaggle/kaggle.json)"
    echo "  chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi
chmod 600 "$HOME/.kaggle/kaggle.json"

# ---- STEP 2: install kaggle CLI if not already present ----
if ! python3 -m pip show kaggle >/dev/null 2>&1; then
    python3 -m pip install kaggle
fi

# ---- STEP 3: skip download if data already present ----
if [ -f "$DATA_DIR/HAM10000_metadata.csv" ] && [ -d "$DATA_DIR/HAM10000_images_part_1" ]; then
    echo "Dataset already present at $DATA_DIR — skipping download."
    exit 0
fi

# ---- STEP 4: download and unzip ----
echo "Downloading $KAGGLE_DATASET ..."
mkdir -p "$DATA_DIR"
kaggle datasets download -d "$KAGGLE_DATASET" -p "$DATA_DIR"

echo "Unzipping..."
cd "$DATA_DIR"
if command -v unzip >/dev/null 2>&1; then
    unzip -q ./*.zip
else
    python3 - <<'PY'
import glob
import zipfile
for archive in glob.glob('*.zip'):
    with zipfile.ZipFile(archive, 'r') as zf:
        zf.extractall()
PY
fi
rm -f ./*.zip
cd ..

# ---- STEP 5: verify expected files exist ----
if [ -f "$DATA_DIR/HAM10000_metadata.csv" ] && [ -d "$DATA_DIR/HAM10000_images_part_1" ]; then
    echo "Verified: metadata + images present."
else
    echo "ERROR: expected files not found after extraction — check dataset structure."
    exit 1
fi

echo "===== Dataset ready at $DATA_DIR ====="
