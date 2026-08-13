#!/bin/bash
set -e

CONFIG_PATH="${1:?Usage: bash scripts/download_dataset.sh <path-to-top-level-config.yaml>}"

# ---- STEP 0: read data_dir / use_segmented from the config's data sub-config ----
CONFIG_OUT=$(python3 -c "
import yaml
top = yaml.safe_load(open('$CONFIG_PATH'))
data_cfg = yaml.safe_load(open(f\"configs/data/{top['defaults']['data']}.yaml\"))
print(data_cfg.get('use_segmented', False), data_cfg['data_dir'])
")
read -r USE_SEGMENTED DATA_DIR <<< "$CONFIG_OUT"

METADATA_DATASET="kmader/skin-cancer-mnist-ham10000"
if [ "$USE_SEGMENTED" = "True" ]; then
    IMAGE_DATASET="krishu22/segmented-ham10000"
    echo "===== Segmented HAM10000 dataset download (config: $CONFIG_PATH) ====="
else
    IMAGE_DATASET="$METADATA_DATASET"
    echo "===== HAM10000 dataset download (config: $CONFIG_PATH) ====="
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

mkdir -p "$DATA_DIR"

# ---- STEP 3: metadata CSV (always sourced from the raw HAM10000 dataset) ----
if [ -f "$DATA_DIR/HAM10000_metadata.csv" ]; then
    echo "Metadata already present at $DATA_DIR — skipping."
else
    echo "Downloading metadata from $METADATA_DATASET ..."
    kaggle datasets download -d "$METADATA_DATASET" -f HAM10000_metadata.csv -p "$DATA_DIR"
    if ls "$DATA_DIR"/*.zip >/dev/null 2>&1; then
        cd "$DATA_DIR" && unzip -q -o ./*.zip && rm -f ./*.zip && cd ..
    fi
fi

# ---- STEP 4: images ----
if [ -d "$DATA_DIR/HAM10000_images_part_1" ]; then
    echo "Images already present at $DATA_DIR — skipping download."
else
    echo "Downloading images from $IMAGE_DATASET ..."
    kaggle datasets download -d "$IMAGE_DATASET" -p "$DATA_DIR"
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
fi

# ---- STEP 5: verify expected files exist ----
if [ -f "$DATA_DIR/HAM10000_metadata.csv" ] && [ -d "$DATA_DIR/HAM10000_images_part_1" ]; then
    echo "Verified: metadata + images present."
else
    echo "ERROR: expected files not found after download — check dataset structure."
    exit 1
fi

echo "===== Dataset ready at $DATA_DIR ====="
