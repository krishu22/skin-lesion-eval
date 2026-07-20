#!/bin/bash
set -e

echo "===== HAM10000 dataset download ====="

DATA_DIR="ham10000"
KAGGLE_DATASET="kmader/skin-cancer-mnist-ham10000"

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
if [ -f "$DATA_DIR/HAM10000_metadata.csv" ]; then
    echo "Dataset already present at $DATA_DIR — skipping download."
    exit 0
fi

# ---- STEP 4: download and unzip ----
echo "Downloading $KAGGLE_DATASET ..."
mkdir -p "$DATA_DIR"
kaggle datasets download -d "$KAGGLE_DATASET" -p "$DATA_DIR"

echo "Unzipping..."
cd "$DATA_DIR"
unzip -q ./*.zip
rm -f ./*.zip
cd ..

# ---- STEP 5: verify expected files exist ----
if [ -f "$DATA_DIR/HAM10000_metadata.csv" ]; then
    echo "Verified: HAM10000_metadata.csv present."
else
    echo "ERROR: metadata CSV not found after extraction — check dataset structure."
    exit 1
fi

echo "===== Dataset ready at $DATA_DIR ====="