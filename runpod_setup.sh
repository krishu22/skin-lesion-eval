#!/bin/bash
set -e  # stop immediately if any command fails, instead of silently continuing

echo "===== RunPod environment setup ====="

# ---- STEP 1: system check ----
echo "GPU check:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---- STEP 2: install Python dependencies ----
echo "Installing requirements..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# ---- STEP 3: W&B login ----
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "ERROR: WANDB_API_KEY is required for non-interactive training."
    echo "Set WANDB_API_KEY in the pod environment before running this script."
    exit 1
fi

echo "Logging into W&B using WANDB_API_KEY env var..."
python3 -m wandb login "$WANDB_API_KEY"

# ---- STEP 4: confirm PyTorch sees the GPU ----
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

echo "===== Setup complete. Run scripts/download_dataset.sh [ham10000|ham10000_segmented] next, then scripts/train.py ====="