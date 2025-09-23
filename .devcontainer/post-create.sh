#!/usr/bin/env bash
set -euo pipefail

# Refresh package index for any additional development headers required by optional extensions.
apt-get update && apt-get install -y --no-install-recommends \
    libavcodec-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    libswscale-dev \
  && rm -rf /var/lib/apt/lists/*

# Ensure pip itself is current inside the devcontainer runtime.
python -m pip install --upgrade pip

# Install PyTorch that matches the CUDA 12.8 toolkit baked into the image.
python -m pip install --upgrade \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch torchvision torchaudio

# Install project Python dependencies (torch will already be satisfied).
python -m pip install -r requirements.txt

# Build and install the custom PointNet++ ops against the freshly installed torch.
python -m pip install ./pointnet2_ops

# Install the M2T2 package itself in editable mode for development.
python -m pip install -e .

# Confirm CUDA visibility inside the container (useful during post-create logs).
python - <<'PY'
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
PY
