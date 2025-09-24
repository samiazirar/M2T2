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
    openssh-client \
  && rm -rf /var/lib/apt/lists/*

# Ensure pip itself is current inside the devcontainer runtime.
python -m pip install --upgrade pip

# Install PyTorch matching the CUDA toolkit baked into the image.
# Auto-detect CUDA version and use appropriate PyTorch wheels.
if [ -z "${PYTORCH_CUDA_CHANNEL}" ]; then
  CUDA_VERSION_DETECTED=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
  case "${CUDA_VERSION_DETECTED}" in
    12.1)
      PYTORCH_CUDA_CHANNEL="https://download.pytorch.org/whl/cu121"
      echo "Detected CUDA ${CUDA_VERSION_DETECTED}, using PyTorch cu121 wheels"
      ;;
    12.8)
      PYTORCH_CUDA_CHANNEL="https://download.pytorch.org/whl/cu128"
      echo "Detected CUDA ${CUDA_VERSION_DETECTED}, using PyTorch cu128 wheels"
      ;;
    *)
      # Default to cu121 for compatibility
      PYTORCH_CUDA_CHANNEL="https://download.pytorch.org/whl/cu121"
      echo "CUDA version ${CUDA_VERSION_DETECTED} not specifically handled, defaulting to cu121 wheels"
      ;;
  esac
fi

python -m pip install --upgrade \
  --index-url "${PYTORCH_CUDA_CHANNEL}" \
  torch torchvision torchaudio

# Install project Python dependencies (torch will already be satisfied).
python -m pip install -r requirements.txt

# Build and install the custom PointNet++ ops against the freshly installed torch.
python -m pip install ./pointnet2_ops

# Install the M2T2 package itself in editable mode for development.
python -m pip install -e .

# Configure SSH for GitHub access (using SSH agent forwarding from host)
# Handle SSH config permission issues when mounted from host
if [ -f "/root/.ssh/config" ] && [ ! -w "/root/.ssh/config" ]; then
    echo "⚠️  SSH config is read-only (mounted from host), using GIT_SSH_COMMAND workaround..."
    # Set Git to use SSH with options that bypass the problematic config
    export GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ControlMaster=no -F /dev/null"
    git config --global core.sshCommand "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ControlMaster=no -F /dev/null"
    echo "✅ SSH configuration bypassed for Git operations"
else
    # Standard SSH configuration when files are writable
    export GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
    git config --global core.sshCommand "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
    echo "✅ Standard SSH configuration applied"
fi


# Install git lfs
if ! command -v git-lfs &> /dev/null; then
  curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
  apt-get install -y git-lfs
  git lfs install
fi

#if it does not already exist, clone the model repo 
if [ ! -d "models" ]; then
  git clone https://huggingface.co/wentao-yuan/m2t2 models
fi


# Confirm CUDA visibility inside the container (useful during post-create logs).
python - <<'PY'
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
PY

echo "✅ Development container setup complete."
echo "Run demo with meshcat-server and python demo.py eval.checkpoint=models/m2t2.pth eval.data_dir=sample_data/real_world/00 eval.mask_thresh=0.4 eval.num_runs=5"