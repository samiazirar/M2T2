# Configurable CUDA version support - works with both old and new drivers
# CUDA 12.1: For RTX 40 series and earlier, works with driver >=530
# CUDA 12.8: For Blackwell GPUs (RTX 50 series), requires driver >=560
ARG CUDA_VERSION=12.1.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Ensure CUDA extensions cover the architectures we actively target.
# Default: Pascal (7.0) through Hopper (9.0) - compatible with CUDA 12.1
# Note: Blackwell (10.0) support requires PyTorch updates, not yet available
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

# Base development tools and common libraries used by the project.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        ffmpeg \
        git \
        libgl1-mesa-glx \
        libxkbcommon-x11-0 \
        ninja-build \
        openssh-client \
        pkg-config \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        wget \
        x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Make "python" available alongside "python3".
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip to the latest release.
RUN python -m pip install --upgrade pip

# Helpful tooling for Hugging Face artifact downloads.
RUN pip install -U "huggingface_hub[hf_xet]"

# Prefer SSH for GitHub remotes to leverage mounted host SSH keys.
RUN git config --system url."ssh://git@github.com/".insteadOf https://github.com/


# Pre-copy requirement manifests to improve Docker build caching.
COPY requirements.txt /tmp/requirements.txt

# Install additional development libraries for FFmpeg and video processing
RUN apt-get update && apt-get install -y --no-install-recommends \
        libavcodec-dev \
        libavdevice-dev \
        libavfilter-dev \
        libavformat-dev \
        libavutil-dev \
        libswresample-dev \
        libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

# Pre-populate GitHub host key to avoid prompts in fresh containers
RUN mkdir -p /etc/ssh  && ssh-keyscan -t rsa,ecdsa,ed25519 github.com >> /etc/ssh/ssh_known_hosts 2>/dev/null || true

# Avoid git safety prompts when working in bind-mounted workspace
RUN git config --global --add safe.directory /workspace

# Install Git LFS
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
    && apt-get install -y git-lfs \
    && git lfs install

WORKDIR /workspace

# Copy project files
COPY requirements.txt .
COPY pointnet2_ops ./pointnet2_ops
COPY setup.py .
COPY m2t2 ./m2t2

# Install PyTorch with CUDA version detection
RUN CUDA_VERSION_DETECTED=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p') && \
    echo "Detected CUDA version: ${CUDA_VERSION_DETECTED}" && \
    if [ "${CUDA_VERSION_DETECTED}" = "12.1" ]; then \
        PYTORCH_CUDA_CHANNEL="https://download.pytorch.org/whl/cu121"; \
        echo "Using PyTorch cu121 wheels"; \
    elif [ "${CUDA_VERSION_DETECTED}" = "12.8" ]; then \
        PYTORCH_CUDA_CHANNEL="https://download.pytorch.org/whl/cu128"; \
        echo "Using PyTorch cu128 wheels"; \
    else \
        PYTORCH_CUDA_CHANNEL="https://download.pytorch.org/whl/cu121"; \
        echo "Defaulting to PyTorch cu121 wheels"; \
    fi && \
    python -m pip install --upgrade \
        --index-url "${PYTORCH_CUDA_CHANNEL}" \
        torch torchvision torchaudio

# Install project dependencies
RUN python -m pip install -r requirements.txt

# Build and install PointNet++ ops
RUN python -m pip install ./pointnet2_ops

# Install M2T2 package in editable mode
RUN python -m pip install -e .

# Clone M2T2 models from Hugging Face
RUN git clone https://huggingface.co/wentao-yuan/m2t2 models

# Verify CUDA installation works with PyTorch
RUN python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); torch.cuda.is_available() and print('CUDA device:', torch.cuda.get_device_name(0))"

EXPOSE 7860

CMD ["bash"]
