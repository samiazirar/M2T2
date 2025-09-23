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

# Pre-populate the GitHub host key to avoid authenticity prompts in fresh containers.
RUN mkdir -p /etc/ssh && ssh-keyscan -t rsa,ecdsa,ed25519 github.com >> /etc/ssh/ssh_known_hosts 2>/dev/null || true

# Pre-copy requirement manifests to improve Docker build caching.
COPY requirements.txt /tmp/requirements.txt

# Mark the repository workspace as safe for Git operations.
RUN git config --global --add safe.directory /workspace

WORKDIR /workspace

EXPOSE 7860

CMD ["bash"]
