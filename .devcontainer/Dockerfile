# CUDA >=12.4 is required to build for sm_100 (Blackwell)
# Use CUDA 12.8 to match the latest cu128 PyTorch wheels.
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Ensure any CUDA extensions compile for Ampere, Ada, Hopper, and Blackwell GPUs.
ENV TORCH_CUDA_ARCH_LIST="80;86;89;90;100"

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
