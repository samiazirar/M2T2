# M2T2 Dev Container Configuration

This dev container supports both older and newer NVIDIA drivers with flexible CUDA version support.

## CUDA Version Support

### Default Configuration (CUDA 12.1) - Recommended
- **GPUs Supported**: Pascal through Ada Lovelace (GTX 10 series to RTX 40 series)
- **Driver Requirement**: NVIDIA driver ≥530
- **PyTorch**: Automatically uses cu121 wheels
- **Status**: Stable and well-tested

### Experimental Configuration (CUDA 12.8) 
- **GPUs Supported**: Same as above (Pascal through Ada Lovelace)
- **Driver Requirement**: NVIDIA driver ≥560
- **PyTorch**: Automatically uses cu128 wheels
- **Status**: Experimental - for future Blackwell support
- **Note**: Blackwell GPUs (RTX 50 series) not yet supported by PyTorch

## How to Switch CUDA Versions

### Method 1: Environment Variable (Recommended)
Set the environment variable before building:

```bash
# For CUDA 12.1 (recommended - stable and well-tested)
export CUDA_VERSION=12.1.1

# For CUDA 12.8 (experimental - for future Blackwell support)
export CUDA_VERSION=12.8.0

# Then rebuild container
```

### Method 2: Manual Override
You can also override the PyTorch CUDA channel:

```bash
# Force PyTorch cu121 wheels regardless of detected CUDA version
export PYTORCH_CUDA_CHANNEL=https://download.pytorch.org/whl/cu121

# Force PyTorch cu128 wheels
export PYTORCH_CUDA_CHANNEL=https://download.pytorch.org/whl/cu128
```

## Supported GPU Architectures

| Compute Capability | Architecture | Example GPUs |
|-------------------|--------------|---------------|
| 7.0 | Pascal | GTX 1080, Tesla P100 |
| 7.5 | Turing | GTX 1660, RTX 2080 |
| 8.0 | Ampere | A100 |
| 8.6 | Ampere | RTX 3090, A40 |
| 8.9 | Ada Lovelace | RTX 4090 |
| 9.0 | Hopper | H100 |
| ~~10.0~~ | ~~Blackwell~~ | ~~RTX 50 series, H200~~ (Not yet supported) |

## Troubleshooting

### "CUDA driver version is insufficient"
- Update your NVIDIA drivers:
  - For CUDA 12.1: Driver ≥530
  - For CUDA 12.8: Driver ≥560

### "Unknown CUDA arch" during PointNet2 compilation
- This should be automatically handled, but if issues persist:
  - Check that your GPU architecture is in the supported list above
  - Try rebuilding the container with a different CUDA version