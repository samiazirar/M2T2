#!/bin/bash
# Quick configuration script for M2T2 dev container CUDA version

set -e

show_help() {
    echo "Usage: $0 [12.1|12.8|auto]"
    echo ""
    echo "Configure CUDA version for M2T2 dev container:"
    echo "  12.1    Use CUDA 12.1 (driver ≥530, recommended for stability)"
    echo "  12.8    Use CUDA 12.8 (driver ≥560, experimental - no Blackwell support yet)"
    echo "  auto    Auto-detect based on current driver version"
    echo ""
    echo "Note: Blackwell GPU support (compute 10.0) is not yet available in PyTorch"
    echo ""
    echo "Current configuration:"
    echo "  CUDA_VERSION: ${CUDA_VERSION:-12.1.1 (default)}"
    echo "  PYTORCH_CUDA_CHANNEL: ${PYTORCH_CUDA_CHANNEL:-auto-detect}"
}

if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

case "$1" in
    "12.1")
        export CUDA_VERSION=12.1.1
        export PYTORCH_CUDA_CHANNEL=https://download.pytorch.org/whl/cu121
        echo "✅ Configured for CUDA 12.1"
        echo "   - Driver requirement: ≥530"
        echo "   - Supports GPUs up to RTX 40 series"
        echo "   - PyTorch: cu121 wheels"
        ;;
    "12.8")
        export CUDA_VERSION=12.8.0
        export PYTORCH_CUDA_CHANNEL=https://download.pytorch.org/whl/cu128
        echo "✅ Configured for CUDA 12.8"
        echo "   - Driver requirement: ≥560"
        echo "   - PyTorch: cu128 wheels"
        echo "   - Note: Blackwell GPUs not yet supported by PyTorch"
        ;;
    "auto")
        # Try to detect driver version
        if command -v nvidia-smi &> /dev/null; then
            DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
            if [ -n "$DRIVER_VERSION" ]; then
                DRIVER_MAJOR=$(echo $DRIVER_VERSION | cut -d. -f1)
                if [ "$DRIVER_MAJOR" -ge 560 ]; then
                    echo "🔍 Detected driver $DRIVER_VERSION (≥560)"
                    echo "✅ Auto-configuring for CUDA 12.8 (experimental)"
                    echo "   Note: Blackwell GPU support pending PyTorch updates"
                    export CUDA_VERSION=12.8.0
                    unset PYTORCH_CUDA_CHANNEL  # Let post-create auto-detect
                elif [ "$DRIVER_MAJOR" -ge 530 ]; then
                    echo "🔍 Detected driver $DRIVER_VERSION (≥530, <560)"
                    echo "✅ Auto-configuring for CUDA 12.1 (legacy support)"
                    export CUDA_VERSION=12.1.1
                    unset PYTORCH_CUDA_CHANNEL  # Let post-create auto-detect
                else
                    echo "⚠️  Driver $DRIVER_VERSION is too old (<530)"
                    echo "   Please update your NVIDIA drivers"
                    exit 1
                fi
            else
                echo "❌ Could not detect driver version"
                exit 1
            fi
        else
            echo "❌ nvidia-smi not found. Cannot auto-detect."
            echo "   Please specify '12.1' or '12.8' manually"
            exit 1
        fi
        ;;
    *)
        echo "❌ Invalid option: $1"
        show_help
        exit 1
        ;;
esac

echo ""
echo "🔄 Environment variables set. Now rebuild your dev container."
echo "   In VS Code: Ctrl+Shift+P → 'Dev Containers: Rebuild Container'"