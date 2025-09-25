#!/usr/bin/env bash
# Launch the M2T2 inference container and expose the Flask API

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HOST_WORKSPACE=${HOST_WORKSPACE:-$SCRIPT_DIR}
IMAGE_NAME=${IMAGE_NAME:-m2t2:cu121}
HOST_PORT=${HOST_PORT:-5000}
CONTAINER_PORT=${CONTAINER_PORT:-5000}
DISPLAY_NUMBER=2



docker run --gpus all -it --rm \
  -p "${HOST_PORT}:${CONTAINER_PORT}" \
  -p 7000:7000 \
  -v "${HOST_WORKSPACE}:/workspace" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=":${DISPLAY_NUMBER}" \
  -e LIBGL_ALWAYS_SOFTWARE=1 \
  -e MESA_GL_VERSION_OVERRIDE=3.3 \
  -e M2T2_VISUALIZE="${M2T2_VISUALIZE:-0}" \
  "${IMAGE_NAME}" \
  bash -c "Xvfb :${DISPLAY_NUMBER} -ac -screen 0 1024x768x24 -nolisten tcp > /dev/null 2>&1 & \
           sleep 2; \
           export DISPLAY=:${DISPLAY_NUMBER}; \
           cd /workspace; \
           if [ -f requirements.txt ]; then pip3 install --no-cache-dir -r requirements.txt || echo 'Warning: pip install failed; ensure dependencies are baked into the image.'; fi; \
           python3 app_server.py --host 0.0.0.0 --port ${CONTAINER_PORT}"