#!/usr/bin/env bash
#
# One-shot script: create/ensure conda env + pip install -r requirements.txt
# Optionally installs common_utils from a local tools2 directory and writes an env helper.
#

set -euo pipefail

# Accept ToS for Anaconda channels
# Remove these lines (not supported on macOS)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-grokking}"            # override: ENV_NAME=new-env-name ./setup_dev_env.sh
PY_VER="${PY_VER:-3.11}"                    # override: PY_VER=3.10 ./setup_dev_env.sh
REQ_FILE="${REQ_FILE:-requirement.txt}"
TOOLS2_DIR="${TOOLS2_DIR:-$ROOT_DIR/tools2}"
COMMON_UTILS_DIR="${COMMON_UTILS_DIR:-$TOOLS2_DIR/common_utils}"
INSTALL_COMMON_UTILS="${INSTALL_COMMON_UTILS:-1}" # set to 0 to skip common_utils install

if [ ! -f "$REQ_FILE" ]; then
  echo "❌ requirements file not found: $REQ_FILE"
  exit 1
fi

# Create env if it doesn't exist
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "📦 Creating conda env '$ENV_NAME' (Python $PY_VER)..."
  conda create -y -n "$ENV_NAME" "python=$PY_VER"
else
  echo "✅ Conda env '$ENV_NAME' already exists."
fi

echo "⬆️  Upgrading pip in '$ENV_NAME'..."
conda run -n "$ENV_NAME" python -m pip install --upgrade pip

echo "📥 Installing dependencies from $REQ_FILE into '$ENV_NAME'..."
conda run -n "$ENV_NAME" python -m pip install -r "$REQ_FILE"

if [ "$INSTALL_COMMON_UTILS" = "1" ]; then
  if [ ! -d "$COMMON_UTILS_DIR" ]; then
    echo "❌ common_utils not found at: $COMMON_UTILS_DIR"
    echo "   Set COMMON_UTILS_DIR or TOOLS2_DIR to the correct path."
    exit 1
  fi
  echo "📦 Installing common_utils from $COMMON_UTILS_DIR into '$ENV_NAME'..."
  conda run -n "$ENV_NAME" python -m pip install -e "$COMMON_UTILS_DIR"
fi

echo -e "\n✅ Done:"
echo "   • conda activate $ENV_NAME"
