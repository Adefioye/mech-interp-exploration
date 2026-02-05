#!/usr/bin/env bash
#
# One-shot script: create/ensure conda env + pip install -r requirements.txt
#

set -euo pipefail

# Accept ToS for Anaconda channels
# Remove these lines (not supported on macOS)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

ENV_NAME="${ENV_NAME:-grokking}"      # override: ENV_NAME=new-env-name ./setup_dev_env.sh
PY_VER="${PY_VER:-3.11}"              # override: PY_VER=3.10 ./setup_dev_env.sh
REQ_FILE="${REQ_FILE:-requirements.txt}"

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

echo -e "\n✅ Done:"
echo "   • conda activate $ENV_NAME"
echo "   • pip show <package>  (to verify)"
