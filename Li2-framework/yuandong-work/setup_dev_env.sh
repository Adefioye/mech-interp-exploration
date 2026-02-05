#!/usr/bin/env bash
#
# One-shot bootstrap:
# - Install Miniconda (if missing)
# - Create / reuse conda env
# - Install requirements.txt
#

set -euo pipefail

ENV_NAME="${ENV_NAME:-grokking}"
PY_VER="${PY_VER:-3.11}"
REQ_FILE="${REQ_FILE:-requirements.txt}"
MINICONDA_DIR="$HOME/miniconda"

# ------------------------------------------------------------------
# 1. Install Miniconda if missing
# ------------------------------------------------------------------
if [ ! -x "$MINICONDA_DIR/bin/conda" ]; then
  echo "📦 Miniconda not found. Installing..."

  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
       -O /tmp/miniconda.sh

  bash /tmp/miniconda.sh -b -p "$MINICONDA_DIR"
  rm /tmp/miniconda.sh

  echo "✅ Miniconda installed at $MINICONDA_DIR"
else
  echo "✅ Miniconda already installed."
fi

# ------------------------------------------------------------------
# 2. Load conda into *this* shell (no restart needed)
# ------------------------------------------------------------------
echo "🔧 Initializing conda for this script..."
eval "$($MINICONDA_DIR/bin/conda shell.bash hook)"

# Optional: initialize for future shells
if ! grep -q "conda initialize" ~/.bashrc 2>/dev/null; then
  echo "📎 Running conda init bash (for future shells)..."
  conda init bash
fi

# ------------------------------------------------------------------
# 3. Create env if needed
# ------------------------------------------------------------------
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "📦 Creating conda env '$ENV_NAME' (Python $PY_VER)..."
  conda create -y -n "$ENV_NAME" "python=$PY_VER"
else
  echo "✅ Conda env '$ENV_NAME' already exists."
fi

# ------------------------------------------------------------------
# 4. Install requirements.txt
# ------------------------------------------------------------------
if [ ! -f "$REQ_FILE" ]; then
  echo "❌ requirements file not found: $REQ_FILE"
  exit 1
fi

echo "⬆️  Upgrading pip..."
conda run -n "$ENV_NAME" python -m pip install --upgrade pip

echo "📥 Installing dependencies from $REQ_FILE..."
conda run -n "$ENV_NAME" python -m pip install -r "$REQ_FILE"

# ------------------------------------------------------------------
# Done
# ------------------------------------------------------------------
echo -e "\n✅ Environment ready:"
echo "   • conda activate $ENV_NAME"
echo "   • python -c \"import sys; print(sys.executable)\""

