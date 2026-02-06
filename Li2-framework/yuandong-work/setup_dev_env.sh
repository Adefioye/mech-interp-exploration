#!/usr/bin/env bash
#
# One-shot script: create/ensure conda env + pip install -r requirements.txt
# Also installs tools2/common_utils and writes an env helper that sets PYTHONPATH.
#

set -euo pipefail

# Accept ToS for Anaconda channels
# Remove these lines (not supported on macOS)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

ENV_NAME="${ENV_NAME:-grokking}"      # override: ENV_NAME=new-env-name ./setup_dev_env.sh
PY_VER="${PY_VER:-3.11}"              # override: PY_VER=3.10 ./setup_dev_env.sh
REQ_FILE="${REQ_FILE:-requirements.txt}"
TOOLS2_DIR="${TOOLS2_DIR:-$HOME/tools2}"
INSTALL_TOOLS2="${INSTALL_TOOLS2:-1}" # set to 0 to skip tools2/common_utils install
WRITE_ENV_SH="${WRITE_ENV_SH:-1}"     # set to 0 to skip writing env.sh

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

# if [ "$INSTALL_TOOLS2" = "1" ]; then
#   if [ ! -d "$TOOLS2_DIR" ]; then
#     echo "📥 Cloning tools2 into $TOOLS2_DIR..."
#     git clone https://github.com/yuandong-tian/tools2 "$TOOLS2_DIR"
#   else
#     echo "✅ tools2 already exists at $TOOLS2_DIR."
#   fi
#   echo "📦 Installing common_utils from tools2 into '$ENV_NAME'..."
#   conda run -n "$ENV_NAME" python -m pip install -e "$TOOLS2_DIR/common_utils"
# fi

# if [ "$WRITE_ENV_SH" = "1" ]; then
#   ENV_SH="$PWD/env.sh"
#   cat > "$ENV_SH" <<'EOF'
# #!/usr/bin/env bash
# # Source this file from the yuandong-work folder.
# ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# SSL_DIR="${ROOT_DIR}/ssl"
# REAL_DATASET_DIR="${SSL_DIR}/real-dataset"
# export PYTHONPATH="${SSL_DIR}:${REAL_DATASET_DIR}:${PYTHONPATH:-}"
# EOF
#   chmod +x "$ENV_SH"
#   echo "🧩 Wrote $ENV_SH"
# fi

echo -e "\n✅ Done:"
echo "   • conda activate $ENV_NAME"
# if [ "$WRITE_ENV_SH" = "1" ]; then
#   echo "   • source $PWD/env.sh"
# fi
# echo "   • pip show <package>  (to verify)"
