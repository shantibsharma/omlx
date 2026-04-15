#!/bin/bash
# Setup script for OMLX development on macOS (M4 Pro optimized)

set -e # Exit on error

# Determine if the script is being sourced or executed
IS_SOURCED=$( [[ "${BASH_SOURCE[0]}" != "${0}" ]] && echo "true" || echo "false" )

if [ "$IS_SOURCED" == "false" ]; then
    set -e
    echo "💡 TIP: Run 'source $0' to keep the environment active in this terminal."
fi

# Locate project root without changing the user's CWD if sourced
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VENV_NAME=".omlxvnv"

echo "🚀 Starting OMLX Setup..."

# 1. Create Virtual Environment if it doesn't exist
if [ ! -d "$PROJECT_ROOT/$VENV_NAME" ]; then
    echo "📦 Creating virtual environment: $VENV_NAME..."
    python3 -m venv "$PROJECT_ROOT/$VENV_NAME"
else
    echo "✅ Virtual environment $VENV_NAME already exists."
fi

# 2. Activate for the rest of this script (and session if sourced)
echo "🔌 Activating environment..."
source "$PROJECT_ROOT/$VENV_NAME/bin/activate"

# 3. Upgrade pip and install build tools
echo "🛠️ Installing build requirements..."
pip install --upgrade pip
pip install setuptools wheel

# 4. Install MLX
echo "🧠 Installing MLX (Metal acceleration)..."
pip install mlx

# 5. Install OMLX in editable mode
echo "🏗️ Building and installing OMLX with native extensions..."
cd "$PROJECT_ROOT"
pip install --no-build-isolation -e .

echo ""
echo "------------------------------------------------"
echo "✨ SETUP COMPLETE ✨"
if [ "$IS_SOURCED" == "true" ]; then
    echo "👉 Environment is now ACTIVE in this terminal."
else
    echo "👉 To activate, run: source $PROJECT_ROOT/$VENV_NAME/bin/activate"
fi
echo "------------------------------------------------"
