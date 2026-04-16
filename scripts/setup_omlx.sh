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

# 5. Install OMLX in editable mode (includes native C++/FP8 extensions)
echo "🏗️ Building and installing OMLX with Native FP8 acceleration..."
cd "$PROJECT_ROOT"
pip install --no-build-isolation -e .

# 6. Install vllm-metal (PagedAttention Metal kernels for Apple Silicon)
echo "⚡ Installing vllm-metal (varlen PagedAttention Metal kernels)..."
if [ -d "$PROJECT_ROOT/vllm-metal" ]; then
    echo "  📁 Found local vllm-metal checkout, installing in editable mode..."
    # PyO3 requires forward compat flag for Python >=3.14
    PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 pip install -e "$PROJECT_ROOT/vllm-metal" 2>&1 || {
        echo "  ⚠️  vllm-metal install failed (Rust toolchain may be missing)."
        echo "  ℹ️  Install Rust via: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
        echo "  ℹ️  oMLX will still work without vllm-metal (using standard MLX attention)."
    }
else
    echo "  📥 Cloning vllm-metal from GitHub..."
    git clone https://github.com/vllm-project/vllm-metal.git "$PROJECT_ROOT/vllm-metal"
    # Relax Python version constraint for >=3.14
    sed -i '' 's/requires-python = ">=3.12,<3.14"/requires-python = ">=3.12"/' "$PROJECT_ROOT/vllm-metal/pyproject.toml" 2>/dev/null || true
    PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 pip install -e "$PROJECT_ROOT/vllm-metal" 2>&1 || {
        echo "  ⚠️  vllm-metal install failed. oMLX will use standard MLX attention as fallback."
    }
fi

echo ""
echo "------------------------------------------------"
echo "✨ SETUP COMPLETE ✨"
if [ "$IS_SOURCED" == "true" ]; then
    echo "👉 Environment is now ACTIVE in this terminal."
else
    echo "👉 To activate, run: source $PROJECT_ROOT/$VENV_NAME/bin/activate"
fi
echo "------------------------------------------------"
