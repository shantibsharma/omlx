#!/bin/bash
# ==============================================================================
# cMLX C++ Native Core Build Script
# ==============================================================================
# Compiles:
# 1. cmlx.cmlx_fast_io (Python extension)
# 2. agent_runner (Standalone C++ binary for Claude Code)

set -e

# Configuration
VNV_PYTHON=".cmlxvnv/bin/python3"
MLX_INC_DIR=$($VNV_PYTHON -c "import site; import os; print(os.path.join(site.getsitepackages()[0], 'mlx/include'))")
MLX_LIB_DIR=$($VNV_PYTHON -c "import site; import os; print(os.path.join(site.getsitepackages()[0], 'mlx/lib'))")
METAL_CPP_DIR="$MLX_INC_DIR/metal_cpp"

echo "🔍 Detected MLX Include: $MLX_INC_DIR"
echo "🔍 Detected MLX Lib: $MLX_LIB_DIR"

# 1. Build Python Extension
echo "📦 Building Python extension..."
$VNV_PYTHON setup.py build_ext --inplace

# 2. Build Standalone Agent Runner
echo "🚀 Building standalone agent_runner binary..."
mkdir -p bin

SOURCES=(
    "src/agent_runner.cpp"
    "src/native_engine.cpp"
    "src/llama_model.cpp"
    "src/metal_ops.cpp"
    "src/scheduler_core.cpp"
    "src/cache_core.cpp"
    "src/native_ssd_cache.cpp"
    "src/cmlx_fast_io.cpp"
    "src/paged_attention.cpp"
)

# Compile flags
# Note: -D_METAL_ and -DACCELERATE_NEW_LAPACK match MLX build requirements
CXX_FLAGS="-std=c++17 -O3 -Wall -fPIC -D_METAL_ -DACCELERATE_NEW_LAPACK"
INC_FLAGS="-Isrc -Isrc/metal -I$MLX_INC_DIR -I$METAL_CPP_DIR"
LD_FLAGS="-L$MLX_LIB_DIR -lmlx -framework Metal -framework Foundation -Wl,-rpath,$MLX_LIB_DIR"

clang++ $CXX_FLAGS $INC_FLAGS ${SOURCES[@]} $LD_FLAGS -o bin/agent_runner

echo "✅ Build complete!"
echo "📍 Python library: cmlx/cmlx_fast_io*.so"
echo "📍 Standalone binary: bin/agent_runner"
