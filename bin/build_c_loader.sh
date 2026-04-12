#!/bin/bash
# ==============================================================================
# Compile oMLX Fast Loader – links against the pre-built libmlx.dylib
# from the PyPI mlx wheel (no Xcode Metal toolchain required).
# ==============================================================================
set -e

echo "🔨 Compiling oMLX C++ Fast-IO Extension (O3 + libmlx.dylib)..."

# Get script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT="$(dirname "$DIR")"

# Activate virtual environment
if [ -f "$ROOT/.venv/bin/activate" ]; then
    source "$ROOT/.venv/bin/activate"
elif [ -f "/Users/shantibhusansharma/work/code/.venv/bin/activate" ]; then
    source "/Users/shantibhusansharma/work/code/.venv/bin/activate"
fi

# Paths
MLX_SITE=$(python -c "import mlx.core, pathlib; print(pathlib.Path(mlx.core.__file__).parent)")
MLX_INCLUDE="${MLX_SITE}/include"
MLX_LIB="${MLX_SITE}/lib"
SRCS="$ROOT/src/omlx_fast_io.cpp $ROOT/src/cache_core.cpp"
OUT="$ROOT/src/omlx_fast_io.so"

echo "   MLX include: ${MLX_INCLUDE}"
echo "   MLX lib:     ${MLX_LIB}"
echo "   Sources:     ${SRCS}"

# Compile
clang++ -shared -fPIC -O3 -std=c++17 \
    -I"${MLX_INCLUDE}" \
    -L"${MLX_LIB}" \
    -lmlx \
    -Wl,-rpath,"${MLX_LIB}" \
    ${SRCS} \
    -o "${OUT}"

echo "✅ Compiled omlx_fast_io.so successfully!"
echo "   Output: ${OUT}"
echo "   Size:   $(du -h ${OUT} | cut -f1)"
