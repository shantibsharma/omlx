#!/bin/bash
# ==============================================================================
# Compile oMLX Fast Loader – links against the pre-built libmlx.dylib
# from the PyPI mlx wheel (no Xcode Metal toolchain required).
# ==============================================================================
set -e

echo "🔨 Compiling oMLX C++ Fast-IO Extension (O3 + libmlx.dylib)..."

# Activate virtual environment
source /Users/shantibhusansharma/work/code/.venv/bin/activate

# Paths
MLX_SITE=$(python -c "import mlx.core, pathlib; print(pathlib.Path(mlx.core.__file__).parent)")
MLX_INCLUDE="${MLX_SITE}/include"
MLX_LIB="${MLX_SITE}/lib"
SRCS="/Users/shantibhusansharma/work/code/omlx/src/omlx_fast_io.cpp /Users/shantibhusansharma/work/code/omlx/src/cache_core.cpp"
OUT="/Users/shantibhusansharma/work/code/omlx/src/omlx_fast_io.so"

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
