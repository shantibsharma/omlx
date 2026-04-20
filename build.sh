#!/bin/bash
# ==============================================================================
# oMLX Unified Build & Setup Script
# ==============================================================================
# 1. Creates/Updates virtual environment
# 2. Installs all dependencies (MLX, vllm-metal, etc.)
# 3. Compiles the C++ Native Core (Python Extension + Standalone Binary)

set -e

echo "🚀 Starting oMLX Unified Build..."

# 1. Run environment setup
./scripts/setup_omlx.sh

# 2. Activate environment
source .omlxvnv/bin/activate

# 3. Run high-performance C++ core build
./bin/build_cpp_core.sh

echo ""
echo "------------------------------------------------"
echo "✨ BUILD COMPLETE ✨"
echo "👉 Python API: Ready (use ./run.sh)"
echo "👉 C++ Standalone: bin/agent_runner"
echo "------------------------------------------------"
