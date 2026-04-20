#!/bin/bash
# ==============================================================================
# cMLX Server Run Script
# ==============================================================================
# Starts the cMLX server with auto-calculated hardware optimizations.

set -e

# Default settings
MODEL_DIR="${1:-$HOME/.cmlx/models}"
PORT="${2:-8000}"

# Activate environment
if [ -f ".cmlxvnv/bin/activate" ]; then
    source .cmlxvnv/bin/activate
else
    echo "❌ Error: Virtual environment not found. Please run ./build.sh first."
    exit 1
fi

echo "🚀 Launching cMLX Server..."
echo "📍 Model Directory: $MODEL_DIR"
echo "🔌 Port: $PORT"
echo "📝 Logs: ~/.cmlx/logs/server.log"
echo "🛠️ Hardware: Auto-optimized for $(sysctl -n hw.model)"
echo ""

# Start server
# Use --model-dir directly (cmlx.server handles hardware-aware auto-scaling)
python3 -m cmlx.server --model-dir "$MODEL_DIR" --port "$PORT"
