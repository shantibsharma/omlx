#!/bin/bash
# ==============================================================================
# oMLX Server Run Script
# ==============================================================================
# Starts the oMLX server with auto-calculated hardware optimizations.

set -e

# Default settings
MODEL_DIR="${1:-$HOME/.omlx/models}"
PORT="${2:-8000}"

# Activate environment
if [ -f ".omlxvnv/bin/activate" ]; then
    source .omlxvnv/bin/activate
else
    echo "❌ Error: Virtual environment not found. Please run ./build.sh first."
    exit 1
fi

echo "🚀 Launching oMLX Server..."
echo "📍 Model Directory: $MODEL_DIR"
echo "🔌 Port: $PORT"
echo "📝 Logs: ~/.omlx/logs/server.log"
echo "🛠️ Hardware: Auto-optimized for $(sysctl -n hw.model)"
echo ""

# Start server
# Use --model-dir directly (omlx.server handles hardware-aware auto-scaling)
python3 -m omlx.server --model-dir "$MODEL_DIR" --port "$PORT"
