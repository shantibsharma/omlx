#!/bin/bash
# ==============================================================================
# cMLX Server Run Script
# ==============================================================================
# Starts the cMLX server with auto-calculated hardware optimizations.

# Activate environment
if [ -f ".cmlxvnv/bin/activate" ]; then
    source .cmlxvnv/bin/activate
else
    echo "❌ Error: Virtual environment not found. Please run ./build.sh first."
    exit 1
fi

echo "🚀 Launching cMLX Server..."
echo "📍 Model Directory: ${1:-$HOME/.cmlx/models}"
echo "🔌 Port: ${2:-8000}"
echo "📝 Logs: ~/.cmlx/logs/server.log"
echo "🛠️ Hardware: Auto-optimized for $(sysctl -n hw.model)"
echo ""

# The key to Ctrl+C working is having the Python process own the TTY
# or having the shell explicitly pass signals.
# We will use 'exec' but we'll first make sure the native threads are handled.
# By using 'exec', the shell replaces itself with the python process.
# This means the Python process becomes the direct child of your terminal
# and receives Ctrl+C immediately.

exec python3 -m cmlx.server --model-dir "${1:-$HOME/.cmlx/models}" --port "${2:-8000}"
