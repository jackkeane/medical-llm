#!/bin/bash
# Launch TensorBoard to monitor training

set -e

# Activate conda environment
source ~/anaconda3/bin/activate py312

# Navigate to project root
cd "$(dirname "$0")/.."

# Check if output directory exists
if [ ! -d "outputs/biomistral-medical" ]; then
    echo "⚠️  Warning: No training runs found yet."
    echo "   Output directory will be created when training starts."
    echo ""
fi

# Default port
PORT=${1:-6006}

echo "📊 Starting TensorBoard..."
echo "Log directory: outputs/biomistral-medical"
echo "URL: http://localhost:$PORT"
echo ""
echo "Press Ctrl+C to stop"
echo "==============================================="

# Launch TensorBoard
tensorboard --logdir=outputs/biomistral-medical --port=$PORT --bind_all
