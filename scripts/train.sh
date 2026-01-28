#!/bin/bash
# Medical LLM Training Script

set -e

# Activate conda environment
source ~/anaconda3/bin/activate py312

# Navigate to project root
cd "$(dirname "$0")/.."

# Set CUDA visible devices (adjust based on your GPU setup)
export CUDA_VISIBLE_DEVICES=0

# Run training
echo "🏥 Starting Medical LLM Fine-Tuning..."
echo "Model: BioMistral-7B"
echo "Method: QLoRA (4-bit)"
echo "Dataset: 800 medical samples"
echo "==============================================="

llamafactory-cli train configs/biomistral_qlora.yaml

echo ""
echo "✅ Training complete!"
echo "Outputs saved to: outputs/biomistral-medical/"
