#!/bin/bash
# Chat with the fine-tuned medical model

set -e

# Activate conda environment
source ~/anaconda3/bin/activate py312

# Navigate to project root
cd "$(dirname "$0")/.."

# Check if model exists
if [ ! -d "outputs/biomistral-medical" ]; then
    echo "❌ Error: No trained model found at outputs/biomistral-medical"
    echo "   Run training first: ./scripts/train.sh"
    exit 1
fi

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=0

echo "🏥 Medical LLM Chat Interface"
echo "Model: BioMistral-7B (QLoRA fine-tuned)"
echo "==============================================="
echo ""
echo "⚠️  MEDICAL DISCLAIMER:"
echo "This is an AI model for educational purposes only."
echo "Do NOT use for medical diagnosis or treatment decisions."
echo "Always consult qualified healthcare providers."
echo ""
echo "Type your medical questions below (Ctrl+C to exit):"
echo ""

# Launch chat interface
llamafactory-cli chat configs/biomistral_qlora.yaml
