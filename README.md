# Medical LLM Fine-Tuning Project

Educational medical AI assistant using BioMistral-7B + QLoRA fine-tuning.

⚠️ **DISCLAIMER:** For educational/research purposes only. NOT for clinical use.

## Project Structure

```
medical-llm/
├── data/
│   ├── raw/              # Original datasets
│   └── processed/        # Cleaned & formatted data (800 train, 100 val, 100 test)
├── scripts/              # Data preparation & utilities
│   ├── prepare_medical_data.py  # Dataset preparation
│   ├── train.sh          # Training script
│   └── chat.sh           # Chat interface
├── configs/              # Training configurations
│   ├── dataset_info.json # Dataset registry
│   └── biomistral_qlora.yaml  # QLoRA training config
├── models/
│   └── biomistral-7b/    # BioMistral-7B base model (14GB)
├── outputs/              # Training outputs & checkpoints
├── logs/                 # Training logs
└── medical-llm-plan.md   # Detailed implementation plan
```

## Environment

- Python 3.12 conda environment (`py312`)
- LLaMA Factory v0.9.4 (global installation)
- PyTorch 2.4.0 with CUDA support

## Quick Start

### 1. Train the Model

```bash
./scripts/train.sh
```

**Training details:**
- Base model: BioMistral-7B (pre-trained on medical literature)
- Method: QLoRA (4-bit quantization + LoRA)
- Dataset: 800 PubMedQA samples
- Epochs: 3
- Estimated time: ~2-4 hours (depending on GPU)
- VRAM: ~16-18GB

### 2. Chat with the Model

```bash
./scripts/chat.sh
```

## Configuration

Edit `configs/biomistral_qlora.yaml` to adjust:
- Learning rate: `5.0e-5`
- LoRA rank: `32`
- Batch size: `2` (increase if you have more VRAM)
- Max sequence length: `4096`
- Number of epochs: `3`

## Monitoring Training

### TensorBoard (Real-time)

Launch TensorBoard to monitor training in real-time:

```bash
./scripts/tensorboard.sh
```

Then open your browser to: `http://localhost:6006`

**Custom port:**
```bash
./scripts/tensorboard.sh 8080
```

**Metrics tracked:**
- Training loss
- Learning rate
- Gradient norm
- Evaluation metrics
- GPU memory usage

### Log Files

Logs are saved to `outputs/biomistral-medical/`:
- `trainer_log.jsonl` - Training metrics
- `runs/` - TensorBoard event files
- Loss plots (if plot_loss enabled)

## Dataset Details

**Source:** PubMedQA (biomedical literature Q&A)
- Training: 800 samples
- Validation: 100 samples
- Test: 100 samples
- Format: Alpaca (instruction, input, output)
- Safety: All responses include medical disclaimers

## Model Outputs

After training, find checkpoints in:
```
outputs/biomistral-medical/
├── checkpoint-XXX/       # Intermediate checkpoints
└── adapter_model.bin     # Final LoRA adapter
```

## Safety & Ethics

This model is for **educational purposes only**:
- ❌ NOT for diagnosing patients
- ❌ NOT for prescribing medications
- ❌ NOT for clinical decision-making
- ✅ For learning about medical AI
- ✅ For research and experimentation
- ✅ For understanding medical NLP

## Status

✅ **Ready to train!**
- Environment configured
- Dataset prepared (1000 samples)
- Base model downloaded (BioMistral-7B)
- Training config ready
- Scripts ready to run

## Requirements - Recommended Updates

| Package | Current | Recommendation |
|---------|---------|----------------|
| torch | >=2.0.0 | Update to >=2.10.0 |
| bitsandbytes | >=0.43.0 | Update to >=0.45.0 (CUDA 12.6 support) |
| transformers | >=4.38.0 | Update to >=4.48.0 for PyTorch 2.10 |
| accelerate | >=0.28.0 | Update to >=0.36.0 |
| peft | >=0.9.0 | Update to >=0.14.0 |
| trl | >=0.7.10 | Update to >=0.14.0 |
| llamafactory | >=0.8.0 | Update to >=0.9.0 (check their PyTorch 2.10 support) |
