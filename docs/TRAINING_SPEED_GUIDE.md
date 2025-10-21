# Training Speed Guide

This guide helps you select the right training method based on your hardware, dataset size, and time constraints.

## Overview of Training Methods

We offer three training methods with different speed/quality tradeoffs:

1. **Standard Training**: Best quality, slowest training
2. **Fast Training**: Good quality, faster training
3. **Ultra-Fast Training**: Acceptable quality, extreme speed

## Quick Selection Guide

| Situation | Recommended Method | Command |
|-----------|-------------------|---------|
| Small dataset, no GPU | Standard | `python src/train.py` |
| Small/medium dataset, consumer GPU | Fast | `python scripts/fast_train.py` |
| Large dataset, high-end GPU | Ultra-Fast | `python scripts/ultra_fast_train.py` |
| Large model (>1B parameters) | Ultra-Fast | `python scripts/ultra_fast_train.py --no-peft` |
| Production-quality model | Standard | `python src/train.py --epochs 3` |

## Hardware Requirements

### Standard Training
- **CPU**: Any modern CPU with 4+ cores
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: Optional, any CUDA-compatible

### Fast Training
- **CPU**: 4+ cores, 2.5GHz+
- **RAM**: 16GB recommended
- **GPU**: CUDA-compatible with 8GB+ VRAM

### Ultra-Fast Training
- **CPU**: 8+ cores, 3.0GHz+
- **RAM**: 32GB recommended
- **GPU**: CUDA-compatible with 16GB+ VRAM
- **SSD**: Fast storage recommended

## Detailed Optimization Table

| Feature | Standard | Fast | Ultra-Fast |
|---------|----------|------|------------|
| Training precision | FP32 | FP16/BF16 | INT8/FP16 |
| Gradient checkpointing | No | Yes | Yes |
| Gradient accumulation | Basic | Optimized | Advanced |
| Batch size | Small | Medium | Dynamic |
| Data loading | Standard | Optimized | Parallel |
| Tokenization | Standard | Parallel | Parallel + Cache |
| Memory offloading | No | No | Yes (ZeRO-3) |
| Parameter-efficient | No | No | Yes (LoRA) |
| Learning rate schedule | Standard | Cosine | Cosine + Warmup |

## Command Line Examples

### Standard Training

```bash
# Basic training
python src/train.py --model gpt2 --data-file data/processed/training_data.jsonl

# Longer training for better quality
python src/train.py --model gpt2 --epochs 3 --batch-size 4 --generate
```

### Fast Training

```bash
# Basic fast training
python scripts/fast_train.py --model gpt2 --fp16

# Optimized for consumer GPU
python scripts/fast_train.py --model gpt2 --batch-size 8 --gradient-accumulation 16 --fp16
```

### Ultra-Fast Training

```bash
# Maximum speed on high-end GPU
python scripts/ultra_fast_train.py --model gpt2

# For larger models (1B+ parameters)
python scripts/ultra_fast_train.py --model facebook/opt-1.3b --batch-size 1 --gradient-accumulation 32

# Speed vs quality tradeoff
python scripts/ultra_fast_train.py --model gpt2 --epochs 0.5 --max-samples 10000
```

## Troubleshooting

### Out of Memory Errors

1. Reduce batch size: `--batch-size 1`
2. Increase gradient accumulation: `--gradient-accumulation 32`
3. Use a smaller model: Try `distilgpt2` instead of `gpt2`
4. For ultra-fast, ensure 8-bit is enabled (default)

### Slow Training

1. Verify GPU is being used (check with `nvidia-smi`)
2. Try enabling mixed precision: `--fp16`
3. Reduce sequence length: `--max-length 128`
4. Use streaming mode for large datasets: `--streaming`

### Poor Quality Results

1. Increase training epochs: `--epochs 3`
2. Use standard training for final models
3. Verify dataset quality
4. For ultra-fast with LoRA, try adjusting rank: `--lora-rank 32`