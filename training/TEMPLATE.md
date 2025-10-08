# [Training Script Name]

**Quick Summary**: [One sentence describing what this training script does]

## Overview

[Provide a comprehensive 2-3 paragraph description. Include:
- What the script trains (pre-training, fine-tuning, etc.)
- Key features and capabilities
- Target use cases
- What makes this script unique or useful]

## Keywords

`#training` `#pipeline` `#distributed` `#fine-tuning` `#[framework]` `#[technique]`

[Add relevant tags from the following categories:
- Training type: #pretraining, #fine-tuning, #instruction-tuning, #RLHF, #LoRA, #QLoRA
- Scale: #single-gpu, #multi-gpu, #distributed, #multi-node
- Framework: #pytorch, #tensorflow, #jax, #deepspeed, #FSDP, #accelerate
- Optimization: #mixed-precision, #gradient-checkpointing, #flash-attention
- Monitoring: #wandb, #tensorboard, #mlflow]

## Features

- ✅ [e.g., Distributed training with DeepSpeed/FSDP]
- ✅ [e.g., Mixed precision training (FP16/BF16)]
- ✅ [e.g., Gradient checkpointing for memory efficiency]
- ✅ [e.g., Automatic checkpoint saving and resumption]
- ✅ [e.g., Integration with Weights & Biases]
- ✅ [e.g., Custom data loading and preprocessing]
- ✅ [e.g., Learning rate scheduling]
- ✅ [e.g., Evaluation during training]

## Requirements

### Dependencies

**Core Requirements:**
```bash
pip install torch>=2.0.0
pip install transformers>=4.30.0
pip install datasets>=2.12.0
pip install accelerate>=0.20.0
```

**Optional Requirements:**
```bash
# For distributed training
pip install deepspeed>=0.9.0
# or
pip install fairscale

# For monitoring
pip install wandb
pip install tensorboard

# For optimization
pip install flash-attn
pip install bitsandbytes
```

**Full Requirements File:**
```bash
pip install -r requirements.txt
```

**Create requirements.txt:**
```txt
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
accelerate>=0.20.0
deepspeed>=0.9.0
wandb
tensorboard
numpy
tqdm
pyyaml
```

### Hardware Requirements

**Minimum:**
- GPU: [e.g., 1x NVIDIA GPU with 16GB VRAM]
- RAM: [e.g., 32GB]
- Storage: [e.g., 100GB for checkpoints]

**Recommended:**
- GPU: [e.g., 4x A100 (40GB) or 8x V100 (32GB)]
- RAM: [e.g., 128GB]
- Storage: [e.g., 500GB NVMe SSD]

**For Large Models:**
- GPU: [e.g., 16+ A100 GPUs]
- RAM: [e.g., 256GB+]
- Network: [e.g., High-speed interconnect (InfiniBand)]

## Configuration

### Configuration File

The training script uses a YAML configuration file. Create `config.yaml`:

```yaml
# Model configuration
model:
  name_or_path: "gpt2"  # Model name or path
  tokenizer_name: null   # Optional, defaults to model name
  
# Dataset configuration
data:
  train_file: "data/train.jsonl"
  validation_file: "data/val.jsonl"
  test_file: "data/test.jsonl"
  max_seq_length: 2048
  preprocessing_num_workers: 4
  
# Training hyperparameters
training:
  output_dir: "./output"
  num_train_epochs: 3
  per_device_train_batch_size: 8
  per_device_eval_batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 5e-5
  warmup_steps: 500
  weight_decay: 0.01
  max_grad_norm: 1.0
  
# Optimization
optimization:
  fp16: false
  bf16: true
  gradient_checkpointing: true
  
# Logging and evaluation
logging:
  logging_steps: 10
  eval_steps: 500
  save_steps: 1000
  save_total_limit: 3
  logging_dir: "./logs"
  report_to: ["wandb", "tensorboard"]
  
# DeepSpeed configuration (optional)
deepspeed:
  config_file: "deepspeed_config.json"
  
# Weights & Biases
wandb:
  project: "my-llm-training"
  entity: "my-team"
  name: "experiment-1"
```

### DeepSpeed Configuration

Create `deepspeed_config.json` for distributed training:

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "overlap_comm": true,
    "contiguous_gradients": true
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": false,
    "contiguous_memory_optimization": false,
    "number_checkpoints": null
  },
  "wall_clock_breakdown": false
}
```

## Usage

### Basic Training

**Single GPU:**
```bash
python train.py --config config.yaml
```

**Multiple GPUs (Same Node):**
```bash
# Using torchrun (recommended for PyTorch 2.0+)
torchrun --nproc_per_node=4 train.py --config config.yaml

# Using accelerate
accelerate launch --multi_gpu --num_processes=4 train.py --config config.yaml
```

### Distributed Training

**Multi-Node Setup:**
```bash
# On node 0 (master)
torchrun \
  --nproc_per_node=8 \
  --nnodes=2 \
  --node_rank=0 \
  --master_addr=master_ip \
  --master_port=29500 \
  train.py --config config.yaml

# On node 1
torchrun \
  --nproc_per_node=8 \
  --nnodes=2 \
  --node_rank=1 \
  --master_addr=master_ip \
  --master_port=29500 \
  train.py --config config.yaml
```

**With DeepSpeed:**
```bash
deepspeed --num_gpus=8 train.py \
  --config config.yaml \
  --deepspeed deepspeed_config.json
```

**With Accelerate:**
```bash
# First, configure accelerate
accelerate config

# Then launch training
accelerate launch train.py --config config.yaml
```

### Resuming from Checkpoint

```bash
python train.py \
  --config config.yaml \
  --resume_from_checkpoint output/checkpoint-5000
```

### Custom Arguments

Override config file settings:
```bash
python train.py \
  --config config.yaml \
  --learning_rate 1e-4 \
  --num_train_epochs 5 \
  --output_dir ./custom_output
```

## Data Preparation

### Data Format

Expected data format (JSONL):
```json
{"text": "This is a training example."}
{"text": "Another example for training."}
```

For instruction tuning:
```json
{
  "instruction": "Translate to French:",
  "input": "Hello, how are you?",
  "output": "Bonjour, comment allez-vous?"
}
```

### Preprocessing Script

```python
# preprocess_data.py
import json
from datasets import load_dataset

# Load raw data
dataset = load_dataset("json", data_files="raw_data.jsonl")

# Preprocess
def preprocess(examples):
    # Add your preprocessing logic
    return examples

processed = dataset.map(preprocess, batched=True)

# Save
processed['train'].to_json("data/train.jsonl")
```

Run preprocessing:
```bash
python preprocess_data.py
```

## Monitoring

### Weights & Biases

```bash
# Login to W&B
wandb login

# Training will automatically log to W&B
python train.py --config config.yaml
```

View training progress at: https://wandb.ai/your-entity/your-project

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir ./logs

# Open browser to http://localhost:6006
```

### Logging

The script logs:
- Training loss
- Learning rate
- Gradient norm
- Validation metrics
- Training speed (samples/sec)
- GPU memory usage

Example log output:
```
Step 100: loss=2.456, lr=0.0001, grad_norm=0.823
Step 200: loss=2.234, lr=0.00009, grad_norm=0.712
Evaluation: perplexity=11.23, accuracy=0.654
```

## Advanced Features

### Gradient Checkpointing

Enable in config:
```yaml
optimization:
  gradient_checkpointing: true
```

Saves memory at the cost of ~20% slower training.

### Mixed Precision Training

**BF16 (recommended for A100):**
```yaml
optimization:
  bf16: true
  fp16: false
```

**FP16 (for V100 and older):**
```yaml
optimization:
  fp16: true
  bf16: false
```

### LoRA/QLoRA Fine-tuning

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,  # For QLoRA
    device_map="auto"
)

# Prepare for training
model = prepare_model_for_kbit_training(model)

# Add LoRA adapters
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
```

### Custom Learning Rate Schedulers

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

# Cosine annealing with warmup
warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
main_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)

# Combine schedulers
from torch.optim.lr_scheduler import SequentialLR
scheduler = SequentialLR(optimizer, 
                        schedulers=[warmup_scheduler, main_scheduler],
                        milestones=[warmup_steps])
```

## Troubleshooting

### Out of Memory (OOM)

**Solutions:**
1. Reduce `per_device_train_batch_size`
2. Increase `gradient_accumulation_steps`
3. Enable `gradient_checkpointing`
4. Use smaller sequence length
5. Use DeepSpeed ZeRO stage 2 or 3
6. Use 8-bit or 4-bit quantization

```yaml
# Memory-efficient config
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 32
optimization:
  gradient_checkpointing: true
```

### Slow Training

**Solutions:**
1. Use mixed precision (BF16/FP16)
2. Increase batch size if memory allows
3. Use Flash Attention
4. Optimize data loading (increase `num_workers`)
5. Use faster data format (Parquet instead of JSON)

### Loss Not Decreasing

**Check:**
1. Learning rate (try 1e-5 to 1e-4)
2. Data quality and preprocessing
3. Model initialization
4. Gradient clipping value
5. Warmup steps

### Distributed Training Issues

**Common fixes:**
1. Check network connectivity between nodes
2. Verify same PyTorch version on all nodes
3. Ensure synchronized clocks (NTP)
4. Check firewall settings
5. Use correct master address and port

## Performance Optimization

### Expected Training Speed

| Setup | Hardware | Speed | Notes |
|-------|----------|-------|-------|
| Single GPU | 1x A100 40GB | ~1000 tokens/sec | Base model |
| Multi-GPU | 8x A100 40GB | ~7000 tokens/sec | Linear scaling |
| Multi-Node | 16x A100 40GB | ~13000 tokens/sec | ~90% efficiency |

### Optimization Tips

1. **Data Loading**: Use multiple workers
   ```yaml
   preprocessing_num_workers: 8
   dataloader_num_workers: 4
   ```

2. **Pin Memory**: Speed up GPU transfer
   ```yaml
   dataloader_pin_memory: true
   ```

3. **Prefetch Factor**: Load batches ahead
   ```yaml
   dataloader_prefetch_factor: 2
   ```

## Examples

### Example 1: Fine-tune GPT-2 on Custom Data

```bash
# 1. Prepare data
python prepare_data.py --input raw.txt --output data/

# 2. Configure training
cat > config.yaml << EOF
model:
  name_or_path: "gpt2"
data:
  train_file: "data/train.jsonl"
  validation_file: "data/val.jsonl"
training:
  num_train_epochs: 3
  per_device_train_batch_size: 8
  learning_rate: 5e-5
EOF

# 3. Train
python train.py --config config.yaml
```

### Example 2: Pre-train Model from Scratch

```bash
# Configure for pre-training
python train.py \
  --config pretraining_config.yaml \
  --model_config model_architecture.json \
  --train_from_scratch
```

### Example 3: Instruction Tuning

```bash
# Use instruction tuning format
python train.py \
  --config config.yaml \
  --data_format instruction \
  --instruction_column instruction \
  --input_column input \
  --output_column output
```

## Citation

If you use this training script, please cite:

```bibtex
@software{training_script_name,
  author = {Your Name},
  title = {Training Script Name},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yesh00008/GigaBase}
}
```

## License

This training script is released under the [LICENSE_NAME] license.

## Contact

**Maintainer**: [Your Name]
- **Email**: [your.email@example.com]
- **GitHub**: [@yourusername](https://github.com/yourusername)

## Acknowledgments

- [Framework/Library]: [Acknowledgment]
- [Contributors]: [Names]

## Related Resources

- [Tutorial]: [Link to detailed tutorial]
- [Example Notebooks]: [Link to example notebooks]
- [Best Practices Guide]: [Link to guide]

---

**Last Updated**: [YYYY-MM-DD]

**Keywords**: #training #pipeline #distributed #fine-tuning #LLM #deep-learning #pytorch #open-source
