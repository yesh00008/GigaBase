#!/usr/bin/env python3
"""High-speed GPU fine-tuning entrypoint for datasets in data/downloaded_datasets."""

import argparse
import inspect
import json
import os
import random
from datetime import datetime
from typing import Iterable, List, Optional

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def _bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    is_supported = getattr(torch.cuda, "is_bf16_supported", None)
    return bool(is_supported() if callable(is_supported) else False)


def _list_jsonl_files(data_dir: str) -> List[str]:
    files = []
    for entry in os.listdir(data_dir):
        path = os.path.join(data_dir, entry)
        if os.path.isfile(path) and entry.endswith(".jsonl"):
            files.append(path)
    return sorted(files)


def _extract_text(sample: dict) -> Optional[str]:
    preferred = (
        "text",
        "content",
        "response",
        "output",
        "answer",
        "story",
    )
    for key in preferred:
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value
    question = sample.get("question")
    answer = sample.get("answer")
    if isinstance(question, str) and isinstance(answer, str):
        combo = f"Q: {question.strip()}\nA: {answer.strip()}"
        if combo.strip():
            return combo
    instruction = sample.get("instruction")
    completion = sample.get("completion") or sample.get("output")
    if isinstance(instruction, str) and isinstance(completion, str):
        combo = f"{instruction.strip()}\n{completion.strip()}"
        if combo.strip():
            return combo
    return None


def _stream_jsonl(path: str, limit: int) -> Iterable[str]:
    collected = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if limit and collected >= limit:
                break
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = _extract_text(payload)
            if text:
                collected += 1
                yield text


def _collect_corpus(files: List[str], limit_per_file: int, max_samples: int) -> List[str]:
    bucket: List[str] = []
    for path in files:
        for text in _stream_jsonl(path, limit_per_file):
            bucket.append(text)
            if max_samples and len(bucket) >= max_samples:
                return bucket
    return bucket


def _maybe_prepare_lora(model: AutoModelForCausalLM, r: int, alpha: int, dropout: float, target_modules: Optional[List[str]]):
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise RuntimeError("peft is required for LoRA fine-tuning; install per requirements.txt") from exc

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    return get_peft_model(model, config)


def _load_model(model_name: str, use_4bit: bool, lora: bool, lora_r: int, lora_alpha: int, lora_dropout: float, target_modules: Optional[List[str]]):
    kwargs = {}
    device_map = "auto" if torch.cuda.is_available() else None
    if use_4bit and torch.cuda.is_available():
        try:
            kwargs.update({
                "load_in_4bit": True,
                "device_map": device_map,
            })
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if _bf16_supported() else torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        except ImportError as exc:
            raise RuntimeError("bitsandbytes is required for 4-bit loading; disable --use-4bit or install it") from exc
    else:
        dtype = None
        if torch.cuda.is_available():
            if _bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16
        kwargs.update({
            "torch_dtype": dtype,
            "device_map": device_map,
        })

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    if lora:
        model = _maybe_prepare_lora(model, lora_r, lora_alpha, lora_dropout, target_modules)

    return model


def _prepare_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _build_dataset(texts: List[str], tokenizer, seq_len: int):
    dataset = Dataset.from_dict({"text": texts})

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=seq_len, padding="max_length")

    return dataset.map(tokenize, batched=True, remove_columns="text")


def _split_dataset(dataset: Dataset, eval_ratio: float):
    if eval_ratio <= 0 or len(dataset) < 2:
        return dataset, None
    ratio = min(0.5, max(1 / len(dataset), eval_ratio))
    split = dataset.train_test_split(test_size=ratio, seed=42)
    return split["train"], split["test"]


def _configure_training_args(args, default_batch: int, steps: int, output: str, eval_steps: int, learning_rate: float, weight_decay: float, grad_accum: int):
    tf32 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.benchmark = True

    bf16 = _bf16_supported()
    fp16 = torch.cuda.is_available() and not bf16

    base_kwargs = {
        "output_dir": output,
        "overwrite_output_dir": True,
        "per_device_train_batch_size": default_batch,
        "per_device_eval_batch_size": default_batch,
        "gradient_accumulation_steps": grad_accum,
        "max_steps": steps,
        "num_train_epochs": 1,
        "warmup_steps": 0,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "logging_steps": max(1, steps // 5),
        "save_strategy": "no",
        "report_to": "none",
        "fp16": fp16,
        "bf16": bf16,
        "optim": "adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        "dataloader_num_workers": args.dataloader_workers,
    }

    optional_kwargs = {
        "evaluation_strategy": "steps" if eval_steps else "no",
        "eval_steps": eval_steps if eval_steps else None,
    }

    init_params = inspect.signature(TrainingArguments.__init__).parameters
    for key, value in optional_kwargs.items():
        if key in init_params and value is not None:
            base_kwargs[key] = value

    # Remove keys unsupported by older Transformers releases
    supported_kwargs = {k: v for k, v in base_kwargs.items() if k in init_params}

    return TrainingArguments(**supported_kwargs)


def _train(args):
    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {args.data_dir}")

    files = _list_jsonl_files(args.data_dir)
    if not files:
        raise RuntimeError(f"No .jsonl files in {args.data_dir}")

    start = datetime.now()
    texts = _collect_corpus(files, args.limit_per_file, args.max_samples)
    if len(texts) < 2:
        raise RuntimeError("Not enough samples to train")

    tokenizer = _prepare_tokenizer(args.model)
    dataset = _build_dataset(texts, tokenizer, args.sequence_length)
    train_dataset, eval_dataset = _split_dataset(dataset, args.eval_ratio)

    model = _load_model(
        model_name=args.model,
        use_4bit=args.use_4bit,
        lora=args.enable_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = _configure_training_args(
        args,
        default_batch=args.batch_size,
        steps=args.max_steps,
        output=args.output_dir,
        eval_steps=args.eval_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_accum=args.gradient_accumulation,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    elapsed = (datetime.now() - start).total_seconds()
    print(f"Training finished in {elapsed:.2f}s using {len(texts)} samples")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Time-constrained GPU micro-finetuning")
    parser.add_argument("--data-dir", default="data/downloaded_datasets", help="Folder with JSONL corpora")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Base checkpoint")
    parser.add_argument("--output-dir", default="models/fast_gpu_run", help="Where to store adapters/model")
    parser.add_argument("--limit-per-file", type=int, default=2048, help="Samples per JSONL to read")
    parser.add_argument("--max-samples", type=int, default=4096, help="Global sample budget")
    parser.add_argument("--sequence-length", type=int, default=512, help="Token sequence length")
    parser.add_argument("--batch-size", type=int, default=16, help="Per device batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--max-steps", type=int, default=40, help="Total optimizer steps")
    parser.add_argument("--eval-ratio", type=float, default=0.05, help="Evaluation split ratio")
    parser.add_argument("--eval-steps", type=int, default=20, help="Evaluation interval in steps")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use-4bit", action="store_true", help="Enable 4-bit loading when GPUs present")
    parser.add_argument("--enable-lora", action="store_true", help="Use LoRA adapters for training speed")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lora-target-modules", nargs="*", default=None, help="Specific module names for LoRA")
    parser.add_argument("--dataloader-workers", type=int, default=2, help="PyTorch DataLoader workers")
    parser.add_argument("--max-samples-fraction", type=float, default=0.0, help="If >0, override samples using fraction of corpus")
    parser.add_argument("--dry-run", action="store_true", help="Load data and model without training")
    return parser.parse_args()


def main():
    args = _parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.max_samples_fraction > 0:
        files = _list_jsonl_files(args.data_dir)
        corpus_count = 0
        for path in files:
            with open(path, "r", encoding="utf-8") as handle:
                corpus_count += sum(1 for _ in handle)
        args.max_samples = max(1, int(corpus_count * args.max_samples_fraction))

    if args.dry_run:
        files = _list_jsonl_files(args.data_dir)
        texts = _collect_corpus(files, args.limit_per_file, args.max_samples)
        print(f"Dry run collected {len(texts)} samples from {len(files)} files")
        tokenizer = _prepare_tokenizer(args.model)
        model = _load_model(
            model_name=args.model,
            use_4bit=args.use_4bit,
            lora=args.enable_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
        )
        print(f"Tokenizer vocab: {tokenizer.vocab_size}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        return

    _train(args)


if __name__ == "__main__":
    main()
