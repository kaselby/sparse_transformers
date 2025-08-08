#!/usr/bin/env python3
"""
Calculate ground-truth sparsities for various base models on a given dataset.

This script takes a list of HuggingFace-compatible models and runs each model 
on a number of samples from a given dataset. Activation statistics are captured
from the models' forward passes, and used to determine the average ground-truth
sparsity of each layer for each model.

This data can then be plotted or saved in a json file to be used as thresholds
for the topk or statistical-topk sparsity methods using trained predictors. 


Usage examples:
  # Capture ground truth sparsity values for a particular model or models
  python measure_gt_sparsity.py \
    --models meta-llama/Llama-3.2-3B-Instruct \
    --num_samples 2048 \
    --max_length 512 \
    --output_dir sparsities \
    --device cuda

  # Generate a plot of ground truth sparsity values by layer and model
  python measure_gt_sparsity.py \
    --models meta-llama/Llama-3.2-3B-Instruct Qwen/Qwen2-1.5B google/gemma-3n-E2B \
    --num_samples 2048 \
    --max_length 512 \
    --output_dir sparsities \
    --device cuda \
    --make_plots
"""



import argparse
from collections import defaultdict
import json
import logging
import os
from typing import Dict

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import set_seed

import matplotlib.pyplot as plt
from src.activation_capture import Hook

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContextualSparsityAnalyzer:
    """Analyzer for measuring contextual sparsity patterns in LLaMA models."""

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        model.activation_capture = model.ACTIVATION_CAPTURE(model)
        model.activation_capture.register_hooks(hooks=[Hook.ACT])
        self.num_layers = len(self.model.activation_capture.get_layers())

        self.reset_buffers()

    def reset_buffers(self):
        self.mlp_sparsity = defaultdict(list)
        self.num_seqs = 0

    def process_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        batch_size = input_ids.size(0)

        # Clear previous captures and GPU cache
        self.model.activation_capture.clear_captures()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # Forward pass
        with torch.no_grad():
            _ = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Compute sparsity
        for layer_idx in range(self.num_layers):
            sparsity_masks = (
                self.model.activation_capture.mlp_activations[Hook.ACT][layer_idx] <= 0
            )

            # Naive sparsity computation
            self.mlp_sparsity[layer_idx].append(
                sparsity_masks.float().mean().item()
            )

            # Level of sparsity after union over batch dim
            # union_sparsity_mask = sparsity_masks.any(dim=0)
            # self.union_sparsity[batch_size][layer_idx].append(union_sparsity_mask.float().mean().item())

            # TODO: Add HNSW sparsity computation for both attn heads and mlp neurons
            # TODO: Compute union sparsity over multiple different batch sizes

        # Clear GPU tensors from capture to free memory
        self.model.activation_capture.clear_captures()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        self.num_seqs += batch_size


def analyze_sparsity(args, model_name, device):
    # Load model and tokenizer
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )

    if device.type != "cuda":
        model = model.to(device)

    # Load C4 dataset
    dataset = C4Dataset(tokenizer, args.max_length, args.num_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    analyzer = ContextualSparsityAnalyzer(model, tokenizer, device)
    try:
        # Process dataset
        logger.info("Starting contextual sparsity analysis...")

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Analyzing sequences")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            analyzer.process_batch(input_ids, attention_mask)

            # Log progress
            if (batch_idx + 1) % 100 == 0:
                logger.info(f"Processed {batch_idx + 1}/{len(dataloader)} sequences")

        analyzer.mlp_sparsity = [
            sum(analyzer.mlp_sparsity[layer_idx]) / len(analyzer.mlp_sparsity[layer_idx])
            for layer_idx in range(len(analyzer.mlp_sparsity))
        ]
    finally:
        analyzer.model.activation_capture.remove_hooks()
    return analyzer.mlp_sparsity


def plot_sparsities(sparsities, output_dir=None):
    plt.figure(figsize=(10, 6))
    for model, model_sparsities in sparsities.items():
        model_name = model.split("/")[1].capitalize()
        plt.plot([i*100/len(model_sparsities) for i in range(len(model_sparsities))], [x*100 for x in model_sparsities], label=model_name)
    plt.xlabel("Layer Index Percentage (layer_idx/num_layers)")
    plt.ylabel(f"% of Neurons Inactive")
    plt.title(f"ACtivation Sparsity By Layer")
    plt.legend()
    plt.minorticks_on()
    if output_dir:
        plt.savefig(
            os.path.join(output_dir, f"sparsity_analysis.png"),
            dpi=300,
            bbox_inches="tight",
        )


class C4Dataset(Dataset):
    """C4 dataset for contextual sparsity analysis."""

    def __init__(self, tokenizer, max_length: int = 512, num_samples: int = 1000):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load C4 dataset
        logger.info("Loading C4 dataset...")
        dataset = load_dataset(
            "allenai/c4", "realnewslike", split="train", streaming=True
        )

        # Process samples
        self.samples = []
        for i, sample in enumerate(dataset):
            if i >= num_samples:
                break

            text = sample["text"]
            if len(text.strip()) > 50:  # Filter out very short texts
                encoding = tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                    return_tensors="pt",
                )

                if (
                    encoding["input_ids"].shape[1] > 10
                ):  # Ensure minimum sequence length
                    self.samples.append(
                        {
                            "input_ids": encoding["input_ids"].squeeze(),
                            "attention_mask": encoding["attention_mask"].squeeze(),
                            "text": text[:200] + "..." if len(text) > 200 else text,
                        }
                    )

        logger.info(f"Loaded {len(self.samples)} C4 samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def main():
    parser = argparse.ArgumentParser(
        description="Measure contextual sparsity in LLaMA models"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[
            "meta-llama/Llama-3.2-3B-Instruct",
            "Qwen/Qwen2-1.5B",
        ],
        help="HuggingFace model names or paths",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for results"
    )
    parser.add_argument(
        "--num_samples", type=int, default=1000, help="Number of C4 samples to analyze"
    )
    parser.add_argument(
        "--max_length", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (recommend 1 for token-by-token analysis)",
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--make_plots", action="store_true", help="Generate and save analysis plots"
    )

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")

    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)

    outs = defaultdict(dict)
    for model in args.models:
        model_sparsities = analyze_sparsity(args, model, device)
        for k, v in model_sparsities.items():
            outs[k][model] = v
    json.dump(outs, open(os.path.join(args.output_dir, "sparsity.json"), "w"))

    if args.make_plots:
        plot_sparsities(outs)


if __name__ == "__main__":
    main()
