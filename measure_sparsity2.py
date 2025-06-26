import argparse
import json
import logging
import os
import time
import math
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    set_seed
)
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle

from src.activation_capture import ActivationCapture

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContextualSparsityAnalyzer:
    """Analyzer for measuring contextual sparsity patterns in LLaMA models."""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.capture = ActivationCapture(model)
        self.mlp_sparsity = defaultdict(list)  # Layer -> sparsity ratio
        self.union_sparsity = defaultdict(lambda: defaultdict(list))  # Batch Size -> Layer -> sparsity ratio

        self.num_seqs = 0


    def process_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict:      
        batch_size = input_ids.size(0)

        # Clear previous captures and GPU cache
        self.capture.clear_captures()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Compute sparsity
        for layer_idx in range(len(self.model.model.layers)):
            if layer_idx in self.capture.hidden_states:
                sparsity_masks = (self.capture.hidden_states[layer_idx] > 0)

                # Naive sparsity computation
                self.mlp_sparsity[layer_idx].append(sparsity_masks.long().mean().item())

                # Level of sparsity after union over batch dim
                union_sparsity_mask = sparsity_masks.any(dim=0)
                self.union_sparsity[batch_size][layer_idx].append(union_sparsity_mask.long().mean().item())

                # TODO: Add HNSW sparsity computation for both attn heads and mlp neurons
                # TODO: Compute union sparsity over multiple different batch sizes

        
        # Clear GPU tensors from capture to free memory
        self.capture.clear_captures()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        self.num_seqs += batch_size

    def save_results(self, save_dir: str):
        """Save analysis results to files."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save raw data
        results = {
            'mlp_sparsity': self.mlp_sparsity,
            'union_sparsity': self.union_sparsity
        }
        
        with open(os.path.join(save_dir, 'sparsity_analysis.pkl'), 'wb') as f:
            pickle.dump(results, f)
        
        # Save summary statistics
        
        summary = {
            'total_sequences_analyzed': self.num_seqs,
            'total_layers': len(self.mlp_sparsity),
            #'sparsity_thresholds': self.sparsity_thresholds,
            #'context_window': self.context_window,
            #'consistency_metrics': results['consistency_metrics']
        }
        
        with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        
        logger.info(f"Results saved to {save_dir}")


    def plot_sparsity_analysis(self, save_dir: str):
        """Generate visualization plots for sparsity analysis."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Plot 1: Sparsity ratios across layers and thresholds
        plt.figure(figsize=(12, 8))
        
        layer_names = sorted(self.mlp_sparsity.keys())
        layer_indices = range(len(layer_names))

        # Plot MLP sparsity by layer
        mlp_sparsity_by_layer = np.array([self.mlp_sparsity[layer_name] for layer_name in layer_names])
        plt.figure(figsize=(15, 10))
        
        plt.plot(layer_indices, mlp_sparsity_by_layer)
        plt.xlabel('Layer Index')
        plt.ylabel('\% of Neurons Active')
        plt.title('MLP Sparsity By Layer')
        plt.xticks(layer_indices[::4], [f'L{i}' for i in layer_indices[::4]])

        # Plot union sparsity for each batch size
        plt.figure(figsize=(15, 10))

        for batch_size, sparsity in self.union_sparsity.items():
            union_sparsity_by_layer = np.array([sparsity[layer_name] for layer_name in layer_names])
            plt.plot(layer_indices, union_sparsity_by_layer, label=str(batch_size))
        plt.xlabel('Layer Index')
        plt.ylabel('\% of Neurons Active')
        plt.title('MLP Sparsity By Layer')
        plt.legend()
        plt.xticks(layer_indices[::4], [f'L{i}' for i in layer_indices[::4]])

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'sparsity_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plots saved to {save_dir}")


def analyze_sparsity(args, analyzer, dataloader, device):
    # Setup activation capture
    analyzer.capture.register_hooks(analyzer.model)

    try:
        # Process dataset
        logger.info("Starting contextual sparsity analysis...")
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Analyzing sequences")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            analyzer.process_batch(input_ids, attention_mask)

            # Log progress
            if (batch_idx + 1) % 100 == 0:
                logger.info(f"Processed {batch_idx + 1}/{len(dataloader)} sequences")

        # Save results
        logger.info("Saving analysis results...")
        analyzer.save_results(args.output_dir)
        
        # Generate plots if requested
        if args.save_plots:
            logger.info("Generating visualization plots...")
            analyzer.plot_sparsity_analysis(args.output_dir)

        num_layers = len(analyzer.mlp_sparsity)
        layer_names = sorted(analyzer.mlp_sparsity.keys())
        def quantile_avg(sparsity_by_layer, num_quantiles=4):
            quantiles = [(math.round(i* num_layers/num_quantiles), math.round((i+1)*num_layers/num_quantiles)) for i in range(num_quantiles)]
            quantile_sparsities = [sum([sparsity_by_layer[layer_names[i]] for i in range(quantile[0], quantile[1]+1)]) \
                                    / (quantile[1]-quantile[0]) for quantile in quantiles]
            return quantiles, quantile_sparsities

        print(f"\n{'='*60}")
        print(f"🎯 CONTEXTUAL SPARSITY ANALYSIS SUMMARY")
        print(f"📊 Model: {args.model_name}")
        print(f"📝 Sequences analyzed: {analyzer.num_seqs}")
        print(f"🧠 Layers analyzed: {len(analyzer.mlp_sparsity)}")
        #print(f"🔍 Context patterns found: {len(analyzer.contextual_patterns)}")

        # MLP Sparsity By Layer (Naive)
        print(f"\n🎯 MLP Sparsity by Layer (Naive):")
        quantiles, sparsities = quantile_avg(analyzer.mlp_sparsity, num_quantiles=5)
        for (l0,l1), sparsity in zip(quantiles,sparsities):
            print(f"   Layer {l0} to Layer {l1} Sparsity: {sparsity}")

        print(f"\n🎯 Union Sparsity by Layer and Batch Size (Naive):")
        for batch_size in sorted(analyzer.union_sparsity.keys()):
            print(f"\nBatch Size: {batch_size}")
            quantiles, sparsities = quantile_avg(analyzer.mlp_sparsity, num_quantiles=5)
            for (l0,l1), sparsity in zip(quantiles,sparsities):
                print(f"   Layer {l0} to Layer {l1} Sparsity: {sparsity}")
        
        print(f"\n✅ Analysis completed! Results saved to: {args.output_dir}")
        print(f"📁 Files generated:")
        print(f"   - sparsity_analysis.pkl (raw data)")
        print(f"   - summary.json (summary statistics)")
        if args.save_plots:
            print(f"   - sparsity_analysis.png (visualization)")

    finally:
        analyzer.capture.remove_hooks()


class C4Dataset(Dataset):
    """C4 dataset for contextual sparsity analysis."""
    
    def __init__(self, tokenizer, max_length: int = 512, num_samples: int = 1000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load C4 dataset
        logger.info("Loading C4 dataset...")
        dataset = load_dataset("allenai/c4", "realnewslike", split="train", streaming=True)
        
        # Process samples
        self.samples = []
        for i, sample in enumerate(dataset):
            if i >= num_samples:
                break
                
            text = sample['text']
            if len(text.strip()) > 50:  # Filter out very short texts
                encoding = tokenizer(
                    text,
                    truncation=True,
                    padding=False,
                    max_length=max_length,
                    return_tensors='pt'
                )
                
                if encoding['input_ids'].shape[1] > 10:  # Ensure minimum sequence length
                    self.samples.append({
                        'input_ids': encoding['input_ids'].squeeze(),
                        'attention_mask': encoding['attention_mask'].squeeze(),
                        'text': text[:200] + "..." if len(text) > 200 else text
                    })
        
        logger.info(f"Loaded {len(self.samples)} C4 samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
def main():
    parser = argparse.ArgumentParser(description="Measure contextual sparsity in LLaMA models")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                       help="HuggingFace model name or path")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for results")
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="Number of C4 samples to analyze")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size (recommend 1 for token-by-token analysis)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--save_plots", action="store_true",
                       help="Generate and save analysis plots")
    
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
    
    # Load model and tokenizer
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True
    )
    
    if device.type != "cuda":
        model = model.to(device)
    
    # Load C4 dataset
    dataset = C4Dataset(tokenizer, args.max_length, args.num_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    analyzer = ContextualSparsityAnalyzer(model, tokenizer, device)

    analyze_sparsity(analyzer, dataloader, device)



if __name__ == "__main__":
    main() 

