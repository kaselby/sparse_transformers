from collections import defaultdict
import logging
import os
import json
from tqdm import tqdm
import argparse

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_utils import set_seed

from src.activation_capture import ActivationCapture, Hook

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


import copy
class ThresholdEvaluator():
    def __init__(self, model, thresholds):
        self.model = model
        self.thresholds = thresholds

        self.compute_neuron_thresholds(thresholds)

        self.mlp_outputs = defaultdict(list)
        self.handles = []

    def get_layers(self):
        return self.model.model.layers

    def compute_neuron_thresholds(self, thresholds):
        n_layers = len(self.get_layers())
        self.neuron_thresholds = torch.zeros(n_layers, self.model.config.intermediate_size)
        with torch.no_grad():
            for layer_idx, layer in self.get_layers():
                norms = layer.mlp.down_proj.weight.norm(dim=0)
                self.neuron_thresholds[layer_idx] = thresholds[layer_idx] * norms

    def _inspect_hook(self, layer_idx):
        def hook(module, input, output):
            # Just detach, don't clone or move to CPU yet
            out = output.view(-1, output.size(-1)).clone().detach()
            self.mlp_outputs[layer_idx].append(out)
            return output
        return hook
    
    def _threshold_hook(self, layer_idx):
        def hook(module, input, output):
            # Just detach, don't clone or move to CPU yet
            mask = (output > self.neuron_thresholds[layer_idx]).bool()
            return output * mask
        return hook
    
    def apply_thresholds(self):
        for layer_idx, layer in enumerate(self.get_layers()):
            handle = layer.mlp.act_fn.register_forward_hook(
                self._threshold_hook(layer_idx)
            )
            self.handles.append(handle)

    def apply_hooks(self):
        for layer_idx, layer in enumerate(self.get_layers()):
            handle = layer.mlp.register_forward_hook(
                self._inspect_hook(layer_idx)
            )
            self.handles.append(handle)

    def clear_captures(self):
        self.mlp_outputs = defaultdict(list)

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def evaluate(self, inputs):
        self.apply_hooks()
                
        with torch.no_grad():
            for inp in inputs:
                _ = self.model(**inp)
        
        ground_truth_outputs = {
            idx: torch.cat(outputs_idx, dim=0) for idx,outputs_idx in self.mlp_outputs
        }
        self.clear_captures()

        self.apply_thresholds()
        with torch.no_grad():
            for inp in inputs:
                _ = self.model(**inp)

        threshold_outputs = {
            idx: torch.cat(outputs_idx, dim=0) for idx,outputs_idx in self.mlp_outputs
        }
        self.clear_captures()



#
#   TODO:
#       1. Test out precomputing down_proj norms and see if that improves performance
#       2. Ensure that the thresholds lead to reasonable results for downstream evaluation
#
#




def cett_from_threshold(neuron_outputs, threshold, norms=None, tot_norm=None):
    if not norms:   # pass both or neither
        norms = norms = neuron_outputs.norm(dim=-2).unsqueeze(-2)
        tot_norm = neuron_outputs.sum(dim=-1).norm(dim=-1)
    threshold_norm = ((norms < threshold) * neuron_outputs).sum(dim=-1).norm(dim=-1)
    return threshold_norm / tot_norm

'''
def calculate_threshold_by_token(neuron_outputs, cett_target, n_thresholds=10000):
    neuron_outputs = neuron_outputs.view(-1, *neuron_outputs.size()[-2:])
    norms = neuron_outputs.norm(dim=-2).unsqueeze(-2)
    min_value = norms.min()
    max_value = norms.quantile(0.99)
    threshold_grid = torch.linspace(min_value, max_value, n_thresholds)
    tot_norm = neuron_outputs.sum(dim=-1).norm(dim=-1)
    thresholds = torch.zeros(neuron_outputs.size(0))
    
    initial_cett = cett_from_threshold(neuron_outputs, max_value, norms=norms, tot_norm=tot_norm)
    thresholds[initial_cett < cett_target] = max_value
    
    for j in tqdm(range(neuron_outputs.size(0))):
        if thresholds[j] == 0:
            left = 0
            right = n_thresholds
            while left < right:
                mid = (left + right) // 2
                cett = cett_from_threshold(neuron_outputs[j], threshold_grid[mid], norms=norms[j], tot_norm=tot_norm[j])
                if cett <= cett_target:
                    left = mid + 1
                else:
                    right = mid
            thresholds[j] = threshold_grid[left]
    return thresholds
'''
        
def calculate_threshold(neuron_outputs, cett_target, n_thresholds=10000):
    neuron_outputs = neuron_outputs.view(-1, *neuron_outputs.size()[-2:])
    norms = neuron_outputs.norm(dim=-2).unsqueeze(-2)
    tot_norm = neuron_outputs.sum(dim=-1).norm(dim=-1)
    
    min_value = norms.min()
    max_value = norms.quantile(0.99)
    threshold_grid = torch.linspace(min_value, max_value, n_thresholds)
    max_cett = cett_from_threshold(neuron_outputs, max_value, norms=norms, tot_norm=tot_norm)
    outlier_mask = max_cett > cett_target
    
    left = 0
    right = n_thresholds
    while left < right:
        print(left,right)
        mid = (left + right) // 2
        cett = cett_from_threshold(neuron_outputs, threshold_grid[mid], norms=norms, tot_norm=tot_norm) # Compute CETT for each token
        cett = cett[outlier_mask].mean()    # Remove outliers and take average
        if cett <= cett_target:
            left = mid + 1
        else:
            right = mid
    return threshold_grid[left]


def find_thresholds(
        model_name: str, 
        dataset_name: str, 
        dataset_config: str,
        save_path: str,
        batch_size: int = 8,
        max_samples: int = 128, 
        max_length: int = 256,
        cett_target: float = 0.2, 
        n_thresholds: int = 10000,
        num_workers: int = 8,
        seed: int = 42,
        device: torch.device = torch.device("cpu"),
    ):

    # Load tokenizer and model
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )

    if device.type != "cuda":
        model = model.to(device)

    model.eval()
    model.activation_capture = ActivationCapture(model)
    model.activation_capture.register_hooks(hooks=[Hook.UP])

    # Load dataset
    logger.info(f"Loading dataset: {dataset_name}")
    if dataset_config:
        dataset = load_dataset(
            dataset_name, dataset_config, split="train", streaming=True
        )
    else:
        dataset = load_dataset(dataset_name, split="train", streaming=True)
    dataset = dataset.shuffle(buffer_size=10000, seed=seed)

    def sample_and_tokenize(examples):
        """Sample text chunks before tokenization for efficiency using vectorized operations."""
        texts = examples["text"]
        tokenized = tokenizer(
            texts, 
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )

        # Convert to lists
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"]
        }

    # Tokenize
    dataset = dataset.take(max_samples).map(sample_and_tokenize, batched=False)
    dataset = dataset.with_format("torch")

    dataloader = TorchDataLoader(dataset, batch_size=1, num_workers=num_workers, pin_memory=False, prefetch_factor=2)  # type: ignore

    # Compute thresholds for each layer across all dataset entries
    logger.info(f"Beginning to compute thresholds using {max_samples} samples")
    thresholds = defaultdict(list)
    with torch.no_grad():
        for batch in tqdm(dataloader, total=max_samples):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
        
            _ = model(input_ids=input_ids.squeeze(0), attention_mask=attention_mask.squeeze(0))

            for layer_idx, layer in enumerate(model.activation_capture.get_layers()):
                down_weight = layer.mlp.down_proj.weight
                activations = model.activation_capture.mlp_activations[Hook.UP][layer_idx]
                neuron_outputs = activations.unsqueeze(-2) * down_weight
                threshold = calculate_threshold(neuron_outputs, cett_target, n_thresholds)
                thresholds[layer_idx].append(threshold)
            
            model.activation_capture.clear_captures()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    for layer_idx, layer_thresholds in thresholds.items():
        thresholds[layer_idx] = sum(layer_thresholds) / len(layer_thresholds)

    # Save layerwise thresholds as record in central json file
    if not os.path.exists(save_path):
        with open("save_path", mode="r", encoding="utf-8") as read_file:
            threshold_dict = json.load(read_file)
    else:
        threshold_dict = {}
    threshold_dict[model_name] = thresholds
    with open("save_path", mode="r", encoding="utf-8") as write_file:
        json.dump(threshold_dict, write_file)



def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate training dataset for sparsity predictors"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name or path of the base model (e.g., meta-llama/Llama-2-7b-hf)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="allenai/c4",
        help="Dataset name (default: allenai/c4)",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="en",
        help="Dataset configuration (e.g., en for C4)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="thresholds.json",
        help="Path to json file for thresholds",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=500,
        help="Maximum number of samples to process",
    )
    parser.add_argument(
        "--cett_target",
        type=float,
        default=0.2,
        help="Optimal CETT value for threshold-finding",
    )
    parser.add_argument(
        "--n_quantiles",
        type=int,
        default=500,
        help="Number of quantiles to sort neuron outputs into for threshold-finding",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)"
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Set seed
    set_seed(args.seed)
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    find_thresholds(
        model_name=args.model_name, 
        dataset_name=args.dataset, 
        dataset_config=args.dataset_config,
        max_samples=args.max_samples, 
        cett_target=args.cett_target, 
        n_quantiles=args.n_quantiles,
        save_path=args.save_path,
        seed=args.seed,
        device=device,
    )
                
