"""
Demo: Bidirectional Steering Effect (Batch Mode)

This script iterates ALL samples in the dataset, applies steering, and saves
both before-steering and after-steering results to a JSON file.

- Positive direction: negative -> positive (add +1 on top-k dimensions)
- Negative direction: positive -> negative (add -1 on top-k dimensions)

Key: Steering is applied to ALL tokens on TOP-K important dimensions only.

Usage:
    python demo_steering.py --direction positive --layer 9 --alpha 1.0 --top_k 10 --output_path results.json
"""

import argparse
import json
import random
import numpy as np

import torch
import torch.nn as nn
from nnsight import LanguageModel
from transformers import AutoTokenizer


class LogisticRegression(nn.Module):
    """Probe model for extracting important dimensions."""
    def __init__(self, input_dim, num_classes=2, use_bias=True):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes, bias=use_bias)

    def forward(self, x):
        return torch.softmax(self.linear(x), dim=1)


def get_top_k_dimensions(probe_model_path: str, top_k: int = 10):
    """
    Load probe model and extract top-k most important dimensions.
    
    Args:
        probe_model_path: Path to the probe model .pth file
        top_k: Number of top dimensions to extract
    
    Returns:
        top_dims: List of top-k dimension indices
        weight_diffs: Weight differences for those dimensions
    """
    print(f"Loading probe model from {probe_model_path}...")

    # Infer input_dim from saved weights to avoid hardcoding
    device = 'cpu'
    state_dict = torch.load(probe_model_path, map_location=device)
    input_dim = state_dict['linear.weight'].shape[1]

    probe_model = LogisticRegression(input_dim=input_dim, num_classes=2, use_bias=True)
    probe_model.load_state_dict(state_dict)
    probe_model.eval()

    # Extract weights
    weights = probe_model.linear.weight.data.cpu().numpy()  # Shape: [2, input_dim]
    
    class_0_weights = weights[0, :]  # Benign
    class_1_weights = weights[1, :]  # Malicious
    
    # Calculate importance: absolute weight difference
    weight_diff = np.abs(class_1_weights - class_0_weights)
    
    # Get top-k dimensions
    top_indices = np.argsort(weight_diff)[::-1][:top_k]
    top_weight_diffs = weight_diff[top_indices]
    
    print(f"✓ Extracted top {top_k} dimensions:")
    for i, (dim, diff) in enumerate(zip(top_indices, top_weight_diffs)):
        print(f"  {i+1}. Dimension {dim}: weight_diff = {diff:.4f}")
    
    return list(top_indices), top_weight_diffs


def load_samples(data_path: str):
    """Load all samples from data file."""
    print(f"Loading data from {data_path}...")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Separate by label
    positive_samples = [item for item in data if item.get('label') == 0]
    negative_samples = [item for item in data if item.get('label') == 1]
    
    print(f"✓ Found {len(positive_samples)} positive samples (benign)")
    print(f"✓ Found {len(negative_samples)} negative samples (malicious)")
    
    return positive_samples, negative_samples


def conversation_to_text(conversation, tokenizer):
    """Convert conversation to text using chat template."""
    return tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False  # Return string, not tokens
    )


def extract_next_step(full_output: str) -> str:
    """
    Extract only the next step (assistant's response) from the full output.
    Looks for the last assistant message in the output.
    """
    # Try to find the last <|start_header_id|>assistant<|end_header_id|> marker
    # This is for Llama 3 format
    marker = "<|start_header_id|>assistant<|end_header_id|>"
    
    if marker in full_output:
        # Find the last occurrence
        last_idx = full_output.rfind(marker)
        # Extract content after the marker
        content = full_output[last_idx + len(marker):]
        # Remove trailing special tokens
        content = content.replace("<|eot_id|>", "").strip()
        return content
    
    # Fallback: try to find [/INST] marker (for other formats)
    if "[/INST]" in full_output:
        last_idx = full_output.rfind("[/INST]")
        content = full_output[last_idx + len("[/INST]"):]
        return content.strip()
    
    # If no marker found, return the full output
    return full_output


@torch.no_grad()
def generate_without_steering(model, tokenizer, conversation, max_new_tokens=100):
    """Generate text without steering (baseline) using nnsight generate."""
    # Convert conversation to text prompt
    prompt = conversation_to_text(conversation, tokenizer)
    
    # Get input length to extract only new tokens
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_length = input_ids.shape[1]
    
    # Use nnsight's generate directly with text - it handles tokenization internally
    with model.generate(prompt, max_new_tokens=max_new_tokens, do_sample=False) as generator:
        # No intervention, just save the output
        output_ids = model.generator.output.save()
    
    # Decode only the newly generated tokens (next step)
    new_tokens = output_ids[0][input_length:]
    next_step = model.tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    return next_step.strip()


@torch.no_grad()
def generate_with_steering(model, tokenizer, conversation, layer, top_dims=None, alpha=1.0, direction='positive', max_new_tokens=100):
    """
    Generate text with steering applied to ALL tokens at every generation step.
    Uses PyTorch forward hooks for reliable steering.
    
    Steering direction: Directly add +1 or -1 on top_dims dimensions
    - direction='positive': add +1 (negative -> positive steering)
    - direction='negative': add -1 (positive -> negative steering)
    
    Args:
        model: Language model (nnsight LanguageModel)
        tokenizer: Tokenizer
        conversation: Input conversation
        layer: Layer to apply steering
        top_dims: List of dimension indices to steer (required)
        alpha: Steering strength (multiplier for +1/-1)
        direction: 'positive' (add +1) or 'negative' (add -1)
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        Generated text (only the next step)
    """
    # Convert conversation to text prompt
    prompt = conversation_to_text(conversation, tokenizer)
    
    # Get input length to extract only new tokens
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_length = input_ids.shape[1]
    
    # Determine steering value based on direction
    if direction == 'positive':
        steer_value = 1.0  # Steer towards positive (benign)
    elif direction == 'negative':
        steer_value = -1.0  # Steer towards negative (malicious)
    else:
        raise ValueError(f"Invalid direction: {direction}. Must be 'positive' or 'negative'")
    
    # Create steering vector: only modify specified dimensions
    if top_dims is None:
        raise ValueError("top_dims must be specified for steering")

    # Get the underlying transformers model
    underlying_model = model._model if hasattr(model, '_model') else model

    hidden_dim = underlying_model.config.hidden_size
    steering_vec = torch.zeros(hidden_dim)
    for dim in top_dims:
        steering_vec[dim] = steer_value

    print(f"  Steering direction: {direction} (value={steer_value})")
    print(f"  Steering on {len(top_dims)} dimensions: {top_dims}")
    print(f"  Steering vector norm: {steering_vec.norm().item():.4f}")
    
    # Get the target layer
    target_layer = underlying_model.model.layers[layer]
    
    # Define steering hook - will be called at every forward pass
    def steering_hook(module, input, output):
        # For LLaMA layers, output is a tuple: (hidden_states, ...) or just hidden_states
        # We need to modify the hidden_states in-place to avoid return type issues
        
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        
        # Move steering vector to the same device as hidden states
        sv = steering_vec.to(hidden_states.device).to(hidden_states.dtype)
        
        # Add steering vector to ALL tokens IN-PLACE
        # hidden_states shape: [batch, seq_len, hidden_dim]
        # sv shape: [hidden_dim] (sparse if top_dims specified)
        hidden_states.add_(alpha * sv)
        
        # Don't return anything - in-place modification
        return None
    
    # Register the hook
    hook_handle = target_layer.register_forward_hook(steering_hook)
    
    try:
        # Move input to device
        device = next(underlying_model.parameters()).device
        input_ids_device = input_ids.to(device)
        
        # Generate with steering hook active
        output_ids = underlying_model.generate(
            input_ids_device,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode only the newly generated tokens (next step)
        new_tokens = output_ids[0][input_length:]
        next_step = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
    finally:
        # Always remove the hook
        hook_handle.remove()
    
    return next_step.strip()


def main(args):
    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"\n{'='*70}")
    print("STEERING EFFECT DEMO (Batch Mode, Top-K Dimensions)")
    print(f"{'='*70}")
    print(f"Top-K dimensions: {args.top_k}")
    print(f"Probe model path: {args.probe_model_path}")
    print(f"Data path: {args.data_path}")
    print(f"Layer: {args.layer}")
    print(f"Alpha (steering strength): {args.alpha}")
    print(f"Direction: {args.direction}")
    print(f"Output path: {args.output_path}")
    print(f"{'='*70}\n")

    # Load top-k dimensions from probe model
    top_dims, top_weight_diffs = get_top_k_dimensions(args.probe_model_path, args.top_k)

    # Load all samples
    positive_samples, negative_samples = load_samples(args.data_path)
    all_samples = positive_samples + negative_samples
    print(f"Total samples to process: {len(all_samples)}\n")

    # Load model (once, reused for all samples)
    print(f"Loading model from {args.model}...")
    model = LanguageModel(args.model, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✓ Model loaded successfully!\n")

    results = []

    for idx, sample in enumerate(all_samples):
        rollout_id = sample.get('rollout_id', idx)
        label = sample.get('label')
        sample_type = "benign" if label == 0 else "malicious"
        conversation = sample['input']

        print(f"\n{'='*70}")
        print(f"[{idx+1}/{len(all_samples)}] rollout_id={rollout_id}  label={label} ({sample_type})")
        print(f"{'='*70}")

        # Generate WITHOUT steering (baseline)
        print("  >> Generating WITHOUT steering...")
        output_before = generate_without_steering(
            model, tokenizer, conversation,
            max_new_tokens=args.max_new_tokens
        )
        print(f"  BEFORE: {output_before[:200]}{'...' if len(output_before) > 200 else ''}")

        # Generate WITH steering
        print(f"  >> Generating WITH steering (direction={args.direction}, alpha={args.alpha})...")
        output_after = generate_with_steering(
            model, tokenizer, conversation,
            layer=args.layer,
            top_dims=top_dims,
            alpha=args.alpha,
            direction=args.direction,
            max_new_tokens=args.max_new_tokens
        )
        print(f"  AFTER:  {output_after[:200]}{'...' if len(output_after) > 200 else ''}")

        # Collect result entry (preserve all original fields)
        entry = {k: v for k, v in sample.items()}
        entry['output_before_steering'] = output_before
        entry['output_after_steering'] = output_after
        entry['steering_config'] = {
            'direction': args.direction,
            'layer': args.layer,
            'alpha': args.alpha,
            'top_k': args.top_k,
            'top_dims': [int(d) for d in top_dims],  # convert numpy int64 -> Python int
        }
        results.append(entry)

        # Incrementally save after each sample so progress is not lost on crash
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print(f"DONE. Processed {len(results)} samples.")
    print(f"Results saved to: {args.output_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo steering effect on negative samples towards a specific positive sample")

    parser.add_argument(
        "--model",
        type=str,
        default="models/LLM-Research/gemma-2-9b-it",
        help="Path to the language model"
    )
    parser.add_argument(
        "--probe_model_path",
        type=str,
        default="models/LLM-Research/gemma-2-9b-it/Gemma/hidden_state/14/best_model.pth",
        help="Path to the probe model .pth file for extracting top-k dimensions"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top important dimensions to steer"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="MAS/Gemma/testset.json",
        help="Path to the labeled data JSON file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/Gemma/manual_steering.json",
        help="Path to save the before/after steering results JSON file"
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="negative",
        choices=["positive", "negative"],
        help="Steering direction: 'positive' (add +1, negative->positive) or 'negative' (add -1, positive->negative)"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=14,
        help="Layer to apply steering"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.9,
        help="Steering strength (1.0 = full steering)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection"
    )
    
    args = parser.parse_args()
    main(args)
