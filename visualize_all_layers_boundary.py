"""
Visualize Decision Boundary for All Layers

This script generates decision boundary visualization for each layer of the model.
It supports both hidden_state and self_attention types.

Author: Auto-generated
"""

import json
import os
import torch
import numpy as np
import matplotlib

# Disable torch._dynamo to avoid CUDA launch failures caused by accelerate device hooks
os.environ["TORCHDYNAMO_DISABLE"] = "1"

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
import torch.nn as nn
from typing import List, Tuple, Optional
import argparse
from pathlib import Path


# Fixed model name prefix for Llama datasets (local / default server path)
_LLAMA_MODEL_PREFIX = 'models_LLM-Research_Meta-Llama-3.1-8B-Instruct'
# Fixed model name prefix for Gemma-2-9B datasets
# Derived from hook_16_no_mask.py default: /home/lcwt/.cache/modelscope/hub/models/LLM-Research/gemma-2-9b-it
# with "/" replaced by "_", matching the model_name_clean convention used during training.
_GEMMA_MODEL_PREFIX = '_home_lcwt_.cache_modelscope_hub_models_LLM-Research_gemma-2-9b-it'
# Model prefix for the agentx2 server layout:
# /home/lcwt/eddy/agentx2/agentx/models/LLM-Research/Meta-Llama-3.1-8B-Instruct
_LLAMA_OS2_PREFIX = '_home_lcwt_eddy_agentx2_agentx_models_LLM-Research_Meta-Llama-3.1-8B-Instruct'
# Model prefix for Qwen2.5-7B-Instruct on agentx2 server:
# /home/lcwt/eddy/agentx2/agentx/models/Qwen/Qwen2.5-7B-Instruct
# Derived from hook_16_no_mask.py default model path with "/" replaced by "_".
_QWEN_OS2_PREFIX = '_home_lcwt_eddy_agentx2_agentx_models_Qwen_Qwen2.5-7B-Instruct'
# Model prefix for Gemma-2-9B-it on agentx2 server:
# /home/lcwt/eddy/agentx2/agentx/models/LLM-Research/gemma-2-9b-it
_GEMMA_OS2_PREFIX = '_home_lcwt_eddy_agentx2_agentx_models_LLM-Research_gemma-2-9b-it'

# Dataset configs: base_path, train/test/eval JSON paths, num_layers (mirrors hook_16.py logic)
# eval_path=None means no eval split exists for that dataset.
DATASET_CONFIGS = {
    'llama_OS': {
        'base_path':  f'{_LLAMA_MODEL_PREFIX}/llama_OS',
        'train_path': 'data/vrap2/previous_detection/trainset.json',
        'test_path':  'data/vrap2/previous_detection/testset.json',
        'eval_path':  'data/vrap2/previous_detection/evalset.json',
        'num_layers': 32,
    },
    'llama_agentharm': {
        'base_path':  f'{_LLAMA_MODEL_PREFIX}/llama_agentharm',
        'train_path': 'data/agentharm/previous_detection/train_harmful_rollout_labeled_clean.json',
        'test_path':  'data/agentharm/previous_detection/test_harmful_rollout_labeled_clean.json',
        'eval_path':  'data/agentharm/previous_detection/eval_harmful_rollout_labeled_clean.json',
        'num_layers': 32,
    },
    'gemma2_agentharm': {
        'base_path':  f'{_GEMMA_MODEL_PREFIX}/gemma2_agentharm',
        'train_path': 'data/agentharm/previous_detection_gemma/trainset.json',
        'test_path':  'data/agentharm/previous_detection_gemma/testset.json',
        'eval_path':  None,  # No eval split available for this dataset
        'num_layers': 42,    # Gemma-2-9B-it has 42 transformer layers
    },
    # agentx2 server: /home/lcwt/eddy/agentx2/agentx/
    # model  → models/LLM-Research/Meta-Llama-3.1-8B-Instruct
    # data   → data/vrap2/LLaMA/{train,test,eval}.json  (682/469/13 samples)
    'llama_OS2': {
        'base_path':  f'{_LLAMA_OS2_PREFIX}/llama_OS2',
        'train_path': 'data/vrap2/LLaMA/train.json',
        'test_path':  'data/vrap2/LLaMA/test.json',
        'eval_path':  'data/vrap2/LLaMA/eval.json',
        'num_layers': 32,
    },
    # agentx2 server: /home/lcwt/eddy/agentx2/agentx/
    # model  → models/Qwen/Qwen2.5-7B-Instruct
    # data   → data/vrap2/Qwen/{train,test}.json (mirrors hook_16_no_mask.py qwen_OS2 config)
    'qwen_OS2': {
        'base_path':  f'{_QWEN_OS2_PREFIX}/qwen_OS2',
        'train_path': 'data/vrap2/Qwen/train.json',
        'test_path':  'data/vrap2/Qwen/test.json',
        'eval_path':  None,  # No eval split available for qwen_OS2
        'num_layers': 28,    # Qwen2.5-7B-Instruct has 28 transformer layers
    },
    # agentx2 server: /home/lcwt/eddy/agentx2/agentx/
    # model  → models/LLM-Research/gemma-2-9b-it
    # data   → data/vrap2/Gemma/{trainset,testset}.json (mirrors hook_16_no_mask.py gemma2_OS config)
    'gemma2_OS': {
        'base_path':  f'{_GEMMA_OS2_PREFIX}/gemma2_OS',
        'train_path': 'data/vrap2/Gemma/trainset.json',
        'test_path':  'data/vrap2/Gemma/testset.json',
        'eval_path':  None,  # No eval split available for gemma2_OS
        'num_layers': 42,    # Gemma-2-9B-it has 42 transformer layers
    },
}


def get_train_size(dataset: str) -> Optional[int]:
    """Return the number of train samples. Returns None if the train file does not exist."""
    config = DATASET_CONFIGS[dataset]
    train_path = config.get('train_path')
    if not train_path or not os.path.exists(train_path):
        return None
    with open(train_path, 'r', encoding='utf-8') as f:
        return len(json.load(f))


def get_embeddings_file(base_path: str, state_type: str, layer_idx: int, split: str) -> str:
    """Return the path to the appropriate precomputed embeddings file for the given split."""
    filename = 'precomputed_embeddings_eval.json' if split == 'eval' else 'precomputed_embeddings.json'
    return f"embeddings/{base_path}/{state_type}/{layer_idx}/{filename}"


def precompute_all_eval_embeddings(
    dataset: str,
    base_path: str,
    state_type: str,
    layers: List[int],
    model_name: str,
    device: str,
) -> None:
    """
    Compute and cache eval embeddings for all layers that are missing a cache file.
    Loads the LLM only once and iterates over all missing layers.

    Args:
        dataset: Dataset name (key in DATASET_CONFIGS)
        base_path: Base path derived from dataset config
        state_type: Layer component type ('hidden_state', 'self_attention', 'MLP')
        layers: Layer indices to ensure are cached
        model_name: HuggingFace model name / local path
        device: Device string for tokenizer input placement (e.g. 'cuda:0')
    """
    missing_layers = [
        layer_idx for layer_idx in layers
        if not os.path.exists(get_embeddings_file(base_path, state_type, layer_idx, 'eval'))
    ]
    if not missing_layers:
        print("All eval embedding caches already exist, skipping computation.")
        return

    print(f"Computing eval embeddings for {len(missing_layers)} missing layers: {missing_layers}")

    # Load eval conversations (mirrors hook_16.py load_and_prepare_rollout_data)
    eval_path = DATASET_CONFIGS[dataset]['eval_path']
    if eval_path is None:
        print(f"Dataset '{dataset}' has no eval split configured. Skipping eval embedding precomputation.")
        return

    with open(eval_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)

    conversations, labels = [], []
    for item in eval_data:
        input_content = item.get('input', [])
        if input_content and isinstance(input_content, list):
            valid_messages = [
                {'role': msg['role'], 'content': msg['content'].strip()}
                for msg in input_content
                if isinstance(msg, dict) and 'content' in msg and 'role' in msg
            ]
            if valid_messages:
                conversations.append(valid_messages)
                labels.append(item.get('label', 0))

    print(f"Loaded {len(conversations)} eval conversations")

    # Lazy-import heavy dependencies (only needed when computing eval embeddings)
    from nnsight import LanguageModel
    from transformers import AutoTokenizer
    from tqdm import tqdm

    llm = LanguageModel(model_name, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for layer_idx in missing_layers:
        cache_path = get_embeddings_file(base_path, state_type, layer_idx, 'eval')
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        embeddings_data = {
            'embeddings': [],
            'labels': labels,
            'layer': layer_idx,
            'layer_select': state_type,
            'model_name': model_name,
            'total_samples': len(conversations),
            'split': 'eval',
        }

        for conv_idx, conversation in enumerate(tqdm(conversations, desc=f"Layer {layer_idx} eval embeddings")):
            input_ids = tokenizer.apply_chat_template(
                conversation, add_generation_prompt=True, return_tensors='pt'
            ).to(device)

            with llm.trace(input_ids):
                if state_type == 'self_attention':
                    hidden_states = llm.model.layers[layer_idx].self_attn[0].output.save()
                elif state_type == 'MLP':
                    hidden_states = llm.model.layers[layer_idx].mlp.output.save()
                else:  # hidden_state
                    hidden_states = llm.model.layers[layer_idx].output[0].save()

            hs = hidden_states.detach().cpu()
            # Normalize shape: (batch, seq_len, hidden_dim) or (seq_len, hidden_dim)
            if hs.dim() == 3:
                result = hs[:, -1, :]   # (batch, hidden_dim) — take last token
            elif hs.dim() == 2:
                result = hs[-1:, :]     # (1, hidden_dim) — take last token
            else:
                result = hs.unsqueeze(0)
            embeddings_data['embeddings'].append(result.squeeze(0).tolist())

            del hidden_states, result
            # Periodically sync and flush GPU cache to avoid async CUDA error accumulation
            if conv_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        with open(cache_path, 'w') as f:
            json.dump(embeddings_data, f)
        print(f"  Saved eval embeddings for layer {layer_idx} → {cache_path}")

    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class LogisticRegression(nn.Module):
    """Logistic regression model for binary classification."""
    
    def __init__(self, input_dim: int, num_classes: int = 2, use_bias: bool = True):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes, bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.linear(x), dim=1)


def create_decision_boundary_mesh(
    embeddings_2d: np.ndarray, 
    model: nn.Module, 
    embeddings_original: np.ndarray, 
    resolution: int = 300
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a mesh grid for decision boundary visualization.
    Maps 2D points back to high-dimensional space for prediction using KNN regression.
    
    Args:
        embeddings_2d: 2D projected embeddings (N, 2)
        model: Trained classifier model
        embeddings_original: Original high-dimensional embeddings (N, D)
        resolution: Grid resolution for mesh
        
    Returns:
        xx, yy: Mesh grid coordinates
        Z: Predicted probabilities reshaped to grid
    """
    # Create mesh grid in 2D space with padding
    x_min, x_max = embeddings_2d[:, 0].min() - 5, embeddings_2d[:, 0].max() + 5
    y_min, y_max = embeddings_2d[:, 1].min() - 5, embeddings_2d[:, 1].max() + 5
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Train KNN regressor to map 2D -> high-dim
    knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
    knn.fit(embeddings_2d, embeddings_original)
    
    # Predict high-dimensional embeddings for grid points
    grid_embeddings = knn.predict(grid_points)
    
    # Get model predictions
    model.eval()
    with torch.no_grad():
        grid_tensor = torch.FloatTensor(grid_embeddings)
        predictions = model(grid_tensor)
        Z = predictions[:, 1].numpy()  # Probability of class 1 (Malicious)
    
    Z = Z.reshape(xx.shape)
    
    return xx, yy, Z


def load_embeddings(
    embeddings_file: str,
    split: str = 'all',
    train_size: Optional[int] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load embeddings from JSON file, optionally slicing to a specific split.

    Args:
        embeddings_file: Path to embeddings JSON file
        split: Which split to return - 'all', 'train', or 'test'
        train_size: Number of leading samples that belong to the train set.
                    Required when split is 'train' or 'test'.

    Returns:
        embeddings: Embeddings array or None if file doesn't exist
        labels: Labels array or None if file doesn't exist
    """
    if not os.path.exists(embeddings_file):
        return None, None

    with open(embeddings_file, 'r') as f:
        data = json.load(f)

    embeddings = np.array(data['embeddings'])
    labels = np.array(data['labels'])

    if split != 'all' and train_size is not None:
        if split == 'train':
            embeddings = embeddings[:train_size]
            labels = labels[:train_size]
        elif split == 'test':
            embeddings = embeddings[train_size:]
            labels = labels[train_size:]

    return embeddings, labels


def load_model(model_file: str, input_dim: int) -> Optional[nn.Module]:
    """
    Load trained model from checkpoint file.
    
    Args:
        model_file: Path to model checkpoint
        input_dim: Input dimension for the model
        
    Returns:
        Loaded model or None if file doesn't exist
    """
    if not os.path.exists(model_file):
        return None
    
    model = LogisticRegression(input_dim, num_classes=2, use_bias=True)
    
    checkpoint = torch.load(model_file, map_location='cpu')
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'linear.weight' in checkpoint:
            model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def _visualize_layer_boundary_debug(layer_idx: int, state_type: str, output_dir: str) -> bool:
    """Generate an empty layout figure for a single layer without loading any data or model."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))

    plot_configs = [
        (axes[0, 0], 't-SNE with Decision Boundary\n(Black dashed line = decision boundary)',
         't-SNE Dim 1', 't-SNE Dim 2'),
        (axes[0, 1], 'PCA with Decision Boundary\n(Variance explained: N/A)',
         'PC1 (N/A)', 'PC2 (N/A)'),
        (axes[1, 0], 'Model Confidence (t-SNE)\nAccuracy: N/A',
         't-SNE Dim 1', 't-SNE Dim 2'),
        (axes[1, 1], 'Label 2 (Test Data) Analysis\n[Layout Debug Mode]',
         't-SNE Dim 1', 't-SNE Dim 2'),
    ]

    for ax, title, xlabel, ylabel in plot_configs:
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.text(0.5, 0.5, '[Layout Debug — No Data]', transform=ax.transAxes,
                ha='center', va='center', fontsize=14, color='gray',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle(
        f'Decision Boundary Visualization - Layer {layer_idx}\n'
        f'({state_type}, LogisticRegression) [LAYOUT DEBUG]',
        fontsize=16, fontweight='bold', y=0.995,
    )
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"layer_{layer_idx:02d}_decision_boundary.pdf")
    plt.savefig(output_file, bbox_inches='tight')
    plt.close(fig)

    print(f"  [Layout Debug] Saved: {output_file}")
    return True


def visualize_layer_boundary(
    layer_idx: int,
    state_type: str,
    base_path: str,
    output_dir: str,
    resolution: int = 300,
    perplexity: int = 30,
    split: str = 'all',
    train_size: Optional[int] = None,
    layout_debug: bool = False,
) -> bool:
    """
    Generate decision boundary visualization for a specific layer.
    
    Args:
        layer_idx: Layer index (0-31)
        state_type: Type of state ('hidden_state' or 'self_attention')
        base_path: Base path for data (derived from dataset name)
        output_dir: Output directory for visualizations
        resolution: Grid resolution for decision boundary
        perplexity: t-SNE perplexity parameter
        split: Which split to visualize ('all', 'train', 'test')
        train_size: Number of leading samples belonging to train set
        
    Returns:
        True if visualization was successful, False otherwise
    """
    # Construct file paths
    embeddings_file = get_embeddings_file(base_path, state_type, layer_idx, split)
    model_file = f"models/{base_path}/{state_type}/{layer_idx}/best_model.pth"

    print(f"\n{'='*70}")
    print(f"Processing Layer {layer_idx} ({state_type})")
    print(f"{'='*70}")

    if layout_debug:
        return _visualize_layer_boundary_debug(layer_idx, state_type, output_dir)

    # Load embeddings based on split mode.
    # n_train_boundary: end-of-train index (separator between train and test); None if unknown
    # n_eval_boundary:  end-of-test index  (separator between test  and eval); None if no eval
    print(f"Loading embeddings (split={split})")
    n_train_boundary: Optional[int] = None
    n_eval_boundary:  Optional[int] = None

    if split == 'eval':
        # Two-set layout: test (circles) + eval (triangles)
        main_emb_file = get_embeddings_file(base_path, state_type, layer_idx, 'all')
        test_emb, test_labels = load_embeddings(main_emb_file, split='test', train_size=train_size)
        eval_emb, eval_labels = load_embeddings(embeddings_file)
        if test_emb is None or eval_emb is None:
            print(f"⚠ Missing embeddings for eval split (test={main_emb_file}, eval={embeddings_file})")
            return False
        n_eval_boundary = len(test_labels)
        embeddings = np.vstack([test_emb, eval_emb])
        labels = np.concatenate([test_labels, eval_labels])
        print(f"  Test embeddings: {test_emb.shape}  Eval embeddings: {eval_emb.shape}")

    elif split == 'all':
        # Three-set layout: train (circles) + test (triangles) + eval (diamonds, optional)
        main_emb_file = get_embeddings_file(base_path, state_type, layer_idx, 'all')
        eval_emb_file = get_embeddings_file(base_path, state_type, layer_idx, 'eval')

        if train_size is not None:
            train_emb, train_labels = load_embeddings(main_emb_file, split='train', train_size=train_size)
            test_emb,  test_labels  = load_embeddings(main_emb_file, split='test',  train_size=train_size)
        else:
            train_emb, train_labels = None, None
            test_emb,  test_labels  = load_embeddings(main_emb_file)

        eval_emb, eval_labels = load_embeddings(eval_emb_file)

        if train_emb is None and test_emb is None:
            print(f"⚠ Embeddings file not found: {main_emb_file}")
            return False

        if train_emb is not None:
            # Full three-set (or two-set if eval missing)
            n_train_boundary = len(train_labels)
            n_eval_boundary  = (len(train_labels) + len(test_labels)) if eval_emb is not None else None
            parts_emb = [train_emb, test_emb] + ([eval_emb] if eval_emb is not None else [])
            parts_lbl = [train_labels, test_labels] + ([eval_labels] if eval_emb is not None else [])
        else:
            # train_size unknown: combined train+test as one group, plus optional eval
            n_eval_boundary = len(test_labels) if eval_emb is not None else None
            parts_emb = [test_emb] + ([eval_emb] if eval_emb is not None else [])
            parts_lbl = [test_labels] + ([eval_labels] if eval_emb is not None else [])

        embeddings = np.vstack(parts_emb)
        labels     = np.concatenate(parts_lbl)
        print(f"  All-sets shape: {embeddings.shape} "
              f"(n_train_boundary={n_train_boundary}, n_eval_boundary={n_eval_boundary})")

    else:
        embeddings, labels = load_embeddings(embeddings_file, split=split, train_size=train_size)

    if embeddings is None:
        print(f"⚠ Embeddings file not found: {embeddings_file}")
        return False

    print(f"  Combined embeddings shape: {embeddings.shape}")
    print(f"  Label distribution: 0={np.sum(labels==0)}, 1={np.sum(labels==1)}")
    
    # Load model (failure is non-fatal: degrade to data-distribution-only view)
    print(f"Loading model from: {model_file}")
    model     = load_model(model_file, embeddings.shape[1])
    has_model = model is not None
    if has_model:
        print(f"  Model loaded successfully")
    else:
        print(f"⚠ Model not found: {model_file} — showing data distribution only")

    # Get predictions (only when model is available)
    predictions = probabilities = accuracy = None
    mask_01 = (labels == 0) | (labels == 1)
    if has_model:
        with torch.no_grad():
            embeddings_tensor = torch.FloatTensor(embeddings)
            logits            = model(embeddings_tensor)
            predictions       = torch.argmax(logits, dim=1).numpy()
            probabilities     = logits.numpy()
        accuracy = np.mean(predictions[mask_01] == labels[mask_01])
        print(f"  Model accuracy on Label 0/1: {accuracy:.2%}")
    
    # Perform dimensionality reduction
    print("  Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000, verbose=0)
    embeddings_tsne = tsne.fit_transform(embeddings)
    
    print("  Computing PCA...")
    pca = PCA(n_components=2, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    
    colors = ['blue', 'red']
    label_names = ['Label 0 (Benign)', 'Label 1 (Malicious)']

    def _scatter_points(ax, emb_2d, labels_arr, n_train_bnd=None, n_eval_bnd=None):
        """
        Plot scatter with marker shape per dataset split and color per label.

        Dataset → marker mapping:
          n_train=None, n_eval=None → all 'o'  (single-set)
          n_train=None, n_eval=K   → [0:K] 'o' test,  [K:] '^' eval
          n_train=M,    n_eval=None→ [0:M] 'o' train, [M:] '^' test
          n_train=M,    n_eval=K   → [0:M] 'o' train, [M:K] '^' test, [K:] 'D' eval

        Colors: label 0 = blue, label 1 = red
        """
        n   = len(labels_arr)
        idx = np.arange(n)

        if n_train_bnd is None and n_eval_bnd is None:
            groups = [('o', np.ones(n, dtype=bool), '',      50, 0.70, 5)]
        elif n_train_bnd is None:
            groups = [
                ('o', idx < n_eval_bnd,  'Test', 50, 0.70, 5),
                ('^', idx >= n_eval_bnd, 'Eval', 80, 0.85, 6),
            ]
        elif n_eval_bnd is None:
            groups = [
                ('o', idx < n_train_bnd,  'Train', 50, 0.70, 5),
                ('^', idx >= n_train_bnd, 'Test',  80, 0.85, 6),
            ]
        else:
            groups = [
                ('o', idx < n_train_bnd,                              'Train', 50, 0.70, 5),
                ('^', (idx >= n_train_bnd) & (idx < n_eval_bnd),     'Test',  80, 0.85, 6),
                ('D', idx >= n_eval_bnd,                              'Eval',  90, 0.90, 7),
            ]

        for marker, seg_mask, seg_name, sz, alp, zo in groups:
            if not np.any(seg_mask):
                continue
            for label_idx, color in enumerate(colors):
                mask = seg_mask & (labels_arr == label_idx)
                if not np.any(mask):
                    continue
                prefix = f'{seg_name} ' if seg_name else ''
                ax.scatter(
                    emb_2d[mask, 0], emb_2d[mask, 1],
                    c=color,
                    label=f'{prefix}{label_names[label_idx]} ({np.sum(mask)})',
                    alpha=alp, s=sz,
                    edgecolors='black', linewidth=0.8,
                    marker=marker, zorder=zo,
                )

    # ========== Plot 1: t-SNE (+ Decision Boundary when model available) ==========
    print(f"  Creating t-SNE {'decision boundary' if has_model else 'distribution'}...")
    ax1 = axes[0, 0]

    if has_model:
        xx_tsne, yy_tsne, Z_tsne = create_decision_boundary_mesh(
            embeddings_tsne, model, embeddings, resolution=resolution
        )
        contour = ax1.contourf(xx_tsne, yy_tsne, Z_tsne, levels=20, cmap='RdBu', alpha=0.3)
        ax1.contour(xx_tsne, yy_tsne, Z_tsne, levels=[0.5], colors='black', linewidths=3, linestyles='--')

    if not has_model:
        ax1.set_facecolor('#e8e8e8')
    _scatter_points(ax1, embeddings_tsne, labels, n_train_boundary, n_eval_boundary)
    ax1.set_title(
        't-SNE with Decision Boundary\n(Black dashed line = decision boundary)' if has_model
        else 't-SNE Data Distribution',
        fontsize=13, fontweight='bold')
    ax1.set_xlabel('t-SNE Dim 1', fontsize=11)
    ax1.set_ylabel('t-SNE Dim 2', fontsize=11)
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)
    if has_model:
        plt.colorbar(contour, ax=ax1, label='P(Malicious)')

    # ========== Plot 2: PCA (+ Decision Boundary when model available) ==========
    print(f"  Creating PCA {'decision boundary' if has_model else 'distribution'}...")
    ax2 = axes[0, 1]

    if has_model:
        xx_pca, yy_pca, Z_pca = create_decision_boundary_mesh(
            embeddings_pca, model, embeddings, resolution=resolution
        )
        contour2 = ax2.contourf(xx_pca, yy_pca, Z_pca, levels=20, cmap='RdBu', alpha=0.3)
        ax2.contour(xx_pca, yy_pca, Z_pca, levels=[0.5], colors='black', linewidths=3, linestyles='--')

    if not has_model:
        ax2.set_facecolor('#e8e8e8')
    _scatter_points(ax2, embeddings_pca, labels, n_train_boundary, n_eval_boundary)
    ax2.set_title(
        f'PCA with Decision Boundary\n(Variance explained: {pca.explained_variance_ratio_.sum():.2%})' if has_model
        else f'PCA Data Distribution\n(Variance explained: {pca.explained_variance_ratio_.sum():.2%})',
        fontsize=13, fontweight='bold')
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=11)
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=11)
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3)
    if has_model:
        plt.colorbar(contour2, ax=ax2, label='P(Malicious)')

    # ========== Plot 3: Prediction Confidence (t-SNE) — or distribution if no model ==========
    ax3 = axes[1, 0]
    idx_arr = np.arange(len(labels))

    if has_model:
        confidence = np.max(probabilities, axis=1)
        conf_vmin, conf_vmax = confidence.min(), confidence.max()

        if n_train_boundary is None and n_eval_boundary is None:
            conf_groups = [('o', np.ones(len(labels), dtype=bool), 'All',   50, 0.70, 5)]
        elif n_train_boundary is None:
            conf_groups = [
                ('o', idx_arr < n_eval_boundary,  'Test', 50, 0.70, 5),
                ('^', idx_arr >= n_eval_boundary, 'Eval', 80, 0.85, 6),
            ]
        elif n_eval_boundary is None:
            conf_groups = [
                ('o', idx_arr < n_train_boundary,  'Train', 50, 0.70, 5),
                ('^', idx_arr >= n_train_boundary, 'Test',  80, 0.85, 6),
            ]
        else:
            conf_groups = [
                ('o', idx_arr < n_train_boundary,                                  'Train', 50, 0.70, 5),
                ('^', (idx_arr >= n_train_boundary) & (idx_arr < n_eval_boundary), 'Test',  80, 0.85, 6),
                ('D', idx_arr >= n_eval_boundary,                                  'Eval',  90, 0.90, 7),
            ]

        scatter3 = None
        for marker, seg_mask, seg_name, sz, alp, zo in conf_groups:
            if not np.any(seg_mask):
                continue
            sc = ax3.scatter(
                embeddings_tsne[seg_mask, 0], embeddings_tsne[seg_mask, 1],
                c=confidence[seg_mask], cmap='viridis',
                alpha=alp, s=sz, edgecolors='black', linewidth=0.5,
                marker=marker, label=seg_name, zorder=zo,
                vmin=conf_vmin, vmax=conf_vmax,
            )
            if scatter3 is None:
                scatter3 = sc
        if len(conf_groups) > 1:
            ax3.legend(fontsize=9)

        # Highlight misclassified points (label 0/1 only)
        correct = predictions[mask_01] == labels[mask_01]
        incorrect_indices = np.where(mask_01)[0][~correct]
        if len(incorrect_indices) > 0:
            ax3.scatter(embeddings_tsne[incorrect_indices, 0],
                        embeddings_tsne[incorrect_indices, 1],
                        c='red', s=200, alpha=0.5, marker='x', linewidths=3,
                        label=f'Misclassified ({len(incorrect_indices)})', zorder=10)
            ax3.legend(fontsize=9)

        ax3.set_title(f'Model Confidence (t-SNE)\nAccuracy: {accuracy:.2%}',
                      fontsize=13, fontweight='bold')
        plt.colorbar(scatter3, ax=ax3, label='Prediction Confidence')
    else:
        # No model: show data distribution as a duplicate of Plot 1
        ax3.set_facecolor('#e8e8e8')
        _scatter_points(ax3, embeddings_tsne, labels, n_train_boundary, n_eval_boundary)
        ax3.set_title('Data Distribution (t-SNE)',
                      fontsize=13, fontweight='bold')
        ax3.legend(fontsize=9, loc='best')

    ax3.set_xlabel('t-SNE Dim 1', fontsize=11)
    ax3.set_ylabel('t-SNE Dim 2', fontsize=11)
    ax3.grid(True, alpha=0.3)

    # ========== Plot 4: Eval Set Analysis ==========
    ax4 = axes[1, 1]
    ax4.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1],
                c='lightgray', alpha=0.3, s=30, zorder=1)

    if n_eval_boundary is not None:
        eval_mask_4 = idx_arr >= n_eval_boundary
        eval_emb_2d = embeddings_tsne[eval_mask_4]
        eval_true   = labels[eval_mask_4]

        if has_model:
            eval_preds = predictions[eval_mask_4]
            eval_probs = probabilities[eval_mask_4]

            for label_idx, color in enumerate(colors):
                true_mask = eval_true == label_idx
                if not np.any(true_mask):
                    continue
                correct_e = eval_preds[true_mask] == label_idx
                if np.any(correct_e):
                    ax4.scatter(eval_emb_2d[true_mask][correct_e, 0],
                                eval_emb_2d[true_mask][correct_e, 1],
                                c=color, s=150, alpha=0.85,
                                edgecolors='black', linewidth=1.5, marker='D',
                                label=f'Eval {label_names[label_idx]} ✓ ({correct_e.sum()})',
                                zorder=8)
                if np.any(~correct_e):
                    ax4.scatter(eval_emb_2d[true_mask][~correct_e, 0],
                                eval_emb_2d[true_mask][~correct_e, 1],
                                c=color, s=200, alpha=0.7,
                                edgecolors='yellow', linewidth=2.5, marker='X',
                                label=f'Eval {label_names[label_idx]} ✗ ({(~correct_e).sum()})',
                                zorder=9)

            avg_conf_e = np.mean(np.max(eval_probs, axis=1))
            eval_acc   = np.mean(eval_preds == eval_true) if len(eval_true) > 0 else float('nan')
            ax4.set_title(f'Eval Set Analysis\nEval Acc: {eval_acc:.2%}, Avg Conf: {avg_conf_e:.2%}',
                          fontsize=13, fontweight='bold')
        else:
            # No model: show eval distribution by true label only
            for label_idx, color in enumerate(colors):
                mask = eval_true == label_idx
                if np.any(mask):
                    ax4.scatter(eval_emb_2d[mask, 0], eval_emb_2d[mask, 1],
                                c=color, s=150, alpha=0.85,
                                edgecolors='black', linewidth=1.5, marker='D',
                                label=f'Eval {label_names[label_idx]} ({mask.sum()})',
                                zorder=8)
            ax4.set_title(f'Eval Set Distribution\n[No model — {len(eval_true)} samples]',
                          fontsize=13, fontweight='bold')
    else:
        ax4.set_title('Eval Set Analysis\nNo eval data in current split',
                      fontsize=13, fontweight='bold')

    ax4.set_xlabel('t-SNE Dim 1', fontsize=11)
    ax4.set_ylabel('t-SNE Dim 2', fontsize=11)
    ax4.legend(fontsize=9, loc='best')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(
        f'Decision Boundary Visualization - Layer {layer_idx}\n({state_type}, LogisticRegression)'
        if has_model else
        f'Data Distribution - Layer {layer_idx}\n({state_type}) [No model]',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"layer_{layer_idx:02d}_decision_boundary.pdf")
    plt.savefig(output_file, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  ✓ Saved: {output_file}")
    
    return True


def visualize_all_layers(
    base_path: str,
    state_type: str,
    layers: List[int],
    output_dir: str,
    resolution: int = 300,
    perplexity: int = 30,
    split: str = 'all',
    train_size: Optional[int] = None,
    layout_debug: bool = False,
) -> None:
    """
    Generate decision boundary visualizations for multiple layers.
    
    Args:
        base_path: Base path for data
        state_type: Type of state ('hidden_state' or 'self_attention')
        layers: List of layer indices to process
        output_dir: Output directory for visualizations
        resolution: Grid resolution for decision boundary
        perplexity: t-SNE perplexity parameter
        split: Which split to visualize ('all', 'train', 'test')
        train_size: Number of leading samples belonging to train set
    """
    print("\n" + "="*70)
    print("All Layers Decision Boundary Visualization")
    print("="*70)
    print(f"Base path: {base_path}")
    print(f"State type: {state_type}")
    print(f"Layers to process: {layers}")
    print(f"Output directory: {output_dir}")
    print(f"Resolution: {resolution}")
    print(f"Perplexity: {perplexity}")
    print(f"Split: {split} (train_size={train_size})")
    
    successful_layers = []
    failed_layers = []
    
    for layer_idx in layers:
        try:
            success = visualize_layer_boundary(
                layer_idx=layer_idx,
                state_type=state_type,
                base_path=base_path,
                output_dir=output_dir,
                resolution=resolution,
                perplexity=perplexity,
                split=split,
                train_size=train_size,
                layout_debug=layout_debug,
            )
            if success:
                successful_layers.append(layer_idx)
            else:
                failed_layers.append(layer_idx)
        except Exception as e:
            print(f"  ✗ Error processing layer {layer_idx}: {e}")
            failed_layers.append(layer_idx)
    
    # Print summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"Successfully processed: {len(successful_layers)} layers")
    print(f"  Layers: {successful_layers}")
    print(f"Failed/skipped: {len(failed_layers)} layers")
    if failed_layers:
        print(f"  Layers: {failed_layers}")
    print(f"Output directory: {output_dir}")


def _create_summary_grid_debug(
    state_type: str,
    layers: List[int],
    output_dir: str,
    n_cols: int = 8,
) -> None:
    """Generate an empty summary grid layout for debugging, without loading any data or model."""
    n_layers = len(layers)
    n_rows = (n_layers + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(32, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()

    for idx, layer_idx in enumerate(layers):
        ax = axes[idx]
        ax.set_title(f'Layer {layer_idx}', fontsize=18, fontweight='bold')
        ax.text(0.5, 0.5, 'debug', transform=ax.transAxes,
                ha='center', va='center', fontsize=8, color='gray')
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(len(layers), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"all_layers_summary_{state_type}.pdf")
    plt.savefig(output_file, bbox_inches='tight')
    plt.close(fig)

    print(f"\n[Layout Debug] Summary grid saved to: {output_file}")


def create_summary_grid(
    base_path: str,
    state_type: str,
    layers: List[int],
    output_dir: str,
    resolution: int = 200,
    perplexity: int = 30,
    split: str = 'all',
    train_size: Optional[int] = None,
    layout_debug: bool = False,
) -> None:
    """
    Create a summary grid showing decision boundaries for all layers.
    
    Args:
        base_path: Base path for data
        state_type: Type of state ('hidden_state' or 'self_attention')
        layers: List of layer indices to process
        output_dir: Output directory for visualizations
        resolution: Grid resolution for decision boundary
        perplexity: t-SNE perplexity parameter
        split: Which split to visualize ('all', 'train', 'test')
        train_size: Number of leading samples belonging to train set
    """
    print("\n" + "="*70)
    print("Creating Summary Grid")
    print("="*70)

    if layout_debug:
        _create_summary_grid_debug(state_type, layers, output_dir)
        return

    # Calculate grid dimensions
    n_layers = len(layers)
    n_cols = 8
    n_rows = (n_layers + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(32, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()
    
    colors = ['blue', 'red', 'green']
    
    for idx, layer_idx in enumerate(layers):
        ax = axes[idx]
        
        embeddings_file = get_embeddings_file(base_path, state_type, layer_idx, split)
        model_file = f"models/{base_path}/{state_type}/{layer_idx}/best_model.pth"

        # Determine n_train_bnd / n_eval_bnd for this layer (mirrors visualize_layer_boundary logic)
        s_n_train_bnd: Optional[int] = None
        s_n_eval_bnd:  Optional[int] = None

        if split == 'eval':
            main_emb_file = get_embeddings_file(base_path, state_type, layer_idx, 'all')
            test_emb, test_labels = load_embeddings(main_emb_file, split='test', train_size=train_size)
            eval_emb, eval_labels = load_embeddings(embeddings_file)
            if test_emb is not None and eval_emb is not None:
                s_n_eval_bnd = len(test_labels)
                embeddings = np.vstack([test_emb, eval_emb])
                labels = np.concatenate([test_labels, eval_labels])
            else:
                embeddings, labels = None, None

        elif split == 'all':
            main_emb_file = get_embeddings_file(base_path, state_type, layer_idx, 'all')
            eval_emb_file = get_embeddings_file(base_path, state_type, layer_idx, 'eval')
            if train_size is not None:
                tr_emb, tr_lbl = load_embeddings(main_emb_file, split='train', train_size=train_size)
                te_emb, te_lbl = load_embeddings(main_emb_file, split='test',  train_size=train_size)
            else:
                tr_emb, tr_lbl = None, None
                te_emb, te_lbl = load_embeddings(main_emb_file)
            ev_emb, ev_lbl = load_embeddings(eval_emb_file)

            if tr_emb is not None:
                s_n_train_bnd = len(tr_lbl)
                s_n_eval_bnd  = (len(tr_lbl) + len(te_lbl)) if ev_emb is not None else None
                parts_e = [tr_emb, te_emb] + ([ev_emb] if ev_emb is not None else [])
                parts_l = [tr_lbl, te_lbl] + ([ev_lbl] if ev_emb is not None else [])
            elif te_emb is not None:
                s_n_eval_bnd = len(te_lbl) if ev_emb is not None else None
                parts_e = [te_emb] + ([ev_emb] if ev_emb is not None else [])
                parts_l = [te_lbl] + ([ev_lbl] if ev_emb is not None else [])
            else:
                parts_e, parts_l = [], []

            if parts_e:
                embeddings = np.vstack(parts_e)
                labels     = np.concatenate(parts_l)
            else:
                embeddings, labels = None, None

        else:
            embeddings, labels = load_embeddings(embeddings_file, split=split, train_size=train_size)

        if embeddings is None:
            ax.set_title(f'Layer {layer_idx}\n(No data)', fontsize=10)
            ax.axis('off')
            continue

        # Build scatter groups (needed regardless of whether model is available)
        sg_idx = np.arange(len(labels))
        if s_n_train_bnd is None and s_n_eval_bnd is None:
            sg_groups = [('o', np.ones(len(labels), dtype=bool), 20, 0.60)]
        elif s_n_train_bnd is None:
            sg_groups = [
                ('o', sg_idx < s_n_eval_bnd,  20, 0.60),
                ('^', sg_idx >= s_n_eval_bnd, 30, 0.80),
            ]

        # s_colors is always needed (defined here to avoid NameError when model is missing)
        s_colors = ['blue', 'red']

        model = load_model(model_file, embeddings.shape[1])

        if model is None:
            # No model: show t-SNE data distribution with consistent gray background
            tsne_nd = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                           max_iter=500, verbose=0)
            emb_tsne_nd = tsne_nd.fit_transform(embeddings)
            # Consistent neutral background matching the probe-present panels
            ax.set_facecolor('#e8e8e8')
            for marker, seg_mask, sz, alp in sg_groups:
                for label_idx, color in enumerate(s_colors):
                    m = seg_mask & (labels == label_idx)
                    if np.any(m):
                        ax.scatter(emb_tsne_nd[m, 0], emb_tsne_nd[m, 1],
                                   c=color, alpha=alp, s=sz, edgecolors='black',
                                   linewidth=0.3, marker=marker)
            ax.set_title(f'Layer {layer_idx}', fontsize=18, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Get predictions
        with torch.no_grad():
            embeddings_tensor = torch.FloatTensor(embeddings)
            logits = model(embeddings_tensor)
            predictions = torch.argmax(logits, dim=1).numpy()

        # Calculate accuracy on label 0/1 only
        mask_01  = (labels == 0) | (labels == 1)
        accuracy = np.mean(predictions[mask_01] == labels[mask_01])

        # Perform t-SNE
        print(f"  Processing layer {layer_idx}...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=500, verbose=0)
        embeddings_tsne = tsne.fit_transform(embeddings)

        # Create decision boundary
        xx, yy, Z = create_decision_boundary_mesh(
            embeddings_tsne, model, embeddings, resolution=resolution
        )

        # Plot background decision boundary
        ax.contourf(xx, yy, Z, levels=15, cmap='RdBu', alpha=0.3)
        ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')

        # Complete the remaining sg_groups cases (first two already set above)
        if s_n_train_bnd is not None and s_n_eval_bnd is None:
            sg_groups = [
                ('o', sg_idx < s_n_train_bnd,  20, 0.60),
                ('^', sg_idx >= s_n_train_bnd, 30, 0.80),
            ]
        elif s_n_train_bnd is not None and s_n_eval_bnd is not None:
            sg_groups = [
                ('o', sg_idx < s_n_train_bnd,                                 20, 0.60),
                ('^', (sg_idx >= s_n_train_bnd) & (sg_idx < s_n_eval_bnd),   30, 0.80),
                ('D', sg_idx >= s_n_eval_bnd,                                  35, 0.90),
            ]

        s_colors = ['blue', 'red']
        for marker, seg_mask, sz, alp in sg_groups:
            for label_idx, color in enumerate(s_colors):
                m = seg_mask & (labels == label_idx)
                if np.any(m):
                    ax.scatter(embeddings_tsne[m, 0], embeddings_tsne[m, 1],
                               c=color, alpha=alp, s=sz, edgecolors='black',
                               linewidth=0.3, marker=marker)
        
        ax.set_title(f'Layer {layer_idx}', fontsize=18, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide unused axes
    for idx in range(len(layers), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"all_layers_summary_{state_type}.pdf")
    plt.savefig(output_file, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n✓ Summary grid saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate decision boundary visualizations for all layers'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=list(DATASET_CONFIGS.keys()),
        help='Dataset name - determines base path, train/test split boundary, and layer count. '
             f'Choices: {list(DATASET_CONFIGS.keys())}'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='all',
        choices=['all', 'train', 'test', 'eval'],
        help='Which subset of embeddings to visualize (default: all). '
             '"eval" uses a separate evalset; embeddings are computed and cached on first run.'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='/home/lcwt/eddy/agentx2/agentx/models/LLM-Research/gemma-2-9b-it',
        help='LLM model name / local path (only used when --split eval cache is missing)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device for LLM inference when computing eval embeddings (default: cuda:0)'
    )
    parser.add_argument(
        '--state-type', 
        type=str, 
        default='hidden_state',
        choices=['hidden_state', 'self_attention'],
        help='Type of state to visualize'
    )
    parser.add_argument(
        '--layers', 
        type=str, 
        default='all',
        help='Layers to process (e.g., "0,1,2,3" or "0-15" or "all")'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='visualizations',
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--resolution', 
        type=int, 
        default=300,
        help='Grid resolution for decision boundary'
    )
    parser.add_argument(
        '--perplexity', 
        type=int, 
        default=30,
        help='t-SNE perplexity parameter'
    )
    parser.add_argument(
        '--summary-only', 
        action='store_true',
        help='Only generate summary grid'
    )
    parser.add_argument(
        '--no-summary', 
        action='store_true',
        help='Skip summary grid generation'
    )
    parser.add_argument(
        '--layout-debug',
        action='store_true',
        help='Debug mode: skip model/data loading and generate blank layout figures only'
    )
    parser.add_argument(
        '--num-layers',
        type=int,
        default=None,
        help='[Layout debug only] Override total layer count (e.g. 28 or 32). '
             'When set, layers is reset to range(num_layers), ignoring --layers.'
    )

    args = parser.parse_args()

    # Resolve base_path and train_size from dataset config
    config = DATASET_CONFIGS[args.dataset]
    base_path  = config['base_path']
    # Always compute train_size so all/train/test splits can distinguish train from test
    train_size = get_train_size(args.dataset)
    print(f"Dataset: {args.dataset}  base_path={base_path}  split={args.split}  train_size={train_size}")

    # Validate that eval split is supported for this dataset
    if args.split == 'eval' and config['eval_path'] is None:
        raise ValueError(
            f"Dataset '{args.dataset}' has no eval split (eval_path=None). "
            f"Use --split all/train/test instead."
        )

    # Parse layers argument; use dataset-specific layer count when 'all' is requested
    num_layers = config.get('num_layers', 32)
    if args.layers == 'all':
        layers = list(range(num_layers))
    elif '-' in args.layers:
        start, end = map(int, args.layers.split('-'))
        layers = list(range(start, end + 1))
    else:
        layers = [int(x) for x in args.layers.split(',')]

    # In layout-debug mode, --num-layers overrides the layer list to simulate different model sizes
    if args.layout_debug and args.num_layers is not None:
        layers = list(range(args.num_layers))
        print(f"[Layout Debug] Overriding layers to range({args.num_layers})")

    # Ensure eval embedding caches exist for split='eval' or split='all' (when eval_path exists).
    # Skip in layout-debug mode — no data or model loading needed there.
    if args.split in ('eval', 'all') and config['eval_path'] is not None and not args.layout_debug:
        precompute_all_eval_embeddings(
            dataset=args.dataset,
            base_path=base_path,
            state_type=args.state_type,
            layers=layers,
            model_name=args.model,
            device=args.device,
        )

    # Create output directory path (include split suffix when not 'all')
    split_suffix = f'_{args.split}' if args.split != 'all' else ''
    output_dir = os.path.join(args.output_dir, base_path, args.state_type + split_suffix)

    if args.summary_only:
        create_summary_grid(
            base_path=base_path,
            state_type=args.state_type,
            layers=layers,
            output_dir=output_dir,
            resolution=min(args.resolution, 200),
            perplexity=args.perplexity,
            split=args.split,
            train_size=train_size,
            layout_debug=args.layout_debug,
        )
    else:
        visualize_all_layers(
            base_path=base_path,
            state_type=args.state_type,
            layers=layers,
            output_dir=output_dir,
            resolution=args.resolution,
            perplexity=args.perplexity,
            split=args.split,
            train_size=train_size,
            layout_debug=args.layout_debug,
        )

        if not args.no_summary:
            create_summary_grid(
                base_path=base_path,
                state_type=args.state_type,
                layers=layers,
                output_dir=output_dir,
                resolution=min(args.resolution, 200),
                perplexity=args.perplexity,
                split=args.split,
                train_size=train_size,
                layout_debug=args.layout_debug,
            )

    print("\n✓ All visualizations completed successfully!")


if __name__ == "__main__":
    main()
