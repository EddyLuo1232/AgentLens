"""Train linear probes on MAS train/test splits."""

import argparse
import json
import os
import sys

MAS_DATASETS = {
    "Gemma": {
        "dir": os.path.join("MAS", "Gemma"),
        "train": "trainset.json",
        "test": "testset.json",
    },
    "LLaMA": {
        "dir": os.path.join("MAS", "LLaMA"),
        "train": "train.json",
        "test": "test.json",
    },
    "Qwen": {
        "dir": os.path.join("MAS", "Qwen"),
        "train": "train.json",
        "test": "test.json",
    },
}


def normalize_dataset_name(dataset: str) -> str:
    for canonical_name in MAS_DATASETS:
        if dataset.lower() == canonical_name.lower():
            return canonical_name
    supported = ", ".join(MAS_DATASETS)
    raise ValueError(f"Unknown dataset: {dataset}. Supported MAS datasets: {supported}")


def build_model_output_dir(model_path: str) -> str:
    normalized = os.path.normpath(model_path)
    if os.path.isabs(normalized):
        return os.path.join("models", os.path.basename(normalized))
    if normalized == "models" or normalized.startswith(f"models{os.sep}"):
        return normalized
    return os.path.join("models", normalized.replace("/", "_").replace(os.sep, "_"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--layer", type=str, default="hidden_state",
                        choices=["self_attention", "MLP", "hidden_state"])
    parser.add_argument("--dataset", type=str, default="Qwen",
                        help="MAS dataset to use: Gemma, LLaMA, or Qwen")
    parser.add_argument("--epoch", type=int, default=100,
                        help="Number of training epochs for each layer probe")
    parser.add_argument("--learning_rate", "--lr", dest="learning_rate", type=float, default=1e-2,
                        help="Learning rate for the Adam optimizer")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--force_recompute", action="store_true",
                        help="Force recompute embeddings even if precomputed file exists")
    parser.add_argument("--max_layers", type=int, default=28,
                        help="Maximum number of layers to process")
    return parser


if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    build_parser().parse_args()
    raise SystemExit

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import pandas as pd
    import numpy as np
    from torch.utils.data import DataLoader, Dataset, Subset
    from tqdm import tqdm
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ModuleNotFoundError as exc:
    print(
        f"Missing runtime dependency: {exc.name}. "
        "Run train_probe.py in the project ML environment or install the required "
        "packages: torch, pandas, numpy, tqdm, transformers.",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


@torch.no_grad()
def logistic_regression_eval(args,model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    labels = []
    #for batch_X, batch_y in tqdm(dataloader, desc="Processing Training Batches"):
    for hidden_state, label in tqdm(dataloader, desc="Processing Test Batches"):
        hidden_state, label = hidden_state.to(device), label.to(device)
        output = model(hidden_state)
        predicted = torch.argmax(output, dim=1)  # 适用于多分类问题
        correct += (predicted == label).sum().item()
        total += label.size(0)

        predictions.extend(output.cpu().tolist())
        labels.extend(label.cpu().tolist())
        if args.debug:
            break
    ACC = correct / total

    return {"ACC": ACC, "AUC": 0, "AUPRC": 0}


def load_and_prepare_rollout_data(dataset="Qwen"):
    """Load one MAS dataset with its fixed train/test split."""
    dataset = normalize_dataset_name(dataset)
    dataset_config = MAS_DATASETS[dataset]
    data_dir = dataset_config["dir"]
    train_path = os.path.join(data_dir, dataset_config["train"])
    test_path = os.path.join(data_dir, dataset_config["test"])

    print(f"Loading MAS dataset '{dataset}' from: {data_dir}")
    with open(train_path, 'r', encoding='utf-8') as file:
        train_data = json.load(file)
    with open(test_path, 'r', encoding='utf-8') as file:
        test_data = json.load(file)

    raw_train_size = len(train_data)

    print(f"Loaded {len(train_data)} train samples from {train_path}")
    print(f"Loaded {len(test_data)} test samples from {test_path}")

    def extract_split(items):
        conversations = []
        labels = []
        for item in items:
            input_content = item.get('input', [])
            if input_content and isinstance(input_content, list):
                valid_messages = []
                for msg in input_content:
                    if isinstance(msg, dict) and 'content' in msg and 'role' in msg:
                        valid_messages.append({
                            'role': msg['role'],
                            'content': msg['content'].strip()
                        })

                if valid_messages:
                    conversations.append(valid_messages)
                    labels.append(item.get('label', 0))
        return conversations, labels

    train_conversations, train_labels = extract_split(train_data)
    test_conversations, test_labels = extract_split(test_data)
    conversations = train_conversations + test_conversations
    labels = train_labels + test_labels
    train_size = len(train_conversations)

    label_0_count = sum(1 for label in labels if label == 0)
    label_1_count = sum(1 for label in labels if label == 1)

    print(f"Valid train conversations: {len(train_conversations)}/{raw_train_size}")
    print(f"Valid test conversations: {len(test_conversations)}/{len(test_data)}")
    print(f"Loaded {label_0_count} samples with label 0")
    print(f"Loaded {label_1_count} samples with label 1")
    print(f"Total valid samples: {len(conversations)} (MAS pre-defined split)")

    return conversations, labels, train_size


class ConversationExtractor:
    """Extract embeddings from conversations using chat template."""

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def _hidden_size(self):
        config = getattr(self.model, "config", None)
        return int(getattr(config, "hidden_size", 0) or 1)

    @torch.no_grad()
    def extract_conversation_embedding(self, conversation, layer, layer_select):
        """Extract embedding from a conversation using chat template.

        Args:
            conversation: List of message dicts with 'role' and 'content' keys
            layer: Target layer number (0-31 for Llama models)
            layer_select: Type of layer component ('self_attention', 'MLP', 'hidden_state')

        Returns:
            Hidden state tensor from the last token position
        """
        # Validate conversation input
        if not conversation or not isinstance(conversation, list):
            print(f"Invalid conversation format: {type(conversation)}")
            return torch.zeros(1, self._hidden_size(), device=self.device)

        # Apply chat template to structured conversation
        input_ids = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)

        full_ids = input_ids

        # Step 2: extract hidden states via a plain PyTorch register_forward_hook.
        # self.model is now a plain AutoModelForCausalLM (not nnsight LanguageModel),
        # so there is no meta-tensor scan and no device conflict.
        if layer_select == "self_attention":
            target_module = self.model.model.layers[layer].self_attn
        elif layer_select == "MLP":
            target_module = self.model.model.layers[layer].mlp
        else:
            target_module = self.model.model.layers[layer]

        hidden_container = [None]

        def _capture_hook(module, inp, out):
            # Layer / MLP output may be a plain tensor or a tuple; index 0 is hidden state
            tensor = out[0] if isinstance(out, tuple) else out
            hidden_container[0] = tensor.detach().clone()

        handle = target_module.register_forward_hook(_capture_hook)
        try:
            self.model(full_ids)
        finally:
            handle.remove()

        hidden_states = hidden_container[0]
        print(f"Hidden states shape: {hidden_states.shape}")
        print(f"Hidden states tensor: {hidden_states}")

        # Extract last token's hidden state with dimension checking
        if hidden_states.dim() == 3:
            # Standard format: [batch_size, seq_len, hidden_dim] -> take last token
            result = hidden_states[:, -1, :].detach().clone()
        elif hidden_states.dim() == 2:
            # Already in [seq_len, hidden_dim] format -> take last row
            result = hidden_states[-1:, :].detach().clone()  # Keep batch dimension
        elif hidden_states.dim() == 1:
            # Single vector [hidden_dim] -> add batch dimension
            result = hidden_states.unsqueeze(0).detach().clone()
        else:
            print(f"Unexpected hidden states dimension: {hidden_states.dim()}")
            result = torch.zeros(1, self._hidden_size(), device=self.device)

        # Clear intermediate variables and GPU cache
        del hidden_states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result


def precompute_embeddings(conversations, labels, layer, model, tokenizer, layer_select, device, save_path, model_name=None):
    """
    Precompute embeddings for all conversations and save to JSON file.

    Args:
        conversations: List of conversation data
        labels: List of corresponding labels
        layer: Target layer number
        model: Language model
        tokenizer: Model tokenizer
        layer_select: Type of layer component
        device: Device to run computation on
        save_path: Path to save precomputed embeddings JSON
        model_name: Model name for verification (optional)

    Returns:
        Dictionary containing embeddings and metadata
    """
    print("Precomputing embeddings for all conversations...")

    extractor = ConversationExtractor(model, tokenizer, device)
    embeddings_data = {
        'embeddings': [],
        'labels': labels,
        'layer': layer,
        'layer_select': layer_select,
        'model_name': model_name,
        'total_samples': len(conversations)
    }

    # Compute embeddings for all conversations
    failed_indices = []
    for idx, conversation in enumerate(tqdm(conversations, desc="Computing embeddings")):
        # Extract embedding without augmentation for base version
        embedding = extractor.extract_conversation_embedding(
            conversation, layer, layer_select
        )

        # Convert to list for JSON serialization: [1, hidden_dim] -> squeeze -> [hidden_dim] -> list
        # Detach from computation graph and move to CPU before converting to list
        embeddings_data['embeddings'].append(embedding.squeeze(0).detach().cpu().tolist())

        # Clear GPU memory after each embedding extraction
        del embedding
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if idx % 100 == 0:
            print(f"Processed {idx}/{len(conversations)} conversations")

    # Report processing results
    if failed_indices:
        print(f"Warning: {len(failed_indices)} conversations failed to process: {failed_indices[:10]}...")
        embeddings_data['failed_indices'] = failed_indices
    else:
        print("All conversations processed successfully!")

    # Save to JSON file
    print(f"Saving precomputed embeddings to {save_path}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(embeddings_data, f)

    print(f"Precomputed embeddings saved successfully. Total samples: {len(conversations)}")
    print(f"Successfully processed: {len(conversations) - len(failed_indices)}/{len(conversations)}")
    return embeddings_data


class MyDataset(Dataset):
    def __init__(self, conversations, labels, layer, model=None, tokenizer=None, layer_select=None, device=None,
                 use_precomputed=False, precomputed_path=None, model_name=None):
        """
        Initialize dataset with conversation data and corresponding labels.

        Args:
            conversations: List of conversation data (only used when not using precomputed)
            labels: List of corresponding labels
            layer: Target layer number
            model: Language model (only required when not using precomputed)
            tokenizer: Model tokenizer (only required when not using precomputed)
            layer_select: Type of layer component (only required when not using precomputed)
            device: Device to run computation on (only required when not using precomputed)
            use_precomputed: Whether to use precomputed embeddings
            precomputed_path: Path to precomputed embeddings JSON file
            model_name: Model name for verification (optional)
        """
        self.use_precomputed = use_precomputed
        self.layer = layer
        self.layer_select = layer_select

        if use_precomputed and precomputed_path and os.path.exists(precomputed_path):
            # Load precomputed embeddings
            print(f"Loading precomputed embeddings from {precomputed_path}")
            with open(precomputed_path, 'r') as f:
                embeddings_data = json.load(f)

            # Validate that precomputed embeddings match current parameters
            stored_layer = embeddings_data.get('layer')
            stored_layer_select = embeddings_data.get('layer_select')
            stored_model_name = embeddings_data.get('model_name')

            if stored_layer != layer:
                print(f"Layer mismatch! Precomputed embeddings were generated for layer {stored_layer}, "
                      f"but current request is for layer {layer}. Use --force_recompute to regenerate.")
                # Will cause error when accessing mismatched data
                embeddings_data['embeddings'][layer]

            if stored_layer_select != layer_select:
                print(f"Layer select mismatch! Precomputed embeddings were generated for {stored_layer_select}, "
                      f"but current request is for {layer_select}. Use --force_recompute to regenerate.")
                # Will cause error when accessing mismatched data
                embeddings_data['embeddings'][layer_select]

            if model_name is not None and stored_model_name != model_name:
                print(f"Model mismatch! Precomputed embeddings were generated for model {stored_model_name}, "
                      f"but current request is for model {model_name}. Use --force_recompute to regenerate.")
                # Will cause error when accessing mismatched model data
                embeddings_data[model_name]

            self.embeddings = [torch.tensor(emb, dtype=torch.float32) for emb in embeddings_data['embeddings']]
            self.labels = torch.tensor(embeddings_data['labels'], dtype=torch.long)
            self.conversations = None  # Not needed when using precomputed
            self.extractor = None

            print(f"Loaded {len(self.embeddings)} precomputed embeddings")
            print(f"Verified parameters: layer={stored_layer}, layer_select={stored_layer_select}, model={stored_model_name}")

        else:
            # Use original real-time computation method
            self.conversations = conversations
            self.labels = torch.tensor(labels, dtype=torch.long)
            self.embeddings = None

            # Initialize conversation extractor for real-time computation
            if model is not None and tokenizer is not None:
                self.extractor = ConversationExtractor(model, tokenizer, device)
            else:
                print("Model and tokenizer are required when not using precomputed embeddings")
                # Will cause AttributeError when trying to use None
                self.extractor = ConversationExtractor(model, tokenizer, device)

    def __len__(self):
        if self.use_precomputed:
            return len(self.embeddings)
        else:
            return len(self.conversations)

    def __getitem__(self, idx):
        if self.use_precomputed:
            # Return precomputed embedding
            return self.embeddings[idx], self.labels[idx]
        else:
            # Original real-time computation with augmentation
            conversation = self.conversations[idx]

            # Extract embedding using conversation extractor
            embedding = self.extractor.extract_conversation_embedding(
                conversation, self.layer, self.layer_select
            )

            # Handle different embedding dimensions properly and detach from computation graph
            if embedding.dim() == 1:
                # Already [hidden_dim] format
                embedding_vec = embedding.detach().cpu()
            elif embedding.dim() == 2:
                # [1, hidden_dim] or [batch_size, hidden_dim] format -> squeeze to [hidden_dim]
                embedding_vec = embedding.squeeze(0).detach().cpu()
            else:
                # Unexpected dimension - will cause error when trying to process
                print(f"Warning: Unexpected embedding dimension {embedding.dim()} for conversation {idx}")
                embedding_vec = embedding.squeeze(0).detach().cpu()  # This will fail for unexpected dims

            # Clear GPU cache after processing each sample
            del embedding
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return embedding_vec, self.labels[idx].cpu()


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes=2, use_bias=True):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes, bias=use_bias)

    def forward(self, x):
        return torch.softmax(self.linear(x), dim=1)  # Softmax for binary/multi-class

    @torch.inference_mode()
    def predict(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(self.linear.weight.device)
        probs = torch.softmax(self.linear(x), dim=1)
        return torch.argmax(probs, dim=1)  # 返回类别索引 (0, 1)


def main(args):
    # Clear GPU cache at the start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    args.dataset = normalize_dataset_name(args.dataset)

    # Load and prepare rollout data from the new datasets in conversation format
    conversations, labels, train_size = load_and_prepare_rollout_data(dataset=args.dataset)

    # Load language model and tokenizer.
    # AutoModelForCausalLM is used instead of nnsight's LanguageModel because nnsight
    # replaces every submodule's forward() with nnsight_forward() at load time.
    # This means even a direct call to _model(inputs) still goes through nnsight's
    # meta-tensor scan, which conflicts with Qwen2's RoPE buffers already on cuda:0:
    #   "Tensor on device cuda:0 is not on the expected device meta!"
    # Using the plain HF model and register_forward_hook avoids this entirely.
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto', torch_dtype='auto')
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Ensure tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loaded {len(conversations)} conversation samples with {len(set(labels))} unique labels")
    print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # Create output directories for all layers (include dataset name to avoid conflicts)
    model_output_dir = build_model_output_dir(args.model)
    result_output_dir = os.path.join("results", os.path.relpath(model_output_dir, "models"))
    embedding_output_dir = os.path.join("embeddings", os.path.relpath(model_output_dir, "models"))
    os.makedirs(os.path.join(model_output_dir, args.dataset, args.layer), exist_ok=True)
    os.makedirs(os.path.join(result_output_dir, args.dataset, args.layer), exist_ok=True)
    os.makedirs(os.path.join(embedding_output_dir, args.dataset, args.layer), exist_ok=True)

    # Initialize comprehensive results storage
    all_results = []

    # Process all layers (0-31 for Llama models, or user specified)
    total_layers = min(args.max_layers, 32)  # Cap at 32 for safety
    print(f"Training probes on {total_layers} layers (0 to {total_layers-1})...")

    # Add overall progress tracking
    start_time = pd.Timestamp.now()

    for target_layer in range(total_layers):
        print(f"\n{'='*60}")
        print(f"Processing Layer {target_layer}/{total_layers-1}")
        print(f"{'='*60}")

        # Create layer-specific directories (include dataset name in path)
        os.makedirs(os.path.join(model_output_dir, args.dataset, args.layer, str(target_layer)), exist_ok=True)
        os.makedirs(os.path.join(result_output_dir, args.dataset, args.layer, str(target_layer)), exist_ok=True)
        os.makedirs(os.path.join(embedding_output_dir, args.dataset, args.layer, str(target_layer)), exist_ok=True)

        best_model_path = os.path.join(model_output_dir, args.dataset, args.layer, str(target_layer), "best_model.pth")
        result_path = os.path.join(result_output_dir, args.dataset, args.layer, str(target_layer), "best_model.csv")

        # Define path for precomputed embeddings (include dataset name in path)
        embeddings_path = os.path.join(embedding_output_dir, args.dataset, args.layer, str(target_layer), "precomputed_embeddings.json")

        # Check if precomputed embeddings exist, if not, compute and save them
        use_precomputed = False
        if os.path.exists(embeddings_path) and not args.force_recompute:
            print(f"Found precomputed embeddings at {embeddings_path}")
            use_precomputed = True
        else:
            if args.force_recompute:
                print(f"Force recompute flag set. Computing and saving embeddings to {embeddings_path}")
            else:
                print(f"Precomputed embeddings not found. Computing and saving to {embeddings_path}")

            # Precompute embeddings and save to JSON
            precompute_embeddings(
                conversations=conversations,
                labels=labels,
                layer=target_layer,
                model=model,
                tokenizer=tokenizer,
                layer_select=args.layer,
                device=args.device,
                save_path=embeddings_path,
                model_name=args.model,
            )
            use_precomputed = True

        # Create dataset - use precomputed embeddings if available
        if use_precomputed:
            dataset = MyDataset(
                conversations=None,  # Not needed when using precomputed
                labels=labels,
                layer=target_layer,
                layer_select=args.layer,  # Pass layer_select for validation
                use_precomputed=True,
                precomputed_path=embeddings_path,
                model_name=args.model
            )
        else:
            # Fallback to real-time computation (should not happen in normal flow)
            dataset = MyDataset(
                conversations=conversations,
                labels=labels,
                layer=target_layer,
                model=model,
                tokenizer=tokenizer,
                layer_select=args.layer,
                device=args.device,
                use_precomputed=False,
                model_name=args.model
            )

        # Split dataset using the fixed MAS train/test files.
        indices = list(range(len(dataset)))
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        print(f"Using MAS split: {len(train_indices)} train, {len(test_indices)} test")
        trainset = Subset(dataset, train_indices)
        testset = Subset(dataset, test_indices)

        sample_embedding, _ = dataset[0]
        input_dim = int(sample_embedding.numel())

        # Create data loaders
        dataloader = DataLoader(
            trainset,
            batch_size=args.batch_size,
            shuffle=True,
        )

        test_dataloader = DataLoader(
            testset,
            batch_size=args.batch_size,
            shuffle=True,
        )
        device = args.device

        # Create probe model for binary classification
        probe_model = LogisticRegression(input_dim, num_classes=2, use_bias=True).to(device)

        # 交叉熵损失函数
        criterion = nn.CrossEntropyLoss()

        test_acc = {"ACC": 0, "AUC": 0, "AUPRC": 0}

        # Adam 优化器
        optimizer = optim.Adam(probe_model.parameters(), lr=args.learning_rate)

        num_epochs = args.epoch
        for epoch in range(num_epochs):
            loss = 0
            probe_model.train()
            for batch_X, batch_y in tqdm(dataloader, desc=f"Layer {target_layer} - Epoch {epoch+1}/{num_epochs}"):
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                batch_X, batch_y = batch_X.detach(), batch_y.detach()

                optimizer.zero_grad()
                logits = probe_model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                # Clear batch variables to free memory
                del batch_X, batch_y, logits

                if args.debug:
                    break

            # Clear GPU cache after each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 计算测试集准确率
            cur_test_acc = logistic_regression_eval(args, probe_model, test_dataloader, device)
            if test_acc["ACC"] < cur_test_acc["ACC"]:
                test_acc["ACC"] = cur_test_acc["ACC"]
                torch.save(probe_model.state_dict(), best_model_path)

            if args.debug:
                print(f"Layer {target_layer} - Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Test: {cur_test_acc}")

        # Save results for current layer
        layer_results_df = pd.DataFrame({
            "Layer": [target_layer],
            "ACC": [test_acc["ACC"]],
            "AUC": [test_acc["AUC"]],
            "AUPRC": [test_acc["AUPRC"]]
        })
        layer_results_df.to_csv(result_path, index=False)

        # Add to comprehensive results
        all_results.append({
            "Layer": target_layer,
            "ACC": test_acc["ACC"],
            "AUC": test_acc["AUC"],
            "AUPRC": test_acc["AUPRC"]
        })

        print(f"Layer {target_layer} completed - Best ACC: {test_acc['ACC']:.4f}")

        # Progress reporting
        elapsed_time = pd.Timestamp.now() - start_time
        completed_layers = target_layer + 1
        avg_time_per_layer = elapsed_time / completed_layers
        remaining_layers = total_layers - completed_layers
        estimated_remaining = avg_time_per_layer * remaining_layers

        print(f"Progress: {completed_layers}/{total_layers} layers completed")
        print(f"Elapsed time: {elapsed_time}")
        if remaining_layers > 0:
            print(f"Estimated remaining time: {estimated_remaining}")

        # Memory reporting
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated()/1024**3
            gpu_memory_reserved = torch.cuda.memory_reserved()/1024**3
            print(f"GPU Memory - Used: {gpu_memory_used:.2f} GB, Reserved: {gpu_memory_reserved:.2f} GB")

        # Clear model and data from memory
        del probe_model, optimizer, criterion, dataset, dataloader, test_dataloader
        del trainset, testset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save comprehensive results for all layers (include dataset name in path)
    comprehensive_results_df = pd.DataFrame(all_results)
    comprehensive_results_path = os.path.join(result_output_dir, args.dataset, args.layer, "all_layers_results.csv")
    comprehensive_results_df.to_csv(comprehensive_results_path, index=False)

    # Calculate total execution time
    total_time = pd.Timestamp.now() - start_time

    print(f"\n{'='*60}")
    print("ALL LAYERS PROCESSING COMPLETED!")
    print(f"{'='*60}")
    print(f"Total execution time: {total_time}")
    print(f"Comprehensive results saved to: {comprehensive_results_path}")
    print("\nSummary of results:")
    print(comprehensive_results_df.to_string(index=False))

    # Find best performing layer
    best_layer_idx = comprehensive_results_df['ACC'].idxmax()
    best_layer = comprehensive_results_df.loc[best_layer_idx]
    print(f"\nBest performing layer: {best_layer['Layer']} (ACC: {best_layer['ACC']:.4f})")

    # Report statistics
    successful_layers = len([r for r in all_results if r['ACC'] > 0])
    failed_layers = total_layers - successful_layers
    print(f"\nProcessing statistics:")
    print(f"Successfully processed: {successful_layers}/{total_layers} layers")
    if failed_layers > 0:
        print(f"Failed layers: {failed_layers}")
    print(f"Average accuracy: {comprehensive_results_df['ACC'].mean():.4f}")
    print(f"Max accuracy: {comprehensive_results_df['ACC'].max():.4f}")
    print(f"Min accuracy: {comprehensive_results_df['ACC'].min():.4f}")


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args)
