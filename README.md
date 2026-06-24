# AgentLens / Agent MechSuits

This is the official repository for the paper **AgentLens: Interpretable Safety Steering via Mechanistic Subspaces for Multi-Turn Coding Agent**.

Paper PDF: [`agentlens.pdf`](./agentlens.pdf)

AgentLens is a white-box safety framework for multi-turn coding agents. Instead of treating agent safety as a one-time task-level decision, AgentLens monitors the agent during execution. It trains lightweight linear probes on step-level hidden states, detects harmful execution states at runtime, and steers the model in a sparse safety-relevant subspace when risk is detected.

The paper also introduces **MAS (Mechanistic Agent Safety)**, a step-annotated benchmark for mechanistic safety analysis of coding agents. This repository includes MAS splits for three white-box coding-agent backbones:

| Model family | Train file | Test file |
| --- | --- | --- |
| Gemma-2-9B | `MAS/Gemma/trainset.json` | `MAS/Gemma/testset.json` |
| LLaMA-3.1-8B | `MAS/LLaMA/train.json` | `MAS/LLaMA/test.json` |
| Qwen-2.5-7B | `MAS/Qwen/train.json` | `MAS/Qwen/test.json` |

## Set Up

Create a Python environment with the core dependencies:

```bash
pip install -r requirement.txt
```

The scripts use relative paths by default. Put model checkpoints under `models/` and keep MAS data under `MAS/`.

If you use the LLM judge / adaptive steering controller, provide an OpenRouter key either through `.env`:

```bash
OPENROUTER_API_KEY=your_key
```

or pass it directly:

```bash
python agentlens.py --openrouter_api_key your_key
```

## Data

MAS contains step-level coding-agent trajectories with labels:

- `label = 0`: benign or safety-aware/refusal step
- `label = 1`: harmful execution step

The current local split sizes are:

| Dataset | Train | Test |
| --- | ---: | ---: |
| Gemma | 146 | 86 |
| LLaMA | 682 | 469 |
| Qwen | 328 | 236 |

`train_probe.py` is restricted to these MAS train/test splits.

## Training Probes

Train a LLaMA probe:

```bash
python train_probe.py \
  --model models/meta-llama/Llama-3.1-8B-Instruct \
  --dataset LLaMA \
  --layer hidden_state \
  --max_layers 32 \
  --epoch 100 \
  --learning_rate 0.01 \
  --batch_size 16 \
  --device cuda:0
```

This creates:

```text
models/meta-llama/Llama-3.1-8B-Instruct/LLaMA/hidden_state/<layer>/best_model.pth
results/meta-llama/Llama-3.1-8B-Instruct/LLaMA/hidden_state/<layer>/best_model.csv
embeddings/meta-llama/Llama-3.1-8B-Instruct/LLaMA/hidden_state/<layer>/precomputed_embeddings.json
```

## AgentLens Steering

Run adaptive AgentLens steering with:

```bash
python agentlens.py \
  --model models/meta-llama/Llama-3.1-8B-Instruct \
  --probe_model_path models/meta-llama/Llama-3.1-8B-Instruct/LLaMA/hidden_state/28/best_model.pth \
  --judge_model deepseek/deepseek-v4-pro \
  --data_path MAS/LLaMA/test.json \
  --output_path results/LLaMA/agentlens.json \
  --layer 28 \
  --top_k 10 \
  --max_iterations 100 \
  --no_improve_patience 5 \
  --openrouter_api_key "$OPENROUTER_API_KEY" \
  --max_new_tokens 200 \
  --seed 42
```

The script:

1. Loads the trained linear probe.
2. Extracts the top safety-relevant hidden dimensions.
3. Detects harmful execution states from step-level hidden representations.
4. Uses `deepseek/deepseek-v4-pro` as the default LLM judge/controller to search steering strength.
5. Saves before/after steering outputs and judge iterations.

Use `--debug` to bypass probe detection and steer every sample.

## Manual Steering

For fixed-direction steering experiments:

```bash
python manual_steering.py \
  --model models/meta-llama/Llama-3.1-8B-Instruct \
  --probe_model_path models/meta-llama/Llama-3.1-8B-Instruct/LLaMA/hidden_state/14/best_model.pth \
  --data_path MAS/LLaMA/test.json \
  --output_path results/LLaMA/manual_steering.json \
  --direction negative \
  --layer 14 \
  --alpha 1.9 \
  --top_k 10
```

This script is useful for causal validation, including negative steering experiments that test whether the learned subspace can push refusal states toward unsafe behavior.

## Evaluation

Evaluate safety of generated trajectories:

```bash
python eval.py \
  --task safety \
  --input results/LLaMA/agentlens.json \
  --field after_steer \
  --name llama_agentlens
```

Evaluate output collapse before/after steering:

```bash
python eval.py \
  --task collapse \
  --input results/LLaMA/agentlens.json \
  --name llama_agentlens
```

`eval.py` uses OpenAI-compatible chat completion calls. Set the API key in the script or adapt it to read from your environment.

## Citation

If you find this code or MAS useful, please cite:

```bibtex
@article{luo2026agentlens,
  title={AgentLens: Interpretable Safety Steering via Mechanistic Subspaces for Multi-Turn Coding Agent},
  author={Luo, Weidi and Zhang, Qiming and Quan, Yihao and Jin, Mingyu and Cai, Jie and Xiao, Chaowei and Niu, Jingcheng and Xiang, Zhen},
  journal={arXiv preprint},
  year={2026}
}
```
