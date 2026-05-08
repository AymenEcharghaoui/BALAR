# BALAR: Bayesian Agentic Loop for Active Reasoning

BALAR is a Bayesian framework for resolving ambiguous prompts through adaptive question-asking. Given an ambiguous task and a set of users who hold private information, an LLM agent maintains a **joint belief state** over latent dimensions of ambiguity, selects questions that maximize mutual information (MI) with the belief, updates its beliefs after each answer via Bayesian inference, and dynamically expands the hypothesis space when current dimensions are insufficient. Once uncertainty drops below a threshold, the agent produces a final answer.

**Datasets supported:**
- `ar-bench-dc` — Detective / murder-case reasoning (100 observations)
- `ar-bench-sp` — Situation puzzles (100 observations)
- `icraft-md` — Medical diagnosis via iCraft-MD (140 observations)

---

## Setup

### 1. Create and activate a conda environment

```bash
conda create -n balar python=3.11 -y
conda activate balar
```

### 2. Install dependencies

```bash
pip install openai pydantic torch numpy python-dotenv vllm
```

### 3. Set environment variables

**If using the OpenAI API:**
```bash
export OPENAI_AGENT_API_KEY=sk-...
export OPENAI_USER_API_KEY=sk-...
```

**If using local vLLM (models from HuggingFace):**
```bash
# HuggingFace credentials and cache
export HUGGING_FACE_HUB_TOKEN=hf_...
export HF_HUB_CACHE=/path/to/hf-cache

# Compiler vars 
export CC=$(which gcc)
export CXX=$(which g++)
unset CUDAHOSTCXX

# Required for Qwen models with extended context
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Set after launching vLLM servers
export VLLM_AGENT_BASE_URL=http://127.0.0.1:<agent_port>/v1
export VLLM_USER_BASE_URL=http://127.0.0.1:<user_port>/v1
export OPENAI_API_KEY="_"  
```

---

## Option A — OpenAI API

```bash
python run.py \
  --dataset_name          icraft-md \
  --agent_model_name      gpt-5-nano \
  --agent_model_class     openai \
  --user_model_name       gpt-5-nano \
  --user_model_class      openai \
  --max_rounds            100 \
  --max_ask_rounds        10 \
  --alpha                 0.1 \
  --beta                  1.0 \
  --num_initial_dims      5 \
  --num_initial_questions 3 \
  --temperature           0.1 \
  --top_p                 1.0 \
  --max_tokens            4096 \
  --max_num_values_per_dim      4 \
  --max_choices_per_question    4 \
  --max_new_questions_per_round 4 \
  --num_top_uncertainty_dims    2 \
  --expand_multiplier           1.0
```

---

## Option B — Local vLLM (e.g., Qwen2.5-32B)

Launch one vLLM server per model (agent and user can share the same model), then point the environment variables at their ports. Qwen models require the yarn rope-scaling override. Omit `--hf-overrides` for Llama.

```bash
# User model on GPU 0
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-32B-Instruct \
    --host 127.0.0.1 --port <user_port> \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --quantization bitsandbytes \
    --max-model-len 131072 \
    --hf-overrides '{"rope_scaling":{"type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}' &

# Agent model on GPU 1
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-32B-Instruct \
    --host 127.0.0.1 --port <agent_port> \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --quantization bitsandbytes \
    --max-model-len 131072 \
    --hf-overrides '{"rope_scaling":{"type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}' &
```

Once both servers respond at `http://127.0.0.1:<port>/health`, run:

```bash
python run.py \
  --dataset_name          icraft-md \
  --agent_model_name      Qwen/Qwen2.5-32B-Instruct \
  --agent_model_class     qwen \
  --user_model_name       Qwen/Qwen2.5-32B-Instruct \
  --user_model_class      qwen \
  --max_rounds            100 \
  --max_ask_rounds        10 \
  --alpha                 0.1 \
  --beta                  1.0 \
  --num_initial_dims      5 \
  --num_initial_questions 3 \
  --temperature           0.1 \
  --top_p                 1.0 \
  --max_tokens            100000 \
  --max_num_values_per_dim      4 \
  --max_choices_per_question    4 \
  --max_new_questions_per_round 4 \
  --num_top_uncertainty_dims    2 \
  --expand_multiplier           1.0
```

---

## Key arguments

### Model & API

| Argument | Default | Description |
|---|---|---|
| `--agent_model_name` | required | HuggingFace or OpenAI model ID for the **agent** (e.g. `Qwen/Qwen2.5-32B-Instruct`, `gpt-5-nano`) |
| `--agent_model_class` | required | Backend for the agent model: `openai`, `qwen`, or `llama` |
| `--user_model_name` | required | Model ID for the **user simulator** |
| `--user_model_class` | required | Backend for the user model: `openai`, `qwen`, or `llama` |
| `--openai_api_key` | `None` | OpenAI API key (overrides `OPENAI_AGENT_API_KEY` env var) |
| `--agent_vllm_base_url` | `None` | Base URL of the agent vLLM server (overrides `VLLM_AGENT_BASE_URL`) |
| `--user_vllm_base_url` | `None` | Base URL of the user vLLM server (overrides `VLLM_USER_BASE_URL`) |

### Generation

| Argument | Default | Description |
|---|---|---|
| `--temperature` | required | Sampling temperature for both models |
| `--top_p` | required | Top-p sampling parameter |
| `--max_tokens` | required | Max tokens per LLM call |

### Dataset

| Argument | Default | Description |
|---|---|---|
| `--dataset_name` | `icraft-md` | Which benchmark to run: `ar-bench-dc`, `ar-bench-sp`, or `icraft-md` |
| `--obs_start` | `0` | Index of the first observation to process |
| `--obs_end` | `-1` | Index of the last observation (exclusive). `-1` means run to the end of the dataset |

### Agentic loop

| Argument | Default | Description |
|---|---|---|
| `--max_rounds` | required | Hard cap on total loop iterations (both ASK and EXPAND steps count). Set high (e.g. `100`) to let other criteria dominate |
| `--max_ask_rounds` | required | Max number of questions the agent can ask users before stopping |
| `--alpha` | required | Convergence threshold. **When the dataset has a fixed answer set** (`ar-bench-dc`, `icraft-md`): stops when the MAP answer probability exceeds `1 - alpha`. **When there is no answer set** (`ar-bench-sp`): stops when the fraction of dimensions whose MAP marginal exceeds `1 - alpha` reaches `beta`. Lower = requires higher confidence before stopping |
| `--beta` | required | Only used when there is **no fixed answer set**: the fraction of dimensions that must each have their MAP marginal ≥ `1 - alpha` for the loop to converge (e.g. `1.0` = all dims must converge) |
| `--expand_multiplier` | required | Controls the expand-vs-ask trade-off. The agent expands the belief space when `entropy_gap > expand_multiplier × best_MI × remaining_rounds`. Higher values favour asking over expanding |

### Belief space initialisation

| Argument | Default | Description |
|---|---|---|
| `--num_initial_dims` | required | Number of latent dimensions (axes of ambiguity) created at round 0 |
| `--num_initial_questions` | required | Number of clarifying questions generated at round 0 |
| `--max_num_values_per_dim` | required | Max discrete values each dimension can take (caps hypothesis space per axis) |

### Question generation

| Argument | Default | Description |
|---|---|---|
| `--max_choices_per_question` | required | Max answer choices offered per question |
| `--max_new_questions_per_round` | required | Max new questions generated each time the belief space is expanded |
| `--num_top_uncertainty_dims` | required | Number of highest-entropy dimensions passed to the LLM when generating new questions after an expand step |

### Optional

| Argument | Default | Description |
|---|---|---|
| `--use_verifier` / `--no-use_verifier` | `False` | Enable a self-verification step: after each LLM response, a second call checks validity and optionally corrects it before it is used |

---

## Citation

If you use BALAR in your research, please cite:

```bibtex
@article{echarghaoui2026balar,
  title   = {BALAR: A Bayesian Agentic Loop for Active Reasoning},
  author  = {Echarghaoui, Aymen and Wu, Dongxia and Fox, Emily B.},
  journal = {arXiv preprint arXiv:2605.05386},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.05386}
}
```

## License

This code is released under the [MIT License](LICENSE). The accompanying paper is distributed on arXiv under the [arXiv non-exclusive license](http://arxiv.org/licenses/nonexclusive-distrib/1.0/).
