# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**REVISE** (Reasoning with Video Sparsity) is a framework for question-aware sparse video understanding. It addresses information overload and insufficient key-information awareness through a multi-round agent loop that iteratively selects informative frames and maintains a compact **summary-as-state** using the POHR format (Previously seen, Observations, Hypotheses, Uncertainties, Reasons).

Two operating modes:
- **Plug-and-play**: wraps any VLM (including proprietary APIs) as a frozen black-box, no parameter updates
- **RL fine-tuning**: GRPO with **EAGER** reward (Evidence-Adjusted Gain for Efficient Reasoning) — combines confidence gain, summary sufficiency, and correct-and-early-stop bonuses; annotation-free

Built on a trimmed version of the `verl` RL training library.

## Common Commands

### Installation
```bash
conda create -n verlrun python=3.10 -y && conda activate verlrun
pip install -U pip && pip install -e .
# Backends:
pip install -r requirements_sglang.txt   # SGLang (recommended)
pip install -r requirements.txt          # vLLM
pip install -e ".[gpu]"                  # flash-attention, liger-kernel
```

### Evaluation (plug-and-play)
```bash
ENGINE=sglang ./examples/revise/run_revise_nextqa_eval.sh
ENGINE=vllm ./examples/revise/run_revise_nextqa_eval.sh --config-name revise_nextqa_eval_vllm
# Smoke test (tiny sample):
ENGINE=sglang ./examples/revise/run_revise_nextqa_smoke.sh
```

### RL Training (GRPO + EAGER)
```bash
ENGINE=sglang ./examples/revise/run_revise_nextqa_grpo.sh
```

### Main Entry Point
All training/eval goes through Hydra:
```bash
python3 -m verl.trainer.main_ppo \
  --config-path $(pwd)/examples/revise/config \
  --config-name <config_name> \
  actor_rollout_ref.rollout.name=sglang \
  [hydra overrides ...]
```

### Standalone Evaluation Scripts (bypass verl trainer, use vLLM directly)
```bash
python examples/revise/plug_and_play_nextqa_vllm.py           # NExT-QA
python examples/revise/plug_and_play_egoschema_vllm.py         # EgoSchema
python examples/revise/plug_and_play_videomme_lvbench_vllm.py  # Video-MME / LVBench
python examples/revise/plug_and_play_lvbench_hf.py             # LVBench (HF backend)
python examples/revise/oneshot_lvbench_hf.py                   # One-shot baseline
python examples/revise/eval_nextqa_caption_vllm.py             # Caption-only baseline
```

### Linting & Type Checking
```bash
ruff check .          # lint (line-length 120)
ruff format .         # auto-format
mypy verl/            # type check (mostly lenient, see below)
```

**No test suite** — This repo has no pytest tests or test infrastructure.

## Architecture

### Directory Layout
- `verl/` — Core RL training library (trimmed from upstream verl)
- `examples/revise/` — REVISE evaluation scripts, shell runners, Hydra configs
- `examples/videoagent/` — VideoAgent baseline implementations
- `paper/` — LaTeX source (synced from Overleaf, gitignored)
- `docs/` — Planning documents

### Key Subsystems

**Trainer** (`verl/trainer/`):
- `main_ppo.py` — Hydra entry point (`@hydra.main`); initializes Ray, builds trainer
- `ppo/ray_trainer.py` — `RayPPOTrainer`: orchestrates rollout → reward → advantage → actor update
- `ppo/core_algos.py` — GRPO, GAE, REINFORCE advantage estimators
- `ppo/reward.py` — Reward function loading; custom reward via `custom_reward_function.path` + `.name`

**Agent Loop** (`verl/experimental/agent_loop/`):
- `revise_agent_loop.py` — Core REVISE loop (~1600 lines): multi-round frame selection, POHR summary parsing via `<summary>/<frames>/<answer>` XML tags, retry on parse failure
- `agent_loop.py` — `AgentLoopBase` abstract class + `@register` decorator for loop registry
- `single_turn_agent_loop.py` / `tool_agent_loop.py` — Variants

**Rollout Backends** (`verl/workers/rollout/`):
- `sglang_rollout/` — SGLang inference (preferred)
- `vllm_rollout/` — vLLM inference
- `hf_server/` — HuggingFace transformers server

**Dataset Loaders** (`verl/utils/dataset/`):
- `nextqa_dataset.py` — NExT-QA: local CSV + videos, configured via `data.nextqa.video_root` / `data.nextqa.map_json`
- `lvbench_dataset.py` — LVBench: HF dataset + local video cache via `data.lvbench.video_cache_dir`
- `rl_dataset.py` — General RL training wrapper
- `vision_utils.py` — Video frame extraction

**Reward** (`verl/workers/reward_manager/`):
- Naive, batch, and prime reward computation strategies
- EAGER reward components: confidence gain, summary sufficiency, correct-and-early-stop, format bonus

### Configuration System

Hydra with composable defaults:
- **Base config**: `verl/trainer/config/ppo_trainer.yaml`
- **Experiment configs**: `examples/revise/config/*.yaml` — ~14 configs for different datasets, reward ablations, staged training
- REVISE-specific block: `actor_rollout_ref.rollout.revise` (max_rounds, max_frames_per_round, initial_sampling)
- Agent loop selection: `actor_rollout_ref.rollout.agent.default_agent_loop: revise_agent`

### Data Flow

1. Dataset loader produces prompts (question + MCQ options + video path)
2. Rollout backend runs agent loop: model outputs `<summary>` + `<frames>` (request) or `<answer>` (terminate) per round
3. Agent loop manages state across rounds — only the summary persists, not raw frames or history
4. Completed rollouts scored by reward manager (EAGER components configurable per-experiment)
5. GRPO advantage estimation → actor model update

### Supported Models
Primary: Qwen2.5-VL (3B, 7B), InternVL2 (8B). Also tested with GPT-4o (plug-and-play only).

### Infrastructure
- Ray for distributed multi-GPU training (typically 4 GPUs with tensor parallelism)
- wandb for experiment tracking (`trainer.logger='["console","wandb"]'`)
- FSDP for model sharding

## Code Conventions
- Python 3.10+
- Ruff linter: line-length 120, rules E/F/UP/B/I/G enabled; isort with `verl` as first-party; `scripts/legacy_model_merger.py` excluded
- Mypy: `ignore_errors = true` globally, but **strict** for `verl.trainer.config.algorithm`, `verl.trainer.ppo.core_algos`, `verl.trainer.ppo.reward`, `verl.workers.reward_manager.*`
- Version in `verl/version/version` (currently `0.7.0.dev`)
- Agent loops registered via `@register` decorator
- Data passed between components via `DataProto` (tensordict-based)
