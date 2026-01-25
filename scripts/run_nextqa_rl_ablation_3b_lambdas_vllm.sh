#!/usr/bin/env bash
set -euo pipefail

# Run EAGER (paper) reward ablations on NExT-QA with Qwen2.5-VL-3B using vLLM (4 GPUs).
#
# Usage:
#   WANDB_MODE=online ENGINE=vllm ./scripts/run_nextqa_rl_ablation_3b_lambdas_vllm.sh
#
# Notes:
# - Default is a quick sweep (30 steps). Set STEPS=100 for the full run.
# - Outputs are written under outputs/YYYY-MM-DD/.

ENGINE=${ENGINE:-vllm}
CONDA_ENV=${CONDA_ENV:-verlrun}
DATE=${DATE:-2026-01-24}
ROOT_DIR="$(pwd)"
CONFIG_PATH="$ROOT_DIR/examples/revise/config"
CONFIG_NAME=${CONFIG_NAME:-revise_nextqa_grpo_eager_paper_reward_100}
STEPS=${STEPS:-30}
SAVE_FREQ=${SAVE_FREQ:-$STEPS}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# Default W&B behavior:
# - If the user didn't specify WANDB_MODE, prefer online when a key is present, otherwise offline.
if [[ -z "${WANDB_MODE:-}" ]]; then
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    export WANDB_MODE=online
  else
    export WANDB_MODE=offline
  fi
fi

BASE_OUT="${ROOT_DIR}/outputs/${DATE}/rl_ablation_nextqa_3b_eager_paper"
mkdir -p "$BASE_OUT"

declare -a SETTINGS=(
  "1.0 1.0 0.5"  # baseline
  "0.0 1.0 0.5"  # no conf
  "1.0 0.0 0.5"  # no sum
  "1.0 1.0 0.0"  # no stop
  "2.0 1.0 0.5"  # high conf
  "1.0 2.0 0.5"  # high sum
)

for triple in "${SETTINGS[@]}"; do
  read -r LCONF LSUM LSTOP <<<"$triple"
  RUN_NAME="qwen2p5vl3b_nextqa_eagerPaper_c${LCONF}_s${LSUM}_st${LSTOP}"
  OUT_DIR="${BASE_OUT}/${RUN_NAME}"

  echo "==> ${RUN_NAME}"
  if [[ -f "${OUT_DIR}/latest_checkpointed_iteration.txt" ]]; then
    existing_step="$(tr -d ' \n' < "${OUT_DIR}/latest_checkpointed_iteration.txt" || true)"
    if [[ "${existing_step}" == "${STEPS}" ]]; then
      echo "    skip: already has global_step_${STEPS} (${OUT_DIR})"
      continue
    fi
  fi
  conda run -n "$CONDA_ENV" python -m verl.trainer.main_ppo \
    --config-path "$CONFIG_PATH" \
    --config-name "$CONFIG_NAME" \
    actor_rollout_ref.rollout.name="$ENGINE" \
    custom_reward_function.reward_kwargs.lambda_conf="$LCONF" \
    custom_reward_function.reward_kwargs.lambda_sum="$LSUM" \
    custom_reward_function.reward_kwargs.lambda_stop="$LSTOP" \
    trainer.experiment_name="$RUN_NAME" \
    trainer.total_training_steps="$STEPS" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.default_local_dir="$OUT_DIR"
done
