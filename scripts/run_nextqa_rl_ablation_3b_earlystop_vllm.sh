#!/usr/bin/env bash
set -euo pipefail

# Run early-stop reward design ablations (β, λ_stop, τ) on NExT-QA with
# Qwen2.5-VL-3B using vLLM (4 GPUs).
#
# We fix (λ_conf, λ_sum) to the best setting from `reports/nextqa_rl_ablation_3b_eager_paper.md`:
#   (λ_conf=0.0, λ_sum=1.0)
#
# Usage:
#   WANDB_MODE=online ENGINE=vllm EVAL=1 MAX_SAMPLES=500 ./scripts/run_nextqa_rl_ablation_3b_earlystop_vllm.sh
#
# Notes:
# - Default is a quick sweep (30 RL steps). Set STEPS=100 for the longer run.
# - If EVAL=1, we run plug-and-play eval after training and write val_pnp_*.jsonl under each run dir.

ENGINE=${ENGINE:-vllm}
CONDA_ENV=${CONDA_ENV:-verlrun}
DATE=${DATE:-2026-01-26}
ROOT_DIR="$(pwd)"
CONFIG_PATH="$ROOT_DIR/examples/revise/config"
CONFIG_NAME=${CONFIG_NAME:-revise_nextqa_grpo_eager_paper_reward_100}
STEPS=${STEPS:-30}
SAVE_FREQ=${SAVE_FREQ:-$STEPS}
EVAL=${EVAL:-0}

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

BASE_OUT="${ROOT_DIR}/outputs/${DATE}/rl_ablation_nextqa_3b_eager_paper_earlystop"
mkdir -p "$BASE_OUT"

# (β, λ_stop, τ)
declare -a SETTINGS=(
  "0.0 0.5 2"  # baseline: binary stop reward, τ=2
  "0.0 0.5 1"  # stricter early-stop window
  "0.0 0.5 3"  # more lenient early-stop window
  "0.0 0.25 2" # weaker stop reward
  "0.0 1.0 2"  # stronger stop reward
  "0.5 0.5 2"  # add extra early-stop bonus within τ
  "1.0 0.5 2"  # larger early-stop bonus within τ
  "1.0 1.0 2"  # strongest: λ_stop + β
)

if [[ -n "${SETTINGS_OVERRIDE:-}" ]]; then
  # Format: "BETA LSTOP TAU;BETA LSTOP TAU;..."
  IFS=";" read -r -a SETTINGS <<<"${SETTINGS_OVERRIDE}"
fi

for triple in "${SETTINGS[@]}"; do
  read -r BETA LSTOP TAU <<<"$triple"
  RUN_NAME="qwen2p5vl3b_nextqa_eagerPaper_stopB${BETA}_st${LSTOP}_tau${TAU}"
  OUT_DIR="${BASE_OUT}/${RUN_NAME}"
  mkdir -p "$OUT_DIR"

  echo "==> ${RUN_NAME}"
  if [[ -f "${OUT_DIR}/latest_checkpointed_iteration.txt" ]]; then
    existing_step="$(tr -d ' \n' < "${OUT_DIR}/latest_checkpointed_iteration.txt" || true)"
    if [[ "${existing_step}" == "${STEPS}" ]]; then
      echo "    skip train: already has global_step_${STEPS} (${OUT_DIR})"
    else
      existing_step=""
    fi
  else
    existing_step=""
  fi

  if [[ -z "${existing_step}" ]]; then
    PYTHONUNBUFFERED=1 conda run -n "$CONDA_ENV" python -u -m verl.trainer.main_ppo \
      --config-path "$CONFIG_PATH" \
      --config-name "$CONFIG_NAME" \
      actor_rollout_ref.rollout.name="$ENGINE" \
      custom_reward_function.reward_kwargs.lambda_conf=0.0 \
      custom_reward_function.reward_kwargs.lambda_sum=1.0 \
      custom_reward_function.reward_kwargs.lambda_stop="$LSTOP" \
      custom_reward_function.reward_kwargs.stop_round_threshold="$TAU" \
      custom_reward_function.reward_kwargs.stop_bonus_beta="$BETA" \
      trainer.experiment_name="$RUN_NAME" \
      trainer.total_training_steps="$STEPS" \
      trainer.save_freq="$SAVE_FREQ" \
      trainer.default_local_dir="$OUT_DIR"
  fi

  if [[ "${EVAL}" == "1" ]]; then
    MAX_SAMPLES=${MAX_SAMPLES:-500}
    SUMMARY_JSON="${OUT_DIR}/val_pnp_${MAX_SAMPLES}_summary.json"
    if [[ -f "${SUMMARY_JSON}" ]]; then
      echo "    skip eval: already has ${SUMMARY_JSON}"
      continue
    fi
    MODEL_PATH="${OUT_DIR}/global_step_${STEPS}/actor/huggingface"
    if [[ ! -d "${MODEL_PATH}" ]]; then
      echo "    warn: missing model path ${MODEL_PATH}; skip eval"
      continue
    fi
    MAX_SAMPLES="$MAX_SAMPLES" bash ./scripts/eval_nextqa_pnp_vllm.sh "$MODEL_PATH" "$OUT_DIR"
  fi
done
