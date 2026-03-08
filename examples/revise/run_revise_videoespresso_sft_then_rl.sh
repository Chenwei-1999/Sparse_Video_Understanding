#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/revise/config"
ENGINE="${ENGINE:-sglang}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SFT_CKPT_DIR="${SFT_CKPT_DIR:-$PROJECT_DIR/outputs/revise_videoespresso_sft}"
TEACHER_LOG="${TEACHER_LOG:-$PROJECT_DIR/outputs/videoespresso_pnp_7b_train_log.jsonl}"

if [ ! -f "$TEACHER_LOG" ]; then
    echo "ERROR: Teacher data log not found at $TEACHER_LOG"
    echo "Run ./examples/revise/run_generate_teacher_data_videoespresso.sh first."
    exit 1
fi

SFT_INPUT="$TEACHER_LOG" ./examples/revise/run_revise_videoespresso_sft.sh

LATEST_STEP=$(ls -d "$SFT_CKPT_DIR"/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1)
if [ -z "$LATEST_STEP" ]; then
    echo "ERROR: No SFT checkpoint found in $SFT_CKPT_DIR"
    exit 1
fi

HF_MODEL_PATH="$LATEST_STEP/huggingface"
if [ ! -d "$HF_MODEL_PATH" ]; then
    HF_MODEL_PATH="$LATEST_STEP/hf_model"
fi
if [ ! -d "$HF_MODEL_PATH" ]; then
    echo "ERROR: HF model not found under $LATEST_STEP"
    exit 1
fi

"$PYTHON_BIN" -m verl.trainer.main_ppo \
  --config-path "$CONFIG_PATH" \
  --config-name revise_videoespresso_grpo_after_sft \
  actor_rollout_ref.model.path="$HF_MODEL_PATH" \
  actor_rollout_ref.rollout.name="$ENGINE" \
  trainer.logger='["console","wandb"]' \
  "$@"
