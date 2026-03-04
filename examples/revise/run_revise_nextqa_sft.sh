#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./examples/revise/run_revise_nextqa_sft.sh [hydra overrides ...]
#
# Generates SFT data from 7B TRAIN-split eval logs (if not already present),
# then runs FSDP SFT training on Qwen2.5-VL-3B to teach the REVISE format.
#
# Prerequisites:
#   1. Generate teacher data on train split first:
#      ./examples/revise/run_generate_teacher_data.sh

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/revise/config"

SFT_INPUT="${SFT_INPUT:-$PROJECT_DIR/outputs/nextqa_pnp_7b_train_log.jsonl}"
SFT_OUTPUT="${SFT_OUTPUT:-$PROJECT_DIR/outputs/sft_data/revise_sft.parquet}"
N_GPUS="${N_GPUS:-4}"

# Step 1: Generate SFT parquet data if not present
TRAIN_FILE="${SFT_OUTPUT%.parquet}_train.parquet"
if [ ! -f "$TRAIN_FILE" ]; then
    if [ ! -f "$SFT_INPUT" ]; then
        echo "ERROR: Teacher data not found at $SFT_INPUT"
        echo "Run ./examples/revise/run_generate_teacher_data.sh first to generate train-split teacher data."
        exit 1
    fi
    echo "=== Generating SFT data ==="
    python3 "$PROJECT_DIR/examples/revise/generate_sft_data.py" \
        --input "$SFT_INPUT" \
        --output "$SFT_OUTPUT"
else
    echo "=== SFT data already exists at $TRAIN_FILE, skipping generation ==="
fi

# Step 2: Run SFT training
echo "=== Starting SFT training ==="
torchrun --nproc_per_node="$N_GPUS" \
    -m verl.trainer.fsdp_sft_trainer \
    --config-path "$CONFIG_PATH" \
    --config-name revise_nextqa_sft \
    "$@"

echo "=== SFT training complete ==="
