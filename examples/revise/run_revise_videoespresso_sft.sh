#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/revise/config"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

SFT_INPUT="${SFT_INPUT:-$PROJECT_DIR/outputs/videoespresso_pnp_7b_train_log.jsonl}"
SFT_OUTPUT="${SFT_OUTPUT:-$PROJECT_DIR/outputs/sft_data/videoespresso_revise_sft.parquet}"
N_GPUS="${N_GPUS:-4}"

TRAIN_FILE="${SFT_OUTPUT%.parquet}_train.parquet"
if [ ! -f "$TRAIN_FILE" ]; then
    if [ ! -f "$SFT_INPUT" ]; then
        echo "ERROR: Teacher data not found at $SFT_INPUT"
        echo "Run ./examples/revise/run_generate_teacher_data_videoespresso.sh first."
        exit 1
    fi
    "$PYTHON_BIN" "$PROJECT_DIR/examples/revise/generate_sft_data.py" \
        --input "$SFT_INPUT" \
        --output "$SFT_OUTPUT" \
        --val-csv ""
fi

"$TORCHRUN_BIN" --nproc_per_node="$N_GPUS" \
    -m verl.trainer.fsdp_sft_trainer \
    --config-path "$CONFIG_PATH" \
    --config-name revise_videoespresso_sft \
    "$@"

BASE_MODEL="${BASE_MODEL:-/shares/hlw3876/chenwei/hf_cache/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3}"
SFT_CKPT_DIR="${SFT_CKPT_DIR:-$PROJECT_DIR/outputs/revise_videoespresso_sft}"
for hf_dir in "$SFT_CKPT_DIR"/global_step_*/huggingface; do
    if [ -d "$hf_dir" ] && [ ! -f "$hf_dir/preprocessor_config.json" ]; then
        cp "$BASE_MODEL/preprocessor_config.json" "$hf_dir/"
    fi
done
