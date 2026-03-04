#!/usr/bin/env bash
set -euo pipefail

# Generate SFT teacher data by running 7B plug-and-play on NExT-QA TRAIN split.
#
# This avoids data leakage: SFT trains on train-split teacher outputs,
# evaluation uses val/test split.
#
# Usage:
#   ./examples/revise/run_generate_teacher_data.sh [extra plug_and_play args...]
#
# Environment variables:
#   MODEL_PATH  — HF model id or local path (default: local 7B snapshot)
#   VIDEO_ROOT  — NExT-QA video root
#   MAP_JSON    — vid → vidorID mapping
#   CSV         — NExT-QA CSV split to use (default: train.csv)
#   MAX_SAMPLES — max samples (default: 0 = all)
#   NUM_SHARDS  — for multi-GPU data-parallel (default: 1)
#   SHARD_IDX   — shard index (default: 0)
#   LOG_PATH    — output JSONL path

PROJECT_DIR="$(pwd)"

MODEL_PATH="${MODEL_PATH:-/shares/hlw3876/chenwei/hf_cache/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5}"
VIDEO_ROOT="${VIDEO_ROOT:-/shares/hlw3876/chenwei/NExT-QA/NExTVideo}"
MAP_JSON="${MAP_JSON:-/shares/hlw3876/chenwei/NExT-QA/map_vid_vidorID.json}"
CSV="${CSV:-/shares/hlw3876/chenwei/NExT-QA/nextqa/train.csv}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_IDX="${SHARD_IDX:-0}"
LOG_PATH="${LOG_PATH:-$PROJECT_DIR/outputs/nextqa_pnp_7b_train_log.jsonl}"

echo "=== Generating teacher data ==="
echo "  Model:       $MODEL_PATH"
echo "  CSV:         $CSV"
echo "  Max samples: $MAX_SAMPLES"
echo "  Log:         $LOG_PATH"
echo "  Shards:      $NUM_SHARDS (idx=$SHARD_IDX)"

python3 "$PROJECT_DIR/examples/revise/plug_and_play_nextqa_vllm.py" \
    --model-path "$MODEL_PATH" \
    --video-root "$VIDEO_ROOT" \
    --map-json "$MAP_JSON" \
    --csv "$CSV" \
    --max-samples "$MAX_SAMPLES" \
    --max-rounds 5 \
    --max-frames-per-round 5 \
    --log-jsonl "$LOG_PATH" \
    --start-server \
    --num-shards "$NUM_SHARDS" \
    --shard-idx "$SHARD_IDX" \
    --resume-from-log \
    "$@"

echo "=== Teacher data generation complete ==="
echo "  Log: $LOG_PATH"
echo ""
echo "Next step: generate SFT parquet data:"
echo "  python examples/revise/generate_sft_data.py --input $LOG_PATH --output outputs/sft_data/revise_sft.parquet"
