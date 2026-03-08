#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL_PATH="${MODEL_PATH:-/shares/hlw3876/chenwei/hf_cache/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5}"
VIDEO_ROOT="${VIDEO_ROOT:-/shares/hlw3876/chenwei/VideoEspresso}"
JSON="${JSON:-$PROJECT_DIR/outputs/videoespresso_train_mc.json}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_IDX="${SHARD_IDX:-0}"
LOG_PATH="${LOG_PATH:-$PROJECT_DIR/outputs/videoespresso_pnp_7b_train_log.jsonl}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.55}"

if [ ! -f "$JSON" ]; then
    echo "ERROR: MC train JSON not found at $JSON"
    echo "Run: python scripts/repro/prepare_videoespresso_mc_train.py --output $JSON"
    exit 1
fi

"$PYTHON_BIN" "$PROJECT_DIR/examples/revise/plug_and_play_egoschema_vllm.py" \
    --model-path "$MODEL_PATH" \
    --dataset-name videoespresso \
    --json "$JSON" \
    --video-root "$VIDEO_ROOT" \
    --max-samples "$MAX_SAMPLES" \
    --max-rounds 5 \
    --max-frames-per-round 5 \
    --log-jsonl "$LOG_PATH" \
    --start-server \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --num-shards "$NUM_SHARDS" \
    --shard-idx "$SHARD_IDX" \
    --resume-from-log \
    "$@"
