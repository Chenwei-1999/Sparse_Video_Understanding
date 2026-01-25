#!/usr/bin/env bash
set -euo pipefail

# Evaluate a model on NExT-QA with the plug-and-play REVISE agent (vLLM).
#
# Usage:
#   ./scripts/eval_nextqa_pnp_vllm.sh /path/to/model /path/to/out_dir
#
# Environment overrides:
#   CONDA_ENV=verlrun
#   CUDA_VISIBLE_DEVICES=0,1,2,3
#   MAX_SAMPLES=500
#   MAX_ROUNDS=4
#   MAX_FRAMES_PER_ROUND=3
#   MAX_RETRIES_PER_ROUND=0
#   TP=4
#   MAX_MODEL_LEN=12288
#   GPU_MEM_UTIL=0.6

MODEL_PATH=${1:?MODEL_PATH required}
OUT_DIR=${2:?OUT_DIR required}

CONDA_ENV=${CONDA_ENV:-verlrun}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

MAX_SAMPLES=${MAX_SAMPLES:-500}
MAX_ROUNDS=${MAX_ROUNDS:-4}
MAX_FRAMES_PER_ROUND=${MAX_FRAMES_PER_ROUND:-3}
MAX_RETRIES_PER_ROUND=${MAX_RETRIES_PER_ROUND:-0}
TP=${TP:-4}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-12288}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.6}

VIDEO_ROOT=${VIDEO_ROOT:-/shares/hlw3876/chenwei/NExT-QA/NExTVideo}
MAP_JSON=${MAP_JSON:-/shares/hlw3876/chenwei/NExT-QA/map_vid_vidorID.json}
CSV=${CSV:-/shares/hlw3876/chenwei/NExT-QA/nextqa/val.csv}

mkdir -p "$OUT_DIR"

conda run -n "$CONDA_ENV" python examples/revise/plug_and_play_nextqa_vllm.py \
  --start-server \
  --model-path "$MODEL_PATH" \
  --video-root "$VIDEO_ROOT" \
  --map-json "$MAP_JSON" \
  --csv "$CSV" \
  --max-samples "$MAX_SAMPLES" \
  --max-rounds "$MAX_ROUNDS" \
  --max-frames-per-round "$MAX_FRAMES_PER_ROUND" \
  --max-retries-per-round "$MAX_RETRIES_PER_ROUND" \
  --use-candidate-frames \
  --use-candidate-frame-ids \
  --max-model-len "$MAX_MODEL_LEN" \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --log-jsonl "$OUT_DIR/val_pnp_${MAX_SAMPLES}.jsonl" \
  --summary-json "$OUT_DIR/val_pnp_${MAX_SAMPLES}_summary.json"

