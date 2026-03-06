#!/usr/bin/env bash
# Generate SFT teacher data: 7B plug-and-play on NExT-QA TRAIN split
# Adapted for local paths on p32027
set -euo pipefail

export PYTHONNOUSERSITE=1

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate verlrun

# ---- Paths ----
MODEL_PATH="/gpfs/projects/p32027/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots"
MODEL_SNAPSHOT=$(ls -1d "$MODEL_PATH"/*/ 2>/dev/null | head -1)
if [ -z "$MODEL_SNAPSHOT" ]; then
    echo "ERROR: Qwen2.5-VL-7B-Instruct not found in $MODEL_PATH"
    exit 1
fi

VIDEO_ROOT="/gpfs/projects/p32027/NExT-QA/NExTVideo"
MAP_JSON="/gpfs/projects/p32027/NExT-QA/map_vid_vidorID.json"
CSV="/gpfs/projects/p32027/NExT-QA/nextqa/train.csv"

MAX_SAMPLES="${MAX_SAMPLES:-0}"  # 0 = all ~34k samples
PORT=18100

LOG_DIR="outputs"
mkdir -p "$LOG_DIR"
LOG_JSONL="${LOG_DIR}/nextqa_pnp_7b_train_log.jsonl"
SERVER_LOG="debug_prompt_logs/vllm_server_teacher_$(date +%Y%m%d_%H%M%S).log"
mkdir -p debug_prompt_logs

echo "=== Generating Teacher Data ==="
echo "Model:   $MODEL_SNAPSHOT"
echo "CSV:     $CSV (train split)"
echo "Samples: $MAX_SAMPLES (0=all)"
echo "Log:     $LOG_JSONL"
echo ""

cd /gpfs/projects/p32027/VideoReasoning
export PYTHONPATH="${PYTHONPATH:-}:/gpfs/projects/p32027/VideoReasoning"

python examples/revise/plug_and_play_nextqa_vllm.py \
    --model-path "$MODEL_SNAPSHOT" \
    --video-root "$VIDEO_ROOT" \
    --map-json "$MAP_JSON" \
    --csv "$CSV" \
    --max-samples "$MAX_SAMPLES" \
    --max-rounds 4 \
    --max-frames-per-round 3 \
    --start-server \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --port "$PORT" \
    --log-jsonl "$LOG_JSONL" \
    --server-log "$SERVER_LOG" \
    --server-timeout-s 600 \
    --request-timeout-s 120 \
    --temperature 0.2 \
    --top-p 0.9 \
    --seed 42 \
    --progress-interval 100 \
    --resume-from-log \
    --force-final-answer \
    --max-retries-per-round 1

echo ""
echo "=== Teacher data generation complete ==="
echo "Log: $LOG_JSONL"
echo "Lines: $(wc -l < "$LOG_JSONL")"
echo ""
echo "Next: python examples/revise/generate_sft_data.py --input $LOG_JSONL --output outputs/sft_data/revise_sft.parquet"
