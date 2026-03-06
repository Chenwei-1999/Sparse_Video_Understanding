#!/usr/bin/env bash
# Evaluate SFT-trained Qwen2.5-VL-3B on NExT-QA val
set -euo pipefail

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook)"
conda activate verlrun

SFT_MODEL="/gpfs/projects/p32027/VideoReasoning/outputs/revise_nextqa_sft/global_step_844/huggingface"
VIDEO_ROOT="/gpfs/projects/p32027/NExT-QA/NExTVideo"
MAP_JSON="/gpfs/projects/p32027/NExT-QA/map_vid_vidorID.json"
CSV="/gpfs/projects/p32027/NExT-QA/nextqa/val.csv"

MAX_SAMPLES="${MAX_SAMPLES:-0}"  # 0 = all
MAX_ROUNDS=4
MAX_FRAMES=3
PORT=18100

LOG_DIR="debug_prompt_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_JSONL="${LOG_DIR}/sft_nextqa_val_${TIMESTAMP}.jsonl"
SUMMARY_JSON="${LOG_DIR}/sft_nextqa_val_${TIMESTAMP}_summary.json"
SERVER_LOG="${LOG_DIR}/vllm_server_sft_${TIMESTAMP}.log"

echo "=== REVISE NExT-QA SFT Model Evaluation ==="
echo "Model:       SFT Qwen2.5-VL-3B (global_step_844)"
echo "Dataset:     NExT-QA val (4996 samples)"
echo "Max rounds:  $MAX_ROUNDS"
echo "Frames/round: $MAX_FRAMES"
echo "Log:         $LOG_JSONL"
echo ""

cd /gpfs/projects/p32027/VideoReasoning
export PYTHONPATH="${PYTHONPATH:-}:/gpfs/projects/p32027/VideoReasoning"

python examples/revise/plug_and_play_nextqa_vllm.py \
    --model-path "$SFT_MODEL" \
    --video-root "$VIDEO_ROOT" \
    --map-json "$MAP_JSON" \
    --csv "$CSV" \
    --max-samples "$MAX_SAMPLES" \
    --max-rounds "$MAX_ROUNDS" \
    --max-frames-per-round "$MAX_FRAMES" \
    --start-server \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --port "$PORT" \
    --log-jsonl "$LOG_JSONL" \
    --summary-json "$SUMMARY_JSON" \
    --server-log "$SERVER_LOG" \
    --server-timeout-s 600 \
    --request-timeout-s 120 \
    --temperature 0.2 \
    --top-p 0.9 \
    --seed 42 \
    --progress-interval 50 \
    --resume-from-log \
    --force-final-answer \
    --max-retries-per-round 1

echo ""
echo "=== SFT Model Results ==="
cat "$SUMMARY_JSON" 2>/dev/null || echo "Summary not written; check $LOG_JSONL"
