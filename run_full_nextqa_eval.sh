#!/usr/bin/env bash
# Full NExT-QA val evaluation with REVISE plug-and-play
# Usage: bash run_full_nextqa_eval.sh
set -euo pipefail

# Disable user site-packages to avoid polluting the conda env
export PYTHONNOUSERSITE=1

# ---- Paths ----
MODEL_PATH="/gpfs/projects/p32027/hf_cache/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots"
MODEL_SNAPSHOT=$(ls -1d "$MODEL_PATH"/*/ 2>/dev/null | head -1)
if [ -z "$MODEL_SNAPSHOT" ]; then
    echo "ERROR: Qwen2.5-VL-3B-Instruct not found in $MODEL_PATH"
    exit 1
fi

VIDEO_ROOT="/gpfs/projects/p32027/NExT-QA/NExTVideo"
MAP_JSON="/gpfs/projects/p32027/NExT-QA/map_vid_vidorID.json"
CSV="/gpfs/projects/p32027/NExT-QA/nextqa/val.csv"

# Paper settings: max_rounds=4, max_frames=3 (matches revise_nextqa_eval_vllm.yaml)
MAX_SAMPLES=0  # 0 = all samples
MAX_ROUNDS=4
MAX_FRAMES=3
PORT=18100

LOG_DIR="debug_prompt_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_JSONL="${LOG_DIR}/full_nextqa_val_${TIMESTAMP}.jsonl"
SUMMARY_JSON="${LOG_DIR}/full_nextqa_val_${TIMESTAMP}_summary.json"
SERVER_LOG="${LOG_DIR}/vllm_server_full_${TIMESTAMP}.log"

echo "=== REVISE NExT-QA Full Val Evaluation ==="
echo "Model:       Qwen2.5-VL-3B-Instruct"
echo "Dataset:     NExT-QA val (4996 samples)"
echo "Max rounds:  $MAX_ROUNDS"
echo "Frames/round: $MAX_FRAMES"
echo "Log:         $LOG_JSONL"
echo ""

cd /gpfs/projects/p32027/VideoReasoning
export PYTHONPATH="${PYTHONPATH:-}:/gpfs/projects/p32027/VideoReasoning"

python examples/revise/plug_and_play_nextqa_vllm.py \
    --model-path "$MODEL_SNAPSHOT" \
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
echo "=== Final Results ==="
cat "$SUMMARY_JSON" 2>/dev/null || echo "Summary not written; check $LOG_JSONL"
