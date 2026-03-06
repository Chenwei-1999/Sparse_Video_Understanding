#!/usr/bin/env bash
# Smoke test: run REVISE plug-and-play on NExT-QA val set (small sample)
# Usage: bash run_smoke_nextqa.sh [--max-samples N]
set -euo pipefail

# Disable user site-packages to avoid polluting the conda env
export PYTHONNOUSERSITE=1

# ---- Paths ----
MODEL_PATH="/gpfs/projects/p32027/hf_cache/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots"
# Auto-detect the snapshot dir
MODEL_SNAPSHOT=$(ls -1d "$MODEL_PATH"/*/ 2>/dev/null | head -1)
if [ -z "$MODEL_SNAPSHOT" ]; then
    echo "ERROR: Qwen2.5-VL-3B-Instruct not found in $MODEL_PATH"
    exit 1
fi
echo "Using model: $MODEL_SNAPSHOT"

VIDEO_ROOT="/gpfs/projects/p32027/NExT-QA/NExTVideo"
MAP_JSON="/gpfs/projects/p32027/NExT-QA/map_vid_vidorID.json"
CSV="/gpfs/projects/p32027/NExT-QA/nextqa/val.csv"

MAX_SAMPLES="${1:-20}"  # default 20 samples for smoke test
MAX_ROUNDS=4
MAX_FRAMES=3
PORT=18100

LOG_DIR="debug_prompt_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_JSONL="${LOG_DIR}/smoke_nextqa_${TIMESTAMP}.jsonl"
SUMMARY_JSON="${LOG_DIR}/smoke_nextqa_${TIMESTAMP}_summary.json"
SERVER_LOG="${LOG_DIR}/vllm_server_${TIMESTAMP}.log"

echo "=== REVISE NExT-QA Smoke Test ==="
echo "Model:       $(basename "$MODEL_SNAPSHOT")"
echo "Samples:     $MAX_SAMPLES"
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
    --temperature 0.2 \
    --top-p 0.9 \
    --seed 42

echo ""
echo "=== Results ==="
if [ -f "$SUMMARY_JSON" ]; then
    cat "$SUMMARY_JSON"
else
    echo "(No summary JSON produced — check $LOG_JSONL for per-sample results)"
    if [ -f "$LOG_JSONL" ]; then
        echo "Samples logged: $(wc -l < "$LOG_JSONL")"
        # Quick accuracy from JSONL
        python3 -c "
import json, sys
correct = total = 0
with open('$LOG_JSONL') as f:
    for line in f:
        rec = json.loads(line)
        total += 1
        if rec.get('correct'):
            correct += 1
if total:
    print(f'Accuracy: {correct}/{total} = {correct/total*100:.1f}%')
    avg_rounds = sum(1 for _ in open('$LOG_JSONL')) # placeholder
else:
    print('No results')
"
    fi
fi
