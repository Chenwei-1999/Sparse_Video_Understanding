#!/usr/bin/env bash
set -euo pipefail

# Run a VideoAgent-style caption retrieval baseline on NExT-QA val (4 shards, TP=1 per GPU),
# in a tmux session.
#
# Usage:
#   ./scripts/run_nextqa_videoagent_caption_val_tp1.sh
#
# Environment overrides:
#   SESSION=nextqa_videoagent_caption_val_tp1
#   DATE=2026-01-25
#   PY=/shares/hlw3876/chenwei/miniconda3/envs/verlrun/bin/python
#   MODEL_PATH=...
#   CSV=...
#   CAPTIONS_DIR=...

cd /home/cxk2993/verl

export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-online}"

SESSION="${SESSION:-nextqa_videoagent_caption_val_tp1}"
DATE="${DATE:-$(date +%F)}"

PY="${PY:-/shares/hlw3876/chenwei/miniconda3/envs/verlrun/bin/python}"
MODEL_PATH="${MODEL_PATH:-/shares/hlw3876/chenwei/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5}"

VIDEO_ROOT="${VIDEO_ROOT:-/shares/hlw3876/chenwei/NExT-QA/NExTVideo}"
MAP_JSON="${MAP_JSON:-/shares/hlw3876/chenwei/NExT-QA/map_vid_vidorID.json}"
CSV="${CSV:-/shares/hlw3876/chenwei/NExT-QA/nextqa/val.csv}"
CAPTIONS_DIR="${CAPTIONS_DIR:-data/nextqa_allcaps_1fps}"

OUT_DIR="${OUT_DIR:-outputs/${DATE}/nextqa_caption_compare_qwen2p5vl7b/videoagent_caption_val_tp1}"
mkdir -p "${OUT_DIR}"

MAX_ROUNDS="${MAX_ROUNDS:-5}"
MAX_FRAMES="${MAX_FRAMES:-5}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.6}"

WANDB_PROJECT="${WANDB_PROJECT:-revise_nextqa}"
WANDB_ENTITY="${WANDB_ENTITY:-cxu-research}"
WANDB_GROUP="${WANDB_GROUP:-nextqa_val_videoagent_caption_tp1}"

LOG_JSONL="${LOG_JSONL:-${OUT_DIR}/log.jsonl}"
SUMMARY_JSON="${SUMMARY_JSON:-${OUT_DIR}/summary.json}"

tmux kill-session -t "${SESSION}" 2>/dev/null || true

mkcmd() {
  local shard="$1"
  local gpu="$2"
  local port="$3"
  local run_log="${OUT_DIR}/run.shard${shard}of4.log"
  local server_log="${OUT_DIR}/server.shard${shard}of4.log"
  local wandb_name="videoagent_caption_val_tp1_s${shard}"

  printf "%s" \
    "cd /home/cxk2993/verl && export CUDA_VISIBLE_DEVICES=${gpu} WANDB_MODE=online && ("\
" ${PY} examples/videoagent/eval_nextqa_videoagent_caption_vllm.py"\
" --model-path '${MODEL_PATH}'"\
" --captions-dir '${CAPTIONS_DIR}'"\
" --video-root '${VIDEO_ROOT}'"\
" --map-json '${MAP_JSON}'"\
" --csv '${CSV}'"\
" --max-samples 0"\
" --num-shards 4 --shard-idx ${shard}"\
" --max-rounds ${MAX_ROUNDS}"\
" --max-frames-per-round ${MAX_FRAMES}"\
" --host 127.0.0.1 --port ${port}"\
" --tensor-parallel-size 1 --dtype bfloat16 --max-model-len ${MAX_MODEL_LEN}"\
" --gpu-memory-utilization ${GPU_MEM_UTIL}"\
" --start-server --server-log '${server_log}'"\
" --log-jsonl '${LOG_JSONL}' --summary-json '${SUMMARY_JSON}'"\
" --resume-from-log"\
" --use-wandb --wandb-project '${WANDB_PROJECT}' --wandb-entity '${WANDB_ENTITY}'"\
" --wandb-group '${WANDB_GROUP}' --wandb-name '${wandb_name}'"\
") 2>&1 | tee -a '${run_log}'"
}

cmd0="$(mkcmd 0 0 18200)"
cmd1="$(mkcmd 1 1 18201)"
cmd2="$(mkcmd 2 2 18202)"
cmd3="$(mkcmd 3 3 18203)"

tmux new-session -d -s "${SESSION}" -n shard0 "bash -lc \"${cmd0}\""
tmux new-window -t "${SESSION}" -n shard1 "bash -lc \"${cmd1}\""
tmux new-window -t "${SESSION}" -n shard2 "bash -lc \"${cmd2}\""
tmux new-window -t "${SESSION}" -n shard3 "bash -lc \"${cmd3}\""

echo "[tmux] started ${SESSION} (4 windows)."

