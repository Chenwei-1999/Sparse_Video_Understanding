#!/usr/bin/env bash
set -euo pipefail

# Plug-and-play eval on lmms-lab/LVBench (train split) using 4 shards (1 GPU each, TP=1).

cd /home/cxk2993/verl

export WANDB_MODE="${WANDB_MODE:-online}"

PY="${PY:-/shares/hlw3876/chenwei/miniconda3/envs/verlrun/bin/python}"
MODEL_PATH="${MODEL_PATH:-/shares/hlw3876/chenwei/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5}"

OUT_DIR="${OUT_DIR:-outputs/2026-01-24/lvbench_train_pnp_qwen2p5vl7b_think_ids_tp1}"
LOG_DIR="${LOG_DIR:-debug_runs}"
PROMPT_LOG_JSONL="${PROMPT_LOG_JSONL:-debug_prompt_logs/lvbench_train_pnp_qwen2p5vl7b_think_ids_tp1.jsonl}"
SUMMARY_JSON="${SUMMARY_JSON:-${OUT_DIR}/summary.json}"
PIDS_FILE="${PIDS_FILE:-${LOG_DIR}/lvbench_train_pnp_qwen2p5vl7b_think_ids_tp1_pids.txt}"

VIDEO_CACHE_DIR="${VIDEO_CACHE_DIR:-/tmp/chenwei_video_cache}"

MAX_ROUNDS="${MAX_ROUNDS:-5}"
MAX_FRAMES_PER_ROUND="${MAX_FRAMES_PER_ROUND:-5}"
MAX_RETRIES_PER_ROUND="${MAX_RETRIES_PER_ROUND:-2}"
CANDIDATE_K="${CANDIDATE_K:-20}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"
YT_DLP_TIMEOUT_S="${YT_DLP_TIMEOUT_S:-600}"

WANDB_PROJECT="${WANDB_PROJECT:-revise_benchmarks}"
WANDB_ENTITY="${WANDB_ENTITY:-cxu-research}"
WANDB_GROUP="${WANDB_GROUP:-lvbench_train_pnp_qwen2p5vl7b_think_ids_tp1}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}" debug_prompt_logs
rm -f "${PIDS_FILE}"

for shard in 0 1 2 3; do
  gpu="${shard}"
  port="$((20000 + shard))"
  log="${LOG_DIR}/lvbench_train_pnp_qwen2p5vl7b_think_ids_tp1_s${shard}.log"
  server_log="${LOG_DIR}/vllm_server_lvbench_train_pnp_qwen2p5vl7b_think_ids_tp1_s${shard}.log"
  wandb_name="pnp_qwen2p5vl7b_lvbench_train_think_ids_tp1_s${shard}"

  cmd=(
    "${PY}"
    examples/revise/plug_and_play_videomme_lvbench_vllm.py
    --dataset lvbench
    --split train
    --video-cache-dir "${VIDEO_CACHE_DIR}"
    --max-samples 0
    --model-path "${MODEL_PATH}"
    --host 127.0.0.1
    --port "${port}"
    --tensor-parallel-size 1
    --dtype bfloat16
    --max-model-len "${MAX_MODEL_LEN}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-rounds "${MAX_ROUNDS}"
    --max-frames-per-round "${MAX_FRAMES_PER_ROUND}"
    --candidate-k "${CANDIDATE_K}"
    --use-candidate-frame-ids
    --require-candidate-frames
    --max-retries-per-round "${MAX_RETRIES_PER_ROUND}"
    --timeout-s 120
    --force-final-answer
    --num-shards 4
    --shard-idx "${shard}"
    --start-server
    --restart-server-on-failure
    --server-log "${server_log}"
    --log-jsonl "${PROMPT_LOG_JSONL}"
    --summary-json "${SUMMARY_JSON}"
    --yt-dlp-timeout-s "${YT_DLP_TIMEOUT_S}"
    --use-wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-entity "${WANDB_ENTITY}"
    --wandb-group "${WANDB_GROUP}"
    --wandb-name "${wandb_name}"
  )

  echo "[lvbench] launch shard ${shard} gpu ${gpu} port ${port}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" > "${log}" 2>&1 &
  echo $! >> "${PIDS_FILE}"
  sleep 0.2
done

echo "[lvbench] pids:"
cat "${PIDS_FILE}"

echo "[lvbench] waiting for all shards..."
wait
echo "[lvbench] all shards finished."
