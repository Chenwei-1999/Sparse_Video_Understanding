#!/usr/bin/env bash
set -euo pipefail

# Start NExT-QA test-set plug-and-play eval (vLLM) after an existing val run finishes.
#
# Assumptions:
# - A val run is already running and recorded its PIDs to:
#     debug_runs/nextqa_val_full_ids_tp1_pids.txt
# - The verlrun environment is available at:
#     /shares/hlw3876/chenwei/miniconda3/envs/verlrun
# - Uses 4 shards, one GPU per shard (TP=1).

cd /home/cxk2993/verl

export WANDB_MODE="${WANDB_MODE:-online}"

PY="${PY:-/shares/hlw3876/chenwei/miniconda3/envs/verlrun/bin/python}"
MODEL_PATH="${MODEL_PATH:-/shares/hlw3876/chenwei/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5}"

VIDEO_ROOT="${VIDEO_ROOT:-/shares/hlw3876/chenwei/NExT-QA/NExTVideo}"
MAP_JSON="${MAP_JSON:-/shares/hlw3876/chenwei/NExT-QA/map_vid_vidorID.json}"
VAL_PIDS_FILE="${VAL_PIDS_FILE:-debug_runs/nextqa_val_full_ids_tp1_pids.txt}"
TEST_CSV="${TEST_CSV:-/shares/hlw3876/chenwei/NExT-QA/test-data-nextqa/test.csv}"

TEST_PIDS_FILE="${TEST_PIDS_FILE:-debug_runs/nextqa_test_full_ids_tp1_pids.txt}"
TEST_LOG_JSONL="${TEST_LOG_JSONL:-debug_prompt_logs/nextqa_test_full_qwen2p5vl7b_ids_tp1.jsonl}"
TEST_SUMMARY_JSON="${TEST_SUMMARY_JSON:-outputs/2026-01-24/nextqa_test_full_pnp_qwen2p5vl7b_ids_tp1/summary.json}"

MAX_ROUNDS="${MAX_ROUNDS:-5}"
MAX_FRAMES_PER_ROUND="${MAX_FRAMES_PER_ROUND:-5}"
MAX_RETRIES_PER_ROUND="${MAX_RETRIES_PER_ROUND:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"

WANDB_PROJECT="${WANDB_PROJECT:-revise_nextqa}"
WANDB_ENTITY="${WANDB_ENTITY:-cxu-research}"
WANDB_GROUP="${WANDB_GROUP:-nextqa_test_full_ids_tp1}"

if [[ ! -f "${VAL_PIDS_FILE}" ]]; then
  echo "[chain] Missing ${VAL_PIDS_FILE}; nothing to wait for."
  exit 1
fi

VAL_PIDS="$(tr '\n' ' ' < "${VAL_PIDS_FILE}")"
echo "[chain] Waiting for val PIDs: ${VAL_PIDS}"

while true; do
  any_alive=0
  for pid in ${VAL_PIDS}; do
    if kill -0 "${pid}" 2>/dev/null; then
      any_alive=1
      break
    fi
  done
  if [[ "${any_alive}" -eq 0 ]]; then
    break
  fi
  sleep 60
  echo "[chain] val still running at $(date)"
done

echo "[chain] val finished at $(date); launching test on 4 GPUs"

rm -f "${TEST_PIDS_FILE}"

mkdir -p debug_runs

for shard in 0 1 2 3; do
  gpu="${shard}"
  port="$((18000 + shard))"
  log="debug_runs/nextqa_test_full_ids_tp1_s${shard}.log"
  server_log="debug_runs/vllm_server_nextqa_test_full_ids_tp1_s${shard}.log"
  wandb_name="pnp_qwen2p5vl7b_nextqa_test_full_ids_tp1_s${shard}"

  cmd=(
    "${PY}"
    examples/revise/plug_and_play_nextqa_vllm.py
    --model-path "${MODEL_PATH}"
    --video-root "${VIDEO_ROOT}"
    --map-json "${MAP_JSON}"
    --csv "${TEST_CSV}"
    --max-samples 0
    --num-shards 4
    --shard-idx "${shard}"
    --max-rounds "${MAX_ROUNDS}"
    --max-frames-per-round "${MAX_FRAMES_PER_ROUND}"
    --max-retries-per-round "${MAX_RETRIES_PER_ROUND}"
    --hide-seen-frames-in-prompt
    --use-candidate-frames
    --use-candidate-frame-ids
    --require-candidate-frames
    --host 127.0.0.1
    --port "${port}"
    --tensor-parallel-size 1
    --dtype bfloat16
    --max-model-len "${MAX_MODEL_LEN}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --start-server
    --restart-server-on-failure
    --server-log "${server_log}"
    --log-jsonl "${TEST_LOG_JSONL}"
    --summary-json "${TEST_SUMMARY_JSON}"
    --force-final-answer
    --use-wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-entity "${WANDB_ENTITY}"
    --wandb-group "${WANDB_GROUP}"
    --wandb-name "${wandb_name}"
  )

  echo "[chain] launch test shard ${shard} gpu ${gpu} port ${port}"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup "${cmd[@]}" > "${log}" 2>&1 &
  echo $! >> "${TEST_PIDS_FILE}"
  sleep 0.2
done

echo "[chain] test pids:"
cat "${TEST_PIDS_FILE}"
