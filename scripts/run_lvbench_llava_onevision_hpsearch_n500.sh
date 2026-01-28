#!/usr/bin/env bash
set -euo pipefail

# Grid search (max_rounds, max_frames_per_round) for REVISE plug-and-play on LVBench.
# Runs 4 shards / 4 GPUs per setting with LLaVA-OneVision (Qwen2-7B backbone).
#
# Outputs:
#   outputs/YYYY-MM-DD/lvbench_train_rev_llava_onevision_hpsearch_n500/<setting>/merged.json
#
# Notes:
# - Uses the existing vLLM-based runner: examples/revise/plug_and_play_videomme_lvbench_vllm.py
# - Starts/stops vLLM servers per shard per setting (robust for non-interactive shells).

DATE="${DATE:-$(date +%F)}"
ROOT_OUT="outputs/${DATE}/lvbench_train_rev_llava_onevision_hpsearch_n500"
mkdir -p "${ROOT_OUT}"

PY="${PY:-/shares/hlw3876/chenwei/miniconda3/envs/verlrun/bin/python}"
MODEL_PATH="${MODEL_PATH:-/shares/hlw3876/chenwei/hf_cache/models--llava-hf--llava-onevision-qwen2-7b-ov-hf/snapshots/0d50680527681998e456c7b78950205bedd8a068}"
VIDEO_CACHE_DIR="${VIDEO_CACHE_DIR:-/tmp/chenwei_video_cache}"

# W&B (optional)
WANDB_PROJECT="${WANDB_PROJECT:-revise_benchmarks}"
WANDB_ENTITY="${WANDB_ENTITY:-cxu-research}"
export WANDB_MODE="${WANDB_MODE:-online}"

# Hyperparameter grid (edit as needed).
# Format: NAME:R:F
GRID=(
  "R1_F10:1:10"
  "R1_F7:1:7"
  "R2_F7:2:7"
  "R4_F7:4:7"
  "R6_F4:6:4"
  "R6_F2:6:2"
)

echo "ROOT_OUT=${ROOT_OUT}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "PY=${PY}"
echo "VIDEO_CACHE_DIR=${VIDEO_CACHE_DIR}"

idx=0
for entry in "${GRID[@]}"; do
  idx=$((idx+1))
  name="${entry%%:*}"
  rest="${entry#*:}"
  max_rounds="${rest%%:*}"
  max_frames="${rest#*:}"

  OUT_DIR="${ROOT_OUT}/${name}"
  mkdir -p "${OUT_DIR}"

  echo ""
  echo "=== [${idx}/${#GRID[@]}] ${name} (max_rounds=${max_rounds}, max_frames_per_round=${max_frames}) ==="

  # Use distinct ports per setting to avoid TIME_WAIT edge cases.
  port_base=$((27000 + idx * 10))
  rm -f "${OUT_DIR}/pids.txt"

  for shard in 0 1 2 3; do
    port=$((port_base + shard))
    log="${OUT_DIR}/run.shard${shard}of4.log"
    server_log="${OUT_DIR}/server.shard${shard}of4.log"
    wandb_group="lvbench_train_llava_onevision_hpsearch_n500_${name}"
    wandb_name="lvbench_${name}_s${shard}"

    echo "[launch] shard=${shard} gpu=${shard} port=${port}"
    CUDA_VISIBLE_DEVICES="${shard}" "${PY}" examples/revise/plug_and_play_videomme_lvbench_vllm.py \
      --dataset lvbench \
      --split train \
      --video-cache-dir "${VIDEO_CACHE_DIR}" \
      --max-samples 500 \
      --model-path "${MODEL_PATH}" \
      --host 127.0.0.1 \
      --port "${port}" \
      --tensor-parallel-size 1 \
      --dtype bfloat16 \
      --max-model-len 32768 \
      --gpu-memory-utilization 0.6 \
      --max-rounds "${max_rounds}" \
      --max-frames-per-round "${max_frames}" \
      --candidate-k 20 \
      --use-candidate-frame-ids \
      --require-candidate-frames \
      --max-retries-per-round 2 \
      --timeout-s 180 \
      --force-final-answer \
      --num-shards 4 \
      --shard-idx "${shard}" \
      --start-server \
      --restart-server-on-failure \
      --server-log "${server_log}" \
      --log-jsonl "${OUT_DIR}/log.jsonl" \
      --summary-json "${OUT_DIR}/summary.json" \
      --resume-from-log \
      --yt-dlp-timeout-s 600 \
      --use-wandb \
      --wandb-project "${WANDB_PROJECT}" \
      --wandb-entity "${WANDB_ENTITY}" \
      --wandb-group "${wandb_group}" \
      --wandb-name "${wandb_name}" \
      >"${log}" 2>&1 &

    echo $! >> "${OUT_DIR}/pids.txt"
    sleep 0.2
  done

  echo "[pids]"; cat "${OUT_DIR}/pids.txt"
  wait

  "${PY}" scripts/merge_pnp_summaries.py --glob "${OUT_DIR}/summary.shard*of4.json" -o "${OUT_DIR}/merged.json"
  "${PY}" - <<PY
import json
from pathlib import Path
p=Path("${OUT_DIR}/merged.json")
d=json.load(p.open())
m=d["merged_results"]
print("merged:", p)
print("acc", m.get("accuracy"), "acc_nonfailed", m.get("accuracy_nonfailed"))
print("avg_rounds", m.get("avg_rounds"), "avg_effective_rounds", m.get("avg_effective_rounds"))
print("failed", m.get("failed"), "invalid_outputs", m.get("invalid_outputs"))
PY
done

echo ""
echo "=== Done. Best setting summary ==="
"${PY}" - <<PY
import json
from pathlib import Path
root=Path("${ROOT_OUT}")
best=None
rows=[]
for cfg in sorted([p for p in root.iterdir() if p.is_dir()]):
    mpath=cfg/"merged.json"
    if not mpath.exists():
        continue
    d=json.load(mpath.open())
    m=d["merged_results"]
    acc=float(m.get("accuracy_nonfailed") or m.get("accuracy") or 0.0)
    rows.append((acc, cfg.name, m.get("accuracy"), m.get("accuracy_nonfailed"), m.get("avg_rounds"), m.get("failed")))
rows.sort(reverse=True)
for acc, name, a, an, ar, failed in rows:
    print(f"{name}: acc={a} acc_nonfailed={an} avg_rounds={ar} failed={failed}")
if rows:
    print("BEST:", rows[0][1])
PY
