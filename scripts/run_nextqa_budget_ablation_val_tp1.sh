#!/usr/bin/env bash
set -euo pipefail

# Ablation study on NExT-QA val (4996) with Qwen2.5-VL-7B:
# Under a fixed TOTAL frame budget (R * F constant), compare:
#   - more rounds with fewer frames/round
#   - fewer rounds with more frames/round
#
# This runs sequentially (one config at a time) but uses 4 GPUs via 4 shards (TP=1 per GPU).
#
# Recommended usage (survives disconnects):
#   tmux new-session -d -s nextqa_budget_ablation "bash -lc 'DATE=2026-01-25 bash scripts/run_nextqa_budget_ablation_val_tp1.sh'"
#
# Environment overrides:
#   DATE=...
#   PY=...
#   MODEL_PATH=...
#   CSV=...
#   VIDEO_ROOT=...
#   MAP_JSON=...
#   OUT_BASE=...
#   TOTAL_BUDGET_FRAMES=20
#   CANDIDATE_K=20
#   MAX_RETRIES=2
#   MAX_MODEL_LEN=12288
#   GPU_MEM_UTIL=0.6
#   WANDB_PROJECT=revise_nextqa
#   WANDB_ENTITY=cxu-research
#   WANDB_GROUP=nextqa_val_budget_ablation

cd /home/cxk2993/verl

export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-online}"

DATE="${DATE:-$(date +%F)}"

PY="${PY:-/shares/hlw3876/chenwei/miniconda3/envs/verlrun/bin/python}"
MODEL_PATH="${MODEL_PATH:-/shares/hlw3876/chenwei/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5}"

VIDEO_ROOT="${VIDEO_ROOT:-/shares/hlw3876/chenwei/NExT-QA/NExTVideo}"
MAP_JSON="${MAP_JSON:-/shares/hlw3876/chenwei/NExT-QA/map_vid_vidorID.json}"
CSV="${CSV:-/shares/hlw3876/chenwei/NExT-QA/nextqa/val.csv}"

OUT_BASE="${OUT_BASE:-outputs/${DATE}/nextqa_budget_ablation_qwen2p5vl7b}"
mkdir -p "${OUT_BASE}"

TOTAL_BUDGET_FRAMES="${TOTAL_BUDGET_FRAMES:-20}"
CANDIDATE_K="${CANDIDATE_K:-20}"
MAX_RETRIES="${MAX_RETRIES:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.6}"

WANDB_PROJECT="${WANDB_PROJECT:-revise_nextqa}"
WANDB_ENTITY="${WANDB_ENTITY:-cxu-research}"
WANDB_GROUP="${WANDB_GROUP:-nextqa_val_budget_ablation}"

EXPECTED_SAMPLES_PER_SHARD=1249

configs=(
  # name:max_rounds:max_frames_per_round
  "more_rounds_fewer_frames:10:2"
  "fewer_rounds_more_frames:4:5"
)

wait_for_summaries() {
  local dir="$1"
  local expected="$2"
  echo "[wait] ${dir} expecting samples=${expected} per shard"
  while true; do
    ok=1
    for shard in 0 1 2 3; do
      f="${dir}/summary.shard${shard}of4.json"
      if [[ ! -f "${f}" ]]; then
        ok=0
        break
      fi
      samples="$(${PY} - <<PY
import json
p='${f}'
try:
  j=json.load(open(p,'r'))
  print(int(j.get('results',{}).get('samples',-1)))
except Exception:
  print(-1)
PY
)"
      if [[ "${samples}" != "${expected}" ]]; then
        ok=0
        break
      fi
    done
    if [[ "${ok}" -eq 1 ]]; then
      echo "[wait] done: ${dir}"
      return 0
    fi
    sleep 300
    echo "[wait] still waiting at $(date)"
  done
}

merge_summaries() {
  local dir="$1"
  local out="${dir}/merged.json"
  echo "[merge] ${dir} -> ${out}"
  ${PY} scripts/merge_pnp_summaries.py --glob "${dir}/summary.shard*of4.json" -o "${out}"
}

run_config_tmux() {
  local name="$1"
  local max_rounds="$2"
  local max_frames="$3"
  local cfg_idx="$4"

  local out_dir="${OUT_BASE}/${name}_R${max_rounds}_F${max_frames}_B${TOTAL_BUDGET_FRAMES}"
  mkdir -p "${out_dir}"

  local session="nextqa_budget_${name}_R${max_rounds}_F${max_frames}"
  tmux kill-session -t "${session}" 2>/dev/null || true

  local log_jsonl="${out_dir}/log.jsonl"
  local summary_json="${out_dir}/summary.json"

  mkcmd() {
    local shard="$1"
    local gpu="$2"
    local port="$3"
    local run_log="${out_dir}/run.shard${shard}of4.log"
    local server_log="${out_dir}/server.shard${shard}of4.log"
    local wandb_name="${name}_R${max_rounds}_F${max_frames}_s${shard}"
    local wandb_group="${WANDB_GROUP}_${name}_R${max_rounds}_F${max_frames}_B${TOTAL_BUDGET_FRAMES}"

    printf "%s" \
      "cd /home/cxk2993/verl && export CUDA_VISIBLE_DEVICES=${gpu} WANDB_MODE=online && ("\
" ${PY} examples/revise/plug_and_play_nextqa_vllm.py"\
" --model-path '${MODEL_PATH}'"\
" --video-root '${VIDEO_ROOT}'"\
" --map-json '${MAP_JSON}'"\
" --csv '${CSV}'"\
" --max-samples 0"\
" --num-shards 4 --shard-idx ${shard}"\
" --max-rounds ${max_rounds}"\
" --max-frames-per-round ${max_frames}"\
" --use-candidate-frames --candidate-k ${CANDIDATE_K}"\
" --use-candidate-frame-ids --require-candidate-frames"\
" --hide-seen-frames-in-prompt"\
" --max-retries-per-round ${MAX_RETRIES}"\
" --answer-only-final-round --force-final-answer"\
" --host 127.0.0.1 --port ${port}"\
" --tensor-parallel-size 1 --dtype bfloat16 --max-model-len ${MAX_MODEL_LEN}"\
" --gpu-memory-utilization ${GPU_MEM_UTIL}"\
" --start-server --restart-server-on-failure --server-log '${server_log}'"\
" --log-jsonl '${log_jsonl}' --summary-json '${summary_json}'"\
" --resume-from-log"\
" --use-wandb --wandb-project '${WANDB_PROJECT}' --wandb-entity '${WANDB_ENTITY}'"\
" --wandb-group '${wandb_group}' --wandb-name '${wandb_name}'"\
") 2>&1 | tee -a '${run_log}'"
  }

  local base_port=$((18500 + cfg_idx * 100))
  local cmd0 cmd1 cmd2 cmd3
  cmd0="$(mkcmd 0 0 $((base_port + 0)))"
  cmd1="$(mkcmd 1 1 $((base_port + 1)))"
  cmd2="$(mkcmd 2 2 $((base_port + 2)))"
  cmd3="$(mkcmd 3 3 $((base_port + 3)))"

  tmux new-session -d -s "${session}" -n shard0 "bash -lc \"${cmd0}\""
  tmux new-window -t "${session}" -n shard1 "bash -lc \"${cmd1}\""
  tmux new-window -t "${session}" -n shard2 "bash -lc \"${cmd2}\""
  tmux new-window -t "${session}" -n shard3 "bash -lc \"${cmd3}\""

  echo "[tmux] started ${session} -> ${out_dir}"

  wait_for_summaries "${out_dir}" "${EXPECTED_SAMPLES_PER_SHARD}"
  merge_summaries "${out_dir}"
}

echo "[ablation] start at $(date) DATE=${DATE} budget_frames=${TOTAL_BUDGET_FRAMES}"

idx=0
for spec in "${configs[@]}"; do
  IFS=":" read -r name max_rounds max_frames <<<"${spec}"
  budget=$((max_rounds * max_frames))
  if [[ "${budget}" -ne "${TOTAL_BUDGET_FRAMES}" ]]; then
    echo "[ablation] ERROR: ${name} has budget=${budget} != TOTAL_BUDGET_FRAMES=${TOTAL_BUDGET_FRAMES}" >&2
    exit 1
  fi
  echo "[ablation] running ${name} (R=${max_rounds}, F=${max_frames}, budget=${budget})"
  run_config_tmux "${name}" "${max_rounds}" "${max_frames}" "${idx}"
  idx=$((idx + 1))
done

echo "[ablation] all configs finished at $(date)"

# Write a short markdown summary.
REPORT_PATH="reports/nextqa_budget_ablation_qwen2p5vl7b.md"
export CONFIG_SPECS="$(printf '%s\n' "${configs[@]}")"
${PY} - <<PY
import json
import os
from pathlib import Path

out_base = Path("${OUT_BASE}")
specs = [s for s in os.environ.get("CONFIG_SPECS", "").splitlines() if s.strip()]
rows = []
for spec in specs:
    name, r, f = spec.split(":")
    merged = out_base / f"{name}_R{r}_F{f}_B${TOTAL_BUDGET_FRAMES}" / "merged.json"
    d = json.load(open(merged))
    m = d["merged_results"]
    rows.append({
        "name": name,
        "rounds": int(r),
        "frames": int(f),
        "acc": m.get("accuracy"),
        "avg_rounds": m.get("avg_rounds"),
        "avg_frames_used": m.get("avg_frames_used"),
        "calls": m.get("total_model_calls"),
        "prompt_bytes": m.get("prompt_log_bytes"),
        "merged": str(merged),
    })

lines = []
lines.append("# NExT-QA budget ablation (Qwen2.5-VL-7B)")
lines.append("")
lines.append(f"- Fixed total frame budget: **{int(${TOTAL_BUDGET_FRAMES})}** (= max_rounds × max_frames_per_round)")
lines.append("- Policy: REVISE plug-and-play (frames), candidate IDs, answer only in final round.")
lines.append("")
lines.append("| Setting | max_rounds | max_frames/round | Accuracy | avg_rounds | avg_frames_used | total_calls | prompt_log_bytes | Artifacts |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
for r in rows:
    lines.append(
        f"| {r['name']} | {r['rounds']} | {r['frames']} | {r['acc']:.4f} | {r['avg_rounds']:.3f} | {r['avg_frames_used']:.2f} | {r['calls']} | {r['prompt_bytes']} | `{r['merged']}` |"
    )

Path("${REPORT_PATH}").write_text("\\n".join(lines) + "\\n")
print(f\"Wrote {REPORT_PATH}\")
PY

echo "[ablation] report: ${REPORT_PATH}"
