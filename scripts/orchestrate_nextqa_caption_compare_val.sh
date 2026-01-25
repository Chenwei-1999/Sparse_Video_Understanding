#!/usr/bin/env bash
set -euo pipefail

# Orchestrate NExT-QA val runs for caption/frames comparisons.
# - Wait for the *currently running* caption-only REVISE run to finish (summary.shard*of4.json present).
# - Merge caption-only REVISE summaries.
# - Launch REVISE + caption (frames+captions) and wait/merge.
# - Launch VideoAgent baseline (caption retrieval) and wait/merge.
#
# This script is safe to start while the caption-only REVISE run is still in progress.
#
# Usage (recommended):
#   tmux new-session -d -s nextqa_caption_compare_orchestrator "bash scripts/orchestrate_nextqa_caption_compare_val.sh"
#
# Environment overrides:
#   DATE=2026-01-25

cd /home/cxk2993/verl

DATE="${DATE:-$(date +%F)}"
export DATE

CAPTION_REVISE_DIR="outputs/2026-01-25/nextqa_caption_compare_qwen2p5vl7b/caption_revise_val_tp1_ts"
REVISE_PLUS_CAPTION_DIR="outputs/${DATE}/nextqa_caption_compare_qwen2p5vl7b/revise_plus_caption_val_tp1_ts"
VIDEOAGENT_DIR="outputs/${DATE}/nextqa_caption_compare_qwen2p5vl7b/videoagent_caption_val_tp1"

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
      samples="$(/shares/hlw3876/chenwei/miniconda3/envs/verlrun/bin/python - <<PY\nimport json\np='${f}'\ntry:\n  j=json.load(open(p,'r'))\n  print(int(j.get('results',{}).get('samples',-1)))\nexcept Exception:\n  print(-1)\nPY\n)"
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
  /shares/hlw3876/chenwei/miniconda3/envs/verlrun/bin/python scripts/merge_pnp_summaries.py \
    --glob "${dir}/summary.shard*of4.json" \
    -o "${out}"
}

echo "[orchestrate] start at $(date) DATE=${DATE}"

wait_for_summaries "${CAPTION_REVISE_DIR}" 1249
merge_summaries "${CAPTION_REVISE_DIR}"

echo "[orchestrate] launching REVISE+caption at $(date)"
bash scripts/run_nextqa_revise_plus_caption_val_tp1.sh
wait_for_summaries "${REVISE_PLUS_CAPTION_DIR}" 1249
merge_summaries "${REVISE_PLUS_CAPTION_DIR}"

echo "[orchestrate] launching VideoAgent baseline at $(date)"
bash scripts/run_nextqa_videoagent_caption_val_tp1.sh
wait_for_summaries "${VIDEOAGENT_DIR}" 1249
merge_summaries "${VIDEOAGENT_DIR}"

echo "[orchestrate] all done at $(date)"
echo "[orchestrate] merged:"
echo "- ${CAPTION_REVISE_DIR}/merged.json"
echo "- ${REVISE_PLUS_CAPTION_DIR}/merged.json"
echo "- ${VIDEOAGENT_DIR}/merged.json"
