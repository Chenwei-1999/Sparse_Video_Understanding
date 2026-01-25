# NExT-QA: caption vs frames (Qwen2.5-VL-7B)

## Setup
- Model: `Qwen/Qwen2.5-VL-7B-Instruct` (local snapshot)
- Dataset split: NExT-QA `val.csv` (4996 samples)
- Captions: `data/nextqa_allcaps_1fps/<video_id>_cap.json` (1fps)

### Settings compared
1) **Caption + Qwen7B**: caption-only prompt, answer letter only.
2) **Our method (REVISE, frames)**: plug-and-play multi-round frame selection + answer.
3) **Our method + caption**: REVISE with captions injected for (a) shown frames and (b) candidate frame IDs.
4) **Caption-only REVISE (multi-turn)**: REVISE loop without images; per-round observations are 1fps captions.
5) **VideoAgent (caption retrieval, local vLLM)**: VideoAgent-style “query + segment” planning, but retrieval is done over 1fps captions (TF-IDF), and the agent is Qwen2.5-VL-7B via vLLM (no GPT-4, no CLIP features).

## Results (val)
| Setting | Accuracy | Notes / artifacts |
|---|---:|---|
| Caption + Qwen7B | **0.6519** (3257/4996) | `outputs/2026-01-25/nextqa_caption_compare_qwen2p5vl7b/caption_only_val_tp1/merged.json` (avg caption chars ≈ 9720) |
| Our method (REVISE, frames) | **0.6239** (3117/4996) | `outputs/2026-01-24/nextqa_val_full_pnp_qwen2p5vl7b_ids_tp1/merged.json` |
| Our method + caption | **0.6207** (3101/4996) | `outputs/2026-01-25/nextqa_caption_compare_qwen2p5vl7b/revise_plus_caption_val_tp1_ts/merged.json` |
| Caption-only REVISE (multi-turn) | **0.5633** (2814/4996) | `outputs/2026-01-25/nextqa_caption_compare_qwen2p5vl7b/caption_revise_val_tp1_ts/merged.json` (invalid_action_terminated=258) |
| VideoAgent (caption retrieval, local vLLM) | **0.5873** (2934/4996) | `outputs/2026-01-25/nextqa_caption_compare_qwen2p5vl7b/videoagent_caption_val_tp1/merged.json` |

## Commands (reference)
Caption-only (4 shards, TP=1 per GPU):
- `python examples/revise/eval_nextqa_caption_vllm.py --start-server --num-shards 4 --shard-idx {0..3} ...`

REVISE + caption (4 shards, TP=1 per GPU):
- `python examples/revise/plug_and_play_nextqa_vllm.py --start-server --num-shards 4 --shard-idx {0..3} --captions-dir ... --caption-include both ...`

Caption-only REVISE (4 shards, TP=1 per GPU):
- `python examples/revise/plug_and_play_nextqa_vllm.py --start-server --num-shards 4 --shard-idx {0..3} --observation-mode caption --captions-dir ...`

REVISE + caption (tmux helper):
- `bash scripts/run_nextqa_revise_plus_caption_val_tp1.sh`

VideoAgent baseline (tmux helper):
- `bash scripts/run_nextqa_videoagent_caption_val_tp1.sh`
