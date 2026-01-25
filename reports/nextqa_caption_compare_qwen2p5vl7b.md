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

## Results (val)
| Setting | Accuracy | Notes / artifacts |
|---|---:|---|
| Caption + Qwen7B | **0.6519** (3257/4996) | `outputs/2026-01-25/nextqa_caption_compare_qwen2p5vl7b/caption_only_val_tp1/merged.json` (avg caption chars ≈ 9720) |
| Our method (REVISE, frames) | **0.6239** (3117/4996) | `outputs/2026-01-24/nextqa_val_full_pnp_qwen2p5vl7b_ids_tp1/merged.json` |
| Our method + caption | **INCOMPLETE (needs rerun)** | Previous run stopped early (~200/4996). Root cause was caption index mismatch (captions are 1fps seconds, videos are ~25–30fps). Fixed by mapping `caption_idx=floor(frame_idx/fps)` for frame-indexed runs. |
| Caption-only REVISE (multi-turn) | **TBD** | Implemented via `examples/revise/plug_and_play_nextqa_vllm.py --observation-mode caption` (no images, captions shown per round). |

## Commands (reference)
Caption-only (4 shards, TP=1 per GPU):
- `python examples/revise/eval_nextqa_caption_vllm.py --start-server --num-shards 4 --shard-idx {0..3} ...`

REVISE + caption (4 shards, TP=1 per GPU):
- `python examples/revise/plug_and_play_nextqa_vllm.py --start-server --num-shards 4 --shard-idx {0..3} --captions-dir ... --caption-include both ...`

Caption-only REVISE (4 shards, TP=1 per GPU):
- `python examples/revise/plug_and_play_nextqa_vllm.py --start-server --num-shards 4 --shard-idx {0..3} --observation-mode caption --captions-dir ...`
