# Plug-and-play multi-turn evaluation report (Qwen2.5-VL-7B, vLLM)

## Setup
- Model: `Qwen/Qwen2.5-VL-7B-Instruct`
- Engine: vLLM OpenAI-compatible server (one server per GPU)
- GPUs: 4×A100-80GB (TP=1, 4 shards)
- Prompt protocol: `REVISE` multi-round with `<think>` + `<summary>` + either `<frames>` (request) or `<answer>` (final)
- Plug-and-play settings:
  - `max_rounds=5`
  - `max_frames_per_round=5`
  - candidate frames: `candidate_k=20` (IDs shown to the model; only IDs allowed)
  - dtype: `bfloat16`
  - `max_model_len=12288`
  - `gpu_memory_utilization=0.6`
- Logging:
  - W&B: online (`WANDB_MODE=online`)
  - Per-round prompt logs: `debug_prompt_logs/*.jsonl`

## NExT-QA (real video frames, full splits)

### Validation (4996)
- Merged: `outputs/2026-01-24/nextqa_val_full_pnp_qwen2p5vl7b_ids_tp1/merged.json`
- Accuracy: **0.6239** (3117 / 4996)
- Avg rounds: **4.6241**; avg effective rounds: **3.6743**
- Failed: **2**; invalid-action terminated: **2**

### Test (8564)
- Merged: `outputs/2026-01-24/nextqa_test_full_pnp_qwen2p5vl7b_ids_tp1/merged.json`
- Accuracy: **0.6391** (5473 / 8564)
- Avg rounds: **4.6303**; avg effective rounds: **3.6874**
- Failed: **2**; invalid-action terminated: **2**

## Video-MME (lmms-lab/Video-MME, test split)
- Status: **done**
- Dataset split: `test` (2700)
- Notes:
  - HF dataset provides **YouTube URLs** only; videos are downloaded on-demand and cached locally.
  - Cache dir used: `/tmp/chenwei_video_cache/videomme` (videos under a nested `videomme/` subfolder)
  - yt-dlp config: use Node runtime + `youtube:player_client=android` to avoid common 403/HLS failures.

### Results (raw)
- Merged summary: `outputs/2026-01-24/videomme_test_pnp_qwen2p5vl7b_think_ids_tp1/merged.json`
- Accuracy (includes download failures as wrong): **0.1515** (409 / 2700)
- Failed samples (download/probe): **1696 / 2700**
- Invalid-action terminated: **5**
- W&B runs:
  - `https://wandb.ai/cxu-research/revise_benchmarks/runs/c5zu0vgh` (shard0)
  - `https://wandb.ai/cxu-research/revise_benchmarks/runs/ci42jhzu` (shard1)
  - `https://wandb.ai/cxu-research/revise_benchmarks/runs/v6vgl8oh` (shard2)
  - `https://wandb.ai/cxu-research/revise_benchmarks/runs/1i2lru3a` (shard3)

### Results (post-processed on accessible videos)
- Prompt-log summary: `outputs/2026-01-24/videomme_test_pnp_qwen2p5vl7b_think_ids_tp1/prompt_log_summary.json`
- Nonfailed samples: **1004**; answered (nonfailed): **989**; correct: **406**
- Accuracy on nonfailed: **0.4044** (406 / 1004)
- Accuracy on answered: **0.4105** (406 / 989)
- Avg rounds (answered): **3.0819**; avg model calls (answered): **3.1183**
- Failure buckets (counts):
  - `youtube_auth_required`: 1599
  - `youtube_private`: 54
  - `youtube_unavailable`: 24
  - `yt_dlp_failed_other`: 10
  - `video_probe_failed`: 9

## LVBench / LongVideoBench (lmms-lab/LVBench, train split)
- Status: **done** (real videos from HF `video_chunks`, resized frames)
- Dataset split: `train` (1549)
- Notes:
  - The HF dataset itself is metadata-only; we use the **packaged LVBench videos** in the HF repo under `video_chunks/*.zip`.
  - Videos extracted (103 mp4) to: `/tmp/chenwei_video_cache/lvbench` (via `scripts/fetch_lvbench_videos.py`).
  - Frames are resized (max edge 512) before sending to vLLM to avoid `max_model_len=12288` overflow on some high-res frames.
- W&B runs:
  - `https://wandb.ai/cxu-research/revise_benchmarks/runs/5tkicu8y` (shard0)
  - `https://wandb.ai/cxu-research/revise_benchmarks/runs/7exxn915` (shard1)
  - `https://wandb.ai/cxu-research/revise_benchmarks/runs/xceqmbr8` (shard2)
  - `https://wandb.ai/cxu-research/revise_benchmarks/runs/sj09zbep` (shard3)
- Prompt logs:
  - `debug_prompt_logs/lvbench_train_pnp_qwen2p5vl7b_think_ids_tp1_hfvid_resized.shard0of4.jsonl` … `shard3of4.jsonl`
- Output dir:
  - `outputs/2026-01-24/lvbench_train_pnp_qwen2p5vl7b_think_ids_tp1_hfvid_resized`

### Results
- Merged summary: `outputs/2026-01-24/lvbench_train_pnp_qwen2p5vl7b_think_ids_tp1_hfvid_resized/merged.json`
- Accuracy: **0.2834** (439 / 1549)
- Avg rounds: **3.4687**; avg effective rounds: **2.5332**
- Failed: **0**; invalid-action terminated: **41**
- Prompt log size: **9,116,405 bytes** across **5,578 lines**
