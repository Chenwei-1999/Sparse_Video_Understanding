# REVISE: Sparse Video Understanding (NExT-QA)

This directory provides a REVISE-style multi-round agent loop, dataset loader, and configs for NExT-QA experiments.

## Data
- NExT-QA root: `/shares/hlw3876/chenwei/NExT-QA`
- Videos: `/shares/hlw3876/chenwei/NExT-QA/NExTVideo`
- Mapping file: `/shares/hlw3876/chenwei/NExT-QA/map_vid_vidorID.json`
- CSVs: `/shares/hlw3876/chenwei/NExT-QA/nextqa/train.csv` and `val.csv`

These are already referenced in the config files under `examples/revise/config`.

## Plug-and-play evaluation (REVISE multi-round)

```bash
ENGINE=sglang ./examples/revise/run_revise_nextqa_eval.sh
```

vLLM backend:

```bash
ENGINE=vllm ./examples/revise/run_revise_nextqa_eval.sh \
  --config-name revise_nextqa_eval_vllm
```

Notes:
- Uses `Qwen/Qwen2.5-VL-7B-Instruct` by default.
- Settings: `max_frames_per_round=3`, `max_rounds=4`, `temperature=0.2`, `top_p=0.9`, `max_response_length=256`.

## Reinforcement fine-tuning (GRPO + EAGER-style reward)

```bash
ENGINE=sglang ./examples/revise/run_revise_nextqa_grpo.sh
```

## Minimal smoke test (4 GPUs, 16 samples, 2 rounds)

```bash
ENGINE=sglang ./examples/revise/run_revise_nextqa_smoke.sh
```

This uses 4 GPUs (CUDA_VISIBLE_DEVICES defaults to 0,1,2,3), `max_samples=16`, and short round/length settings to
validate the loop quickly. Set `ENGINE=vllm` to smoke-test vLLM.

Default training settings follow the paper:
- `lr=1e-6`, `kl_loss_coef=0.001`, `entropy_coeff=0`.
- `max_prompt_length=8192`, `max_response_length=512`, `train_batch_size=8`.
- `max_frames_per_round=3`, `max_rounds=4`.

## Customization
- Change VLM backbone via `actor_rollout_ref.model.path=...`.
- Override dataset paths or batch sizes via CLI flags.
- REVISE-specific settings live under `actor_rollout_ref.rollout.revise`.

## Notes on EAGER reward
The included `eager_videoqa` reward approximates EAGER when full margin signals are not available. If you compute per-round
confidence gains (margins) externally, pass them via `extra_info['revise']` to fully match the paper.
