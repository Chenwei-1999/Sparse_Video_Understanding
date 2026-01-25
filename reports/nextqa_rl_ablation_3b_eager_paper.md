# NExT-QA RL Ablations (Qwen2.5-VL-3B, REVISE + EAGER)

This report summarizes RL ablations on NExT-QA using the EAGER reward:

\[
r_t=\lambda_1 r^{conf}_t + \lambda_2 r^{sum}_t + \lambda_3 r^{stop}_t + r^{format}_t,
\qquad
R(H)=\sum_{t=1}^{\tau}\gamma^{t-1}r_t.
\]

All evaluations below use the **plug-and-play** REVISE agent with:
- `max_rounds=4`, `max_frames_per_round=3`, `use_candidate_frames=true`, `use_candidate_frame_ids=true`
- `tensor_parallel_size=4`, `max_model_len=12288`, `dtype=bfloat16`

## Results (val subset)

| Model | RL steps | (λ_conf, λ_sum, λ_stop) | Acc | Avg rounds | Invalid terminations | Prompt log (lines / bytes) |
|---|---:|---:|---:|---:|---:|---:|
| EAGER-paper | 30 | (1.0, 1.0, 0.5) | 0.444 | 2.778 | 95 | 1399 / 8120308 |
| EAGER-paper | 30 | (0.0, 1.0, 0.5) | 0.464 | 2.492 | 90 | 1249 / 7227792 |
| EAGER-paper | 30 | (1.0, 0.0, 0.5) | 0.448 | 2.620 | 105 | 1321 / 7648208 |
| EAGER-paper | 30 | (1.0, 1.0, 0.0) | 0.398 | 3.032 | 118 | 1536 / 8942809 |
| EAGER-paper | 30 | (2.0, 1.0, 0.5) | 0.408 | 2.948 | 108 | 1483 / 8625638 |
| EAGER-paper | 30 | (1.0, 2.0, 0.5) | 0.422 | 2.790 | 102 | 1406 / 8156129 |
| Base (no RL) | 0 | n/a | 0.316 | 2.860 | 172 | 1454 / 8462211 |

## Takeaways (from this sweep)

- **EAGER-style RL helps** under the strict plug-and-play eval: base is **0.316**, best RL run reaches **0.464**.
- **λ_conf appears harmful/noisy** here: increasing it to 2.0 drops accuracy to **0.408**; setting it to 0.0 gives the best result (**0.464**).
- **λ_stop matters**: removing stop reward increases rounds and decreases accuracy (**0.398**, avg_rounds **3.03**).
- **λ_sum alone isn’t sufficient**: removing it slightly reduces performance and increases invalid terminations (**0.448**, invalid **105**).

## 100-step follow-up (best λ)

We ran a longer training for the best λ from the sweep (λ_conf=0, λ_sum=1, λ_stop=0.5):

| Model | RL steps | (λ_conf, λ_sum, λ_stop) | Acc | Avg rounds | Invalid terminations | Prompt log (lines / bytes) |
|---|---:|---:|---:|---:|---:|---:|
| EAGER-paper (best λ) | 100 | (0.0, 1.0, 0.5) | 0.498 | 2.830 | 57 | 1429 / 8292692 |

## Checkpoints

- (1.0, 1.0, 0.5): `outputs/2026-01-24/rl_ablation_nextqa_3b_eager_paper/qwen2p5vl3b_nextqa_eagerPaper_c1.0_s1.0_st0.5/global_step_30/actor/huggingface`
- (0.0, 1.0, 0.5): `outputs/2026-01-24/rl_ablation_nextqa_3b_eager_paper/qwen2p5vl3b_nextqa_eagerPaper_c0.0_s1.0_st0.5/global_step_30/actor/huggingface`
- (1.0, 0.0, 0.5): `outputs/2026-01-24/rl_ablation_nextqa_3b_eager_paper/qwen2p5vl3b_nextqa_eagerPaper_c1.0_s0.0_st0.5/global_step_30/actor/huggingface`
- (1.0, 1.0, 0.0): `outputs/2026-01-24/rl_ablation_nextqa_3b_eager_paper/qwen2p5vl3b_nextqa_eagerPaper_c1.0_s1.0_st0.0/global_step_30/actor/huggingface`
- (2.0, 1.0, 0.5): `outputs/2026-01-24/rl_ablation_nextqa_3b_eager_paper/qwen2p5vl3b_nextqa_eagerPaper_c2.0_s1.0_st0.5/global_step_30/actor/huggingface`
- (1.0, 2.0, 0.5): `outputs/2026-01-24/rl_ablation_nextqa_3b_eager_paper/qwen2p5vl3b_nextqa_eagerPaper_c1.0_s2.0_st0.5/global_step_30/actor/huggingface`
- Best λ @ step100: `outputs/2026-01-24/rl_ablation_nextqa_3b_eager_paper_100/qwen2p5vl3b_nextqa_eagerPaper_c0.0_s1.0_st0.5/global_step_100/actor/huggingface`
