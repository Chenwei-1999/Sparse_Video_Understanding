# Notes: RL Ablations (EAGER reward) on NExT-QA (3B VLM)

## Reward definition (paper)
- Per-step: `r_t = λ1 r_conf_t + λ2 r_sum_t + λ3 r_stop_t + r_format_t`
- Return: `R(H) = Σ γ^{t-1} r_t`
- Margin: `m_t = log p(y* | p_t, z_t, S_t) - max_{y≠y*} log p(y | p_t, z_t, S_t)`
- Conf gain (on Select): `r_conf_t = [m_t - m_{t-1}]_+`
- Summary sufficiency (at Answer): `r_sum_t = 1[argmax_y p(y | p_t, z_t) = y*]` (summary-only)
- No frame-level annotations required; only answer labels + model scores.

## Implementation mapping (this repo)
- `ReviseAgentLoop` computes:
  - `margins` list via vLLM logprobs over answer letters at each decision state
  - `summary_only_correct` via a summary-only scoring call at answer time
  - `actions` and `format_by_round` aligned with decision steps
- Reward function `eager_videoqa_paper.compute_score` consumes these signals and returns discounted return.

## Ablation grid (λ_conf, λ_sum, λ_stop)
- baseline: (1.0, 1.0, 0.5)
- no conf:  (0.0, 1.0, 0.5)
- no sum:   (1.0, 0.0, 0.5)
- no stop:  (1.0, 1.0, 0.0)
- high conf: (2.0, 1.0, 0.5)
- high sum:  (1.0, 2.0, 0.5)

## Interim results (val subset, 500 samples, strict eval)
- Eval settings: `max_rounds=4`, `max_frames_per_round=3`, `max_retries_per_round=0`, candidate frames + candidate IDs enabled.
- Base Qwen2.5-VL-3B-Instruct (no RL): acc=0.316, avg_rounds=2.860, invalid_term=172
- baseline (1.0, 1.0, 0.5): acc=0.444, avg_rounds=2.778, invalid_term=95
- no conf  (0.0, 1.0, 0.5): acc=0.464, avg_rounds=2.492, invalid_term=90
- no sum   (1.0, 0.0, 0.5): acc=0.448, avg_rounds=2.620, invalid_term=105
- no stop  (1.0, 1.0, 0.0): acc=0.398, avg_rounds=3.032, invalid_term=118
- high conf (2.0, 1.0, 0.5): acc=0.408, avg_rounds=2.948, invalid_term=108
- high sum  (1.0, 2.0, 0.5): acc=0.422, avg_rounds=2.790, invalid_term=102

## 100-step follow-up (best λ)
- (0.0, 1.0, 0.5) @ step100: acc=0.498, avg_rounds=2.830, invalid_term=57
