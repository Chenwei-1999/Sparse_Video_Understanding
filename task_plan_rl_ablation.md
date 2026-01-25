# Task Plan: RL Ablations (EAGER reward) on NExT-QA (3B VLM)

## Goal
Run reinforcement learning ablations on NExT-QA with a 3B VLM (Qwen2.5-VL-3B), varying EAGER reward weights (λ) and reporting accuracy/efficiency metrics.

## Phases
- [x] Phase 1: Implement paper-faithful EAGER reward (margins + summary-only) plumbing
- [x] Phase 2: Add ablation configs/scripts (λ grid)
- [x] Phase 3: Smoke test training (few steps) to validate signals
- [ ] Phase 4: Run quick 30-step RL sweeps for each λ setting
- [ ] Phase 5: Select best λ, run 100-step, re-evaluate

## Key Questions
1. Do margins (m_t) and summary-only correctness compute reliably during rollout (no crashes, reasonable ranges)?
2. Which λ setting improves val accuracy and/or reduces rounds while keeping formatting stable?
3. How sensitive are results to λ_conf vs λ_sum vs λ_stop?

## Decisions Made
- Use vLLM rollout (TP=4) on 4×A100 for stable multimodal + logprobs-based scoring.
- Keep format reward as a small constant (`format_reward=0.05`) and ablate λ weights.

## Errors Encountered
- (none yet)

## Status
**Currently in Phase 4** - sweeping λ settings (30 steps) and evaluating on NExT-QA val (500-sample subset).
