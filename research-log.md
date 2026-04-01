# Research Log

## 2026-04-01

### Bootstrap

- Initialized autoresearch workspace in `/home/cxk2993/VideoReasoning`.
- Confirmed local resources:
  - 3 x A100 80GB available.
  - NExT-QA present at `/shares/hlw3876/chenwei/NExT-QA`.
  - Existing REVISE SFT checkpoint present at `outputs/revise_nextqa_sft/global_step_844/huggingface`.
  - VideoCritic synthetic/error-analysis artifacts present at `/shares/hlw3876/chenwei/VideoCritic_outputs`.
- Inspected current repo positioning:
  - REVISE is strong on sparse evidence acquisition and compact multi-round state.
  - The repo does not yet make a solid causal claim about reasoning-chain error detection.
  - Existing prompt logs expose rich round-by-round process traces, making process-level analysis feasible.
- Key empirical observations from existing artifacts:
  - Base Qwen2.5-VL-3B plug-and-play REVISE run on NExT-QA val has low accuracy (40.8%) and many invalid outputs.
  - REVISE SFT model reaches 56.5% on the same split and nearly eliminates invalid output failures.
  - Prior VideoCritic revision experiments indicate that broad correction/revision often harms accuracy.
  - Therefore, the research opportunity is not "always correct the chain", but "detect risky chains precisely enough to intervene selectively."
- Initial hypotheses recorded in `research-state.yaml`:
  - H1: synthetic process critic transfers to REVISE traces.
  - H2: critic-gated intervention beats always-revise.
  - H3: annotation-free process inconsistency signals are a meaningful baseline.
