# H1 Protocol: Process-Risk Calibration On Existing REVISE Logs

## Question

Can a process critic, trained or calibrated without human-labeled REVISE traces, identify which REVISE reasoning chains are likely to end in a wrong final answer?

## Hypothesis

Synthetic chain errors plus annotation-free process features will produce a risk score that correlates with downstream answer failure on NExT-QA REVISE traces. This risk score will outperform simple invalid-format heuristics.

## Why This Matters

If this fails, the current "reasoning error" direction is weak and should pivot. If this succeeds, it supports a solid second-stage intervention paper story.

## Inputs

- REVISE prompt logs:
  - `debug_prompt_logs/full_nextqa_val_20260305_120329.jsonl`
  - `debug_prompt_logs/sft_nextqa_val_20260305_190153.jsonl`
- Synthetic error supervision:
  - `/shares/hlw3876/chenwei/VideoCritic_outputs/training_data/train.jsonl`
  - `/shares/hlw3876/chenwei/VideoCritic_outputs/training_data/test.jsonl`

## Planned Signals

- Structural validity:
  - invalid tags / retries / duplicate or stale summaries
- Summary dynamics:
  - uncertainty persists but final answer is emitted
  - hypothesis statement changes sharply across rounds
  - answer letter changes under revision
- Textual risk:
  - generic unsupported claims
  - contradiction markers
  - hallucination-like insertions
- Learned critic:
  - a small text classifier or scoring model on synthetic clean/error chains

## Metrics

- AUROC and AUPRC for predicting final answer correctness
- Recall on the top-k highest-risk chains
- Accuracy lift on the high-risk subset versus random selection

## Baselines

- Invalid-output-only heuristic
- Logistic regression on hand-built process features
- Synthetic-chain critic score alone

## Confirmatory Criteria

- At least one risk model must show meaningful separation between correct and incorrect chains on REVISE logs.
- The best model must outperform invalid-output-only heuristics by a non-trivial margin on AUROC or top-k recall.

## Failure Conditions

- No risk model generalizes beyond formatting artifacts.
- Signal exists only for the weak base model but disappears on the stronger SFT model.

## Immediate Deliverables

- A reusable trace-extraction script under `src/`
- Risk-analysis tables under `experiments/h1-process-risk-calibration/results/`
- A recommendation on whether to proceed to H2 selective intervention
