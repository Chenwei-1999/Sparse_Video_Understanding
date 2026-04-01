# Literature Survey

## Scope

Target literature for this project:

- Video reasoning and VideoQA failure analysis
- Process supervision / verifier models for chain-of-thought
- Multimodal hallucination and self-correction
- Confidence calibration and selective prediction for VLMs

## Working Questions

1. What prior work detects reasoning-process errors rather than only answer errors?
2. What prior work shows that verifier-guided or self-corrective intervention improves multimodal QA?
3. Which claims can be made without human-labeled process annotations?

## Local Evidence To Anchor The Survey

- `debug_prompt_logs/full_nextqa_val_20260305_120329_summary.json`
- `debug_prompt_logs/sft_nextqa_val_20260305_190153_summary.json`
- `/shares/hlw3876/chenwei/VideoCritic_outputs/training_data/summary.json`
- `/shares/hlw3876/chenwei/VideoCritic_outputs/revision_experiment/revision_results.json`
- `/shares/hlw3876/chenwei/VideoCritic_outputs/protocol_comparison/results.json`

## Initial Position

The likely novelty is not another VideoQA benchmark and not generic self-correction. The more defensible angle is:

"A process critic trained with synthetic/annotation-free supervision can detect unreliable video reasoning traces, and critic-gated intervention improves downstream VideoQA accuracy."

This needs to be tested against at least three failure modes:

- detector only learns formatting artifacts
- detector transfers poorly from synthetic chains to real REVISE summaries
- intervention changes too many correct answers and harms net accuracy
