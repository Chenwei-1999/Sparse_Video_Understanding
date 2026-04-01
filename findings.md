# Findings

## Current Understanding

The current repo has a credible systems/method story around sparse frame acquisition, but the "video reasoning error" story is not yet solid enough on its own. The missing piece is a reliable process-level signal that says when the reasoning chain is likely wrong and a controlled demonstration that acting on this signal helps VideoQA.

The strongest available angle is to treat REVISE's multi-round summaries as process traces and connect them to the existing `VideoCritic_outputs` assets. This creates a path to process-error research without requiring human-labeled reasoning chains:

1. Use synthetic or automatically constructed chain corruption data to train or calibrate a process critic.
2. Test whether that critic transfers to real REVISE traces by predicting final answer failure.
3. Use the critic only for selective intervention, because prior broad revision appears harmful.

## Patterns And Insights

- Invalid output behavior is itself informative but insufficient.
  The weak base REVISE run fails heavily on formatting/action validity, while the SFT model mostly fixes this. That suggests process quality is not just syntax; a stronger detector must model semantic inconsistency and unsupported updates.

- Broad correction is dangerous.
  Existing `VideoCritic_outputs/revision_experiment/revision_results.json` shows that revising too many chains can break many already-correct answers. This makes selective routing a core design requirement, not an optional ablation.

- Synthetic supervision is already available.
  The `VideoCritic_outputs/training_data/train.jsonl` file contains 12.6k clean/error chains across logical, temporal, spatial, perceptual, and hallucination error types. Even if its distribution differs from REVISE traces, it is enough to test whether transferable error scoring exists.

- The strongest likely contribution is calibration, not raw taxonomy classification.
  High-granularity error typing appears noisy and high-FP in prior artifacts. A more solid claim may be:
  "process risk scoring predicts when a video reasoning chain is unreliable, and risk-aware intervention improves QA."

## Lessons And Constraints

- Do not center the story on human annotation. There is already enough synthetic and outcome-based supervision to make the idea solid.
- Do not use unconditional revise/reread as the main method; existing evidence suggests it often hurts.
- The paper claim should be framed around transfer and utility:
  detect process risk from chain text plus interaction signals, then use that risk to improve downstream QA.
- The first experiments should be cheap and discriminative:
  if critic scores do not correlate with answer correctness on existing logs, the current direction should pivot quickly.

## Open Questions

- How transferable are synthetic chain errors to real REVISE traces?
- What minimal set of process features is enough to build a strong no-extra-label baseline?
- Which intervention is safest:
  re-answer from the same evidence, request more frames, or ask the model to explicitly revise its POHR summary?
- Does risk gating help mostly on base models, or also on the stronger REVISE SFT checkpoint?
