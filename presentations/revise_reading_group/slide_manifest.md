# Slide Manifest

## Opening
- `s01_title`: title slide and talk framing; visual anchor for the deck.
- `s02_preview`: one-slide preview of the paper's core claim and payoff.
- `s03_roadmap`: agenda and section map for the rest of the talk.

## Problem Setup
- `s04_task`: define the long-video QA setting and what the model is asked to do.
- `s05_why_hard`: explain redundancy, distractors, and context-budget pressure.
- `s06_sparse_evidence`: show that relevant evidence is sparse relative to total video length.
- `s07_benchmarks`: orient the audience to the benchmark landscape used in this area.
- `s08_failure_case`: motivate the problem with a concrete failure case from dense processing.

## Landscape
- `s09_field_shift`: summarize the shift from dense video processing to selective reasoning.
- `s10_taxonomy`: group prior work into method families and tradeoffs.
- `s11_caption_memory`: cover caption-then-reason and memory-centric pipelines.
- `s12_frame_selection`: cover query-aware and learned frame-selection methods.
- `s13_agentic_reasoning`: cover multi-round, agentic, and tool-using video methods.
- `s14_gap_to_revise`: isolate the gap that REVISE fills.

## REVISE Core Idea
- `s15_overview`: introduce REVISE at a high level.
- `s16_one_shot_vs_iterative`: contrast one-shot inference with iterative evidence gathering.
- `s17_summary_as_state`: explain summary-as-state as the central memory representation.
- `s18_pohr`: define the POHR step and why it matters.
- `s19_loop`: show the reasoning loop and how state evolves over rounds.
- `s20_worked_example`: walk through a concrete example of the loop in action.
- `s21_pnp_mode`: describe plug-and-play mode and where it fits.
- `s22_rl_mode`: describe reinforcement-learning mode and where it fits.
- `s23_eager_reward`: explain the eager-reward signal used in training.
- `s24_why_it_works`: summarize the mechanism-level explanation for why REVISE works.
- `s25_method_takeaway`: close the method section with the main takeaway.

## Experiments
- `s26_setup`: describe datasets, metrics, and evaluation protocol.
- `s27_main_results`: show the primary accuracy or QA performance results.
- `s28_efficiency`: show efficiency, token, or frame-budget comparisons.
- `s29_ablation_rounds`: isolate the effect of iterative rounds.
- `s30_ablation_components`: isolate the effect of individual method components.
- `s31_empirical_takeaway`: summarize the empirical evidence in one slide.

## Closing
- `s32_takeaways`: restate the three most important conclusions.
- `s33_open_questions`: surface limitations and open research questions.
- `s34_qa`: final Q&A slide and discussion prompt.
