# NeurIPS-ready baselines and revision plan (REVISE)

Last updated: 2026-02-21

Note: The LaTeX paper lives in an Overleaf mirror under `paper/`, which is intentionally gitignored in this repo.
File references like `paper/sec/*` refer to that separate paper repo.

## Goal
Turn the current CVPR-style draft into a NeurIPS-ready submission that can credibly target "strong accept".
The main blockers are (1) baseline fairness, (2) statistical credibility, and (3) a tighter claim-evidence chain.

This doc lists (A) what we already compare against, (B) what is implemented in this repo, and (C) what we still need to add.

## A. Baselines already present in the paper

### A.1 VideoEspresso (fine-grained categories)
From `paper/tables/VideoEspresso.tex`:
- VLM baselines: LLaVA-1.5, LLaVA-NeXT (Inter/Video), LongVA-DPO, mPLUG-Owl3, VideoEspresso.
- Backbones with and without REVISE: InternVL2 vs InternVL2 + REVISE; Qwen2-VL vs Qwen2-VL + REVISE; GPT-4o vs GPT-4o + REVISE.
- Closed-source: Qwen-VL-Max, GPT-4o.

Notes:
- This table mixes "different backbones" and "same backbone + wrapper". For NeurIPS, reviewers will ask for a same-backbone/same-budget view.

### A.2 NExT-QA (plug-and-play)
From `paper/tables/nextqa.tex`:
- VideoTree, VideoAgent, LLoVi, ProViQ, SeViLA, LVNet.
- GPT-4o + REVISE.

Notes:
- Numbers are quoted from prior work; paper states this explicitly. NeurIPS reviewers often still ask for a controlled comparison under a unified backbone and budget.

### A.3 EgoSchema (plug-and-play)
From `paper/tables/egoschma.tex`:
- VideoAgent, VideoTree, LVNet, LLoVi, MC-ViT-L.
- GPT-4o + REVISE.

### A.4 Video-MME and LVBench (supplementary)
From `paper/tables/more_benchmarks.tex`:
- "Adaptive Keyframes (CVPR 25)", "MDP3 (ICCV 25)", REVISE.

Notes:
- These baseline names currently have no citations in `paper/main.bib`. NeurIPS reviewers will flag missing references.

### A.5 Reinforcement fine-tuning (RFT)
From `paper/tables/RL Results.tex`:
- Baseline categories: Direct Reasoning, Plug-and-Play, Supervised Format Fine-Tuning, Reinforced Fine-Tuning.

## B. Baselines implemented in this repo (runnable)

### B.1 REVISE plug-and-play (multi-round)
- NExT-QA: `examples/revise/plug_and_play_nextqa_vllm.py`
- EgoSchema: `examples/revise/plug_and_play_egoschema_vllm.py`
- Video-MME + LVBench: `examples/revise/plug_and_play_videomme_lvbench_vllm.py`
- LVBench (HF): `examples/revise/plug_and_play_lvbench_hf.py`

### B.2 One-shot uniform sampling (single call, k frames)
- Video-MME + LVBench: `examples/revise/oneshot_videomme_lvbench_vllm.py`
- LVBench (HF): `examples/revise/oneshot_lvbench_hf.py`

### B.3 Caption-only baseline (single call, captions instead of frames)
- NExT-QA caption-only: `examples/revise/eval_nextqa_caption_vllm.py`

### B.4 VideoAgent-style caption baseline (NExT-QA)
- VideoAgent (captioned timeline + LLM reasoning): `examples/videoagent/eval_nextqa_videoagent_caption_vllm.py`
- VideoAgent "officialstyle" variant: `examples/videoagent/eval_nextqa_videoagent_officialstyle_caption_vllm.py`

## C. Baselines we still need for a NeurIPS-strength story

This section is the "what to add" list. Items are grouped by how reviewers typically critique papers in this space.

### C.1 Must-have (fairness / protocol)
- Same-backbone, same-budget baselines for every dataset:
  - One-shot uniform-k (already for Video-MME/LVBench; add for NExT-QA/EgoSchema/VideoEspresso if missing).
  - One-shot random-k.
  - One-shot "dense" within budget (k up to the allowed total frames) to show the Pareto boundary.
- Report cost in a way reviewers accept:
  - #frames used, #rounds, #VLM calls, prompt tokens, latency (or wall-clock time).
  - Main claim should be framed as a Pareto improvement (accuracy vs cost), not just "better accuracy".

### C.2 Must-have (statistical credibility)
- Multi-seed evaluation for all RL/RFT results:
  - At least 3 seeds; report mean and confidence intervals.
  - Training curves and variance across seeds (to show stability).
- Bootstrap confidence intervals for plug-and-play evaluation where feasible.

### C.3 Strongly recommended (dataset-specific expectations)
- Video-MME / LVBench:
  - Add citations for "Adaptive Keyframes" and "MDP3" if we keep those rows.
  - Add at least one additional widely-recognized long-video baseline under a unified backbone + budget (even if it is a simplified, clearly-defined selector policy).
- NExT-QA / EgoSchema:
  - Either (i) re-run key baselines under a shared backbone and budget, or (ii) explicitly separate "quoted leaderboards" from "controlled comparisons" and make the controlled comparison the main evidence.
  - Resolve LVNet citation inconsistency (`park2024too` vs `awasthi2022lvnet`) and keep one canonical reference.

### C.4 Nice-to-have (extra lift)
- Layered error analysis:
  - Missing evidence vs selector mistake vs summary drift vs formatting.
  - Breakdowns by question type and video length.
- Pareto curves:
  - Accuracy vs frames, vs calls, vs tokens, vs latency (AUC or curves, not only a table).
- Qualitative examples:
  - A small set of cases that show why multi-round helps (and where it fails).

## D. Concrete revision checklist (what to change in the paper)

### D.1 Experimental protocol section (must be explicit)
- Define the evaluation protocol once (sampling FPS, max frames/round, max rounds, decoding params, max output length).
- State what counts as "cost" and how it is measured.
- State how invalid-format outputs are handled (count as wrong vs filtered).

### D.2 Fix inconsistencies that reviewers will notice
- Caption ablation table currently mentions a different closed model than the rest of the paper; unify naming and ensure the model actually appears in the method/results narrative.
- Add missing citations for baselines in `paper/tables/more_benchmarks.tex` or remove the rows.
- Remove duplicate package usage and keep LaTeX clean enough for a NeurIPS template switch.

### D.3 Reframe contributions as claims backed by specific evidence
- For each key claim in Abstract/Intro, point to the exact table/figure/ablation that supports it.
- Prefer claims that are robust under controlled budgets, not claims that depend on mixing backbones.

## E. Work plan (engineering + writing)

### E.1 Code tasks (to unblock controlled comparisons)
- Add NExT-QA and EgoSchema one-shot baselines (uniform-k and random-k) in `examples/revise/`.
- Add a common evaluation harness that logs:
  - frames, rounds, calls, tokens, runtime, validity, and answer.
- Add aggregation scripts that output:
  - mean/CI tables and Pareto CSVs suitable for plotting.

### E.2 Experiment runs (minimum set to upgrade the story)
- Plug-and-play:
  - (backbone A) controlled comparisons across datasets.
  - (backbone B) replicate key headline results.
- RFT:
  - 3 seeds on at least two datasets, with ablations for reward terms and stopping.

### E.3 Paper edits (after evidence is ready)
- Rewrite `paper/sec/4_exp.tex` around the controlled protocol first, and move quoted leaderboards into a separate paragraph/table.
- Add an explicit "Controlled protocol vs quoted leaderboards" note to preempt fairness critiques.
- Add a short "Reproducibility" paragraph (hardware, versions, seed policy, dataset preprocessing).

## F. Open decisions (need owner sign-off)
- Which open-source backbone is the "unified backbone" for controlled comparisons (e.g., Qwen2.5-VL-7B vs InternVL2)?
- Whether to add LongVideoBench (or another long-video QA benchmark) as a main benchmark.
- Whether to keep proprietary-model headline results as primary evidence or as a secondary demonstration.
