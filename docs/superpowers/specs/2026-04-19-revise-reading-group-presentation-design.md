# REVISE Reading Group Presentation Design

## Context

This design covers a 60-minute internal company reading-group presentation on the paper in this repository:

- `REVISE / Towards Sparse Video Understanding and Reasoning`

The talk format is:

- `45 minutes` presentation
- `15 minutes` Q&A

The audience is mixed. Many attendees are not specialists in video understanding or long-video reasoning, so the talk must establish the field context before presenting the paper itself.

The requested output is:

- an editable `PPTX` deck
- speaker notes / talk track

The user prefers:

- English slides
- a context-heavy framing
- an explanatory tone rather than a reviewer-style critique
- an `Editorial Research` visual style

## Presentation Goal

The presentation should help a mixed technical audience do three things by the end of the talk:

1. Understand why long-video QA and sparse video reasoning are difficult problems.
2. Understand the broader research landscape around efficient video reasoning and adaptive frame selection.
3. Understand what REVISE contributes, how it works at a high level, and why it is interesting in context.

The deck should not behave like a line-by-line paper summary. It should behave like a well-structured seminar that uses one paper as the anchor for a broader field introduction.

## Recommended Framing

The talk should follow a `landscape-first, paper-in-context` structure rather than a paper-first journal-club structure.

Reasoning:

- This matches the audience profile better than a method-heavy walkthrough.
- It supports the user's request to include research landscape, state-of-the-art methods, and related work.
- It allows the talk to introduce REVISE as one point in a larger methodological trend rather than an isolated contribution.

The tone should be primarily explanatory:

- emphasize intuition and field orientation
- present strengths clearly
- include limitations and open questions, but keep critique measured and constructive

## Narrative Structure

The 45-minute main talk should follow this arc:

### Part I: Why Long-Video QA Is Hard

Introduce the task and establish the practical constraints:

- long videos are temporally redundant
- models have finite context budgets
- only a small subset of frames may matter for a given question
- naive uniform sampling can miss critical evidence or waste capacity on irrelevant frames

This section should make the problem intuitive even for people who do not work on video understanding.

### Part II: What The Field Has Tried

Present a compact taxonomy of the main methodological strategies:

- dense or uniform frame processing
- caption or memory mediated pipelines
- query-aware / adaptive frame selection
- agentic / multi-round reasoning approaches

The purpose is not exhaustive survey coverage. The purpose is to show the design space and identify the gap REVISE addresses.

### Part III: REVISE In Context

Present REVISE as a method that combines:

- adaptive multi-round frame acquisition
- persistent summary-as-state memory
- structured reasoning traces via POHR
- both plug-and-play and RL fine-tuning modes

This section should explain the design motivation and operational mechanism clearly, but it should not get lost in low-level implementation details.

### Part IV: What The Results Mean

Summarize the empirical message:

- REVISE improves sparse, question-aware video reasoning
- it can operate with very small frame budgets
- multi-round selection and compact state are the core story

Conclude with:

- practical takeaways
- measured limitations
- open questions for the field

## Timing Plan

The main talk should be budgeted as follows:

- `12-14 min` field and problem setup
- `8-10 min` research landscape and SOTA taxonomy
- `12-13 min` REVISE method
- `7-8 min` experiments, results, and takeaways
- `2-3 min` conclusion and transition into Q&A

This leaves a small timing buffer inside the 45-minute presentation and preserves the full 15-minute Q&A block.

## Slide Architecture

The recommended deck size is:

- `34-38 total slides`
- `30-33 main presentation slides`
- `4-6 appendix / backup slides`

### Section 1: Opening And Framing

`3 slides`

- title slide
- one-slide preview of the talk's key message
- roadmap slide

### Section 2: Problem Setup

`6-7 slides`

- what VideoQA / long-video reasoning is
- why the setting is hard
- why uniform sampling is often inadequate
- benchmark setup and task diversity
- one motivating example or failure case

### Section 3: Research Landscape And SOTA

`7-8 slides`

- field evolution
- taxonomy slide
- representative method buckets
- explicit positioning of REVISE relative to nearby approaches
- bridge slide defining the remaining gap

### Section 4: REVISE Core Idea

`8-9 slides`

- overview figure
- summary-as-state intuition
- POHR structure
- multi-round interaction loop
- plug-and-play mode
- RL mode and EAGER reward
- why the method differs from one-shot selection

### Section 5: Experiments And Takeaways

`6-7 slides`

- benchmarks and setup
- main results
- efficiency framing
- ablations
- high-level empirical takeaways

### Section 6: Conclusion

`3 slides`

- three final takeaways
- limitations and open questions
- Q&A slide

### Section 7: Appendix / Backup

`4-6 slides`

- extra ablations
- benchmark details
- RL reward details
- additional comparison material for questions

## Visual Design System

The deck should use the selected `Editorial Research` style.

### Design Direction

- warm off-white or paper-like background
- dark navy / charcoal text
- restrained accent colors
- strong visual hierarchy
- generous white space
- clean academic polish without default PowerPoint aesthetics

### Slide Types

The deck should rotate across a small set of slide templates:

- section divider slide
- concept / intuition slide
- taxonomy / comparison slide
- paper figure slide
- result comparison slide
- takeaway slide

This prevents the presentation from feeling repetitive while keeping the deck visually coherent.

### Text Density

The deck should remain figure-first:

- use short bullets
- avoid paragraph slides
- prefer one strong message per slide
- keep citations visible but unobtrusive

Because this is a seminar, some slides can be denser than a conference talk, but the deck should still avoid paper-like text blocks.

## Content Strategy

### Research Landscape Coverage

The deck should include a curated set of representative methods rather than a full survey dump.

Planned categories:

- long-video VLM / compression style methods
- caption and memory mediated pipelines
- query-aware adaptive frame selection
- agentic multi-round reasoning

Representative methods should likely include several of the following, depending on space and citation cleanliness:

- VideoAgent
- VideoTree
- SeViLA
- VideoEspresso
- Flexible Frame Selection
- Adaptive Keyframe Sampling
- Q-Frame
- BIMBA

The comparison should explain what each family optimizes for:

- accuracy
- efficiency
- temporal coverage
- plug-and-play compatibility
- interpretability

### REVISE Coverage

The REVISE section should prioritize:

- the motivating problem
- summary-as-state as the conceptual centerpiece
- POHR as the explicit state format
- the multi-round frame request loop
- the distinction between plug-and-play and RL modes
- the practical efficiency story

The following should be included but kept concise:

- GRPO
- EAGER reward
- detailed hyperparameters
- exhaustive table-by-table commentary

### Empirical Framing

Results should be interpreted for a mixed audience:

- what improves
- what gets more efficient
- what the ablations imply about why the method works

The talk should not attempt to litigate every benchmark detail. It should instead help the audience remember the empirical message.

## Source Material Plan

The deck should preferentially reuse local assets from this repository when they are already presentation-ready:

- `assets/overview.png`
- `assets/summary_as_state.png`
- `assets/sketch.png`
- `assets/pareto.png`

The paper source and local notes should be used to extract:

- method explanation
- benchmark setup
- reported results
- ablation conclusions

Primary local sources:

- `README.md`
- `paper/main.tex`
- `paper/sec/1_intro.tex`
- `paper/sec/2_related.tex`
- `paper/sec/3_method.tex`
- `paper/sec/4_exp.tex`

External literature should be used selectively to strengthen the landscape section and related-work framing. The landscape should rely on representative, primary-source papers rather than secondary summaries.

## Speaker Notes Strategy

The deliverable should include speaker notes, but they should not be a verbatim script.

For each slide, notes should contain:

- the main point of the slide
- a short talk track for what to say
- an optional expansion point if time allows

This keeps the deck usable for live speaking while still giving the user enough structure for rehearsal.

## Deliverables

The final design targets these artifacts:

1. Editable `PPTX` slide deck.
2. Speaker notes / talk track aligned slide-by-slide.
3. Backup slides for Q&A.
4. Supporting slide map or outline if useful during production.

## Quality-Control Requirements

Before the deck is considered complete, it should pass a presentation-specific QA check:

- timing sanity check for a 45-minute talk
- no text overflow
- readable fonts and figure labels
- consistent citations and section structure
- smooth narrative transitions
- appendix slides ready for common follow-up questions

## Risks And Guardrails

### Risk 1: Too Much Related Work

If the landscape section becomes a citation wall, the audience will lose the thread.

Guardrail:

- use taxonomy first, representative methods second
- compare families, not just individual papers

### Risk 2: Too Much Paper Detail

If REVISE is presented like a methods lecture, the audience fit breaks.

Guardrail:

- explain mechanism and intuition first
- compress training details and hyperparameters

### Risk 3: Too Little Experimental Interpretation

If the results section only repeats numbers, the audience will not remember the point.

Guardrail:

- emphasize what the results mean for sparse video reasoning
- explicitly connect metrics back to the method claims

### Risk 4: Style Drift

If the deck mixes too many layouts or color logics, it will feel amateurish.

Guardrail:

- use a small number of layout types
- keep a single coherent visual language throughout

## Out Of Scope

This design does not cover:

- a full literature-review document beyond what is needed for the deck
- a deeply adversarial or reviewer-style critique of the paper
- a job-talk style personal narrative
- implementation of new experiments beyond what is already reported and needed for presentation framing

## Decision Summary

Confirmed decisions:

- audience: mixed technical internal reading group
- format: 60 minutes total, `45 + 15`
- output: editable `PPTX` plus speaker notes
- slide language: English
- emphasis: context-heavy
- framing: landscape-first, paper-in-context
- tone: mostly explanatory
- visual style: `Editorial Research`
