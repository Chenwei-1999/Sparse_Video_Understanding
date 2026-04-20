# Speaker Notes

## Opening

## s01_title
- Main point: REVISE is a paper about treating long-video QA as sparse evidence gathering rather than dense video consumption.
- Talk track: I want to frame this paper less as one more video benchmark result and more as a proposal for how to interact with long videos. The key idea is that most frames are irrelevant to a specific question, so the system should not try to stuff the whole video into one context window. REVISE instead requests a few frames at a time, updates a compact state, and keeps going only if the evidence is still incomplete.
- Optional expansion: The title also hints at the paper's scope. This is not generic video generation or summarization. It is question-aware reasoning under a strict evidence budget.

## s02_preview
- Main point: The paper matters because it turns long-video QA into an iterative retrieval-and-reasoning loop.
- Talk track: The preview version is: long videos are mostly redundant, the useful evidence is question-specific, and REVISE operationalizes that observation. Instead of paying the cost of dense viewing upfront, it treats frame access as something the model should earn by showing why the next frames are needed. That is the conceptual shift I want to keep in mind through the talk.
- Optional expansion: One useful lens is to compare this with retrieval-augmented language systems. The hard part is not only reasoning over evidence, but choosing the right evidence at the right time.

## s03_roadmap
- Main point: The roadmap is organized around field setup, the method landscape, and then REVISE itself.
- Talk track: I want the audience to know where we are going before we dive into details. First I will set up the task, explain why long-video QA is hard, and anchor that with the benchmark choices. Then I will map the landscape by method family and state-of-the-art framing. Only after that context do we move into REVISE itself: the core idea, the results, and the takeaways.
- Optional expansion: This ordering is intentional for a mixed audience. The method is easier to evaluate once the retrieval and memory design space is visible.

## Problem Setup

## s04_task
- Main point: Long-video QA asks the model to answer a question from evidence that may be scattered across a long temporal sequence.
- Talk track: The task sounds simple at a high level, but it is structurally hard. The answer may depend on causal order, a brief event, or a detail that only becomes meaningful later. On top of that, the model has a fixed context budget, so it cannot just look at every frame at full fidelity. The practical problem is deciding what to inspect and what to ignore.
- Optional expansion: The benchmarks used here stretch that problem in different directions, from medium-length causal reasoning to longer egocentric video and sparse core-frame reasoning.

## s05_why_hard
- Main point: The difficulty reduces to redundancy, finite reading budget, and question-dependent selectivity.
- Talk track: Most frames in a long video do not help with a specific question, so dense sampling is immediately wasteful. At the same time, the model cannot inspect everything because visual context is expensive. That means the system has to be selective, and the right frames depend on the question being asked. Those three pressures together are why long-video QA is not solved by just scaling up the input window.
- Optional expansion: The benchmark mix matters here because each dataset stresses a different version of this same constraint triangle.

## s06_sparse_evidence
- Main point: REVISE claims that the real structure of the problem is semantic sparsity.
- Talk track: The paper's central abstraction is that only a small subset of frames matters for any given question. That subset changes with the prompt, so static selection policies are mismatched. The model therefore needs a compact record of verified evidence and a way to request more frames specifically to resolve the remaining uncertainty.
- Optional expansion: This is where the summary-as-state idea starts to appear. It is not just memory for past frames; it is memory shaped around the question.

## s07_benchmarks
- Main point: The chosen benchmarks matter because they probe different temporal regimes and reasoning demands.
- Talk track: NExT-QA is closer to short or medium videos with causal and temporal reasoning. EgoSchema pushes toward longer egocentric clips where informative moments are buried inside routine activity. VideoEspresso emphasizes sparse core-frame reasoning over a diverse set of reasoning types. Taken together, they are a good testbed for asking whether a sparse, iterative controller generalizes across different notions of long-video difficulty.
- Optional expansion: I read this benchmark mix as a stress test for the claim that sparse evidence selection is a general principle rather than a dataset-specific trick.

## s08_failure_case
- Main point: A one-shot pass over uniformly sampled frames often fails because it lacks a follow-up mechanism.
- Talk track: If the decisive evidence is missing from the first sample, the model can only guess or hedge. What it cannot do in a standard one-shot setup is say which missing observation would resolve the ambiguity and then request it. The motivating example in the paper is useful because it makes clear that the missing capability is not just better reasoning over a fixed set of frames; it is adaptive evidence gathering.
- Optional expansion: This also explains why simply adding more frames is not an ideal fix. More context is expensive and still may not be the right context.

## Landscape

## s09_field_shift
- Main point: The broader field is moving from dense video ingestion toward selective evidence use.
- Talk track: Earlier systems mostly tried to expand the amount of video that could fit into one pass, either through more frames or more aggressive textual compression. The newer trend is to treat visual access as something that should be selected, scheduled, and justified. That shift is what makes REVISE timely: the field is no longer asking only how to scale context, but how to spend it intelligently.
- Optional expansion: I would describe this as a move from capacity-centric design to control-centric design.

## s10_taxonomy
- Main point: A useful taxonomy is dense video-LLMs, caption or memory pipelines, query-aware selectors, and agentic multi-round systems.
- Talk track: These categories are not perfect silos, but they surface the key design tradeoffs. Dense models preserve raw visual evidence but waste budget. Caption or memory pipelines scale better but may lose detail. Query-aware selectors are more efficient, yet often depend on a single selection pass. Agentic systems add iteration, which is promising, but then the main design problem becomes how to represent state across rounds.
- Optional expansion: That last point is the bridge to REVISE. Once you allow multiple rounds, memory design becomes first-class.

## s11_caption_memory
- Main point: Caption and memory pipelines win on scale, but they accept an information bottleneck early.
- Talk track: The advantage of these systems is obvious: convert the video into language, then let the LLM operate in a domain it handles well. The downside is that the abstraction happens before the question has been fully resolved. If the answer depends on a subtle motion cue, object state, or temporal relation that the captioner flattened away, the downstream reasoner cannot recover it. So these methods are strong for coarse narrative understanding, but weaker for fine-grained evidence tracking.
- Optional expansion: This is why REVISE's direct frame access matters. It avoids committing to text-only compression as the sole intermediate representation.

## s12_frame_selection
- Main point: Query-aware frame selection improves efficiency, but many methods still behave like one-shot retrieval.
- Talk track: The selector family is appealing because it addresses the obvious waste in dense sampling. The limitation is that a single ranking step often has to predict relevance before enough evidence has been gathered to know what is truly missing. Without a persistent state describing current beliefs and gaps, the system can rank frames, but it has a weaker basis for iterative follow-up.
- Optional expansion: Put differently, selection alone is not yet a reasoning loop. It becomes one only when the model can update a state and use that state to decide the next action.

## s13_agentic_reasoning
- Main point: Agentic video methods add the missing loop, but they still need a stable memory representation.
- Talk track: This family is closest in spirit to REVISE because it allows multiple rounds of observe, reason, and request. That better matches the structure of long-video QA. But once you allow iteration, you inherit a new systems problem: how do you preserve what the model has learned so far without dragging along the entire history? REVISE answers that with a structured summary state rather than raw frame replay or unconstrained dialogue history.
- Optional expansion: I think this is the most important comparative point. REVISE is not just agentic; it is agentic with a specific memory discipline.

## s14_gap_to_revise
- Main point: REVISE sits at the intersection of direct visual reasoning, iterative frame requests, and compact structured state.
- Talk track: The gap the paper targets is fairly specific. Caption pipelines compress too early, selection methods are often too static, and agentic methods still need a disciplined way to remember what has already been established. REVISE's bet is that a summary-as-state, updated every round, is enough to stabilize multi-round video reasoning while staying compatible with both closed and open models.
- Optional expansion: This is also why the method has two appealing modes. It can act as a controller around proprietary APIs, and it can become a training target for reinforcement learning in open models.

## REVISE Core Idea

## s15_overview
- Main point:
- Talk track:
- Optional expansion:

## s16_one_shot_vs_iterative
- Main point:
- Talk track:
- Optional expansion:

## s17_summary_as_state
- Main point:
- Talk track:
- Optional expansion:

## s18_pohr
- Main point:
- Talk track:
- Optional expansion:

## s19_loop
- Main point:
- Talk track:
- Optional expansion:

## s20_worked_example
- Main point:
- Talk track:
- Optional expansion:

## s21_pnp_mode
- Main point:
- Talk track:
- Optional expansion:

## s22_rl_mode
- Main point:
- Talk track:
- Optional expansion:

## s23_eager_reward
- Main point:
- Talk track:
- Optional expansion:

## s24_why_it_works
- Main point:
- Talk track:
- Optional expansion:

## s25_method_takeaway
- Main point:
- Talk track:
- Optional expansion:

## Experiments

## s26_setup
- Main point:
- Talk track:
- Optional expansion:

## s27_main_results
- Main point:
- Talk track:
- Optional expansion:

## s28_efficiency
- Main point:
- Talk track:
- Optional expansion:

## s29_ablation_rounds
- Main point:
- Talk track:
- Optional expansion:

## s30_ablation_components
- Main point:
- Talk track:
- Optional expansion:

## s31_empirical_takeaway
- Main point:
- Talk track:
- Optional expansion:

## Closing

## s32_takeaways
- Main point:
- Talk track:
- Optional expansion:

## s33_open_questions
- Main point:
- Talk track:
- Optional expansion:

## s34_qa
- Main point:
- Talk track:
- Optional expansion:
