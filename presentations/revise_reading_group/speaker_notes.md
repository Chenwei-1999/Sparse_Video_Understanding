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
- Main point: REVISE should be understood as a three-part system for sparse evidence gathering, not as a single clever sampler.
- Talk track: This is the anchor slide for the method section. The paper's claim is that long-video QA breaks because models both see too much irrelevant content and lack a clean way to represent what key evidence is still missing. REVISE answers that with three linked pieces: a multi-round controller, a structured response format, and a summary-as-state. I want the audience to see that those pieces only really make sense together. The controller chooses what to inspect next, the protocol forces the reasoning into a visible format, and the summary keeps the memory small and cumulative.
- Optional expansion: If someone asks what is novel here, I would say the distinctive part is not iteration alone. It is iteration plus disciplined state.

## s16_one_shot_vs_iterative
- Main point: The real contrast is between one-shot relevance guessing and iterative evidence revision.
- Talk track: A one-shot selector has to guess the important frames before it has learned much from the video. That can work when the signal is obvious, but it breaks when the first evidence is only partially informative. REVISE instead treats selection as something that can be revised. After each round, the model has an explicit account of what it saw, what it currently believes, and what still needs to be checked. That is why I would not describe REVISE as just a better selector. It is closer to a controller that can recover from incomplete first guesses.
- Optional expansion: For a mixed audience, I would translate this as the difference between taking one educated guess and being allowed to ask one more targeted question.

## s17_summary_as_state
- Main point: The summary-as-state is the main memory discipline of the method.
- Talk track: This is the design move I most want people to remember. Only the structured summary persists across rounds. The system does not keep replaying all previous frames or the entire dialogue history as if bigger context were automatically better. Instead, the latest summary is supposed to be cumulative, meaning it already carries forward what matters from prior rounds. That keeps the active context compact, and it also makes the method more inspectable because we can read the state directly.
- Optional expansion: This is a good place to emphasize that memory compression here is question-aware. The state is not a generic video summary; it is a reasoning state for this question.

## s18_pohr
- Main point: POHR is how REVISE turns hidden reasoning state into a readable schema.
- Talk track: The POHR fields look simple, but they do important work. P says what has already been covered, so the loop has a sense of prior evidence. O records the current observations so the state stays tied to what was actually seen. Then H, U, and R together represent the reasoning frontier: what the model now thinks, what is still uncertain, and why the next frames are needed. For the audience, the important point is that this is not decorative formatting. It is the mechanism that makes the loop explicit and inspectable.
- Optional expansion: If someone is skeptical about structured fields, I would point out that the paper's ablations suggest this structure matters, not just the presence of any summary text.

## s19_loop
- Main point: Each round is intentionally small, and the stop decision is part of the algorithm rather than an afterthought.
- Talk track: The loop is operationally straightforward. Start from a small seed set of frames, update the POHR summary, and then choose between two actions: request a few more frame indices or answer the question. The controller repeats that process until either the evidence is sufficient or the turn budget is exhausted. I want to stress that stopping early is a feature here, not a failure to use the full budget. The method is designed to answer once uncertainty has been reduced enough, which is why the efficiency numbers later are meaningful.
- Optional expansion: This is also where the visible tags matter. The output format makes the controller's action legible to the external system.

## s20_worked_example
- Main point: The value of REVISE is easier to understand as evidence accumulation than as an abstract controller.
- Talk track: This is the slide where I want you to picture the loop in action. The first few frames give only a partial story, so the model cannot answer responsibly yet. Instead, it writes down what it has seen and, crucially, what specific uncertainty remains. That uncertainty then justifies the follow-up request. The second round is not broader; it is narrower and more targeted. By the end, the answer comes from a small number of frames because the controller used the first round to decide what evidence was actually missing.
- Optional expansion: This is a useful moment to connect the method to interpretability. We can inspect why the extra frames were requested instead of treating the whole decision process as a dense black box.

## s21_pnp_mode
- Main point: Plug-and-play mode matters because the core idea works even when the backbone model is frozen.
- Talk track: One strength of the paper is that it can test the method without asking us to buy into a new trained model from day one. In plug-and-play mode, REVISE wraps an existing VLM and handles the multi-round orchestration externally. That means the weights do not change, the visual encoder does not change, and even proprietary APIs can be used. Conceptually, this helps isolate the contribution of the controller design itself. If performance improves under this setup, it suggests the interaction pattern is doing real work.
- Optional expansion: For the audience, the takeaway is that REVISE is not only a training recipe. It is also an interface pattern for how to use strong vision-language models more selectively.

## s22_rl_mode
- Main point: In the RL version, the same loop becomes a learned policy over select-versus-answer decisions.
- Talk track: Once the paper moves to open models, it reinterprets the interaction as a finite-horizon MDP. The state includes the current prompt, the latest summary, and the frames already admitted. The actions are exactly the same as before: request more frames or answer now. What changes is that the model is now trained to improve the quality of those choices. That framing is important because it shows the method is not split into two unrelated systems. The plug-and-play and RL variants share the same basic control loop.
- Optional expansion: If someone asks why RL instead of supervised learning, the short answer is that good frame requests and good stopping behavior are sequential decisions with delayed payoff.

## s23_eager_reward
- Main point: EAGER tries to reward the right kind of behavior without requiring frame-level annotations.
- Talk track: I would present EAGER as a practical reward decomposition rather than a formula to memorize. First, it rewards confidence gain, meaning a frame request should earn credit only if the new evidence actually sharpens the answer. Second, it rewards summary sufficiency, which checks whether the final answer is recoverable from the summary alone. Third, it rewards being correct and stopping early. Then there is a small format-validity bonus because the protocol itself has to remain usable. Together these terms push the model toward sparse, faithful, and well-formed reasoning rather than just longer interaction.
- Optional expansion: The summary-sufficiency term is especially nice because it directly trains the state to carry useful information instead of becoming empty boilerplate.

## s24_why_it_works
- Main point: REVISE helps because it ties together sparse retrieval, explicit uncertainty, and disciplined memory.
- Talk track: By this point in the section, I want to synthesize the method logic. REVISE reduces overload by keeping the per-round visual budget small. It improves key-information awareness by forcing the model to externalize what is known and unknown. And it becomes easier to inspect because the control state is visible in POHR instead of buried in hidden activations or long chat history. So the method's advantage is not any single trick. It is the combination of a compact state, uncertainty-driven follow-up, and an explicit stop rule.
- Optional expansion: This is also the place to remind the audience that sparse video reasoning is partly a systems problem. Better control can matter as much as a larger backbone.

## s25_method_takeaway
- Main point: The one-slide memory is that REVISE is a sparse evidence controller with explicit state.
- Talk track: If the audience forgets the notation, I still want them to leave with three statements. First, REVISE is not just selecting frames once; it is iterating. Second, the memory is not free-form; it is structured and cumulative. Third, the same core design supports both frozen plug-and-play use and learned RL control. That is the conceptual payload of the method section, and it sets up why the later results should be interpreted as evidence for the control design rather than only for a particular backbone model.
- Optional expansion: This is a good transition slide into experiments because it tells the audience exactly what claims the empirical section needs to validate.

## Experiments

## s26_setup
- Main point: Read the experiments as an accuracy-versus-budget argument, not as a raw leaderboard dump.
- Talk track: Before looking at numbers, I want to set the lens. REVISE is capped at three frames per round and four rounds overall, so the method is operating under a very small visual budget by design. The important metrics are therefore paired: answer accuracy and total frames used. The paper matters if iterative control lets a frozen model buy much better answers without escaping that sparse regime.
- Optional expansion: The plug-and-play setup is also important because it isolates the controller. If scores improve without retraining the backbone, the interaction pattern itself is doing real work.

## s27_main_results
- Main point: The main result is that iterative control produces large accuracy gains while staying in a single-digit frame regime.
- Talk track: I removed the figure here because the available local image is the turn-ablation plot, not a main-results figure. The slide now reads as a compact results board, which is a better fit for the evidence we actually have. The sharpest example is still VideoEspresso with GPT-4o: 26.4 rises to 48.9 using about eight frames total. The same pattern continues on NExT-QA and EgoSchema, and it is not exclusive to GPT-4o because open backbones like Qwen2-VL and InternVL2 also improve on VideoEspresso.
- Optional expansion: If I had to compress the message to one line, it would be: the controller changes the operating point of the backbone more than simple prompt variation would suggest.

## s28_efficiency
- Main point: The paper's strongest systems claim is efficiency at low frame counts, not absolute dominance on every benchmark.
- Talk track: REVISE usually finishes in the high-single-digit frame range, which is dramatically smaller than the input counts used by many long-video baselines. On NExT-QA, the paper highlights that it uses three to four times fewer inputs than systems like LVNet and SeViLA, and much less than heavier methods like VideoTree or LLoVi. On EgoSchema it stays close to VideoAgent's low-frame regime while avoiding an external captioner. The fair reading is that REVISE trades some peak accuracy for a simple, efficient, and inspectable controller.
- Optional expansion: That trade-off is actually part of the appeal in an internal systems setting, where frame cost and latency are often first-class constraints.

## s29_ablation_rounds
- Main point: More turns help because they create opportunities to revise the search, not because they force the model to look at many more frames.
- Talk track: The visible slide is intentionally terse, because the figure is already doing part of the work. The key numbers are the endpoints: one turn gets 38.3 accuracy with 4.60 frames, while four turns reach 42.1 with 2.89 selected frames. The subtle but important detail is that the mean number of rounds is only about 2.28, so allowing more turns does not mean the controller always spends them. It means the controller can ask a targeted follow-up when needed and stop when it is not.
- Optional expansion: This is one of the cleanest pieces of evidence that iteration is doing conceptual work beyond just increasing context.

## s30_ablation_components
- Main point: The summary state is load-bearing, and both carryover and structure matter.
- Talk track: The full system reaches 41.48 accuracy. If we remove state carryover, the score falls to 23.14. If we keep carryover but remove the structured POHR fields, it falls to 24.27. Removing both drops it further to 20.24. I read this as very strong evidence that REVISE is not winning simply because it can ask for more frames. It wins because each round leaves behind a usable, cumulative reasoning state.
- Optional expansion: This ablation is what makes the summary-as-state claim credible rather than rhetorical.

## s31_empirical_takeaway
- Main point: The experiments validate the controller design more than any one backbone choice.
- Talk track: If I compress the entire empirical section into one statement, it is this: strong backbones benefit from a better interface to sparse evidence. The method works best when we judge it on the frontier of accuracy versus frames, and the ablations trace the gains back to the paper's central design idea of a structured cumulative state. That is why I think the experiments mostly support the conceptual thesis of the paper.
- Optional expansion: In other words, the method section and the results section line up unusually well here. The experiments test the exact thing the paper says matters.

## Closing

## s32_takeaways
- Main point: The closing takeaway is about problem framing, controller design, and memory design.
- Talk track: My three takeaways are intentionally simple. First, long-video QA is best seen as selective evidence gathering, not dense viewing. Second, REVISE turns that into a concrete iterative loop over read, summarize, and either request or answer. Third, the summary-as-state idea is the most reusable conceptual contribution, because it says what kind of memory object an agentic video system should carry forward.
- Optional expansion: Even if future methods outperform REVISE, I would expect many of them to inherit this same basic framing.

## s33_open_questions
- Main point: The open questions are mostly about how far this memory-and-control pattern can scale.
- Talk track: I want to frame this slide gently rather than as reviewer criticism. REVISE seems strongest when a few targeted glimpses can resolve the question. Harder cases may require many weak clues spread across time, or richer state than POHR alone can comfortably hold. So the discussion question is not whether the paper works in its intended setting. It is what the next memory object should look like when the sparse-evidence story becomes more demanding.
- Optional expansion: Another useful discussion axis is when we should value efficiency and interpretability over chasing the strongest high-frame leaderboard.

## s34_qa
- Main point: The appendix is there to support discussion, not to reopen the whole talk.
- Talk track: For Q and A, I have backup on three likely directions: extra benchmark numbers including the RL setting, more detail on how EAGER shapes behavior, and a stronger read of the component ablation. I also kept one related-work slide in reserve in case the conversation turns to where REVISE sits relative to caption pipelines, selectors, and agentic systems.
- Optional expansion: If time is short, the single discussion prompt I would keep is: what is the right persistent state for iterative long-video reasoning?

## Appendix

## a01_more_benchmarks
- Main point: The RL results are backup evidence that the sparse-control pattern is learnable, not just promptable.
- Talk track: This slide is now self-contained because it keeps the plug-and-play GPT-4o reference numbers beside the RL numbers. On VideoEspresso, plug-and-play reaches 48.9 with about eight frames, while RL gets 27.8 with 4.1 frames. On NExT-QA, the comparison is 63.8 with 8.4 frames versus 51.3 with 3.9 frames. So the point is not that RL beats plug-and-play here. The point is that the learned policy keeps roughly half the visual budget while staying recognizably functional.
- Optional expansion: I would use this slide mainly if someone asks whether the paper is an interface trick or a learnable decision problem. The answer is: both.

## a02_reward_details
- Main point: EAGER tries to turn good sparse-browsing behavior into a shaped reward without dense supervision.
- Talk track: I would not present EAGER as a formula to memorize. The conceptual pieces are enough. Confidence gain says extra frames should earn reward only when they help. Summary sufficiency says the carried state should be useful enough that the answer is recoverable from it. Correct-and-early stop keeps the policy from browsing forever. Format validity is small but practical, because the whole loop depends on well-formed structured output.
- Optional expansion: The summary-sufficiency term is the most distinctive part because it directly trains the memory object, not only the final answer.

## a03_component_ablation
- Main point: The component ablation is the strongest evidence that memory design is causal here.
- Talk track: I changed this backup slide so it does more than restate s30. The useful readout is the size and shape of the degradation. The full system reaches 41.48 with 2.79 turns and 22.71 seconds, so the baseline is not secretly buying accuracy by running forever. Removing carryover costs 18.34 points, removing POHR structure costs 17.21, and removing both costs 21.24. That pattern says the two ingredients are individually important and jointly reinforcing.
- Optional expansion: For me, the most important part is the operational readout column: it translates the table from raw scores into an argument about what kind of loop failure each ablation induces.

## a04_related_work_extra
- Main point: The cleanest related-work comparison is to ask what each family remembers and how it decides what to inspect next.
- Talk track: Caption and memory systems remember compressed text. Single-pass selectors remember a relevance ranking. REVISE-like agentic systems remember an evolving reasoning state. That framing makes the differences easier to discuss than a raw method list, because it keeps attention on the control problem and the memory problem the paper is actually addressing.
- Optional expansion: This is also a helpful slide if the group wants to generalize beyond video QA and ask whether similar state designs could matter in other multimodal agent settings.
