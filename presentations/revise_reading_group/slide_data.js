const LOCAL_ASSETS = {
  overview: "assets/overview.png",
  summary: "assets/summary_as_state.png",
  sketch: "assets/sketch.png",
  pareto: "assets/pareto.png"
};

const slides = [
  {
    id: "s01_title",
    section: "opening",
    layout: "title_hero",
    title: "REVISE: Towards Sparse Video Understanding and Reasoning",
    subtitle: "Reading group framing for long-video QA under sparse evidence",
    bullets: [
      "Long videos contain far more visual evidence than a model can process productively in one pass",
      "The paper reframes video QA as iterative evidence gathering instead of dense viewing",
      "The central design move is to keep a compact summary state while requesting only new, useful frames"
    ],
    asset: {
      kind: "image",
      path: LOCAL_ASSETS.overview,
      caption: "REVISE overview"
    },
    citations: ["REVISE"],
    notesKey: "s01_title"
  },
  {
    id: "s02_preview",
    section: "opening",
    layout: "preview",
    title: "Why This Paper Matters",
    bullets: [
      "Long-video QA is mostly a retrieval problem disguised as a context-window problem",
      "Relevant evidence is sparse and depends on the question, so uniform sampling wastes budget",
      "REVISE turns frame access into an iterative reasoning loop that can stop early when evidence is sufficient"
    ],
    asset: null,
    citations: ["REVISE"],
    notesKey: "s02_preview"
  },
  {
    id: "s03_roadmap",
    section: "opening",
    layout: "comparison",
    title: "Roadmap",
    columns: [
      { header: "Field Setup", items: ["Task", "Why hard", "Benchmarks"] },
      { header: "Landscape", items: ["Method families", "SOTA framing"] },
      { header: "REVISE", items: ["Core idea", "Results", "Takeaways"] }
    ],
    asset: null,
    citations: [],
    notesKey: "s03_roadmap"
  },
  {
    id: "s04_task",
    section: "context",
    layout: "context",
    title: "Task Setup: Long-Video Question Answering",
    bullets: [
      "Input is a user question plus a long sequence of video frames spread across time",
      "The model must recover the answer from temporally distributed evidence, not just local appearance",
      "In practice the system also has to respect a fixed visual-context budget, so it cannot inspect everything"
    ],
    asset: null,
    citations: ["REVISE", "NExT-QA", "EgoSchema", "VideoEspresso"],
    notesKey: "s04_task"
  },
  {
    id: "s05_why_hard",
    section: "context",
    layout: "comparison",
    title: "Why Long-Video QA Is Hard",
    columns: [
      { header: "Redundancy", items: ["Most frames add little evidence"] },
      { header: "Budget", items: ["Models cannot read everything"] },
      { header: "Selectivity", items: ["Useful frames depend on the question"] }
    ],
    asset: null,
    citations: ["NExT-QA", "EgoSchema", "VideoEspresso"],
    notesKey: "s05_why_hard"
  },
  {
    id: "s06_sparse_evidence",
    section: "context",
    layout: "claim",
    title: "The Core Claim Is Semantic Sparsity",
    bullets: [
      "Only a small subset of frames is actually relevant to a given question",
      "The target subset changes with the question, so a fixed sampling policy is structurally mismatched",
      "A good system should accumulate verified evidence while keeping the active context compact"
    ],
    asset: {
      kind: "image",
      path: LOCAL_ASSETS.summary,
      caption: "Summary-as-state as compact carryover memory"
    },
    citations: ["REVISE"],
    notesKey: "s06_sparse_evidence"
  },
  {
    id: "s07_benchmarks",
    section: "context",
    layout: "benchmark_compare",
    title: "Benchmarks Stress Different Time Scales",
    bullets: [
      "NExT-QA focuses on short-to-medium videos with causal, temporal, and descriptive questions",
      "EgoSchema pushes toward longer egocentric clips where useful evidence is buried in routine activity",
      "VideoEspresso emphasizes sparse core-frame reasoning across diverse fine-grained reasoning categories"
    ],
    asset: null,
    citations: ["NExT-QA", "EgoSchema", "VideoEspresso"],
    notesKey: "s07_benchmarks"
  },
  {
    id: "s08_failure_case",
    section: "context",
    layout: "failure_case",
    title: "What Fails In A One-Shot Pass",
    bullets: [
      "Uniformly sampled frames give broad coverage but weak diagnostic evidence for the actual question",
      "If the first pass misses the decisive moment, the model has no mechanism to ask for targeted follow-up evidence",
      "The result is either bloated context or brittle answers built from partial observations"
    ],
    asset: {
      kind: "image",
      path: LOCAL_ASSETS.sketch,
      caption: "Multi-round example showing targeted follow-up frames"
    },
    citations: ["REVISE", "VideoAgent", "VideoTree"],
    notesKey: "s08_failure_case"
  },
  {
    id: "s09_field_shift",
    section: "landscape",
    layout: "trend",
    title: "The Field Is Shifting From Dense Viewing To Selective Reasoning",
    bullets: [
      "Early video-VLM systems mostly tried to pack more frames or captions into one inference call",
      "Newer work treats frame access as a bottleneck to manage, not just a larger context to fill",
      "That shift creates room for methods that retrieve evidence, reason over it, and decide when to continue"
    ],
    asset: null,
    citations: ["LLoVi", "MovieChat", "VideoAgent", "VideoTree"],
    notesKey: "s09_field_shift"
  },
  {
    id: "s10_taxonomy",
    section: "landscape",
    layout: "taxonomy",
    title: "A Useful Taxonomy Of Long-Video QA Methods",
    bullets: [
      "Dense video-LLMs trade simplicity for heavy context use and redundancy",
      "Caption or memory pipelines compress the video into text, gaining scale but often losing fine visual detail",
      "Query-aware frame selectors and agentic multi-round systems move closer to question-specific evidence gathering"
    ],
    categories: [
      {
        name: "Dense video-LLMs",
        mechanism: "Process many frames in one pass with direct visual tokens",
        strength: "Preserve raw visual detail and keep the pipeline simple",
        weakness: "Consume context budget quickly and revisit redundant evidence",
        fit: "Shorter videos or tasks where broad visual coverage matters more than targeted retrieval"
      },
      {
        name: "Caption / memory pipelines",
        mechanism: "Compress video into textual descriptions or memory before reasoning",
        strength: "Scale to longer videos by moving reasoning into text space",
        weakness: "May discard the fine-grained visual cues needed for the final answer",
        fit: "Narrative or event-level questions that tolerate abstraction"
      },
      {
        name: "Query-aware frame selectors",
        mechanism: "Rank or sample likely-relevant frames conditioned on the prompt",
        strength: "Improve efficiency by spending budget on candidate evidence",
        weakness: "Often make an early relevance decision without an evolving uncertainty state",
        fit: "Settings where one good retrieval pass is often enough"
      },
      {
        name: "Agentic multi-round reasoning",
        mechanism: "Alternate between reasoning, requesting evidence, and deciding when to stop",
        strength: "Support adaptive follow-up when the first pass is insufficient",
        weakness: "Need a stable memory representation to avoid history bloat or drift",
        fit: "Long-form QA where evidence is sparse and question-specific"
      }
    ],
    asset: null,
    citations: ["LLoVi", "MovieChat", "VideoAgent", "VideoTree", "Q-Frame", "FlexibleFrameSelection", "ActiveVideoPerception"],
    notesKey: "s10_taxonomy"
  },
  {
    id: "s11_caption_memory",
    section: "landscape",
    layout: "comparison",
    title: "Caption / Memory Pipelines Compress First, Reason Later",
    bullets: [
      "Their strength is scale: captions or textual memories let an LLM traverse long videos without raw visual overload",
      "Their weakness is fidelity: once the abstraction is textual, subtle visual cues may already be gone",
      "These methods are strongest when the question is answerable from high-level event summaries rather than precise frame evidence"
    ],
    comparisons: [
      {
        aspect: "Primary representation",
        familyA: "Text summaries, captions, or compressed memory slots",
        familyB: "Direct frame evidence remains available only indirectly"
      },
      {
        aspect: "Best-case payoff",
        familyA: "Long-horizon coverage with manageable context cost",
        familyB: "Strong for coarse event structure and high-level narrative recall"
      },
      {
        aspect: "Failure mode",
        familyA: "Early compression can hide the exact visual cue the question depends on",
        familyB: "Downstream reasoning cannot recover detail that was never preserved"
      },
      {
        aspect: "When to prefer it",
        familyA: "Questions are answerable from semantic summaries",
        familyB: "The talk task values scale more than frame-level fidelity"
      }
    ],
    asset: null,
    citations: ["LLoVi", "MovieChat", "VideoAgent"],
    notesKey: "s11_caption_memory"
  },
  {
    id: "s12_frame_selection",
    section: "landscape",
    layout: "comparison",
    title: "Query-Aware Frame Selection Improves Efficiency, But Often Stays Single-Pass",
    bullets: [
      "Selection methods try to rank or sample frames that are likely to matter for the prompt",
      "They reduce wasted compute, but many still make the key decision before enough evidence has been gathered",
      "Without an explicit evolving state, it is harder to represent what is known, what is missing, and why another frame is needed"
    ],
    columns: [
      {
        label: "What improves",
        points: [
          "Cuts redundant frame processing relative to dense uniform sampling",
          "Makes visual budget explicitly question-aware",
          "Can be paired with strong back-end VLMs without redesigning the whole stack"
        ]
      },
      {
        label: "What remains limited",
        points: [
          "Many selectors still commit to relevance judgments in effectively one pass",
          "Selection quality depends on weak early signals before ambiguity is resolved",
          "They usually lack an explicit state for beliefs, uncertainties, and next-step rationale"
        ]
      }
    ],
    asset: null,
    citations: ["VideoTree", "Q-Frame", "FlexibleFrameSelection", "AdaptiveKeyframeSampling"],
    notesKey: "s12_frame_selection"
  },
  {
    id: "s13_agentic_reasoning",
    section: "landscape",
    layout: "comparison",
    title: "Agentic Methods Add Iteration And Tool Use",
    bullets: [
      "Agentic systems can observe, reflect, and request new evidence instead of committing to a single retrieval step",
      "This is closer to how a human would solve long-video QA: inspect, update the hypothesis, then probe the missing piece",
      "The open question is what memory representation keeps those loops stable instead of letting them drift or bloat"
    ],
    columns: [
      {
        label: "What agentic loops add",
        points: [
          "Iterative evidence requests instead of fixed retrieval",
          "A natural place for uncertainty-driven follow-up",
          "The option to stop early once the answer is sufficiently supported"
        ]
      },
      {
        label: "Why memory becomes central",
        points: [
          "The agent must preserve verified evidence across rounds",
          "Full history replay is expensive and can amplify drift",
          "A compact state representation is needed to keep the loop interpretable and stable"
        ]
      }
    ],
    asset: null,
    citations: ["VideoAgent", "ActiveVideoPerception", "AIR", "VideoBrain", "FrameMind", "FrameThinker"],
    notesKey: "s13_agentic_reasoning"
  },
  {
    id: "s14_gap_to_revise",
    section: "landscape",
    layout: "gap",
    title: "Where REVISE Fits",
    bullets: [
      "REVISE combines direct visual reasoning with multi-round frame requests, so it does not have to collapse everything into captions first",
      "Its distinctive bet is summary-as-state: carry forward only compact, structured evidence instead of the full visual or conversational history",
      "That makes the method interesting both as a plug-and-play controller for closed models and as an RL target for open models"
    ],
    asset: {
      kind: "image",
      path: LOCAL_ASSETS.overview,
      caption: "Bridge from landscape to REVISE"
    },
    citations: ["REVISE", "VideoAgent", "VideoTree", "RAGEN"],
    notesKey: "s14_gap_to_revise"
  },
  {
    id: "s15_overview",
    section: "revise",
    layout: "claim",
    title: "REVISE In One View",
    figureLabel: "Three-part method",
    bullets: [
      "REVISE targets two failure modes together: information overload and weak awareness of which missing evidence actually matters",
      "The method combines a multi-round controller, a structured response protocol, and a summary-as-state carried across rounds",
      "The result is a question-aware loop that can inspect a few frames, update its state, and stop once the evidence is sufficient"
    ],
    asset: {
      kind: "image",
      path: LOCAL_ASSETS.overview,
      caption: "Overview of the multi-round REVISE loop"
    },
    citations: ["REVISE"],
    notesKey: "s15_overview"
  },
  {
    id: "s16_one_shot_vs_iterative",
    section: "revise",
    layout: "comparison",
    title: "REVISE Is Not Just Another One-Shot Selector",
    bullets: [
      "The main difference is not only which frames are chosen, but whether the model can revise its search after seeing partial evidence"
    ],
    columns: [
      {
        header: "One-shot selector",
        items: [
          "Ranks frames once from weak initial clues",
          "Has little way to represent what remains unresolved",
          "Usually spends budget before ambiguity has been reduced"
        ]
      },
      {
        header: "REVISE loop",
        items: [
          "Alternates evidence gathering with explicit state updates",
          "Uses uncertainty and rationale fields to drive the next request",
          "Can stop early when the current summary already supports an answer"
        ]
      }
    ],
    citations: ["REVISE", "VideoTree", "Q-Frame"],
    notesKey: "s16_one_shot_vs_iterative"
  },
  {
    id: "s17_summary_as_state",
    section: "revise",
    layout: "claim",
    title: "Only The Summary Persists Across Rounds",
    figureLabel: "Memory discipline",
    bullets: [
      "Each round conditions on the current question, the newly shown frames, and the latest structured summary state",
      "Raw frames and full dialogue history are not replayed as the persistent memory, which keeps the active context compact",
      "That discipline matters because the state is cumulative: the latest summary is meant to subsume what has already been established"
    ],
    asset: {
      kind: "image",
      path: LOCAL_ASSETS.summary,
      caption: "Summary-as-state as the only carryover memory"
    },
    citations: ["REVISE"],
    notesKey: "s17_summary_as_state"
  },
  {
    id: "s18_pohr",
    section: "revise",
    layout: "comparison",
    title: "POHR Makes The State Explicit",
    bullets: [
      "The summary is not free-form chat history. It is a fixed schema that separates evidence, belief updates, and next-step rationale."
    ],
    columns: [
      { header: "P", items: ["Previously seen evidence", "What has already been inspected", "Prevents the loop from forgetting its coverage"] },
      { header: "O", items: ["Current observations", "What these new frames visibly show", "Anchors the state in concrete evidence"] },
      { header: "H / U / R", items: ["Hypotheses updated from the observations", "Uncertainties that remain unresolved", "Reasons for the next frames or for stopping now"] }
    ],
    citations: ["REVISE"],
    notesKey: "s18_pohr"
  },
  {
    id: "s19_loop",
    section: "revise",
    layout: "comparison",
    title: "The Loop Is Read, Summarize, Then Either Select Or Answer",
    bullets: [
      "Each round is deliberately small: look at a few frames, update the summary, then either ask for more evidence or stop"
    ],
    columns: [
      {
        header: "1. Read",
        items: [
          "Start from a small seed set",
          "Show timestamps and basic video meta",
          "Condition on the latest summary plus current frames"
        ]
      },
      {
        header: "2. Summarize",
        items: [
          "Write a new cumulative POHR state",
          "Capture what changed and what remains unclear"
        ]
      },
      {
        header: "3. Act",
        items: [
          "Emit `<frames>` for targeted follow-up",
          "Or emit `<answer>` when uncertainty is low",
          "Maximum rounds cap cost, but early stopping is preferred"
        ]
      }
    ],
    citations: ["REVISE"],
    notesKey: "s19_loop"
  },
  {
    id: "s20_worked_example",
    section: "revise",
    layout: "failure_case",
    title: "A Worked Example Makes The Loop Concrete",
    figureLabel: "Qualitative trajectory",
    bullets: [
      "The example starts from a few uniformly sampled frames, which are enough to form an initial guess but not enough to answer confidently",
      "After the first summary, the uncertainty and rationale fields point to a more targeted follow-up request",
      "A small second round resolves the missing evidence, so the model answers using a handful of frames rather than dense coverage"
    ],
    asset: {
      kind: "image",
      path: LOCAL_ASSETS.sketch,
      caption: "Worked example of targeted follow-up across rounds"
    },
    citations: ["REVISE"],
    notesKey: "s20_worked_example"
  },
  {
    id: "s21_pnp_mode",
    section: "revise",
    layout: "comparison",
    title: "Plug-And-Play Mode Wraps A Frozen VLM",
    bullets: [
      "One practical attraction of REVISE is that the controller lives outside the model, so it can wrap both open and closed VLMs without changing weights"
    ],
    columns: [
      {
        header: "What stays fixed",
        items: [
          "The backbone VLM remains frozen",
          "Its original visual encoder and API behavior are unchanged",
          "No frame-level labels or retraining are required"
        ]
      },
      {
        header: "What REVISE adds",
        items: [
          "Multi-round orchestration around the model",
          "The structured `<summary>` plus `<frames>` or `<answer>` protocol",
          "External validity checks and iterative frame requests"
        ]
      },
      {
        header: "Why this matters",
        items: [
          "Works with proprietary APIs like GPT-4o",
          "Lets the paper test the idea before committing to training",
          "Separates the reasoning policy from the backbone weights"
        ]
      }
    ],
    citations: ["REVISE"],
    notesKey: "s21_pnp_mode"
  },
  {
    id: "s22_rl_mode",
    section: "revise",
    layout: "comparison",
    title: "RL Mode Turns The Controller Into A Learned Policy",
    bullets: [
      "For open models, the same interaction loop is recast as a finite-horizon decision problem and optimized with reinforcement learning"
    ],
    columns: [
      {
        header: "State",
        items: [
          "Current prompt with formatting and meta",
          "Latest cumulative summary state",
          "Frames already admitted so far"
        ]
      },
      {
        header: "Actions",
        items: [
          "Select a few new frame indices",
          "Or stop and emit the answer",
          "The action space mirrors the plug-and-play protocol"
        ]
      },
      {
        header: "Optimization view",
        items: [
          "Finite-horizon MDP over rounds",
          "GRPO trains the policy at the token level",
          "The goal is better frame requests, better summaries, and earlier correct stopping"
        ]
      }
    ],
    citations: ["REVISE", "RAGEN"],
    notesKey: "s22_rl_mode"
  },
  {
    id: "s23_eager_reward",
    section: "revise",
    layout: "comparison",
    title: "EAGER Rewards Evidence, State Quality, And Early Stopping",
    bullets: [
      "EAGER trains the controller without frame-level labels by rewarding useful evidence gathering and disciplined stopping"
    ],
    columns: [
      {
        header: "Evidence gain",
        items: [
          "Give credit only when new frames increase answer confidence",
          "This discourages pointless extra browsing"
        ]
      },
      {
        header: "State quality",
        items: [
          "Check whether the final answer is recoverable from the summary alone",
          "Reward valid structured output so the state stays usable"
        ]
      },
      {
        header: "Correct early stop",
        items: [
          "Reward answering correctly within a small turn budget",
          "This aligns the learned policy with sparse inference"
        ]
      }
    ],
    citations: ["REVISE"],
    notesKey: "s23_eager_reward"
  },
  {
    id: "s24_why_it_works",
    section: "revise",
    layout: "comparison",
    title: "Why This Design Helps Sparse Video Reasoning",
    bullets: [
      "REVISE works because it couples retrieval, reasoning, and memory discipline instead of treating them as separate afterthoughts"
    ],
    columns: [
      {
        header: "Less overload",
        items: [
          "Only a few frames are admitted each round",
          "The model does not keep replaying all prior visual context",
          "Budget is spent on likely evidence rather than blanket coverage"
        ]
      },
      {
        header: "Better key-information awareness",
        items: [
          "Uncertainty is explicit rather than implicit",
          "Reasons for the next frames are written down",
          "Follow-up requests become question-specific instead of generic"
        ]
      },
      {
        header: "More interpretable control",
        items: [
          "POHR exposes what the controller thinks it knows",
          "The stop-or-select decision is externally visible",
          "Failures are easier to inspect than in a dense black-box pass"
        ]
      }
    ],
    citations: ["REVISE"],
    notesKey: "s24_why_it_works"
  },
  {
    id: "s25_method_takeaway",
    section: "revise",
    layout: "context",
    title: "Method Takeaway: REVISE Is A Sparse Evidence Controller",
    bullets: [
      "Not just frame selection: it is an iterative controller with an explicit state and a visible stopping rule",
      "Not just memory: the summary is structured so the next request is grounded in observations, hypotheses, and uncertainties",
      "Not just a training recipe: the same core loop supports frozen plug-and-play use and RL fine-tuning"
    ],
    asset: {
      kind: "image",
      path: LOCAL_ASSETS.summary,
      caption: "One-slide memory of the REVISE core"
    },
    citations: ["REVISE", "RAGEN"],
    notesKey: "s25_method_takeaway"
  },
  {
    id: "s26_setup",
    section: "results",
    layout: "comparison",
    title: "How To Read The Experiments",
    bullets: [
      "The experimental question is not only whether REVISE can improve accuracy, but whether it can do so while staying in a deliberately small visual budget"
    ],
    columns: [
      {
        header: "Controller budget",
        items: [
          "Maximum 3 frames per round",
          "Maximum 4 rounds",
          "Early stopping is allowed and preferred"
        ]
      },
      {
        header: "Evaluation lens",
        items: [
          "Track both answer accuracy and total frames used",
          "Interpret results as an accuracy-efficiency trade-off",
          "Ask whether iterative access changes the regime, not just the score"
        ]
      },
      {
        header: "Plug-and-play scope",
        items: [
          "Backbones include GPT-4o, Qwen2-VL, Qwen2.5-VL, and InternVL2",
          "The controller wraps the model instead of retraining it",
          "This isolates the value of the interaction pattern itself"
        ]
      }
    ],
    citations: ["REVISE", "VideoEspresso", "NExT-QA", "EgoSchema"],
    notesKey: "s26_setup"
  },
  {
    id: "s27_main_results",
    section: "results",
    layout: "claim",
    title: "REVISE Improves Accuracy In A Small-Frame Regime",
    figureLabel: "Plug-and-play results",
    bullets: [
      "On VideoEspresso, GPT-4o rises from 26.4 to 48.9 accuracy while using about 8.0 total frames, showing that iterative access changes the answer quality regime rather than nudging it slightly",
      "The same pattern holds on NExT-QA and the EgoSchema subset: GPT-4o + REVISE reaches 63.8 with 8.4 frames and 60.6 with 9.8 frames",
      "The controller is not tied to one proprietary model: on VideoEspresso, Qwen2-VL improves from 28.5 to 37.8 and InternVL2 improves from 28.7 to 32.1"
    ],
    asset: {
      kind: "image",
      path: LOCAL_ASSETS.pareto,
      caption: "Accuracy and frame budget move together when the controller can iterate."
    },
    citations: ["REVISE", "VideoEspresso", "NExT-QA", "EgoSchema"],
    notesKey: "s27_main_results"
  },
  {
    id: "s28_efficiency",
    section: "results",
    layout: "comparison",
    title: "The Main Win Is The Small-Frame Operating Regime",
    bullets: [
      "The paper's efficiency claim is best read as: REVISE stays in single-digit frame counts while remaining competitive enough to make sparse reasoning practical"
    ],
    columns: [
      {
        header: "What small-frame means here",
        items: [
          "Most reported REVISE runs finish with roughly 8 to 10 frames total",
          "The controller is allowed four rounds, but often stops earlier",
          "This is closer to targeted evidence inspection than to dense viewing"
        ]
      },
      {
        header: "Why that matters",
        items: [
          "On NExT-QA, REVISE uses 3 to 4 times fewer inputs than LVNet and SeViLA",
          "It uses far fewer inputs than heavy-context systems like VideoTree and LLoVi",
          "On EgoSchema, it stays near VideoAgent's low-frame regime without relying on a captioner"
        ]
      },
      {
        header: "Trade-off to state plainly",
        items: [
          "The strongest high-frame baselines can still win on absolute accuracy",
          "REVISE is appealing because the controller is simple, efficient, and plug-and-play",
          "So the result is not 'best everywhere' but 'strong accuracy for very little visual budget'"
        ]
      }
    ],
    citations: ["REVISE", "LVNet", "SeViLA", "VideoTree", "LLoVi", "VideoAgent"],
    notesKey: "s28_efficiency"
  },
  {
    id: "s29_ablation_rounds",
    section: "results",
    layout: "claim",
    title: "More Iteration Can Help Without Blowing Up The Budget",
    figureLabel: "Turn-budget ablation",
    bullets: [
      "The turn ablation is useful because it shows iteration is not equivalent to indiscriminately adding more frames: one turn gets 38.3 accuracy with 4.60 frames, while four turns reach 42.1 with only 2.89 selected frames on average",
      "The three-turn and four-turn settings are especially revealing: accuracy rises to 41.6 and 42.1 while average rounds stay near the low twos, so the model is learning to use optional follow-up rather than always taking it",
      "The practical reading is that extra turns buy opportunities to correct an incomplete first guess, not a license to flood the model with context"
    ],
    asset: {
      kind: "image",
      path: LOCAL_ASSETS.pareto,
      caption: "Turn ablation shows a better accuracy-budget frontier, not just more viewing."
    },
    citations: ["REVISE"],
    notesKey: "s29_ablation_rounds"
  },
  {
    id: "s30_ablation_components",
    section: "results",
    layout: "comparison",
    title: "State Carryover And Structure Are Doing Real Work",
    bullets: [
      "The component ablation strongly suggests that REVISE works because it preserves a disciplined state, not just because it can ask for more frames"
    ],
    columns: [
      {
        header: "Full system",
        items: [
          "41.48 accuracy",
          "2.79 average turns",
          "22.71 seconds",
          "State carryover plus structured POHR"
        ]
      },
      {
        header: "No state carryover",
        items: [
          "23.14 accuracy",
          "Large drop despite the same general loop",
          "Shows that round-to-round memory is essential"
        ]
      },
      {
        header: "No structured POHR",
        items: [
          "24.27 accuracy",
          "Free-form summary is not a drop-in substitute",
          "The schema helps preserve usable reasoning state"
        ]
      },
      {
        header: "Neither",
        items: [
          "20.24 accuracy",
          "Worst result in the ablation",
          "Removing carryover and structure largely collapses the method"
        ]
      }
    ],
    citations: ["REVISE"],
    notesKey: "s30_ablation_components"
  },
  {
    id: "s31_empirical_takeaway",
    section: "results",
    layout: "context",
    title: "Empirical Takeaway: The Controller Matters",
    bullets: [
      "The experiments support the claim that better control over frame access can produce large gains even when the backbone is unchanged",
      "Those gains appear in a sparse regime: REVISE is most compelling when we care about what accuracy we can buy with only a few visual glimpses",
      "The ablations point back to the paper's conceptual center: iterative retrieval helps, but it helps most when the state is cumulative and structured"
    ],
    citations: ["REVISE"],
    notesKey: "s31_empirical_takeaway"
  },
  {
    id: "s32_takeaways",
    section: "closing",
    layout: "context",
    title: "Three Takeaways",
    bullets: [
      "Long-video QA is fundamentally a selective evidence problem",
      "REVISE turns frame access into an iterative reasoning loop",
      "Summary-as-state is the conceptual center of the paper"
    ],
    citations: [],
    notesKey: "s32_takeaways"
  },
  {
    id: "s33_open_questions",
    section: "closing",
    layout: "comparison",
    title: "Open Questions After A Sympathetic Reading",
    bullets: [
      "The remaining questions are mostly about where this controller pattern scales cleanly and where richer memory or stronger retrieval policies would be needed"
    ],
    columns: [
      {
        header: "Where might it break",
        items: [
          "Questions that require many weak clues spread over long time ranges",
          "Tasks where the missing evidence is hard to localize from an early summary",
          "Settings where fine spatial detail matters more than sparse event cues"
        ]
      },
      {
        header: "What seems promising",
        items: [
          "Better state representations without replaying raw history",
          "Controllers that combine sparse frame access with stronger temporal tools",
          "Hybrid systems that keep direct frame access but add lightweight external memory"
        ]
      },
      {
        header: "Discussion lens",
        items: [
          "How much of the gain comes from the protocol versus the backbone following it well",
          "When should we prefer efficiency and inspectability over absolute peak accuracy",
          "What would a harder benchmark for iterative evidence gathering look like"
        ]
      }
    ],
    citations: ["REVISE", "VideoAgent", "VideoTree"],
    notesKey: "s33_open_questions"
  },
  {
    id: "s34_qa",
    section: "closing",
    layout: "context",
    title: "Q&A",
    bullets: [
      "Appendix backup slides cover extra benchmark numbers, EAGER reward design, component ablations, and a wider related-work comparison",
      "If useful, we can discuss the paper either as a systems idea for sparse video access or as an RL problem over select-versus-answer decisions",
      "The question I would keep on the table is simple: what is the right memory object for iterative long-video reasoning"
    ],
    citations: [],
    notesKey: "s34_qa"
  },
  {
    id: "a01_more_benchmarks",
    section: "appendix",
    layout: "comparison",
    title: "Appendix: Extra Benchmark Numbers",
    bullets: [
      "These backup numbers are useful when the discussion shifts from the main plug-and-play story to how the learned policy behaves in the same sparse-control framework"
    ],
    columns: [
      {
        header: "VideoEspresso RL",
        items: [
          "27.8 accuracy",
          "4.1 frames",
          "1.37 rounds",
          "1.02 seconds"
        ]
      },
      {
        header: "NExT-QA RL",
        items: [
          "51.3 accuracy",
          "3.9 frames",
          "1.32 rounds",
          "0.62 seconds"
        ]
      },
      {
        header: "How to read it",
        items: [
          "RL keeps the sparse regime even tighter than plug-and-play",
          "The learned policy is still making select-versus-answer trade-offs under a tiny frame budget",
          "This supports the idea that the loop itself is learnable, not only promptable"
        ]
      }
    ],
    citations: ["REVISE", "VideoEspresso", "NExT-QA"],
    notesKey: "a01_more_benchmarks"
  },
  {
    id: "a02_reward_details",
    section: "appendix",
    layout: "comparison",
    title: "Appendix: EAGER Reward Details",
    bullets: [
      "EAGER is best read as a reward decomposition that tries to teach useful browsing behavior without requiring dense supervision over which frames were 'correct'"
    ],
    columns: [
      {
        header: "Confidence gain",
        items: [
          "Reward new frames only if they increase answer confidence",
          "Discourages pointless extra queries",
          "Pushes the controller toward informative follow-up"
        ]
      },
      {
        header: "Summary sufficiency",
        items: [
          "Checks whether the final answer is recoverable from the summary state",
          "Directly trains the carryover state to stay useful",
          "Aligns memory quality with final-task utility"
        ]
      },
      {
        header: "Correct-and-early stop",
        items: [
          "Reward correct answers that stop within the budget",
          "Makes sparse completion a first-class objective",
          "Prevents the policy from treating more rounds as free"
        ]
      },
      {
        header: "Format validity",
        items: [
          "Small reward for valid structured output",
          "Keeps the protocol machine-readable",
          "Important because the loop depends on well-formed tags and fields"
        ]
      }
    ],
    citations: ["REVISE"],
    notesKey: "a02_reward_details"
  },
  {
    id: "a03_component_ablation",
    section: "appendix",
    layout: "comparison",
    title: "Appendix: Stronger Read Of The Component Ablation",
    bullets: [
      "The ablation table is worth keeping in backup because it sharpens the paper's mechanistic claim: summary design is not cosmetic, it is load-bearing"
    ],
    columns: [
      {
        header: "Carryover",
        items: [
          "Removing carryover drops accuracy from 41.48 to 23.14",
          "The model loses cumulative evidence across rounds",
          "Iteration without memory becomes mostly reset behavior"
        ]
      },
      {
        header: "POHR structure",
        items: [
          "Removing structured fields drops accuracy to 24.27",
          "The model still writes text, but the state becomes less usable",
          "Explicit evidence and uncertainty slots matter"
        ]
      },
      {
        header: "Combined removal",
        items: [
          "No carryover plus no structure falls to 20.24",
          "That is below either single ablation",
          "The two design choices reinforce each other"
        ]
      }
    ],
    citations: ["REVISE"],
    notesKey: "a03_component_ablation"
  },
  {
    id: "a04_related_work_extra",
    section: "appendix",
    layout: "comparison",
    title: "Appendix: Extra Related-Work Framing",
    bullets: [
      "If the discussion turns outward, REVISE is easiest to place by asking what each family chooses as its primary memory object and control policy"
    ],
    columns: [
      {
        header: "Caption / memory systems",
        items: [
          "Primary memory is text or compressed summaries",
          "Strength is scale across long videos",
          "Risk is losing answer-critical visual detail too early"
        ]
      },
      {
        header: "Single-pass selectors",
        items: [
          "Primary control move is one relevance decision",
          "Strength is cheaper inference than dense viewing",
          "Risk is committing before uncertainty is well characterized"
        ]
      },
      {
        header: "REVISE-like agentic loops",
        items: [
          "Primary memory is an evolving reasoning state",
          "Strength is targeted follow-up and visible stopping logic",
          "Open question is how rich that state must become for harder tasks"
        ]
      }
    ],
    citations: ["REVISE", "MovieChat", "VideoAgent", "VideoTree", "Q-Frame"],
    notesKey: "a04_related_work_extra"
  }
];

module.exports = { slides };
