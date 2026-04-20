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
      path: "assets/overview.png",
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
      path: "assets/summary_as_state.png",
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
      path: "assets/sketch.png",
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
      path: "assets/overview.png",
      caption: "Bridge from landscape to REVISE"
    },
    citations: ["REVISE", "VideoAgent", "VideoTree", "RAGEN"],
    notesKey: "s14_gap_to_revise"
  }
];

module.exports = { slides };
