## [LRN-20260307-001] best_practice

**Logged**: 2026-03-07T10:11:19-06:00
**Priority**: medium
**Status**: pending
**Area**: config

### Summary
Example entrypoints under `examples/` must bootstrap the repo root before importing `examples.*` modules.

### Details
Several README-documented commands use `python examples/...py`. When Python executes a script by path, the script directory is added to `sys.path`, not the repository root. That caused `ModuleNotFoundError: No module named 'examples'` for multiple entrypoints even though the imports were otherwise valid.

Cross-script reuse was also fragile because some scripts imported stale private helper names from sibling files. Shared helpers in `examples/revise/pnp_utils.py` are the stable import source for utilities such as `get_model_id`, `wait_port`, `stop_server`, `extract_frames_1fps`, and related helpers.

### Suggested Action
Keep the repo-root bootstrap snippet at the top of direct-execution scripts, and prefer importing shared helpers from `examples/revise/pnp_utils.py` instead of duplicating or depending on private aliases in other scripts.

### Metadata
- Source: conversation
- Related Files: examples/revise/plug_and_play_nextqa_vllm.py, examples/revise/oneshot_videomme_lvbench_vllm.py, examples/videoagent/eval_nextqa_videoagent_caption_vllm.py
- Tags: examples, imports, entrypoint, pythonpath

---
## [LRN-20260307-002] best_practice

**Logged**: 2026-03-07T11:02:08-06:00
**Priority**: medium
**Status**: pending
**Area**: tooling

### Summary
Standalone scripts under `scripts/` need the same repo-root bootstrap as `examples/` entrypoints when they are executed via `python path/to/script.py`.

### Details
`python scripts/repro/doctor.py` initially failed with `ModuleNotFoundError: No module named 'scripts'` because Python added `scripts/repro/` to `sys.path`, not the repository root. Adding the repo root explicitly before importing `scripts.repro.*` fixed both `doctor.py` and `paper_suite.py`.

### Suggested Action
For any new direct-execution script outside an installed package, prepend the repository root to `sys.path` before importing sibling modules.

### Metadata
- Source: conversation
- Related Files: scripts/repro/doctor.py, scripts/repro/paper_suite.py
- Tags: scripts, pythonpath, entrypoint

---
## [LRN-20260307-003] best_practice

**Logged**: 2026-03-07T11:02:08-06:00
**Priority**: medium
**Status**: pending
**Area**: infra

### Summary
Environment doctor scripts should not rely only on `importlib.metadata.version()` when checking packages installed from nonstandard layouts.

### Details
`decord` imported successfully in the target env but `importlib.metadata.version('decord')` returned `PackageNotFoundError`, which produced a false negative in `doctor.py`. Falling back to importing the module and reading `__version__` fixed the check.

### Suggested Action
When a package version lookup is advisory rather than security-critical, use a metadata lookup first and then a module import fallback.

### Metadata
- Source: conversation
- Related Files: scripts/repro/common.py
- Tags: doctor, packaging, importlib

---
## [LRN-20260307-004] best_practice

**Logged**: 2026-03-07T11:37:00-06:00
**Priority**: medium
**Status**: pending
**Area**: infra

### Summary
Paper-level smoke runs should prefer the local 3B Qwen2.5-VL checkpoint even when the full experiment uses 7B.

### Details
Real smoke validation showed that the 7B checkpoint on shared storage can spend long periods in model-load startup, while the 3B checkpoint consistently reaches a serving state and exercises the same request/response code path. For smoke tests the goal is to validate orchestration, dataset loading, prompt formatting, and API compatibility, not final benchmark quality.

### Suggested Action
Keep smoke variants on 3B by default and reserve 7B for full reproductions or remote API runs.

### Metadata
- Source: conversation
- Related Files: scripts/repro/paper_suite.py
- Tags: smoke, vllm, model-size, reproducibility

---
## [LRN-20260307-005] best_practice

**Logged**: 2026-03-07T11:37:00-06:00
**Priority**: high
**Status**: pending
**Area**: config

### Summary
Editable installs for paper reproduction should be done after backend-pinned requirements, using `pip install -e . --no-deps`.

### Details
Running `pip install -e .` first in a clean env caused pip to resolve broad dependencies to current latest releases such as new `torch`, `transformers`, and `ray` versions, which drifted away from the versions actually used by the repo's working environment. Installing the backend-specific runtime stack first and then adding the editable repo without dependency resolution preserved a reproducible set of packages.

### Suggested Action
For reproducibility tooling, install pinned backend/runtime packages first, then run `pip install -e . --no-deps`.

### Metadata
- Source: conversation
- Related Files: scripts/repro/setup_env.sh, setup.py
- Tags: pip, editable, dependency-resolution, environment

---
