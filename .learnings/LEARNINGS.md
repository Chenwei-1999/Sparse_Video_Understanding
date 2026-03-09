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
## [LRN-20260307-006] best_practice

**Logged**: 2026-03-07T22:58:00-06:00
**Priority**: high
**Status**: pending
**Area**: infra

### Summary
EgoSchema HF fallback should extract only the requested mp4 from the public chunked zip archives instead of assuming a rawvideo repo contains every subset video.

### Details
The `VLM2Vec/egoschema` metadata split is small and public, but many `Subset` video IDs are not present as top-level files in `VLM2Vec/egoschema-rawvideo`. The public `videos_chunked_*.zip` archives in the main dataset repo do contain the required mp4s, and they can be accessed lazily with `fsspec` + `zipfile` over Hugging Face HTTP range requests. That makes on-demand EgoSchema video retrieval feasible without pre-downloading ~100 GB of archives.

### Suggested Action
Keep the EgoSchema fallback on the main chunked dataset repo, cache resolved chunk lookups, and extract only requested members into a local video cache.

### Metadata
- Source: conversation
- Related Files: examples/revise/plug_and_play_egoschema_vllm.py
- Tags: egoschema, huggingface, zip, lazy-download, reproducibility

---
## [LRN-20260307-007] best_practice

**Logged**: 2026-03-07T22:58:00-06:00
**Priority**: high
**Status**: pending
**Area**: config

### Summary
Manual training shell pipelines should accept explicit `PYTHON_BIN` and conservative server defaults instead of assuming the caller already activated the right environment and has four visible GPUs.

### Details
The `run_generate_teacher_data*.sh` and downstream SFT/RL shell wrappers previously hard-coded `python3` and inherited the evaluator default `--tensor-parallel-size 4`. In practice that broke smoke validation when the caller had not activated the intended conda env or only had one or two visible GPUs. Passing `PYTHON_BIN`, `TORCHRUN_BIN`, `TENSOR_PARALLEL_SIZE`, and `GPU_MEMORY_UTILIZATION` through the shell entrypoints makes the same scripts usable both in full runs and in constrained smoke environments.

### Suggested Action
Keep shell entrypoints parameterized via environment variables and have reproduction tooling export the validated interpreter path explicitly.

### Metadata
- Source: conversation
- Related Files: examples/revise/run_generate_teacher_data.sh, examples/revise/run_generate_teacher_data_videoespresso.sh, examples/revise/run_revise_nextqa_sft.sh, examples/revise/run_revise_videoespresso_sft.sh, scripts/repro/paper_suite.py
- Tags: shell, python-bin, tensor-parallel, smoke, environment

---
## [LRN-20260308-008] best_practice

**Logged**: 2026-03-08T17:17:06-05:00
**Priority**: high
**Status**: pending
**Area**: config

### Summary
In `verl.trainer.ppo.ray_trainer`, `trainer.total_training_steps` does not extend training past dataloader exhaustion; the loop still stops when `trainer.total_epochs * len(train_dataloader)` is exhausted.

### Details
A VideoEspresso smoke dataset with 4 training examples consistently stopped after 4 steps even when `trainer.total_training_steps=20` was overridden. The trainer uses `total_training_steps` for scheduler/progress bookkeeping and `is_last_step`, but the outer loop is still `for epoch in range(total_epochs)` over the finite dataloader. To actually execute 20 PPO updates on a 4-sample smoke dataset, `trainer.total_epochs` must be at least 5.

### Suggested Action
When running long smoke jobs on tiny datasets, set both `trainer.total_training_steps` and `trainer.total_epochs` consistently so `total_epochs * len(train_dataloader) >= total_training_steps`.

### Metadata
- Source: conversation
- Related Files: verl/trainer/ppo/ray_trainer.py, examples/revise/config/revise_videoespresso_grpo_smoke.yaml
- Tags: trainer, epochs, dataloader, smoke, grpo
- See Also: ERR-20260308-010

---
