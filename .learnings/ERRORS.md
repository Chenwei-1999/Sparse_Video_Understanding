## [ERR-20260307-001] exec_command

**Logged**: 2026-03-07T10:11:19-06:00
**Priority**: low
**Status**: pending
**Area**: infra

### Summary
A compound shell validation command was rejected by the local execution policy even though the underlying script was valid.

### Error
```text
Rejected("`/usr/bin/bash -lc 'rm -f /tmp/nextqa_sft_train.parquet /tmp/nextqa_sft_val.parquet && /shares/hlw3876/chenwei/miniconda3/envs/verlrun/bin/python examples/revise/generate_sft_data.py --input outputs/nextqa_pnp_7b_train_log.jsonl --val-csv /shares/hlw3876/chenwei/NExT-QA/nextqa/val.csv --output /tmp/nextqa_sft_train.parquet --val_ratio 0.1 && ls -lh /tmp/nextqa_sft_train.parquet /tmp/nextqa_sft_val.parquet'` rejected: blocked by policy")
```

### Context
- Command attempted to clear prior temp files, run `generate_sft_data.py`, and list generated outputs in one shell line.
- Retrying with a simpler command structure succeeded.

### Suggested Fix
When validating scripts through the execution tool, avoid overly compound shell lines and prefer smaller commands with fresh output paths.

### Metadata
- Reproducible: unknown
- Related Files: examples/revise/generate_sft_data.py

---
## [ERR-20260307-002] direct_exec_import

**Logged**: 2026-03-07T11:02:08-06:00
**Priority**: low
**Status**: fixed
**Area**: tooling

### Summary
Direct execution of `scripts/repro/doctor.py` failed because the repository root was not on `sys.path`.

### Error
```text
Traceback (most recent call last):
  File "/home/cxk2993/VideoReasoning/scripts/repro/doctor.py", line 9, in <module>
    from scripts.repro.common import discover_assets
ModuleNotFoundError: No module named 'scripts'
```

### Context
- Reproduced with `/shares/hlw3876/chenwei/miniconda3/envs/verlrun/bin/python scripts/repro/doctor.py`.
- Fixed by prepending the repo root to `sys.path` in both `doctor.py` and `paper_suite.py`.

### Suggested Fix
Direct-execution scripts that import repo-local modules should bootstrap the repo root explicitly.

### Metadata
- Reproducible: yes
- Related Files: scripts/repro/doctor.py, scripts/repro/paper_suite.py

---
## [ERR-20260307-003] shared_storage_model_load

**Logged**: 2026-03-07T11:02:08-06:00
**Priority**: medium
**Status**: pending
**Area**: infra

### Summary
Smoke inference processes loading local VLM weights from shared storage entered `D` state before GPU allocation, which stalled end-to-end validation.

### Error
```text
ps showed python / vllm processes in `D` (uninterruptible sleep) while loading Qwen2.5-VL checkpoints from /shares/hlw3876/chenwei/hf_cache, and `nvidia-smi` reported no compute process yet.
```

### Context
- Affected commands included `examples/revise/oneshot_local_mc_vllm.py` and `examples/revise/oneshot_lvbench_hf.py` smoke runs.
- This appears to be an infrastructure / filesystem latency issue rather than a Python exception in repo code.

### Suggested Fix
For smoke validation on shared clusters, prefer one of:
- a remote OpenAI-compatible API via `--base-url` / `--model-id`
- a local SSD copy of the checkpoint
- a prewarmed long-lived vLLM service instead of per-run model startup

### Metadata
- Reproducible: unknown
- Related Files: examples/revise/oneshot_local_mc_vllm.py, examples/revise/oneshot_lvbench_hf.py, scripts/repro/paper_suite.py

---
## [ERR-20260307-004] env_setup_backend_conflict

**Logged**: 2026-03-07T11:37:00-06:00
**Priority**: high
**Status**: fixed
**Area**: infra

### Summary
`scripts/repro/setup_env.sh` originally installed `vllm` and `sglang` into the same environment by default, leaving incompatible transitive dependencies.

### Error
```text
pip check after running setup_env.sh reported vllm requirements broken by the later sglang install, including grpcio, grpcio-reflection, llguidance, outlines-core, and xgrammar mismatches.
```

### Context
- Reproduced by running `bash scripts/repro/setup_env.sh` in the target reproduction environment.
- The mixed-backend default made the environment non-reproducible for local vLLM smoke tests.

### Suggested Fix
Default to a single backend per environment, refuse mixed installs, and direct users to create separate envs for vLLM and SGLang.

### Metadata
- Reproducible: yes
- Related Files: scripts/repro/setup_env.sh, paper/REPRODUCE.md

---
## [ERR-20260307-005] env_setup_decord_version

**Logged**: 2026-03-07T11:37:00-06:00
**Priority**: medium
**Status**: fixed
**Area**: infra

### Summary
Fresh vLLM environment creation failed because `decord==3.0.0` is not available on PyPI.

### Error
```text
ERROR: Could not find a version that satisfies the requirement decord==3.0.0
ERROR: No matching distribution found for decord==3.0.0
```

### Context
- Reproduced while creating a fresh `vllm`-only conda env with `scripts/repro/setup_env.sh`.
- The existing machine had `decord 3.0.0` from a nonstandard installation path, but that version is not installable through pip in a clean environment.

### Suggested Fix
Use an installable `decord` spec in the reproduction script instead of pinning the local non-PyPI build version.

### Metadata
- Reproducible: yes
- Related Files: scripts/repro/setup_env.sh

---
## [ERR-20260307-006] longvideo_pnp_model_id_scope

**Logged**: 2026-03-07T11:37:00-06:00
**Priority**: high
**Status**: fixed
**Area**: backend

### Summary
The long-video plug-and-play script shadowed `model_id` inside its nested sample processor and crashed before the first request.

### Error
```text
server_error: UnboundLocalError: local variable 'model_id' referenced before assignment
```

### Context
- Reproduced by running `python scripts/repro/paper_suite.py run --experiment videomme_pnp --smoke`.
- The nested `_process_one()` function reassigned `model_id` on restart but did not declare it `nonlocal`.

### Suggested Fix
Declare `model_id` as `nonlocal` in `_process_one()` so restart logic updates the outer value.

### Metadata
- Reproducible: yes
- Related Files: examples/revise/plug_and_play_videomme_lvbench_vllm.py

---
## [ERR-20260307-007] longvideo_oneshot_start_order

**Logged**: 2026-03-07T11:37:00-06:00
**Priority**: high
**Status**: fixed
**Area**: backend

### Summary
The long-video one-shot script queried `/v1/models` before starting the local vLLM server in `--start-server` mode.

### Error
```text
requests.exceptions.ConnectionError: HTTPConnectionPool(host='127.0.0.1', port=18123): Max retries exceeded with url: /v1/models
```

### Context
- Reproduced by running `python scripts/repro/paper_suite.py run --experiment lvbench_oneshot --smoke`.
- The script resolved `model_id` before the `args.start_server` branch launched and waited for the local server.

### Suggested Fix
Start and wait for the local server first, then resolve `model_id`.

### Metadata
- Reproducible: yes
- Related Files: examples/revise/oneshot_videomme_lvbench_vllm.py

---
## [ERR-20260307-008] egoschema_rawvideo_missing_subset_files

**Logged**: 2026-03-07T22:58:00-06:00
**Priority**: high
**Status**: fixed
**Area**: infra

### Summary
The public `VLM2Vec/egoschema-rawvideo` repo does not expose every video referenced by the public EgoSchema `Subset` metadata.

### Error
```text
EntryNotFoundError 404 Client Error: Entry Not Found for url: https://huggingface.co/datasets/VLM2Vec/egoschema-rawvideo/resolve/main/0074f737-11cb-497d-8d07-77c3a8127391.mp4
```

### Context
- Reproduced while implementing HF fallback for `examples/revise/plug_and_play_egoschema_vllm.py`.
- The `Subset` metadata row referenced `video_idx=0074f737-11cb-497d-8d07-77c3a8127391`, but that file was absent from the rawvideo repo.
- The required videos were instead found inside the main dataset repo's `videos_chunked_*.zip` archives.

### Suggested Fix
Fall back to the chunked zip archives in `VLM2Vec/egoschema` and lazily extract the requested member via HTTP range reads.

### Metadata
- Reproducible: yes
- Related Files: examples/revise/plug_and_play_egoschema_vllm.py

---
## [ERR-20260307-009] videoespresso_mc_prepare_quadratic_sampling

**Logged**: 2026-03-07T22:58:00-06:00
**Priority**: medium
**Status**: fixed
**Area**: backend

### Summary
The first implementation of `prepare_videoespresso_mc_train.py` scaled poorly on the full 200k-row train split because it repeatedly shuffled near-global distractor pools per sample.

### Error
```text
The conversion process stayed CPU-bound at 100% for tens of seconds without producing output because distractor selection rebuilt and shuffled large candidate lists for each row.
```

### Context
- Reproduced while generating `outputs/videoespresso_train_mc.json` from the public VideoEspresso train JSON.
- The script was effectively doing repeated large-pool shuffles, which is unnecessary when each row only needs four distractors.

### Suggested Fix
Deduplicate answer pools once, then sample small deterministic candidate subsets per row instead of shuffling the full pool every time.

### Metadata
- Reproducible: yes
- Related Files: scripts/repro/prepare_videoespresso_mc_train.py

---
## [ERR-20260307-010] exec_command_rm_blocked

**Logged**: 2026-03-07T22:58:00-06:00
**Priority**: low
**Status**: pending
**Area**: tooling

### Summary
`functions.exec_command` rejected a simple `rm -f` cleanup command under the active policy.

### Error
```text
Rejected("`/usr/bin/bash -lc 'rm -f outputs/paper_suite_smoke/egoschema_pnp.summary.json outputs/paper_suite_smoke/egoschema_pnp.jsonl'` rejected: blocked by policy")
```

### Context
- Triggered while trying to remove stale EgoSchema smoke artifacts before rerunning the experiment.
- Non-destructive overwrite paths still worked, so the run was able to continue.

### Suggested Fix
Prefer overwrite-in-place or use non-`rm` cleanup patterns when the command policy is restrictive.

### Metadata
- Reproducible: unknown
- Related Files: outputs/paper_suite_smoke/egoschema_pnp.summary.json

---
## [ERR-20260307-011] videoespresso_teacher_shell_default_env

**Logged**: 2026-03-07T22:58:00-06:00
**Priority**: high
**Status**: fixed
**Area**: config

### Summary
The teacher-generation shell wrappers depended on the caller's ambient `python3` and the evaluator's default tensor-parallel size, which broke smoke validation outside an activated conda env.

### Error
```text
RuntimeError: No samples loaded (check dataset source, JSON, video cache, or HF video availability).
```

### Context
- Reproduced while running `run_generate_teacher_data_videoespresso.sh` from the workspace shell without exporting the validated conda python.
- A separate issue was that the script inherited the evaluator default `--tensor-parallel-size 4`, which is invalid on the current two-GPU machine.

### Suggested Fix
Parameterize the shell wrappers with `PYTHON_BIN`, `TORCHRUN_BIN`, `TENSOR_PARALLEL_SIZE`, and `GPU_MEMORY_UTILIZATION`, then export those from reproduction tooling.

### Metadata
- Reproducible: yes
- Related Files: examples/revise/run_generate_teacher_data.sh, examples/revise/run_generate_teacher_data_videoespresso.sh, scripts/repro/paper_suite.py

---
## [ERR-20260308-001] videoespresso_rl_smoke_missing_micro_batch

**Logged**: 2026-03-08T03:26:00-06:00
**Priority**: high
**Status**: fixed
**Area**: config

### Summary
`verl` GRPO smoke runs fail early if `actor.use_dynamic_bsz=False` is paired with no explicit actor/log-prob micro-batch sizes.

### Error
```text
AssertionError: [actor] Please set at least one of 'actor.ppo_micro_batch_size' or 'actor.ppo_micro_batch_size_per_gpu' if use_dynamic_bsz is not enabled.
```

### Context
- Reproduced while launching a `1-step` VideoEspresso RL smoke from `revise_videoespresso_grpo_after_sft`.
- The smoke override intentionally disabled dynamic batch sizing and reduced `ppo_mini_batch_size` to `1`, but did not also set the required actor micro-batch size.
- `verl/workers/engine_workers.py` also expects `rollout.log_prob_micro_batch_size_per_gpu` when dynamic batching is disabled.

### Suggested Fix
Provide a dedicated smoke config that sets `actor.ppo_micro_batch_size_per_gpu=1`, `ref.log_prob_micro_batch_size_per_gpu=1`, and `rollout.log_prob_micro_batch_size_per_gpu=1` together with the 1-GPU overrides.

### Metadata
- Reproducible: yes
- Related Files: examples/revise/config/revise_videoespresso_grpo_after_sft.yaml, verl/workers/config/actor.py, verl/workers/engine_workers.py

---
## [ERR-20260308-002] local_mc_dataset_dictconfig_and_all_video_paths

**Logged**: 2026-03-08T03:36:00-06:00
**Priority**: high
**Status**: fixed
**Area**: backend

### Summary
`LocalMCDataset` returned zero samples under Hydra because it only read plain `dict` configs and did not resolve `VideoEspresso` `all_video/...` validation paths correctly.

### Error
```text
ValueError: num_samples should be a positive integer value, but got num_samples=0
```

### Context
- Reproduced during a clean-env `1-step` VideoEspresso RL smoke using `revise_videoespresso_grpo_smoke.yaml`.
- The dataset loader received a Hydra `DictConfig`, but `LocalMCDataset` only extracted `local_mc.video_root` when `isinstance(config, dict)`, so the effective `video_root` became empty and all relative videos were dropped.
- Validation JSON rows also use paths like `all_video/...`, which need to resolve against `test_video/all_video/...` instead of naively appending another `all_video` segment.

### Suggested Fix
Treat config inputs as generic mappings / `DictConfig`, and add explicit handling for `all_video/...` relative paths in the local video resolver.

### Metadata
- Reproducible: yes
- Related Files: verl/utils/dataset/local_mc_dataset.py, examples/revise/config/revise_videoespresso_grpo_smoke.yaml

---
## [ERR-20260308-003] clean_env_flash_attn_default_breaks_training

**Logged**: 2026-03-08T03:44:00-06:00
**Priority**: high
**Status**: fixed
**Area**: backend

### Summary
Training-side model loading still defaulted to `flash_attention_2`, which breaks the clean `vllm` env because `flash_attn` is intentionally not installed there.

### Error
```text
ImportError: FlashAttention2 has been toggled on, but it cannot be used due to the following error: the package flash_attn seems to be not installed.
```

### Context
- Reproduced after fixing `LocalMCDataset` and rerunning the clean-env `1-step` VideoEspresso RL smoke.
- The smoke config explicitly requested `sdpa`, but multiple trainer / worker paths still constructed Hugging Face configs and models with `flash_attention_2`.
- This only surfaced in the fresh env because the older shared env already had extra packages and looser defaults.

### Suggested Fix
Centralize attention-implementation resolution and automatically fall back from `flash_attention_2` / `flash_attention_3` to `sdpa` when `flash_attn` is unavailable, then use that resolver in every training-side model loader.

### Metadata
- Reproducible: yes
- Related Files: verl/utils/model.py, verl/workers/config/model.py, verl/workers/fsdp_workers.py, verl/trainer/fsdp_sft_trainer.py

---
## [ERR-20260308-004] agent_loop_smoke_batch_smaller_than_num_workers

**Logged**: 2026-03-08T03:52:00-06:00
**Priority**: medium
**Status**: fixed
**Area**: config

### Summary
Agent-loop rollout defaults to `num_workers=8`, so a `train_batch_size=1` smoke run crashes when `DataProto.chunk()` requires equal splits.

### Error
```text
AssertionError: only support equal chunk. Got size of DataProto 1 and chunk 8.
```

### Context
- Reproduced in the clean-env `1-step` VideoEspresso RL smoke after the model and rollout engine had fully initialized.
- `revise_videoespresso_grpo_smoke.yaml` reduced the training batch to `1` but inherited the rollout default `actor_rollout_ref.rollout.agent.num_workers=8` from `ppo_trainer`.
- `AgentLoopManager.generate_sequences()` chunks the prompt batch by the exact worker count and currently does not support uneven splits.

### Suggested Fix
For smoke configs, explicitly set `actor_rollout_ref.rollout.agent.num_workers=1` whenever the rollout batch is reduced to a single sample.

### Metadata
- Reproducible: yes
- Related Files: examples/revise/config/revise_videoespresso_grpo_smoke.yaml, verl/experimental/agent_loop/agent_loop.py

---
## [ERR-20260308-005] revise_agent_loop_imageio_inf_nframes

**Logged**: 2026-03-08T03:38:00-06:00
**Priority**: high
**Status**: fixed
**Area**: backend

### Summary
`ReviseAgentLoop` crashed on some `VideoEspresso` mp4 files because `imageio/ffmpeg` reported `nframes=inf`, and the loop cast that directly to `int`.

### Error
```text
OverflowError: cannot convert float infinity to integer
```

### Context
- Reproduced during the clean-env `1-step` VideoEspresso GRPO smoke after rollout reached the first agent-loop worker.
- For `/shares/hlw3876/chenwei/VideoEspresso/train_video/all_video/Moviechat/videos/1/AWG-5.mp4`, `imageio.get_reader(...).get_meta_data()` returned `{'nframes': inf, 'fps': 25.0, 'duration': 450.0, ...}`.
- `decord.VideoReader` on the same file returned a valid frame count (`11250`), so the failure was caused by trusting unsanitized fallback metadata rather than by a broken asset.

### Suggested Fix
Prefer `decord` for video metadata, and sanitize all frame-count / fps conversions so non-finite values fall back to `None` or a derived finite estimate before any integer cast.

### Metadata
- Reproducible: yes
- Related Files: verl/experimental/agent_loop/revise_agent_loop.py

---
## [ERR-20260308-006] decord_cp36_wheel_metadata_breaks_pip_check

**Logged**: 2026-03-08T03:45:00-06:00
**Priority**: medium
**Status**: fixed
**Area**: environment

### Summary
The published `decord` wheel installed into the clean Python 3.10 env carried a stale `cp36-cp36m` wheel tag, so `pip check` reported the environment as unsupported even though runtime import worked.

### Error
```text
decord 0.6.0 is not supported on this platform
```

### Context
- Reproduced in `/shares/hlw3876/chenwei/miniconda3/envs/vr-paper-vllm-clean` immediately after a clean `scripts/repro/setup_env.sh` install.
- The installed `WHEEL` metadata was `Tag: cp36-cp36m-manylinux2010_x86_64`, but the package content is Python wrappers plus `libdecord.so`, not a CPython-ABI-bound extension wheel.
- `decord` imported and decoded video correctly on Python 3.10; only metadata validation was failing.

### Suggested Fix
Post-install, normalize the stale `decord` wheel tag to a generic `py3-none-manylinux2010_x86_64` tag after verifying that `decord` imports successfully.

### Metadata
- Reproducible: yes
- Related Files: scripts/repro/setup_env.sh

---
## [ERR-20260308-007] videoespresso_rl_smoke_dataloader_worker_killed_on_exit

**Logged**: 2026-03-08T03:42:00-06:00
**Priority**: medium
**Status**: fixed
**Area**: config

### Summary
The `1-step` VideoEspresso RL smoke could finish the training step and save `global_step_1`, but still terminate noisily because background DataLoader workers were killed during teardown.

### Error
```text
RuntimeError: DataLoader worker (pid ...) is killed by signal: Killed.
```

### Context
- Reproduced after the clean-env smoke had already logged `training/global_step:1` and saved the actor checkpoint.
- The default `data.dataloader_num_workers=8` is unnecessary for a 4-sample smoke dataset and inflated host memory usage during teardown.
- The same run reported `perf/cpu_memory_used_gb` in the high hundreds of GB, which is excessive for smoke validation.

### Suggested Fix
Set `data.dataloader_num_workers=0` in the dedicated smoke config so the validation run exits cleanly after checkpoint save.

### Metadata
- Reproducible: yes
- Related Files: examples/revise/config/revise_videoespresso_grpo_smoke.yaml

---
## [ERR-20260308-008] workspace_quota_breaks_smoke_checkpoint_save

**Logged**: 2026-03-08T03:47:00-06:00
**Priority**: medium
**Status**: fixed
**Area**: config

### Summary
The clean-env VideoEspresso RL smoke could reach `global_step_1`, but saving a full 3B Hugging Face checkpoint into the repo workspace failed under the home-directory disk quota.

### Error
```text
OSError: [Errno 122] Disk quota exceeded
```

### Context
- Reproduced when `trainer.default_local_dir` was overridden to `/home/cxk2993/VideoReasoning/outputs/...` for the smoke run.
- The checkpoint save failed in `transformers.save_pretrained()` while writing `model.safetensors.index.json`, after the training step itself had already completed.
- Smoke validation does not need to spend home-directory quota; `/tmp` is sufficient and available on this machine.

### Suggested Fix
Point the dedicated smoke config's default output directory to `/tmp` (via an env-overridable path) so checkpoint-save validation does not depend on workspace quota.

### Metadata
- Reproducible: yes
- Related Files: examples/revise/config/revise_videoespresso_grpo_smoke.yaml

---
## [ERR-20260308-009] hydra_default_run_dir_still_hits_workspace_quota

**Logged**: 2026-03-08T03:48:00-06:00
**Priority**: medium
**Status**: fixed
**Area**: config

### Summary
Even after moving `trainer.default_local_dir` to `/tmp`, Hydra still tried to create its own default run directory under the repo workspace (`outputs/YYYY-MM-DD/...`), which hit the same disk quota.

### Error
```text
OSError: [Errno 122] Disk quota exceeded: 'outputs/2026-03-08/03-47-43'
```

### Context
- Reproduced immediately at process start when rerunning the clean-env smoke with `REVISE_SMOKE_OUTPUT_DIR=/tmp/...`.
- Hydra's default `run.dir` remained unset, so it fell back to its standard per-date folder under the current working directory.
- For smoke validation, both trainer checkpoints and Hydra artifacts must point at the same large-volume temp root.

### Suggested Fix
Set `hydra.run.dir` in the dedicated smoke config to an env-overridable path under `${trainer.default_local_dir}`.

### Metadata
- Reproducible: yes
- Related Files: examples/revise/config/revise_videoespresso_grpo_smoke.yaml
