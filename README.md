# REVISE: Sparse Video Understanding

This repository contains the code for **REVISE**, a sparse, multi-round video understanding method:
- **Plug-and-play evaluation**: iteratively select a small number of frames, maintain a **text summary state**, and answer MCQ video QA.
- **RL fine-tuning**: GRPO-style training with an **EAGER-style** reward design for efficient, summary-driven reasoning.

The implementation is built on top of the `verl` RL training library and has been trimmed to keep only the components needed for REVISE.

## What’s inside
- `examples/revise/`: REVISE agent loop runners + dataset adapters + configs (NExT-QA, LVBench, Video-MME, EgoSchema).
- `verl/`: core RL training code + REVISE-specific agent loop + reward implementations.

## Installation

We recommend using a dedicated conda environment.

```bash
conda create -n verlrun python=3.10 -y
conda activate verlrun

pip install -U pip
pip install -e .
```

Install optional dependencies depending on your rollout backend:
- vLLM: use `requirements.txt` + a compatible vLLM build.
- SGLang: see `requirements_sglang.txt`.

## Quickstart: NExT-QA plug-and-play evaluation

Run the REVISE multi-round evaluation loop via the `verl` trainer entrypoint:

```bash
ENGINE=sglang ./examples/revise/run_revise_nextqa_eval.sh
```

To use vLLM:

```bash
ENGINE=vllm ./examples/revise/run_revise_nextqa_eval.sh --config-name revise_nextqa_eval_vllm
```

## RL fine-tuning (GRPO + EAGER-style reward)

```bash
ENGINE=sglang ./examples/revise/run_revise_nextqa_grpo.sh
```

## Datasets

This repo expects datasets to be available on local disk (paths are configured in `examples/revise/config/*`).
Typical datasets used in our experiments:
- NExT-QA (videos + CSVs)
- LVBench (HF dataset + cached videos)
- Video-MME (HF dataset + cached videos)
- EgoSchema

## Notes
- The agent loop uses a **summary state** that is updated across rounds; format robustness matters (invalid outputs will reduce accuracy).
- For best performance, run on 4 GPUs with tensor-parallel vLLM/SGLang configurations matching your cluster.

## License

Apache-2.0 (see `LICENSE`). This repo includes code adapted from the original `verl` project.

