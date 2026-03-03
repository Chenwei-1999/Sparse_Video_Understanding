#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd
import requests

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_CAPTION_CACHE: dict[str, dict[int, str]] = {}


SYSTEM_PROMPT = (
    "You are a multiple-choice QA assistant.\n"
    "You will be given a question with options and video captions sampled at ~1 fps (caption index ≈ seconds).\n"
    "Answer with ONLY the option letter (A/B/C/D/E). Do not output any other text.\n"
)


def _wait_port(host: str, port: int, timeout_s: int) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"Timed out waiting for {host}:{port} to accept connections after {timeout_s}s")


def _get_model_id(base_url: str, timeout: int = 30) -> str:
    resp = requests.get(f"{base_url}/v1/models", timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    models = data.get("data", [])
    if not models:
        raise RuntimeError(f"No models returned from {base_url}/v1/models")
    return models[0]["id"]


def _chat_once(
    *,
    base_url: str,
    model_id: str,
    system_prompt: str,
    user_text: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout_s: int,
) -> str:
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _format_question(question: str, choices: list[str]) -> str:
    labels = [chr(ord("A") + i) for i in range(len(choices))]
    lines = [f"Question: {question}", "Options:"]
    for label, choice in zip(labels, choices, strict=False):
        lines.append(f"{label}. {choice}")
    if labels:
        lines.append(f"Return ONLY the option letter: {', '.join(labels)}.")
    return "\n".join(lines)


def _extract_answer_letter(text: str) -> Optional[str]:
    raw = (text or "").strip()
    if not raw:
        return None
    m = _ANSWER_RE.search(raw)
    if m:
        cand = (m.group(1) or "").strip()
        if cand:
            return cand[:1].upper()
    # Fall back to first non-whitespace character.
    for ch in raw:
        if ch.isspace():
            continue
        if ch.upper() in {"A", "B", "C", "D", "E"}:
            return ch.upper()
        break
    return None


def _stable_sample_id(video_id: str, question: str, choices: list[str], answer_idx: int) -> str:
    payload = {
        "video_id": str(video_id),
        "question": str(question),
        "choices": [str(c) for c in (choices or [])],
        "answer_idx": int(answer_idx),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _normalize_video_id(video_id: Any) -> str:
    if isinstance(video_id, int):
        return str(video_id)
    if isinstance(video_id, float):
        return str(int(video_id))
    return str(video_id)


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars].rstrip() + "…"


def _clip_middle(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    text = text or ""
    if len(text) <= max_chars:
        return text
    if max_chars < 50:
        return text[:max_chars]
    keep = max_chars - 15
    head = keep // 2
    tail = keep - head
    return text[:head].rstrip() + "\n...\n" + text[-tail:].lstrip()


def _load_video_captions(captions_dir: str, video_id: str) -> dict[int, str]:
    cached = _CAPTION_CACHE.get(video_id)
    if cached is not None:
        return cached
    path = os.path.join(captions_dir, f"{video_id}_cap.json")
    if not os.path.exists(path):
        _CAPTION_CACHE[video_id] = {}
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        _CAPTION_CACHE[video_id] = {}
        return {}
    captions: dict[int, str] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                idx = int(k)
            except Exception:
                continue
            if isinstance(v, str):
                captions[idx] = v.strip()
    _CAPTION_CACHE[video_id] = captions
    return captions


def _format_caption_timeline(
    captions: dict[int, str],
    *,
    per_caption_max_chars: int,
    total_max_chars: int,
) -> tuple[str, int]:
    if not captions:
        return "(no captions found)", 0
    lines: list[str] = []
    total_chars = 0
    for idx in sorted(captions):
        cap = _truncate_text(captions.get(idx) or "", per_caption_max_chars)
        line = f"{idx}: {cap}"
        lines.append(line)
        total_chars += len(line) + 1
    text = "\n".join(lines)
    return _clip_middle(text, total_max_chars), total_chars


@dataclass
class NextQASample:
    sample_id: str
    qid: str
    video_id: str
    video_path: str
    question: str
    choices: list[str]
    answer_idx: int


def _load_nextqa_samples(
    csv_path: str,
    map_json: str,
    video_root: str,
    max_samples: int,
    seed: int,
) -> list[NextQASample]:
    with open(map_json, "r", encoding="utf-8") as f:
        video_map = json.load(f)
    video_map = {str(k): v for k, v in video_map.items()}

    df = pd.read_csv(csv_path)
    if max_samples > 0:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    samples: list[NextQASample] = []
    for _, row in df.iterrows():
        video_id = _normalize_video_id(row["video"])
        rel = video_map.get(video_id)
        if rel is None:
            continue
        if not str(rel).endswith(".mp4"):
            rel = f"{rel}.mp4"
        video_path = os.path.join(video_root, rel)
        if not os.path.exists(video_path):
            continue
        choices = [str(row[f"a{i}"]) for i in range(5)]
        answer_idx = int(row.get("answer", 0))
        samples.append(
            NextQASample(
                sample_id=_stable_sample_id(video_id, str(row.get("question", "")), choices, answer_idx),
                qid=str(row.get("qid", "")),
                video_id=video_id,
                video_path=video_path,
                question=str(row.get("question", "")),
                choices=choices,
                answer_idx=answer_idx,
            )
        )
        if max_samples > 0 and len(samples) >= max_samples:
            break
    return samples


def _start_vllm_server(args: argparse.Namespace) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env.pop("ROCR_VISIBLE_DEVICES", None)
    env.pop("HIP_VISIBLE_DEVICES", None)
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")

    py_bin = os.path.dirname(sys.executable)
    if py_bin:
        env["PATH"] = py_bin + os.pathsep + env.get("PATH", "")
    vllm_bin = shutil.which("vllm", path=env.get("PATH"))

    cmd = [
        vllm_bin or "vllm",
        "serve",
        args.model_path,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        args.dtype,
        "--load-format",
        "auto",
        "--max-model-len",
        str(args.max_model_len),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    server_stdout = subprocess.DEVNULL
    server_stderr = subprocess.DEVNULL
    if args.server_log:
        log_dir = os.path.dirname(args.server_log)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        log_f = open(args.server_log, "a", encoding="utf-8")
        server_stdout = log_f
        server_stderr = log_f
    return subprocess.Popen(cmd, env=env, stdout=server_stdout, stderr=server_stderr)


def _stop_server(proc: subprocess.Popen[str]) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def _maybe_log_jsonl(path: str, record: dict[str, Any]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="HF model id or local snapshot path")
    ap.add_argument("--captions-dir", required=True, help="Directory containing <video_id>_cap.json caption files.")
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--map-json", required=True)
    ap.add_argument("--csv", required=True, help="NExT-QA CSV (e.g., val.csv)")
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--per-caption-max-chars", type=int, default=220)
    ap.add_argument("--caption-max-total-chars", type=int, default=40000)

    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=4)
    ap.add_argument("--request-timeout-s", type=int, default=90)
    ap.add_argument("--progress-interval", type=int, default=100)

    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=19000)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--max-model-len", type=int, default=12288)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.6)

    ap.add_argument("--start-server", action="store_true")
    ap.add_argument("--server-log", default=None)
    ap.add_argument("--server-timeout-s", type=int, default=600)

    ap.add_argument("--log-jsonl", default=None)
    ap.add_argument("--summary-json", default=None)

    ap.add_argument("--use-wandb", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--wandb-project", default="revise_nextqa")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default=None)
    ap.add_argument("--wandb-mode", default=None, choices=[None, "online", "offline"])

    args = ap.parse_args()
    if not os.path.isdir(args.captions_dir):
        raise ValueError(f"--captions-dir does not exist or is not a directory: {args.captions_dir}")

    random.seed(args.seed)

    server_proc: Optional[subprocess.Popen[str]] = None
    run = None
    try:
        if args.start_server:
            server_proc = _start_vllm_server(args)
            _wait_port(args.host, args.port, timeout_s=args.server_timeout_s)

        base_url = f"http://{args.host}:{args.port}"
        model_id = _get_model_id(base_url)

        samples = _load_nextqa_samples(
            csv_path=args.csv,
            map_json=args.map_json,
            video_root=args.video_root,
            max_samples=args.max_samples,
            seed=args.seed or 42,
        )
        if not samples:
            raise RuntimeError("No samples loaded (check csv/map/video_root).")

        num_shards = max(1, int(args.num_shards))
        shard_idx = int(args.shard_idx)
        if num_shards > 1:
            if not (0 <= shard_idx < num_shards):
                raise ValueError(f"--shard-idx must be in [0, {num_shards}) (got {shard_idx}).")

            def _suffix_path(path: str, *, suffix: str) -> str:
                root, ext = os.path.splitext(path)
                if ext:
                    return f"{root}{suffix}{ext}"
                return f"{path}{suffix}"

            shard_suffix = f".shard{shard_idx}of{num_shards}"
            if args.log_jsonl and shard_suffix not in args.log_jsonl:
                args.log_jsonl = _suffix_path(args.log_jsonl, suffix=shard_suffix)
            if args.summary_json and shard_suffix not in args.summary_json:
                args.summary_json = _suffix_path(args.summary_json, suffix=shard_suffix)
            samples = [s for i, s in enumerate(samples) if (i % num_shards) == shard_idx]
            if not samples:
                raise RuntimeError(
                    f"No samples selected for shard {shard_idx}/{num_shards} (check --max-samples / sharding)."
                )

        if args.use_wandb and wandb is not None:
            mode = args.wandb_mode or os.getenv("WANDB_MODE") or ("online" if os.getenv("WANDB_API_KEY") else "offline")
            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_name,
                group=args.wandb_group,
                config={
                    "task": "nextqa_caption_only_vllm",
                    "csv": args.csv,
                    "captions_dir": args.captions_dir,
                    "model_path": args.model_path,
                    "num_shards": num_shards,
                    "shard_idx": shard_idx,
                    "max_samples": args.max_samples,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": args.max_tokens,
                    "per_caption_max_chars": args.per_caption_max_chars,
                    "caption_max_total_chars": args.caption_max_total_chars,
                },
            )

        correct = 0
        processed = 0
        invalid_outputs = 0
        failed = 0
        total_model_calls = 0
        total_caption_chars = 0

        start_eval = time.time()
        for s in samples:
            processed += 1
            try:
                captions = _load_video_captions(args.captions_dir, s.video_id)
                caption_block, caption_chars = _format_caption_timeline(
                    captions,
                    per_caption_max_chars=args.per_caption_max_chars,
                    total_max_chars=args.caption_max_total_chars,
                )
                total_caption_chars += caption_chars
                user_text = (
                    f"{_format_question(s.question, s.choices)}\n\n"
                    "Video captions (1fps timeline; index ≈ seconds):\n"
                    f"{caption_block}\n"
                )
                raw = _chat_once(
                    base_url=base_url,
                    model_id=model_id,
                    system_prompt=SYSTEM_PROMPT,
                    user_text=user_text,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    timeout_s=args.request_timeout_s,
                )
                total_model_calls += 1
                pred = _extract_answer_letter(raw)
                gt = chr(ord("A") + int(s.answer_idx))
                is_correct = bool(pred == gt)
                if pred is None:
                    invalid_outputs += 1
                if is_correct:
                    correct += 1

                _maybe_log_jsonl(
                    args.log_jsonl,
                    {
                        "ts": time.time(),
                        "sample_id": s.sample_id,
                        "qid": s.qid,
                        "video_id": s.video_id,
                        "question": s.question,
                        "choices": s.choices,
                        "ground_truth": gt,
                        "caption_chars": caption_chars,
                        "user_text": user_text,
                        "raw_output": raw,
                        "pred": pred,
                        "correct": is_correct,
                    },
                )
            except Exception as e:
                failed += 1
                _maybe_log_jsonl(
                    args.log_jsonl,
                    {
                        "ts": time.time(),
                        "sample_id": s.sample_id,
                        "qid": s.qid,
                        "video_id": s.video_id,
                        "error": repr(e),
                    },
                )
            if args.progress_interval and (processed % int(args.progress_interval) == 0):
                elapsed = time.time() - start_eval
                acc_now = correct / max(1, processed - failed)
                print(
                    f"[{processed}/{len(samples)}] acc={acc_now:.4f} failed={failed} invalid={invalid_outputs} "
                    f"avg_caption_chars={total_caption_chars / max(1, processed):.1f} elapsed_s={elapsed:.1f}",
                    flush=True,
                )

        elapsed_s = time.time() - start_eval
        samples_n = max(1, processed - failed)
        acc = correct / samples_n
        avg_caption_chars = total_caption_chars / samples_n

        if run is not None:
            run.summary["final_acc"] = acc
            run.summary["final_correct"] = correct
            run.summary["final_samples"] = processed
            run.summary["avg_caption_chars"] = avg_caption_chars
            run.summary["invalid_outputs"] = invalid_outputs
            run.summary["failed"] = failed
            run.finish()

        results = {
            "samples": processed,
            "correct": correct,
            "accuracy": acc,
            "total_rounds": processed,
            "avg_rounds": 1.0,
            "total_effective_rounds": processed,
            "avg_effective_rounds": 1.0,
            "failed": failed,
            "elapsed_s": elapsed_s,
            "invalid_outputs": invalid_outputs,
            "invalid_action_terminated": 0,
            "total_retries": 0,
            "total_model_calls": total_model_calls,
            "fallback_frames_used": 0,
            "avg_caption_chars": avg_caption_chars,
        }

        prompt_log_lines = 0
        prompt_log_bytes = 0
        if args.log_jsonl and os.path.exists(args.log_jsonl):
            with open(args.log_jsonl, "r", encoding="utf-8") as _f:
                prompt_log_lines = sum(1 for _ in _f)
            prompt_log_bytes = os.path.getsize(args.log_jsonl)

        if args.summary_json:
            os.makedirs(os.path.dirname(args.summary_json) or ".", exist_ok=True)
            with open(args.summary_json, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "task": "nextqa_caption_only_vllm",
                        "dataset_csv": args.csv,
                        "video_root": args.video_root,
                        "map_json": args.map_json,
                        "captions_dir": args.captions_dir,
                        "model_path": args.model_path,
                        "engine": "vllm",
                        "tensor_parallel_size": args.tensor_parallel_size,
                        "dtype": args.dtype,
                        "max_model_len": args.max_model_len,
                        "gpu_memory_utilization": args.gpu_memory_utilization,
                        "max_samples": args.max_samples,
                        "num_shards": num_shards,
                        "shard_idx": shard_idx,
                        "per_caption_max_chars": args.per_caption_max_chars,
                        "caption_max_total_chars": args.caption_max_total_chars,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "max_tokens": args.max_tokens,
                        "results": results,
                        "prompt_log_jsonl": args.log_jsonl,
                        "prompt_log_lines": prompt_log_lines,
                        "prompt_log_bytes": prompt_log_bytes,
                        "wandb": {
                            "enabled": bool(args.use_wandb),
                            "mode": args.wandb_mode
                            or os.getenv("WANDB_MODE")
                            or ("online" if os.getenv("WANDB_API_KEY") else "offline"),
                            "project": args.wandb_project,
                            "entity": args.wandb_entity,
                            "name": args.wandb_name,
                            "group": args.wandb_group,
                            "id": getattr(run, "id", None),
                            "run_dir": getattr(run, "dir", None),
                            "url": getattr(run, "url", None),
                        },
                        "command": "python " + " ".join(os.sys.argv),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        return 0
    finally:
        if server_proc is not None:
            _stop_server(server_proc)


if __name__ == "__main__":
    raise SystemExit(main())
