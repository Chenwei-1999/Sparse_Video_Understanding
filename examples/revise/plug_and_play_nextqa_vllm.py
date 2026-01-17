#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import re
import socket
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd
import requests
from PIL import Image

from verl.experimental.agent_loop.revise_agent_loop import (
    DEFAULT_SYSTEM_PROMPT,
    _extract_tag,
    _normalize_answer_letter,
    _parse_frame_indices,
    _sample_uniform_indices,
)

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None


_SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL | re.IGNORECASE)
_FRAMES_RE = re.compile(r"<frames>(.*?)</frames>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def _b64_jpeg(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _wait_port(host: str, port: int, timeout_s: int = 300) -> None:
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


def _format_question(question: str, choices: list[str]) -> str:
    labels = [chr(ord("A") + i) for i in range(len(choices))]
    lines = [f"Question: {question}", "Options:"]
    for label, choice in zip(labels, choices, strict=False):
        lines.append(f"{label}. {choice}")
    if labels:
        lines.append(f"Answer with exactly one letter only: {', '.join(labels)}.")
    return "\n".join(lines)


def _build_user_text(
    question_block: str,
    summary: str,
    frame_count: int,
    round_idx: int,
    frame_indices: list[int],
    seen_frames: list[int],
) -> str:
    lines = [
        f"Round {round_idx} / Question:\n{question_block}",
        f"Total frames L = {frame_count}.",
        f"Seen frames: {seen_frames}",
        "Current summary:",
        f"<summary>{summary}</summary>",
        "Frames for this round:",
    ]
    for idx in frame_indices:
        lines.append(f"Frame {idx} <image>")
    return "\n".join(lines)


def _extract_frames(video_path: str, frame_indices: list[int]) -> list[Image.Image]:
    if not frame_indices:
        return []
    # Try decord first (fast).
    try:
        import decord

        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        frames = vr.get_batch(frame_indices).asnumpy()
        return [Image.fromarray(frame) for frame in frames]
    except Exception:
        pass
    # Fall back to imageio.
    import imageio

    reader = imageio.get_reader(video_path, "ffmpeg")
    try:
        return [Image.fromarray(reader.get_data(idx)) for idx in frame_indices]
    finally:
        reader.close()


@dataclass
class NextQASample:
    qid: str
    video_id: str
    video_path: str
    question: str
    choices: list[str]
    answer_idx: int
    frame_count: int


def _normalize_video_id(video_id: Any) -> str:
    if isinstance(video_id, int):
        return str(video_id)
    if isinstance(video_id, float):
        return str(int(video_id))
    return str(video_id)


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
        # Deterministic shuffle, but keep picking until we have enough valid samples.
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
        samples.append(
            NextQASample(
                qid=str(row.get("qid", "")),
                video_id=video_id,
                video_path=video_path,
                question=str(row.get("question", "")),
                choices=choices,
                answer_idx=int(row.get("answer", 0)),
                frame_count=int(row.get("frame_count", 0)),
            )
        )
        if max_samples > 0 and len(samples) >= max_samples:
            break
    return samples


def _sample_unseen_frames(frame_count: int, seen: set[int], k: int, rng: random.Random) -> list[int]:
    if frame_count <= 0 or k <= 0:
        return []
    if len(seen) >= frame_count:
        return []
    candidates = [i for i in range(frame_count) if i not in seen]
    if not candidates:
        return []
    if len(candidates) <= k:
        return candidates
    return sorted(rng.sample(candidates, k=k))


def _maybe_log_jsonl(path: Optional[str], obj: dict[str, Any]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _maybe_init_wandb(args: argparse.Namespace, run_config: dict[str, Any]) -> Any:
    if not getattr(args, "use_wandb", False) or wandb is None:
        return None

    mode = getattr(args, "wandb_mode", "") or os.getenv("WANDB_MODE")
    if not mode:
        mode = "online" if os.getenv("WANDB_API_KEY") else "offline"

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        group=args.wandb_group,
        tags=args.wandb_tags.split(",") if args.wandb_tags else None,
        mode=mode,
        config=run_config,
        reinit=True,
    )


def _wandb_log(run: Any, metrics: dict[str, Any], step: int) -> None:
    if run is None or wandb is None:
        return
    wandb.log(metrics, step=step)


def _load_progress_from_log(
    log_path: str,
    max_rounds: int,
    default_num_choices: int = 5,
) -> tuple[int, int, int]:
    if not log_path or not os.path.exists(log_path):
        return 0, 0, 0

    completed = 0
    correct = 0
    total_rounds = 0

    in_sample = False
    current_gt_idx = -1
    current_num_choices = default_num_choices
    current_answer: Optional[str] = None
    current_answer_round: Optional[int] = None

    def _finalize_sample() -> None:
        nonlocal completed, correct, total_rounds
        nonlocal in_sample, current_gt_idx, current_num_choices, current_answer, current_answer_round
        if not in_sample:
            return
        if current_answer is None or current_answer_round is None:
            return
        completed += 1
        pred_idx = ord(current_answer) - ord("A")
        if current_gt_idx >= 0 and pred_idx == current_gt_idx:
            correct += 1
        total_rounds += min(current_answer_round, max_rounds)

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            forced = bool(obj.get("forced_answer", False))
            raw = str(obj.get("raw_output", ""))

            try:
                round_idx = int(obj.get("round_idx", 0))
            except Exception:
                round_idx = 0

            # New sample boundary: every sample starts from round_idx=1 (forced answer is logged as max_rounds+1).
            if not in_sample or (round_idx == 1 and not forced):
                _finalize_sample()
                in_sample = True
                try:
                    current_gt_idx = int(obj.get("ground_truth_idx", -1))
                except Exception:
                    current_gt_idx = -1
                choices = obj.get("choices", [])
                current_num_choices = len(choices) if isinstance(choices, list) and choices else default_num_choices
                current_answer = None
                current_answer_round = None

            answer = _extract_tag(raw, _ANSWER_RE)
            if answer and current_answer is None:
                answer_letter = _normalize_answer_letter(answer, current_num_choices)
                if answer_letter is not None:
                    current_answer = answer_letter
                    # For forced answers, we log an extra request after max_rounds, but the "round budget" is max_rounds.
                    current_answer_round = min(round_idx, max_rounds)

    _finalize_sample()
    return completed, correct, total_rounds


def _count_file_lines(path: str) -> int:
    if not path or not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _chat_once(
    base_url: str,
    model_id: str,
    system_prompt: str,
    user_text: str,
    images: list[Image.Image],
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout_s: int,
) -> str:
    # Prefer interleaving text + images to match the "<image>" placeholders in text.
    # This is more robust across OpenAI-compatible multimodal frontends.
    content: list[dict[str, Any]] = []
    parts = user_text.split("<image>") if images else [user_text]
    if images and (len(parts) - 1) == len(images):
        for i, img in enumerate(images):
            if parts[i]:
                content.append({"type": "text", "text": parts[i]})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{_b64_jpeg(img)}"},
                }
            )
        if parts[-1]:
            content.append({"type": "text", "text": parts[-1]})
    else:
        for img in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{_b64_jpeg(img)}"},
                }
            )
        content.append({"type": "text", "text": user_text})

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _start_vllm_server(args: argparse.Namespace) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env.pop("ROCR_VISIBLE_DEVICES", None)
    env.pop("HIP_VISIBLE_DEVICES", None)
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    cmd = [
        "vllm",
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
        "--limit-mm-per-prompt",
        json.dumps({"image": int(args.max_frames_per_round)}),
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="HF model id or local snapshot path")
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--map-json", required=True)
    parser.add_argument("--csv", required=True, help="NExT-QA CSV (e.g., val.csv)")
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument(
        "--max-frames-per-round",
        "--max-frames",
        type=int,
        default=5,
        dest="max_frames_per_round",
        help="Max frames to show/request per round.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--start-server", action="store_true", help="Start vLLM server subprocess")
    parser.add_argument(
        "--restart-server-on-failure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When using --start-server, restart vLLM if a request fails (e.g., timeout).",
    )
    parser.add_argument(
        "--server-log",
        default=None,
        help="Optional file path to append vLLM server logs (stdout+stderr).",
    )
    parser.add_argument("--server-timeout-s", type=int, default=600)
    parser.add_argument("--request-timeout-s", type=int, default=600)
    parser.add_argument("--log-jsonl", default=os.getenv("REVISE_LOG_PATH", "debug_prompt_logs/revise_samples.jsonl"))
    parser.add_argument("--summary-json", default=None, help="Optional path to save a run summary JSON.")
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument(
        "--resume-from-log",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If log-jsonl exists, skip already-completed samples and continue appending to the same log.",
    )
    parser.add_argument(
        "--force-final-answer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If no <answer> is produced within max rounds, issue a final answer-only request (matches ReviseAgentLoop).",
    )
    parser.add_argument("--use-wandb", action="store_true", help="Log eval metrics to Weights & Biases.")
    parser.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "verl-revise"))
    parser.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY"))
    parser.add_argument("--wandb-name", default=os.getenv("WANDB_RUN_NAME"))
    parser.add_argument("--wandb-group", default=os.getenv("WANDB_RUN_GROUP"))
    parser.add_argument("--wandb-tags", default=os.getenv("WANDB_TAGS", ""))
    parser.add_argument("--wandb-mode", default=os.getenv("WANDB_MODE", ""))
    args = parser.parse_args()

    random.seed(args.seed)
    rng = random.Random(args.seed)

    server_proc: Optional[subprocess.Popen[str]] = None
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

        resume_completed = 0
        correct = 0
        total_rounds = 0
        if args.resume_from_log and args.log_jsonl and os.path.exists(args.log_jsonl):
            resume_completed, correct, total_rounds = _load_progress_from_log(
                args.log_jsonl, max_rounds=args.max_rounds
            )
            resume_completed = min(resume_completed, len(samples))

        processed = resume_completed
        failed = 0
        run = _maybe_init_wandb(
            args,
            run_config={
                "task": "revise_plug_and_play_nextqa_vllm",
                "dataset_csv": args.csv,
                "video_root": args.video_root,
                "map_json": args.map_json,
                "model_path": args.model_path,
                "engine": "vllm",
                "tensor_parallel_size": args.tensor_parallel_size,
                "dtype": args.dtype,
                "max_model_len": args.max_model_len,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "max_samples": args.max_samples,
                "max_rounds": args.max_rounds,
                "max_frames_per_round": args.max_frames_per_round,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
                "force_final_answer": args.force_final_answer,
                "resume_from_log": args.resume_from_log,
                "initial_completed": resume_completed,
                "log_jsonl": args.log_jsonl,
            },
        )
        start_eval = time.time()
        for sample in samples[resume_completed:]:
            processed += 1
            try:
                frame_count = sample.frame_count
                if frame_count <= 0:
                    try:
                        import decord

                        vr = decord.VideoReader(sample.video_path, ctx=decord.cpu(0))
                        frame_count = int(len(vr))
                    except Exception:
                        frame_count = 0

                question_block = _format_question(sample.question, sample.choices)
                system_prompt = DEFAULT_SYSTEM_PROMPT.format(max_frames_per_round=args.max_frames_per_round)

                summary_state = "O: none; H: none; R: need evidence; P: []; U: unknown"
                seen_frames: list[int] = []

                init_frames = _sample_uniform_indices(frame_count, args.max_frames_per_round)
                next_frames = [int(i) for i in init_frames if i >= 0]
                answer_letter: Optional[str] = None
                last_user_text: Optional[str] = None
                last_images: list[Image.Image] = []
                last_frames: list[int] = []

                for round_idx in range(1, args.max_rounds + 1):
                    # Frames shown in this round.
                    frames_this_round = [i for i in next_frames if i not in seen_frames]
                    if not frames_this_round:
                        frames_this_round = _sample_uniform_indices(frame_count, 1)
                    frames_this_round = frames_this_round[: args.max_frames_per_round]
                    for i in frames_this_round:
                        if i not in seen_frames:
                            seen_frames.append(i)

                    images = _extract_frames(sample.video_path, frames_this_round)
                    user_text = _build_user_text(
                        question_block=question_block,
                        summary=summary_state,
                        frame_count=frame_count,
                        round_idx=round_idx,
                        frame_indices=frames_this_round,
                        seen_frames=seen_frames,
                    )
                    if args.force_final_answer and round_idx >= args.max_rounds:
                        user_text = (
                            f"{user_text}\n\n"
                            "This is the final round. You MUST answer now using <answer>LETTER</answer>."
                        )
                    last_user_text = user_text
                    last_images = images
                    last_frames = frames_this_round

                    raw = _chat_once(
                        base_url=base_url,
                        model_id=model_id,
                        system_prompt=system_prompt,
                        user_text=user_text,
                        images=images,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens,
                        timeout_s=args.request_timeout_s,
                    )

                    _maybe_log_jsonl(
                        args.log_jsonl,
                        {
                            "ts": time.time(),
                            "qid": sample.qid,
                            "video_id": sample.video_id,
                            "video_path": sample.video_path,
                            "round_idx": round_idx,
                            "question": sample.question,
                            "choices": sample.choices,
                            "ground_truth_idx": sample.answer_idx,
                            "seen_frames": seen_frames,
                            "current_frames": frames_this_round,
                            "summary_in": summary_state,
                            "system_prompt": system_prompt,
                            "user_text": user_text,
                            "raw_output": raw,
                        },
                    )

                    answer = _extract_tag(raw, _ANSWER_RE)
                    if answer:
                        answer_letter = _normalize_answer_letter(answer, len(sample.choices))
                        break

                    frames_text = _extract_tag(raw, _FRAMES_RE)
                    summary = _extract_tag(raw, _SUMMARY_RE)
                    if summary:
                        summary_state = summary
                    if frames_text:
                        requested = _parse_frame_indices(frames_text)
                        requested = [i for i in requested if 0 <= i < frame_count and i not in seen_frames]
                        if not requested:
                            requested = _sample_unseen_frames(
                                frame_count, set(seen_frames), args.max_frames_per_round, rng=rng
                            )
                        next_frames = requested[: args.max_frames_per_round]
                    else:
                        next_frames = _sample_uniform_indices(frame_count, 1)

                total_rounds += round_idx
                if args.force_final_answer and answer_letter is None and last_user_text is not None:
                    forced_user_text = (
                        f"{last_user_text}\n\n"
                        "Max rounds reached. Provide the final answer now using <answer>LETTER</answer> only."
                    )
                    raw = _chat_once(
                        base_url=base_url,
                        model_id=model_id,
                        system_prompt=system_prompt,
                        user_text=forced_user_text,
                        images=last_images,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens,
                        timeout_s=args.request_timeout_s,
                    )
                    _maybe_log_jsonl(
                        args.log_jsonl,
                        {
                            "ts": time.time(),
                            "qid": sample.qid,
                            "video_id": sample.video_id,
                            "video_path": sample.video_path,
                            "round_idx": args.max_rounds + 1,
                            "forced_answer": True,
                            "question": sample.question,
                            "choices": sample.choices,
                            "ground_truth_idx": sample.answer_idx,
                            "seen_frames": seen_frames,
                            "current_frames": last_frames,
                            "summary_in": summary_state,
                            "system_prompt": system_prompt,
                            "user_text": forced_user_text,
                            "raw_output": raw,
                        },
                    )
                    answer = _extract_tag(raw, _ANSWER_RE)
                    if answer:
                        answer_letter = _normalize_answer_letter(answer, len(sample.choices))

                if answer_letter is not None:
                    pred_idx = ord(answer_letter) - ord("A")
                    if pred_idx == sample.answer_idx:
                        correct += 1
            except Exception as e:
                failed += 1
                total_rounds += args.max_rounds
                if (
                    args.start_server
                    and args.restart_server_on_failure
                    and server_proc is not None
                    and isinstance(e, requests.exceptions.RequestException)
                ):
                    try:
                        _stop_server(server_proc)
                    except Exception:
                        pass
                    server_proc = _start_vllm_server(args)
                    _wait_port(args.host, args.port, timeout_s=args.server_timeout_s)
                    model_id = _get_model_id(base_url)

            if args.progress_interval > 0 and processed % args.progress_interval == 0:
                elapsed = time.time() - start_eval
                acc = correct / max(1, processed)
                avg_rounds = total_rounds / max(1, processed)
                print(
                    f"[{processed}/{len(samples)}] acc={acc:.4f} avg_rounds={avg_rounds:.3f} "
                    f"failed={failed} elapsed_s={elapsed:.1f}",
                    flush=True,
                )
                _wandb_log(
                    run,
                    {
                        "eval/acc": acc,
                        "eval/avg_rounds": avg_rounds,
                        "eval/failed": failed,
                        "eval/processed": processed,
                        "eval/elapsed_s": elapsed,
                    },
                    step=processed,
                )

        acc = correct / max(1, processed)
        avg_rounds = total_rounds / max(1, processed)
        elapsed = time.time() - start_eval
        prompt_log_lines = _count_file_lines(args.log_jsonl) if args.log_jsonl else 0
        prompt_log_bytes = os.path.getsize(args.log_jsonl) if args.log_jsonl and os.path.exists(args.log_jsonl) else 0
        results = {
            "samples": processed,
            "accuracy": acc,
            "avg_rounds": avg_rounds,
            "failed": failed,
            "elapsed_s": elapsed,
        }
        print(json.dumps(results, indent=2))
        _wandb_log(
            run,
            {
                "eval/final_acc": acc,
                "eval/final_avg_rounds": avg_rounds,
                "eval/final_failed": failed,
                "eval/final_elapsed_s": elapsed,
                "eval/prompt_log_lines": prompt_log_lines,
                "eval/prompt_log_bytes": prompt_log_bytes,
            },
            step=processed,
        )
        if run is not None:
            run.summary["prompt_log_jsonl"] = args.log_jsonl
            run.summary["prompt_log_lines"] = prompt_log_lines
            run.summary["prompt_log_bytes"] = prompt_log_bytes
            run.summary["final_acc"] = acc
            run.summary["final_avg_rounds"] = avg_rounds
            run.summary["final_failed"] = failed
            run.finish()

        if args.summary_json:
            summary_dir = os.path.dirname(args.summary_json)
            if summary_dir:
                os.makedirs(summary_dir, exist_ok=True)
            with open(args.summary_json, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "task": "revise_plug_and_play_nextqa_vllm",
                        "dataset_csv": args.csv,
                        "video_root": args.video_root,
                        "map_json": args.map_json,
                        "model_path": args.model_path,
                        "engine": "vllm",
                        "tensor_parallel_size": args.tensor_parallel_size,
                        "dtype": args.dtype,
                        "max_model_len": args.max_model_len,
                        "gpu_memory_utilization": args.gpu_memory_utilization,
                        "max_samples": args.max_samples,
                        "max_rounds": args.max_rounds,
                        "max_frames_per_round": args.max_frames_per_round,
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
