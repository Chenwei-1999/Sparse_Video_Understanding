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
    _parse_frame_indices,
    _sample_uniform_indices,
)


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


def _maybe_log_jsonl(path: Optional[str], obj: dict[str, Any]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


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
    args = parser.parse_args()

    random.seed(args.seed)

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

        correct = 0
        total_rounds = 0
        failed = 0
        start_eval = time.time()
        for idx, sample in enumerate(samples, start=1):
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
                        answer_letter = answer.strip().upper()[:1]
                        break

                    frames_text = _extract_tag(raw, _FRAMES_RE)
                    summary = _extract_tag(raw, _SUMMARY_RE)
                    if summary:
                        summary_state = summary
                    if frames_text:
                        requested = _parse_frame_indices(frames_text)
                        requested = [i for i in requested if 0 <= i < frame_count and i not in seen_frames]
                        next_frames = requested[: args.max_frames_per_round]
                    else:
                        next_frames = _sample_uniform_indices(frame_count, 1)

                total_rounds += round_idx

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

            if args.progress_interval > 0 and idx % args.progress_interval == 0:
                elapsed = time.time() - start_eval
                print(
                    f"[{idx}/{len(samples)}] acc={correct/idx:.4f} avg_rounds={total_rounds/idx:.3f} "
                    f"failed={failed} elapsed_s={elapsed:.1f}",
                    flush=True,
                )

        processed = len(samples)
        acc = correct / max(1, processed)
        avg_rounds = total_rounds / max(1, processed)
        elapsed = time.time() - start_eval
        results = {
            "samples": processed,
            "accuracy": acc,
            "avg_rounds": avg_rounds,
            "failed": failed,
            "elapsed_s": elapsed,
        }
        print(json.dumps(results, indent=2))

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
