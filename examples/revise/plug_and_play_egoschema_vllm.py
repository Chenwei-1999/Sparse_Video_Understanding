#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import hashlib
import io
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

import requests
from PIL import Image

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None


_SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL | re.IGNORECASE)
_FRAMES_RE = re.compile(r"<frames>(.*?)</frames>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


DEFAULT_SYSTEM_PROMPT = (
    "You are REVISE, a multi-round video reasoning agent.\n"
    "Each round you will see: (1) a multiple-choice question with options, (2) the current belief summary, "
    "and (3) a few sampled video frames.\n"
    "If you are confident, answer the question.\n"
    "If you are NOT confident, request MORE video frames to view NEXT.\n"
    "Frames are sampled at ~1 fps; frame index ≈ timestamp in seconds.\n\n"
    "IMPORTANT: Output must follow EXACTLY ONE of the two formats below. Do not output any text outside tags.\n"
    "Output ONLY <summary> plus either <frames> (request) OR <answer> (final).\n"
    "Do NOT output <think>.\n"
    "Do NOT output bare placeholders like '...', 'none', or 'N/A' as your summary fields.\n"
    "It's OK to say something is unclear/unknown *within a sentence*, but do not leave fields empty.\n\n"
    "Format 1 — Request more frames (use this only if NOT confident):\n"
    "<summary>P: previously seen frames show a person in the scene interacting with objects; "
    "O: I observe an action that may be relevant to the question; "
    "H: based on the evidence so far, my belief is updated but still incomplete; "
    "U: a key detail needed to answer is still unclear; "
    "R: request additional frames to gather the missing evidence</summary>\n"
    "<frames>1, 3</frames>\n\n"
    "If Candidate Frame IDs are provided in the user prompt, request using those IDs (e.g., <frames>1, 3</frames>).\n\n"
    "Format 2 — Answer now (use this if confident):\n"
    "<summary>P: previously seen frames already contain the key evidence; "
    "O: the answer-relevant evidence is visible in the shown frames; "
    "H: my belief is updated based on the observed evidence; "
    "U: no remaining ambiguity that affects the answer; "
    "R: answered</summary>\n"
    "<answer>B</answer>\n\n"
    "Tag meanings:\n"
    "- <summary>: the ONLY persistent memory across rounds. Keep it short and update it EVERY round.\n"
    "  The summary MUST be written in this exact order: P → O → H → U → R.\n"
    "  - P (Previously seen): which frames have already been used/seen (write as a sentence; no Python lists).\n"
    "  - O (Observations): what you currently observe in the selected frames.\n"
    "  - H (Belief updates): updated belief based on what has been observed so far (do NOT include the final answer letter).\n"
    "  - U (Uncertainties): what is still unknown or ambiguous.\n"
    "  - R (Reasons): why you need more frames and what evidence you are looking for next (or 'answered').\n\n"
    "Rules:\n"
    "- Frame indices are 0-based in [0, L-1].\n"
    "- If you are confident, answer instead of requesting more frames.\n"
    "- If requesting, choose 1 to {max_frames_per_round} NEW frames to view NEXT.\n"
    "- Do NOT output any frame index from the Seen frames list; those are already viewed.\n"
    "- When provided, request frames ONLY within the allowed unseen frame ranges.\n"
    "- When Candidate Frame IDs are provided, output those IDs (1..K) in <frames> instead of raw frame indices.\n"
    "- In <frames>, output comma-separated integers only (no brackets, no text).\n"
    "- In <summary>, include P/O/H/U/R as short natural-language sentences that reflect your current understanding.\n"
    "- The order inside <summary> MUST be: P then O then H then U then R.\n"
    "- In P, describe previously seen frames in a sentence (describe content; do NOT list frame indices or use Python lists like [4, 8, 12]).\n"
    "- In <answer>, output EXACTLY ONE option letter shown in the question (e.g., A/B/C/D/E). No words/punctuation.\n"
    "- Never copy the example text; replace it with information from the current video.\n"
)


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _contains_banned_example(text: str) -> bool:
    t = _collapse_ws(text).lower()
    if not t:
        return False
    if "george approaching a shelf" in t:
        return True
    if "george pauses" in t and "shelf" in t:
        return True
    return False


def _extract_tag(text: str, pattern: re.Pattern[str]) -> Optional[str]:
    matches = list(pattern.finditer(text or ""))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def _parse_frame_indices(text: str) -> list[int]:
    return [int(n) for n in re.findall(r"\d+", text or "")]


def _dedupe_preserve_order(indices: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for idx in indices or []:
        if idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
    return out


def _normalize_answer_letter(answer_text: str, num_choices: int) -> Optional[str]:
    allowed = {chr(ord("A") + i) for i in range(max(0, num_choices))}
    if not allowed:
        allowed = {"A", "B", "C", "D", "E"}
    candidate = str(answer_text).strip().upper()
    if candidate in allowed:
        return candidate
    m = re.search(r"\b([A-E])\b", candidate)
    if m:
        letter = m.group(1).upper()
        return letter if letter in allowed else None
    m = re.search(r"([A-E])", candidate)
    if m:
        letter = m.group(1).upper()
        return letter if letter in allowed else None
    return None


def _summary_has_ohrpu(summary: str) -> bool:
    if summary is None:
        return False
    s = _collapse_ws(summary)
    order = ["P", "O", "H", "U", "R"]
    positions = []
    for key in order:
        m = re.search(rf"\b{key}\s*:\s*", s, re.IGNORECASE)
        if m is None:
            return False
        positions.append(m.start())
    return all(a < b for a, b in zip(positions, positions[1:], strict=False))


def _is_placeholder(text: str) -> bool:
    t = _collapse_ws(text).lower()
    if not t:
        return True
    if "..." in t or "…" in t:
        return True
    if t in {"...", "…", "none", "n/a", "na", "null", "unknown", "unsure", "uncertain"}:
        return True
    if re.fullmatch(r"[.·•…]+", t):
        return True
    alnum = re.findall(r"[a-z0-9]+", t)
    if len(alnum) <= 1 and len(t) <= 6:
        return True
    return False


def _sample_uniform_indices(frame_count: int, n: int) -> list[int]:
    if n <= 0:
        return []
    if frame_count <= 0:
        return [0]
    if n == 1:
        return [frame_count // 2]
    if frame_count == 1:
        return [0]
    return [round(i * (frame_count - 1) / (n - 1)) for i in range(n)]


def _unseen_intervals(frame_count: int, seen_frames: list[int]) -> list[tuple[int, int]]:
    if frame_count <= 0:
        return [(0, 0)]
    seen = sorted({int(i) for i in (seen_frames or []) if 0 <= int(i) < frame_count})
    anchors = [-1, *seen, frame_count]
    intervals: list[tuple[int, int]] = []
    for a, b in zip(anchors, anchors[1:], strict=False):
        start = a + 1
        end = b - 1
        if start <= end:
            intervals.append((start, end))
    return intervals if intervals else [(0, frame_count - 1)]


def _format_intervals(intervals: list[tuple[int, int]]) -> str:
    out: list[str] = []
    for a, b in intervals:
        out.append(str(a) if a == b else f"{a}-{b}")
    return ", ".join(out) if out else ""


def _format_frame_list(frames: list[int]) -> str:
    f = [int(x) for x in (frames or [])]
    if not f:
        return "(none)"
    if len(f) > 25:
        return ", ".join([str(x) for x in f[:25]]) + f", ... (+{len(f) - 25} more)"
    return ", ".join(str(x) for x in f)


def _indices_to_intervals(indices: list[int]) -> list[tuple[int, int]]:
    ids = sorted({int(i) for i in (indices or [])})
    if not ids:
        return []
    out: list[tuple[int, int]] = []
    start = prev = ids[0]
    for x in ids[1:]:
        if x == prev + 1:
            prev = x
            continue
        out.append((start, prev))
        start = prev = x
    out.append((start, prev))
    return out


def _in_intervals(x: int, intervals: list[tuple[int, int]]) -> bool:
    for a, b in intervals:
        if a <= x <= b:
            return True
    return False


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
    raise TimeoutError(f"Timed out waiting for {host}:{port} after {timeout_s}s")


def _get_model_id(base_url: str, timeout: int = 30) -> str:
    resp = requests.get(f"{base_url}/v1/models", timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    models = data.get("data", [])
    if not models:
        raise RuntimeError(f"No models returned from {base_url}/v1/models")
    return models[0]["id"]


def _normalize_option_text(opt: str) -> str:
    # EgoSchema options may already start with "(A):". Strip that to avoid double labels.
    t = str(opt).strip()
    t = re.sub(r"^\(?[A-E]\)?\s*[:.)-]\s*", "", t)
    return t.strip()


def _format_question(question: str, choices: list[str]) -> str:
    labels = [chr(ord("A") + i) for i in range(len(choices))]
    lines = [f"Question: {question}", "Options:"]
    for label, choice in zip(labels, choices, strict=False):
        lines.append(f"{label}. {choice}")
    if labels:
        lines.append(
            "To answer, output <summary>...</summary> then <answer>LETTER</answer> "
            f"(LETTER must be one of: {', '.join(labels)})."
        )
    return "\n".join(lines)


def _build_user_text(
    question_block: str,
    summary: str,
    frame_count: int,
    round_idx: int,
    frame_indices: list[int],
    seen_frames: list[int],
    *,
    hide_seen_frames: bool = False,
    candidate_unseen_frames: Optional[list[int]] = None,
    use_candidate_frame_ids: bool = False,
    require_candidate_frames: bool = False,
) -> str:
    lines: list[str] = [f"Round {round_idx} / Question:\n{question_block}", f"Total frames L = {frame_count}."]
    if hide_seen_frames:
        lines.append(
            f"Seen frames: {len(seen_frames)} frames already viewed "
            "(do NOT request any previously shown frames; follow the selection constraints below)."
        )
    else:
        lines.append(f"Seen frames (already viewed; do NOT request these again): {_format_frame_list(seen_frames)}")

    if candidate_unseen_frames and use_candidate_frame_ids:
        lines.append(
            "Candidate unseen frames available as IDs (all NEW): "
            f"choose IDs in [1, {len(candidate_unseen_frames)}]."
        )
        lines.append(
            "In <frames>, output ONLY candidate IDs (comma-separated). Do NOT output raw frame indices when IDs exist."
        )
    else:
        lines.append(
            "Allowed unseen frame ranges for <frames> (choose NEW indices only from these ranges): "
            f"{_format_intervals(_unseen_intervals(frame_count, seen_frames))}"
        )
        if candidate_unseen_frames:
            prefix = (
                "Candidate unseen frame ranges to request (REQUIRED, all NEW): "
                if require_candidate_frames
                else "Candidate unseen frame ranges to request (optional, all NEW): "
            )
            lines.append(prefix + f"{_format_intervals(_indices_to_intervals(candidate_unseen_frames))}")
            if require_candidate_frames:
                lines.append("In <frames>, output ONLY indices within the Candidate unseen frame ranges above.")
    lines.extend(["Current summary:", f"<summary>{summary}</summary>", "Frames shown in this round:"])
    if hide_seen_frames or (candidate_unseen_frames and use_candidate_frame_ids):
        for i, _ in enumerate(frame_indices):
            label = chr(ord("A") + i)
            lines.append(f"Shown frame {label} <image>")
    else:
        for idx in frame_indices:
            lines.append(f"Frame {idx} <image>")
    return "\n".join(lines)


def _extract_frames(video_path: str, frame_indices: list[int]) -> list[Image.Image]:
    if not frame_indices:
        return []
    try:
        import decord

        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        frames = vr.get_batch(frame_indices).asnumpy()
        return [Image.fromarray(frame) for frame in frames]
    except Exception:
        pass
    import imageio

    reader = imageio.get_reader(video_path, "ffmpeg")
    try:
        return [Image.fromarray(reader.get_data(idx)) for idx in frame_indices]
    finally:
        reader.close()


def _sample_unseen_frames(frame_count: int, seen: set[int], k: int, rng: random.Random) -> list[int]:
    if frame_count <= 0 or k <= 0:
        return []
    if len(seen) >= frame_count:
        return []
    candidates = [i for i in range(frame_count) if i not in seen]
    rng.shuffle(candidates)
    return sorted(candidates[:k])


def _propose_candidate_unseen_frames(frame_count: int, seen: set[int], k: int, rng: random.Random) -> list[int]:
    if frame_count <= 0 or k <= 0:
        return []
    if len(seen) >= frame_count:
        return []
    seen_sorted = sorted(i for i in seen if 0 <= i < frame_count)
    anchors = sorted(set([0, frame_count - 1, *seen_sorted]))
    # Prefer sampling near "gaps" between seen frames to encourage covering new evidence.
    candidates: list[int] = []
    for a, b in zip(anchors, anchors[1:], strict=False):
        gap = max(0, b - a - 1)
        if gap <= 0:
            continue
        mid = a + 1 + gap // 2
        candidates.append(mid)
    candidates = [c for c in candidates if c not in seen and 0 <= c < frame_count]
    # Fill remaining with random unseen.
    if len(candidates) < k:
        remaining = [i for i in range(frame_count) if i not in seen and i not in candidates]
        rng.shuffle(remaining)
        candidates.extend(remaining[: max(0, k - len(candidates))])
    return sorted(candidates[:k])


def _call_chat_completions(
    base_url: str,
    model_id: str,
    messages: list[dict[str, Any]],
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout_s: int,
) -> str:
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
    }
    resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    return str(data["choices"][0]["message"]["content"])


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
        "--limit-mm-per-prompt",
        json.dumps({"image": int(args.max_frames_per_round)}),
    ]

    server_stdout: Any = subprocess.DEVNULL
    server_stderr: Any = subprocess.DEVNULL
    if args.server_log:
        os.makedirs(os.path.dirname(args.server_log) or ".", exist_ok=True)
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


def _wandb_init(args: argparse.Namespace, summary: dict[str, Any]) -> Any:
    if not args.use_wandb or wandb is None:
        return None

    def _has_wandb_credentials() -> bool:
        if os.getenv("WANDB_API_KEY"):
            return True
        if os.getenv("WANDB_IDENTITY_TOKEN_FILE"):
            return True
        try:
            from wandb.sdk.lib.wbauth import wbnetrc  # type: ignore

            base_url = os.getenv("WANDB_BASE_URL") or "https://api.wandb.ai"
            return bool(wbnetrc.read_netrc_auth(host=base_url))
        except Exception:
            return False

    mode = (args.wandb_mode or os.getenv("WANDB_MODE") or "").strip()
    if not mode:
        mode = "online" if _has_wandb_credentials() else "offline"
    if mode.lower() == "online" and not _has_wandb_credentials():
        mode = "offline"

    return wandb.init(
        project=args.wandb_project or "revise_egoschema",
        entity=args.wandb_entity,
        name=args.wandb_name,
        group=args.wandb_group,
        tags=(args.wandb_tags.split(",") if args.wandb_tags else None),
        mode=mode,
        config=summary,
        reinit=True,
    )


@dataclass
class EgoSchemaSample:
    sample_id: str
    qid: str
    video_path: str
    question: str
    choices: list[str]
    answer_idx: int
    frame_count: int


def _stable_sample_id(video_path: str, question: str, choices: list[str], answer_idx: int) -> str:
    payload = {
        "video_path": str(video_path),
        "question": str(question),
        "choices": [str(c) for c in (choices or [])],
        "answer_idx": int(answer_idx),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _load_egoschema_samples(
    json_path: str,
    video_root: str,
    max_samples: int,
    seed: int,
) -> list[EgoSchemaSample]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected list in {json_path}, got {type(data)}")

    rng = random.Random(seed)
    if max_samples > 0 and len(data) > max_samples:
        rng.shuffle(data)
        data = data[:max_samples]

    samples: list[EgoSchemaSample] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("question_idx") or row.get("qid") or row.get("id") or "")
        question = str(row.get("question") or "")
        options = row.get("options") or row.get("choices") or []
        if not isinstance(options, list) or len(options) < 2:
            continue
        choices = [_normalize_option_text(o) for o in options]
        correct = str(row.get("correct_answer") or row.get("answer") or "").strip()
        m = re.search(r"([A-E])", correct.upper())
        if not m:
            continue
        answer_idx = ord(m.group(1)) - ord("A")
        rel = str(row.get("video_path") or row.get("video") or "")
        if not rel:
            continue
        video_path = os.path.join(video_root, rel)
        if not os.path.exists(video_path):
            # Try alternate: if the json stores only a filename, assume videos/videos/<id>.mp4
            alt = os.path.join(video_root, "videos", "videos", os.path.basename(rel))
            if os.path.exists(alt):
                video_path = alt
            else:
                continue

        sample_id = _stable_sample_id(video_path, question, choices, answer_idx)
        samples.append(
            EgoSchemaSample(
                sample_id=sample_id,
                qid=qid,
                video_path=video_path,
                question=question,
                choices=choices,
                answer_idx=answer_idx,
                frame_count=0,
            )
        )
    return samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="HF model id or local snapshot path")
    parser.add_argument("--video-root", required=True, help="Root directory containing videos referenced by JSON")
    parser.add_argument("--json", required=True, help="EgoSchema JSON list (e.g., pnp_subset_500.json)")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--num-shards", type=int, default=1, help="Shard the dataset for data-parallel evaluation.")
    parser.add_argument("--shard-idx", type=int, default=0, help="Shard index in [0, num_shards).")
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--max-frames-per-round", "--max-frames", type=int, default=5)
    parser.add_argument(
        "--use-candidate-frames",
        action="store_true",
        help="Include a small list of candidate unseen frame indices in the prompt to help frame selection.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=0,
        help="Number of candidate unseen frames to propose when --use-candidate-frames is set (default: max(12, max_frames_per_round*4)).",
    )
    parser.add_argument(
        "--use-candidate-frame-ids",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When --use-candidate-frames is enabled, expose candidates as IDs 1..K (not raw indices) and require <frames> to output candidate IDs.",
    )
    parser.add_argument(
        "--require-candidate-frames",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When candidate frames are provided, treat them as an allowlist: <frames> must only contain indices within the candidate unseen ranges.",
    )
    parser.add_argument(
        "--hide-seen-frames-in-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Do not print explicit seen frame indices in the prompt (reduces copying); rely on unseen ranges instead.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--dtype", type=str, choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=12288)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--start-server", action="store_true", help="Start vLLM server subprocess")
    parser.add_argument(
        "--restart-server-on-failure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When using --start-server, restart vLLM if a request fails (e.g., timeout).",
    )
    parser.add_argument("--server-log", type=str, default=None, help="Optional file path to append vLLM server logs")
    parser.add_argument("--server-timeout-s", type=int, default=300)
    parser.add_argument("--request-timeout-s", type=int, default=300)
    parser.add_argument("--max-retries-per-round", type=int, default=2)
    parser.add_argument(
        "--strict-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Terminate the sample immediately on illegal actions (e.g., invalid <frames>, missing required tags, <think>).",
    )
    parser.add_argument("--log-jsonl", type=str, default=None)
    parser.add_argument("--summary-json", type=str, default=None, help="Optional path to save a run summary JSON.")
    parser.add_argument("--progress-interval", type=int, default=25)
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
        help="If no <answer> is produced within max rounds, issue a final answer-only request.",
    )
    parser.add_argument("--use-wandb", action="store_true", help="Log eval metrics to Weights & Biases.")
    parser.add_argument("--wandb-project", type=str, default="revise_egoschema")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default=None)

    args = parser.parse_args()

    if args.num_shards <= 0:
        raise ValueError("--num-shards must be >= 1")
    if not (0 <= args.shard_idx < args.num_shards):
        raise ValueError("--shard-idx must be in [0, num_shards)")

    rng = random.Random(args.seed)

    samples = _load_egoschema_samples(args.json, args.video_root, args.max_samples, args.seed)
    if args.num_shards > 1:
        samples = [s for i, s in enumerate(samples) if (i % args.num_shards) == args.shard_idx]

    if args.candidate_k <= 0:
        args.candidate_k = max(12, args.max_frames_per_round * 4)

    os.makedirs("debug_runs", exist_ok=True)
    log_jsonl = args.log_jsonl
    if log_jsonl:
        os.makedirs(os.path.dirname(log_jsonl) or ".", exist_ok=True)

    summary: dict[str, Any] = {
        "task": "revise_plug_and_play_egoschema_vllm",
        "dataset_json": args.json,
        "video_root": args.video_root,
        "model_path": args.model_path,
        "engine": "vllm",
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_samples": args.max_samples,
        "num_shards": args.num_shards,
        "shard_idx": args.shard_idx,
        "max_rounds": args.max_rounds,
        "max_frames_per_round": args.max_frames_per_round,
        "max_retries_per_round": args.max_retries_per_round,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "use_candidate_frames": bool(args.use_candidate_frames),
        "candidate_k": int(args.candidate_k),
        "use_candidate_frame_ids": bool(args.use_candidate_frame_ids),
        "require_candidate_frames": bool(args.require_candidate_frames),
        "hide_seen_frames_in_prompt": bool(args.hide_seen_frames_in_prompt),
    }

    wandb_run = _wandb_init(args, summary)

    server_proc: Optional[subprocess.Popen[str]] = None
    base_url = f"http://{args.host}:{args.port}"
    model_id: Optional[str] = None

    def _ensure_server() -> None:
        nonlocal server_proc, model_id
        if args.start_server and server_proc is None:
            server_proc = _start_vllm_server(args)
            _wait_port(args.host, args.port, timeout_s=args.server_timeout_s)
        if model_id is None:
            model_id = _get_model_id(base_url)

    def _restart_server() -> None:
        nonlocal server_proc, model_id
        if server_proc is not None:
            _stop_server(server_proc)
            server_proc = None
            model_id = None
        if args.start_server:
            server_proc = _start_vllm_server(args)
            _wait_port(args.host, args.port, timeout_s=args.server_timeout_s)
            model_id = _get_model_id(base_url)

    # Resume logic
    done_ids: set[str] = set()
    if log_jsonl and args.resume_from_log and os.path.exists(log_jsonl):
        with open(log_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                sid = str(rec.get("sample_id") or "")
                if sid and rec.get("done"):
                    done_ids.add(sid)

    # Eval loop
    correct = 0
    total = 0
    failed = 0
    total_rounds = 0
    invalid_outputs = 0
    total_retries = 0
    total_model_calls = 0
    fallback_frames_used = 0
    start_time = time.time()

    for i, sample in enumerate(samples):
        if sample.sample_id in done_ids:
            continue

        total += 1
        question_block = _format_question(sample.question, sample.choices)
        summary_state = "P: none yet; O: no observations yet; H: no belief yet; U: key details are unknown; R: need frames"
        seen_frames: list[int] = []
        used_images: list[Image.Image] = []
        final_answer: Optional[str] = None
        answer_round: Optional[int] = None
        illegal_action = False
        per_sample_invalid = 0
        per_sample_retries = 0

        # Compute frame_count lazily.
        frame_count = sample.frame_count
        if frame_count <= 0:
            try:
                import decord

                vr = decord.VideoReader(sample.video_path, ctx=decord.cpu(0))
                frame_count = int(len(vr))
            except Exception:
                frame_count = 0

        init_frames = _sample_uniform_indices(frame_count, args.max_frames_per_round)

        for round_idx in range(1, args.max_rounds + 1):
            if round_idx == 1:
                frames_this_round = init_frames
            else:
                if not frames_this_round:
                    frames_this_round = _sample_uniform_indices(frame_count, 1)

            images = _extract_frames(sample.video_path, frames_this_round)
            if not images:
                # fallback: if decoding fails, skip sample
                failed += 1
                illegal_action = True
                break

            used_images.extend(images)
            seen_frames = _dedupe_preserve_order(seen_frames + frames_this_round)

            candidate_unseen = None
            if args.use_candidate_frames:
                candidate_unseen = _propose_candidate_unseen_frames(
                    frame_count, set(seen_frames), args.candidate_k, rng=rng
                )

            user_text = _build_user_text(
                question_block,
                summary_state,
                frame_count,
                round_idx,
                frame_indices=frames_this_round,
                seen_frames=seen_frames,
                hide_seen_frames=args.hide_seen_frames_in_prompt,
                candidate_unseen_frames=candidate_unseen,
                use_candidate_frame_ids=args.use_candidate_frame_ids,
                require_candidate_frames=args.require_candidate_frames,
            )

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT.format(max_frames_per_round=args.max_frames_per_round)},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        *[
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{_b64_jpeg(img)}"},
                            }
                            for img in images
                        ],
                    ],
                },
            ]

            try:
                _ensure_server()
                assert model_id is not None
                raw = _call_chat_completions(
                    base_url,
                    model_id,
                    messages,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    timeout_s=args.request_timeout_s,
                )
                total_model_calls += 1
            except Exception as e:
                if args.restart_server_on_failure and isinstance(e, requests.exceptions.RequestException):
                    _restart_server()
                    try:
                        _ensure_server()
                        assert model_id is not None
                        raw = _call_chat_completions(
                            base_url,
                            model_id,
                            messages,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            max_tokens=args.max_tokens,
                            timeout_s=args.request_timeout_s,
                        )
                        total_model_calls += 1
                    except Exception:
                        failed += 1
                        illegal_action = True
                        break
                else:
                    failed += 1
                    illegal_action = True
                    break

            answer_text = _extract_tag(raw, _ANSWER_RE)
            frames_text = _extract_tag(raw, _FRAMES_RE)
            summary_text = _extract_tag(raw, _SUMMARY_RE)
            think_text = _extract_tag(raw, _THINK_RE)

            def _log_event(done: bool, **extra: Any) -> None:
                if not log_jsonl:
                    return
                rec = {
                    "sample_id": sample.sample_id,
                    "qid": sample.qid,
                    "video_path": sample.video_path,
                    "round": round_idx,
                    "done": bool(done),
                    "system_prompt": DEFAULT_SYSTEM_PROMPT.format(max_frames_per_round=args.max_frames_per_round),
                    "user_text": user_text,
                    "shown_frames": frames_this_round,
                    "seen_frames": seen_frames,
                    "candidate_unseen": candidate_unseen,
                    "raw_output": raw,
                    **extra,
                }
                with open(log_jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # Strict protocol checks + retry loop
            retries_left = int(args.max_retries_per_round)
            while True:
                invalid = False
                invalid_reason = None
                if think_text is not None:
                    invalid = True
                    invalid_reason = "invalid_think"
                if summary_text is None or _is_placeholder(summary_text) or _contains_banned_example(summary_text) or not _summary_has_ohrpu(summary_text):
                    invalid = True
                    invalid_reason = "invalid_summary"
                if answer_text:
                    norm = _normalize_answer_letter(answer_text, len(sample.choices))
                    if norm is None:
                        invalid = True
                        invalid_reason = "invalid_answer"
                    else:
                        final_answer = norm
                        answer_round = round_idx
                else:
                    if frames_text is None:
                        invalid = True
                        invalid_reason = "missing_frames"

                if not invalid:
                    break

                per_sample_invalid += 1
                invalid_outputs += 1
                if retries_left <= 0:
                    illegal_action = True
                    if args.strict_actions:
                        _log_event(True, illegal_action=True, invalid_reason=invalid_reason, retries_used=per_sample_retries)
                        break
                    break
                retries_left -= 1
                per_sample_retries += 1
                total_retries += 1
                feedback = (
                    "Invalid response: Output ONLY <summary> plus either <frames> (request) or <answer> (final). "
                    "Do NOT output <think>. Summary must contain P/O/H/U/R in order, with meaningful text (no placeholders). "
                    "If answering, <answer> must be exactly one letter (A/B/C/D/E)."
                )
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": feedback})
                try:
                    _ensure_server()
                    assert model_id is not None
                    raw = _call_chat_completions(
                        base_url,
                        model_id,
                        messages,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens,
                        timeout_s=args.request_timeout_s,
                    )
                    total_model_calls += 1
                except Exception:
                    illegal_action = True
                    break

                answer_text = _extract_tag(raw, _ANSWER_RE)
                frames_text = _extract_tag(raw, _FRAMES_RE)
                summary_text = _extract_tag(raw, _SUMMARY_RE)
                think_text = _extract_tag(raw, _THINK_RE)

            if illegal_action:
                break

            if summary_text:
                summary_state = summary_text.strip()

            if final_answer is not None:
                _log_event(True, final_answer=final_answer, retries_used=per_sample_retries, invalid_outputs=per_sample_invalid)
                break

            # Frame request path.
            assert frames_text is not None
            requested = _dedupe_preserve_order(_parse_frame_indices(frames_text))
            if args.use_candidate_frame_ids and candidate_unseen:
                mapped: list[int] = []
                invalid_id = False
                for cid in requested:
                    if 1 <= cid <= len(candidate_unseen):
                        mapped.append(int(candidate_unseen[cid - 1]))
                    else:
                        invalid_id = True
                if invalid_id:
                    per_sample_invalid += 1
                    invalid_outputs += 1
                    illegal_action = True if args.strict_actions else False
                    break
                requested = _dedupe_preserve_order(mapped)
            elif args.require_candidate_frames and candidate_unseen:
                allowed = {int(i) for i in candidate_unseen}
                requested = [i for i in requested if int(i) in allowed]
            if not requested:
                per_sample_invalid += 1
                invalid_outputs += 1
                illegal_action = True if args.strict_actions else False
                break

            allowed_ranges = _unseen_intervals(frame_count, seen_frames)
            next_frames = [i for i in requested if 0 <= i < frame_count and i not in seen_frames and _in_intervals(i, allowed_ranges)]
            if not next_frames:
                fallback_frames_used += 1
                # fallback: sample a new unseen frame
                fallback = _sample_unseen_frames(frame_count, set(seen_frames), args.max_frames_per_round, rng=rng)
                next_frames = fallback[: args.max_frames_per_round] if fallback else _sample_uniform_indices(frame_count, 1)
            frames_this_round = next_frames[: args.max_frames_per_round]

            _log_event(False, requested_frames=requested, next_frames=frames_this_round, retries_used=per_sample_retries)

        # End per-sample
        if final_answer is None:
            if args.force_final_answer and not illegal_action:
                # Final forced answer-only prompt (no images).
                user_text = (
                    f"Final round: answer now.\n{question_block}\n"
                    f"Total frames L = {frame_count}.\n"
                    "You must answer now.\n"
                    "Output ONLY <summary>...</summary> then <answer>LETTER</answer>."
                )
                messages = [
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT.format(max_frames_per_round=args.max_frames_per_round)},
                    {"role": "user", "content": user_text},
                ]
                try:
                    _ensure_server()
                    assert model_id is not None
                    raw = _call_chat_completions(
                        base_url,
                        model_id,
                        messages,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens,
                        timeout_s=args.request_timeout_s,
                    )
                    total_model_calls += 1
                    answer_text = _extract_tag(raw, _ANSWER_RE)
                    final_answer = _normalize_answer_letter(answer_text or "", len(sample.choices))
                    if final_answer is not None:
                        answer_round = int(args.max_rounds)
                        if log_jsonl:
                            rec = {
                                "sample_id": sample.sample_id,
                                "qid": sample.qid,
                                "video_path": sample.video_path,
                                "round": int(args.max_rounds) + 1,
                                "done": True,
                                "forced_answer": True,
                                "system_prompt": DEFAULT_SYSTEM_PROMPT.format(
                                    max_frames_per_round=args.max_frames_per_round
                                ),
                                "user_text": user_text,
                                "raw_output": raw,
                                "final_answer": final_answer,
                            }
                            with open(log_jsonl, "a", encoding="utf-8") as f:
                                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                except Exception:
                    final_answer = None

        if final_answer is None:
            failed += 1
        else:
            if final_answer == chr(ord("A") + int(sample.answer_idx)):
                correct += 1
            if answer_round is not None:
                total_rounds += int(min(int(answer_round), int(args.max_rounds)))

        if total % args.progress_interval == 0:
            acc = correct / max(1, total - failed)
            avg_r = total_rounds / max(1, total - failed)
            msg = f"[progress] {total}/{len(samples)} done | acc={acc:.3f} avg_rounds={avg_r:.2f} failed={failed} invalid={invalid_outputs} calls={total_model_calls}"
            print(msg, flush=True)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "progress/samples": total,
                        "progress/failed": failed,
                        "metrics/accuracy": acc,
                        "metrics/avg_rounds": avg_r,
                        "debug/invalid_outputs": invalid_outputs,
                        "debug/total_model_calls": total_model_calls,
                    }
                )

    elapsed_s = time.time() - start_time
    denom = max(1, total)
    acc = correct / max(1, total - failed) if total > failed else 0.0
    avg_rounds = total_rounds / max(1, total - failed) if total > failed else 0.0

    results = {
        "samples": total,
        "accuracy": acc,
        "avg_rounds": avg_rounds,
        "failed": failed,
        "elapsed_s": elapsed_s,
        "invalid_outputs": invalid_outputs,
        "total_retries": total_retries,
        "total_model_calls": total_model_calls,
        "fallback_frames_used": fallback_frames_used,
    }
    summary["results"] = results
    summary["log_jsonl"] = log_jsonl

    if log_jsonl and os.path.exists(log_jsonl):
        try:
            with open(log_jsonl, "r", encoding="utf-8") as f:
                prompt_lines = sum(1 for _ in f)
            prompt_bytes = os.path.getsize(log_jsonl)
        except Exception:
            prompt_lines = None
            prompt_bytes = None
        summary["prompt_log_lines"] = prompt_lines
        summary["prompt_log_bytes"] = prompt_bytes

    if wandb_run is not None:
        summary["wandb"] = {
            "enabled": True,
            "mode": wandb_run.settings.mode,
            "project": wandb_run.project,
            "entity": getattr(wandb_run, "entity", None),
            "name": wandb_run.name,
            "group": getattr(wandb_run, "group", None),
            "id": wandb_run.id,
            "run_dir": wandb_run.dir,
            "url": getattr(wandb_run, "url", None),
        }
        wandb_run.log(
            {
                "final/samples": total,
                "final/failed": failed,
                "final/accuracy": acc,
                "final/avg_rounds": avg_rounds,
                "final/invalid_outputs": invalid_outputs,
                "final/total_model_calls": total_model_calls,
                "final/fallback_frames_used": fallback_frames_used,
            }
        )
        wandb_run.finish()

    if args.summary_json:
        os.makedirs(os.path.dirname(args.summary_json) or ".", exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    if server_proc is not None:
        _stop_server(server_proc)

    print(json.dumps({"results": results, "summary_json": args.summary_json, "log_jsonl": log_jsonl}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
