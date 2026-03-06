"""Shared utility functions for REVISE plug-and-play evaluation scripts.

This module consolidates functions that were previously duplicated across multiple
standalone evaluation scripts (plug_and_play_nextqa_vllm.py, plug_and_play_egoschema_vllm.py,
plug_and_play_videomme_lvbench_vllm.py, plug_and_play_lvbench_hf.py, eval_nextqa_caption_vllm.py)
and the RL agent loop (verl/experimental/agent_loop/revise_agent_loop.py).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import time
from typing import Any, Optional

import requests
from PIL import Image

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None


# ---------------------------------------------------------------------------
# Regex constants for parsing model output tags
# ---------------------------------------------------------------------------

SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL | re.IGNORECASE)
FRAMES_RE = re.compile(r"<frames>(.*?)</frames>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

PLACEHOLDER_SET = {"...", "…", "none", "n/a", "na", "null", "unknown", "unsure", "uncertain"}


# ---------------------------------------------------------------------------
# Text & parsing helpers
# ---------------------------------------------------------------------------


def collapse_ws(text: str) -> str:
    """Collapse all whitespace to single spaces and strip."""
    return re.sub(r"\s+", " ", str(text)).strip()


def extract_tag(text: str, pattern: re.Pattern[str]) -> Optional[str]:
    """Extract the *last* match of a regex tag pattern, stripped."""
    matches = list(pattern.finditer(text or ""))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def dedupe_preserve_order(indices: list[int]) -> list[int]:
    """De-duplicate a list of ints, preserving first-occurrence order."""
    seen: set[int] = set()
    out: list[int] = []
    for idx in indices or []:
        if idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
    return out


def parse_int_list(text: str) -> list[int]:
    """Extract all integers from *text* via regex."""
    return [int(n) for n in re.findall(r"\d+", text or "")]


def normalize_answer_letter(answer_text: str, num_choices: int) -> Optional[str]:
    """Normalize a model answer to a single uppercase letter within the allowed range."""
    allowed = {chr(ord("A") + i) for i in range(max(0, num_choices))}
    if not allowed:
        allowed = {"A", "B", "C", "D", "E"}

    candidate = (answer_text or "").strip().upper()
    if candidate in allowed:
        return candidate

    match = re.search(r"\b([A-E])\b", candidate)
    if match and match.group(1).upper() in allowed:
        return match.group(1).upper()

    match = re.search(r"([A-E])", candidate)
    if match and match.group(1).upper() in allowed:
        return match.group(1).upper()
    return None


def is_placeholder(text: str) -> bool:
    """Detect placeholder / empty summary text."""
    t = collapse_ws(text).lower()
    if not t:
        return True
    if "..." in t or "…" in t:
        return True
    if t in PLACEHOLDER_SET:
        return True
    if re.fullmatch(r"[.·•…]+", t):
        return True
    alnum = re.findall(r"[a-z0-9]+", t)
    if len(alnum) <= 1 and len(t) <= 6:
        return True
    return False


def summary_has_ohrpu(summary_text: str) -> bool:
    """Check that summary contains P/O/H/U/R keys in the correct order."""
    if summary_text is None:
        return False
    s = collapse_ws(summary_text)
    keys = ["P", "O", "H", "U", "R"]
    positions = []
    for key in keys:
        m = re.search(rf"\b{key}\s*:\s*", s, re.IGNORECASE)
        if m is None:
            return False
        positions.append(m.start())
    return all(a < b for a, b in zip(positions, positions[1:], strict=False))


def contains_banned_example(text: str) -> bool:
    """Detect accidental copying of legacy few-shot example content."""
    t = collapse_ws(text).lower()
    if not t:
        return False
    if "george approaching a shelf" in t:
        return True
    if "george pauses" in t and "shelf" in t:
        return True
    return False


def truncate_text(text: str, max_chars: int) -> str:
    """Truncate *text* to *max_chars*, appending ellipsis if needed."""
    if max_chars <= 0:
        return text
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def normalize_video_id(video_id: Any) -> str:
    """Normalize a video ID to a string (handles int/float from pandas)."""
    if isinstance(video_id, int):
        return str(video_id)
    if isinstance(video_id, float):
        return str(int(video_id))
    return str(video_id)


# ---------------------------------------------------------------------------
# Frame selection & interval math
# ---------------------------------------------------------------------------


def sample_uniform_indices(frame_count: int, n: int) -> list[int]:
    """Uniformly sample *n* indices from [0, frame_count-1]."""
    if n <= 0:
        return []
    if frame_count <= 0:
        return list(range(n))
    if n == 1:
        return [frame_count // 2]
    if frame_count == 1:
        return [0]
    return [round(i * (frame_count - 1) / (n - 1)) for i in range(n)]


def linspace(a: float, b: float, n: int) -> list[float]:
    """Pure-Python linspace (no numpy dependency)."""
    if n <= 1:
        return [a]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]


def sample_uniform_indices_inclusive(start: int, end: int, k: int) -> list[int]:
    """Uniformly sample *k* indices in [start, end] inclusive."""
    if k <= 0:
        return []
    if end < start:
        return []
    if start == end:
        return [start]
    out = [int(round(i)) for i in linspace(float(start), float(end), k)]
    out = [max(start, min(i, end)) for i in out]
    return dedupe_preserve_order(out)


def indices_to_intervals(indices: list[int]) -> list[tuple[int, int]]:
    """Convert a list of indices to inclusive [start, end] intervals."""
    if not indices:
        return []
    sorted_unique = sorted({int(i) for i in indices})
    intervals: list[tuple[int, int]] = []
    start = prev = sorted_unique[0]
    for idx in sorted_unique[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        intervals.append((start, prev))
        start = prev = idx
    intervals.append((start, prev))
    return intervals


def unseen_intervals(frame_count: int, seen_frames: list[int]) -> list[tuple[int, int]]:
    """Return unseen frame ranges as inclusive [start, end] intervals."""
    if frame_count <= 0:
        return []
    seen = sorted({int(i) for i in (seen_frames or []) if 0 <= int(i) < frame_count})
    anchors = [-1, *seen, frame_count]
    intervals: list[tuple[int, int]] = []
    for a, b in zip(anchors, anchors[1:], strict=False):
        s = a + 1
        e = b - 1
        if s <= e:
            intervals.append((s, e))
    return intervals


def in_intervals(idx: int, intervals: list[tuple[int, int]]) -> bool:
    """Check if *idx* falls within any of the given inclusive intervals."""
    for start, end in intervals:
        if start <= idx <= end:
            return True
    return False


def format_intervals(intervals: list[tuple[int, int]]) -> str:
    """Format intervals as a semicolon-separated string."""
    if not intervals:
        return "none"
    parts: list[str] = []
    for start, end in intervals:
        if start == end:
            parts.append(str(start))
        else:
            parts.append(f"{start}-{end}")
    return "; ".join(parts)


def format_frame_list(frames: list[int]) -> str:
    """Format a frame index list for display in prompts."""
    if not frames:
        return "no frames yet"
    return ", ".join(str(int(i)) for i in frames)


def propose_candidate_frames(frame_count: int, seen: set[int], k: int, rng: random.Random) -> list[int]:
    """Propose *k* candidate NEW frame indices using gap-filling + random fill.

    Strategy: pick midpoints of the largest gaps between already-seen frames,
    then fill remaining slots with random unseen frames.
    """
    if frame_count <= 0 or k <= 0:
        return []
    if len(seen) >= frame_count:
        return []

    seen_sorted = sorted(i for i in seen if 0 <= i < frame_count)
    anchors = sorted(set([0, frame_count - 1, *seen_sorted]))

    candidates: list[int] = []
    gaps: list[tuple[int, int, int]] = []
    for a, b in zip(anchors, anchors[1:], strict=False):
        if b - a <= 1:
            continue
        gaps.append((b - a, a, b))
    gaps.sort(reverse=True)
    for _, a, b in gaps:
        mid = (a + b) // 2
        for d in (0, -1, 1, -2, 2, -3, 3):
            idx = mid + d
            if a < idx < b and idx not in seen and idx not in candidates:
                candidates.append(idx)
                break
        if len(candidates) >= k:
            return sorted(candidates[:k])

    need = k - len(candidates)
    if need > 0:
        remaining = [i for i in range(frame_count) if i not in seen and i not in candidates]
        if remaining:
            fill = rng.sample(remaining, k=min(need, len(remaining)))
            candidates.extend(sorted(fill))

    return sorted(candidates[:k])


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------


def extract_video_info(video_path: str) -> tuple[int, float]:
    """Return (total_frames, fps) using decord."""
    import decord

    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    total_frames = int(len(vr))
    fps = float(vr.get_avg_fps() or 0.0)
    if fps <= 0:
        fps = 30.0
    return total_frames, fps


def timeline_len_1fps(total_frames: int, fps: float) -> int:
    """Convert (total_frames, fps) to a 1fps timeline length (seconds)."""
    if total_frames <= 0:
        return 0
    duration_s = total_frames / max(1e-6, fps)
    return max(1, int(math.ceil(duration_s)))


def timeline_to_frame_idx(timeline_idx: int, fps: float, total_frames: int) -> int:
    """Map a 1fps timeline index (seconds) to a raw frame index."""
    if total_frames <= 0:
        return 0
    t = max(0.0, float(timeline_idx))
    idx = int(t * fps)
    return max(0, min(idx, total_frames - 1))


def extract_frames_1fps(video_path: str, timeline_indices: list[int]) -> list[Image.Image]:
    """Extract frames mapped from a 1fps timeline to actual frame indices."""
    if not timeline_indices:
        return []
    import decord

    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    total_frames = int(len(vr))
    fps = float(vr.get_avg_fps() or 0.0)
    if fps <= 0:
        fps = 30.0

    frame_indices = [timeline_to_frame_idx(i, fps, total_frames) for i in timeline_indices]
    try:
        frames = vr.get_batch(frame_indices).asnumpy()
        return [Image.fromarray(frame) for frame in frames]
    except Exception:
        out: list[Image.Image] = []
        for idx in frame_indices:
            out.append(Image.fromarray(vr[idx].asnumpy()))
        return out


def extract_frames(video_path: str, frame_indices: list[int]) -> list[Image.Image]:
    """Extract frames by raw index using decord (with imageio fallback)."""
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


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------


def b64_jpeg(img: Image.Image, *, max_edge: int = 0, quality: int = 90) -> str:
    """Base64-encode an image as JPEG, optionally resizing to *max_edge*.

    Args:
        img: PIL Image to encode.
        max_edge: If > 0, thumbnail the image so its longest edge is at most this.
        quality: JPEG quality (1-100).
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    if max_edge > 0:
        w, h = img.size
        if max(w, h) > max_edge:
            img = img.copy()
            img.thumbnail((max_edge, max_edge), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Time-reference parsing (LVBench)
# ---------------------------------------------------------------------------


def parse_time_to_seconds(text: str) -> Optional[float]:
    """Parse 'MM:SS' or 'HH:MM:SS' to seconds."""
    raw = collapse_ws(text)
    if not raw:
        return None
    parts = [p for p in raw.split(":") if p]
    if len(parts) == 2:
        try:
            mm = int(parts[0])
            ss = float(parts[1])
            return max(0.0, mm * 60.0 + ss)
        except Exception:
            return None
    if len(parts) == 3:
        try:
            hh = int(parts[0])
            mm = int(parts[1])
            ss = float(parts[2])
            return max(0.0, hh * 3600.0 + mm * 60.0 + ss)
        except Exception:
            return None
    return None


def parse_time_reference_range(time_reference: str, timeline_len: int) -> Optional[tuple[int, int]]:
    """Parse LVBench time_reference (e.g. '04:19-08:41') into a (start, end) range on the 1fps timeline."""
    tr = collapse_ws(time_reference)
    if not tr or tr.lower() in {"n/a", "na", "none"}:
        return None
    if "-" not in tr:
        return None
    left, right = (s.strip() for s in tr.split("-", 1))
    start_s = parse_time_to_seconds(left)
    end_s = parse_time_to_seconds(right)
    if start_s is None or end_s is None:
        return None
    if timeline_len <= 0:
        return None

    start = int(math.floor(start_s))
    end = int(math.ceil(end_s))
    if end < start:
        start, end = end, start
    start = max(0, min(start, timeline_len - 1))
    end = max(0, min(end, timeline_len - 1))
    if end < start:
        return None
    return start, end


# ---------------------------------------------------------------------------
# Question formatting
# ---------------------------------------------------------------------------


def format_question_block(question: str, options: list[str]) -> str:
    """Format a question with labeled options (A, B, C, ...)."""
    q = str(question).strip()
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines = ["Question: " + q, "Options:"]
    for i, opt in enumerate(options):
        prefix = letters[i] if i < len(letters) else str(i)
        lines.append(f"{prefix}. {str(opt).strip()}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Logging & monitoring
# ---------------------------------------------------------------------------


def maybe_log_jsonl(path: Optional[str], obj: dict[str, Any]) -> None:
    """Append a JSON object to a JSONL file (no-op if *path* is falsy)."""
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def maybe_init_wandb(args: argparse.Namespace, run_config: dict[str, Any]) -> Any:
    """Initialize a wandb run if ``--use-wandb`` is set and credentials are available.

    Expects *args* to have: ``use_wandb``, ``wandb_project``, ``wandb_entity``,
    ``wandb_name``, ``wandb_group``, ``wandb_tags``, ``wandb_mode``.
    """
    if not getattr(args, "use_wandb", False) or wandb is None:
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

    mode = getattr(args, "wandb_mode", "") or os.getenv("WANDB_MODE")
    if not mode:
        mode = "online" if _has_wandb_credentials() else "offline"
    if str(mode).lower() == "online" and not _has_wandb_credentials():
        mode = "offline"

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


def wandb_log(run: Any, metrics: dict[str, Any], step: int) -> None:
    """Log metrics to wandb (no-op if *run* is None)."""
    if run is None or wandb is None:
        return
    wandb.log(metrics, step=step)


# ---------------------------------------------------------------------------
# Stable sample ID
# ---------------------------------------------------------------------------


def stable_sample_id(*, keys: dict[str, Any]) -> str:
    """SHA1 hash of arbitrary key-value metadata for deterministic sample IDs."""
    return hashlib.sha1(json.dumps(keys, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def stable_sample_id_nextqa(video_id: str, question: str, choices: list[str], answer_idx: int) -> str:
    """NExT-QA specific sample ID."""
    return stable_sample_id(
        keys={
            "video_id": str(video_id),
            "question": str(question),
            "choices": [str(c) for c in (choices or [])],
            "answer_idx": int(answer_idx),
        }
    )


def stable_sample_id_dataset(dataset: str, video_key: str, uid: str) -> str:
    """Video-MME / LVBench style sample ID."""
    return stable_sample_id(keys={"dataset": str(dataset), "video": str(video_key), "uid": str(uid)})


# ---------------------------------------------------------------------------
# LVBench question parsing
# ---------------------------------------------------------------------------


_OPT_RE = re.compile(r"^\(([A-Z])\)\s*(.*)$")


def parse_options_from_lvbench_question(question: str) -> tuple[str, list[str]]:
    """Parse LVBench question format with ``(A)`` prefix options."""
    text = str(question or "").strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "", []

    q_lines: list[str] = []
    opts: dict[str, str] = {}
    for ln in lines:
        m = _OPT_RE.match(ln)
        if m:
            opts[m.group(1).upper()] = m.group(2).strip()
        else:
            q_lines.append(ln)

    q_text = " ".join(q_lines).strip()
    if not opts:
        return q_text, []
    letters_sorted = sorted(opts.keys())
    options = [opts[k] for k in letters_sorted]
    return q_text, options


# ---------------------------------------------------------------------------
# Sharding
# ---------------------------------------------------------------------------


def shard_by_video(samples: list[Any], num_shards: int, shard_idx: int, *, video_key_attr: str = "video_key") -> list:
    """Shard a sample list by video key (deterministic round-robin)."""
    if num_shards <= 1:
        return samples
    if not (0 <= shard_idx < num_shards):
        raise ValueError(f"--shard-idx must be in [0, {num_shards}) (got {shard_idx})")

    by_video: dict[str, list] = {}
    for s in samples:
        vk = getattr(s, video_key_attr)
        by_video.setdefault(vk, []).append(s)
    video_keys = sorted(by_video.keys())
    my_videos = {vk for i, vk in enumerate(video_keys) if (i % num_shards) == shard_idx}

    out: list = []
    for vk in video_keys:
        if vk not in my_videos:
            continue
        out.extend(by_video[vk])
    return out


# ---------------------------------------------------------------------------
# vLLM server management
# ---------------------------------------------------------------------------


def pick_free_port() -> int:
    """Pick a free TCP port on localhost."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    _, port = s.getsockname()
    s.close()
    return int(port)


def port_is_open(host: str, port: int) -> bool:
    """Check if a TCP port is accepting connections."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        try:
            return sock.connect_ex((host, port)) == 0
        except Exception:
            return False


def wait_port(host: str, port: int, timeout_s: int = 300) -> None:
    """Wait until *host:port* accepts TCP connections."""
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"Timed out waiting for {host}:{port} to accept connections after {timeout_s}s")


def wait_for_server(host: str, port: int, timeout_s: int) -> None:
    """Wait until the vLLM OpenAI server is actually ready (polls ``/v1/models``).

    vLLM can open the TCP port before the model is fully initialized; issuing a chat
    request too early can return transient HTTP 400/503 errors.
    """
    base_url = f"http://{host}:{port}"
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            resp = requests.get(f"{base_url}/v1/models", timeout=1.0)
            if resp.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"vLLM server did not become ready at {host}:{port} within {timeout_s}s")


def get_model_id(base_url: str, timeout: int = 30) -> str:
    """Fetch the model ID from a running vLLM/OpenAI-compatible server."""
    resp = requests.get(f"{base_url}/v1/models", timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    models = data.get("data", [])
    if not models:
        raise RuntimeError(f"No models returned from {base_url}/v1/models")
    return models[0]["id"]


def stop_server(proc: subprocess.Popen) -> None:
    """Gracefully terminate a server subprocess."""
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def suffix_path(path: str, suffix: str) -> str:
    """Add a suffix before the file extension (e.g. ``'log.jsonl'`` -> ``'log.shard0of4.jsonl'``)."""
    root, ext = os.path.splitext(path)
    return f"{root}{suffix}{ext}" if ext else f"{path}{suffix}"
