#!/usr/bin/env python3

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import requests
from datasets import load_dataset
from PIL import Image

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL | re.IGNORECASE)
_FRAMES_RE = re.compile(r"<frames>(.*?)</frames>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


DEFAULT_SYSTEM_PROMPT_WITH_THINK = (
    "You are REVISE, a multi-round video reasoning agent.\n"
    "Each round you will see: (1) a multiple-choice question with options, (2) the current belief summary, "
    "and (3) a few sampled video frames.\n"
    "If you are confident, answer the question.\n"
    "If you are NOT confident, request MORE video frames to view NEXT.\n"
    "Frames are sampled at 1 fps; frame index ≈ timestamp in seconds.\n\n"
    "IMPORTANT: Output must follow EXACTLY ONE of the two formats below. Do not output any text outside tags.\n"
    "Output <think> then <summary> plus either <frames> (request) OR <answer> (final).\n"
    "Do NOT output any other tags.\n\n"
    "Format 1 — Request more frames (use this only if NOT confident):\n"
    "<think>short reasoning based on shown frames; no final answer letter</think>\n"
    "<summary>P: ...; O: ...; H: ...; U: ...; R: request additional frames</summary>\n"
    "<frames>1, 3</frames>\n\n"
    "Format 2 — Answer now (use this if confident):\n"
    "<think>short reasoning based on shown frames; no extra text</think>\n"
    "<summary>P: ...; O: ...; H: ...; U: ...; R: answered</summary>\n"
    "<answer>B</answer>\n\n"
    "Tag meanings:\n"
    "- <think>: scratchpad reasoning for this round (will NOT be carried across rounds).\n"
    "- <summary>: the ONLY persistent memory across rounds. Keep it short and update it EVERY round.\n"
    "  The summary MUST be written in this exact order: P → O → H → U → R.\n"
    "  - P (Previously seen): what was seen so far (describe content; do NOT list indices).\n"
    "  - O (Observations): what you observe in the current frames.\n"
    "  - H (Belief updates): updated belief based on evidence so far (do NOT include the final answer letter).\n"
    "  - U (Uncertainties): what is still unknown or ambiguous.\n"
    "  - R (Reasons): what evidence you need next (or 'answered').\n\n"
    "Rules:\n"
    "- Frame indices are 0-based in [0, L-1] and correspond to seconds on the 1-fps timeline.\n"
    "- If requesting, choose 1 to {max_frames_per_round} NEW frames to view NEXT.\n"
    "- Do NOT output any frame index from the Seen frames list; those are already viewed.\n"
    "- When Candidate Frame IDs are provided, output those IDs (1..K) in <frames> instead of raw frame indices.\n"
    "- In <frames>, output comma-separated integers only (no brackets, no text).\n"
    "- In <answer>, output EXACTLY ONE option letter shown in the question (e.g., A/B/C/D/E). No words.\n"
)


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _extract_tag(text: str, pattern: re.Pattern[str]) -> Optional[str]:
    matches = list(pattern.finditer(text or ""))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def _parse_int_list(text: str) -> list[int]:
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


def _parse_time_to_seconds(text: str) -> Optional[float]:
    raw = _collapse_ws(text)
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


def _parse_time_reference_range(time_reference: str, timeline_len: int) -> Optional[tuple[int, int]]:
    """Parse LVBench time_reference (e.g. '04:19-08:41') into a (start,end) range on the 1fps timeline."""
    tr = _collapse_ws(time_reference)
    if not tr or tr.lower() in {"n/a", "na", "none"}:
        return None
    if "-" not in tr:
        return None
    left, right = (s.strip() for s in tr.split("-", 1))
    start_s = _parse_time_to_seconds(left)
    end_s = _parse_time_to_seconds(right)
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


def _b64_jpeg(img: Image.Image) -> str:
    # Qwen2.5-VL can exceed `max_model_len` when fed high-resolution frames (image token count depends on
    # resolution/aspect ratio). Downscale to keep prompts within budget while still using real frames.
    max_edge = 512
    try:
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > max_edge:
            img = img.copy()
            img.thumbnail((max_edge, max_edge), Image.LANCZOS)
    except Exception:
        pass
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _format_question_block(question: str, options: list[str]) -> str:
    q = str(question).strip()
    lines = ["Question: " + q, "Options:"]
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i, opt in enumerate(options):
        prefix = letters[i] if i < len(letters) else str(i)
        lines.append(f"{prefix}. {str(opt).strip()}")
    return "\n".join(lines)


def _normalize_answer_letter(ans: str, num_choices: int) -> Optional[str]:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if not ans:
        return None
    a = _collapse_ws(ans).strip().upper()
    if len(a) == 1 and a in letters[: max(1, num_choices)]:
        return a
    m = re.search(r"\b([A-Z])\b", a)
    if m:
        cand = m.group(1)
        if cand in letters[: max(1, num_choices)]:
            return cand
    return None


def _stable_sample_id(dataset: str, video_key: str, uid: str) -> str:
    payload = {"dataset": str(dataset), "video": str(video_key), "uid": str(uid)}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _sample_uniform_indices(frame_count: int, k: int) -> list[int]:
    if frame_count <= 0 or k <= 0:
        return []
    if frame_count == 1:
        return [0]
    if k == 1:
        return [frame_count // 2]
    return [int(round(i)) for i in list(_linspace(0, frame_count - 1, k))]


def _sample_uniform_indices_inclusive(start: int, end: int, k: int) -> list[int]:
    if k <= 0:
        return []
    if end < start:
        return []
    if start == end:
        return [start]
    out = [int(round(i)) for i in _linspace(float(start), float(end), k)]
    out = [max(start, min(i, end)) for i in out]
    return _dedupe_preserve_order(out)


def _linspace(a: float, b: float, n: int) -> list[float]:
    if n <= 1:
        return [a]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]


def _propose_candidate_frames(frame_count: int, seen: set[int], k: int, rng: random.Random) -> list[int]:
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


def _maybe_log_jsonl(path: Optional[str], obj: dict[str, Any]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _pick_free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    _, port = s.getsockname()
    s.close()
    return int(port)


def _port_is_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        try:
            return sock.connect_ex((host, port)) == 0
        except Exception:
            return False


def _wait_for_server(host: str, port: int, timeout_s: int) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        if _port_is_open(host, port):
            return
        time.sleep(0.5)
    raise TimeoutError(f"vLLM server did not open {host}:{port} within {timeout_s}s")


def _start_vllm_server(args: argparse.Namespace) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env.pop("ROCR_VISIBLE_DEVICES", None)
    env.pop("HIP_VISIBLE_DEVICES", None)
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")

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
    stdout = subprocess.DEVNULL
    stderr = subprocess.DEVNULL
    if args.server_log:
        log_dir = os.path.dirname(args.server_log)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        log_f = open(args.server_log, "a", encoding="utf-8")
        stdout = log_f
        stderr = log_f
    return subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr)


def _stop_server(proc: subprocess.Popen[str]) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


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
    content: list[dict[str, Any]] = []
    parts = user_text.split("<image>") if images else [user_text]
    if images and (len(parts) - 1) == len(images):
        for i, img in enumerate(images):
            if parts[i]:
                content.append({"type": "text", "text": parts[i]})
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64_jpeg(img)}"}}
            )
        if parts[-1]:
            content.append({"type": "text", "text": parts[-1]})
    else:
        for img in images:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64_jpeg(img)}"}})
        content.append({"type": "text", "text": user_text})

    payload = {
        "model": model_id,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=timeout_s)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        body = ""
        try:
            body = (resp.text or "").strip()
        except Exception:
            body = ""
        if len(body) > 2000:
            body = body[:2000] + "…"
        raise RuntimeError(f"vLLM HTTP {resp.status_code}: {body}") from e
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _maybe_init_wandb(args: argparse.Namespace, run_config: dict[str, Any]) -> Any:
    if not getattr(args, "use_wandb", False) or wandb is None:
        return None

    def _has_wandb_credentials() -> bool:
        if os.getenv("WANDB_API_KEY"):
            return True
        if os.getenv("WANDB_IDENTITY_TOKEN_FILE"):
            return True
        try:
            from wandb.sdk.lib.wbauth import wbnetrc

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


def _wandb_log(run: Any, metrics: dict[str, Any], step: int) -> None:
    if run is None or wandb is None:
        return
    wandb.log(metrics, step=step)


def _retry_feedback_text(feedback: str, *, force_answer: bool) -> str:
    if force_answer:
        return (
            f"{feedback}\n"
            "You MUST answer now. Output <think>...</think> then <summary>...</summary> then <answer>LETTER</answer>."
        )
    return f"{feedback}\nPlease respond with one of the required formats."


def _extract_video_info(video_path: str) -> tuple[int, float]:
    """Return (total_frames, fps)."""
    import decord

    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    total_frames = int(len(vr))
    fps = float(vr.get_avg_fps() or 0.0)
    if fps <= 0:
        # Conservative fallback (30fps).
        fps = 30.0
    return total_frames, fps


def _timeline_len_1fps(total_frames: int, fps: float) -> int:
    if total_frames <= 0:
        return 0
    duration_s = total_frames / max(1e-6, fps)
    return max(1, int(math.ceil(duration_s)))


def _timeline_to_frame_idx(timeline_idx: int, fps: float, total_frames: int) -> int:
    if total_frames <= 0:
        return 0
    t = max(0.0, float(timeline_idx))
    idx = int(t * fps)
    return max(0, min(idx, total_frames - 1))


def _extract_frames_1fps(video_path: str, timeline_indices: list[int]) -> list[Image.Image]:
    if not timeline_indices:
        return []
    import decord

    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    total_frames = int(len(vr))
    fps = float(vr.get_avg_fps() or 0.0)
    if fps <= 0:
        fps = 30.0

    frame_indices = [_timeline_to_frame_idx(i, fps, total_frames) for i in timeline_indices]
    try:
        frames = vr.get_batch(frame_indices).asnumpy()
        return [Image.fromarray(frame) for frame in frames]
    except Exception:
        # Fallback to per-frame reads.
        out: list[Image.Image] = []
        for idx in frame_indices:
            out.append(Image.fromarray(vr[idx].asnumpy()))
        return out


def _ensure_yt_dlp(py_bin: str) -> list[str]:
    """Return command prefix to invoke yt-dlp (binary or `python -m yt_dlp`)."""
    if shutil.which("yt-dlp"):
        return ["yt-dlp"]
    # Fall back to module invocation.
    return [py_bin, "-m", "yt_dlp"]


def _download_youtube(url: str, out_mp4: str, *, py_bin: str, timeout_s: int) -> None:
    out_mp4_path = Path(out_mp4)
    out_mp4_path.parent.mkdir(parents=True, exist_ok=True)
    out_tmpl = str(out_mp4_path.with_suffix("")) + ".%(ext)s"

    # yt-dlp increasingly requires a JS runtime for YouTube extraction. We prefer Node if present.
    node_path = shutil.which("node")
    js_runtime_args: list[str] = []
    if node_path:
        js_runtime_args = ["--js-runtimes", f"node:{node_path}"]

    cmd = [
        *_ensure_yt_dlp(py_bin),
        *js_runtime_args,
        "--no-playlist",
        "--merge-output-format",
        "mp4",
        "--extractor-args",
        "youtube:player_client=android",
        "-f",
        "best[ext=mp4][height<=480]/best[ext=mp4]/best",
        "-o",
        out_tmpl,
        url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0:
        raise RuntimeError(f"yt-dlp failed ({proc.returncode}): {proc.stderr.strip()[:500]}")

    # Find the merged mp4.
    if out_mp4_path.exists() and out_mp4_path.stat().st_size > 0:
        return
    # Sometimes yt-dlp still leaves a different extension; try to locate by stem.
    candidates = list(out_mp4_path.parent.glob(out_mp4_path.stem + ".*"))
    for c in candidates:
        if c.suffix.lower() == ".mp4" and c.stat().st_size > 0:
            c.rename(out_mp4_path)
            return
    raise FileNotFoundError(f"Downloaded file not found for {url} (expected {out_mp4_path})")


@dataclass
class MCVideoSample:
    dataset: str
    uid: str
    video_key: str
    video_url: str
    question: str
    options: list[str]
    answer_letter: str
    time_reference: str = ""

    @property
    def sample_id(self) -> str:
        return _stable_sample_id(self.dataset, self.video_key, self.uid)


def _parse_options_from_lvbench_question(question: str) -> tuple[str, list[str]]:
    text = str(question or "").strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "", []

    opt_re = re.compile(r"^\(([A-Z])\)\s*(.*)$")
    q_lines: list[str] = []
    opts: dict[str, str] = {}
    for ln in lines:
        m = opt_re.match(ln)
        if m:
            opts[m.group(1)] = m.group(2).strip()
        else:
            q_lines.append(ln)

    q_text = " ".join(q_lines).strip()
    if not opts:
        return q_text, []
    letters_sorted = sorted(opts.keys())
    options = [opts[k] for k in letters_sorted]
    return q_text, options


def _load_videomme_samples(split: str) -> list[MCVideoSample]:
    ds = load_dataset("lmms-lab/Video-MME", split=split)
    samples: list[MCVideoSample] = []
    for ex in ds:
        video_id = str(ex.get("videoID") or ex.get("video_id") or "").strip()
        url = str(ex.get("url") or "").strip()
        qid = str(ex.get("question_id") or ex.get("qid") or "").strip()
        question = str(ex.get("question") or "").strip()
        options_raw = ex.get("options") or []
        if not isinstance(options_raw, list):
            options_raw = []
        options: list[str] = []
        for opt in options_raw:
            s = str(opt).strip()
            m = re.match(r"^[A-Z]\s*[.)]\s*(.*)$", s)
            options.append(m.group(1).strip() if m else s)
        answer = str(ex.get("answer") or "").strip().upper()
        samples.append(
            MCVideoSample(
                dataset="videomme",
                uid=qid or _stable_sample_id("videomme", video_id, question),
                video_key=f"{video_id}.mp4",
                video_url=url,
                question=question,
                options=options,
                answer_letter=answer,
            )
        )
    return samples


def _load_lvbench_samples(split: str) -> list[MCVideoSample]:
    ds = load_dataset("lmms-lab/LVBench", split=split)
    samples: list[MCVideoSample] = []
    for ex in ds:
        video_path = str(ex.get("video_path") or "").strip()
        uid = str(ex.get("uid") or ex.get("key") or "").strip()
        q_raw = str(ex.get("question") or "").strip()
        q_text, options = _parse_options_from_lvbench_question(q_raw)
        answer = str(ex.get("answer") or "").strip().upper()
        time_reference = str(ex.get("time_reference") or "").strip()
        video_id = Path(video_path).stem
        url = f"https://www.youtube.com/watch?v={video_id}"
        samples.append(
            MCVideoSample(
                dataset="lvbench",
                uid=uid or _stable_sample_id("lvbench", video_path, q_raw),
                video_key=video_path,
                video_url=url,
                question=q_text if q_text else q_raw,
                options=options,
                answer_letter=answer,
                time_reference=time_reference,
            )
        )
    return samples


def _shard_by_video(samples: list[MCVideoSample], num_shards: int, shard_idx: int) -> list[MCVideoSample]:
    if num_shards <= 1:
        return samples
    if not (0 <= shard_idx < num_shards):
        raise ValueError(f"--shard-idx must be in [0, {num_shards}) (got {shard_idx})")

    by_video: dict[str, list[MCVideoSample]] = {}
    for s in samples:
        by_video.setdefault(s.video_key, []).append(s)
    video_keys = sorted(by_video.keys())
    my_videos = {vk for i, vk in enumerate(video_keys) if (i % num_shards) == shard_idx}

    out: list[MCVideoSample] = []
    for vk in video_keys:
        if vk not in my_videos:
            continue
        out.extend(by_video[vk])
    return out


def _build_user_text(
    question_block: str,
    summary: str,
    timeline_len: int,
    round_idx: int,
    current_frames: list[int],
    seen_frames: list[int],
    candidate_unseen_frames: list[int],
    use_candidate_frame_ids: bool,
    require_candidate_frames: bool,
    time_reference: str = "",
) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    allowed_letters = ", ".join(list(letters[: max(1, question_block.count("\n") - 1)]))  # heuristic

    lines: list[str] = []
    lines.append(f"Round {round_idx} / Question:")
    lines.append(question_block)
    if allowed_letters:
        lines.append(
            f"To answer, output <think>...</think> then <summary>...</summary> then <answer>LETTER</answer> "
            f"(LETTER must be one of: {allowed_letters})."
        )
    lines.append(f"Total frames L = {timeline_len} (1 fps timeline).")
    if time_reference:
        lines.append(f"Relevant time window for this question: {time_reference} (focus on this segment).")
    lines.append(
        f"Seen frames: {len(seen_frames)} frames already viewed (do NOT request any previously shown frames)."
    )
    if use_candidate_frame_ids and candidate_unseen_frames:
        lines.append(
            f"Candidate unseen frames available as IDs (all NEW): choose IDs in [1, {len(candidate_unseen_frames)}]."
        )
        id_map = ", ".join(f"{i+1}->{t}s" for i, t in enumerate(candidate_unseen_frames))
        lines.append(f"Candidate ID -> timeline second: {id_map}")
        lines.append("In <frames>, output ONLY candidate IDs (comma-separated). Do NOT output raw indices when IDs exist.")
        if require_candidate_frames:
            lines.append("IMPORTANT: You MUST choose frames only from the Candidate IDs.")
    lines.extend(["Current summary:", f"<summary>{summary}</summary>", "Frames shown in this round:"])
    for i in range(len(current_frames)):
        lines.append(f"Shown frame {letters[i]} <image>")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["videomme", "lvbench"], required=True)
    ap.add_argument("--split", default="")
    ap.add_argument("--video-cache-dir", default="/tmp/chenwei_video_cache")
    ap.add_argument("--max-samples", type=int, default=0)

    ap.add_argument("--model-path", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--start-server", action="store_true")
    ap.add_argument("--restart-server-on-failure", action="store_true")
    ap.add_argument("--server-log", default="")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-model-len", type=int, default=12288)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.6)

    ap.add_argument("--max-rounds", type=int, default=5)
    ap.add_argument("--max-frames-per-round", type=int, default=5)
    ap.add_argument("--candidate-k", type=int, default=20)
    ap.add_argument("--use-candidate-frame-ids", action="store_true", default=True)
    ap.add_argument("--require-candidate-frames", action="store_true", default=True)
    ap.add_argument("--max-retries-per-round", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout-s", type=int, default=120)
    ap.add_argument("--force-final-answer", action="store_true", default=True)

    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-idx", type=int, default=0)

    ap.add_argument("--log-jsonl", default="")
    ap.add_argument("--summary-json", default="")
    ap.add_argument("--resume-from-log", action="store_true")

    ap.add_argument("--use-wandb", action="store_true")
    ap.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "verl-revise"))
    ap.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY"))
    ap.add_argument("--wandb-name", default=os.getenv("WANDB_RUN_NAME"))
    ap.add_argument("--wandb-group", default=os.getenv("WANDB_RUN_GROUP"))
    ap.add_argument("--wandb-tags", default=os.getenv("WANDB_TAGS", ""))
    ap.add_argument("--wandb-mode", default=os.getenv("WANDB_MODE", ""))

    ap.add_argument("--yt-dlp-timeout-s", type=int, default=600)

    args = ap.parse_args()

    if args.port <= 0:
        args.port = _pick_free_port()

    split = args.split
    if not split:
        split = "test" if args.dataset == "videomme" else "train"

    if args.dataset == "videomme":
        samples = _load_videomme_samples(split)
    else:
        samples = _load_lvbench_samples(split)

    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    samples = _shard_by_video(samples, args.num_shards, args.shard_idx)

    # Stable order to improve download locality.
    samples.sort(key=lambda s: (s.video_key, s.uid))

    if not samples:
        raise SystemExit("No samples selected (check --split/--max-samples/--sharding).")

    # Auto-suffix output paths for multi-shard runs.
    if args.num_shards > 1:
        suffix = f".shard{args.shard_idx}of{args.num_shards}"

        def _suffix_path(path: str) -> str:
            root, ext = os.path.splitext(path)
            return f"{root}{suffix}{ext}" if ext else f"{path}{suffix}"

        if args.log_jsonl and suffix not in args.log_jsonl:
            args.log_jsonl = _suffix_path(args.log_jsonl)
        if args.summary_json and suffix not in args.summary_json:
            args.summary_json = _suffix_path(args.summary_json)

    resume_completed = 0
    correct = 0
    total_rounds = 0
    invalid_outputs = 0
    invalid_action_terminated = 0
    failed = 0
    total_model_calls = 0
    total_retries = 0
    total_effective_rounds = 0
    think_present = 0

    if args.resume_from_log and args.log_jsonl and os.path.exists(args.log_jsonl):
        # Cheap resume heuristic: count answered samples in log.
        seen_samples: set[str] = set()
        with open(args.log_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                sid = obj.get("sample_id")
                if sid and "<answer>" in str(obj.get("raw_output", "")).lower():
                    seen_samples.add(sid)
        resume_completed = len(seen_samples)
        if resume_completed > 0:
            print(f"[resume] detected {resume_completed} completed samples in {args.log_jsonl}")

    if resume_completed > 0:
        samples = samples[resume_completed:]

    server_proc: Optional[subprocess.Popen[str]] = None
    if args.start_server:
        server_proc = _start_vllm_server(args)
        _wait_for_server(args.host, args.port, timeout_s=240)

    run_config = {
        "task": "revise_plug_and_play_videomme_lvbench_vllm",
        "dataset": args.dataset,
        "split": split,
        "model_path": args.model_path,
        "video_cache_dir": args.video_cache_dir,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_samples": args.max_samples,
        "num_shards": args.num_shards,
        "shard_idx": args.shard_idx,
        "max_rounds": args.max_rounds,
        "max_frames_per_round": args.max_frames_per_round,
        "candidate_k": args.candidate_k,
        "max_retries_per_round": args.max_retries_per_round,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "use_candidate_frame_ids": bool(args.use_candidate_frame_ids),
        "require_candidate_frames": bool(args.require_candidate_frames),
        "force_final_answer": bool(args.force_final_answer),
        "log_jsonl": args.log_jsonl,
        "summary_json": args.summary_json,
    }
    run = _maybe_init_wandb(args, run_config)

    base_url = f"http://{args.host}:{args.port}"
    model_id = args.model_path
    system_prompt = DEFAULT_SYSTEM_PROMPT_WITH_THINK.format(max_frames_per_round=args.max_frames_per_round)

    rng = random.Random(42 + int(args.shard_idx))

    start_t = time.time()

    def _process_one(sample: MCVideoSample) -> None:
        nonlocal correct, total_rounds, invalid_outputs, invalid_action_terminated, failed
        nonlocal total_model_calls, total_retries, total_effective_rounds, think_present
        nonlocal server_proc

        cache_dir = Path(args.video_cache_dir) / sample.dataset
        cache_dir.mkdir(parents=True, exist_ok=True)
        video_path = str(cache_dir / sample.video_key)
        failed_marker = video_path + ".failed"

        video_ok = os.path.exists(video_path) and os.path.getsize(video_path) > 0
        if video_ok and os.path.exists(failed_marker):
            try:
                os.remove(failed_marker)
            except Exception:
                pass

        # Negative-cache download failures (e.g., private/unavailable YouTube videos) to avoid repeated yt-dlp calls
        # across many questions from the same video.
        if not video_ok and os.path.exists(failed_marker):
            failed += 1
            _maybe_log_jsonl(
                args.log_jsonl,
                {
                    "ts": time.time(),
                    "dataset": sample.dataset,
                    "split": split,
                    "sample_id": sample.sample_id,
                    "uid": sample.uid,
                    "video_key": sample.video_key,
                    "video_url": sample.video_url,
                    "error": "download_failed_cached",
                },
            )
            return

        if not video_ok:
            try:
                _download_youtube(sample.video_url, video_path, py_bin=sys.executable, timeout_s=args.yt_dlp_timeout_s)
            except Exception as e:
                failed += 1
                try:
                    with open(failed_marker, "w", encoding="utf-8") as f:
                        f.write(f"download_failed: {type(e).__name__}: {str(e)}\n")
                except Exception:
                    pass
                _maybe_log_jsonl(
                    args.log_jsonl,
                    {
                        "ts": time.time(),
                        "dataset": sample.dataset,
                        "split": split,
                        "sample_id": sample.sample_id,
                        "uid": sample.uid,
                        "video_key": sample.video_key,
                        "video_url": sample.video_url,
                        "error": f"download_failed: {type(e).__name__}: {str(e)[:400]}",
                    },
                )
                return

        try:
            total_frames, fps = _extract_video_info(video_path)
            timeline_len = _timeline_len_1fps(total_frames, fps)
        except Exception as e:
            failed += 1
            _maybe_log_jsonl(
                args.log_jsonl,
                {
                    "ts": time.time(),
                    "dataset": sample.dataset,
                    "split": split,
                    "sample_id": sample.sample_id,
                    "uid": sample.uid,
                    "video_key": sample.video_key,
                    "video_path": video_path,
                    "error": f"video_probe_failed: {type(e).__name__}: {str(e)[:400]}",
                },
            )
            return

        if timeline_len <= 0:
            failed += 1
            return

        # LVBench provides `time_reference` that localizes where the evidence is in the video.
        # We bias/scope frame sampling to this window to avoid spending rounds on irrelevant frames.
        time_range = None
        if sample.dataset == "lvbench" and sample.time_reference:
            time_range = _parse_time_reference_range(sample.time_reference, timeline_len)
        if time_range is None:
            range_start, range_end = 0, timeline_len - 1
        else:
            range_start, range_end = time_range

        question_block = _format_question_block(sample.question, sample.options)
        summary_state = (
            "P: the agent has not seen any frames yet; "
            "O: no reliable observation yet; "
            "H: my belief will be updated based on what is observed; "
            "U: key detail is still unclear; "
            "R: need evidence from frames"
        )
        seen_frames: list[int] = []
        answer_letter: Optional[str] = None
        effective_rounds = 0
        terminated_invalid = False

        init_frames = _sample_uniform_indices_inclusive(range_start, range_end, args.max_frames_per_round)
        next_frames = [int(i) for i in init_frames if i >= 0]

        for round_idx in range(1, args.max_rounds + 1):
            frames_this_round = [i for i in next_frames if i not in seen_frames]
            if not frames_this_round:
                frames_this_round = _sample_uniform_indices_inclusive(range_start, range_end, 1)
            frames_this_round = frames_this_round[: args.max_frames_per_round]
            for i in frames_this_round:
                if i not in seen_frames:
                    seen_frames.append(i)

            # Propose candidate frames within the active window.
            local_len = max(0, range_end - range_start + 1)
            if local_len <= 0:
                candidate_next_frames = []
            else:
                seen_local = {int(i - range_start) for i in seen_frames if range_start <= i <= range_end}
                cand_local = _propose_candidate_frames(
                    frame_count=local_len,
                    seen=seen_local,
                    k=int(args.candidate_k),
                    rng=rng,
                )
                candidate_next_frames = [int(i + range_start) for i in cand_local]

            images = _extract_frames_1fps(video_path, frames_this_round)
            user_text = _build_user_text(
                question_block=question_block,
                summary=summary_state,
                timeline_len=timeline_len,
                round_idx=round_idx,
                current_frames=frames_this_round,
                seen_frames=seen_frames,
                candidate_unseen_frames=candidate_next_frames,
                use_candidate_frame_ids=bool(args.use_candidate_frame_ids),
                require_candidate_frames=bool(args.require_candidate_frames),
                time_reference=sample.time_reference,
            )
            if args.force_final_answer and round_idx >= args.max_rounds:
                user_text = (
                    f"{user_text}\n\n"
                    "This is the final round. You MUST answer now using <think>...</think> then "
                    "<summary>...</summary> then <answer>LETTER</answer>."
                )

            retry_feedback: Optional[str] = None
            raw_output = ""
            for retry_idx in range(args.max_retries_per_round + 1):
                try:
                    raw_output = _chat_once(
                        base_url=base_url,
                        model_id=model_id,
                        system_prompt=system_prompt,
                        user_text=user_text if retry_feedback is None else _retry_feedback_text(
                            retry_feedback, force_answer=bool(args.force_final_answer and round_idx >= args.max_rounds)
                        ),
                        images=images,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens,
                        timeout_s=args.timeout_s,
                    )
                    total_model_calls += 1
                except Exception as e:
                    err_txt = f"{type(e).__name__}: {str(e)[:400]}"
                    if args.restart_server_on_failure and args.start_server:
                        if server_proc is not None:
                            _stop_server(server_proc)
                        server_proc = _start_vllm_server(args)
                        _wait_for_server(args.host, args.port, timeout_s=240)
                        retry_feedback = f"Server error; please retry. ({err_txt})"
                        total_retries += 1
                        if retry_idx < args.max_retries_per_round:
                            continue
                    failed += 1
                    _maybe_log_jsonl(
                        args.log_jsonl,
                        {
                            "ts": time.time(),
                            "dataset": sample.dataset,
                            "split": split,
                            "sample_id": sample.sample_id,
                            "uid": sample.uid,
                            "video_key": sample.video_key,
                            "video_url": sample.video_url,
                            "video_path": video_path,
                            "round_idx": round_idx,
                            "error": f"server_error: {err_txt}",
                        },
                    )
                    return

                think = _extract_tag(raw_output, _THINK_RE)
                if think is not None:
                    think_present += 1

                summary = _extract_tag(raw_output, _SUMMARY_RE)
                frames_text = _extract_tag(raw_output, _FRAMES_RE)
                answer = _extract_tag(raw_output, _ANSWER_RE)

                if summary is None:
                    retry_feedback = "Invalid response: missing <summary> tag."
                    invalid_outputs += 1
                    total_retries += 1
                    continue

                if answer is not None:
                    letter = _normalize_answer_letter(answer, len(sample.options))
                    if letter is None:
                        retry_feedback = "Invalid response: <answer> must be a single option letter."
                        invalid_outputs += 1
                        total_retries += 1
                        continue
                    answer_letter = letter
                    summary_state = summary
                    break

                if frames_text is None:
                    retry_feedback = "Invalid response: missing <frames> tag for requesting more frames."
                    invalid_outputs += 1
                    total_retries += 1
                    continue

                requested = _dedupe_preserve_order(_parse_int_list(frames_text))
                if not requested:
                    retry_feedback = "Invalid response: <frames> must contain at least one integer."
                    invalid_outputs += 1
                    total_retries += 1
                    continue

                if len(requested) > args.max_frames_per_round:
                    requested = requested[: args.max_frames_per_round]
                    invalid_outputs += 1

                mapped = requested
                if args.use_candidate_frame_ids and candidate_next_frames:
                    mapped2: list[int] = []
                    allowed = set(int(x) for x in candidate_next_frames)
                    for cid in requested:
                        if 1 <= cid <= len(candidate_next_frames):
                            mapped2.append(int(candidate_next_frames[cid - 1]))
                        elif cid in allowed:
                            # Allow direct timeline indices if they correspond to candidate frames.
                            mapped2.append(int(cid))
                    mapped = mapped2
                if args.require_candidate_frames and candidate_next_frames:
                    allowed = set(int(x) for x in candidate_next_frames)
                    if any(x not in allowed for x in mapped):
                        retry_feedback = "Invalid response: requested frames must be within candidate IDs."
                        invalid_outputs += 1
                        total_retries += 1
                        continue

                # Must be unseen.
                if any(x in seen_frames for x in mapped):
                    retry_feedback = "Invalid response: requested frames must be NEW (unseen)."
                    invalid_outputs += 1
                    total_retries += 1
                    continue

                if mapped:
                    next_frames = mapped
                    effective_rounds += 1
                else:
                    next_frames = candidate_next_frames[: args.max_frames_per_round]
                    invalid_outputs += 1

                summary_state = summary
                break

            _maybe_log_jsonl(
                args.log_jsonl,
                {
                    "ts": time.time(),
                    "dataset": sample.dataset,
                    "split": split,
                    "sample_id": sample.sample_id,
                    "uid": sample.uid,
                    "video_key": sample.video_key,
                    "video_url": sample.video_url,
                    "video_path": video_path,
                    "timeline_len": timeline_len,
                    "round_idx": round_idx,
                    "retry_feedback": retry_feedback,
                    "question": sample.question,
                    "options": sample.options,
                    "answer_gt": sample.answer_letter,
                    "seen_frames": seen_frames,
                    "current_frames": frames_this_round,
                    "candidate_unseen_frames": candidate_next_frames,
                    "summary_in": summary_state,
                    "raw_output": raw_output,
                    "answer_letter": answer_letter,
                },
            )

            if answer_letter is not None:
                total_rounds += round_idx
                total_effective_rounds += effective_rounds
                gt = _normalize_answer_letter(sample.answer_letter, len(sample.options))
                if gt is not None and answer_letter == gt:
                    correct += 1
                return

        # If we still have no answer, force one extra call (answer-only).
        if answer_letter is None and args.force_final_answer:
            images = _extract_frames_1fps(video_path, frames_this_round)
            user_text2 = (
                f"{question_block}\n"
                "You MUST answer now. Output <think>...</think> then <summary>...</summary> then <answer>LETTER</answer>."
            )
            try:
                raw = _chat_once(
                    base_url=base_url,
                    model_id=model_id,
                    system_prompt=system_prompt,
                    user_text=user_text2,
                    images=images,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    timeout_s=args.timeout_s,
                )
            except Exception as e:
                failed += 1
                _maybe_log_jsonl(
                    args.log_jsonl,
                    {
                        "ts": time.time(),
                        "dataset": sample.dataset,
                        "split": split,
                        "sample_id": sample.sample_id,
                        "uid": sample.uid,
                        "video_key": sample.video_key,
                        "video_url": sample.video_url,
                        "video_path": video_path,
                        "round_idx": args.max_rounds,
                        "error": f"server_error_final_answer: {type(e).__name__}: {str(e)[:400]}",
                    },
                )
                return
            total_model_calls += 1
            ans = _extract_tag(raw, _ANSWER_RE)
            summary = _extract_tag(raw, _SUMMARY_RE)
            letter = _normalize_answer_letter(ans or "", len(sample.options))
            if letter is not None:
                answer_letter = letter
                total_rounds += args.max_rounds
                total_effective_rounds += effective_rounds
                gt = _normalize_answer_letter(sample.answer_letter, len(sample.options))
                if gt is not None and answer_letter == gt:
                    correct += 1
            else:
                terminated_invalid = True

        if terminated_invalid:
            invalid_action_terminated += 1

    processed = 0
    for sample in samples:
        _process_one(sample)
        processed += 1
        if processed % 20 == 0:
            acc = correct / max(1, processed)
            avg_rounds = total_rounds / max(1, processed)
            print(
                f"[{processed}/{len(samples)}] acc={acc:.4f} avg_rounds={avg_rounds:.3f} "
                f"failed={failed} invalid_term={invalid_action_terminated} calls={total_model_calls} "
                f"elapsed_s={time.time()-start_t:.1f}",
                flush=True,
            )
            _wandb_log(
                run,
                {
                    "eval/acc": acc,
                    "eval/avg_rounds": avg_rounds,
                    "eval/failed": failed,
                    "eval/invalid_action_terminated": invalid_action_terminated,
                    "eval/invalid_outputs": invalid_outputs,
                    "eval/total_calls": total_model_calls,
                    "eval/think_present": think_present,
                },
                step=processed,
            )

    elapsed = time.time() - start_t
    acc = correct / max(1, processed)
    avg_rounds = total_rounds / max(1, processed)
    avg_effective_rounds = total_effective_rounds / max(1, processed)
    prompt_log_lines = 0
    prompt_log_bytes = 0
    if args.log_jsonl and os.path.exists(args.log_jsonl):
        prompt_log_bytes = os.path.getsize(args.log_jsonl)
        with open(args.log_jsonl, "r", encoding="utf-8") as f:
            prompt_log_lines = sum(1 for _ in f)

    results = {
        "samples": processed,
        "correct": correct,
        "accuracy": acc,
        "avg_rounds": avg_rounds,
        "avg_effective_rounds": avg_effective_rounds,
        "failed": failed,
        "elapsed_s": elapsed,
        "prompt_log_lines": prompt_log_lines,
        "prompt_log_bytes": prompt_log_bytes,
        "invalid_outputs": invalid_outputs,
        "invalid_action_terminated": invalid_action_terminated,
        "total_retries": total_retries,
        "total_model_calls": total_model_calls,
        "think_present_rounds": think_present,
    }
    print(json.dumps(results, indent=2), flush=True)

    wandb_info: Optional[dict[str, Any]] = None
    if run is not None:
        run.summary["final_acc"] = acc
        run.summary["final_avg_rounds"] = avg_rounds
        run.summary["final_avg_effective_rounds"] = avg_effective_rounds
        run.summary["failed"] = failed
        run.summary["invalid_outputs"] = invalid_outputs
        run.summary["invalid_action_terminated"] = invalid_action_terminated
        run.summary["prompt_log_jsonl"] = args.log_jsonl
        run.summary["prompt_log_lines"] = prompt_log_lines
        run.summary["prompt_log_bytes"] = prompt_log_bytes
        run.summary["think_present_rounds"] = think_present
        run.finish()
        wandb_info = {
            "enabled": True,
            "mode": getattr(args, "wandb_mode", "") or os.getenv("WANDB_MODE"),
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "name": args.wandb_name,
            "group": args.wandb_group,
            "id": getattr(run, "id", None),
            "url": getattr(run, "url", None),
            "run_dir": getattr(run, "dir", None),
        }

    if args.summary_json:
        out_dir = os.path.dirname(args.summary_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    **run_config,
                    "results": results,
                    "prompt_log_jsonl": args.log_jsonl,
                    "wandb": wandb_info,
                    "command": " ".join(["python", *sys.argv]),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    if server_proc is not None and args.start_server:
        _stop_server(server_proc)


if __name__ == "__main__":
    main()
