#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from datasets import load_dataset
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL | re.IGNORECASE)
_FRAMES_RE = re.compile(r"<frames>(.*?)</frames>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

_SYSTEM_PROMPT = (
    "You are a video question-answering agent. You may request more frames or answer now.\n"
    "Output <think> then <summary> plus either <frames> (request) OR <answer> (final).\n\n"
    "Example request:\n"
    "<think>...</think>\n"
    "<summary>P: ...; O: ...; H: ...; U: ...; R: request additional frames</summary>\n"
    "<frames>1, 3</frames>\n\n"
    "Example answer:\n"
    "<think>...</think>\n"
    "<summary>P: ...; O: ...; H: ...; U: ...; R: answered</summary>\n"
    "<answer>B</answer>\n\n"
    "Rules:\n"
    "- <summary>: the ONLY persistent memory across rounds. Keep it short and update it EVERY round.\n"
    "- In <frames>, output comma-separated integers only (no brackets, no text).\n"
    "- In <answer>, output EXACTLY ONE option letter shown in the question (e.g., A/B/C/D/E). No words.\n"
    "- When Candidate Frame IDs are provided, output those IDs (1..K) in <frames> instead of raw frame indices.\n"
)


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _extract_tag(text: str, pattern: re.Pattern[str]) -> Optional[str]:
    matches = list(pattern.finditer(text or ""))
    if not matches:
        return None
    return (matches[-1].group(1) or "").strip()


def _dedupe_preserve_order(items: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _parse_int_list(text: str) -> list[int]:
    out: list[int] = []
    for tok in re.split(r"[,\s]+", str(text or "").strip()):
        if not tok:
            continue
        try:
            out.append(int(tok))
        except Exception:
            continue
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


def _stable_sample_id(dataset: str, video_key: str, uid: str) -> str:
    payload = {"dataset": str(dataset), "video": str(video_key), "uid": str(uid)}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _linspace(a: float, b: float, n: int) -> list[float]:
    if n <= 1:
        return [a]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]


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


def _extract_video_info(video_path: str) -> tuple[int, float]:
    import decord

    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    total_frames = int(len(vr))
    fps = float(vr.get_avg_fps() or 0.0)
    if fps <= 0:
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
        out: list[Image.Image] = []
        for idx in frame_indices:
            out.append(Image.fromarray(vr[idx].asnumpy()))
        return out


def _prep_image(img: Image.Image, *, max_edge: int = 512) -> Image.Image:
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_edge:
        img = img.copy()
        img.thumbnail((max_edge, max_edge), Image.LANCZOS)
    return img


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


def _retry_feedback_text(feedback: str, *, force_answer: bool) -> str:
    if force_answer:
        return (
            f"{feedback}\n"
            "You MUST answer now. Output <think>...</think> then <summary>...</summary> then <answer>LETTER</answer>."
        )
    return f"{feedback}\nPlease respond with one of the required formats."


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

    def _idx_to_letters(idx: int) -> str:
        # Excel-style: 0->A, 25->Z, 26->AA, ...
        if idx < 0:
            return "?"
        base = len(letters)
        n = idx + 1
        out = ""
        while n > 0:
            n -= 1
            n, rem = divmod(n, base)
            out = letters[rem] + out
        return out

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
        lines.append(f"Shown frame {_idx_to_letters(i)} <image>")
    return "\n".join(lines)


def _build_chat_messages(system_prompt: str, user_text: str, images: list[Image.Image]) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    content: list[dict[str, Any]] = []
    parts = user_text.split("<image>") if images else [user_text]
    if images and (len(parts) - 1) == len(images):
        for i, img in enumerate(images):
            if parts[i]:
                content.append({"type": "text", "text": parts[i]})
            content.append({"type": "image"})
        if parts[-1]:
            content.append({"type": "text", "text": parts[-1]})
    else:
        for _ in images:
            content.append({"type": "image"})
        content.append({"type": "text", "text": user_text})

    conv: list[dict[str, Any]] = []
    if system_prompt:
        conv.append({"role": "system", "content": system_prompt})
    conv.append({"role": "user", "content": content})
    # Processor expects images list aligned with "image" placeholders.
    prepped_images = [_prep_image(img) for img in images]
    return conv, prepped_images


def _chat_once_hf(
    model: Any,
    processor: Any,
    system_prompt: str,
    user_text: str,
    images: list[Image.Image],
    *,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_len: int,
) -> str:
    conv, prepped_images = _build_chat_messages(system_prompt, user_text, images)
    chat = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=chat, images=prepped_images, return_tensors="pt")
    input_len = int(inputs["input_ids"].shape[-1])
    if input_len + max_new_tokens > max_len:
        raise RuntimeError(f"prompt_too_long: input_len={input_len} max_len={max_len} max_new={max_new_tokens}")

    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    gen_kwargs: dict[str, Any] = {"max_new_tokens": int(max_new_tokens)}
    if temperature and temperature > 0:
        gen_kwargs.update({"do_sample": True, "temperature": float(temperature), "top_p": float(top_p)})
    else:
        gen_kwargs.update({"do_sample": False})
    with torch.inference_mode():
        generated = model.generate(**inputs, **gen_kwargs)
    gen_ids = generated[:, input_len:]
    return str(processor.batch_decode(gen_ids, skip_special_tokens=True)[0])


def _maybe_log_jsonl(path: Optional[str], obj: dict[str, Any]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


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


@dataclass
class MCVideoSample:
    uid: str
    video_key: str
    question: str
    options: list[str]
    answer_letter: str
    time_reference: str
    question_type: str
    video_type: str

    @property
    def sample_id(self) -> str:
        return _stable_sample_id("lvbench", self.video_key, self.uid)


_OPT_RE = re.compile(r"^\(([A-Z])\)\s*(.*)$")


def _parse_options_from_lvbench_question(question: str) -> tuple[str, list[str]]:
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


def _load_lvbench_samples(split: str) -> list[MCVideoSample]:
    ds = load_dataset("lmms-lab/LVBench", split=split)
    samples: list[MCVideoSample] = []
    for ex in ds:
        video_path = str(ex.get("video_path") or "").strip()
        uid = str(ex.get("uid") or ex.get("key") or "").strip() or video_path
        q_raw = str(ex.get("question") or "").strip()
        q_text, options = _parse_options_from_lvbench_question(q_raw)
        answer = str(ex.get("answer") or "").strip().upper()
        time_reference = str(ex.get("time_reference") or "").strip()
        q_type = str(ex.get("question_type") or "").strip()
        v_type = str(ex.get("type") or "").strip()
        if not video_path or not answer:
            continue
        samples.append(
            MCVideoSample(
                uid=uid,
                video_key=video_path,
                question=q_text if q_text else q_raw,
                options=options,
                answer_letter=answer,
                time_reference=time_reference,
                question_type=q_type,
                video_type=v_type,
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


def _load_model_and_processor(model_path: str, dtype: str, device: torch.device) -> tuple[Any, Any]:
    torch_dtype = torch.bfloat16
    if dtype == "float16":
        torch_dtype = torch.float16
    if dtype == "float32":
        torch_dtype = torch.float32

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    model.to(device)
    return model, processor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--video-cache-dir", default="/tmp/chenwei_video_cache")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--start-idx", type=int, default=0)
    ap.add_argument("--end-idx", type=int, default=0)

    ap.add_argument("--model-path", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--system-prompt", default=_SYSTEM_PROMPT)

    ap.add_argument("--max-rounds", type=int, default=5)
    ap.add_argument("--max-frames-per-round", type=int, default=5)
    ap.add_argument("--candidate-k", type=int, default=20)
    ap.add_argument("--use-candidate-frame-ids", action="store_true", default=True)
    ap.add_argument("--require-candidate-frames", action="store_true", default=True)
    ap.add_argument("--max-retries-per-round", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--force-final-answer", action="store_true", default=True)

    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-idx", type=int, default=0)

    ap.add_argument("--log-jsonl", default="")
    ap.add_argument("--summary-json", default="")
    ap.add_argument("--resume-from-log", action="store_true")

    ap.add_argument("--use-wandb", action="store_true")
    ap.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "revise_benchmarks"))
    ap.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY"))
    ap.add_argument("--wandb-name", default=os.getenv("WANDB_RUN_NAME"))
    ap.add_argument("--wandb-group", default=os.getenv("WANDB_RUN_GROUP"))
    ap.add_argument("--wandb-tags", default=os.getenv("WANDB_TAGS", ""))
    ap.add_argument("--wandb-mode", default=os.getenv("WANDB_MODE", ""))

    args = ap.parse_args()

    samples = _load_lvbench_samples(args.split)
    start_idx = max(0, int(args.start_idx or 0))
    end_idx = int(args.end_idx or 0)
    if end_idx <= 0:
        end_idx = len(samples)
    samples = samples[start_idx:end_idx]
    if args.max_samples and args.max_samples > 0:
        samples = samples[: args.max_samples]
    samples = _shard_by_video(samples, args.num_shards, args.shard_idx)
    samples.sort(key=lambda s: (s.video_key, s.uid))
    if not samples:
        raise SystemExit("No samples selected (check --split/--start-idx/--max-samples/--sharding).")

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
    if args.resume_from_log and args.log_jsonl and os.path.exists(args.log_jsonl):
        seen_samples: set[str] = set()
        with open(args.log_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                sid = obj.get("sample_id")
                if sid and obj.get("answer_letter"):
                    seen_samples.add(str(sid))
        resume_completed = len(seen_samples)
        if resume_completed > 0:
            print(f"[resume] detected {resume_completed} completed samples in {args.log_jsonl}")
    if resume_completed > 0:
        samples = samples[resume_completed:]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, processor = _load_model_and_processor(args.model_path, args.dtype, device)
    max_len = int(getattr(getattr(model.config, "text_config", model.config), "max_position_embeddings", 32768))

    run_config = {
        "task": "revise_plug_and_play_lvbench_hf",
        "dataset": "lvbench",
        "split": args.split,
        "model_path": args.model_path,
        "max_rounds": args.max_rounds,
        "max_frames_per_round": args.max_frames_per_round,
        "candidate_k": args.candidate_k,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
    }
    run = _maybe_init_wandb(args, run_config)

    rng = random.Random(1337 + int(args.shard_idx))

    correct = 0
    failed = 0
    invalid_outputs = 0
    invalid_action_terminated = 0
    total_rounds = 0
    total_effective_rounds = 0
    total_frames_used = 0
    total_model_calls = 0
    total_retries = 0
    think_present = 0

    cache_dir = Path(args.video_cache_dir) / "lvbench"
    cache_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for sample in samples:
        processed += 1
        processed_global = processed + resume_completed

        video_path = str(cache_dir / sample.video_key)
        if not os.path.exists(video_path) or os.path.getsize(video_path) <= 0:
            failed += 1
            _maybe_log_jsonl(
                args.log_jsonl,
                {
                    "ts": time.time(),
                    "dataset": "lvbench",
                    "split": args.split,
                    "sample_id": sample.sample_id,
                    "uid": sample.uid,
                    "video_key": sample.video_key,
                    "video_path": video_path,
                    "error": "missing_video",
                },
            )
            continue

        try:
            total_frames, fps = _extract_video_info(video_path)
            timeline_len = _timeline_len_1fps(total_frames, fps)
        except Exception as e:
            failed += 1
            _maybe_log_jsonl(
                args.log_jsonl,
                {
                    "ts": time.time(),
                    "dataset": "lvbench",
                    "split": args.split,
                    "sample_id": sample.sample_id,
                    "uid": sample.uid,
                    "video_key": sample.video_key,
                    "video_path": video_path,
                    "error": f"video_probe_failed: {type(e).__name__}: {str(e)[:400]}",
                },
            )
            continue

        if timeline_len <= 0:
            failed += 1
            continue

        time_range = None
        if sample.time_reference:
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

        init_frames = _sample_uniform_indices_inclusive(range_start, range_end, int(args.max_frames_per_round))
        next_frames = [int(i) for i in init_frames if i >= 0]

        for round_idx in range(1, int(args.max_rounds) + 1):
            frames_this_round = [i for i in next_frames if i not in seen_frames]
            if not frames_this_round:
                frames_this_round = _sample_uniform_indices_inclusive(range_start, range_end, 1)
            frames_this_round = frames_this_round[: int(args.max_frames_per_round)]
            for i in frames_this_round:
                if i not in seen_frames:
                    seen_frames.append(i)

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
            if args.force_final_answer and round_idx >= int(args.max_rounds):
                user_text = (
                    f"{user_text}\n\n"
                    "This is the final round. You MUST answer now using <think>...</think> then "
                    "<summary>...</summary> then <answer>LETTER</answer>."
                )

            retry_feedback: Optional[str] = None
            raw_output = ""
            for retry_idx in range(int(args.max_retries_per_round) + 1):
                try:
                    raw_output = _chat_once_hf(
                        model=model,
                        processor=processor,
                        system_prompt=str(args.system_prompt or ""),
                        user_text=user_text
                        if retry_feedback is None
                        else _retry_feedback_text(
                            retry_feedback,
                            force_answer=bool(args.force_final_answer and round_idx >= int(args.max_rounds)),
                        ),
                        images=images,
                        device=device,
                        max_new_tokens=int(args.max_new_tokens),
                        temperature=float(args.temperature),
                        top_p=float(args.top_p),
                        max_len=max_len,
                    )
                    total_model_calls += 1
                except Exception as e:
                    err_txt = f"{type(e).__name__}: {str(e)[:400]}"
                    retry_feedback = f"Model error; please retry. ({err_txt})"
                    total_retries += 1
                    if retry_idx < int(args.max_retries_per_round):
                        continue
                    failed += 1
                    _maybe_log_jsonl(
                        args.log_jsonl,
                        {
                            "ts": time.time(),
                            "dataset": "lvbench",
                            "split": args.split,
                            "sample_id": sample.sample_id,
                            "uid": sample.uid,
                            "video_key": sample.video_key,
                            "video_path": video_path,
                            "round_idx": round_idx,
                            "error": f"model_error: {err_txt}",
                        },
                    )
                    break

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

                if len(requested) > int(args.max_frames_per_round):
                    requested = requested[: int(args.max_frames_per_round)]
                    invalid_outputs += 1

                mapped = requested
                if args.use_candidate_frame_ids and candidate_next_frames:
                    mapped2: list[int] = []
                    allowed = set(int(x) for x in candidate_next_frames)
                    for cid in requested:
                        if 1 <= cid <= len(candidate_next_frames):
                            mapped2.append(int(candidate_next_frames[cid - 1]))
                        elif cid in allowed:
                            mapped2.append(int(cid))
                    mapped = mapped2
                if args.require_candidate_frames and candidate_next_frames:
                    allowed = set(int(x) for x in candidate_next_frames)
                    if any(x not in allowed for x in mapped):
                        retry_feedback = "Invalid response: requested frames must be within candidate IDs."
                        invalid_outputs += 1
                        total_retries += 1
                        continue

                if any(x in seen_frames for x in mapped):
                    retry_feedback = "Invalid response: requested frames must be NEW (unseen)."
                    invalid_outputs += 1
                    total_retries += 1
                    continue

                if mapped:
                    next_frames = mapped
                    effective_rounds += 1
                else:
                    next_frames = candidate_next_frames[: int(args.max_frames_per_round)]
                    invalid_outputs += 1

                summary_state = summary
                break

            _maybe_log_jsonl(
                args.log_jsonl,
                {
                    "ts": time.time(),
                    "dataset": "lvbench",
                    "split": args.split,
                    "sample_id": sample.sample_id,
                    "uid": sample.uid,
                    "video_key": sample.video_key,
                    "video_path": video_path,
                    "timeline_len": timeline_len,
                    "round_idx": round_idx,
                    "retry_feedback": retry_feedback,
                    "question": sample.question,
                    "options": sample.options,
                    "answer_gt": sample.answer_letter,
                    "time_reference": sample.time_reference,
                    "question_type": sample.question_type,
                    "video_type": sample.video_type,
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
                total_frames_used += len(frames_this_round)
                gt = _normalize_answer_letter(sample.answer_letter, len(sample.options))
                if gt is not None and answer_letter == gt:
                    correct += 1
                break

        if answer_letter is None and args.force_final_answer:
            # If we still have no answer, force one extra call (answer-only).
            images = _extract_frames_1fps(video_path, frames_this_round)
            user_text2 = (
                f"{question_block}\n"
                "You MUST answer now. Output <think>...</think> then <summary>...</summary> then <answer>LETTER</answer>."
            )
            try:
                raw = _chat_once_hf(
                    model=model,
                    processor=processor,
                    system_prompt=str(args.system_prompt or ""),
                    user_text=user_text2,
                    images=images,
                    device=device,
                    max_new_tokens=int(args.max_new_tokens),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    max_len=max_len,
                )
            except Exception as e:
                failed += 1
                _maybe_log_jsonl(
                    args.log_jsonl,
                    {
                        "ts": time.time(),
                        "dataset": "lvbench",
                        "split": args.split,
                        "sample_id": sample.sample_id,
                        "uid": sample.uid,
                        "video_key": sample.video_key,
                        "video_path": video_path,
                        "round_idx": int(args.max_rounds),
                        "error": f"model_error_final_answer: {type(e).__name__}: {str(e)[:400]}",
                    },
                )
                continue
            total_model_calls += 1
            ans = _extract_tag(raw, _ANSWER_RE)
            summary = _extract_tag(raw, _SUMMARY_RE)
            letter = _normalize_answer_letter(ans or "", len(sample.options))
            if letter is not None:
                answer_letter = letter
                total_rounds += int(args.max_rounds)
                total_effective_rounds += effective_rounds
                total_frames_used += len(frames_this_round)
                gt = _normalize_answer_letter(sample.answer_letter, len(sample.options))
                if gt is not None and answer_letter == gt:
                    correct += 1
            else:
                terminated_invalid = True

        if terminated_invalid:
            invalid_action_terminated += 1

        if run is not None:
            _wandb_log(
                run,
                {
                    "progress/processed": processed_global,
                    "metrics/acc_so_far": correct / max(1, processed_global - failed),
                    "metrics/failed": failed,
                    "metrics/invalid_outputs": invalid_outputs,
                    "metrics/avg_rounds": total_rounds / max(1, processed_global - failed),
                    "metrics/avg_effective_rounds": total_effective_rounds / max(1, processed_global - failed),
                    "metrics/avg_frames_used": total_frames_used / max(1, processed_global - failed),
                },
                step=processed_global,
            )

        if processed_global % 25 == 0 or processed_global == len(samples):
            acc_so_far = correct / max(1, processed_global - failed)
            print(
                f"[lvbench] processed {processed_global} / {len(samples)+resume_completed} | "
                f"acc={acc_so_far:.4f} failed={failed} invalid={invalid_outputs} calls={total_model_calls}"
            )

    answered = max(0, len(samples) + resume_completed - failed)
    summary = {
        "dataset": "lvbench",
        "split": args.split,
        "model_path": args.model_path,
        "num_samples": len(samples) + resume_completed,
        "answered": answered,
        "correct": correct,
        "accuracy": correct / max(1, answered),
        "failed": failed,
        "invalid_outputs": invalid_outputs,
        "invalid_action_terminated": invalid_action_terminated,
        "avg_rounds": total_rounds / max(1, answered),
        "avg_effective_rounds": total_effective_rounds / max(1, answered),
        "avg_frames_used": total_frames_used / max(1, answered),
        "think_present": think_present,
        "total_model_calls": total_model_calls,
        "total_retries": total_retries,
        "max_len": max_len,
        "config": run_config,
        "ts": time.time(),
    }

    if args.summary_json:
        os.makedirs(os.path.dirname(args.summary_json), exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, indent=2))
    if run is not None:
        run.summary.update(summary)
        run.finish()


if __name__ == "__main__":
    main()
