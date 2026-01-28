#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from datasets import load_dataset
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None


_ANSWER_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


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
    if start_s is None or end_s is None or timeline_len <= 0:
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


def _sample_uniform_indices_inclusive(start: int, end: int, k: int) -> list[int]:
    if k <= 0 or end < start:
        return []
    if start == end:
        return [start]
    if k == 1:
        return [int(round((start + end) / 2))]
    out = [int(round(start + (end - start) * j / (k - 1))) for j in range(k)]
    # Dedupe while preserving order.
    seen: set[int] = set()
    dedup: list[int] = []
    for i in out:
        i = max(start, min(i, end))
        if i in seen:
            continue
        seen.add(i)
        dedup.append(i)
    return dedup


def _format_question_block(question: str, options: list[str]) -> str:
    q = str(question).strip()
    lines = ["Question: " + q, "Options:"]
    for i, opt in enumerate(options):
        prefix = _ANSWER_LETTERS[i] if i < len(_ANSWER_LETTERS) else str(i)
        lines.append(f"{prefix}. {str(opt).strip()}")
    return "\n".join(lines)


def _normalize_answer_letter(ans: str, num_choices: int) -> Optional[str]:
    if not ans:
        return None
    a = _collapse_ws(ans).strip().upper()
    if len(a) == 1 and a in _ANSWER_LETTERS[: max(1, num_choices)]:
        return a
    m = re.search(r"\b([A-Z])\b", a)
    if m:
        cand = m.group(1)
        if cand in _ANSWER_LETTERS[: max(1, num_choices)]:
            return cand
    return None


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
    fps = float(vr.get_avg_fps() or 0.0) or 30.0
    frame_indices = [_timeline_to_frame_idx(i, fps, total_frames) for i in timeline_indices]
    try:
        frames = vr.get_batch(frame_indices).asnumpy()
        return [Image.fromarray(frame) for frame in frames]
    except Exception:
        out: list[Image.Image] = []
        for idx in frame_indices:
            out.append(Image.fromarray(vr[idx].asnumpy()))
        return out


def _ensure_yt_dlp(py_bin: str) -> list[str]:
    if shutil.which("yt-dlp"):
        return ["yt-dlp"]
    return [py_bin, "-m", "yt_dlp"]


def _download_youtube(url: str, out_mp4: str, *, py_bin: str, timeout_s: int) -> None:
    out_mp4_path = Path(out_mp4)
    out_mp4_path.parent.mkdir(parents=True, exist_ok=True)
    out_tmpl = str(out_mp4_path.with_suffix("")) + ".%(ext)s"

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

    if out_mp4_path.exists() and out_mp4_path.stat().st_size > 0:
        return
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


def _maybe_init_wandb(args: argparse.Namespace, run_config: dict[str, Any]) -> Any:
    if not getattr(args, "use_wandb", False) or wandb is None:
        return None

    def _has_wandb_credentials() -> bool:
        if os.getenv("WANDB_API_KEY") or os.getenv("WANDB_IDENTITY_TOKEN_FILE"):
            return True
        return False

    mode = getattr(args, "wandb_mode", "") or os.getenv("WANDB_MODE")
    if not mode:
        mode = "online" if _has_wandb_credentials() else "offline"

    tags = [t.strip() for t in (getattr(args, "wandb_tags", "") or "").split(",") if t.strip()]
    run = wandb.init(
        project=getattr(args, "wandb_project", None),
        entity=getattr(args, "wandb_entity", None),
        name=getattr(args, "wandb_name", None),
        group=getattr(args, "wandb_group", None),
        tags=tags or None,
        mode=mode,
        config=run_config,
        reinit=True,
    )
    return run


def _wandb_log(run: Any, metrics: dict[str, Any], step: int) -> None:
    if run is None or wandb is None:
        return
    wandb.log(metrics, step=step)


def _maybe_log_jsonl(path: Optional[str], obj: dict[str, Any]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _load_model_and_processor(model_path: str, dtype: str, device: torch.device) -> tuple[Any, Any]:
    torch_dtype: Any
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "float32":
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.bfloat16

    processor = AutoProcessor.from_pretrained(model_path)
    model = None
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
        )
    except Exception:
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
    model.eval()
    model.to(device)
    return model, processor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--video-cache-dir", default="/tmp/chenwei_video_cache")
    ap.add_argument("--start-idx", type=int, default=0)
    ap.add_argument("--end-idx", type=int, default=0)
    ap.add_argument("--max-samples", type=int, default=0)

    ap.add_argument("--model-path", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--max-frames", type=int, default=15)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--yt-dlp-timeout-s", type=int, default=600)

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
                if sid and obj.get("pred_answer"):
                    seen_samples.add(sid)
        resume_completed = len(seen_samples)
        if resume_completed > 0:
            print(f"[resume] detected {resume_completed} completed samples in {args.log_jsonl}")
    if resume_completed > 0:
        samples = samples[resume_completed:]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, processor = _load_model_and_processor(args.model_path, args.dtype, device)
    max_len = int(getattr(getattr(model.config, "text_config", model.config), "max_position_embeddings", 32768))

    run_config = {
        "task": "lvbench_oneshot_hf",
        "dataset": "lvbench",
        "split": args.split,
        "model_path": args.model_path,
        "video_cache_dir": args.video_cache_dir,
        "dtype": args.dtype,
        "max_frames": args.max_frames,
        "max_new_tokens": args.max_new_tokens,
        "max_len": max_len,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "max_samples": args.max_samples,
        "num_shards": args.num_shards,
        "shard_idx": args.shard_idx,
    }
    run = _maybe_init_wandb(args, run_config)

    rng = random.Random(42 + int(args.shard_idx))
    start_t = time.time()

    processed = 0
    correct = 0
    failed = 0
    invalid_outputs = 0
    invalid_action_terminated = 0
    total_model_calls = 0
    total_retries = 0
    total_frames_used = 0

    for sample in samples:
        processed += 1

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

        if not video_ok and os.path.exists(failed_marker):
            failed += 1
            _maybe_log_jsonl(
                args.log_jsonl,
                {
                    "ts": time.time(),
                    "dataset": sample.dataset,
                    "split": args.split,
                    "sample_id": sample.sample_id,
                    "uid": sample.uid,
                    "video_key": sample.video_key,
                    "video_url": sample.video_url,
                    "error": "download_failed_cached",
                },
            )
            continue

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
                        "split": args.split,
                        "sample_id": sample.sample_id,
                        "uid": sample.uid,
                        "video_key": sample.video_key,
                        "video_url": sample.video_url,
                        "error": f"download_failed: {type(e).__name__}: {str(e)[:400]}",
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
                    "dataset": sample.dataset,
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

        frame_indices = _sample_uniform_indices_inclusive(range_start, range_end, int(args.max_frames))
        frames = _extract_frames_1fps(video_path, frame_indices)
        if not frames:
            failed += 1
            continue

        # Build structured conversation required by LlavaOnevisionProcessor.
        q_block = _format_question_block(sample.question, sample.options)
        prompt_text = (
            f"{q_block}\n\n"
            "You will be given video frames sampled at 1 fps.\n"
            "Answer with EXACTLY ONE option letter (e.g., A/B/C/D). Do not output any other text."
        )
        content: list[dict[str, Any]] = [{"type": "image"} for _ in frames]
        content.append({"type": "text", "text": prompt_text})
        conv = [{"role": "user", "content": content}]
        chat = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)

        # Ensure input length + generation does not exceed model max length.
        # Llava-OneVision expands each image into many tokens; some videos can exceed max_len even with few frames.
        max_new = int(args.max_new_tokens)
        usable_frames = frames
        input_len = 0
        inputs: dict[str, Any] = {}
        while True:
            inputs = processor(text=chat, images=usable_frames, return_tensors="pt")
            input_len = int(inputs["input_ids"].shape[-1])
            if input_len + max_new <= max_len:
                break
            if len(usable_frames) <= 1:
                break
            usable_frames = usable_frames[:-1]
            content = [{"type": "image"} for _ in usable_frames] + [{"type": "text", "text": prompt_text}]
            conv = [{"role": "user", "content": content}]
            chat = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)

        if input_len + max_new > max_len:
            failed += 1
            _maybe_log_jsonl(
                args.log_jsonl,
                {
                    "ts": time.time(),
                    "dataset": sample.dataset,
                    "split": args.split,
                    "sample_id": sample.sample_id,
                    "uid": sample.uid,
                    "video_key": sample.video_key,
                    "video_path": video_path,
                    "timeline_len": timeline_len,
                    "time_reference": sample.time_reference,
                    "sampled_frames": len(frames),
                    "usable_frames": len(usable_frames),
                    "input_len": input_len,
                    "error": f"prompt_too_long: input_len={input_len} max_len={max_len} max_new={max_new}",
                },
            )
            continue

        total_frames_used += len(usable_frames)
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        pred_text = ""
        for retry_idx in range(3):
            try:
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=max_new,
                        do_sample=False,
                    )
                total_model_calls += 1
                gen_ids = generated[:, inputs["input_ids"].shape[-1] :]
                pred_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
                break
            except Exception as e:
                total_retries += 1
                if retry_idx >= 2:
                    failed += 1
                    _maybe_log_jsonl(
                        args.log_jsonl,
                        {
                            "ts": time.time(),
                            "dataset": sample.dataset,
                            "split": args.split,
                            "sample_id": sample.sample_id,
                            "uid": sample.uid,
                            "video_key": sample.video_key,
                            "video_path": video_path,
                            "input_len": input_len,
                            "error": f"infer_failed: {type(e).__name__}: {str(e)[:400]}",
                        },
                    )
                    pred_text = ""
                else:
                    time.sleep(0.2 + 0.2 * rng.random())

        pred = _normalize_answer_letter(pred_text, len(sample.options))
        gt = _normalize_answer_letter(sample.answer_letter, len(sample.options))
        is_correct = bool(pred and gt and pred == gt)
        if pred is None:
            invalid_outputs += 1
            invalid_action_terminated += 1
        if is_correct:
            correct += 1

        _maybe_log_jsonl(
            args.log_jsonl,
            {
                "ts": time.time(),
                "dataset": sample.dataset,
                "split": args.split,
                "sample_id": sample.sample_id,
                "uid": sample.uid,
                "video_key": sample.video_key,
                "video_url": sample.video_url,
                "video_path": video_path,
                "timeline_len": timeline_len,
                "time_reference": sample.time_reference,
                "sampled_frames": len(frames),
                "usable_frames": len(usable_frames),
                "frame_indices": frame_indices[: len(usable_frames)],
                "input_len": input_len,
                "question": sample.question,
                "options": sample.options,
                "answer_gt": gt,
                "pred_answer": pred,
                "pred_text": pred_text[:200],
                "is_correct": is_correct,
            },
        )

        if processed % 20 == 0:
            acc = correct / max(1, processed)
            print(
                f"[{processed}/{len(samples)}] acc={acc:.4f} failed={failed} invalid={invalid_outputs} "
                f"calls={total_model_calls} elapsed_s={time.time()-start_t:.1f}",
                flush=True,
            )
            _wandb_log(
                run,
                {
                    "eval/acc": acc,
                    "eval/failed": failed,
                    "eval/invalid_outputs": invalid_outputs,
                    "eval/total_calls": total_model_calls,
                },
                step=processed,
            )

    elapsed = time.time() - start_t
    acc = correct / max(1, processed)
    total_rounds = max(0, processed - failed)
    avg_rounds = total_rounds / max(1, processed)
    total_effective_rounds = total_rounds
    avg_effective_rounds = avg_rounds
    avg_frames_used = total_frames_used / max(1, processed)
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
        "total_rounds": total_rounds,
        "avg_rounds": avg_rounds,
        "total_effective_rounds": total_effective_rounds,
        "avg_effective_rounds": avg_effective_rounds,
        "total_frames_used": total_frames_used,
        "avg_frames_used": avg_frames_used,
        "failed": failed,
        "elapsed_s": elapsed,
        "prompt_log_lines": prompt_log_lines,
        "prompt_log_bytes": prompt_log_bytes,
        "invalid_outputs": invalid_outputs,
        "invalid_action_terminated": invalid_action_terminated,
        "total_retries": total_retries,
        "total_model_calls": total_model_calls,
    }
    print(json.dumps(results, indent=2), flush=True)

    wandb_info: Optional[dict[str, Any]] = None
    if run is not None:
        run.summary["final_acc"] = acc
        run.summary["failed"] = failed
        run.summary["invalid_outputs"] = invalid_outputs
        run.summary["prompt_log_jsonl"] = args.log_jsonl
        run.summary["prompt_log_lines"] = prompt_log_lines
        run.summary["prompt_log_bytes"] = prompt_log_bytes
        run.finish()
        wandb_info = {
            "enabled": True,
            "mode": getattr(args, "wandb_mode", "") or os.getenv("WANDB_MODE"),
            "id": getattr(run, "id", None),
            "url": getattr(run, "url", None),
        }

    summary = {
        "task": "lvbench_oneshot_hf",
        "dataset": "lvbench",
        "split": args.split,
        "model_path": args.model_path,
        "video_cache_dir": args.video_cache_dir,
        "dtype": args.dtype,
        "max_frames": args.max_frames,
        "max_new_tokens": args.max_new_tokens,
        "num_shards": args.num_shards,
        "shard_idx": args.shard_idx,
        "log_jsonl": args.log_jsonl,
        "summary_json": args.summary_json,
        "prompt_log_jsonl": args.log_jsonl,
        "results": results,
        "wandb": wandb_info,
        "command": " ".join(sys.argv),
    }

    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
