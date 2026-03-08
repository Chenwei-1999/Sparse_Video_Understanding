#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import requests

# Allow direct execution via `python examples/...py`.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:  # pragma: no cover
    TfidfVectorizer = None

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None

from examples.revise.plug_and_play_nextqa_vllm import (  # noqa: E402
    _chat_once,
    _load_nextqa_samples,
    _start_vllm_server,
)
from examples.revise.caption_utils import ensure_video_captions  # noqa: E402
from examples.revise.pnp_utils import (  # noqa: E402
    get_model_id as _get_model_id,
    maybe_log_jsonl as _maybe_log_jsonl,
    resolve_base_url as _resolve_base_url,
    stop_server as _stop_server,
    truncate_text as _truncate_text,
    wait_port as _wait_port,
    wait_for_server as _wait_for_server,
    wandb_log as _wandb_log,
)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


SYSTEM_PROMPT_ANSWER = (
    "You are a multiple-choice video QA assistant.\n"
    "You will be given a question with options and a subset of video captions (1fps timeline; index≈seconds).\n"
    "Return a JSON object with exactly two keys: 'answer' and 'confidence'.\n"
    "- 'answer' must be exactly one of: A, B, C, D, E.\n"
    "- 'confidence' must be an integer 1, 2, or 3.\n"
    "Do not output any other keys.\n"
)


SYSTEM_PROMPT_REQUEST = (
    "You are a video QA agent that decides what additional evidence to look for.\n"
    "You will be given a question, options, captions already observed, and a list of candidate time segments.\n"
    "Return a JSON object with a single key 'requests' whose value is a list.\n"
    "Each element must have keys: 'segment_id' (int) and 'query' (string).\n"
    "Choose up to K requests. Keep queries short.\n"
)


def _parse_json_obj(text: str) -> Optional[dict[str, Any]]:
    if not text:
        return None
    raw = text.strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        m = _JSON_RE.search(raw)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None


def _normalize_answer_letter(letter: Any, num_choices: int) -> Optional[str]:
    if not letter:
        return None
    s = str(letter).strip().upper()
    if not s:
        return None
    s = s[0]
    if s < "A" or s > chr(ord("A") + max(0, num_choices - 1)):
        return None
    return s


def _normalize_confidence(conf: Any) -> Optional[int]:
    try:
        c = int(conf)
    except Exception:
        return None
    if c in (1, 2, 3):
        return c
    return None


def _format_question(question: str, choices: list[str]) -> str:
    labels = [chr(ord("A") + i) for i in range(len(choices))]
    lines = [f"Question: {question}", "Options:"]
    for label, choice in zip(labels, choices, strict=False):
        lines.append(f"{label}. {choice}")
    return "\n".join(lines)


def _as_caption_list(captions: dict[int, str]) -> tuple[list[int], list[str]]:
    if not captions:
        return [], []
    keys = sorted(int(k) for k in captions.keys())
    return keys, [str(captions[k] or "") for k in keys]


def _format_captions(
    indices: list[int],
    captions: dict[int, str],
    *,
    max_chars: int,
) -> str:
    lines: list[str] = []
    for idx in sorted(set(int(i) for i in indices)):
        cap = captions.get(int(idx))
        if not cap:
            continue
        lines.append(f"{idx}s: {_truncate_text(cap, max_chars)}")
    return "\n".join(lines) if lines else "(no captions available)"


@dataclass
class Segment:
    segment_id: int
    start: int
    end: int

    def label(self) -> str:
        return f"{self.start}s-{self.end}s"


def _build_segments(seen: list[int], max_idx: int) -> list[Segment]:
    seen_sorted = sorted(set(int(i) for i in seen if 0 <= int(i) <= max_idx))
    segments: list[Segment] = []
    for a, b in zip(seen_sorted, seen_sorted[1:], strict=False):
        start = a + 1
        end = b - 1
        if start <= end:
            segments.append(Segment(segment_id=len(segments) + 1, start=start, end=end))
    return segments


def _pick_uniform_indices(keys: list[int], k: int) -> list[int]:
    if not keys or k <= 0:
        return []
    if len(keys) <= k:
        return keys
    positions = np.linspace(0, len(keys) - 1, num=k, dtype=int).tolist()
    return sorted({keys[p] for p in positions})


def _build_video_tfidf(caption_texts: list[str]) -> tuple[TfidfVectorizer, Any]:
    if TfidfVectorizer is None:
        raise ImportError("scikit-learn is required for VideoAgent caption retrieval. Install `scikit-learn`.")
    vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
    matrix = vectorizer.fit_transform([t or "" for t in caption_texts])
    return vectorizer, matrix


def _retrieve_best_index(
    *,
    query: str,
    segments: list[Segment],
    segment_id: int,
    caption_keys: list[int],
    caption_texts: list[str],
    vectorizer: TfidfVectorizer,
    matrix: Any,
    seen: set[int],
) -> Optional[int]:
    if not query:
        return None
    if not (1 <= segment_id <= len(segments)):
        return None
    seg = segments[segment_id - 1]
    cand_rows: list[int] = []
    cand_indices: list[int] = []
    for row, idx in enumerate(caption_keys):
        if idx in seen:
            continue
        if seg.start <= idx <= seg.end and (caption_texts[row] or "").strip():
            cand_rows.append(row)
            cand_indices.append(idx)
    if not cand_rows:
        return None
    q_vec = vectorizer.transform([query])
    sims = (matrix[cand_rows] @ q_vec.T).toarray().ravel()
    if sims.size == 0:
        return None
    best = int(cand_indices[int(sims.argmax())])
    return best


def _answer_with_confidence(
    *,
    base_url: str,
    model_id: str,
    question: str,
    choices: list[str],
    observed_caption_block: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout_s: int,
) -> tuple[Optional[str], Optional[int], str, str]:
    user_text = (
        f"{_format_question(question, choices)}\n\n"
        "Observed captions (index≈seconds):\n"
        f"{observed_caption_block}\n\n"
        "Return JSON: {\"answer\":\"A\",\"confidence\":3}\n"
    )
    raw = _chat_once(
        base_url=base_url,
        model_id=model_id,
        system_prompt=SYSTEM_PROMPT_ANSWER,
        user_text=user_text,
        images=[],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        timeout_s=timeout_s,
    )
    obj = _parse_json_obj(raw)
    answer = _normalize_answer_letter((obj or {}).get("answer"), len(choices))
    conf = _normalize_confidence((obj or {}).get("confidence"))
    return answer, conf, user_text, raw


def _request_more(
    *,
    base_url: str,
    model_id: str,
    question: str,
    choices: list[str],
    observed_caption_block: str,
    segments: list[Segment],
    k: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout_s: int,
) -> tuple[list[dict[str, Any]], str, str]:
    seg_lines = [f"{s.segment_id}) {s.label()}" for s in segments]
    segments_block = "\n".join(seg_lines) if seg_lines else "(no segments available)"
    user_text = (
        f"{_format_question(question, choices)}\n\n"
        "Observed captions (index≈seconds):\n"
        f"{observed_caption_block}\n\n"
        "Candidate segments to search for more evidence:\n"
        f"{segments_block}\n\n"
        f"Choose up to {k} requests.\n"
        "Return JSON: {\"requests\":[{\"segment_id\":1,\"query\":\"...\"}]}\n"
    )
    raw = _chat_once(
        base_url=base_url,
        model_id=model_id,
        system_prompt=SYSTEM_PROMPT_REQUEST,
        user_text=user_text,
        images=[],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        timeout_s=timeout_s,
    )
    obj = _parse_json_obj(raw) or {}
    reqs = obj.get("requests", [])
    if not isinstance(reqs, list):
        reqs = []
    parsed: list[dict[str, Any]] = []
    for r in reqs[: max(0, int(k))]:
        if not isinstance(r, dict):
            continue
        seg_id = r.get("segment_id")
        try:
            seg_id_int = int(seg_id)
        except Exception:
            continue
        query = str(r.get("query", "")).strip()
        if not query:
            continue
        parsed.append({"segment_id": seg_id_int, "query": query})
    return parsed, user_text, raw


def _read_completed_sample_ids(log_path: str) -> set[str]:
    if not log_path or not os.path.exists(log_path):
        return set()
    completed: set[str] = set()
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("event") == "final" and obj.get("sample_id"):
                completed.add(str(obj["sample_id"]))
    return completed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="HF model id or local snapshot path")
    ap.add_argument("--captions-dir", default=None, help="Optional directory containing <video_id>_cap.json caption files.")
    ap.add_argument(
        "--generated-captions-dir",
        default=str(REPO_ROOT / "outputs" / "generated_captions" / "nextqa"),
        help="Directory for auto-generated <video_id>_cap.json files.",
    )
    ap.add_argument("--auto-generate-captions", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--caption-gen-max-frames", type=int, default=16)
    ap.add_argument("--caption-gen-stride-s", type=int, default=4)
    ap.add_argument("--caption-gen-timeout-s", type=int, default=600)
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--map-json", required=True)
    ap.add_argument("--csv", required=True, help="NExT-QA CSV (e.g., val.csv)")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--max-rounds", type=int, default=5)
    ap.add_argument("--max-frames-per-round", type=int, default=5)
    ap.add_argument("--caption-max-chars", type=int, default=200)

    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--request-timeout-s", type=int, default=600)
    ap.add_argument("--progress-interval", type=int, default=50)

    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18200)
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL. Defaults to http://host:port.")
    ap.add_argument("--model-id", default=None, help="Explicit remote model ID for chat completions.")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--max-model-len", type=int, default=12288)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.6)

    ap.add_argument("--start-server", action="store_true")
    ap.add_argument("--server-log", default=None)
    ap.add_argument("--server-timeout-s", type=int, default=600)
    ap.add_argument("--restart-server-on-failure", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--log-jsonl", default=None)
    ap.add_argument("--summary-json", default=None)
    ap.add_argument("--resume-from-log", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--use-wandb", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--wandb-project", default="revise_nextqa")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default=None)
    ap.add_argument("--wandb-mode", default=None, choices=[None, "online", "offline"])

    args = ap.parse_args()
    if args.captions_dir and not os.path.isdir(args.captions_dir):
        raise ValueError(f"--captions-dir does not exist or is not a directory: {args.captions_dir}")
    if args.auto_generate_captions and args.generated_captions_dir:
        os.makedirs(args.generated_captions_dir, exist_ok=True)
    if args.base_url and args.start_server:
        raise ValueError("--base-url cannot be combined with --start-server.")

    random.seed(args.seed)
    rng = random.Random(args.seed)

    server_proc = None
    run = None
    try:
        if args.start_server:
            server_proc = _start_vllm_server(args)
            _wait_port(args.host, args.port, timeout_s=args.server_timeout_s)
            _wait_for_server(args.host, args.port, timeout_s=args.server_timeout_s)

        base_url = _resolve_base_url(args.base_url, args.host, args.port)
        model_id = _get_model_id(base_url, model_id=args.model_id)

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

        completed_ids: set[str] = set()
        if args.resume_from_log and args.log_jsonl:
            completed_ids = _read_completed_sample_ids(args.log_jsonl)

        if args.use_wandb and wandb is not None:
            mode = args.wandb_mode or os.getenv("WANDB_MODE") or ("online" if os.getenv("WANDB_API_KEY") else "offline")
            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_name,
                group=args.wandb_group,
                config={
                    "task": "nextqa_videoagent_caption_vllm",
                    "csv": args.csv,
                    "captions_dir": args.captions_dir,
                    "generated_captions_dir": args.generated_captions_dir,
                    "auto_generate_captions": bool(args.auto_generate_captions),
                    "model_path": args.model_path,
                    "num_shards": num_shards,
                    "shard_idx": shard_idx,
                    "max_samples": args.max_samples,
                    "max_rounds": args.max_rounds,
                    "max_frames_per_round": args.max_frames_per_round,
                    "caption_max_chars": args.caption_max_chars,
                    "caption_gen_max_frames": args.caption_gen_max_frames,
                    "caption_gen_stride_s": args.caption_gen_stride_s,
                    "caption_gen_timeout_s": args.caption_gen_timeout_s,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": args.max_tokens,
                    "resume_from_log": bool(args.resume_from_log),
                    "initial_completed": len(completed_ids),
                },
            )
            if mode:
                os.environ["WANDB_MODE"] = str(mode)

        processed = 0
        correct = 0
        failed = 0
        invalid_outputs = 0
        total_rounds = 0
        total_frames_used = 0
        total_model_calls = 0
        generated_caption_samples = 0
        start_eval = time.time()

        for sample in samples:
            if sample.sample_id in completed_ids:
                continue
            processed += 1
            gt_letter = chr(ord("A") + int(sample.answer_idx))
            try:
                generated_path = (
                    os.path.join(args.generated_captions_dir, f"{sample.video_id}_cap.json")
                    if args.generated_captions_dir
                    else None
                )
                had_generated_before = bool(generated_path and os.path.exists(generated_path))
                caps = ensure_video_captions(
                    captions_dir=args.captions_dir,
                    generated_captions_dir=args.generated_captions_dir,
                    video_id=sample.video_id,
                    video_path=sample.video_path,
                    base_url=base_url,
                    model_id=model_id,
                    auto_generate=bool(args.auto_generate_captions),
                    max_frames=int(args.caption_gen_max_frames),
                    stride_s=int(args.caption_gen_stride_s),
                    timeout_s=int(args.caption_gen_timeout_s),
                )
                if generated_path and os.path.exists(generated_path) and not had_generated_before:
                    generated_caption_samples += 1
                caption_keys, caption_texts = _as_caption_list(caps)
                if not caption_keys:
                    raise RuntimeError("No captions found")

                # Precompute TF-IDF for retrieval.
                vectorizer, matrix = _build_video_tfidf([_truncate_text(t, int(args.caption_max_chars)) for t in caption_texts])

                seen = set(_pick_uniform_indices(caption_keys, k=5))
                answer_letter: Optional[str] = None
                conf: Optional[int] = None
                used_rounds = 0

                for round_idx in range(1, int(args.max_rounds) + 1):
                    used_rounds = round_idx
                    observed_block = _format_captions(sorted(seen), caps, max_chars=int(args.caption_max_chars))

                    answer_letter, conf, ans_prompt, ans_raw = _answer_with_confidence(
                        base_url=base_url,
                        model_id=model_id,
                        question=sample.question,
                        choices=sample.choices,
                        observed_caption_block=observed_block,
                        temperature=float(args.temperature),
                        top_p=float(args.top_p),
                        max_tokens=int(args.max_tokens),
                        timeout_s=int(args.request_timeout_s),
                    )
                    total_model_calls += 1
                    _maybe_log_jsonl(
                        args.log_jsonl,
                        {
                            "ts": time.time(),
                            "event": "answer",
                            "sample_id": sample.sample_id,
                            "qid": sample.qid,
                            "video_id": sample.video_id,
                            "round_idx": round_idx,
                            "question": sample.question,
                            "choices": sample.choices,
                            "ground_truth": gt_letter,
                            "seen_indices": sorted(seen),
                            "answer": answer_letter,
                            "confidence": conf,
                            "user_text": ans_prompt,
                            "raw_output": ans_raw,
                        },
                    )

                    if answer_letter is None:
                        invalid_outputs += 1

                    if conf == 3 or round_idx >= int(args.max_rounds):
                        break

                    max_new = int(args.max_frames_per_round)
                    max_idx = int(max(caption_keys))
                    segments = _build_segments(sorted(seen), max_idx=max_idx)
                    requests, req_prompt, req_raw = _request_more(
                        base_url=base_url,
                        model_id=model_id,
                        question=sample.question,
                        choices=sample.choices,
                        observed_caption_block=observed_block,
                        segments=segments,
                        k=max_new,
                        temperature=float(args.temperature),
                        top_p=float(args.top_p),
                        max_tokens=int(args.max_tokens),
                        timeout_s=int(args.request_timeout_s),
                    )
                    total_model_calls += 1
                    _maybe_log_jsonl(
                        args.log_jsonl,
                        {
                            "ts": time.time(),
                            "event": "request",
                            "sample_id": sample.sample_id,
                            "qid": sample.qid,
                            "video_id": sample.video_id,
                            "round_idx": round_idx,
                            "segments": [s.__dict__ for s in segments],
                            "requests": requests,
                            "user_text": req_prompt,
                            "raw_output": req_raw,
                        },
                    )

                    new_indices: list[int] = []
                    for r in requests:
                        idx = _retrieve_best_index(
                            query=str(r.get("query", "")),
                            segments=segments,
                            segment_id=int(r.get("segment_id", 0)),
                            caption_keys=caption_keys,
                            caption_texts=[_truncate_text(t, int(args.caption_max_chars)) for t in caption_texts],
                            vectorizer=vectorizer,
                            matrix=matrix,
                            seen=seen,
                        )
                        if idx is None:
                            continue
                        if idx not in seen:
                            new_indices.append(idx)
                            seen.add(idx)
                        if len(new_indices) >= max_new:
                            break

                    if not new_indices:
                        # Fallback: sample unseen indices uniformly from caption timeline.
                        unseen = [k for k in caption_keys if k not in seen]
                        if unseen:
                            extra = rng.sample(unseen, k=min(max_new, len(unseen)))
                            for idx in extra:
                                seen.add(int(idx))

                if answer_letter is None:
                    failed += 1
                else:
                    if answer_letter == gt_letter:
                        correct += 1

                total_rounds += used_rounds
                total_frames_used += len(seen)

                _maybe_log_jsonl(
                    args.log_jsonl,
                    {
                        "ts": time.time(),
                        "event": "final",
                        "sample_id": sample.sample_id,
                        "qid": sample.qid,
                        "video_id": sample.video_id,
                        "ground_truth": gt_letter,
                        "pred": answer_letter,
                        "correct": bool(answer_letter == gt_letter),
                        "rounds": used_rounds,
                        "frames_used": len(seen),
                    },
                )
            except Exception as e:
                failed += 1
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
                    _wait_for_server(args.host, args.port, timeout_s=args.server_timeout_s)
                    model_id = _get_model_id(base_url, model_id=args.model_id)
                _maybe_log_jsonl(
                    args.log_jsonl,
                    {
                        "ts": time.time(),
                        "event": "error",
                        "sample_id": sample.sample_id,
                        "qid": sample.qid,
                        "video_id": sample.video_id,
                        "error": repr(e),
                    },
                )

            if args.progress_interval and (processed % int(args.progress_interval) == 0):
                elapsed = time.time() - start_eval
                acc_now = correct / max(1, processed)
                avg_rounds = total_rounds / max(1, processed)
                avg_frames = total_frames_used / max(1, processed)
                calls_per_sample = total_model_calls / max(1, processed)
                print(
                    f"[{processed}/{len(samples)}] acc={acc_now:.4f} avg_rounds={avg_rounds:.3f} "
                    f"avg_frames={avg_frames:.2f} failed={failed} invalid={invalid_outputs} "
                    f"calls={total_model_calls} calls/sample={calls_per_sample:.2f} elapsed_s={elapsed:.1f}",
                    flush=True,
                )
                _wandb_log(
                    run,
                    {
                        "eval/acc": acc_now,
                        "eval/avg_rounds": avg_rounds,
                        "eval/avg_frames_used": avg_frames,
                        "eval/failed": failed,
                        "eval/processed": processed,
                        "eval/elapsed_s": elapsed,
                        "eval/invalid_outputs": invalid_outputs,
                        "eval/total_model_calls": total_model_calls,
                        "eval/calls_per_sample": calls_per_sample,
                    },
                    step=processed,
                )

        elapsed_s = time.time() - start_eval
        samples_n = max(1, processed)
        results = {
            "samples": processed,
            "correct": correct,
            "accuracy": correct / samples_n,
            "failed": failed,
            "invalid_outputs": invalid_outputs,
            "total_rounds": total_rounds,
            "avg_rounds": total_rounds / samples_n,
            "total_frames_used": total_frames_used,
            "avg_frames_used": total_frames_used / samples_n,
            "total_model_calls": total_model_calls,
            "generated_caption_samples": generated_caption_samples,
            "elapsed_s": elapsed_s,
        }

        if args.summary_json:
            os.makedirs(os.path.dirname(args.summary_json) or ".", exist_ok=True)
            with open(args.summary_json, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "results": results,
                        "captions_dir": args.captions_dir,
                        "generated_captions_dir": args.generated_captions_dir,
                        "auto_generate_captions": bool(args.auto_generate_captions),
                        "caption_gen_max_frames": args.caption_gen_max_frames,
                        "caption_gen_stride_s": args.caption_gen_stride_s,
                        "caption_gen_timeout_s": args.caption_gen_timeout_s,
                        "prompt_log_jsonl": args.log_jsonl,
                        "wandb": {
                            "enabled": bool(args.use_wandb),
                            "project": args.wandb_project,
                            "entity": args.wandb_entity,
                            "name": args.wandb_name,
                            "group": args.wandb_group,
                            "id": getattr(run, "id", None),
                            "url": getattr(run, "url", None),
                        },
                        "command": "python " + " ".join(os.sys.argv),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        if run is not None:
            run.summary["final_acc"] = results["accuracy"]
            run.summary["final_correct"] = correct
            run.summary["final_samples"] = processed
            run.summary["avg_rounds"] = results["avg_rounds"]
            run.summary["avg_frames_used"] = results["avg_frames_used"]
            run.summary["failed"] = failed
            run.summary["invalid_outputs"] = invalid_outputs
            run.summary["elapsed_s"] = elapsed_s
            run.summary["total_model_calls"] = total_model_calls
            run.summary["generated_caption_samples"] = generated_caption_samples
            run.finish()

        return 0
    finally:
        if server_proc is not None:
            try:
                _stop_server(server_proc)
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
