#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from openai import BadRequestError, OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.revise import plug_and_play_nextqa_vllm as pnp  # type: ignore


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


ANSWER_SCHEMA = {
    "name": "videoagent_answer",
    "schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "enum": ["A", "B", "C", "D", "E"]},
            "confidence": {"type": "integer", "enum": [1, 2, 3]},
        },
        "required": ["answer", "confidence"],
        "additionalProperties": False,
    },
    "strict": True,
}


REQUEST_SCHEMA = {
    "name": "videoagent_requests",
    "schema": {
        "type": "object",
        "properties": {
            "requests": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "segment_id": {"type": "integer", "minimum": 1},
                        "query": {"type": "string"},
                    },
                    "required": ["segment_id", "query"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["requests"],
        "additionalProperties": False,
    },
    "strict": True,
}


def _safe_json_loads(line: str) -> Optional[dict[str, Any]]:
    try:
        obj = json.loads(line)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = _safe_json_loads(line)
            if obj:
                yield obj


def _resume_openai_batch(client: OpenAI, *, out_dir: Path, poll_interval_s: float) -> Optional[Path]:
    meta_path = out_dir / "batch_meta.json"
    out_jsonl_path = out_dir / "batch_output.jsonl"
    if out_jsonl_path.exists():
        return out_jsonl_path
    if not meta_path.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    batch_id = meta.get("batch_id")
    if not isinstance(batch_id, str) or not batch_id:
        return None

    batch = client.batches.retrieve(batch_id)
    status = batch.status
    while status not in {"completed", "failed", "cancelled", "expired"}:
        time.sleep(float(poll_interval_s))
        batch = client.batches.retrieve(batch_id)
        status = batch.status

    if status != "completed":
        raise RuntimeError(f"Batch did not complete successfully: status={status} batch_id={batch_id}")

    if not batch.output_file_id:
        if batch.error_file_id:
            err_path = out_dir / "batch_error.jsonl"
            err_path.write_bytes(client.files.content(batch.error_file_id).read())
            raise RuntimeError(
                f"Batch completed but only produced errors (no output_file_id). batch_id={batch_id} saved={err_path}"
            )
        raise RuntimeError(f"Batch completed but has no output_file_id: batch_id={batch_id}")

    out_jsonl_path.write_bytes(client.files.content(batch.output_file_id).read())
    return out_jsonl_path


def _run_openai_batch(
    client: OpenAI,
    *,
    input_jsonl_path: Path,
    out_dir: Path,
    completion_window: str,
    metadata: dict[str, str],
    poll_interval_s: float,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    file_obj = client.files.create(file=input_jsonl_path.open("rb"), purpose="batch")
    try:
        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window=str(completion_window),
            metadata=metadata,
        )
    except BadRequestError as e:
        msg = ""
        code = ""
        try:
            payload = getattr(e, "response", None).json() if getattr(e, "response", None) is not None else {}
            msg = str(((payload or {}).get("error") or {}).get("message") or "")
            code = str(((payload or {}).get("error") or {}).get("code") or "")
        except Exception:
            pass
        raise RuntimeError(f"OpenAI Batch create failed: code={code!r} message={msg!r}") from e

    (out_dir / "batch_meta.json").write_text(
        json.dumps(
            {
                "batch_id": batch.id,
                "input_file_id": file_obj.id,
                "endpoint": "/v1/chat/completions",
                "completion_window": str(completion_window),
                "metadata": metadata,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    status = batch.status
    while status not in {"completed", "failed", "cancelled", "expired"}:
        time.sleep(float(poll_interval_s))
        batch = client.batches.retrieve(batch.id)
        status = batch.status

    if status != "completed":
        raise RuntimeError(f"Batch did not complete successfully: status={status} batch_id={batch.id}")

    if not batch.output_file_id:
        if batch.error_file_id:
            err_path = out_dir / "batch_error.jsonl"
            err_path.write_bytes(client.files.content(batch.error_file_id).read())
            raise RuntimeError(
                f"Batch completed but only produced errors (no output_file_id). batch_id={batch.id} saved={err_path}"
            )
        raise RuntimeError(f"Batch completed but has no output_file_id: batch_id={batch.id}")

    out_jsonl_path = out_dir / "batch_output.jsonl"
    out_jsonl_path.write_bytes(client.files.content(batch.output_file_id).read())
    return out_jsonl_path


def _extract_content_and_json(line_obj: dict[str, Any]) -> tuple[str, Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    resp = line_obj.get("response") or {}
    if int(resp.get("status_code") or 0) != 200:
        return "", None, None
    body = resp.get("body") or {}
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else None
    choices = body.get("choices") or []
    if not choices:
        return "", None, usage
    msg = (choices[0] or {}).get("message") or {}
    content = msg.get("content") if isinstance(msg.get("content"), str) else ""
    try:
        obj = json.loads(content) if content else None
    except Exception:
        obj = None
    return content, obj if isinstance(obj, dict) else None, usage


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


def _format_captions(indices: list[int], captions: dict[int, str], *, max_chars: int) -> str:
    lines: list[str] = []
    for idx in sorted(set(int(i) for i in indices)):
        cap = captions.get(int(idx))
        if not cap:
            continue
        lines.append(f"{idx}s: {pnp._truncate_text(cap, max_chars)}")
    return "\n".join(lines) if lines else "(no captions available)"


@dataclass
class Segment:
    segment_id: int
    start: int
    end: int

    def label(self) -> str:
        return f"{self.start}s-{self.end}s"


def _build_segments(seen: set[int], max_idx: int) -> list[Segment]:
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
    if not (1 <= int(segment_id) <= len(segments)):
        return None
    seg = segments[int(segment_id) - 1]

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
    return int(cand_indices[int(sims.argmax())])


@dataclass
class SampleState:
    sample: Any  # pnp.NextQASample
    gt_letter: str
    captions: dict[int, str]
    caption_keys: list[int]
    caption_texts: list[str]
    vectorizer: TfidfVectorizer
    matrix: Any
    seen: set[int]
    answer: Optional[str] = None
    confidence: Optional[int] = None
    rounds: int = 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.1")
    ap.add_argument("--csv", default="/shares/hlw3876/chenwei/NExT-QA/nextqa/val.csv")
    ap.add_argument("--video-root", default="/shares/hlw3876/chenwei/NExT-QA/NExTVideo")
    ap.add_argument("--map-json", default="/shares/hlw3876/chenwei/NExT-QA/map_vid_vidorID.json")
    ap.add_argument("--captions-dir", default="data/nextqa_allcaps_1fps")
    ap.add_argument("--max-samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rounds", type=int, default=5)
    ap.add_argument("--max-frames-per-round", type=int, default=5)
    ap.add_argument("--caption-max-chars", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-completion-tokens", type=int, default=512)
    ap.add_argument("--completion-window", default="24h", choices=["24h"])
    ap.add_argument("--poll-interval-s", type=float, default=10.0)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(str(args.captions_dir)):
        raise ValueError(f"--captions-dir does not exist: {args.captions_dir}")

    stamp = time.strftime("%Y-%m-%d")
    if not args.output_dir:
        args.output_dir = f"outputs/{stamp}/nextqa_videoagent_caption_{args.model}_n{int(args.max_samples)}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the same sample subset as other scripts (pnp._load_nextqa_samples).
    samples = pnp._load_nextqa_samples(
        csv_path=str(args.csv),
        map_json=str(args.map_json),
        video_root=str(args.video_root),
        max_samples=int(args.max_samples),
        seed=int(args.seed),
    )
    if not samples:
        raise RuntimeError("No samples loaded.")

    states: dict[str, SampleState] = {}
    for s in samples:
        sid = str(getattr(s, "sample_id"))
        caps = pnp._load_video_captions(str(args.captions_dir), str(s.video_id))
        keys, texts = _as_caption_list(caps)
        if not keys:
            continue
        trunc_texts = [pnp._truncate_text(t, int(args.caption_max_chars)) for t in texts]
        vectorizer, matrix = _build_video_tfidf(trunc_texts)
        seen = set(_pick_uniform_indices(keys, k=min(5, int(args.max_frames_per_round))))
        gt_letter = chr(ord("A") + int(s.answer_idx))
        states[sid] = SampleState(
            sample=s,
            gt_letter=gt_letter,
            captions=caps,
            caption_keys=keys,
            caption_texts=trunc_texts,
            vectorizer=vectorizer,
            matrix=matrix,
            seen=seen,
        )

    # Deterministically cap to requested sample count in case missing captions.
    all_ids = list(states.keys())
    if len(all_ids) > int(args.max_samples):
        all_ids = all_ids[: int(args.max_samples)]
        states = {sid: states[sid] for sid in all_ids}

    # Save sample list.
    (out_dir / "samples.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": sid,
                    "qid": st.sample.qid,
                    "video_id": st.sample.video_id,
                    "ground_truth": st.gt_letter,
                },
                ensure_ascii=False,
            )
            + "\n"
            for sid, st in states.items()
        ),
        encoding="utf-8",
    )

    preds_path = out_dir / "predictions.jsonl"
    if preds_path.exists() and args.resume:
        # Load previous predictions and only run unfinished.
        finished: set[str] = set()
        with preds_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = _safe_json_loads(line)
                if not obj:
                    continue
                sid = str(obj.get("sample_id") or "")
                if sid:
                    finished.add(sid)
                    st = states.get(sid)
                    if st is not None:
                        st.answer = str(obj.get("pred") or "") or None
                        st.confidence = int(obj.get("confidence") or 0) or None
                        st.rounds = int(obj.get("rounds") or 0) or 0
        active = [sid for sid in states.keys() if sid not in finished]
    else:
        active = list(states.keys())
        if preds_path.exists():
            preds_path.unlink()

    client = OpenAI()
    calls_log = out_dir / "calls.jsonl"
    if calls_log.exists() and not args.resume:
        calls_log.unlink()

    total_model_calls = 0
    invalid_outputs = 0
    rng = random.Random(int(args.seed))

    def write_call(rec: dict[str, Any]) -> None:
        with calls_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    for round_idx in range(1, int(args.max_rounds) + 1):
        if not active:
            break

        # ----------------
        # (1) Answer batch
        # ----------------
        ans_dir = out_dir / "rounds" / f"r{round_idx:02d}" / "answer"
        ans_dir.mkdir(parents=True, exist_ok=True)
        ans_input = ans_dir / "batch_input.jsonl"
        with ans_input.open("w", encoding="utf-8") as fout:
            for sid in active:
                st = states[sid]
                observed = _format_captions(sorted(st.seen), st.captions, max_chars=int(args.caption_max_chars))
                user_text = (
                    f"{_format_question(st.sample.question, st.sample.choices)}\n\n"
                    "Observed captions (index≈seconds):\n"
                    f"{observed}\n\n"
                    'Return JSON: {"answer":"A","confidence":3}\n'
                )
                req = {
                    "custom_id": sid,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": str(args.model),
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT_ANSWER},
                            {"role": "user", "content": user_text},
                        ],
                        "temperature": float(args.temperature),
                        "top_p": float(args.top_p),
                        "max_completion_tokens": int(args.max_completion_tokens),
                        "response_format": {"type": "json_schema", "json_schema": ANSWER_SCHEMA},
                    },
                }
                fout.write(json.dumps(req, ensure_ascii=False) + "\n")

        ans_out = (
            _resume_openai_batch(client, out_dir=ans_dir, poll_interval_s=float(args.poll_interval_s))
            if args.resume
            else None
        ) or _run_openai_batch(
            client,
            input_jsonl_path=ans_input,
            out_dir=ans_dir,
            completion_window=str(args.completion_window),
            metadata={"task": "nextqa_videoagent_answer", "round": str(round_idx)},
            poll_interval_s=float(args.poll_interval_s),
        )

        answered_now: list[str] = []
        need_more: list[str] = []
        for line_obj in _iter_jsonl(ans_out):
            sid = str(line_obj.get("custom_id") or "")
            if sid not in states:
                continue
            content, obj, usage = _extract_content_and_json(line_obj)
            total_model_calls += 1
            st = states[sid]
            st.rounds = max(st.rounds, round_idx)

            if not obj:
                invalid_outputs += 1
                need_more.append(sid)
                write_call(
                    {
                        "ts": time.time(),
                        "sample_id": sid,
                        "qid": st.sample.qid,
                        "video_id": st.sample.video_id,
                        "round_idx": round_idx,
                        "stage": "answer",
                        "status": "invalid_json",
                        "raw_output": content,
                        "usage": usage,
                    }
                )
                continue

            ans = str(obj.get("answer") or "").strip().upper()
            conf = obj.get("confidence")
            if ans not in {"A", "B", "C", "D", "E"} or conf not in {1, 2, 3}:
                invalid_outputs += 1
                need_more.append(sid)
                write_call(
                    {
                        "ts": time.time(),
                        "sample_id": sid,
                        "qid": st.sample.qid,
                        "video_id": st.sample.video_id,
                        "round_idx": round_idx,
                        "stage": "answer",
                        "status": "invalid_fields",
                        "raw_output": content,
                        "parsed": obj,
                        "usage": usage,
                    }
                )
                continue

            st.answer = ans
            st.confidence = int(conf)
            write_call(
                {
                    "ts": time.time(),
                    "sample_id": sid,
                    "qid": st.sample.qid,
                    "video_id": st.sample.video_id,
                    "round_idx": round_idx,
                    "stage": "answer",
                    "status": "ok",
                    "answer": ans,
                    "confidence": int(conf),
                    "usage": usage,
                    "raw_output": content,
                }
            )

            if int(conf) >= 3 or round_idx >= int(args.max_rounds):
                answered_now.append(sid)
            else:
                need_more.append(sid)

        # Persist predictions for the ones that decided to stop (including final round).
        if answered_now:
            with preds_path.open("a", encoding="utf-8") as f:
                for sid in answered_now:
                    st = states[sid]
                    f.write(
                        json.dumps(
                            {
                                "sample_id": sid,
                                "qid": st.sample.qid,
                                "video_id": st.sample.video_id,
                                "ground_truth": st.gt_letter,
                                "pred": st.answer,
                                "confidence": st.confidence,
                                "correct": bool(st.answer == st.gt_letter),
                                "rounds": int(st.rounds),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

        active = need_more
        if not active or round_idx >= int(args.max_rounds):
            break

        # -------------------
        # (2) Request batch
        # -------------------
        req_dir = out_dir / "rounds" / f"r{round_idx:02d}" / "request"
        req_dir.mkdir(parents=True, exist_ok=True)
        req_input = req_dir / "batch_input.jsonl"
        with req_input.open("w", encoding="utf-8") as fout:
            for sid in active:
                st = states[sid]
                observed = _format_captions(sorted(st.seen), st.captions, max_chars=int(args.caption_max_chars))
                max_idx = int(st.caption_keys[-1]) if st.caption_keys else 0
                segments = _build_segments(st.seen, max_idx=max_idx)
                if not segments:
                    # If we somehow covered everything, just force stop next round.
                    continue
                segment_text = "\n".join(f"{seg.segment_id}. {seg.label()}" for seg in segments)
                user_text = (
                    f"{_format_question(st.sample.question, st.sample.choices)}\n\n"
                    "Observed captions (index≈seconds):\n"
                    f"{observed}\n\n"
                    "Candidate segments:\n"
                    f"{segment_text}\n\n"
                    f"Choose up to K={int(args.max_frames_per_round)} requests.\n"
                    'Return JSON: {"requests":[{"segment_id":1,"query":"person action"}]}\n'
                )

                req = {
                    "custom_id": sid,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": str(args.model),
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT_REQUEST},
                            {"role": "user", "content": user_text},
                        ],
                        "temperature": float(args.temperature),
                        "top_p": float(args.top_p),
                        "max_completion_tokens": int(args.max_completion_tokens),
                        "response_format": {"type": "json_schema", "json_schema": REQUEST_SCHEMA},
                    },
                }
                fout.write(json.dumps(req, ensure_ascii=False) + "\n")

        req_out = (
            _resume_openai_batch(client, out_dir=req_dir, poll_interval_s=float(args.poll_interval_s))
            if args.resume
            else None
        ) or _run_openai_batch(
            client,
            input_jsonl_path=req_input,
            out_dir=req_dir,
            completion_window=str(args.completion_window),
            metadata={"task": "nextqa_videoagent_request", "round": str(round_idx)},
            poll_interval_s=float(args.poll_interval_s),
            )

        # Apply retrieval updates.
        for line_obj in _iter_jsonl(req_out):
            sid = str(line_obj.get("custom_id") or "")
            if sid not in states:
                continue
            content, obj, usage = _extract_content_and_json(line_obj)
            total_model_calls += 1
            st = states[sid]
            max_idx = int(st.caption_keys[-1]) if st.caption_keys else 0
            segments = _build_segments(st.seen, max_idx=max_idx)

            if not obj:
                invalid_outputs += 1
                # Fallback: pick a random unseen index.
                unseen = [i for i in st.caption_keys if int(i) not in st.seen]
                if unseen:
                    st.seen.add(int(rng.choice(unseen)))
                write_call(
                    {
                        "ts": time.time(),
                        "sample_id": sid,
                        "qid": st.sample.qid,
                        "video_id": st.sample.video_id,
                        "round_idx": round_idx,
                        "stage": "request",
                        "status": "invalid_json",
                        "raw_output": content,
                        "usage": usage,
                    }
                )
                continue

            reqs = obj.get("requests")
            if not isinstance(reqs, list):
                invalid_outputs += 1
                reqs = []

            chosen: list[int] = []
            for r in reqs[: int(args.max_frames_per_round)]:
                if not isinstance(r, dict):
                    continue
                seg_id = r.get("segment_id")
                query = str(r.get("query") or "").strip()
                try:
                    seg_id_int = int(seg_id)
                except Exception:
                    continue
                idx = _retrieve_best_index(
                    query=query,
                    segments=segments,
                    segment_id=seg_id_int,
                    caption_keys=st.caption_keys,
                    caption_texts=st.caption_texts,
                    vectorizer=st.vectorizer,
                    matrix=st.matrix,
                    seen=st.seen,
                )
                if idx is not None:
                    chosen.append(int(idx))

            # Fallback if nothing selected.
            if not chosen:
                unseen = [i for i in st.caption_keys if int(i) not in st.seen]
                if unseen:
                    chosen = [int(unseen[0])]

            for idx in chosen:
                st.seen.add(int(idx))

            write_call(
                {
                    "ts": time.time(),
                    "sample_id": sid,
                    "qid": st.sample.qid,
                    "video_id": st.sample.video_id,
                    "round_idx": round_idx,
                    "stage": "request",
                    "status": "ok",
                    "requests": obj.get("requests"),
                    "chosen_indices": chosen,
                    "usage": usage,
                    "raw_output": content,
                }
            )

    # Finalize: ensure all have a prediction line (for any leftover actives, keep last answer).
    finished: set[str] = set()
    if preds_path.exists():
        with preds_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = _safe_json_loads(line)
                if obj and obj.get("sample_id"):
                    finished.add(str(obj["sample_id"]))

    remaining = [sid for sid in states.keys() if sid not in finished]
    if remaining:
        with preds_path.open("a", encoding="utf-8") as f:
            for sid in remaining:
                st = states[sid]
                # If never answered, default to A with low confidence.
                pred = st.answer or "A"
                conf = int(st.confidence or 1)
                rounds = int(st.rounds or int(args.max_rounds))
                f.write(
                    json.dumps(
                        {
                            "sample_id": sid,
                            "qid": st.sample.qid,
                            "video_id": st.sample.video_id,
                            "ground_truth": st.gt_letter,
                            "pred": pred,
                            "confidence": conf,
                            "correct": bool(pred == st.gt_letter),
                            "rounds": rounds,
                            "terminated_reason": "forced_finalize",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    # Aggregate summary
    correct = 0
    total = 0
    total_rounds = 0
    with preds_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = _safe_json_loads(line)
            if not obj:
                continue
            total += 1
            if obj.get("correct"):
                correct += 1
            total_rounds += int(obj.get("rounds") or 0)

    results = {
        "samples": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "avg_rounds": (total_rounds / total) if total else 0.0,
        "invalid_outputs": int(invalid_outputs),
        "total_model_calls": int(total_model_calls),
    }

    summary = {
        "task": "openai_batch_nextqa_videoagent_caption",
        "model": str(args.model),
        "dataset_csv": str(args.csv),
        "video_root": str(args.video_root),
        "map_json": str(args.map_json),
        "captions_dir": str(args.captions_dir),
        "max_samples": int(args.max_samples),
        "seed": int(args.seed),
        "max_rounds": int(args.max_rounds),
        "max_frames_per_round": int(args.max_frames_per_round),
        "caption_max_chars": int(args.caption_max_chars),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "max_completion_tokens": int(args.max_completion_tokens),
        "results": results,
        "paths": {
            "output_dir": str(out_dir),
            "samples_jsonl": str(out_dir / "samples.jsonl"),
            "calls_jsonl": str(calls_log),
            "predictions_jsonl": str(preds_path),
        },
        "command": "python " + " ".join(sys.argv),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
