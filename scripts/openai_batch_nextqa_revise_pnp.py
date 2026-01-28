#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI
from openai import BadRequestError
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse the exact prompt + parsing logic from the Qwen/vLLM runner.
from examples.revise import plug_and_play_nextqa_vllm as pnp  # type: ignore


def _encode_jpeg_b64(img: Image.Image, *, max_side: int, quality: int) -> str:
    if max_side and max_side > 0:
        img = img.convert("RGB")
        img.thumbnail((int(max_side), int(max_side)), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_multimodal_user_content(
    user_text: str,
    images: list[Image.Image],
    *,
    image_max_side: int,
    image_quality: int,
) -> list[dict[str, Any]]:
    if not images:
        return [{"type": "text", "text": user_text}]

    content: list[dict[str, Any]] = []
    parts = user_text.split("<image>")
    if (len(parts) - 1) == len(images):
        for i, img in enumerate(images):
            if parts[i]:
                content.append({"type": "text", "text": parts[i]})
            b64 = _encode_jpeg_b64(img, max_side=image_max_side, quality=image_quality)
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        if parts[-1]:
            content.append({"type": "text", "text": parts[-1]})
        return content

    # Fallback: prepend all images, then the full text.
    for img in images:
        b64 = _encode_jpeg_b64(img, max_side=image_max_side, quality=image_quality)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    content.append({"type": "text", "text": user_text})
    return content


def _safe_json_loads(line: str) -> Optional[dict[str, Any]]:
    try:
        obj = json.loads(line)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _batch_write_error_file(client: OpenAI, error_file_id: str, out_path: Path) -> None:
    out_path.write_bytes(client.files.content(error_file_id).read())


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
            _batch_write_error_file(client, batch.error_file_id, err_path)
            raise RuntimeError(
                f"Batch completed but only produced errors (no output_file_id). batch_id={batch.id} saved={err_path}"
            )
        raise RuntimeError(f"Batch completed but has no output_file_id: batch_id={batch.id}")

    out_jsonl_path = out_dir / "batch_output.jsonl"
    out_jsonl_path.write_bytes(client.files.content(batch.output_file_id).read())
    return out_jsonl_path


def _extract_assistant_content_from_batch_line(line_obj: dict[str, Any]) -> tuple[int, Optional[str], Optional[str]]:
    resp = line_obj.get("response") or {}
    status_code = int(resp.get("status_code") or 0)
    body = resp.get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        return status_code, None, None
    choice0 = choices[0] or {}
    finish_reason = choice0.get("finish_reason")
    msg = choice0.get("message") or {}
    content = msg.get("content")
    if not isinstance(content, str):
        content = None
    if finish_reason is not None and not isinstance(finish_reason, str):
        finish_reason = str(finish_reason)
    return status_code, content, finish_reason


@dataclass
class SampleState:
    sample: Any  # pnp.NextQASample
    gt_letter: str
    frame_count: int
    fps: float
    captions: dict[int, str] = field(default_factory=dict)

    summary_state: str = (
        "P: I will summarize what has been shown so far; "
        "O: I will record the key observations from the current evidence; "
        "H: I will update my belief as new evidence arrives; "
        "U: some key detail may still be unclear; "
        "R: request more evidence if needed"
    )
    seen_frames: list[int] = field(default_factory=list)
    next_frames: list[int] = field(default_factory=list)

    current_frames: list[int] = field(default_factory=list)
    candidate_next_frames: list[int] = field(default_factory=list)
    base_user_text: str = ""
    forced_base_user_text: str = ""
    last_frames: list[int] = field(default_factory=list)

    answer_letter: Optional[str] = None
    answer_round: Optional[int] = None
    terminated_invalid_action: bool = False
    terminated_reason: Optional[str] = None
    effective_rounds: int = 0


@dataclass
class Stats:
    processed: int = 0
    correct: int = 0
    failed: int = 0
    invalid_outputs: int = 0
    invalid_action_terminated: int = 0
    total_retries: int = 0
    total_model_calls: int = 0
    fallback_frames_used: int = 0
    total_rounds: int = 0
    total_effective_rounds: int = 0


def _caption_for_index(state: SampleState, idx: int, observation_mode: str) -> str:
    if not state.captions:
        return "[no caption]"
    key = int(idx)
    if observation_mode != "caption":
        key = pnp._caption_key_for_frame_index(int(idx), state.fps)
    return state.captions.get(int(key)) or "[no caption]"


def _prepare_round_context(
    state: SampleState,
    *,
    rng: random.Random,
    round_idx: int,
    args: argparse.Namespace,
) -> None:
    frame_count = int(state.frame_count)

    # Frames shown in this round.
    frames_this_round = [i for i in (state.next_frames or []) if i not in state.seen_frames]
    if not frames_this_round:
        frames_this_round = pnp._sample_uniform_indices(frame_count, 1)
    frames_this_round = frames_this_round[: int(args.max_frames_per_round)]
    for i in frames_this_round:
        if i not in state.seen_frames:
            state.seen_frames.append(int(i))

    # Candidate unseen frames.
    candidate_next_frames: list[int] = []
    if bool(args.use_candidate_frames):
        k = int(args.candidate_k) if args.candidate_k is not None else max(12, int(args.max_frames_per_round) * 4)
        candidate_next_frames = pnp._propose_candidate_frames(
            frame_count=frame_count,
            seen=set(int(i) for i in state.seen_frames),
            k=k,
            rng=rng,
        )

    question_block = pnp._format_question(state.sample.question, state.sample.choices)

    shown_captions: Optional[list[str]] = None
    candidate_captions: Optional[list[str]] = None
    shown_ts: Optional[list[int]] = None
    candidate_ts: Optional[list[int]] = None
    if state.captions:
        include = str(getattr(args, "caption_include", "none"))
        max_chars = int(getattr(args, "caption_max_chars", 0))
        if include in ("shown", "both"):
            shown_captions = [pnp._truncate_text(_caption_for_index(state, int(i), args.observation_mode), max_chars) for i in frames_this_round]
            if args.observation_mode != "caption":
                shown_ts = [pnp._caption_key_for_frame_index(int(i), state.fps) for i in frames_this_round]
        if include in ("candidate", "both") and candidate_next_frames:
            candidate_captions = [
                pnp._truncate_text(_caption_for_index(state, int(i), args.observation_mode), max_chars)
                for i in candidate_next_frames
            ]
            if args.observation_mode != "caption":
                candidate_ts = [pnp._caption_key_for_frame_index(int(i), state.fps) for i in candidate_next_frames]

    user_text = pnp._build_user_text(
        question_block=question_block,
        summary=state.summary_state,
        frame_count=frame_count,
        round_idx=int(round_idx),
        frame_indices=[int(i) for i in frames_this_round],
        seen_frames=[int(i) for i in state.seen_frames],
        render_images=(args.observation_mode != "caption"),
        hide_seen_frames=bool(args.hide_seen_frames_in_prompt),
        candidate_unseen_frames=candidate_next_frames if bool(args.use_candidate_frames) else None,
        use_candidate_frame_ids=bool(args.use_candidate_frame_ids),
        require_candidate_frames=bool(args.require_candidate_frames),
        shown_frame_captions=shown_captions,
        candidate_id_captions=candidate_captions,
        shown_frame_ts=shown_ts,
        candidate_id_ts=candidate_ts,
    )

    if bool(args.force_final_answer) and int(round_idx) >= int(args.max_rounds):
        user_text = (
            f"{user_text}\n\n"
            "This is the final round. You MUST answer now using <summary>...</summary> then <answer>LETTER</answer>."
        )

    state.current_frames = [int(i) for i in frames_this_round]
    state.candidate_next_frames = [int(i) for i in candidate_next_frames]
    state.base_user_text = user_text
    state.last_frames = list(state.current_frames)


def _handle_output_one_attempt(
    state: SampleState,
    *,
    raw: str,
    round_idx: int,
    retry_idx: int,
    args: argparse.Namespace,
    rng: random.Random,
    stats: Stats,
) -> tuple[bool, Optional[str]]:
    """Return (accepted, retry_feedback). If accepted=True the round is resolved for this sample."""

    frame_count = int(state.frame_count)
    seen_frames = state.seen_frames
    candidate_next_frames = state.candidate_next_frames

    summary = pnp._extract_tag(raw, pnp._SUMMARY_RE)
    if (
        summary
        and (not pnp._is_placeholder(summary))
        and (not pnp._contains_banned_example(summary))
        and pnp._summary_has_ohrpu(summary)
        and (not pnp._summary_has_stale_boilerplate(summary, seen_count=len(seen_frames)))
    ):
        state.summary_state = summary

    think = pnp._extract_tag(raw, pnp._THINK_RE)
    if think is not None:
        stats.invalid_outputs += 1
        state.terminated_reason = "invalid_think"
        if retry_idx < int(args.max_retries_per_round):
            stats.total_retries += 1
            return False, pnp._retry_feedback_text(
                "Invalid response: do NOT output <think>. Output ONLY <summary> plus either <frames> (request) or <answer> (final).",
                force_answer=bool(args.force_final_answer and round_idx >= int(args.max_rounds)),
            )
        if args.strict_actions:
            stats.invalid_action_terminated += 1
            state.terminated_invalid_action = True
            state.answer_letter = None
            return True, None
        stats.fallback_frames_used += 1
        requested = pnp._sample_unseen_frames(frame_count, set(seen_frames), int(args.max_frames_per_round), rng=rng)
        state.next_frames = requested[: int(args.max_frames_per_round)] if requested else pnp._sample_uniform_indices(frame_count, 1)
        return True, None

    answer = pnp._extract_tag(raw, pnp._ANSWER_RE)
    if answer:
        answer_letter = pnp._normalize_answer_letter(answer, len(state.sample.choices))
        if answer_letter is None:
            stats.invalid_outputs += 1
            state.terminated_reason = "invalid_answer_letter"
            if retry_idx < int(args.max_retries_per_round):
                stats.total_retries += 1
                return False, pnp._retry_feedback_text(
                    "Invalid response: <answer> must be exactly ONE option letter (A/B/C/D/E). Do not output words or a sentence.",
                    force_answer=True,
                )
            if args.strict_actions:
                stats.invalid_action_terminated += 1
                state.terminated_invalid_action = True
                state.answer_letter = None
                return True, None
            stats.fallback_frames_used += 1
            state.next_frames = pnp._sample_unseen_frames(frame_count, set(seen_frames), int(args.max_frames_per_round), rng=rng)
            if not state.next_frames:
                state.next_frames = pnp._sample_uniform_indices(frame_count, 1)
            state.answer_letter = None
            return True, None

        if bool(args.answer_only_final_round) and int(round_idx) < int(args.max_rounds):
            stats.invalid_outputs += 1
            state.terminated_reason = "early_answer_disallowed"
            if retry_idx < int(args.max_retries_per_round):
                stats.total_retries += 1
                return False, pnp._retry_feedback_text(
                    "Invalid response: do NOT answer yet. Request more frames using <summary>...</summary> and <frames>...</frames>.",
                    force_answer=False,
                )
            if args.strict_actions:
                stats.invalid_action_terminated += 1
                state.terminated_invalid_action = True
                state.answer_letter = None
                return True, None
            stats.fallback_frames_used += 1
            state.next_frames = pnp._sample_unseen_frames(frame_count, set(seen_frames), int(args.max_frames_per_round), rng=rng)
            if not state.next_frames:
                state.next_frames = pnp._sample_uniform_indices(frame_count, 1)
            state.answer_letter = None
            return True, None

        if (
            summary is None
            or pnp._is_placeholder(summary)
            or pnp._contains_banned_example(summary)
            or (not pnp._summary_has_ohrpu(summary))
            or pnp._summary_has_stale_boilerplate(summary, seen_count=len(seen_frames))
        ):
            stats.invalid_outputs += 1
            state.terminated_reason = "invalid_answer_summary"
            if retry_idx < int(args.max_retries_per_round):
                stats.total_retries += 1
                return False, pnp._retry_feedback_text(
                    "Invalid response: when answering, include a meaningful <summary> with P/O/H/U/R in that exact order "
                    "(no placeholders like '.../none/unknown').",
                    force_answer=True,
                )

        state.answer_letter = answer_letter
        state.answer_round = int(round_idx)
        return True, None

    frames_text = pnp._extract_tag(raw, pnp._FRAMES_RE)
    if frames_text is None:
        stats.invalid_outputs += 1
        state.terminated_reason = "missing_frames_tag"
        if retry_idx < int(args.max_retries_per_round):
            stats.total_retries += 1
            return False, pnp._retry_feedback_text(
                "Invalid response: missing <frames> tag for requesting more frames. "
                "Remember: <frames> must list NEW frame indices to view NEXT (not already seen).",
                force_answer=bool(args.force_final_answer and round_idx >= int(args.max_rounds)),
            )
        if args.strict_actions:
            stats.invalid_action_terminated += 1
            state.terminated_invalid_action = True
            state.answer_letter = None
            return True, None
        state.next_frames = pnp._sample_uniform_indices(frame_count, 1)
        return True, None

    if summary is None or pnp._is_placeholder(summary) or pnp._contains_banned_example(summary) or (not pnp._summary_has_ohrpu(summary)):
        stats.invalid_outputs += 1
        state.terminated_reason = "invalid_select_summary"
        if retry_idx < int(args.max_retries_per_round):
            stats.total_retries += 1
            return False, pnp._retry_feedback_text(
                "Invalid response: include a meaningful <summary> with P/O/H/U/R in that exact order "
                "(no placeholders like '.../none/unknown').",
                force_answer=bool(args.force_final_answer and round_idx >= int(args.max_rounds)),
            )
    if summary is not None and pnp._summary_has_stale_boilerplate(summary, seen_count=len(seen_frames)):
        stats.invalid_outputs += 1
        state.terminated_reason = "stale_select_summary"
        if retry_idx < int(args.max_retries_per_round):
            stats.total_retries += 1
            return False, pnp._retry_feedback_text(
                "Invalid response: the <summary> claims no frames/captions were seen, but evidence was shown. "
                "Rewrite <summary> to reflect what was observed so far (P/O/H/U/R), then request frames.",
                force_answer=bool(args.force_final_answer and round_idx >= int(args.max_rounds)),
            )

    if (not bool(args.use_candidate_frame_ids)) and pnp._frames_has_range_syntax(frames_text):
        stats.invalid_outputs += 1
        state.terminated_reason = "frames_range_syntax"
        if retry_idx < int(args.max_retries_per_round):
            stats.total_retries += 1
            return False, pnp._retry_feedback_text(
                "Invalid response: <frames> must be a comma-separated list of integers only (NO ranges like '4-182', no hyphens). "
                "Choose up to {k} NEW frames.".format(k=int(args.max_frames_per_round)),
                force_answer=bool(args.force_final_answer and round_idx >= int(args.max_rounds)),
            )

    requested = pnp._dedupe_preserve_order(pnp._parse_frame_indices(frames_text))
    if bool(args.use_candidate_frame_ids) and candidate_next_frames:
        mapped: list[int] = []
        invalid_id = False
        for cid in requested:
            if 1 <= cid <= len(candidate_next_frames):
                mapped.append(int(candidate_next_frames[cid - 1]))
            else:
                invalid_id = True
        if invalid_id:
            stats.invalid_outputs += 1
            state.terminated_reason = "frames_out_of_range"
            if retry_idx < int(args.max_retries_per_round):
                stats.total_retries += 1
                return False, pnp._retry_feedback_text(
                    "Invalid response: when Candidate Frame IDs are provided, <frames> must contain only IDs in the allowed range [1..K] "
                    "(comma-separated integers).",
                    force_answer=bool(args.force_final_answer and round_idx >= int(args.max_rounds)),
                )
            stats.fallback_frames_used += 1
            requested = candidate_next_frames[: int(args.max_frames_per_round)]
            if not requested:
                requested = pnp._sample_unseen_frames(frame_count, set(seen_frames), int(args.max_frames_per_round), rng=rng)
            state.next_frames = requested[: int(args.max_frames_per_round)] if requested else pnp._sample_uniform_indices(frame_count, 1)
            return True, None

        requested = pnp._dedupe_preserve_order(mapped)
        requested = [i for i in requested if 0 <= i < frame_count and i not in seen_frames]
    else:
        if bool(args.require_candidate_frames) and candidate_next_frames:
            allowed = {int(i) for i in candidate_next_frames}
            disallowed = [i for i in requested if int(i) not in allowed]
            if disallowed:
                stats.invalid_outputs += 1
                state.terminated_reason = "frames_not_in_candidates"
                if retry_idx < int(args.max_retries_per_round):
                    stats.total_retries += 1
                    return False, pnp._retry_feedback_text(
                        "Invalid response: requested frames must be chosen ONLY from the candidate unseen frame list/ranges provided.",
                        force_answer=bool(args.force_final_answer and round_idx >= int(args.max_rounds)),
                    )
                if args.strict_actions:
                    stats.invalid_action_terminated += 1
                    state.terminated_invalid_action = True
                    state.answer_letter = None
                    return True, None
                requested = []
            else:
                requested = [i for i in requested if 0 <= i < frame_count and i not in seen_frames and int(i) in allowed]
        else:
            allowed_ranges = pnp._unseen_intervals(frame_count, seen_frames)
            requested = [i for i in requested if 0 <= i < frame_count and i not in seen_frames and pnp._in_intervals(i, allowed_ranges)]

    if requested and len(requested) > int(args.max_frames_per_round):
        stats.invalid_outputs += 1
        state.terminated_reason = "too_many_frames"
        requested = requested[: int(args.max_frames_per_round)]

    if not requested:
        stats.invalid_outputs += 1
        state.terminated_reason = "invalid_frames"
        if retry_idx < int(args.max_retries_per_round):
            stats.total_retries += 1
            candidate_text = f" Allowed unseen ranges: {pnp._format_intervals(pnp._unseen_intervals(frame_count, seen_frames))}."
            return False, pnp._retry_feedback_text(
                "Invalid response: requested frames must be NEW and within range. "
                "In <frames>, output 1–{k} comma-separated integers NOT in Seen frames.".format(k=int(args.max_frames_per_round))
                + candidate_text,
                force_answer=bool(args.force_final_answer and round_idx >= int(args.max_rounds)),
            )
        if args.strict_actions:
            stats.invalid_action_terminated += 1
            state.terminated_invalid_action = True
            state.answer_letter = None
            return True, None
        stats.fallback_frames_used += 1
        requested = candidate_next_frames[: int(args.max_frames_per_round)]
        if not requested:
            requested = pnp._sample_unseen_frames(frame_count, set(seen_frames), int(args.max_frames_per_round), rng=rng)
        state.next_frames = requested[: int(args.max_frames_per_round)] if requested else pnp._sample_uniform_indices(frame_count, 1)
        return True, None

    state.next_frames = requested[: int(args.max_frames_per_round)]
    state.effective_rounds += 1
    stats.total_effective_rounds += 1
    return True, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.1")
    ap.add_argument("--csv", default="/shares/hlw3876/chenwei/NExT-QA/nextqa/val.csv")
    ap.add_argument("--video-root", default="/shares/hlw3876/chenwei/NExT-QA/NExTVideo")
    ap.add_argument("--map-json", default="/shares/hlw3876/chenwei/NExT-QA/map_vid_vidorID.json")

    ap.add_argument("--max-samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--max-rounds", type=int, default=5)
    ap.add_argument("--max-frames-per-round", type=int, default=5)
    ap.add_argument("--use-candidate-frames", action="store_true", default=True)
    ap.add_argument("--candidate-k", type=int, default=None)
    ap.add_argument("--use-candidate-frame-ids", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--require-candidate-frames", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--hide-seen-frames-in-prompt", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--observation-mode", choices=["image", "caption"], default="image")
    ap.add_argument("--captions-dir", default=None)
    ap.add_argument("--caption-include", choices=["none", "shown", "candidate", "both"], default="none")
    ap.add_argument("--caption-max-chars", type=int, default=200)

    ap.add_argument("--max-retries-per-round", type=int, default=2)
    ap.add_argument("--strict-actions", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--force-final-answer", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--answer-only-final-round", action=argparse.BooleanOptionalAction, default=False)

    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-completion-tokens", type=int, default=1024)
    ap.add_argument("--completion-window", default="24h", choices=["24h"])
    ap.add_argument("--poll-interval-s", type=float, default=10.0)

    ap.add_argument("--image-max-side", type=int, default=256)
    ap.add_argument("--image-jpeg-quality", type=int, default=65)

    ap.add_argument("--output-dir", default=None)
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing --output-dir by reusing any completed batch_output.jsonl files and only submitting missing batches.",
    )
    args = ap.parse_args()

    if args.observation_mode == "caption":
        if not args.captions_dir:
            raise ValueError("--observation-mode caption requires --captions-dir")
        if args.caption_include == "none":
            args.caption_include = "shown"
    if args.caption_include != "none":
        if not args.captions_dir:
            raise ValueError("--caption-include requires --captions-dir")
        if not os.path.isdir(str(args.captions_dir)):
            raise ValueError(f"--captions-dir does not exist: {args.captions_dir}")

    stamp = time.strftime("%Y-%m-%d")
    if not args.output_dir:
        args.output_dir = f"outputs/{stamp}/openai_batch_nextqa_revise_pnp_{args.model}_n{int(args.max_samples)}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.resume) and (out_dir / "summary.json").exists():
        print((out_dir / "summary.json").read_text(encoding="utf-8"))
        return 0

    samples = pnp._load_nextqa_samples(
        csv_path=str(args.csv),
        map_json=str(args.map_json),
        video_root=str(args.video_root),
        max_samples=int(args.max_samples),
        seed=int(args.seed),
    )
    if not samples:
        raise RuntimeError("No samples loaded (check csv/map/video_root).")

    rng = random.Random(int(args.seed))

    # Initialize per-sample state (no images stored).
    states: dict[str, SampleState] = {}
    for s in samples:
        gt = chr(ord("A") + int(s.answer_idx))
        captions: dict[int, str] = {}
        fps = 0.0
        frame_count = int(s.frame_count)
        if args.captions_dir and args.caption_include != "none":
            captions = pnp._load_video_captions(str(args.captions_dir), str(s.video_id))
        if args.observation_mode == "caption":
            if captions:
                frame_count = max(captions.keys(), default=-1) + 1
            frame_count = max(1, int(frame_count))
        else:
            # For image mode, fps is only used for caption mapping.
            if captions:
                fps = pnp._get_video_fps(str(s.video_path))
        init_frames = pnp._sample_uniform_indices(frame_count, int(args.max_frames_per_round))
        state = SampleState(
            sample=s,
            gt_letter=gt,
            frame_count=int(frame_count),
            fps=float(fps),
            captions=captions,
            next_frames=[int(i) for i in init_frames if int(i) >= 0],
        )
        states[str(s.sample_id)] = state

    (out_dir / "samples.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": st.sample.sample_id,
                    "qid": st.sample.qid,
                    "video_id": st.sample.video_id,
                    "video_path": st.sample.video_path,
                    "ground_truth": st.gt_letter,
                },
                ensure_ascii=False,
            )
            + "\n"
            for st in states.values()
        ),
        encoding="utf-8",
    )

    calls_log = out_dir / "calls.jsonl"
    if calls_log.exists() and not bool(args.resume):
        calls_log.unlink()

    client = OpenAI()
    stats = Stats(processed=len(states))

    # System prompt (matches the baseline exactly).
    prompt_template = pnp.DEFAULT_SYSTEM_PROMPT_CAPTION_ONLY if args.observation_mode == "caption" else pnp.DEFAULT_SYSTEM_PROMPT
    system_prompt = prompt_template.format(max_frames_per_round=int(args.max_frames_per_round))

    active_ids = [sid for sid, st in states.items() if st.answer_letter is None and not st.terminated_invalid_action]

    for round_idx in range(1, int(args.max_rounds) + 1):
        if not active_ids:
            break

        # Prepare the per-sample prompt context for this round.
        for sid in active_ids:
            _prepare_round_context(states[sid], rng=rng, round_idx=round_idx, args=args)

        pending: dict[str, str] = {sid: "" for sid in active_ids}  # sid -> retry_feedback

        for retry_idx in range(0, int(args.max_retries_per_round) + 1):
            if not pending:
                break

            attempt_dir = out_dir / "rounds" / f"r{round_idx:02d}" / f"try{retry_idx}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            input_path = attempt_dir / "batch_input.jsonl"
            output_path_existing = attempt_dir / "batch_output.jsonl"
            reuse_existing_output = bool(args.resume) and output_path_existing.exists()

            if reuse_existing_output:
                # Re-run in resume mode: reuse the existing Batch output file and avoid re-submitting requests.
                out_path = output_path_existing
                stats.total_model_calls += int(len(pending))
            else:
                # If resuming and an input file already exists (but there's no output yet), reuse it.
                # This is helpful for runs that stopped mid-flight (e.g., billing hard limit) after writing batch_input.jsonl.
                if bool(args.resume) and input_path.exists():
                    stats.total_model_calls += int(len(pending))
                else:
                    with input_path.open("w", encoding="utf-8") as fout:
                        for sid, fb in pending.items():
                            st = states[sid]
                            user_text = st.base_user_text
                            if fb:
                                user_text = f"{user_text}\n\n{fb}"

                            images: list[Image.Image] = []
                            if args.observation_mode != "caption":
                                images = pnp._extract_frames(str(st.sample.video_path), [int(i) for i in st.current_frames])

                            req = {
                                "custom_id": f"{sid}",
                                "method": "POST",
                                "url": "/v1/chat/completions",
                                "body": {
                                    "model": str(args.model),
                                    "messages": [
                                        {"role": "system", "content": system_prompt},
                                        {
                                            "role": "user",
                                            "content": _build_multimodal_user_content(
                                                user_text,
                                                images,
                                                image_max_side=int(args.image_max_side),
                                                image_quality=int(args.image_jpeg_quality),
                                            ),
                                        },
                                    ],
                                    "temperature": float(args.temperature),
                                    "top_p": float(args.top_p),
                                    "max_completion_tokens": int(args.max_completion_tokens),
                                },
                            }
                            fout.write(json.dumps(req, ensure_ascii=False) + "\n")
                            stats.total_model_calls += 1

                out_path = _run_openai_batch(
                    client,
                    input_jsonl_path=input_path,
                    out_dir=attempt_dir,
                    completion_window=str(args.completion_window),
                    metadata={
                        "task": "nextqa_revise_pnp",
                        "model": str(args.model),
                        "round": str(round_idx),
                        "retry": str(retry_idx),
                        "n": str(len(pending)),
                    },
                    poll_interval_s=float(args.poll_interval_s),
                )

            # Consume outputs.
            next_pending: dict[str, str] = {}
            with out_path.open("r", encoding="utf-8") as f:
                for line in f:
                    obj = _safe_json_loads(line)
                    if not obj:
                        continue
                    cid = obj.get("custom_id")
                    if not isinstance(cid, str):
                        continue
                    sid = cid
                    if sid not in pending:
                        continue
                    st = states[sid]
                    status_code, content, finish_reason = _extract_assistant_content_from_batch_line(obj)
                    raw = content or ""
                    st.forced_base_user_text = st.base_user_text

                    if not reuse_existing_output:
                        # Log without embedding images.
                        with calls_log.open("a", encoding="utf-8") as flog:
                            flog.write(
                                json.dumps(
                                    {
                                        "ts": time.time(),
                                        "sample_id": sid,
                                        "qid": st.sample.qid,
                                        "video_id": st.sample.video_id,
                                        "round_idx": round_idx,
                                        "retry_idx": retry_idx,
                                        "status_code": status_code,
                                        "finish_reason": finish_reason,
                                        "seen_frames": st.seen_frames,
                                        "current_frames": st.current_frames,
                                        "candidate_next_frames": st.candidate_next_frames,
                                        "summary_in": st.summary_state,
                                        "user_text": st.base_user_text,
                                        "retry_feedback": pending[sid] or None,
                                        "raw_output": raw,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )

                    if status_code != 200:
                        stats.invalid_outputs += 1
                        st.terminated_reason = f"http_{status_code}"
                        if retry_idx < int(args.max_retries_per_round):
                            stats.total_retries += 1
                            next_pending[sid] = pnp._retry_feedback_text(
                                f"Request failed (HTTP {status_code}). Please follow the required output format exactly.",
                                force_answer=bool(args.force_final_answer and round_idx >= int(args.max_rounds)),
                            )
                            continue
                        if args.strict_actions:
                            stats.invalid_action_terminated += 1
                            st.terminated_invalid_action = True
                            continue
                        stats.fallback_frames_used += 1
                        st.next_frames = pnp._sample_uniform_indices(int(st.frame_count), 1)
                        continue

                    accepted, retry_feedback = _handle_output_one_attempt(
                        st,
                        raw=raw,
                        round_idx=round_idx,
                        retry_idx=retry_idx,
                        args=args,
                        rng=rng,
                        stats=stats,
                    )
                    if not accepted:
                        next_pending[sid] = retry_feedback or ""
                        continue

                    # Resolved this round: if answered/terminated, remove from active later.
                    if st.answer_letter is not None:
                        pass

            pending = next_pending

        # End of retries: update active set.
        next_active: list[str] = []
        for sid in active_ids:
            st = states[sid]
            if st.answer_letter is not None:
                continue
            if st.terminated_invalid_action and args.strict_actions:
                continue
            next_active.append(sid)
        active_ids = next_active
        stats.total_rounds += int(round_idx) * 0  # keep for compatibility; recomputed below

    # Forced final answers (same as vLLM runner).
    need_forced = [
        sid
        for sid, st in states.items()
        if bool(args.force_final_answer) and st.answer_letter is None and not (args.strict_actions and st.terminated_invalid_action)
    ]
    if need_forced:
        forced_dir = out_dir / "forced_answer"
        forced_dir.mkdir(parents=True, exist_ok=True)
        input_path = forced_dir / "batch_input.jsonl"
        forced_output_existing = forced_dir / "batch_output.jsonl"
        reuse_forced = bool(args.resume) and forced_output_existing.exists()

        if reuse_forced:
            out_path = forced_output_existing
            stats.total_model_calls += int(len(need_forced))
        else:
            if bool(args.resume) and input_path.exists():
                stats.total_model_calls += int(len(need_forced))
            else:
                with input_path.open("w", encoding="utf-8") as fout:
                    for sid in need_forced:
                        st = states[sid]
                        forced_user_text = (
                            f"{st.base_user_text}\n\n"
                            "Max rounds reached. Provide the final answer now using <summary>...</summary> then <answer>LETTER</answer>."
                        )
                        images: list[Image.Image] = []
                        if args.observation_mode != "caption":
                            images = pnp._extract_frames(str(st.sample.video_path), [int(i) for i in st.last_frames])
                        req = {
                            "custom_id": sid,
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": {
                                "model": str(args.model),
                                "messages": [
                                    {"role": "system", "content": system_prompt},
                                    {
                                        "role": "user",
                                        "content": _build_multimodal_user_content(
                                            forced_user_text,
                                            images,
                                            image_max_side=int(args.image_max_side),
                                            image_quality=int(args.image_jpeg_quality),
                                        ),
                                    },
                                ],
                                "temperature": float(args.temperature),
                                "top_p": float(args.top_p),
                                "max_completion_tokens": int(args.max_completion_tokens),
                            },
                        }
                        fout.write(json.dumps(req, ensure_ascii=False) + "\n")
                        stats.total_model_calls += 1

            out_path = _run_openai_batch(
                client,
                input_jsonl_path=input_path,
                out_dir=forced_dir,
                completion_window=str(args.completion_window),
                metadata={
                    "task": "nextqa_revise_pnp_forced_answer",
                    "model": str(args.model),
                    "n": str(len(need_forced)),
                },
                poll_interval_s=float(args.poll_interval_s),
            )
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = _safe_json_loads(line)
                if not obj:
                    continue
                sid = obj.get("custom_id")
                if not isinstance(sid, str) or sid not in states:
                    continue
                st = states[sid]
                status_code, content, _ = _extract_assistant_content_from_batch_line(obj)
                if status_code != 200:
                    continue
                answer = pnp._extract_tag(content or "", pnp._ANSWER_RE)
                if answer:
                    st.answer_letter = pnp._normalize_answer_letter(answer, len(st.sample.choices))
                    if st.answer_letter is not None:
                        st.answer_round = int(args.max_rounds) + 1

    # Score + write per-sample outputs.
    preds_path = out_dir / "predictions.jsonl"
    with preds_path.open("w", encoding="utf-8") as f:
        for sid, st in states.items():
            pred = st.answer_letter
            correct = bool(pred == st.gt_letter) if pred is not None else False
            if pred is None:
                stats.failed += 1
            elif correct:
                stats.correct += 1
            f.write(
                json.dumps(
                    {
                        "sample_id": sid,
                        "qid": st.sample.qid,
                        "video_id": st.sample.video_id,
                        "ground_truth": st.gt_letter,
                        "pred": pred,
                        "correct": correct,
                        "answer_round": st.answer_round,
                        "effective_rounds": st.effective_rounds,
                        "invalid_action_terminated": bool(st.terminated_invalid_action),
                        "terminated_reason": st.terminated_reason,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # Approximate total rounds: use answer_round if available, else max_rounds (like merged summaries).
    total_rounds = 0
    total_effective_rounds = 0
    for st in states.values():
        if st.answer_round is not None:
            total_rounds += min(int(st.answer_round), int(args.max_rounds))
        else:
            total_rounds += int(args.max_rounds)
        total_effective_rounds += int(st.effective_rounds)

    results = {
        "samples": len(states),
        "correct": stats.correct,
        "accuracy": stats.correct / max(1, len(states)),
        "total_rounds": total_rounds,
        "avg_rounds": total_rounds / max(1, len(states)),
        "total_effective_rounds": total_effective_rounds,
        "avg_effective_rounds": total_effective_rounds / max(1, len(states)),
        "failed": stats.failed,
        "invalid_outputs": stats.invalid_outputs,
        "invalid_action_terminated": stats.invalid_action_terminated,
        "total_retries": stats.total_retries,
        "total_model_calls": stats.total_model_calls,
        "fallback_frames_used": stats.fallback_frames_used,
    }

    summary = {
        "task": "openai_batch_nextqa_revise_pnp",
        "model": str(args.model),
        "dataset_csv": str(args.csv),
        "video_root": str(args.video_root),
        "map_json": str(args.map_json),
        "captions_dir": str(args.captions_dir) if args.captions_dir else None,
        "observation_mode": str(args.observation_mode),
        "caption_include": str(args.caption_include),
        "caption_max_chars": int(args.caption_max_chars),
        "max_samples": int(args.max_samples),
        "seed": int(args.seed),
        "max_rounds": int(args.max_rounds),
        "max_frames_per_round": int(args.max_frames_per_round),
        "max_retries_per_round": int(args.max_retries_per_round),
        "use_candidate_frames": bool(args.use_candidate_frames),
        "candidate_k": int(args.candidate_k) if args.candidate_k is not None else None,
        "use_candidate_frame_ids": bool(args.use_candidate_frame_ids),
        "require_candidate_frames": bool(args.require_candidate_frames),
        "hide_seen_frames_in_prompt": bool(args.hide_seen_frames_in_prompt),
        "strict_actions": bool(args.strict_actions),
        "force_final_answer": bool(args.force_final_answer),
        "answer_only_final_round": bool(args.answer_only_final_round),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "max_completion_tokens": int(args.max_completion_tokens),
        "image_max_side": int(args.image_max_side),
        "image_jpeg_quality": int(args.image_jpeg_quality),
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
