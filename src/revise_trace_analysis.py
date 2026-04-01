from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional

from examples.revise.pnp_utils import (
    ANSWER_RE,
    FRAMES_RE,
    SUMMARY_RE,
    THINK_RE,
    contains_banned_example,
    extract_tag,
    is_placeholder,
    normalize_answer_letter,
    parse_int_list,
    summary_has_ohrpu,
)

_FRAME_COUNT_RE = re.compile(r"Total frames L = (\d+)")
_UNCERTAINTY_TOKENS = (
    "unclear",
    "unknown",
    "uncertain",
    "not sure",
    "still need",
    "still unclear",
    "may",
    "might",
    "possibly",
    "cannot tell",
    "can't tell",
)


def _extract_frame_count(user_text: str) -> int:
    match = _FRAME_COUNT_RE.search(user_text or "")
    if match is None:
        return 0
    try:
        return int(match.group(1))
    except Exception:
        return 0


def _summary_has_stale_boilerplate(summary_text: str, *, seen_count: int) -> bool:
    if not summary_text or seen_count <= 0:
        return False
    s = re.sub(r"\s+", " ", str(summary_text)).strip().lower()
    return any(
        phrase in s
        for phrase in (
            "has not seen any frames yet",
            "has not seen any caption yet",
            "has not seen any captions yet",
            "no frames yet",
            "no captions yet",
        )
    )


def _frames_has_range_syntax(frames_text: str) -> bool:
    return bool(frames_text and re.search(r"\d+\s*[-–—]\s*\d+", frames_text))


def _similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a or "", b or "").ratio()


def _count_token_hits(text: str, tokens: tuple[str, ...]) -> int:
    text = (text or "").lower()
    return sum(int(token in text) for token in tokens)


def _extract_field(summary_text: str, field: str) -> str:
    summary_text = summary_text or ""
    pattern = re.compile(r"\b([POHUR])\s*:\s*", re.IGNORECASE)
    matches = list(pattern.finditer(summary_text))
    if not matches:
        return ""
    spans: list[tuple[str, int, int]] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(summary_text)
        spans.append((match.group(1).upper(), start, end))
    for key, start, end in spans:
        if key == field.upper():
            return summary_text[start:end].strip(" ;\n\t")
    return ""


@dataclass
class TraceTurn:
    sample_id: str
    qid: str
    round_idx: int
    retry_idx: int
    forced_answer: bool
    ground_truth_idx: int
    seen_frames: list[int]
    current_frames: list[int]
    requested_raw_frames: list[int]
    requested_mapped_frames: list[int]
    summary_in: str
    summary_out: str
    raw_output: str
    answer_letter: Optional[str]
    frame_count: int
    valid_summary: bool
    valid_answer: bool
    valid_request: bool
    invalid_call: bool
    think_present: bool
    stale_summary: bool
    duplicate_request: bool
    request_repeats_current: bool


@dataclass
class TraceSample:
    sample_id: str
    qid: str
    question: str
    turns: list[TraceTurn]
    correct: int
    predicted_answer: Optional[str]
    answer_round: Optional[int]
    forced_answer_used: bool


def parse_turn(obj: dict[str, Any]) -> TraceTurn:
    raw_output = str(obj.get("raw_output") or "")
    summary_out = extract_tag(raw_output, SUMMARY_RE) or ""
    answer_letter = normalize_answer_letter(
        extract_tag(raw_output, ANSWER_RE) or "",
        len(obj.get("choices") or []),
    )
    frames_text = extract_tag(raw_output, FRAMES_RE) or ""
    frame_count = _extract_frame_count(str(obj.get("user_text") or ""))

    seen_frames = [int(i) for i in (obj.get("seen_frames") or [])]
    current_frames = [int(i) for i in (obj.get("current_frames") or [])]
    requested_raw_frames = [int(i) for i in (obj.get("requested_raw_frames") or [])]
    requested_mapped_frames = [int(i) for i in (obj.get("requested_mapped_frames") or [])]
    requested = requested_mapped_frames or requested_raw_frames

    valid_summary = bool(
        summary_out
        and not is_placeholder(summary_out)
        and not contains_banned_example(summary_out)
        and summary_has_ohrpu(summary_out)
        and not _summary_has_stale_boilerplate(summary_out, seen_count=len(seen_frames))
    )

    valid_answer = bool(answer_letter and valid_summary)
    valid_request = False
    duplicate_request = False
    request_repeats_current = False
    if requested and frames_text and not _frames_has_range_syntax(frames_text):
        request_repeats_current = requested == current_frames and bool(current_frames)
        valid_new = [i for i in requested if 0 <= i < frame_count and i not in seen_frames]
        duplicate_request = bool(requested) and not bool(valid_new)
        valid_request = bool(valid_new) and valid_summary

    think_present = extract_tag(raw_output, THINK_RE) is not None
    stale_summary = valid_summary and _similarity(str(obj.get("summary_in") or ""), summary_out) > 0.98
    invalid_call = not valid_answer and not valid_request

    return TraceTurn(
        sample_id=str(obj.get("sample_id") or ""),
        qid=str(obj.get("qid") or ""),
        round_idx=int(obj.get("round_idx") or 0),
        retry_idx=int(obj.get("retry_idx") or 0),
        forced_answer=bool(obj.get("forced_answer", False)),
        ground_truth_idx=int(obj.get("ground_truth_idx") or -1),
        seen_frames=seen_frames,
        current_frames=current_frames,
        requested_raw_frames=requested_raw_frames,
        requested_mapped_frames=requested_mapped_frames,
        summary_in=str(obj.get("summary_in") or ""),
        summary_out=summary_out,
        raw_output=raw_output,
        answer_letter=answer_letter,
        frame_count=frame_count,
        valid_summary=valid_summary,
        valid_answer=valid_answer,
        valid_request=valid_request,
        invalid_call=invalid_call,
        think_present=think_present,
        stale_summary=stale_summary,
        duplicate_request=duplicate_request,
        request_repeats_current=request_repeats_current,
    )


def load_revise_traces(log_path: str | Path) -> list[TraceSample]:
    traces: list[TraceSample] = []
    current_objs: list[dict[str, Any]] = []

    def finalize(objs: list[dict[str, Any]]) -> Optional[TraceSample]:
        if not objs:
            return None
        turns = [parse_turn(obj) for obj in objs]
        predicted_answer: Optional[str] = None
        answer_round: Optional[int] = None
        correct = 0
        for turn in turns:
            if not turn.valid_answer:
                continue
            predicted_answer = turn.answer_letter
            answer_round = turn.round_idx
            pred_idx = ord(predicted_answer) - ord("A")
            correct = int(turn.ground_truth_idx >= 0 and pred_idx == turn.ground_truth_idx)
            break
        return TraceSample(
            sample_id=turns[0].sample_id,
            qid=turns[0].qid,
            question=str(objs[0].get("question") or ""),
            turns=turns,
            correct=correct,
            predicted_answer=predicted_answer,
            answer_round=answer_round,
            forced_answer_used=any(turn.forced_answer for turn in turns),
        )

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            round_idx = int(obj.get("round_idx") or 0)
            forced = bool(obj.get("forced_answer", False))
            retry_idx = int(obj.get("retry_idx") or 0)
            if current_objs and round_idx == 1 and retry_idx == 0 and not forced:
                sample = finalize(current_objs)
                if sample is not None:
                    traces.append(sample)
                current_objs = [obj]
            else:
                current_objs.append(obj)
    sample = finalize(current_objs)
    if sample is not None:
        traces.append(sample)
    return traces


def trace_text(sample: TraceSample) -> str:
    lines = [f"Question: {sample.question}"]
    for turn in sample.turns:
        if turn.summary_out:
            lines.append(f"Round {turn.round_idx} summary: {turn.summary_out}")
        if turn.answer_letter:
            lines.append(f"Round {turn.round_idx} answer: {turn.answer_letter}")
        elif turn.requested_mapped_frames:
            lines.append(
                "Round "
                f"{turn.round_idx} request: {', '.join(str(i) for i in turn.requested_mapped_frames)}"
            )
    return "\n".join(lines)


def build_trace_features(sample: TraceSample) -> dict[str, Any]:
    turns = sample.turns
    num_turns = len(turns)
    invalid_calls = sum(int(turn.invalid_call) for turn in turns)
    retries = sum(int(turn.retry_idx > 0) for turn in turns)
    stale_summary_calls = sum(int(turn.stale_summary) for turn in turns)
    duplicate_request_calls = sum(int(turn.duplicate_request or turn.request_repeats_current) for turn in turns)
    valid_request_calls = sum(int(turn.valid_request) for turn in turns)
    valid_answer_calls = sum(int(turn.valid_answer) for turn in turns)
    think_calls = sum(int(turn.think_present) for turn in turns)
    mean_summary_similarity = 0.0
    if num_turns:
        mean_summary_similarity = sum(_similarity(turn.summary_in, turn.summary_out) for turn in turns) / num_turns

    final_turn = next((turn for turn in turns if turn.valid_answer), turns[-1])
    final_summary = final_turn.summary_out
    final_uncertainty = _extract_field(final_summary, "U")
    final_reason = _extract_field(final_summary, "R")
    final_observation = _extract_field(final_summary, "O")
    uncertainty_hits = _count_token_hits(final_summary, _UNCERTAINTY_TOKENS)
    answer_with_uncertainty = int(bool(final_turn.valid_answer and uncertainty_hits > 0 and "no remaining ambiguity" not in final_summary.lower()))

    heuristic_risk = (
        1.4 * (invalid_calls / max(1, num_turns))
        + 1.0 * (retries / max(1, num_turns))
        + 0.9 * (duplicate_request_calls / max(1, num_turns))
        + 0.8 * (stale_summary_calls / max(1, num_turns))
        + 0.8 * answer_with_uncertainty
        + 0.5 * int(sample.forced_answer_used)
        + 0.3 * (1.0 - (valid_request_calls + valid_answer_calls) / max(1, num_turns))
    )

    return {
        "sample_id": sample.sample_id,
        "qid": sample.qid,
        "num_turns": num_turns,
        "answer_round": sample.answer_round or 0,
        "correct": sample.correct,
        "incorrect": 1 - int(sample.correct),
        "forced_answer_used": int(sample.forced_answer_used),
        "invalid_calls": invalid_calls,
        "invalid_call_frac": invalid_calls / max(1, num_turns),
        "retries": retries,
        "retry_frac": retries / max(1, num_turns),
        "stale_summary_calls": stale_summary_calls,
        "stale_summary_frac": stale_summary_calls / max(1, num_turns),
        "duplicate_request_calls": duplicate_request_calls,
        "duplicate_request_frac": duplicate_request_calls / max(1, num_turns),
        "valid_request_calls": valid_request_calls,
        "valid_answer_calls": valid_answer_calls,
        "think_calls": think_calls,
        "mean_summary_similarity": mean_summary_similarity,
        "final_summary_len": len(final_summary),
        "final_uncertainty_len": len(final_uncertainty),
        "final_reason_len": len(final_reason),
        "final_observation_len": len(final_observation),
        "final_uncertainty_hits": uncertainty_hits,
        "answer_with_uncertainty": answer_with_uncertainty,
        "heuristic_risk": heuristic_risk,
        "trace_text": trace_text(sample),
    }
