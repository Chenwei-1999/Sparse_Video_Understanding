# Copyright 2026
# Licensed under the Apache License, Version 2.0

"""Safer EAGER-style reward for REVISE video QA.

This version avoids rewarding summary "answer_text" substring matches (which can be trivially
hacked by listing all options). Instead, it rewards the hypothesis letter in the summary's
`H:` field when it matches the ground-truth option letter.
"""

from __future__ import annotations

import re
from typing import Any

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL | re.IGNORECASE)
_HYP_RE = re.compile(r"\bH\s*:\s*([A-E])\b", re.IGNORECASE)


def _extract_tag(text: str, pattern: re.Pattern[str]) -> str | None:
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def _normalize_answer_letter(answer_text: str, num_choices: int) -> str | None:
    allowed = {chr(ord("A") + i) for i in range(max(0, num_choices))}
    if not allowed:
        allowed = {"A", "B", "C", "D", "E"}

    candidate = answer_text.strip().upper()
    if candidate in allowed:
        return candidate

    match = re.search(r"\b([A-E])\b", candidate)
    if match:
        letter = match.group(1).upper()
        return letter if letter in allowed else None

    match = re.search(r"([A-E])", candidate)
    if match:
        letter = match.group(1).upper()
        return letter if letter in allowed else None
    return None


def _extract_summary_hyp_letter(summary_text: str, num_choices: int) -> str | None:
    if not summary_text:
        return None
    match = _HYP_RE.search(summary_text)
    if match:
        return _normalize_answer_letter(match.group(1), num_choices)
    # Fall back to any single letter (avoid rewarding lists like "A/B/C/D/E").
    match = re.search(r"\b([A-E])\b", summary_text, re.IGNORECASE)
    if match:
        return _normalize_answer_letter(match.group(1), num_choices)
    return None


def compute_score(
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any] | None = None,
    reward_weights: dict[str, float] | None = None,
    stop_round_threshold: int = 2,
    **kwargs: Any,
) -> dict[str, float]:
    """Compute a conservative EAGER-style reward.

    Reward components:
    - answer_correct: 1 if final <answer> matches ground-truth option index.
    - format_valid: 1 if <answer> is present and parsable.
    - summary_hyp_correct: 1 if <summary> contains a single-letter hypothesis matching the label.
    - stop_early: 1 if answer is correct and num_rounds <= stop_round_threshold.

    The summary bonus is based on `H:` letter to avoid substring hacks.
    """

    extra_info = extra_info or {}
    revise_info = extra_info.get("revise", {}) if isinstance(extra_info, dict) else {}

    choices = ground_truth.get("choices") or []
    answer_idx = ground_truth.get("answer_idx")
    if answer_idx is None:
        return {
            "score": 0.0,
            "answer_correct": 0.0,
            "format_valid": 0.0,
            "summary_hyp_correct": 0.0,
            "stop_early": 0.0,
        }

    num_choices = len(choices) if isinstance(choices, list) else 0
    correct_letter = chr(ord("A") + int(answer_idx))

    pred_answer_text = _extract_tag(solution_str, _ANSWER_RE)
    pred_letter = _normalize_answer_letter(pred_answer_text or "", num_choices)
    format_valid = float(pred_letter is not None)
    answer_correct = float(pred_letter == correct_letter) if pred_letter is not None else 0.0

    summary_text = revise_info.get("summary") or _extract_tag(solution_str, _SUMMARY_RE) or ""
    hyp_letter = _extract_summary_hyp_letter(str(summary_text), num_choices)
    summary_hyp_correct = float(hyp_letter == correct_letter) if hyp_letter is not None else 0.0

    num_rounds = revise_info.get("num_rounds")
    if num_rounds is None:
        num_rounds = extra_info.get("num_turns")
    stop_early = float(answer_correct > 0.0 and num_rounds is not None and int(num_rounds) <= int(stop_round_threshold))

    # Default weights: keep close to accuracy-only while adding small shaping terms.
    weights = reward_weights or {}
    w_answer = float(weights.get("answer", 1.0))
    w_format = float(weights.get("format", 0.1))
    w_sum = float(weights.get("sum", 0.2))
    w_stop = float(weights.get("stop", 0.2))

    score = w_answer * answer_correct + w_format * format_valid + w_sum * summary_hyp_correct + w_stop * stop_early

    return {
        "score": float(score),
        "answer_correct": float(answer_correct),
        "format_valid": float(format_valid),
        "summary_hyp_correct": float(summary_hyp_correct),
        "stop_early": float(stop_early),
        "pred_letter": float(ord(pred_letter) - ord("A")) if pred_letter is not None else -1.0,
        "hyp_letter": float(ord(hyp_letter) - ord("A")) if hyp_letter is not None else -1.0,
        "num_rounds": float(num_rounds) if num_rounds is not None else -1.0,
    }

