from __future__ import annotations

import re
from typing import Any

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_FRAMES_RE = re.compile(r"<frames>(.*?)</frames>", re.DOTALL | re.IGNORECASE)
_SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

_STRICT_SELECT_RE = re.compile(
    r"^\s*<think>.*?</think>\s*<summary>.*?</summary>\s*<frames>\s*\d+(?:\s*,\s*\d+)*\s*</frames>\s*$",
    re.DOTALL | re.IGNORECASE,
)
_STRICT_ANSWER_RE = re.compile(
    r"^\s*<think>.*?</think>\s*<summary>.*?</summary>\s*<answer>\s*([A-E])\s*</answer>\s*$",
    re.DOTALL | re.IGNORECASE,
)


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _is_placeholder(text: str) -> bool:
    t = _collapse_ws(text).lower()
    if not t:
        return True
    # Treat ellipses anywhere as placeholders.
    if "..." in t or "…" in t:
        return True
    if t in {"...", "…", "none", "n/a", "na", "null", "unknown", "unsure", "uncertain"}:
        return True
    if re.fullmatch(r"[.·•…]+", t):
        return True
    # Too short: often a placeholder like "ok", "idk", etc.
    alnum = re.findall(r"[a-z0-9]+", t)
    if len(alnum) <= 1 and len(t) <= 6:
        return True
    return False


def _text_quality(text: str, *, min_words: int) -> float:
    if text is None:
        return 0.0
    if _is_placeholder(text):
        return 0.0
    words = re.findall(r"[A-Za-z0-9]+", str(text))
    if len(words) < min_words:
        return 0.5
    return 1.0


def _extract_tag(text: str, pattern: re.Pattern[str]) -> str | None:
    matches = list(pattern.finditer(text or ""))
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


def _summary_quality(summary_text: str, num_choices: int) -> float:
    """Return 1.0 for well-formed, non-placeholder O/H/R/P/U summary; else 0.0."""
    if summary_text is None:
        return 0.0
    summary = _collapse_ws(summary_text)
    if _is_placeholder(summary):
        return 0.0

    # Require O/H/R/P/U labels.
    for key in ["O", "H", "R", "P", "U"]:
        if re.search(rf"\b{key}\s*:", summary, re.IGNORECASE) is None:
            return 0.0

    # H: must contain exactly one option letter.
    h_match = re.search(r"\bH\s*:\s*([^;]+)", summary, re.IGNORECASE)
    if not h_match:
        return 0.0
    h_field = h_match.group(1)
    letters = re.findall(r"\b([A-E])\b", h_field, re.IGNORECASE)
    unique_letters = {l.upper() for l in letters}
    if len(unique_letters) != 1:
        return 0.0
    if _normalize_answer_letter(next(iter(unique_letters)), num_choices) is None:
        return 0.0

    # P: should include at least one frame index.
    p_match = re.search(r"\bP\s*:\s*([^;]+)", summary, re.IGNORECASE)
    if not p_match or not re.findall(r"\d+", p_match.group(1)):
        return 0.0

    # Avoid placeholder/empty content in O/R/U.
    for key in ["O", "R", "U"]:
        m = re.search(rf"\b{key}\s*:\s*([^;]+)", summary, re.IGNORECASE)
        if not m or _is_placeholder(m.group(1)):
            return 0.0
    return 1.0


def compute_score(
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any] | None = None,
    reward_weights: dict[str, float] | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """Reward = accuracy + a small conditional formatting bonus.

    The formatting bonus/penalties are only applied when the final answer is correct, to avoid the
    model trading off accuracy for protocol/format hacking.
    """

    choices = ground_truth.get("choices") or []
    answer_idx = ground_truth.get("answer_idx")
    if answer_idx is None:
        return {
            "score": 0.0,
            "answer_correct": 0.0,
            "format_valid": 0.0,
            "format_strict": 0.0,
            "think_quality": 0.0,
            "summary_quality": 0.0,
            "summary_present": 0.0,
        }

    num_choices = len(choices) if isinstance(choices, list) else 0
    correct_letter = chr(ord("A") + int(answer_idx))

    extra_info = extra_info or {}
    revise_info = extra_info.get("revise", {}) if isinstance(extra_info, dict) else {}
    final_response = str(revise_info.get("last_response") or solution_str)

    answer_text = _extract_tag(final_response, _ANSWER_RE)
    pred_letter = _normalize_answer_letter(answer_text or "", num_choices)
    format_valid = float(pred_letter is not None)
    answer_correct = float(pred_letter == correct_letter) if pred_letter is not None else 0.0

    has_answer = answer_text is not None
    if has_answer:
        format_strict = float(_STRICT_ANSWER_RE.match(final_response) is not None)
    else:
        format_strict = float(_STRICT_SELECT_RE.match(final_response) is not None)

    think_text = _extract_tag(final_response, _THINK_RE) or ""
    think_quality = float(_text_quality(think_text, min_words=4))

    summary_text = str(revise_info.get("summary") or _extract_tag(final_response, _SUMMARY_RE) or "")
    summary_present = float(bool(summary_text))
    summary_quality = float(_summary_quality(summary_text, num_choices)) if summary_text else 0.0

    weights = reward_weights or {}
    w_answer = float(weights.get("answer", 1.0))
    w_format = float(weights.get("format", 0.05))
    w_think = float(weights.get("think", 0.02))
    w_summary = float(weights.get("summary", 0.05))
    w_invalid_penalty = float(weights.get("invalid_penalty", 0.0))
    w_all_seen_penalty = float(weights.get("frames_all_seen_penalty", 0.0))

    bonus = (w_format * format_strict) + (w_think * think_quality) + (w_summary * summary_quality)

    invalid_outputs = int(revise_info.get("invalid_outputs") or 0)
    frames_all_seen = int(revise_info.get("frames_all_seen") or 0)
    penalty = (w_invalid_penalty * float(invalid_outputs)) + (w_all_seen_penalty * float(frames_all_seen))

    score = (w_answer * answer_correct) + (answer_correct * bonus) - (answer_correct * penalty)
    score = float(max(0.0, score))

    return {
        "score": float(score),
        "answer_correct": float(answer_correct),
        "format_valid": float(format_valid),
        "format_strict": float(format_strict),
        "think_quality": float(think_quality),
        "summary_present": float(summary_present),
        "summary_quality": float(summary_quality),
        "invalid_outputs": float(invalid_outputs),
        "frames_all_seen": float(frames_all_seen),
        "pred_letter": float(ord(pred_letter) - ord("A")) if pred_letter is not None else -1.0,
    }
