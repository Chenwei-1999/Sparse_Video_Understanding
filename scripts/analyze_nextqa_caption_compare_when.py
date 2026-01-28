#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
import tiktoken

# -----------------------------
# Shared helpers
# -----------------------------


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _super_type(nextqa_type: Optional[str]) -> str:
    s = str(nextqa_type or "").strip().upper()
    if not s:
        return "unknown"
    if s.startswith("D"):
        return "descriptive"
    if s.startswith("T"):
        return "temporal"
    if s.startswith("C"):
        return "causal"
    return "unknown"


def _load_val_type_map(dataset_csv: Path) -> dict[tuple[str, str], str]:
    df = pd.read_csv(dataset_csv)
    df["video"] = df["video"].astype(str)
    df["qid"] = df["qid"].astype(str)
    out: dict[tuple[str, str], str] = {}
    for r in df.itertuples(index=False):
        out[(str(r.video), str(r.qid))] = str(r.type)
    return out


_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def _normalize_answer_letter(answer_text: str, num_choices: int) -> Optional[str]:
    allowed = {chr(ord("A") + i) for i in range(max(0, int(num_choices)))}
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


# Caption block extraction (for reproducible token accounting)
_CAP_START_MARKERS = (
    "Video captions",
    "Captions (1fps timeline",
    "Captions for candidate unseen frame IDs",
    "Captions for shown frames",
    "Captions shown in this round",
    "Observed captions",
)
_CAP_STOP_PREFIXES = (
    "Current summary:",
    "Frames shown",
    "Shown frame",
    "This is the final round",
    "Return",
)
_CAP_LINE_RE = re.compile(r"^\s*(?:ID\s+\d+\s*\(t[^)]*\):|\d+\s*(?:s)?\s*:)\s*.+")


def _extract_caption_text(user_text: str) -> str:
    if not user_text:
        return ""
    out: list[str] = []
    in_cap = False
    for line in str(user_text).splitlines():
        if any(m in line for m in _CAP_START_MARKERS):
            in_cap = True
            continue
        if not in_cap:
            continue

        if line.startswith(_CAP_STOP_PREFIXES):
            in_cap = False
            continue
        if re.match(
            r"^(Round\s+\d+\s*/|Question:|Options:|Total frames|Seen frames:|Candidate unseen|In <frames>|To answer,|Current summary:)",
            line,
        ):
            in_cap = False
            continue

        if _CAP_LINE_RE.match(line):
            out.append(line.strip())
            continue

        # Allow indented continuation lines (rare; mostly single-line captions).
        if out and (line.startswith(" ") or line.startswith("\t")):
            out.append(line.strip())

    return "\n".join(out)


def _extract_caption_json_map(user_text: str) -> str:
    """Extract a JSON object block like:

    {
      "0": "...",
      "2": "..."
    }

    Used by the VideoAgent official-style caption prompts.
    """
    if not user_text:
        return ""
    m = re.search(r"\n(\{\n\s*\"\d+\"\s*:\s*)", str(user_text))
    if not m:
        return ""
    start = m.start(1)

    s = str(user_text)
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == "\"":
                in_str = False
            continue

        if ch == "\"":
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                block = s[start : i + 1]
                try:
                    obj = json.loads(block)
                except Exception:
                    return ""
                if not isinstance(obj, dict):
                    return ""
                # Heuristic: numeric-string keys.
                if not any(str(k).isdigit() for k in obj.keys()):
                    return ""
                return block
    return ""


@dataclass
class SettingReport:
    name: str
    n_samples: int
    acc_by_super: dict[str, float]
    correct_by_super: dict[str, int]
    n_by_super: dict[str, int]
    avg_total_tokens: float | None = None
    avg_caption_tokens: float | None = None
    avg_non_caption_tokens: float | None = None
    notes: str = ""


def _format_pct(x: float) -> str:
    return f"{x*100:.2f}%"


# -----------------------------
# Qwen2.5-VL-7B (local logs; token estimate via tiktoken + OpenAI vision heuristic)
# -----------------------------


def _qwen_parse_caption_only_correct(logs: list[Path]) -> dict[tuple[str, str], bool]:
    correct: dict[tuple[str, str], bool] = {}
    for p in logs:
        for obj in _iter_jsonl(p):
            key = (str(obj.get("video_id") or ""), str(obj.get("qid") or ""))
            if not key[0] or not key[1]:
                continue
            correct[key] = bool(obj.get("correct"))
    return correct


def _qwen_parse_revise_correct(logs: list[Path]) -> dict[tuple[str, str], bool]:
    per_key: dict[tuple[str, str], dict[str, Any]] = {}
    for p in logs:
        for obj in _iter_jsonl(p):
            if "event" in obj:
                continue
            key = (str(obj.get("video_id") or ""), str(obj.get("qid") or ""))
            if not key[0] or not key[1]:
                continue
            rec = per_key.get(key)
            if rec is None:
                rec = {"gt_idx": obj.get("ground_truth_idx"), "pred": None, "num_choices": len(obj.get("choices") or [])}
                per_key[key] = rec

            if rec["gt_idx"] is None and obj.get("ground_truth_idx") is not None:
                rec["gt_idx"] = obj.get("ground_truth_idx")

            if rec["pred"] is not None:
                continue

            raw = str(obj.get("raw_output") or "")
            m = _ANSWER_TAG_RE.search(raw)
            if not m:
                continue
            pred = _normalize_answer_letter(m.group(1), rec["num_choices"] or 5)
            if pred is not None:
                rec["pred"] = pred

    correct: dict[tuple[str, str], bool] = {}
    for key, rec in per_key.items():
        gt_idx = rec.get("gt_idx")
        gt = chr(ord("A") + int(gt_idx)) if gt_idx is not None else None
        correct[key] = bool(gt and rec.get("pred") and (gt == rec.get("pred")))
    return correct


def _qwen_estimate_tokens_from_logs(
    logs: list[Path],
    *,
    enc: tiktoken.Encoding,
    image_tokens_per_image: int = 70,
    chat_overhead_tokens: int = 12,
    has_system_prompt: bool,
) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str], int]]:
    """Return (total_tokens_by_key, caption_tokens_by_key) summed over all calls."""

    sys_cache: dict[str, int] = {}
    total_by_key: dict[tuple[str, str], int] = defaultdict(int)
    caption_by_key: dict[tuple[str, str], int] = defaultdict(int)

    for p in logs:
        for obj in _iter_jsonl(p):
            if "event" in obj:
                continue
            key = (str(obj.get("video_id") or ""), str(obj.get("qid") or ""))
            if not key[0] or not key[1]:
                continue

            user_text = str(obj.get("user_text") or "")
            raw_out = str(obj.get("raw_output") or "")

            in_text = chat_overhead_tokens + len(enc.encode(user_text)) if user_text else chat_overhead_tokens
            if has_system_prompt:
                sys = str(obj.get("system_prompt") or "")
                if sys in sys_cache:
                    in_text += sys_cache[sys]
                else:
                    sys_cache[sys] = len(enc.encode(sys)) if sys else 0
                    in_text += sys_cache[sys]

            img_count = user_text.count("<image>")
            in_image = img_count * int(image_tokens_per_image)
            out_tokens = len(enc.encode(raw_out)) if raw_out else 0

            total_by_key[key] += int(in_text + in_image + out_tokens)

            cap_text = _extract_caption_text(user_text)
            if cap_text:
                caption_by_key[key] += len(enc.encode(cap_text))

    return total_by_key, caption_by_key


def _qwen_parse_videoagent_final_correct(logs: list[Path]) -> dict[tuple[str, str], bool]:
    correct: dict[tuple[str, str], bool] = {}
    for p in logs:
        for obj in _iter_jsonl(p):
            if obj.get("event") != "final":
                continue
            key = (str(obj.get("video_id") or ""), str(obj.get("qid") or ""))
            if not key[0] or not key[1]:
                continue
            correct[key] = bool(obj.get("correct"))
    return correct


def _qwen_estimate_tokens_from_videoagent_logs(
    logs: list[Path],
    *,
    enc: tiktoken.Encoding,
    chat_overhead_tokens: int = 12,
) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str], int]]:
    """Estimate tokens for VideoAgent-style JSON event logs."""

    SYSTEM_PROMPT_ANSWER = (
        "You are a multiple-choice video QA assistant.\n"
        "Return ONLY a JSON object with exactly one key: 'final_answer'.\n"
        "- 'final_answer' must be exactly one of: A, B, C, D, E.\n"
        "You MUST always output one letter even if guessing.\n"
        "Example: {\"final_answer\":\"A\"}\n"
        "Do not output any other text."
    )
    SYSTEM_PROMPT_CONFIDENCE = (
        "You are a helpful assistant.\n"
        "Return ONLY a JSON object with exactly one key: 'confidence'.\n"
        "- 'confidence' must be an integer 1, 2, or 3.\n"
        "Example: {\"confidence\":3}\n"
        "Do not output any other text."
    )
    SYSTEM_PROMPT_REQUEST = (
        "You are a video QA agent that decides what additional evidence to look for.\n"
        "Return ONLY a JSON object with exactly one key: 'frame_descriptions'.\n"
        "- 'frame_descriptions' must be a list of objects.\n"
        "- Each object must have: 'segment_id' (int) and 'description' (string).\n"
        "Do not output any other text."
    )

    sys_by_event = {
        "answer": SYSTEM_PROMPT_ANSWER,
        "confidence": SYSTEM_PROMPT_CONFIDENCE,
        "request": SYSTEM_PROMPT_REQUEST,
    }

    total_by_key: dict[tuple[str, str], int] = defaultdict(int)
    caption_by_key: dict[tuple[str, str], int] = defaultdict(int)

    for p in logs:
        for obj in _iter_jsonl(p):
            event = str(obj.get("event") or "")
            if event not in sys_by_event:
                continue
            key = (str(obj.get("video_id") or ""), str(obj.get("qid") or ""))
            if not key[0] or not key[1]:
                continue

            user_text = str(obj.get("user_text") or "")
            raw_out = str(obj.get("raw_output") or "")
            sys = sys_by_event[event]

            in_text = chat_overhead_tokens + len(enc.encode(user_text)) + len(enc.encode(sys))
            out_tokens = len(enc.encode(raw_out)) if raw_out else 0
            total_by_key[key] += int(in_text + out_tokens)

            cap_text = _extract_caption_text(user_text) or _extract_caption_json_map(user_text)
            if cap_text:
                caption_by_key[key] += len(enc.encode(cap_text))

    return total_by_key, caption_by_key


def _report_accuracy_by_super(
    correct_map: dict[tuple[str, str], bool],
    val_type_map: dict[tuple[str, str], str],
    *,
    keys: Optional[Iterable[tuple[str, str]]] = None,
) -> tuple[dict[str, float], dict[str, int], dict[str, int]]:
    n_by = Counter()
    c_by = Counter()

    it = keys if keys is not None else val_type_map.keys()
    for key in it:
        nextqa_type = val_type_map.get(key)
        if nextqa_type is None:
            continue
        s = _super_type(nextqa_type)
        n_by[s] += 1
        if correct_map.get(key, False):
            c_by[s] += 1

    acc_by = {k: (c_by[k] / n_by[k]) if n_by[k] else 0.0 for k in sorted(n_by.keys())}
    return acc_by, dict(c_by), dict(n_by)


# -----------------------------
# GPT-5.1 (Batch logs with usage)
# -----------------------------


def _gpt_parse_caption_only_correct(output_dir: Path) -> dict[tuple[str, str], bool]:
    samples_path = output_dir / "samples.jsonl"
    out_path = output_dir / "batch_output.jsonl"

    sid_to_key_gt: dict[str, tuple[tuple[str, str], str]] = {}
    for obj in _iter_jsonl(samples_path):
        sid = str(obj.get("sample_id") or "")
        if not sid:
            continue
        key = (str(obj.get("video_id") or ""), str(obj.get("qid") or ""))
        gt = str(obj.get("ground_truth") or "").strip().upper()
        sid_to_key_gt[sid] = (key, gt)

    pred_by_sid: dict[str, str] = {}
    letter_re = re.compile(r"\b([A-E])\b", re.IGNORECASE)
    for obj in _iter_jsonl(out_path):
        cid = str(obj.get("custom_id") or "")
        if not cid:
            continue
        if ":" in cid:
            cid = cid.split(":", 1)[1]
        body = (obj.get("response") or {}).get("body") or {}
        choices = body.get("choices") or []
        if not choices:
            continue
        content = str(((choices[0] or {}).get("message") or {}).get("content") or "").strip().upper()
        m = letter_re.search(content)
        if m:
            pred_by_sid[cid] = m.group(1).upper()

    correct: dict[tuple[str, str], bool] = {}
    for sid, (key, gt) in sid_to_key_gt.items():
        pred = pred_by_sid.get(sid)
        correct[key] = bool(pred and gt and pred == gt)
    return correct


def _gpt_usage_tokens_caption_only(output_dir: Path) -> dict[str, int]:
    out_path = output_dir / "batch_output.jsonl"
    total_by_sid: dict[str, int] = defaultdict(int)
    for obj in _iter_jsonl(out_path):
        cid = str(obj.get("custom_id") or "")
        if not cid:
            continue
        if ":" in cid:
            cid = cid.split(":", 1)[1]
        body = (obj.get("response") or {}).get("body") or {}
        usage = body.get("usage") or {}
        total_by_sid[cid] += int(usage.get("total_tokens") or 0)
    return total_by_sid


def _gpt_usage_tokens_revise(output_dir: Path) -> dict[str, int]:
    total_by_sid: dict[str, int] = defaultdict(int)
    for fp in glob.glob(str(output_dir / "rounds" / "r*" / "try*" / "batch_output.jsonl")):
        for obj in _iter_jsonl(Path(fp)):
            cid = str(obj.get("custom_id") or "")
            if not cid:
                continue
            body = (obj.get("response") or {}).get("body") or {}
            usage = body.get("usage") or {}
            total_by_sid[cid] += int(usage.get("total_tokens") or 0)
    return total_by_sid


def _gpt_caption_tokens_from_calls(calls_path: Path, enc: tiktoken.Encoding) -> dict[str, int]:
    cap_by_sid: dict[str, int] = defaultdict(int)
    for obj in _iter_jsonl(calls_path):
        sid = str(obj.get("sample_id") or "")
        if not sid:
            continue
        cap_text = _extract_caption_text(str(obj.get("user_text") or ""))
        if not cap_text:
            continue
        cap_by_sid[sid] += len(enc.encode(cap_text))
    return cap_by_sid


def _gpt_caption_tokens_from_caption_only_batch_input(batch_input: Path, enc: tiktoken.Encoding) -> dict[str, int]:
    cap_by_sid: dict[str, int] = {}
    for obj in _iter_jsonl(batch_input):
        cid = str(obj.get("custom_id") or "")
        if not cid:
            continue
        if ":" in cid:
            cid = cid.split(":", 1)[1]
        body = obj.get("body") or {}
        msgs = body.get("messages") or []
        user = ""
        for m in msgs:
            if (m or {}).get("role") == "user":
                user = str((m or {}).get("content") or "")
                break
        cap_text = _extract_caption_text(user)
        cap_by_sid[cid] = len(enc.encode(cap_text)) if cap_text else 0
    return cap_by_sid


def _gpt_usage_tokens_from_calls_jsonl(calls_jsonl: Path) -> dict[str, int]:
    total_by_sid: dict[str, int] = defaultdict(int)
    for obj in _iter_jsonl(calls_jsonl):
        sid = str(obj.get("sample_id") or "")
        if not sid:
            continue
        usage = obj.get("usage") or {}
        if not isinstance(usage, dict):
            continue
        total_by_sid[sid] += int(usage.get("total_tokens") or 0)
    return total_by_sid


def _gpt_caption_tokens_from_batch_inputs(batch_inputs: list[Path], enc: tiktoken.Encoding) -> dict[str, int]:
    cap_by_sid: dict[str, int] = defaultdict(int)
    for batch_input in batch_inputs:
        for obj in _iter_jsonl(batch_input):
            cid = str(obj.get("custom_id") or "")
            if not cid:
                continue
            if ":" in cid:
                cid = cid.split(":", 1)[1]

            body = obj.get("body") or {}
            msgs = body.get("messages") or []
            user = ""
            for m in msgs:
                if (m or {}).get("role") == "user":
                    user = str((m or {}).get("content") or "")
                    break
            cap_text = _extract_caption_text(user)
            if cap_text:
                cap_by_sid[cid] += len(enc.encode(cap_text))
    return cap_by_sid


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-csv",
        type=Path,
        default=Path("/shares/hlw3876/chenwei/NExT-QA/nextqa/val.csv"),
    )
    ap.add_argument("--encoding", default="o200k_base")
    ap.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = ap.parse_args()

    val_type_map = _load_val_type_map(args.dataset_csv)
    enc = tiktoken.get_encoding(args.encoding)

    # ---- Qwen report ----
    qwen_reports: list[SettingReport] = []

    qwen_caption_only_logs = [
        Path(f"outputs/2026-01-25/nextqa_caption_compare_qwen2p5vl7b/caption_only_val_tp1/log.shard{i}of4.jsonl")
        for i in range(4)
    ]
    qwen_frames_logs = [
        Path(f"debug_prompt_logs/nextqa_val_full_qwen2p5vl7b_ids_tp1.shard{i}of4.jsonl") for i in range(4)
    ]
    qwen_plus_logs = [
        Path(
            f"outputs/2026-01-25/nextqa_caption_compare_qwen2p5vl7b/revise_plus_caption_val_tp1_ts/log.shard{i}of4.jsonl"
        )
        for i in range(4)
    ]
    qwen_caption_revise_logs = [
        Path(
            f"outputs/2026-01-25/nextqa_caption_compare_qwen2p5vl7b/caption_revise_val_tp1_ts/log.shard{i}of4.jsonl"
        )
        for i in range(4)
    ]
    qwen_videoagent_official_logs = [
        Path(
            f"outputs/2026-01-25/nextqa_caption_compare_qwen2p5vl7b/videoagent_officialstyle_caption_val_tp1/log.shard{i}of4.jsonl"
        )
        for i in range(4)
    ]

    # Correctness
    qwen_cap_correct = _qwen_parse_caption_only_correct(qwen_caption_only_logs)
    qwen_frames_correct = _qwen_parse_revise_correct(qwen_frames_logs)
    qwen_plus_correct = _qwen_parse_revise_correct(qwen_plus_logs)
    qwen_caprev_correct = _qwen_parse_revise_correct(qwen_caption_revise_logs)
    qwen_va_official_correct = _qwen_parse_videoagent_final_correct(qwen_videoagent_official_logs)

    # Tokens (estimated)
    qwen_cap_tot, qwen_cap_cap = _qwen_estimate_tokens_from_logs(qwen_caption_only_logs, enc=enc, has_system_prompt=False)
    qwen_frames_tot, qwen_frames_cap = _qwen_estimate_tokens_from_logs(qwen_frames_logs, enc=enc, has_system_prompt=True)
    qwen_plus_tot, qwen_plus_cap = _qwen_estimate_tokens_from_logs(qwen_plus_logs, enc=enc, has_system_prompt=True)
    qwen_caprev_tot, qwen_caprev_cap = _qwen_estimate_tokens_from_logs(
        qwen_caption_revise_logs, enc=enc, has_system_prompt=True
    )
    qwen_va_official_tot, qwen_va_official_cap = _qwen_estimate_tokens_from_videoagent_logs(
        qwen_videoagent_official_logs, enc=enc
    )

    for name, correct_map, tot_map, cap_map, note in [
        ("Caption-only (single-turn)", qwen_cap_correct, qwen_cap_tot, qwen_cap_cap, "token est. via tiktoken"),
        ("REVISE (frames)", qwen_frames_correct, qwen_frames_tot, qwen_frames_cap, "token est. via tiktoken + image=70"),
        ("REVISE + caption", qwen_plus_correct, qwen_plus_tot, qwen_plus_cap, "token est. via tiktoken + image=70"),
        ("Caption-only REVISE (multi-turn)", qwen_caprev_correct, qwen_caprev_tot, qwen_caprev_cap, "token est. via tiktoken"),
        (
            "VideoAgent (official-style prompts)",
            qwen_va_official_correct,
            qwen_va_official_tot,
            qwen_va_official_cap,
            "token est. via tiktoken (JSON prompts)",
        ),
    ]:
        acc_by, c_by, n_by = _report_accuracy_by_super(correct_map, val_type_map)
        n = sum(n_by.values())
        avg_total = sum(tot_map.values()) / n if n else 0.0
        avg_cap = sum(cap_map.values()) / n if n else 0.0
        qwen_reports.append(
            SettingReport(
                name=name,
                n_samples=n,
                acc_by_super=acc_by,
                correct_by_super=c_by,
                n_by_super=n_by,
                avg_total_tokens=avg_total,
                avg_caption_tokens=avg_cap,
                avg_non_caption_tokens=avg_total - avg_cap,
                notes=note,
            )
        )

    # ---- GPT report ----
    gpt_reports: list[SettingReport] = []
    gpt_root = Path("outputs/2026-01-25/nextqa_caption_compare_gpt5p1_n1000")
    gpt_videoagent_dir = Path("outputs/2026-01-26/nextqa_caption_compare_gpt5p1_n1000/videoagent_caption")

    # Correctness
    gpt_cap_correct = _gpt_parse_caption_only_correct(gpt_root / "caption_only")

    def _gpt_correct_from_predictions(pred_path: Path) -> dict[tuple[str, str], bool]:
        out: dict[tuple[str, str], bool] = {}
        for obj in _iter_jsonl(pred_path):
            key = (str(obj.get("video_id") or ""), str(obj.get("qid") or ""))
            if not key[0] or not key[1]:
                continue
            out[key] = bool(obj.get("correct"))
        return out

    gpt_frames_correct = _gpt_correct_from_predictions(gpt_root / "revise_frames" / "predictions.jsonl")
    gpt_plus_correct = _gpt_correct_from_predictions(gpt_root / "revise_plus_caption" / "predictions.jsonl")
    gpt_caprev_correct = _gpt_correct_from_predictions(gpt_root / "caption_revise" / "predictions.jsonl")

    # Tokens (API usage totals)
    gpt_cap_tot_by_sid = _gpt_usage_tokens_caption_only(gpt_root / "caption_only")
    gpt_frames_tot_by_sid = _gpt_usage_tokens_revise(gpt_root / "revise_frames")
    gpt_plus_tot_by_sid = _gpt_usage_tokens_revise(gpt_root / "revise_plus_caption")
    gpt_caprev_tot_by_sid = _gpt_usage_tokens_revise(gpt_root / "caption_revise")

    # Caption tokens (tiktoken on extracted caption blocks)
    gpt_cap_cap_by_sid = _gpt_caption_tokens_from_caption_only_batch_input(
        gpt_root / "caption_only" / "batch_input.jsonl", enc
    )
    gpt_frames_cap_by_sid = _gpt_caption_tokens_from_calls(gpt_root / "revise_frames" / "calls.jsonl", enc)
    gpt_plus_cap_by_sid = _gpt_caption_tokens_from_calls(gpt_root / "revise_plus_caption" / "calls.jsonl", enc)
    gpt_caprev_cap_by_sid = _gpt_caption_tokens_from_calls(gpt_root / "caption_revise" / "calls.jsonl", enc)

    # Map sample_id -> (video_id,qid) for averaging token totals per sample (use predictions.jsonl)
    def _sid_count(pred_path: Path) -> int:
        return sum(1 for _ in _iter_jsonl(pred_path))

    gpt_n = _sid_count(gpt_root / "revise_frames" / "predictions.jsonl")

    def _avg(d: dict[str, int], n: int) -> float:
        return (sum(d.values()) / n) if n else 0.0

    # For caption-only, n is fixed 1000 as well.
    gpt_cap_acc, gpt_cap_c, gpt_cap_n = _report_accuracy_by_super(gpt_cap_correct, val_type_map, keys=gpt_cap_correct.keys())
    gpt_reports.append(
        SettingReport(
            name="Caption-only (single-turn)",
            n_samples=gpt_n,
            acc_by_super=gpt_cap_acc,
            correct_by_super=gpt_cap_c,
            n_by_super=gpt_cap_n,
            avg_total_tokens=_avg(gpt_cap_tot_by_sid, gpt_n),
            avg_caption_tokens=_avg(gpt_cap_cap_by_sid, gpt_n),
            avg_non_caption_tokens=_avg(gpt_cap_tot_by_sid, gpt_n) - _avg(gpt_cap_cap_by_sid, gpt_n),
            notes="total tokens from OpenAI usage; caption tokens via tiktoken",
        )
    )

    for name, correct_map, tot_by_sid, cap_by_sid in [
        ("REVISE (frames)", gpt_frames_correct, gpt_frames_tot_by_sid, gpt_frames_cap_by_sid),
        ("REVISE + caption", gpt_plus_correct, gpt_plus_tot_by_sid, gpt_plus_cap_by_sid),
        ("Caption-only REVISE (multi-turn)", gpt_caprev_correct, gpt_caprev_tot_by_sid, gpt_caprev_cap_by_sid),
    ]:
        acc_by, c_by, n_by = _report_accuracy_by_super(correct_map, val_type_map, keys=correct_map.keys())
        gpt_reports.append(
            SettingReport(
                name=name,
                n_samples=gpt_n,
                acc_by_super=acc_by,
                correct_by_super=c_by,
                n_by_super=n_by,
                avg_total_tokens=_avg(tot_by_sid, gpt_n),
                avg_caption_tokens=_avg(cap_by_sid, gpt_n),
                avg_non_caption_tokens=_avg(tot_by_sid, gpt_n) - _avg(cap_by_sid, gpt_n),
                notes="total tokens from OpenAI usage; caption tokens via tiktoken",
            )
        )

    if gpt_videoagent_dir.is_dir():
        gpt_va_correct = _gpt_correct_from_predictions(gpt_videoagent_dir / "predictions.jsonl")
        gpt_va_tot_by_sid = _gpt_usage_tokens_from_calls_jsonl(gpt_videoagent_dir / "calls.jsonl")
        batch_inputs: list[Path] = []
        for fp in glob.glob(str(gpt_videoagent_dir / "rounds" / "r*" / "answer" / "batch_input.jsonl")):
            batch_inputs.append(Path(fp))
        for fp in glob.glob(str(gpt_videoagent_dir / "rounds" / "r*" / "request" / "batch_input.jsonl")):
            batch_inputs.append(Path(fp))
        gpt_va_cap_by_sid = _gpt_caption_tokens_from_batch_inputs(batch_inputs, enc)
        acc_by, c_by, n_by = _report_accuracy_by_super(gpt_va_correct, val_type_map, keys=gpt_va_correct.keys())
        gpt_reports.append(
            SettingReport(
                name="VideoAgent (caption retrieval)",
                n_samples=gpt_n,
                acc_by_super=acc_by,
                correct_by_super=c_by,
                n_by_super=n_by,
                avg_total_tokens=_avg(gpt_va_tot_by_sid, gpt_n),
                avg_caption_tokens=_avg(gpt_va_cap_by_sid, gpt_n),
                avg_non_caption_tokens=_avg(gpt_va_tot_by_sid, gpt_n) - _avg(gpt_va_cap_by_sid, gpt_n),
                notes="total tokens from OpenAI usage; caption tokens via tiktoken",
            )
        )

    if args.json:
        print(
            json.dumps(
                {
                    "qwen": [r.__dict__ for r in qwen_reports],
                    "gpt": [r.__dict__ for r in gpt_reports],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    # Markdown output (copy/paste into reports or rebuttal)
    def _md_table(reports: list[SettingReport]) -> str:
        lines: list[str] = []
        lines.append(
            "| setting | Acc (D/T/C) | ΔAcc (vs REVISE frames) | Tok_total | Tok_caption | Tok_noncap | caption% |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")

        base = None
        for r in reports:
            if r.name == "REVISE (frames)":
                base = r
                break
        base_acc = base.acc_by_super if base else {}

        def acc_triplet(a: dict[str, float]) -> str:
            return f"{a.get('descriptive', 0):.3f}/{a.get('temporal', 0):.3f}/{a.get('causal', 0):.3f}"

        for r in reports:
            if base and r is not base:
                d = {k: (r.acc_by_super.get(k, 0.0) - base_acc.get(k, 0.0)) for k in ["descriptive", "temporal", "causal"]}
                delta = f"{d['descriptive']:+.3f}/{d['temporal']:+.3f}/{d['causal']:+.3f}"
            else:
                delta = "—"

            tok_total = float(r.avg_total_tokens or 0.0)
            tok_cap = float(r.avg_caption_tokens or 0.0)
            tok_non = float(r.avg_non_caption_tokens or 0.0)
            cap_pct = (tok_cap / tok_total) if tok_total else 0.0

            lines.append(
                f"| {r.name} | {acc_triplet(r.acc_by_super)} | {delta} | {tok_total:.1f} | {tok_cap:.1f} | {tok_non:.1f} | {_format_pct(cap_pct)} |"
            )

        return "\n".join(lines)

    print("## Qwen2.5-VL-7B (val=4996)")
    print(_md_table(qwen_reports))
    print("")
    print("## GPT-5.1 (val subset n=1000, seed=0)")
    print(_md_table(gpt_reports))


if __name__ == "__main__":
    main()
