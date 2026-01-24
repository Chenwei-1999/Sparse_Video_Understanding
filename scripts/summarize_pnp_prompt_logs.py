#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Optional


_LETTER_RE = re.compile(r"\b([A-Z])\b")


def _normalize_letter(text: str) -> Optional[str]:
    if not text:
        return None
    s = str(text).strip().upper()
    if len(s) == 1 and "A" <= s <= "Z":
        return s
    m = _LETTER_RE.search(s)
    return m.group(1) if m else None


def _bucket_error(err: str) -> str:
    s = str(err or "")
    if "Private video" in s:
        return "youtube_private"
    if "Video unavailable" in s:
        return "youtube_unavailable"
    if "Sign in" in s or "cookies" in s:
        return "youtube_auth_required"
    if "yt-dlp failed" in s:
        return "yt_dlp_failed_other"
    if "video_probe_failed" in s:
        return "video_probe_failed"
    return "other"


@dataclass
class SampleAgg:
    failed: bool = False
    error_bucket: str = ""
    answer_gt: str = ""
    last_answer: str = ""
    max_round: int = 0
    model_calls: int = 0
    think_rounds: int = 0


def summarize(paths: list[str]) -> dict[str, Any]:
    per_sample: dict[str, SampleAgg] = {}

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                sid = obj.get("sample_id")
                if not sid:
                    continue

                rec = per_sample.get(sid)
                if rec is None:
                    rec = SampleAgg()
                    per_sample[sid] = rec

                if "error" in obj:
                    rec.failed = True
                    rec.error_bucket = _bucket_error(str(obj.get("error") or ""))
                    continue

                rec.model_calls += 1

                gt = _normalize_letter(str(obj.get("answer_gt") or ""))
                if gt:
                    rec.answer_gt = gt

                raw = str(obj.get("raw_output") or "").lower()
                if "<think>" in raw and "</think>" in raw:
                    rec.think_rounds += 1

                try:
                    r = int(obj.get("round_idx") or 0)
                except Exception:
                    r = 0
                if r > rec.max_round:
                    rec.max_round = r

                ans = _normalize_letter(str(obj.get("answer_letter") or ""))
                if ans:
                    rec.last_answer = ans

    total = len(per_sample)
    failed = sum(1 for v in per_sample.values() if v.failed)
    nonfailed = total - failed
    answered = sum(1 for v in per_sample.values() if v.last_answer)
    answered_nonfailed = sum(1 for v in per_sample.values() if v.last_answer and not v.failed)
    correct = sum(1 for v in per_sample.values() if v.last_answer and v.answer_gt and v.last_answer == v.answer_gt)

    sum_rounds_answered = sum(v.max_round for v in per_sample.values() if v.last_answer)
    sum_calls_answered = sum(v.model_calls for v in per_sample.values() if v.last_answer)
    sum_think_rounds = sum(v.think_rounds for v in per_sample.values())

    error_buckets: dict[str, int] = {}
    for v in per_sample.values():
        if not v.failed:
            continue
        error_buckets[v.error_bucket or "other"] = error_buckets.get(v.error_bucket or "other", 0) + 1

    return {
        "files": paths,
        "total_samples": total,
        "failed_samples": failed,
        "nonfailed_samples": nonfailed,
        "answered_samples": answered,
        "answered_nonfailed_samples": answered_nonfailed,
        "correct": correct,
        "acc_all": (correct / total) if total else 0.0,
        "acc_nonfailed": (correct / nonfailed) if nonfailed else 0.0,
        "acc_answered": (correct / answered) if answered else 0.0,
        "answer_rate_all": (answered / total) if total else 0.0,
        "answer_rate_nonfailed": (answered_nonfailed / nonfailed) if nonfailed else 0.0,
        "avg_rounds_answered": (sum_rounds_answered / answered) if answered else 0.0,
        "avg_model_calls_answered": (sum_calls_answered / answered) if answered else 0.0,
        "think_rounds_total": sum_think_rounds,
        "error_buckets": dict(sorted(error_buckets.items(), key=lambda x: (-x[1], x[0]))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="", help="glob pattern for jsonl files")
    ap.add_argument("--paths", nargs="*", default=[], help="explicit jsonl file paths")
    ap.add_argument("-o", "--output", default="", help="write JSON summary to this path")
    args = ap.parse_args()

    paths: list[str] = []
    if args.glob:
        paths.extend(sorted(glob.glob(args.glob)))
    if args.paths:
        paths.extend(args.paths)

    paths = [p for p in paths if p and os.path.exists(p)]
    if not paths:
        raise SystemExit("No input jsonl files found.")

    out = summarize(paths)
    text = json.dumps(out, indent=2, ensure_ascii=False)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()

