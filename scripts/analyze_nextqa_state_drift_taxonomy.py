#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd


_SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.IGNORECASE | re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


def _iter_jsonl(paths: list[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield obj


def _normalize_answer_letter(text: str, num_choices: int) -> Optional[str]:
    allowed = {chr(ord("A") + i) for i in range(max(0, int(num_choices)))}
    if not allowed:
        allowed = {"A", "B", "C", "D", "E"}
    t = (text or "").strip().upper()
    if len(t) == 1 and t in allowed:
        return t
    m = re.search(r"\b([A-E])\b", t)
    if m and m.group(1).upper() in allowed:
        return m.group(1).upper()
    m = re.search(r"([A-E])", t)
    if m and m.group(1).upper() in allowed:
        return m.group(1).upper()
    return None


_OHRPU_FIELD_RE = re.compile(r"\b([POHUR])\s*:\s*(.*?)(?=\b[POHUR]\s*:|$)", re.IGNORECASE | re.DOTALL)


def _parse_ohrpu(summary: str) -> dict[str, str]:
    out: dict[str, str] = {}
    s = str(summary or "").strip()
    for m in _OHRPU_FIELD_RE.finditer(s):
        key = m.group(1).upper()
        val = m.group(2).strip().strip(";")
        out[key] = re.sub(r"\s+", " ", val).strip()
    return out


_UNCERTAIN_WORDS = {
    "unclear",
    "unknown",
    "not sure",
    "unsure",
    "cannot tell",
    "can't tell",
    "hard to tell",
    "need more",
    "insufficient",
    "ambiguous",
    "uncertain",
}


def _uncertainty_score(text: str) -> int:
    t = (text or "").lower()
    return sum(1 for w in _UNCERTAIN_WORDS if w in t)


def _drift_distance(a: str, b: str) -> float:
    """1 - SequenceMatcher ratio, in [0,1]."""
    a = (a or "").strip()
    b = (b or "").strip()
    if not a and not b:
        return 0.0
    return 1.0 - SequenceMatcher(None, a, b).ratio()


def _format_float(x: float, nd: int = 4) -> str:
    if math.isnan(x) or math.isinf(x):
        return "nan"
    return f"{x:.{nd}f}"


@dataclass
class SampleTrajectory:
    sample_id: str
    video_id: str
    qid: str
    question: str
    choices: list[str]
    gt_letter: Optional[str]
    # per-attempt logs
    attempts: list[dict[str, Any]]

    def final_attempt_with_answer(self) -> Optional[dict[str, Any]]:
        for obj in reversed(self.attempts):
            raw = str(obj.get("raw_output") or "")
            if _ANSWER_RE.search(raw):
                return obj
        return None

    def final_answer(self) -> Optional[str]:
        obj = self.final_attempt_with_answer()
        if obj is None:
            return None
        raw = str(obj.get("raw_output") or "")
        m = _ANSWER_RE.search(raw)
        if not m:
            return None
        return _normalize_answer_letter(m.group(1), len(self.choices) or 5)

    def final_summary(self) -> str:
        obj = self.final_attempt_with_answer() or (self.attempts[-1] if self.attempts else {})
        raw = str(obj.get("raw_output") or "")
        m = _SUMMARY_RE.search(raw)
        return (m.group(1).strip() if m else "").strip()

    def final_round_idx(self) -> int:
        obj = self.final_attempt_with_answer() or (self.attempts[-1] if self.attempts else {})
        return int(obj.get("round_idx") or 0)

    def retries(self) -> int:
        return sum(1 for obj in self.attempts if int(obj.get("retry_idx") or 0) > 0)

    def unique_rounds(self) -> int:
        return len({int(obj.get("round_idx") or 0) for obj in self.attempts if int(obj.get("round_idx") or 0) > 0})

    def summaries_by_round(self) -> list[str]:
        """Use summary_in snapshots as state trajectory; append final summary_out."""
        by_round: dict[int, str] = {}
        for obj in self.attempts:
            r = int(obj.get("round_idx") or 0)
            if r <= 0:
                continue
            if r not in by_round:
                by_round[r] = str(obj.get("summary_in") or "").strip()
        out = [by_round[k] for k in sorted(by_round.keys())]
        final = self.final_summary()
        if final and (not out or final != out[-1]):
            out.append(final)
        return out


def _load_nextqa_types(csv_path: Path) -> dict[tuple[str, str], str]:
    df = pd.read_csv(csv_path)
    df["video"] = df["video"].astype(str)
    df["qid"] = df["qid"].astype(str)
    out: dict[tuple[str, str], str] = {}
    for r in df.itertuples(index=False):
        out[(str(r.video), str(r.qid))] = str(getattr(r, "type", "")).strip().upper()
    return out


def _type_bucket(t: str) -> str:
    if t.startswith("D"):
        return "D"
    if t.startswith("T"):
        return "T"
    if t.startswith("C"):
        return "C"
    return "unknown"


def _taxonomy_for_sample(
    traj: SampleTrajectory,
    *,
    max_rounds: Optional[int],
) -> str:
    pred = traj.final_answer()
    gt = traj.gt_letter
    if pred is None or gt is None:
        return "format_or_missing_answer"
    is_correct = pred == gt
    if is_correct:
        return "correct"

    # Early stop (only meaningful when max_rounds provided).
    if max_rounds is not None and traj.final_round_idx() > 0 and traj.final_round_idx() < max_rounds:
        return "premature_stop"

    # Format-heavy: many retries and still wrong.
    if traj.retries() >= 3:
        return "format_heavy_wrong"

    final_fields = _parse_ohrpu(traj.final_summary())
    u = final_fields.get("U", "")
    r = final_fields.get("R", "")
    unc = _uncertainty_score(u) + _uncertainty_score(r)

    summaries = traj.summaries_by_round()
    drifts = [_drift_distance(a, b) for a, b in zip(summaries, summaries[1:], strict=False)]
    avg_drift = sum(drifts) / len(drifts) if drifts else 0.0
    max_drift = max(drifts) if drifts else 0.0

    # Heuristic buckets (intended for trend analysis; not ground-truth).
    if unc >= 1:
        return "likely_miss_key_moment_or_insufficient_evidence"
    if max_drift >= 0.55 or avg_drift >= 0.35:
        return "likely_state_drift_or_bad_update"
    return "other_wrong"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--logs",
        type=Path,
        action="append",
        required=True,
        help="One or more log.jsonl files (repeatable).",
    )
    ap.add_argument(
        "--dataset-csv",
        type=Path,
        default=Path("/shares/hlw3876/chenwei/NExT-QA/nextqa/val.csv"),
        help="NExT-QA val.csv for D/T/C type labels.",
    )
    ap.add_argument(
        "--merged-json",
        type=Path,
        default=None,
        help="Optional merged.json from the run to pull config like max_rounds.",
    )
    ap.add_argument(
        "--output-md",
        type=Path,
        required=True,
        help="Write markdown report here.",
    )
    args = ap.parse_args()

    max_rounds: Optional[int] = None
    if args.merged_json and args.merged_json.exists():
        try:
            merged = json.loads(args.merged_json.read_text(encoding="utf-8"))
            max_rounds = int(merged.get("max_rounds") or 0) or None
        except Exception:
            max_rounds = None

    type_map = _load_nextqa_types(args.dataset_csv)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for obj in _iter_jsonl(args.logs):
        sid = str(obj.get("sample_id") or "")
        if not sid:
            continue
        grouped[sid].append(obj)

    trajectories: list[SampleTrajectory] = []
    for sid, rows in grouped.items():
        rows.sort(key=lambda r: (int(r.get("round_idx") or 0), int(r.get("retry_idx") or 0)))
        first = rows[0]
        video_id = str(first.get("video_id") or "")
        qid = str(first.get("qid") or "")
        question = str(first.get("question") or "")
        choices = list(first.get("choices") or [])
        gt_idx = first.get("ground_truth_idx")
        gt_letter = None
        try:
            gt_letter = chr(ord("A") + int(gt_idx))
        except Exception:
            gt_letter = None
        trajectories.append(
            SampleTrajectory(
                sample_id=sid,
                video_id=video_id,
                qid=qid,
                question=question,
                choices=[str(c) for c in choices],
                gt_letter=gt_letter,
                attempts=rows,
            )
        )

    trajectories.sort(key=lambda t: (t.video_id, t.qid, t.sample_id))

    # Aggregate stats.
    tax = Counter()
    by_type = defaultdict(Counter)
    drift_by_bucket: dict[str, list[float]] = defaultdict(list)
    retries_by_bucket: dict[str, list[int]] = defaultdict(list)
    rounds_by_bucket: dict[str, list[int]] = defaultdict(list)

    for traj in trajectories:
        bucket = _taxonomy_for_sample(traj, max_rounds=max_rounds)
        tax[bucket] += 1
        t = _type_bucket(type_map.get((traj.video_id, traj.qid), ""))
        by_type[t][bucket] += 1

        summaries = traj.summaries_by_round()
        drifts = [_drift_distance(a, b) for a, b in zip(summaries, summaries[1:], strict=False)]
        drift_by_bucket[bucket].append(sum(drifts) / len(drifts) if drifts else 0.0)
        retries_by_bucket[bucket].append(traj.retries())
        rounds_by_bucket[bucket].append(traj.unique_rounds())

    n = len(trajectories)
    n_correct = tax.get("correct", 0)
    acc = n_correct / n if n else 0.0

    lines: list[str] = []
    lines.append("# NExT-QA state drift / compression loss analysis (heuristic)")
    lines.append("")
    lines.append("This report is computed from REVISE-style prompt logs (no frame-level annotations).")
    lines.append(
        "It provides a *heuristic* failure taxonomy to help decide whether errors are dominated by "
        "(a) missing key moment, (b) summary-state drift/compression, or (c) premature stopping."
    )
    lines.append("")
    if max_rounds is not None:
        lines.append(f"- Config max_rounds: {max_rounds}")
    lines.append(f"- Samples parsed: {n}")
    lines.append(f"- Accuracy (from parsed <answer>): {_format_float(acc, 4)}")
    lines.append("")

    lines.append("## Taxonomy breakdown")
    lines.append("| Bucket | N | % | avg_drift | avg_retries | avg_unique_rounds |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for bucket, cnt in tax.most_common():
        drift = sum(drift_by_bucket[bucket]) / len(drift_by_bucket[bucket]) if drift_by_bucket[bucket] else 0.0
        avg_retries = sum(retries_by_bucket[bucket]) / len(retries_by_bucket[bucket]) if retries_by_bucket[bucket] else 0.0
        avg_rounds = sum(rounds_by_bucket[bucket]) / len(rounds_by_bucket[bucket]) if rounds_by_bucket[bucket] else 0.0
        lines.append(
            f"| {bucket} | {cnt} | {_format_float(cnt/n if n else 0.0, 4)} | "
            f"{_format_float(drift, 4)} | {_format_float(avg_retries, 2)} | {_format_float(avg_rounds, 2)} |"
        )
    lines.append("")

    lines.append("## By question type (D/T/C)")
    for t in ["D", "T", "C", "unknown"]:
        if t not in by_type:
            continue
        lines.append(f"### {t}")
        sub = by_type[t]
        total_t = sum(sub.values())
        lines.append("| Bucket | N | % |")
        lines.append("|---|---:|---:|")
        for bucket, cnt in sub.most_common():
            lines.append(f"| {bucket} | {cnt} | {_format_float(cnt/total_t if total_t else 0.0, 4)} |")
        lines.append("")

    lines.append("## How to use these buckets (actionable next steps)")
    lines.append("- `likely_miss_key_moment_or_insufficient_evidence`: improve evidence acquisition (candidate proposal, retrieval, coverage).")
    lines.append("- `likely_state_drift_or_bad_update`: strengthen state representation (structured state, evidence anchors, faithfulness checks).")
    lines.append("- `premature_stop`: tighten stop gating (require confidence threshold + stability), penalize early stop.")
    lines.append("- `format_heavy_wrong` / `format_or_missing_answer`: increase format reward / stricter decoding / retries.")
    lines.append("")

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()

