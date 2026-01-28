#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
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


def _load_correct_map(predictions_jsonl: Path) -> dict[tuple[str, str], bool]:
    out: dict[tuple[str, str], bool] = {}
    for obj in _iter_jsonl(predictions_jsonl):
        key = (str(obj.get("video_id") or ""), str(obj.get("qid") or ""))
        if not key[0] or not key[1]:
            continue
        out[key] = bool(obj.get("correct"))
    return out


_RX_CAUSAL = re.compile(r"\bwhy\b|\bin order to\b|\bso that\b|\bbecause\b", re.IGNORECASE)
_RX_TEMPORAL = re.compile(
    r"\b(before|after|then|first|next|later|earlier|beginning|at the beginning|end|at the end|while|when|as soon as|"
    r"during|until)\b",
    re.IGNORECASE,
)
_RX_COUNT = re.compile(r"\bhow many\b|\bnumber of\b|\bcount\b", re.IGNORECASE)
_RX_TEXT = re.compile(r"\b(written|says?|read|word|letter|number|text|sign)\b", re.IGNORECASE)
_RX_SPATIAL = re.compile(
    r"\b(where|left|right|front|in front|behind|next to|on top|under|between|near|far|closest|farthest)\b",
    re.IGNORECASE,
)
_RX_COLOR = re.compile(r"\b(colou?r)\b", re.IGNORECASE)
_RX_ATTR = re.compile(r"\b(wear|wearing|held|holding|carry|carrying|eat|eating|drink|drinking)\b", re.IGNORECASE)


def _failure_mode(question: str) -> str:
    q = str(question or "").strip()
    if not q:
        return "other/unknown"

    # Single-label heuristic with precedence.
    if _RX_CAUSAL.search(q):
        return "causal/intent"
    if _RX_TEMPORAL.search(q):
        return "temporal/order"
    if _RX_COUNT.search(q):
        return "descriptive/counting"
    if _RX_TEXT.search(q):
        return "fine-grained/text-ocr"
    if _RX_SPATIAL.search(q):
        return "fine-grained/spatial"
    if _RX_COLOR.search(q):
        return "fine-grained/color"
    if _RX_ATTR.search(q):
        return "descriptive/attribute"
    return "other"


@dataclass
class WinLossRow:
    mode: str
    n: int = 0
    wins: int = 0
    losses: int = 0

    @property
    def net(self) -> int:
        return self.wins - self.losses

    @property
    def changed(self) -> int:
        return self.wins + self.losses

    @property
    def win_rate(self) -> float:
        return (self.wins / self.changed) if self.changed else 0.0


def _win_loss_table(
    *,
    keys: Iterable[tuple[str, str]],
    mode_by_key: dict[tuple[str, str], str],
    baseline: dict[tuple[str, str], bool],
    variant: dict[tuple[str, str], bool],
) -> dict[str, WinLossRow]:
    rows: dict[str, WinLossRow] = defaultdict(lambda: WinLossRow(mode=""))
    for key in keys:
        mode = mode_by_key.get(key, "other/unknown")
        if rows[mode].mode == "":
            rows[mode].mode = mode
        rows[mode].n += 1

        b = bool(baseline.get(key, False))
        v = bool(variant.get(key, False))
        if v and not b:
            rows[mode].wins += 1
        elif b and not v:
            rows[mode].losses += 1
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-csv",
        type=Path,
        default=Path("/shares/hlw3876/chenwei/NExT-QA/nextqa/val.csv"),
        help="NExT-QA CSV (val.csv). Used for question text and qid/video_id keys.",
    )
    ap.add_argument(
        "--baseline-pred",
        type=Path,
        default=Path("outputs/2026-01-25/nextqa_caption_compare_gpt5p1_n1000/revise_frames/predictions.jsonl"),
        help="Baseline predictions.jsonl (e.g., REVISE frames).",
    )
    ap.add_argument(
        "--variant-pred",
        type=Path,
        action="append",
        default=[
            Path("outputs/2026-01-25/nextqa_caption_compare_gpt5p1_n1000/revise_plus_caption/predictions.jsonl"),
            Path("outputs/2026-01-26/nextqa_caption_compare_gpt5p1_n1000/videoagent_caption/predictions.jsonl"),
        ],
        help="Variant predictions.jsonl (repeatable).",
    )
    ap.add_argument(
        "--variant-name",
        type=str,
        action="append",
        default=["REVISE + caption", "VideoAgent (caption retrieval)"],
        help="Display names for variants (repeatable; must match --variant-pred count).",
    )
    args = ap.parse_args()

    if len(args.variant_pred) != len(args.variant_name):
        raise ValueError("--variant-pred and --variant-name must have the same length.")

    df = pd.read_csv(args.dataset_csv)
    df["video"] = df["video"].astype(str)
    df["qid"] = df["qid"].astype(str)
    mode_by_key: dict[tuple[str, str], str] = {}
    for r in df.itertuples(index=False):
        key = (str(r.video), str(r.qid))
        mode_by_key[key] = _failure_mode(getattr(r, "question", ""))

    baseline = _load_correct_map(args.baseline_pred)
    variants = [(name, _load_correct_map(Path(p))) for name, p in zip(args.variant_name, args.variant_pred, strict=False)]

    # Use the key intersection to avoid silently counting missing samples.
    keys = set(baseline.keys())
    for _, v in variants:
        keys &= set(v.keys())

    # Ensure every key has a mode (CSV-backed); otherwise they land in other/unknown.
    all_modes = sorted(set(mode_by_key.get(k, "other/unknown") for k in keys))

    tables: dict[str, dict[str, WinLossRow]] = {}
    for name, v in variants:
        tables[name] = _win_loss_table(keys=keys, mode_by_key=mode_by_key, baseline=baseline, variant=v)

    # Markdown table.
    header = ["failure_mode", "n"]
    for name, _ in variants:
        header += [f"{name} win", f"{name} loss", f"{name} net"]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] + ["---:"] * (len(header) - 1)) + " |"]

    def _get_row(t: dict[str, WinLossRow], mode: str) -> WinLossRow:
        r = t.get(mode)
        return r if r is not None else WinLossRow(mode=mode)

    for mode in all_modes:
        n = _get_row(tables[variants[0][0]], mode).n if variants else 0
        row = [mode, str(n)]
        for name, _ in variants:
            r = _get_row(tables[name], mode)
            row += [str(r.wins), str(r.losses), f"{r.net:+d}"]
        lines.append("| " + " | ".join(row) + " |")

    # Totals row
    totals = ["TOTAL", str(len(keys))]
    for name, _ in variants:
        total_w = sum(r.wins for r in tables[name].values())
        total_l = sum(r.losses for r in tables[name].values())
        totals += [str(total_w), str(total_l), f"{(total_w - total_l):+d}"]
    lines.append("| " + " | ".join(totals) + " |")

    print("\n".join(lines))


if __name__ == "__main__":
    main()

