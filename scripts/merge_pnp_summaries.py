#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


def _as_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _merge_results(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "samples": 0,
        "correct": 0,
        "total_rounds": 0,
        "total_effective_rounds": 0,
        "failed": 0,
        "invalid_outputs": 0,
        "invalid_action_terminated": 0,
        "total_retries": 0,
        "total_model_calls": 0,
        "fallback_frames_used": 0,
        "prompt_log_lines": 0,
        "prompt_log_bytes": 0,
    }
    shard_elapsed: list[float] = []
    weighted_rounds_sum = 0.0
    weighted_effective_rounds_sum = 0.0

    shard_infos: list[dict[str, Any]] = []
    for s in summaries:
        r = s.get("results") or {}
        shard_samples = _as_int(r.get("samples"))
        shard_avg_rounds = _as_float(r.get("avg_rounds"))
        shard_avg_eff_rounds = _as_float(r.get("avg_effective_rounds"))

        totals["samples"] += shard_samples
        totals["correct"] += _as_int(r.get("correct"))

        # Some runners only report averages (no totals). Recover totals via weighted sums.
        shard_total_rounds = r.get("total_rounds")
        shard_total_eff_rounds = r.get("total_effective_rounds")
        if shard_total_rounds is None:
            weighted_rounds_sum += shard_avg_rounds * shard_samples
        else:
            totals["total_rounds"] += _as_int(shard_total_rounds)
        if shard_total_eff_rounds is None:
            weighted_effective_rounds_sum += shard_avg_eff_rounds * shard_samples
        else:
            totals["total_effective_rounds"] += _as_int(shard_total_eff_rounds)

        totals["failed"] += _as_int(r.get("failed"))
        totals["invalid_outputs"] += _as_int(r.get("invalid_outputs"))
        totals["invalid_action_terminated"] += _as_int(r.get("invalid_action_terminated"))
        totals["total_retries"] += _as_int(r.get("total_retries"))
        totals["total_model_calls"] += _as_int(r.get("total_model_calls"))
        totals["fallback_frames_used"] += _as_int(r.get("fallback_frames_used"))
        totals["prompt_log_lines"] += _as_int(r.get("prompt_log_lines"))
        totals["prompt_log_bytes"] += _as_int(r.get("prompt_log_bytes"))

        shard_elapsed.append(_as_float(r.get("elapsed_s")))

        wandb = s.get("wandb") or {}
        shard_infos.append(
            {
                "summary_json": s.get("_summary_json_path"),
                "samples": shard_samples,
                "correct": _as_int(r.get("correct")),
                "accuracy": _as_float(r.get("accuracy")),
                "avg_rounds": shard_avg_rounds,
                "prompt_log_jsonl": s.get("prompt_log_jsonl"),
                "wandb_id": wandb.get("id"),
                "wandb_url": wandb.get("url"),
            }
        )

    samples = max(0, totals["samples"])
    correct = totals["correct"]
    accuracy = correct / samples if samples else 0.0
    nonfailed = max(0, samples - totals["failed"])
    accuracy_nonfailed = correct / nonfailed if nonfailed else 0.0
    total_rounds = totals["total_rounds"] + int(round(weighted_rounds_sum))
    total_effective_rounds = totals["total_effective_rounds"] + int(round(weighted_effective_rounds_sum))
    avg_rounds = total_rounds / samples if samples else 0.0
    avg_effective_rounds = total_effective_rounds / samples if samples else 0.0

    merged = {
        "samples": samples,
        "correct": correct,
        "accuracy": accuracy,
        "nonfailed_samples": nonfailed,
        "accuracy_nonfailed": accuracy_nonfailed,
        "total_rounds": total_rounds,
        "avg_rounds": avg_rounds,
        "total_effective_rounds": total_effective_rounds,
        "avg_effective_rounds": avg_effective_rounds,
        "failed": totals["failed"],
        "elapsed_s_max_shard": max(shard_elapsed) if shard_elapsed else 0.0,
        "elapsed_s_sum_shards": sum(shard_elapsed) if shard_elapsed else 0.0,
        "prompt_log_lines": totals["prompt_log_lines"],
        "prompt_log_bytes": totals["prompt_log_bytes"],
        "invalid_outputs": totals["invalid_outputs"],
        "invalid_action_terminated": totals["invalid_action_terminated"],
        "total_retries": totals["total_retries"],
        "total_model_calls": totals["total_model_calls"],
        "fallback_frames_used": totals["fallback_frames_used"],
        "shards": shard_infos,
    }
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", dest="glob_pattern", help="Glob pattern for summary.json files.")
    ap.add_argument("--output", "-o", help="Write merged JSON to this path (prints to stdout if omitted).")
    ap.add_argument("summaries", nargs="*", help="Per-shard summary.json files.")
    args = ap.parse_args()

    paths: list[Path] = []
    if args.glob_pattern:
        paths.extend(Path(p) for p in glob.glob(args.glob_pattern))
    paths.extend(Path(p) for p in (args.summaries or []))
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise SystemExit("No summary files found.")

    summaries: list[dict[str, Any]] = []
    for p in sorted(paths):
        s = _load_json(p)
        s["_summary_json_path"] = str(p)
        summaries.append(s)

    base = summaries[0].copy()
    base.pop("results", None)
    base.pop("wandb", None)
    base.pop("command", None)

    merged = {
        **base,
        "merged_from": [str(p) for p in sorted(paths)],
        "merged_results": _merge_results(summaries),
    }

    out = json.dumps(merged, indent=2, ensure_ascii=False)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out, encoding="utf-8")
    else:
        print(out)


if __name__ == "__main__":
    main()
