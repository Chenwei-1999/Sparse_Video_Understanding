#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI


def _safe_json_loads(line: str) -> Optional[dict[str, Any]]:
    try:
        obj = json.loads(line)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _extract_chat_content_from_batch_line(line_obj: dict[str, Any]) -> tuple[int, Optional[str], Optional[str]]:
    resp = line_obj.get("response") or {}
    status = int(resp.get("status_code") or 0)
    body = resp.get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        return status, None, None
    choice0 = choices[0] or {}
    finish_reason = choice0.get("finish_reason")
    msg = choice0.get("message") or {}
    content = msg.get("content")
    if not isinstance(content, str):
        content = None
    if finish_reason is not None and not isinstance(finish_reason, str):
        finish_reason = str(finish_reason)
    return status, content, finish_reason


def _answer_letter(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    s = text.strip()
    if not s:
        return None
    ch = s[0].upper()
    return ch if ch in {"A", "B", "C", "D", "E"} else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Directory containing batch_input.jsonl and batch_output.jsonl")
    ap.add_argument("--model", default="gpt-5")
    ap.add_argument("--completion-window", default="24h", choices=["24h"])
    ap.add_argument("--retry-max-completion-tokens", type=int, default=2048)
    ap.add_argument("--poll-interval-s", type=float, default=10.0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    batch_input = run_dir / "batch_input.jsonl"
    batch_output = run_dir / "batch_output.jsonl"
    samples_jsonl = run_dir / "samples.jsonl"
    if not batch_input.exists() or not batch_output.exists() or not samples_jsonl.exists():
        raise FileNotFoundError(f"Missing required files in {run_dir}")

    # Ground truth map.
    gt_by_custom_id: dict[str, str] = {}
    with samples_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            obj = _safe_json_loads(line)
            if not obj:
                continue
            cid = obj.get("custom_id")
            gt = obj.get("ground_truth")
            if isinstance(cid, str) and isinstance(gt, str) and gt:
                gt_by_custom_id[cid] = gt.strip().upper()[:1]

    # Find empty outputs (finish_reason=length and empty content).
    need_retry: set[str] = set()
    orig_pred: dict[str, Optional[str]] = {}
    with batch_output.open("r", encoding="utf-8") as f:
        for line in f:
            obj = _safe_json_loads(line)
            if not obj:
                continue
            cid = obj.get("custom_id")
            if not isinstance(cid, str):
                continue
            status, content, finish_reason = _extract_chat_content_from_batch_line(obj)
            pred = _answer_letter(content)
            orig_pred[cid] = pred
            if status == 200 and (content or "").strip() == "" and finish_reason == "length":
                need_retry.add(cid)

    if not need_retry:
        print("[retry] no empty outputs found; nothing to do.")
        return 0

    retry_dir = run_dir / f"retry_empty_mct{int(args.retry_max_completion_tokens)}"
    retry_dir.mkdir(parents=True, exist_ok=True)
    retry_input = retry_dir / "batch_input.jsonl"

    # Build retry batch_input.jsonl by filtering original requests.
    kept = 0
    with batch_input.open("r", encoding="utf-8") as fin, retry_input.open("w", encoding="utf-8") as fout:
        for line in fin:
            obj = _safe_json_loads(line)
            if not obj:
                continue
            cid = obj.get("custom_id")
            if not isinstance(cid, str) or cid not in need_retry:
                continue
            body = obj.get("body") or {}
            if not isinstance(body, dict):
                continue
            body["model"] = str(args.model)
            body.pop("temperature", None)
            body.pop("top_p", None)
            body["max_completion_tokens"] = int(args.retry_max_completion_tokens)
            obj["body"] = body
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

    print(f"[retry] retrying {kept} empty outputs in {retry_dir}")

    client = OpenAI()
    file_obj = client.files.create(file=retry_input.open("rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window=str(args.completion_window),
        metadata={
            "task": "nextqa_caption_only_retry_empty",
            "model": str(args.model),
            "n": str(kept),
            "source_dir": str(run_dir),
        },
    )
    (retry_dir / "batch_meta.json").write_text(
        json.dumps(
            {
                "batch_id": batch.id,
                "input_file_id": file_obj.id,
                "endpoint": "/v1/chat/completions",
                "completion_window": str(args.completion_window),
                "model": str(args.model),
                "n": kept,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[retry] created batch: {batch.id}")

    status = batch.status
    while status not in {"completed", "failed", "cancelled", "expired"}:
        time.sleep(float(args.poll_interval_s))
        batch = client.batches.retrieve(batch.id)
        status = batch.status
        print(f"[retry] status={status} counts={batch.request_counts}")

    if status != "completed":
        raise RuntimeError(f"Retry batch did not complete successfully: status={status} batch_id={batch.id}")
    if not batch.output_file_id:
        if batch.error_file_id:
            err_path = retry_dir / "batch_error.jsonl"
            err_path.write_bytes(client.files.content(batch.error_file_id).read())
            raise RuntimeError(
                f"Retry batch produced only errors. batch_id={batch.id} error_file_id={batch.error_file_id} saved={err_path}"
            )
        raise RuntimeError(f"Retry batch completed but has no output_file_id: batch_id={batch.id}")

    out_jsonl = retry_dir / "batch_output.jsonl"
    out_jsonl.write_bytes(client.files.content(batch.output_file_id).read())
    print(f"[retry] output saved: {out_jsonl}")

    # Parse retry preds.
    retry_pred: dict[str, Optional[str]] = {}
    retry_empty = 0
    with out_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            obj = _safe_json_loads(line)
            if not obj:
                continue
            cid = obj.get("custom_id")
            if not isinstance(cid, str):
                continue
            status_code, content, finish_reason = _extract_chat_content_from_batch_line(obj)
            if status_code != 200:
                retry_pred[cid] = None
                continue
            pred = _answer_letter(content)
            retry_pred[cid] = pred
            if (content or "").strip() == "" and finish_reason == "length":
                retry_empty += 1

    # Merge + score.
    merged_pred: dict[str, Optional[str]] = dict(orig_pred)
    for cid, pred in retry_pred.items():
        if merged_pred.get(cid) is None and pred is not None:
            merged_pred[cid] = pred

    total = len(gt_by_custom_id)
    correct = 0
    invalid = 0
    for cid, gt in gt_by_custom_id.items():
        pred = merged_pred.get(cid)
        if pred is None:
            invalid += 1
            continue
        if pred == gt:
            correct += 1

    summary = {
        "source_run_dir": str(run_dir),
        "retry_dir": str(retry_dir),
        "retry_batch_id": batch.id,
        "retry_output_file_id": batch.output_file_id,
        "retry_n": kept,
        "retry_still_empty": retry_empty,
        "merged": {"n_samples": total, "correct": correct, "accuracy": correct / max(1, total), "invalid": invalid},
    }
    (retry_dir / "merged_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

