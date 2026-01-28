#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI
from openai import BadRequestError

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.revise import plug_and_play_nextqa_vllm as pnp  # type: ignore
from examples.revise.eval_nextqa_caption_vllm import (  # type: ignore
    _extract_answer_letter,
    _format_caption_timeline,
    _format_question,
    _load_video_captions,
)


SYSTEM_PROMPT = (
    "You are a multiple-choice video QA assistant.\n"
    "You will be given a question with options and video captions sampled at ~1 fps (caption index ≈ seconds).\n"
    "Answer with ONLY the option letter (A/B/C/D/E). Do not output any other text.\n"
)


def _extract_answer_letter_from_response(body: dict[str, Any]) -> Optional[str]:
    choices = body.get("choices", [])
    if not choices:
        return None
    msg = (choices[0] or {}).get("message", {})
    content = (msg or {}).get("content", "")
    if not isinstance(content, str) or not content.strip():
        return None
    return _extract_answer_letter(content)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5")
    ap.add_argument("--csv", default="/shares/hlw3876/chenwei/NExT-QA/nextqa/val.csv")
    ap.add_argument("--video-root", default="/shares/hlw3876/chenwei/NExT-QA/NExTVideo")
    ap.add_argument("--map-json", default="/shares/hlw3876/chenwei/NExT-QA/map_vid_vidorID.json")
    ap.add_argument("--captions-dir", default="data/nextqa_allcaps_1fps")
    ap.add_argument("--max-samples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--completion-window", default="24h", choices=["24h"])
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--dry-run", action="store_true", help="Only write batch_input.jsonl; do not call OpenAI API.")

    ap.add_argument("--per-caption-max-chars", type=int, default=220)
    ap.add_argument("--total-caption-max-chars", type=int, default=40000)
    ap.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature. For GPT-5, only the default value is supported, so leave unset.",
    )
    ap.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus sampling. For GPT-5, only the default value is supported, so leave unset.",
    )
    ap.add_argument(
        "--max-completion-tokens",
        type=int,
        default=512,
        help="GPT-5 uses hidden reasoning tokens; keep this large enough to avoid empty outputs.",
    )
    ap.add_argument("--poll-interval-s", type=float, default=5.0)
    args = ap.parse_args()

    out_dir = args.output_dir
    if not out_dir:
        stamp = time.strftime("%Y-%m-%d")
        out_dir = f"outputs/{stamp}/openai_batch_nextqa_caption_only_{args.model}_n{int(args.max_samples)}"
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    samples = pnp._load_nextqa_samples(
        csv_path=str(args.csv),
        map_json=str(args.map_json),
        video_root=str(args.video_root),
        max_samples=int(args.max_samples),
        seed=int(args.seed),
    )
    if not samples:
        raise RuntimeError(f"No samples loaded from {args.csv}")

    # Prepare batch input jsonl.
    input_jsonl_path = out_path / "batch_input.jsonl"
    samples_json_path = out_path / "samples.jsonl"
    gt_by_custom_id: dict[str, str] = {}

    with input_jsonl_path.open("w", encoding="utf-8") as fout, samples_json_path.open("w", encoding="utf-8") as fsamp:
        for idx, sample in enumerate(samples):
            gt_letter = chr(ord("A") + int(sample.answer_idx))
            custom_id = f"nextqa:{sample.sample_id}"

            captions = _load_video_captions(args.captions_dir, sample.video_id)
            caption_block, caption_chars = _format_caption_timeline(
                captions,
                per_caption_max_chars=int(args.per_caption_max_chars),
                total_max_chars=int(args.total_caption_max_chars),
            )
            user_text = (
                f"{_format_question(sample.question, sample.choices)}\n\n"
                "Captions (1fps timeline; index≈seconds):\n"
                f"{caption_block}\n\n"
                "Return ONLY the option letter.\n"
            )

            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": str(args.model),
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_text},
                    ],
                    "max_completion_tokens": int(args.max_completion_tokens),
                },
            }
            if args.temperature is not None:
                request["body"]["temperature"] = float(args.temperature)
            if args.top_p is not None:
                request["body"]["top_p"] = float(args.top_p)
            fout.write(json.dumps(request, ensure_ascii=False) + "\n")
            fsamp.write(
                json.dumps(
                    {
                        "idx": idx,
                        "custom_id": custom_id,
                        "sample_id": sample.sample_id,
                        "qid": sample.qid,
                        "video_id": sample.video_id,
                        "question": sample.question,
                        "choices": sample.choices,
                        "ground_truth": gt_letter,
                        "caption_chars": caption_chars,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            gt_by_custom_id[custom_id] = gt_letter

    if args.dry_run:
        print(f"[dry-run] wrote: {input_jsonl_path}")
        print(f"[dry-run] wrote: {samples_json_path}")
        print(f"[dry-run] output_dir: {out_path}")
        return 0

    client = OpenAI()
    file_obj = client.files.create(file=input_jsonl_path.open("rb"), purpose="batch")
    try:
        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window=str(args.completion_window),
            metadata={
                "task": "nextqa_caption_only",
                "split": os.path.basename(args.csv),
                "model": str(args.model),
                "n": str(int(args.max_samples)),
            },
        )
    except BadRequestError as e:
        msg = ""
        try:
            payload = getattr(e, "response", None).json() if getattr(e, "response", None) is not None else {}
            msg = str(((payload or {}).get("error") or {}).get("message") or "")
            code = str(((payload or {}).get("error") or {}).get("code") or "")
        except Exception:
            code = ""
        if "billing" in msg.lower() or "billing" in code.lower() or "hard_limit" in code.lower():
            print("[error] OpenAI API rejected the Batch request due to billing limits.")
            print(f"[error] message={msg!r} code={code!r}")
            print("[next] Increase/raise your OpenAI billing hard limit or use an API key with available quota, then rerun.")
            print(f"[next] Reuse the already-written batch file: {input_jsonl_path}")
            return 2
        raise

    (out_path / "batch_meta.json").write_text(
        json.dumps(
            {
                "batch_id": batch.id,
                "input_file_id": file_obj.id,
                "created_at": getattr(batch, "created_at", None),
                "endpoint": "/v1/chat/completions",
                "completion_window": str(args.completion_window),
                "model": str(args.model),
                "max_samples": int(args.max_samples),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[batch] created: {batch.id} (input_file_id={file_obj.id})")
    status = batch.status
    while status not in {"completed", "failed", "cancelled", "expired"}:
        time.sleep(float(args.poll_interval_s))
        batch = client.batches.retrieve(batch.id)
        status = batch.status
        print(f"[batch] status={status}")

    if status != "completed":
        raise RuntimeError(f"Batch did not complete successfully: status={status} batch_id={batch.id}")

    output_file_id = batch.output_file_id
    error_file_id = batch.error_file_id
    if not output_file_id:
        if error_file_id:
            err_path = out_path / "batch_error.jsonl"
            err_content = client.files.content(error_file_id).read()
            err_path.write_bytes(err_content)
            raise RuntimeError(
                f"Batch completed but only produced errors (no output_file_id). "
                f"batch_id={batch.id} error_file_id={error_file_id} saved={err_path}"
            )
        raise RuntimeError(f"Batch completed but has no output_file_id: batch_id={batch.id}")

    out_jsonl_path = out_path / "batch_output.jsonl"
    content = client.files.content(output_file_id).read()
    out_jsonl_path.write_bytes(content)
    print(f"[batch] output saved: {out_jsonl_path} (output_file_id={output_file_id})")

    # Score.
    preds: dict[str, Optional[str]] = {}
    invalid = 0
    correct = 0
    total = 0
    with out_jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            custom_id = obj.get("custom_id")
            resp = obj.get("response") or {}
            body = resp.get("body") or {}
            total += 1
            pred = _extract_answer_letter_from_response(body) if resp.get("status_code") == 200 else None
            preds[str(custom_id)] = pred
            if pred is None:
                invalid += 1
                continue
            gt = gt_by_custom_id.get(str(custom_id))
            if gt and pred == gt:
                correct += 1

    n_samples = len(samples)
    acc = correct / max(1, n_samples)
    summary = {
        "model": str(args.model),
        "batch_id": batch.id,
        "status": status,
        "n_samples": n_samples,
        "correct": correct,
        "accuracy": acc,
        "invalid_outputs": invalid,
        "output_file_id": output_file_id,
        "paths": {
            "output_dir": str(out_path),
            "batch_input_jsonl": str(input_jsonl_path),
            "batch_output_jsonl": str(out_jsonl_path),
            "samples_jsonl": str(samples_json_path),
        },
    }
    (out_path / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
