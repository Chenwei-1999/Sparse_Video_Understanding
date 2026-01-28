#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import tiktoken


@dataclass
class ImageTokenConfig:
    detail: str  # "low" | "high"
    width: int
    height: int


@dataclass
class SettingSpec:
    name: str
    shard_logs: list[Path]


def _ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _estimate_gpt5_image_tokens(cfg: ImageTokenConfig) -> int:
    """Estimate vision tokens for one image.

    Uses the tile-based scheme described in OpenAI Vision docs for models including GPT-5:
    - detail="low": fixed base tokens
    - detail="high": base + tile_tokens * num_tiles after the 2048/768 scaling rule

    We assume the caller will resize images before upload so (width,height) are already final.
    """

    # Per-doc table (gpt-5 / gpt-5-chat-latest):
    base_tokens = 70
    tile_tokens = 140

    detail = (cfg.detail or "low").lower()
    if detail == "low":
        return base_tokens

    # detail="high": OpenAI's heuristic scales the image so the shortest side becomes 768px
    # (after fitting within a 2048px square). Approximate this:
    w, h = int(cfg.width), int(cfg.height)
    if w <= 0 or h <= 0:
        return base_tokens

    # Step 1: fit within 2048x2048 while preserving aspect ratio.
    max_side = max(w, h)
    if max_side > 2048:
        scale = 2048 / float(max_side)
        w = max(1, int(round(w * scale)))
        h = max(1, int(round(h * scale)))

    # Step 2: scale so shortest side is 768.
    min_side = min(w, h)
    if min_side > 0 and min_side != 768:
        scale = 768 / float(min_side)
        w = max(1, int(round(w * scale)))
        h = max(1, int(round(h * scale)))

    tiles = _ceildiv(w, 512) * _ceildiv(h, 512)
    return base_tokens + tile_tokens * int(tiles)


def _iter_log_lines(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _token_len(enc: tiktoken.Encoding, text: Optional[str]) -> int:
    if not text:
        return 0
    return len(enc.encode(str(text)))


@dataclass
class TokenTally:
    samples: int = 0
    calls: int = 0
    text_in_tokens: int = 0
    image_tokens: int = 0
    out_tokens: int = 0
    images: int = 0

    def total_in_tokens(self) -> int:
        return self.text_in_tokens + self.image_tokens


def _estimate_from_shard(
    log_path: Path,
    enc: tiktoken.Encoding,
    *,
    max_samples: int,
    image_cfg: ImageTokenConfig,
    chat_overhead_tokens: int,
) -> TokenTally:
    tally = TokenTally()
    seen: set[str] = set()
    image_tokens_per = _estimate_gpt5_image_tokens(image_cfg)

    for obj in _iter_log_lines(log_path):
        sid = obj.get("sample_id")
        if not isinstance(sid, str) or not sid:
            continue

        if sid not in seen:
            if len(seen) >= max_samples:
                break
            seen.add(sid)

        system_prompt = obj.get("system_prompt", "")
        user_text = obj.get("user_text", "")
        raw_output = obj.get("raw_output", "")

        img_count = str(user_text).count("<image>")
        tally.images += img_count
        tally.image_tokens += img_count * image_tokens_per

        tally.text_in_tokens += _token_len(enc, system_prompt) + _token_len(enc, user_text) + int(
            chat_overhead_tokens
        )
        tally.out_tokens += _token_len(enc, raw_output)
        tally.calls += 1

    tally.samples = len(seen)
    return tally


def _find_existing_logs(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.exists():
            out.append(path)
    return out


def _default_specs() -> list[SettingSpec]:
    return [
        SettingSpec(
            name="caption_only_singleturn",
            shard_logs=_find_existing_logs(
                [
                    "outputs/2026-01-25/nextqa_caption_compare_qwen2p5vl7b/caption_only_val_tp1/log.shard0of4.jsonl",
                    "outputs/2026-01-25/nextqa_caption_compare_qwen2p5vl7b/caption_only_val_tp1/log.shard1of4.jsonl",
                    "outputs/2026-01-25/nextqa_caption_compare_qwen2p5vl7b/caption_only_val_tp1/log.shard2of4.jsonl",
                    "outputs/2026-01-25/nextqa_caption_compare_qwen2p5vl7b/caption_only_val_tp1/log.shard3of4.jsonl",
                ]
            ),
        ),
        SettingSpec(
            name="revise_frames_pnp",
            shard_logs=_find_existing_logs(
                [
                    "debug_prompt_logs/nextqa_val_full_qwen2p5vl7b_ids_tp1.shard0of4.jsonl",
                    "debug_prompt_logs/nextqa_val_full_qwen2p5vl7b_ids_tp1.shard1of4.jsonl",
                    "debug_prompt_logs/nextqa_val_full_qwen2p5vl7b_ids_tp1.shard2of4.jsonl",
                    "debug_prompt_logs/nextqa_val_full_qwen2p5vl7b_ids_tp1.shard3of4.jsonl",
                ]
            ),
        ),
        SettingSpec(
            name="revise_plus_caption_pnp",
            shard_logs=_find_existing_logs(
                [
                    "outputs/2026-01-25/nextqa_caption_compare_qwen2p5vl7b/revise_plus_caption_val_tp1_ts/log.shard0of4.jsonl",
                    "outputs/2026-01-25/nextqa_caption_compare_qwen2p5vl7b/revise_plus_caption_val_tp1_ts/log.shard1of4.jsonl",
                    "outputs/2026-01-25/nextqa_caption_compare_qwen2p5vl7b/revise_plus_caption_val_tp1_ts/log.shard2of4.jsonl",
                    "outputs/2026-01-25/nextqa_caption_compare_qwen2p5vl7b/revise_plus_caption_val_tp1_ts/log.shard3of4.jsonl",
                ]
            ),
        ),
        SettingSpec(
            name="caption_only_revise_multiturn",
            shard_logs=_find_existing_logs(
                [
                    "outputs/2026-01-25/nextqa_caption_compare_qwen2p5vl7b/caption_revise_val_tp1_ts/log.shard0of4.jsonl",
                    "outputs/2026-01-25/nextqa_caption_compare_qwen2p5vl7b/caption_revise_val_tp1_ts/log.shard1of4.jsonl",
                    "outputs/2026-01-25/nextqa_caption_compare_qwen2p5vl7b/caption_revise_val_tp1_ts/log.shard2of4.jsonl",
                    "outputs/2026-01-25/nextqa_caption_compare_qwen2p5vl7b/caption_revise_val_tp1_ts/log.shard3of4.jsonl",
                ]
            ),
        ),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=500, help="Total samples to estimate per setting.")
    ap.add_argument(
        "--per-shard",
        type=int,
        default=125,
        help="Samples per shard (4 shards -> total ~= 4*per_shard).",
    )
    ap.add_argument("--encoding", default="o200k_base", help="tiktoken encoding name.")
    ap.add_argument(
        "--image-detail",
        default="low",
        choices=["low", "high"],
        help="Assumed GPT-5 vision detail setting for image token estimate.",
    )
    ap.add_argument("--image-width", type=int, default=512)
    ap.add_argument("--image-height", type=int, default=512)
    ap.add_argument(
        "--chat-overhead-tokens",
        type=int,
        default=12,
        help="Rough per-request overhead for chat formatting/roles.",
    )
    ap.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = ap.parse_args()

    enc = tiktoken.get_encoding(args.encoding)
    img_cfg = ImageTokenConfig(detail=args.image_detail, width=args.image_width, height=args.image_height)

    specs = _default_specs()
    per_shard = max(1, int(args.per_shard))
    target_total = int(args.max_samples)

    results: dict[str, Any] = {"assumptions": {}, "settings": {}}
    results["assumptions"] = {
        "encoding": args.encoding,
        "chat_overhead_tokens_per_call": int(args.chat_overhead_tokens),
        "image_detail": img_cfg.detail,
        "image_size": [img_cfg.width, img_cfg.height],
        "image_tokens_per_image_estimate": _estimate_gpt5_image_tokens(img_cfg),
        "target_samples_per_setting": target_total,
        "target_samples_per_shard": per_shard,
    }

    for spec in specs:
        shard_logs = spec.shard_logs
        if not shard_logs:
            continue

        per_setting = TokenTally()
        shard_breakdown: list[dict[str, Any]] = []

        # Sample up to `per_shard` from each shard until we hit ~target_total.
        remaining = target_total
        for shard_path in shard_logs:
            if remaining <= 0:
                break
            shard_target = min(per_shard, remaining)
            shard_tally = _estimate_from_shard(
                shard_path,
                enc,
                max_samples=shard_target,
                image_cfg=img_cfg,
                chat_overhead_tokens=int(args.chat_overhead_tokens),
            )

            per_setting.samples += shard_tally.samples
            per_setting.calls += shard_tally.calls
            per_setting.text_in_tokens += shard_tally.text_in_tokens
            per_setting.image_tokens += shard_tally.image_tokens
            per_setting.out_tokens += shard_tally.out_tokens
            per_setting.images += shard_tally.images
            remaining -= shard_tally.samples

            shard_breakdown.append(
                {
                    "log": str(shard_path),
                    "samples": shard_tally.samples,
                    "calls": shard_tally.calls,
                    "text_in_tokens": shard_tally.text_in_tokens,
                    "image_tokens": shard_tally.image_tokens,
                    "out_tokens": shard_tally.out_tokens,
                    "images": shard_tally.images,
                }
            )

        settings_out = {
            "samples": per_setting.samples,
            "calls": per_setting.calls,
            "text_in_tokens": per_setting.text_in_tokens,
            "image_tokens": per_setting.image_tokens,
            "in_tokens_total": per_setting.total_in_tokens(),
            "out_tokens": per_setting.out_tokens,
            "images": per_setting.images,
            "avg_calls_per_sample": (per_setting.calls / per_setting.samples) if per_setting.samples else 0.0,
            "avg_in_tokens_per_sample": (per_setting.total_in_tokens() / per_setting.samples)
            if per_setting.samples
            else 0.0,
            "avg_out_tokens_per_sample": (per_setting.out_tokens / per_setting.samples)
            if per_setting.samples
            else 0.0,
            "avg_images_per_call": (per_setting.images / per_setting.calls) if per_setting.calls else 0.0,
            "avg_in_tokens_per_call": (per_setting.total_in_tokens() / per_setting.calls)
            if per_setting.calls
            else 0.0,
            "avg_out_tokens_per_call": (per_setting.out_tokens / per_setting.calls) if per_setting.calls else 0.0,
            "shards": shard_breakdown,
        }

        results["settings"][spec.name] = settings_out

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    # Human-readable table
    print("# Estimated GPT-5 tokens for 500-sample NextQA caption/frames compare")
    print("")
    print("Assumptions:")
    print(f"- encoding: {args.encoding}")
    print(f"- image detail: {img_cfg.detail}")
    print(f"- image size: {img_cfg.width}x{img_cfg.height}")
    print(f"- image tokens per image (est): {_estimate_gpt5_image_tokens(img_cfg)}")
    print(f"- chat overhead tokens/call: {int(args.chat_overhead_tokens)}")
    print(f"- target samples/setting: {target_total}")
    print("")
    print("| setting | samples | calls | in_tokens(text) | in_tokens(image) | in_tokens(total) | out_tokens | avg_calls/sample | avg_in/sample | avg_out/sample |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, s in results["settings"].items():
        print(
            f"| {name} | {s['samples']} | {s['calls']} | {s['text_in_tokens']} | {s['image_tokens']} | {s['in_tokens_total']} | {s['out_tokens']} | "
            f"{s['avg_calls_per_sample']:.2f} | {s['avg_in_tokens_per_sample']:.1f} | {s['avg_out_tokens_per_sample']:.1f} |"
        )


if __name__ == "__main__":
    main()
