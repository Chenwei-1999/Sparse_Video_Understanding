#!/usr/bin/env python3
"""Generate SFT training data from 7B plug-and-play evaluation logs.

Reads REVISE multi-round eval logs (JSONL), reconstructs multi-turn
conversations, filters for quality, and outputs parquet files compatible
with MultiTurnSFTDataset.

IMPORTANT: The input log must come from the TRAIN split to avoid data
leakage. Use run_generate_teacher_data.sh to generate train-split logs.

Usage:
    # Step 1: Generate teacher data on train split (requires GPU)
    ./examples/revise/run_generate_teacher_data.sh

    # Step 2: Convert to SFT parquet
    python examples/revise/generate_sft_data.py \
        --input outputs/nextqa_pnp_7b_train_log.jsonl \
        --output outputs/sft_data/revise_sft.parquet \
        --val_ratio 0.05
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd


def has_valid_summary(text: str) -> bool:
    """Check that text contains a well-formed <summary>...</summary> block."""
    return bool(re.search(r"<summary>.+?</summary>", text, re.DOTALL))


def has_valid_answer(text: str) -> bool:
    """Check that the final-round output contains <answer>LETTER</answer>."""
    return bool(re.search(r"<answer>\s*[A-E]\s*</answer>", text))


def has_template_text(text: str) -> bool:
    """Detect if the model simply copied the template placeholder."""
    markers = [
        "I will summarize what has been shown so far",
        "I will record the key observations",
        "I will update my belief as new evidence arrives",
    ]
    return any(m in text for m in markers)


def strip_image_placeholders(text: str) -> str:
    """Replace <image> vision placeholders with a text-only marker."""
    return text.replace("<image>", "[frame]")


def build_conversation(rounds: list[dict]) -> list[dict] | None:
    """Build a chat-format conversation from a list of sorted round entries.

    Returns None if any quality check fails.
    """
    if not rounds:
        return None

    messages = []

    # System prompt (same across rounds)
    system_prompt = rounds[0].get("system_prompt", "")
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for i, entry in enumerate(rounds):
        raw_output = entry.get("raw_output", "")
        user_text = entry.get("user_text", "")

        # Quality filter: every round must have valid summary
        if not has_valid_summary(raw_output):
            return None

        # Quality filter: reject template-copied outputs
        if has_template_text(raw_output):
            return None

        is_last = i == len(rounds) - 1
        # Last round must have a valid answer
        if is_last and not has_valid_answer(raw_output):
            return None

        # Strip image placeholders for text-only SFT
        user_content = strip_image_placeholders(user_text)
        assistant_content = raw_output.strip()

        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": assistant_content})

    return messages


def main():
    parser = argparse.ArgumentParser(description="Generate SFT data from REVISE eval logs")
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/nextqa_pnp_7b_train_log.jsonl",
        help="Path to eval log JSONL (should be from train split)",
    )
    parser.add_argument(
        "--val-csv",
        type=str,
        default="/shares/hlw3876/chenwei/NExT-QA/nextqa/val.csv",
        help="NExT-QA val CSV to check for data leakage",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/sft_data/revise_sft.parquet",
        help="Output parquet path (train split; val split auto-named)",
    )
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Fraction held out for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    args = parser.parse_args()

    # 1. Read and group entries by sample_id
    print(f"Reading {args.input} ...")
    sample_rounds: dict[str, list[dict]] = defaultdict(list)
    with open(args.input) as f:
        for line in f:
            entry = json.loads(line)
            sample_rounds[entry["sample_id"]].append(entry)

    print(f"  Total entries: {sum(len(v) for v in sample_rounds.values())}")
    print(f"  Unique samples: {len(sample_rounds)}")

    # 1b. Check for data leakage against val split
    if os.path.exists(args.val_csv):
        val_keys = set()
        with open(args.val_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                val_keys.add((row["video"], row["question"]))
        eval_keys = set()
        for entries in sample_rounds.values():
            e = entries[0]
            eval_keys.add((e["video_id"], e["question"]))
        overlap = eval_keys & val_keys
        if overlap:
            pct = len(overlap) / len(eval_keys) * 100
            print(f"\n  WARNING: {len(overlap)}/{len(eval_keys)} samples ({pct:.0f}%) overlap with val split!")
            print("  This will cause data leakage if you evaluate on val.")
            print("  Use run_generate_teacher_data.sh to generate train-split teacher data.\n")
            if pct > 50:
                print("  ERROR: >50% overlap. Refusing to generate SFT data from val-split logs.")
                print("  Pass --val-csv '' to override (not recommended).")
                sys.exit(1)
    else:
        print(f"  (Skipping leakage check: {args.val_csv} not found)")

    # 2. Sort rounds within each sample and build conversations
    conversations = []
    skipped = 0
    for sample_id, entries in sample_rounds.items():
        # Sort by round_idx, then retry_idx (take retry_idx=0 only)
        entries = [e for e in entries if e.get("retry_idx", 0) == 0]
        entries.sort(key=lambda e: e["round_idx"])
        conv = build_conversation(entries)
        if conv is not None:
            conversations.append(conv)
        else:
            skipped += 1

    print(f"  Valid conversations: {len(conversations)}")
    print(f"  Skipped (quality filter): {skipped}")

    if not conversations:
        print("ERROR: No valid conversations found. Check input file.")
        return

    # 3. Split train / val
    import random

    random.seed(args.seed)
    indices = list(range(len(conversations)))
    random.shuffle(indices)
    n_val = max(1, int(len(conversations) * args.val_ratio))
    val_indices = set(indices[:n_val])
    train_convs = [conversations[i] for i in range(len(conversations)) if i not in val_indices]
    val_convs = [conversations[i] for i in val_indices]

    print(f"  Train: {len(train_convs)}, Val: {len(val_convs)}")

    # 4. Write parquet files
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame({"messages": train_convs})
    val_path = output_path.parent / output_path.name.replace(".parquet", "_val.parquet")
    train_path = output_path.parent / output_path.name.replace(".parquet", "_train.parquet")

    train_df.to_parquet(str(train_path))
    val_df = pd.DataFrame({"messages": val_convs})
    val_df.to_parquet(str(val_path))

    print(f"  Wrote {train_path} ({len(train_df)} rows)")
    print(f"  Wrote {val_path} ({len(val_df)} rows)")

    # 5. Print stats
    turn_counts = [len([m for m in c if m["role"] == "assistant"]) for c in conversations]
    print(f"\nStats:")
    print(f"  Avg assistant turns per conversation: {sum(turn_counts) / len(turn_counts):.1f}")
    print(f"  Min/Max turns: {min(turn_counts)}/{max(turn_counts)}")


if __name__ == "__main__":
    main()
