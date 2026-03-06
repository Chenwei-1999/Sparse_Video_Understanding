#!/usr/bin/env python3
"""Check accuracy from a REVISE JSONL prompt log."""
import json, sys, glob, collections, re

pattern = sys.argv[1] if len(sys.argv) > 1 else "debug_prompt_logs/full_nextqa_val_*.jsonl"
files = sorted(glob.glob(pattern))
if not files:
    print("No JSONL files found matching:", pattern)
    sys.exit(1)
path = files[-1]

# Group rounds by sample_id
samples = collections.OrderedDict()
with open(path) as f:
    for line in f:
        rec = json.loads(line)
        sid = rec.get("sample_id", "")
        if sid not in samples:
            samples[sid] = []
        samples[sid].append(rec)

total = len(samples)
if total == 0:
    print("No samples found")
    sys.exit(0)

correct = 0
answered = 0
total_rounds = 0
total_frames = 0
invalid_no_answer = 0
ANSWER_RE = re.compile(r"<answer>\s*([A-Ea-e])\s*</answer>")

for sid, rounds in samples.items():
    total_rounds += len(rounds)
    last = rounds[-1]
    gt_idx = last.get("ground_truth_idx", -1)
    seen = last.get("seen_frames", [])
    total_frames += len(seen)

    # Try to find answer in any round (last round most likely)
    found_answer = False
    for r in reversed(rounds):
        raw = r.get("raw_output", "")
        m = ANSWER_RE.search(raw)
        if m:
            pred = m.group(1).upper()
            pred_idx = ord(pred) - ord("A")
            answered += 1
            if pred_idx == gt_idx:
                correct += 1
            found_answer = True
            break
    if not found_answer:
        invalid_no_answer += 1

acc = correct / total * 100 if total else 0
ans_rate = answered / total * 100 if total else 0
avg_rounds = total_rounds / total if total else 0
avg_frames = total_frames / total if total else 0

print(f"JSONL: {path}")
print(f"Samples processed: {total}")
print(f"Answered: {answered}/{total} ({ans_rate:.1f}%)")
print(f"Accuracy (all): {correct}/{total} = {acc:.1f}%")
if answered:
    print(f"Accuracy (answered only): {correct}/{answered} = {correct/answered*100:.1f}%")
print(f"No answer (invalid/terminated): {invalid_no_answer}")
print(f"Avg rounds: {avg_rounds:.2f}")
print(f"Avg frames used: {avg_frames:.1f}")
print()
print("Paper reference: Qwen2.5-VL-3B plug-and-play NExT-QA = 31.7% acc, 5.3 frames, 1.74 rounds")
