# Copyright 2026
# Licensed under the Apache License, Version 2.0

"""REVISE-style multi-round agent loop for sparse video QA."""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from typing import Any, Optional
from uuid import uuid4

import numpy as np
from PIL import Image

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


_SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL | re.IGNORECASE)
_FRAMES_RE = re.compile(r"<frames>(.*?)</frames>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

_PLACEHOLDER_SET = {"...", "…", "none", "n/a", "na", "null", "unknown", "unsure", "uncertain"}


DEFAULT_SYSTEM_PROMPT = (
    "You are REVISE, a multi-round video reasoning agent.\n"
    "Each round you will see: (1) a multiple-choice question with options, (2) the current belief summary, "
    "and (3) a few sampled video frames.\n"
    "If you are confident, answer the question.\n"
    "If you are NOT confident, request MORE video frames to view NEXT.\n"
    "Frames are sampled at ~1 fps; frame index ≈ timestamp in seconds.\n\n"
    "IMPORTANT: Output must follow EXACTLY ONE of the two formats below. Do not output any text outside tags.\n"
    "Do NOT output placeholders like '...', 'none', 'unknown', or 'N/A' in your final output.\n\n"
    "Format 1 — Request more frames (use this only if NOT confident):\n"
    "<think>I am not confident yet; I need more visual evidence to confirm my current hypothesis.</think>\n"
    "<summary>O: George seems to pause and think about something, then begins checking the items inside the shelves; "
    "H: his curiosity appears connected to the objects he finds or examines on the shelves; "
    "R: looking at later frames may help confirm what exactly caught his attention or what he discovered; "
    "P: the agent has already seen frames 4, 8, and 12; "
    "U: it is still unclear what first caused his curiosity or made him start investigating</summary>\n"
    "<frames>18, 24</frames>\n\n"
    "Format 2 — Answer now (use this if confident):\n"
    "<think>I am confident now; the observed evidence matches my hypothesis.</think>\n"
    "<summary>O: the key evidence is visible in the shown frames; "
    "H: my belief is updated based on the observed evidence; "
    "R: answered; "
    "P: the agent has already seen frames 0, 12, 18, and 24; "
    "U: no remaining ambiguity that affects the answer</summary>\n"
    "<answer>B</answer>\n\n"
    "Tag meanings:\n"
    "- <think>: 1–2 short sentences describing your decision (NOT the final answer).\n"
    "- <summary>: the ONLY persistent memory across rounds. Keep it short and update it EVERY round.\n"
    "  - O (Observations): what you currently observe in the selected frames.\n"
    "  - H (Belief updates): your updated belief based on what has been observed so far.\n"
    "  - R (Reasons): why you need more frames and what evidence you are looking for next.\n"
    "  - P (Previously seen): which frames have already been used/seen.\n"
    "  - U (Uncertainties): what is still unknown or ambiguous.\n\n"
    "Rules:\n"
    "- Frame indices are 0-based in [0, L-1].\n"
    "- If you are confident, answer instead of requesting more frames.\n"
    "- If requesting, choose 1 to {max_frames_per_round} NEW frames to view NEXT.\n"
    "- Do NOT output any frame index from the Seen frames list; those are already viewed.\n"
    "- In <frames>, output comma-separated integers only (no brackets, no text).\n"
    "- In <summary>, include O/H/R/P/U as short natural-language sentences that reflect your current understanding.\n"
    "- In P, describe previously seen frames in a sentence (e.g., 'the agent has already seen frames 4, 8, and 12'); "
    "do NOT use Python list formatting like [4, 8, 12].\n"
    "- In <answer>, output EXACTLY ONE option letter shown in the question (e.g., A/B/C/D/E). No words/punctuation.\n"
    "- Never copy the example text; replace it with information from the current video.\n"
)


def _extract_tag(text: str, tag_re: re.Pattern[str]) -> Optional[str]:
    matches = list(tag_re.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _format_frame_list(frames: list[int]) -> str:
    if not frames:
        return "no frames yet"
    return ", ".join(str(int(i)) for i in frames)


def _is_placeholder(text: str) -> bool:
    t = _collapse_ws(text).lower()
    if not t:
        return True
    if "..." in t or "…" in t:
        return True
    if re.search(r"\b(none|unknown|unsure|uncertain|null|n/a|na)\b", t):
        return True
    if t in _PLACEHOLDER_SET:
        return True
    if re.fullmatch(r"[.·•…]+", t):
        return True
    # Too short: often a placeholder like "ok", "idk", etc.
    alnum = re.findall(r"[a-z0-9]+", t)
    if len(alnum) <= 1 and len(t) <= 6:
        return True
    return False


def _summary_has_ohrpu(summary_text: str) -> bool:
    if summary_text is None:
        return False
    s = _collapse_ws(summary_text)
    return all(re.search(rf"\b{k}\s*:\s*", s, re.IGNORECASE) for k in ["O", "H", "R", "P", "U"])


def _parse_frame_indices(text: str) -> list[int]:
    numbers = re.findall(r"\d+", text)
    return [int(n) for n in numbers]


def _normalize_answer_letter(answer_text: str, num_choices: int) -> Optional[str]:
    allowed = {chr(ord("A") + i) for i in range(max(0, num_choices))}
    if not allowed:
        allowed = {"A", "B", "C", "D", "E"}

    candidate = answer_text.strip().upper()
    if candidate in allowed:
        return candidate

    # Common variants: "A.", "(A)", "Answer: A", etc.
    match = re.search(r"\b([A-E])\b", candidate)
    if match:
        letter = match.group(1).upper()
        if letter in allowed:
            return letter

    match = re.search(r"([A-E])", candidate)
    if match:
        letter = match.group(1).upper()
        if letter in allowed:
            return letter
    return None


def _sample_uniform_indices(frame_count: int, n: int) -> list[int]:
    if frame_count <= 0:
        return list(range(n))
    if n <= 1:
        return [frame_count // 2]
    return [int(x) for x in np.linspace(0, frame_count - 1, n)]


def _propose_candidate_unseen_frames(
    frame_count: int,
    seen: set[int],
    k: int,
    rng: random.Random,
) -> list[int]:
    """Suggest a small list of unseen frames to help the model avoid repeating seen indices."""
    if frame_count <= 0 or k <= 0:
        return []
    if len(seen) >= frame_count:
        return []

    seen_sorted = sorted(i for i in seen if 0 <= i < frame_count)
    anchors = sorted(set([0, frame_count - 1, *seen_sorted]))

    candidates: list[int] = []
    gaps: list[tuple[int, int, int]] = []
    for a, b in zip(anchors, anchors[1:], strict=False):
        if b - a <= 1:
            continue
        gaps.append((b - a, a, b))
    gaps.sort(reverse=True)
    for _, a, b in gaps:
        mid = (a + b) // 2
        for d in (0, -1, 1, -2, 2, -3, 3):
            idx = mid + d
            if a < idx < b and idx not in seen and idx not in candidates:
                candidates.append(idx)
                break
        if len(candidates) >= k:
            return sorted(candidates[:k])

    need = k - len(candidates)
    if need > 0:
        remaining = [i for i in range(frame_count) if i not in seen and i not in candidates]
        if remaining:
            candidates.extend(rng.sample(remaining, k=min(need, len(remaining))))

    return sorted(candidates[:k])


def _load_video_meta(video_path: str) -> tuple[Optional[int], Optional[float]]:
    # Returns (frame_count, fps)
    try:
        import imageio

        reader = imageio.get_reader(video_path, "ffmpeg")
        meta = reader.get_meta_data()
        fps = meta.get("fps")
        nframes = meta.get("nframes")
        reader.close()
        return nframes, fps
    except Exception:
        return None, None


def _extract_frames(video_path: str, frame_indices: list[int]) -> tuple[list[Image.Image], Optional[float]]:
    if not frame_indices:
        return [], None

    # Try decord first
    try:
        import decord

        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        frames = vr.get_batch(frame_indices).asnumpy()
        images = [Image.fromarray(frame) for frame in frames]
        fps = float(vr.get_avg_fps()) if hasattr(vr, "get_avg_fps") else None
        return images, fps
    except Exception:
        pass

    # Fall back to imageio
    try:
        import imageio

        reader = imageio.get_reader(video_path, "ffmpeg")
        images = [Image.fromarray(reader.get_data(idx)) for idx in frame_indices]
        meta = reader.get_meta_data()
        fps = meta.get("fps")
        reader.close()
        return images, fps
    except Exception as exc:
        raise RuntimeError(f"Failed to extract frames from {video_path}: {exc}")


def _format_question_block(question: str, choices: list[str]) -> str:
    labels = [chr(ord("A") + i) for i in range(len(choices))]
    lines = [f"Question: {question}", "Options:"]
    for label, choice in zip(labels, choices, strict=False):
        lines.append(f"{label}. {choice}")
    if labels:
        lines.append(f"Answer with exactly one letter only: {', '.join(labels)}.")
    return "\n".join(lines)


def _build_user_content(
    question_block: str,
    summary: str,
    frame_count: int,
    round_idx: int,
    frame_indices: list[int],
    timestamps: list[Optional[float]],
    seen_frames: list[int],
) -> str:
    if question_block:
        header = f"Round {round_idx} / Question:\n{question_block}"
    else:
        header = f"Round {round_idx} (same question/options as Round 1)."

    lines = [
        header,
        f"Total frames L = {frame_count}.",
        f"Seen frames (already viewed; do NOT request these again): {_format_frame_list(seen_frames)}",
        "Current summary:",
        f"<summary>{summary}</summary>",
        "Frames shown in this round:",
    ]
    for idx, ts in zip(frame_indices, timestamps, strict=False):
        if ts is not None:
            lines.append(f"Frame {idx} (t={ts:.2f}s) <image>")
        else:
            lines.append(f"Frame {idx} <image>")
    return "\n".join(lines)


def _maybe_log_sample(payload: dict[str, Any]) -> None:
    """Optionally log a REVISE sample to disk for debugging."""
    log_dir = os.getenv("REVISE_LOG_DIR")
    if not log_dir:
        return
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "revise_samples.jsonl")

    def _strip_images(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cleaned = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                # Replace image blobs with tokens and keep text segments
                parts = []
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "image":
                        parts.append("<image>")
                    elif isinstance(c, dict) and c.get("type") == "text":
                        parts.append(c.get("text", ""))
                    else:
                        parts.append(str(c))
                content = "\n".join(parts)
            cleaned.append({**msg, "content": content})
        return cleaned

    safe_payload = payload.copy()
    if "messages" in safe_payload:
        safe_payload["messages"] = _strip_images(safe_payload["messages"])

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(safe_payload, ensure_ascii=False) + "\n")


@register("revise_agent")
class ReviseAgentLoop(AgentLoopBase):
    """Multi-round controller implementing REVISE-style frame selection."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.config
        self.prompt_length = cfg.actor_rollout_ref.rollout.prompt_length
        self.response_length = cfg.actor_rollout_ref.rollout.response_length
        # NOTE: response_length is a padding/trajectory budget; generation is capped separately.
        self.max_new_tokens = int(cfg.actor_rollout_ref.rollout.get("max_new_tokens", self.response_length))
        self.max_model_len = int(cfg.actor_rollout_ref.rollout.get("max_model_len", self.prompt_length))
        revise_cfg = cfg.actor_rollout_ref.rollout.get("revise", {})

        self.max_rounds = int(revise_cfg.get("max_rounds", 4))
        self.max_frames_per_round = int(revise_cfg.get("max_frames_per_round", 3))
        # Cap the total number of vision inputs carried across multi-round context.
        self.max_vision_inputs = int(revise_cfg.get("max_vision_inputs", 2))
        self.max_retries = int(revise_cfg.get("max_retries_per_round", 1))
        self.initial_sampling = revise_cfg.get("initial_sampling", "uniform")
        self.include_timestamps = bool(revise_cfg.get("include_timestamps", True))
        self.system_prompt_template = revise_cfg.get("system_prompt_template", DEFAULT_SYSTEM_PROMPT)
        self.seed = int(revise_cfg.get("seed", 0))

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        extra_info = kwargs.get("extra_info", {})
        reward_model = kwargs.get("reward_model", {})
        ground_truth = reward_model.get("ground_truth", {})
        question = extra_info.get("question", "")
        choices = extra_info.get("choices", [])
        video_path = extra_info.get("video_path")
        frame_count = int(extra_info.get("frame_count", 0))

        if not video_path:
            raise ValueError("extra_info.video_path is required for ReviseAgentLoop")

        # If frame count missing, try to load from video metadata
        if frame_count <= 0:
            frame_count, _ = _load_video_meta(video_path)
            frame_count = int(frame_count or 0)

        rng = random.Random(self.seed)

        # Initial summary state (avoid placeholder tokens like "unknown"/"none" to reduce copying).
        summary_state = (
            "O: no reliable observation yet; "
            "H: my belief will be updated based on what is observed; "
            "R: need evidence from frames; "
            "P: the agent has not seen any frames yet; "
            "U: key detail is still unclear"
        )

        # Sample initial frames
        if self.initial_sampling == "random" and frame_count > 0:
            init_indices = rng.sample(range(frame_count), k=min(self.max_frames_per_round, frame_count))
        else:
            init_indices = _sample_uniform_indices(frame_count, self.max_frames_per_round)

        seen_frames = []
        all_images: list[Image.Image] = []

        init_frames = [idx for idx in init_indices if idx >= 0]
        init_images, fps = _extract_frames(video_path, init_frames)
        if len(init_images) > self.max_vision_inputs:
            init_images = init_images[: self.max_vision_inputs]
            init_frames = init_frames[: len(init_images)]
        timestamps = []
        for idx in init_frames:
            if self.include_timestamps and fps:
                timestamps.append(idx / fps)
            else:
                timestamps.append(None)

        all_images.extend(init_images)
        seen_frames.extend(init_frames)

        question_block = _format_question_block(question, choices)
        system_prompt = self.system_prompt_template.format(max_frames_per_round=self.max_frames_per_round)

        user_content = _build_user_content(
            question_block,
            summary_state,
            frame_count,
            round_idx=1,
            frame_indices=init_frames,
            timestamps=timestamps,
            seen_frames=seen_frames,
        )

        def _with_images(message_list, images):
            """Convert plain-text message content to multimodal lists containing the
            provided images followed by the text. Needed for Qwen2.5-VL so that
            the processor can emit vision placeholders matching `image_data`.

            Only attach images to the final (typically user) message to avoid
            duplicating placeholders across system/assistant turns."""
            if not images:
                return message_list
            new_msgs = []
            for idx, m in enumerate(message_list):
                content = m["content"]
                # Only the last message gets the images.
                if idx != len(message_list) - 1 or isinstance(content, list):
                    new_msgs.append(m)
                    continue
                content_list = [{"type": "image", "image": img} for img in images]
                content_list.append({"type": "text", "text": content})
                new_msgs.append({**m, "content": content_list})
            return new_msgs

        messages = _with_images(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            all_images,
        )

        prompt_ids = await self.apply_chat_template(messages, images=all_images)
        # Guard against rare cases where vision tokens make the prompt exceed vLLM's
        # max_model_len (e.g., very large frames). Fall back to fewer initial frames.
        # Keep some extra headroom beyond max_new_tokens to avoid max_possible_tokens=0 edge cases.
        min_generation_room = int(self.max_new_tokens) + 32
        max_prompt_len = max(0, self.max_model_len - min_generation_room)
        while len(prompt_ids) > max_prompt_len and all_images:
            all_images = all_images[:-1]
            init_frames = init_frames[: len(all_images)]
            seen_frames = seen_frames[: len(all_images)]
            timestamps = timestamps[: len(all_images)]

            user_content = _build_user_content(
                question_block,
                summary_state,
                frame_count,
                round_idx=1,
                frame_indices=init_frames,
                timestamps=timestamps,
                seen_frames=seen_frames,
            )
            messages = _with_images(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                all_images,
            )
            prompt_ids = await self.apply_chat_template(messages, images=all_images)
        response_mask: list[int] = []
        response_logprobs: list[float] = []

        num_rounds = 0
        answer_text: Optional[str] = None
        format_valid = False
        last_response_text = ""
        invalid_outputs = 0
        total_retries = 0
        frames_all_seen = 0

        retries_left = [self.max_retries]

        async def _retry_invalid(feedback: str, *, force_answer: bool = False) -> bool:
            nonlocal invalid_outputs, total_retries
            invalid_outputs += 1
            ok = await self._retry_with_feedback(
                feedback,
                messages,
                prompt_ids,
                response_mask,
                response_logprobs,
                retries_left,
                force_answer=force_answer,
            )
            if ok:
                total_retries += 1
            return ok

        for round_idx in range(1, self.max_rounds + 1):
            num_rounds = round_idx

            if len(prompt_ids) >= self.max_model_len:
                logger.warning(
                    "Prompt already at/over max_model_len (%s >= %s); stopping sample early.",
                    len(prompt_ids),
                    self.max_model_len,
                )
                break

            with simple_timer("generate_sequences", {}):
                try:
                    output = await self.server_manager.generate(
                        request_id=uuid4().hex,
                        prompt_ids=prompt_ids,
                        sampling_params=sampling_params,
                        image_data=all_images,
                        video_data=None,
                    )
                except Exception as exc:
                    logger.warning("vLLM generation failed (%s); stopping sample early.", exc)
                    break

            # Update response tracking
            response_ids = output.token_ids
            prompt_ids += response_ids
            response_mask += [1] * len(response_ids)
            if output.log_probs:
                response_logprobs += output.log_probs

            last_response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": last_response_text})

            # Parse model output
            think = _extract_tag(last_response_text, _THINK_RE)
            answer = _extract_tag(last_response_text, _ANSWER_RE)
            frames_text = _extract_tag(last_response_text, _FRAMES_RE)
            summary = _extract_tag(last_response_text, _SUMMARY_RE)

            if think is None or _is_placeholder(think):
                feedback = (
                    "Invalid response: <think> is missing or a placeholder. "
                    "Provide 1–2 short sentences describing your decision (not the final answer)."
                )
                if not await _retry_invalid(feedback, force_answer=bool(answer)):
                    break
                continue

            if answer:
                if summary is None or _is_placeholder(summary) or not _summary_has_ohrpu(summary):
                    feedback = (
                        "Invalid response: when answering, include a meaningful <summary> with O/H/R/P/U "
                        "(no placeholders like '.../none/unknown')."
                    )
                    if not await _retry_invalid(feedback, force_answer=True):
                        break
                    continue
                normalized = _normalize_answer_letter(answer, len(choices))
                if normalized is None:
                    allowed = [chr(ord("A") + i) for i in range(len(choices) or 5)]
                    feedback = (
                        "Invalid response: <answer> must be exactly ONE option letter "
                        f"({', '.join(allowed)}). Do not output words or a sentence."
                    )
                    if not await _retry_invalid(feedback, force_answer=True):
                        break
                    continue
                if summary is not None:
                    summary_state = summary
                answer_text = normalized
                format_valid = True
                break

            if frames_text is None:
                # invalid: missing frames for select
                feedback = (
                    "Invalid response: missing <frames> tag for requesting more frames. "
                    "Remember: <frames> must list NEW frame indices to view NEXT (not already seen)."
                )
                if not await _retry_invalid(feedback):
                    break
                continue

            # Validate summary
            if summary is None or _is_placeholder(summary) or not _summary_has_ohrpu(summary):
                feedback = (
                    "Invalid response: missing or invalid <summary> tag. "
                    "Include a meaningful <summary> with O/H/R/P/U (no placeholders like '.../none/unknown')."
                )
                if not await _retry_invalid(feedback):
                    break
                continue

            requested = _parse_frame_indices(frames_text)
            if not requested:
                feedback = (
                    "Invalid response: <frames> is empty. "
                    "Provide 1–{max_k} NEW frame indices to view NEXT (comma-separated). "
                    "Do not request frames already in Seen frames."
                ).format(max_k=self.max_frames_per_round)
                if not await _retry_invalid(feedback):
                    break
                continue

            # Enforce constraints
            requested_unique = [i for i in requested if i not in seen_frames]
            if len(requested_unique) == 0:
                frames_all_seen += 1
                candidates = _propose_candidate_unseen_frames(
                    frame_count=frame_count,
                    seen=set(seen_frames),
                    k=max(12, self.max_frames_per_round * 4),
                    rng=rng,
                )
                candidate_text = f" Unseen candidates: {_format_frame_list(candidates)}." if candidates else ""
                feedback = (
                    "Invalid response: all requested frames are already seen. "
                    "In <frames>, output 1–{max_k} NEW indices NOT in the Seen frames list."
                    + candidate_text
                ).format(max_k=self.max_frames_per_round)
                if not await _retry_invalid(feedback):
                    break
                continue
            if len(requested_unique) > self.max_frames_per_round:
                feedback = (
                    f"Invalid response: requested {len(requested_unique)} NEW frames, "
                    f"which exceeds max {self.max_frames_per_round}. "
                    f"Choose at most {self.max_frames_per_round} NEW frames NOT in Seen frames."
                )
                if not await _retry_invalid(feedback):
                    break
                continue

            # Clip to range
            valid_requested = [i for i in requested_unique if 0 <= i < frame_count]
            if not valid_requested:
                feedback = (
                    "Invalid response: frame indices out of range. "
                    f"Valid range is [0, {max(0, frame_count - 1)}]. "
                    "Choose NEW frame indices within range and NOT in Seen frames."
                )
                if not await _retry_invalid(feedback):
                    break
                continue

            # Enforce a global cap on total images carried across rounds.
            slots_left = self.max_vision_inputs - len(all_images)
            if slots_left <= 0:
                feedback = (
                    f"Vision limit of {self.max_vision_inputs} images reached. "
                    "Provide the final answer using <answer>...</answer>."
                )
                await self._retry_with_feedback(
                    feedback,
                    messages,
                    prompt_ids,
                    response_mask,
                    response_logprobs,
                    retries_left,
                    force_answer=True,
                )
                continue

            if len(valid_requested) > slots_left:
                valid_requested = valid_requested[:slots_left]

            summary_state = summary
            candidate_images, fps = _extract_frames(video_path, valid_requested)
            candidate_timestamps: list[Optional[float]] = []
            for idx in valid_requested:
                if self.include_timestamps and fps:
                    candidate_timestamps.append(idx / fps)
                else:
                    candidate_timestamps.append(None)

            # Guard against vLLM context overflow (prompt must be <= max_model_len).
            # Images can contribute many tokens; for rare near-limit cases, reduce the
            # number of frames added for the next round and leave headroom to answer.
            min_generation_room = int(self.max_new_tokens) + 32
            max_prompt_len = max(0, self.max_model_len - min_generation_room)

            selected_frames: list[int] = []
            selected_images: list[Image.Image] = []
            selected_user_ids: Optional[list[int]] = None
            selected_messages: Optional[list[dict[str, Any]]] = None

            for k in range(len(valid_requested), 0, -1):
                trial_frames = valid_requested[:k]
                trial_images = candidate_images[:k]
                trial_timestamps = candidate_timestamps[:k]

                trial_user_content = _build_user_content(
                    "",
                    summary_state,
                    frame_count,
                    round_idx=round_idx + 1,
                    frame_indices=trial_frames,
                    timestamps=trial_timestamps,
                    seen_frames=seen_frames + trial_frames,
                )
                trial_messages = _with_images(
                    [{"role": "user", "content": trial_user_content}],
                    trial_images,
                )
                trial_user_ids = await self.apply_chat_template(
                    trial_messages,
                    images=trial_images,
                    remove_system_prompt=True,
                )

                # Also keep trajectory within the configured response_length budget to avoid
                # truncating inside a vision token block (which can break Qwen2.5-VL RoPE indexing).
                response_budget_ok = (
                    len(response_mask) + len(trial_user_ids) + int(self.max_new_tokens) + 16
                    <= int(self.response_length)
                )

                if len(prompt_ids) + len(trial_user_ids) <= max_prompt_len and response_budget_ok:
                    selected_frames = trial_frames
                    selected_images = trial_images
                    selected_user_ids = trial_user_ids
                    selected_messages = trial_messages
                    break

            if selected_user_ids is None or selected_messages is None:
                feedback = (
                    "Context/trajectory length limit reached. Provide the final answer now using <answer> tags."
                )
                await self._retry_with_feedback(
                    feedback,
                    messages,
                    prompt_ids,
                    response_mask,
                    response_logprobs,
                    retries_left,
                    force_answer=True,
                )
                continue

            all_images.extend(selected_images)
            seen_frames.extend(selected_frames)
            messages.extend(selected_messages)
            prompt_ids += selected_user_ids
            response_mask += [0] * len(selected_user_ids)
            if response_logprobs:
                response_logprobs += [0.0] * len(selected_user_ids)

        # If no answer was produced, force a final answer attempt
        if answer_text is None:
            feedback = "Max rounds reached. Provide final answer now using <answer> tags."
            await self._retry_with_feedback(
                feedback,
                messages,
                prompt_ids,
                response_mask,
                response_logprobs,
                retries_left,
                force_answer=True,
            )
            if len(prompt_ids) < self.max_model_len:
                try:
                    output = await self.server_manager.generate(
                        request_id=uuid4().hex,
                        prompt_ids=prompt_ids,
                        sampling_params=sampling_params,
                        image_data=all_images,
                        video_data=None,
                    )
                except Exception as exc:
                    logger.warning("vLLM final-answer generation failed (%s); leaving answer empty.", exc)
                    output = None
            else:
                output = None

            if output is not None:
                response_ids = output.token_ids
                prompt_ids += response_ids
                response_mask += [1] * len(response_ids)
                if output.log_probs:
                    response_logprobs += output.log_probs
                last_response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                messages.append({"role": "assistant", "content": last_response_text})
                answer_text = _extract_tag(last_response_text, _ANSWER_RE)
                forced_summary = _extract_tag(last_response_text, _SUMMARY_RE)
                if forced_summary is not None:
                    summary_state = forced_summary
                format_valid = answer_text is not None

        # Prepare output
        response_ids = prompt_ids[-len(response_mask) :] if response_mask else []
        prompt_only_ids = prompt_ids[: len(prompt_ids) - len(response_mask)] if response_mask else prompt_ids

        revise_metrics = {
            "num_rounds": num_rounds,
            "frames_used": len(seen_frames),
            "seen_frames": seen_frames,
            "summary": summary_state,
            "answer": answer_text,
            "format_valid": format_valid,
            "last_response": last_response_text,
            "invalid_outputs": invalid_outputs,
            "total_retries": total_retries,
            "frames_all_seen": frames_all_seen,
        }

        # Optional: summary-only correctness approximation
        if ground_truth and summary_state:
            gt_text = ground_truth.get("answer_text")
            if gt_text:
                revise_metrics["summary_contains_answer"] = str(gt_text).lower() in summary_state.lower()

        _maybe_log_sample(
            {
                "timestamp": time.time(),
                "video_id": extra_info.get("video_id"),
                "video_path": video_path,
                "question": question,
                "choices": choices,
                "ground_truth": ground_truth,
                "messages": messages,
                "seen_frames": seen_frames,
                "num_rounds": num_rounds,
                "summary": summary_state,
                "answer": answer_text,
                "format_valid": format_valid,
                "revise_metrics": revise_metrics,
            }
        )

        output = AgentLoopOutput(
            prompt_ids=prompt_only_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            multi_modal_data={"images": all_images},
            num_turns=1 + 2 * num_rounds,
            metrics={},
            extra_fields={"revise": revise_metrics},
        )
        return output

    async def _retry_with_feedback(
        self,
        feedback: str,
        messages: list[dict[str, Any]],
        prompt_ids: list[int],
        response_mask: list[int],
        response_logprobs: list[float],
        retries_left: list[int],
        force_answer: bool = False,
    ) -> bool:
        """Append feedback and request a retry. Returns False if retry budget exceeded."""
        if retries_left[0] <= 0 and not force_answer:
            return False

        feedback_text = (
            f"{feedback}\n"
            "Please respond with one of the required formats."
        )
        if force_answer:
            feedback_text = (
                f"{feedback}\n"
                "Use <answer>LETTER</answer> where LETTER is a single option letter (e.g., A/B/C/D/E)."
            )

        add_messages = [{"role": "user", "content": feedback_text}]
        messages.extend(add_messages)
        user_ids = await self.apply_chat_template(add_messages, remove_system_prompt=True)
        prompt_ids += user_ids
        response_mask += [0] * len(user_ids)
        if response_logprobs:
            response_logprobs += [0.0] * len(user_ids)

        if force_answer:
            return True

        retries_left[0] -= 1
        return True
