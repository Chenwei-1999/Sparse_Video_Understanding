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


DEFAULT_SYSTEM_PROMPT = (
    "You are REVISE, a multi-round video reasoning agent.\n"
    "Each round you will see: (1) a multiple-choice question with options, (2) the current belief summary, "
    "and (3) a few sampled video frames.\n"
    "Goal: answer the question as soon as you have a best guess.\n"
    "Request more frames ONLY if the current frames are insufficient to choose among the options.\n\n"
    "IMPORTANT: Output must follow EXACTLY ONE of the two formats below. Do not output any text outside tags.\n\n"
    "Format 1 — Select more frames (use this only if NOT confident):\n"
    "<think>...</think>\n"
    "<summary>O: ...; H: ...; R: ...; P: ...; U: ...</summary>\n"
    "<frames>i, j, k</frames>\n\n"
    "Format 2 — Answer now (use this if confident):\n"
    "<think>...</think>\n"
    "<answer>LETTER</answer>\n\n"
    "Tag meanings:\n"
    "- <think>: your private reasoning to decide whether to select or answer.\n"
    "- <summary>: the ONLY persistent memory across rounds. Keep it short and update it every time you select frames.\n"
    "  - O (Observations): what you currently observe in the selected frames.\n"
    "  - H (Hypothesis): updated beliefs and your current answer candidate (include the option letter).\n"
    "  - R (Reasons): why you need more frames and what evidence you are looking for next.\n"
    "  - P (Previously seen): which frames have already been used/seen.\n"
    "  - U (Uncertainties): what is still unknown or ambiguous.\n\n"
    "Rules:\n"
    "- Frame indices are 0-based in [0, L-1].\n"
    "- If you are confident, answer instead of requesting more frames.\n"
    "- If selecting, request 1 to {max_frames_per_round} NEW frames (not already seen).\n"
    "- In <frames>, output comma-separated integers only.\n"
    "- In <answer>, output EXACTLY ONE option letter shown in the question (e.g., A/B/C/D/E). No words/punctuation.\n"
)


def _extract_tag(text: str, tag_re: re.Pattern[str]) -> Optional[str]:
    matches = list(tag_re.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1).strip()


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
    lines = [
        f"Round {round_idx} / Question:\n{question_block}",
        f"Total frames L = {frame_count}.",
        f"Seen frames: {seen_frames}",
        "Current summary:",
        f"<summary>{summary}</summary>",
        "Frames for this round:",
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
        revise_cfg = cfg.actor_rollout_ref.rollout.get("revise", {})

        self.max_rounds = int(revise_cfg.get("max_rounds", 4))
        self.max_frames_per_round = int(revise_cfg.get("max_frames_per_round", 3))
        # Qwen2.5-VL currently supports up to 2 vision inputs; keep a hard cap.
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

        # Initial summary state
        summary_state = "O: none; H: none; R: need evidence; P: []; U: unknown"

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
        response_mask: list[int] = []
        response_logprobs: list[float] = []

        num_rounds = 0
        answer_text: Optional[str] = None
        format_valid = False
        last_response_text = ""

        retries_left = [self.max_retries]
        for round_idx in range(1, self.max_rounds + 1):
            num_rounds = round_idx

            with simple_timer("generate_sequences", {}):
                output = await self.server_manager.generate(
                    request_id=uuid4().hex,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    image_data=all_images,
                    video_data=None,
                )

            # Update response tracking
            response_ids = output.token_ids
            prompt_ids += response_ids
            response_mask += [1] * len(response_ids)
            if output.log_probs:
                response_logprobs += output.log_probs

            last_response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": last_response_text})

            # Parse model output
            answer = _extract_tag(last_response_text, _ANSWER_RE)
            frames_text = _extract_tag(last_response_text, _FRAMES_RE)
            summary = _extract_tag(last_response_text, _SUMMARY_RE)

            if answer:
                normalized = _normalize_answer_letter(answer, len(choices))
                if normalized is None:
                    allowed = [chr(ord("A") + i) for i in range(len(choices) or 5)]
                    feedback = (
                        "Invalid response: <answer> must be exactly ONE option letter "
                        f"({', '.join(allowed)}). Do not output words or a sentence."
                    )
                    if not await self._retry_with_feedback(
                        feedback,
                        messages,
                        prompt_ids,
                        response_mask,
                        response_logprobs,
                        retries_left,
                        force_answer=True,
                    ):
                        break
                    continue
                answer_text = normalized
                format_valid = True
                break

            if frames_text is None:
                # invalid: missing frames for select
                feedback = (
                    "Invalid response: missing <frames> tag for selection. "
                    "Please follow the required format."
                )
                if not await self._retry_with_feedback(
                    feedback,
                    messages,
                    prompt_ids,
                    response_mask,
                    response_logprobs,
                    retries_left,
                ):
                    break
                continue

            # Validate summary
            if summary is None:
                feedback = "Invalid response: missing <summary> tag for selection."
                if not await self._retry_with_feedback(
                    feedback,
                    messages,
                    prompt_ids,
                    response_mask,
                    response_logprobs,
                    retries_left,
                ):
                    break
                continue

            requested = _parse_frame_indices(frames_text)
            if not requested:
                feedback = "Invalid response: <frames> is empty. Provide frame indices."
                if not await self._retry_with_feedback(
                    feedback,
                    messages,
                    prompt_ids,
                    response_mask,
                    response_logprobs,
                    retries_left,
                ):
                    break
                continue

            # Enforce constraints
            requested_unique = [i for i in requested if i not in seen_frames]
            if len(requested_unique) == 0:
                feedback = "Invalid response: requested frames already seen."
                if not await self._retry_with_feedback(
                    feedback,
                    messages,
                    prompt_ids,
                    response_mask,
                    response_logprobs,
                    retries_left,
                ):
                    break
                continue
            if len(requested_unique) > self.max_frames_per_round:
                feedback = (
                    f"Invalid response: requested {len(requested_unique)} frames, "
                    f"exceeds max {self.max_frames_per_round}."
                )
                if not await self._retry_with_feedback(
                    feedback,
                    messages,
                    prompt_ids,
                    response_mask,
                    response_logprobs,
                    retries_left,
                ):
                    break
                continue

            # Clip to range
            valid_requested = [i for i in requested_unique if 0 <= i < frame_count]
            if not valid_requested:
                feedback = "Invalid response: frame indices out of range."
                if not await self._retry_with_feedback(
                    feedback,
                    messages,
                    prompt_ids,
                    response_mask,
                    response_logprobs,
                    retries_left,
                ):
                    break
                continue

            # Enforce global vision cap (Qwen2.5-VL supports 2 images max)
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
            new_images, fps = _extract_frames(video_path, valid_requested)
            timestamps = []
            for idx in valid_requested:
                if self.include_timestamps and fps:
                    timestamps.append(idx / fps)
                else:
                    timestamps.append(None)

            all_images.extend(new_images)
            seen_frames.extend(valid_requested)

            user_content = _build_user_content(
                question_block,
                summary_state,
                frame_count,
                round_idx=round_idx + 1,
                frame_indices=valid_requested,
                timestamps=timestamps,
                seen_frames=seen_frames,
            )
            add_messages = _with_images(
                [{"role": "user", "content": user_content}],
                new_images,
            )
            messages.extend(add_messages)

            user_ids = await self.apply_chat_template(
                add_messages,
                images=new_images,
                remove_system_prompt=True,
            )
            prompt_ids += user_ids
            response_mask += [0] * len(user_ids)
            if response_logprobs:
                response_logprobs += [0.0] * len(user_ids)

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
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=all_images,
                video_data=None,
            )
            response_ids = output.token_ids
            prompt_ids += response_ids
            response_mask += [1] * len(response_ids)
            if output.log_probs:
                response_logprobs += output.log_probs
            last_response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": last_response_text})
            answer_text = _extract_tag(last_response_text, _ANSWER_RE)
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
