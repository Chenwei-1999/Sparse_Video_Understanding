from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def _first_existing(candidates: list[str | Path]) -> str | None:
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.exists():
            return str(path)
    return None


def _env_or_existing(env_name: str, candidates: list[str | Path]) -> str | None:
    env_val = os.getenv(env_name, "").strip()
    if env_val and Path(env_val).expanduser().exists():
        return str(Path(env_val).expanduser())
    return _first_existing(candidates)


def _metadata_version(dist_name: str) -> str | None:
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        module_name = dist_name.replace("-", "_")
        try:
            module = __import__(module_name)
        except Exception:
            return None
        return getattr(module, "__version__", "installed")


def _run(cmd: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception as exc:
        return 1, str(exc)
    output = (proc.stdout or proc.stderr or "").strip()
    return proc.returncode, output


def _probe_videoespresso_mc(json_path: str | None) -> dict[str, Any]:
    out = {"path": json_path, "multiple_choice": False, "reason": "missing"}
    if not json_path or not Path(json_path).exists():
        return out
    try:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    except Exception as exc:
        out["reason"] = f"json_error: {type(exc).__name__}"
        return out
    if not isinstance(data, list) or not data:
        out["reason"] = "empty_or_non_list"
        return out
    probe = data[: min(32, len(data))]
    has_mc = True
    for row in probe:
        if not isinstance(row, dict):
            has_mc = False
            break
        options = row.get("options") or row.get("choices")
        answer = row.get("correct_answer")
        if not isinstance(options, list) or len(options) < 2 or answer in (None, ""):
            has_mc = False
            break
    out["multiple_choice"] = has_mc
    out["reason"] = "ok" if has_mc else "missing_options_or_correct_answer"
    return out


def discover_assets() -> dict[str, Any]:
    nextqa_root = _env_or_existing("REVISE_NEXTQA_ROOT", ["/shares/hlw3876/chenwei/NExT-QA"])
    nextqa_video_root = _env_or_existing(
        "REVISE_NEXTQA_VIDEO_ROOT",
        [Path(nextqa_root) / "NExTVideo"] if nextqa_root else [],
    )
    nextqa_map_json = _env_or_existing(
        "REVISE_NEXTQA_MAP_JSON",
        [Path(nextqa_root) / "map_vid_vidorID.json"] if nextqa_root else [],
    )
    nextqa_train_csv = _env_or_existing(
        "REVISE_NEXTQA_TRAIN_CSV",
        [Path(nextqa_root) / "nextqa" / "train.csv"] if nextqa_root else [],
    )
    nextqa_val_csv = _env_or_existing(
        "REVISE_NEXTQA_VAL_CSV",
        [Path(nextqa_root) / "nextqa" / "val.csv"] if nextqa_root else [],
    )
    nextqa_captions_dir = _env_or_existing(
        "REVISE_NEXTQA_CAPTIONS_DIR",
        [
            Path(nextqa_root) / "captions",
            Path(nextqa_root) / "captions_1fps",
            Path(nextqa_root) / "video_captions",
        ]
        if nextqa_root
        else [],
    )

    ve_root = _env_or_existing("REVISE_VIDEOESPRESSO_ROOT", ["/shares/hlw3876/chenwei/VideoEspresso"])
    ve_test_json = _env_or_existing(
        "REVISE_VIDEOESPRESSO_TEST_JSON",
        [Path(ve_root) / "test_video" / "bench_hard.json"] if ve_root else [],
    )
    ve_test_video_root = _env_or_existing(
        "REVISE_VIDEOESPRESSO_TEST_VIDEO_ROOT",
        [Path(ve_root) / "test_video"] if ve_root else [],
    )
    ve_train_video_json = _env_or_existing(
        "REVISE_VIDEOESPRESSO_TRAIN_VIDEO_JSON",
        [Path(ve_root) / "train_video" / "videoespresso_train_video.json"] if ve_root else [],
    )
    ve_train_multi_json = _env_or_existing(
        "REVISE_VIDEOESPRESSO_TRAIN_MULTI_JSON",
        [Path(ve_root) / "train_multi_image" / "videoespresso_train.json"] if ve_root else [],
    )
    ve_mc_train_json = _env_or_existing(
        "REVISE_VIDEOESPRESSO_MC_TRAIN_JSON",
        [
            Path(ve_root) / "train_video" / "videoespresso_train_mc.json",
            Path(ve_root) / "train_video" / "videoespresso_train_multiple_choice.json",
        ]
        if ve_root
        else [],
    )

    egoschema_video_root = _env_or_existing(
        "REVISE_EGOSCHEMA_VIDEO_ROOT",
        [
            "/shares/hlw3876/chenwei/egoschema/hf_egoschema",
            "/shares/hlw3876/chenwei/EgoSchema",
        ],
    )
    egoschema_json = _env_or_existing(
        "REVISE_EGOSCHEMA_JSON",
        [
            "/shares/hlw3876/chenwei/egoschema/pnp_subset_500.json",
            "/shares/hlw3876/chenwei/EgoSchema/pnp_subset_500.json",
        ],
    )

    video_cache_dir = _env_or_existing("REVISE_VIDEO_CACHE_DIR", ["/tmp/chenwei_video_cache", "/tmp/video_cache"])

    qwen_3b = _env_or_existing(
        "REVISE_QWEN25_VL_3B_PATH",
        [
            "/shares/hlw3876/chenwei/hf_cache/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3",
            "/shares/hlw3876/chenwei/hf_cache/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3",
        ],
    )
    qwen_7b = _env_or_existing(
        "REVISE_QWEN25_VL_7B_PATH",
        [
            "/shares/hlw3876/chenwei/hf_cache/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5",
            "/shares/hlw3876/chenwei/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5",
        ],
    )

    gpu_rc, gpu_out = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )

    return {
        "repo_root": str(REPO_ROOT),
        "python": {"path": sys.executable, "version": sys.version.split()[0]},
        "packages": {
            "torch": _metadata_version("torch"),
            "transformers": _metadata_version("transformers"),
            "vllm": _metadata_version("vllm"),
            "sglang": _metadata_version("sglang"),
            "decord": _metadata_version("decord"),
            "imageio": _metadata_version("imageio"),
            "datasets": _metadata_version("datasets"),
            "hydra-core": _metadata_version("hydra-core"),
            "ray": _metadata_version("ray"),
            "wandb": _metadata_version("wandb"),
            "scikit-learn": _metadata_version("scikit-learn"),
        },
        "gpu": {"available": gpu_rc == 0, "raw": gpu_out},
        "models": {
            "qwen25_vl_3b": qwen_3b,
            "qwen25_vl_7b": qwen_7b,
        },
        "remote_api": {
            "base_url": os.getenv("REVISE_API_BASE_URL", "").strip() or None,
            "model_id": os.getenv("REVISE_MODEL_ID", "").strip() or None,
            "api_key_present": bool(
                os.getenv("REVISE_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("VLLM_API_KEY")
            ),
        },
        "datasets": {
            "nextqa": {
                "root": nextqa_root,
                "video_root": nextqa_video_root,
                "map_json": nextqa_map_json,
                "train_csv": nextqa_train_csv,
                "val_csv": nextqa_val_csv,
                "captions_dir": nextqa_captions_dir,
            },
            "videoespresso": {
                "root": ve_root,
                "test_json": ve_test_json,
                "test_video_root": ve_test_video_root,
                "train_video_json": ve_train_video_json,
                "train_multi_json": ve_train_multi_json,
                "mc_train_json": ve_mc_train_json,
                "public_train_probe": _probe_videoespresso_mc(ve_train_video_json),
                "mc_train_probe": _probe_videoespresso_mc(ve_mc_train_json),
            },
            "egoschema": {
                "video_root": egoschema_video_root,
                "json": egoschema_json,
            },
            "video_cache": {
                "root": video_cache_dir,
                "lvbench_dir": str(Path(video_cache_dir) / "lvbench") if video_cache_dir else None,
                "videomme_dir": str(Path(video_cache_dir) / "videomme") if video_cache_dir else None,
            },
        },
    }
