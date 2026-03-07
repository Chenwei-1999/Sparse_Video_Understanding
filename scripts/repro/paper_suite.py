#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.repro.common import REPO_ROOT, discover_assets


ExperimentBuilder = Callable[[dict, bool, Path], list[str] | str]
ExperimentChecker = Callable[[dict], list[str]]


def _python_bin() -> str:
    return os.getenv("REVISE_PYTHON", sys.executable)


def _model_endpoint_args(assets: dict, *, prefer_7b: bool, port: int) -> list[str]:
    remote = assets["remote_api"]
    if remote.get("base_url") and remote.get("model_id"):
        return [
            "--model-path",
            str(remote["model_id"]),
            "--base-url",
            str(remote["base_url"]),
            "--model-id",
            str(remote["model_id"]),
        ]
    models = assets["models"]
    model_path = models.get("qwen25_vl_7b") if prefer_7b else models.get("qwen25_vl_3b")
    if not model_path:
        model_path = models.get("qwen25_vl_3b") or models.get("qwen25_vl_7b")
    return [
        "--model-path",
        str(model_path),
        "--start-server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tensor-parallel-size",
        "1",
        "--dtype",
        "bfloat16",
        "--gpu-memory-utilization",
        "0.55",
    ]


def _require_model_or_api(assets: dict) -> list[str]:
    remote = assets["remote_api"]
    models = assets["models"]
    if remote.get("base_url") and remote.get("model_id"):
        return []
    if models.get("qwen25_vl_3b") or models.get("qwen25_vl_7b"):
        return []
    return ["No remote OpenAI-compatible API configured and no local Qwen2.5-VL model path found."]


def _require_nextqa(assets: dict) -> list[str]:
    nextqa = assets["datasets"]["nextqa"]
    missing = []
    for key in ("video_root", "map_json", "val_csv"):
        if not nextqa.get(key):
            missing.append(f"NExT-QA {key} missing")
    return missing + _require_model_or_api(assets)


def _require_nextqa_captions(assets: dict) -> list[str]:
    missing = _require_nextqa(assets)
    if not assets["datasets"]["nextqa"].get("captions_dir"):
        missing.append("NExT-QA captions_dir missing (set REVISE_NEXTQA_CAPTIONS_DIR).")
    return missing


def _require_videoespresso(assets: dict) -> list[str]:
    ve = assets["datasets"]["videoespresso"]
    missing = []
    for key in ("test_json", "test_video_root"):
        if not ve.get(key):
            missing.append(f"VideoEspresso {key} missing")
    return missing + _require_model_or_api(assets)


def _require_egoschema(assets: dict) -> list[str]:
    eg = assets["datasets"]["egoschema"]
    missing = []
    for key in ("video_root", "json"):
        if not eg.get(key):
            missing.append(f"EgoSchema {key} missing")
    return missing + _require_model_or_api(assets)


def _require_nextqa_training(assets: dict) -> list[str]:
    missing = _require_nextqa(assets)
    if not assets["models"].get("qwen25_vl_3b"):
        missing.append("Local Qwen2.5-VL-3B checkpoint required for training.")
    if not assets["models"].get("qwen25_vl_7b") and not (
        assets["remote_api"].get("base_url") and assets["remote_api"].get("model_id")
    ):
        missing.append("Teacher generation needs local Qwen2.5-VL-7B or a remote API.")
    return missing


def _require_videoespresso_training(assets: dict) -> list[str]:
    missing = _require_videoespresso(assets)
    probe = assets["datasets"]["videoespresso"]["mc_train_probe"]
    if not probe.get("multiple_choice"):
        missing.append(
            "VideoEspresso MC training JSON missing. Public train files on disk are open-ended and cannot drive current RL code."
        )
    if not assets["models"].get("qwen25_vl_3b"):
        missing.append("Local Qwen2.5-VL-3B checkpoint required for training.")
    return missing


def _cmd_nextqa_pnp(assets: dict, smoke: bool, out_dir: Path) -> list[str]:
    nextqa = assets["datasets"]["nextqa"]
    cmd = [
        _python_bin(),
        str(REPO_ROOT / "examples/revise/plug_and_play_nextqa_vllm.py"),
        "--video-root",
        nextqa["video_root"],
        "--map-json",
        nextqa["map_json"],
        "--csv",
        nextqa["val_csv"],
        "--seed",
        "0",
        "--max-rounds",
        "2" if smoke else "4",
        "--max-frames-per-round",
        "3",
        "--max-samples",
        "1" if smoke else "0",
        "--summary-json",
        str(out_dir / "nextqa_pnp.summary.json"),
        "--log-jsonl",
        str(out_dir / "nextqa_pnp.jsonl"),
    ]
    cmd += _model_endpoint_args(assets, prefer_7b=not smoke, port=18000)
    return cmd


def _cmd_nextqa_oneshot(assets: dict, smoke: bool, out_dir: Path) -> list[str]:
    nextqa = assets["datasets"]["nextqa"]
    cmd = [
        _python_bin(),
        str(REPO_ROOT / "examples/revise/oneshot_local_mc_vllm.py"),
        "--dataset",
        "nextqa",
        "--video-root",
        nextqa["video_root"],
        "--map-json",
        nextqa["map_json"],
        "--csv",
        nextqa["val_csv"],
        "--max-frames",
        "8",
        "--max-samples",
        "1" if smoke else "0",
        "--summary-json",
        str(out_dir / "nextqa_oneshot.summary.json"),
        "--log-jsonl",
        str(out_dir / "nextqa_oneshot.jsonl"),
    ]
    cmd += _model_endpoint_args(assets, prefer_7b=False, port=18001)
    return cmd


def _cmd_nextqa_caption(assets: dict, smoke: bool, out_dir: Path) -> list[str]:
    nextqa = assets["datasets"]["nextqa"]
    cmd = [
        _python_bin(),
        str(REPO_ROOT / "examples/revise/eval_nextqa_caption_vllm.py"),
        "--video-root",
        nextqa["video_root"],
        "--map-json",
        nextqa["map_json"],
        "--csv",
        nextqa["val_csv"],
        "--captions-dir",
        nextqa["captions_dir"],
        "--max-samples",
        "1" if smoke else "0",
        "--summary-json",
        str(out_dir / "nextqa_caption.summary.json"),
        "--log-jsonl",
        str(out_dir / "nextqa_caption.jsonl"),
    ]
    cmd += _model_endpoint_args(assets, prefer_7b=False, port=19000)
    return cmd


def _cmd_nextqa_videoagent(assets: dict, smoke: bool, out_dir: Path, *, official: bool) -> list[str]:
    nextqa = assets["datasets"]["nextqa"]
    script = (
        "eval_nextqa_videoagent_officialstyle_caption_vllm.py"
        if official
        else "eval_nextqa_videoagent_caption_vllm.py"
    )
    prefix = "nextqa_videoagent_official" if official else "nextqa_videoagent"
    cmd = [
        _python_bin(),
        str(REPO_ROOT / "examples/videoagent" / script),
        "--video-root",
        nextqa["video_root"],
        "--map-json",
        nextqa["map_json"],
        "--csv",
        nextqa["val_csv"],
        "--captions-dir",
        nextqa["captions_dir"],
        "--max-samples",
        "1" if smoke else "0",
        "--summary-json",
        str(out_dir / f"{prefix}.summary.json"),
        "--log-jsonl",
        str(out_dir / f"{prefix}.jsonl"),
    ]
    cmd += _model_endpoint_args(assets, prefer_7b=False, port=18200 if not official else 18201)
    return cmd


def _cmd_videoespresso_pnp(assets: dict, smoke: bool, out_dir: Path) -> list[str]:
    ve = assets["datasets"]["videoespresso"]
    cmd = [
        _python_bin(),
        str(REPO_ROOT / "examples/revise/plug_and_play_egoschema_vllm.py"),
        "--dataset-name",
        "videoespresso",
        "--json",
        ve["test_json"],
        "--video-root",
        ve["test_video_root"],
        "--max-rounds",
        "2" if smoke else "4",
        "--max-frames-per-round",
        "3",
        "--max-samples",
        "1" if smoke else "0",
        "--summary-json",
        str(out_dir / "videoespresso_pnp.summary.json"),
        "--log-jsonl",
        str(out_dir / "videoespresso_pnp.jsonl"),
    ]
    cmd += _model_endpoint_args(assets, prefer_7b=not smoke, port=18100)
    return cmd


def _cmd_videoespresso_oneshot(assets: dict, smoke: bool, out_dir: Path) -> list[str]:
    ve = assets["datasets"]["videoespresso"]
    cmd = [
        _python_bin(),
        str(REPO_ROOT / "examples/revise/oneshot_local_mc_vllm.py"),
        "--dataset",
        "jsonmc",
        "--dataset-name",
        "videoespresso",
        "--json",
        ve["test_json"],
        "--video-root",
        ve["test_video_root"],
        "--max-frames",
        "8",
        "--max-samples",
        "1" if smoke else "0",
        "--summary-json",
        str(out_dir / "videoespresso_oneshot.summary.json"),
        "--log-jsonl",
        str(out_dir / "videoespresso_oneshot.jsonl"),
    ]
    cmd += _model_endpoint_args(assets, prefer_7b=False, port=18101)
    return cmd


def _cmd_egoschema_pnp(assets: dict, smoke: bool, out_dir: Path) -> list[str]:
    eg = assets["datasets"]["egoschema"]
    cmd = [
        _python_bin(),
        str(REPO_ROOT / "examples/revise/plug_and_play_egoschema_vllm.py"),
        "--dataset-name",
        "egoschema",
        "--json",
        eg["json"],
        "--video-root",
        eg["video_root"],
        "--max-rounds",
        "2" if smoke else "4",
        "--max-frames-per-round",
        "3",
        "--max-samples",
        "1" if smoke else "0",
        "--summary-json",
        str(out_dir / "egoschema_pnp.summary.json"),
        "--log-jsonl",
        str(out_dir / "egoschema_pnp.jsonl"),
    ]
    cmd += _model_endpoint_args(assets, prefer_7b=not smoke, port=18110)
    return cmd


def _cmd_videomme_pnp(assets: dict, smoke: bool, out_dir: Path) -> list[str]:
    cmd = [
        _python_bin(),
        str(REPO_ROOT / "examples/revise/plug_and_play_videomme_lvbench_vllm.py"),
        "--dataset",
        "videomme",
        "--cached-only",
        "--max-samples",
        "1" if smoke else "0",
        "--max-rounds",
        "2" if smoke else "4",
        "--max-frames-per-round",
        "3",
        "--video-cache-dir",
        assets["datasets"]["video_cache"]["root"] or "/tmp/chenwei_video_cache",
        "--summary-json",
        str(out_dir / "videomme_pnp.summary.json"),
        "--log-jsonl",
        str(out_dir / "videomme_pnp.jsonl"),
    ]
    cmd += _model_endpoint_args(assets, prefer_7b=False, port=18120)
    return cmd


def _cmd_lvbench_pnp(assets: dict, smoke: bool, out_dir: Path) -> list[str]:
    cmd = [
        _python_bin(),
        str(REPO_ROOT / "examples/revise/plug_and_play_videomme_lvbench_vllm.py"),
        "--dataset",
        "lvbench",
        "--cached-only",
        "--max-samples",
        "1" if smoke else "0",
        "--max-rounds",
        "2" if smoke else "4",
        "--max-frames-per-round",
        "3",
        "--video-cache-dir",
        assets["datasets"]["video_cache"]["root"] or "/tmp/chenwei_video_cache",
        "--summary-json",
        str(out_dir / "lvbench_pnp.summary.json"),
        "--log-jsonl",
        str(out_dir / "lvbench_pnp.jsonl"),
    ]
    cmd += _model_endpoint_args(assets, prefer_7b=False, port=18121)
    return cmd


def _cmd_videomme_oneshot(assets: dict, smoke: bool, out_dir: Path) -> list[str]:
    cmd = [
        _python_bin(),
        str(REPO_ROOT / "examples/revise/oneshot_videomme_lvbench_vllm.py"),
        "--dataset",
        "videomme",
        "--cached-only",
        "--max-samples",
        "1" if smoke else "0",
        "--video-cache-dir",
        assets["datasets"]["video_cache"]["root"] or "/tmp/chenwei_video_cache",
        "--summary-json",
        str(out_dir / "videomme_oneshot.summary.json"),
        "--log-jsonl",
        str(out_dir / "videomme_oneshot.jsonl"),
    ]
    cmd += _model_endpoint_args(assets, prefer_7b=False, port=18122)
    return cmd


def _cmd_lvbench_oneshot(assets: dict, smoke: bool, out_dir: Path) -> list[str]:
    cmd = [
        _python_bin(),
        str(REPO_ROOT / "examples/revise/oneshot_videomme_lvbench_vllm.py"),
        "--dataset",
        "lvbench",
        "--cached-only",
        "--max-samples",
        "1" if smoke else "0",
        "--video-cache-dir",
        assets["datasets"]["video_cache"]["root"] or "/tmp/chenwei_video_cache",
        "--summary-json",
        str(out_dir / "lvbench_oneshot.summary.json"),
        "--log-jsonl",
        str(out_dir / "lvbench_oneshot.jsonl"),
    ]
    cmd += _model_endpoint_args(assets, prefer_7b=False, port=18123)
    return cmd


def _cmd_nextqa_hydra_resolve(assets: dict, smoke: bool, out_dir: Path) -> list[str]:
    nextqa = assets["datasets"]["nextqa"]
    model_path = assets["models"]["qwen25_vl_3b"]
    return [
        _python_bin(),
        "-m",
        "verl.trainer.main_ppo",
        "--config-path",
        str(REPO_ROOT / "examples/revise/config"),
        "--config-name",
        "revise_nextqa_smoke",
        f"data.nextqa.video_root={nextqa['video_root']}",
        f"data.nextqa.map_json={nextqa['map_json']}",
        f"data.train_files={nextqa['train_csv']}",
        f"data.val_files={nextqa['val_csv']}",
        f"actor_rollout_ref.model.path={model_path}",
        "--cfg",
        "job",
        "--resolve",
    ]


def _manual_nextqa_pipeline(assets: dict, smoke: bool, out_dir: Path) -> str:
    nextqa = assets["datasets"]["nextqa"]
    return (
        f"cd {shlex.quote(str(REPO_ROOT))} && "
        "VIDEO_ROOT="
        + shlex.quote(str(nextqa["video_root"]))
        + " "
        "MAP_JSON="
        + shlex.quote(str(nextqa["map_json"]))
        + " "
        "CSV="
        + shlex.quote(str(nextqa["train_csv"]))
        + " "
        "./examples/revise/run_generate_teacher_data.sh && "
        "./examples/revise/run_revise_nextqa_sft_then_rl.sh "
        "data.nextqa.video_root="
        + shlex.quote(str(nextqa["video_root"]))
        + " "
        "data.nextqa.map_json="
        + shlex.quote(str(nextqa["map_json"]))
        + " "
        "data.train_files="
        + shlex.quote(str(nextqa["train_csv"]))
        + " "
        "data.val_files="
        + shlex.quote(str(nextqa["val_csv"]))
    )


def _manual_videoespresso_pipeline(assets: dict, smoke: bool, out_dir: Path) -> str:
    return (
        "Requires a private or pre-converted VideoEspresso multiple-choice train JSON. "
        "Set REVISE_VIDEOESPRESSO_MC_TRAIN_JSON and then run a dedicated SFT/RL config."
    )


EXPERIMENTS: dict[str, dict[str, object]] = {
    "nextqa_pnp": {
        "title": "NExT-QA plug-and-play",
        "paper_ref": "paper/tables/nextqa.tex, paper/tables/RL Results.tex",
        "check": _require_nextqa,
        "build": _cmd_nextqa_pnp,
        "run_supported": True,
    },
    "nextqa_oneshot": {
        "title": "NExT-QA direct reasoning (one-shot)",
        "paper_ref": "paper/tables/RL Results.tex",
        "check": _require_nextqa,
        "build": _cmd_nextqa_oneshot,
        "run_supported": True,
    },
    "nextqa_caption": {
        "title": "NExT-QA caption-only baseline",
        "paper_ref": "paper/tables/abaltion_caption.tex",
        "check": _require_nextqa_captions,
        "build": _cmd_nextqa_caption,
        "run_supported": True,
    },
    "nextqa_videoagent": {
        "title": "NExT-QA VideoAgent-style caption baseline",
        "paper_ref": "paper/tables/abaltion_caption.tex",
        "check": _require_nextqa_captions,
        "build": lambda assets, smoke, out_dir: _cmd_nextqa_videoagent(assets, smoke, out_dir, official=False),
        "run_supported": True,
    },
    "nextqa_videoagent_official": {
        "title": "NExT-QA VideoAgent official-style baseline",
        "paper_ref": "paper/tables/videoagent.tex",
        "check": _require_nextqa_captions,
        "build": lambda assets, smoke, out_dir: _cmd_nextqa_videoagent(assets, smoke, out_dir, official=True),
        "run_supported": True,
    },
    "videoespresso_pnp": {
        "title": "VideoEspresso plug-and-play",
        "paper_ref": "paper/tables/VideoEspresso.tex, paper/tables/RL Results.tex",
        "check": _require_videoespresso,
        "build": _cmd_videoespresso_pnp,
        "run_supported": True,
    },
    "videoespresso_oneshot": {
        "title": "VideoEspresso direct reasoning (one-shot)",
        "paper_ref": "paper/tables/RL Results.tex",
        "check": _require_videoespresso,
        "build": _cmd_videoespresso_oneshot,
        "run_supported": True,
    },
    "egoschema_pnp": {
        "title": "EgoSchema plug-and-play",
        "paper_ref": "paper/tables/egoschma.tex",
        "check": _require_egoschema,
        "build": _cmd_egoschema_pnp,
        "run_supported": True,
    },
    "videomme_pnp": {
        "title": "Video-MME plug-and-play",
        "paper_ref": "paper/tables/more_benchmarks.tex",
        "check": _require_model_or_api,
        "build": _cmd_videomme_pnp,
        "run_supported": True,
    },
    "lvbench_pnp": {
        "title": "LVBench plug-and-play",
        "paper_ref": "paper/tables/more_benchmarks.tex",
        "check": _require_model_or_api,
        "build": _cmd_lvbench_pnp,
        "run_supported": True,
    },
    "videomme_oneshot": {
        "title": "Video-MME one-shot baseline",
        "paper_ref": "paper/tables/more_benchmarks.tex",
        "check": _require_model_or_api,
        "build": _cmd_videomme_oneshot,
        "run_supported": True,
    },
    "lvbench_oneshot": {
        "title": "LVBench one-shot baseline",
        "paper_ref": "paper/tables/more_benchmarks.tex",
        "check": _require_model_or_api,
        "build": _cmd_lvbench_oneshot,
        "run_supported": True,
    },
    "nextqa_hydra_resolve": {
        "title": "NExT-QA Hydra config resolve",
        "paper_ref": "paper/sec/4_exp.tex",
        "check": _require_nextqa_training,
        "build": _cmd_nextqa_hydra_resolve,
        "run_supported": True,
    },
    "nextqa_train_pipeline": {
        "title": "NExT-QA SFT+GRPO training pipeline",
        "paper_ref": "paper/tables/RL Results.tex",
        "check": _require_nextqa_training,
        "build": _manual_nextqa_pipeline,
        "run_supported": False,
    },
    "videoespresso_train_pipeline": {
        "title": "VideoEspresso SFT+GRPO training pipeline",
        "paper_ref": "paper/tables/RL Results.tex",
        "check": _require_videoespresso_training,
        "build": _manual_videoespresso_pipeline,
        "run_supported": False,
    },
}


def _selected_ids(args: argparse.Namespace) -> list[str]:
    if args.all:
        return list(EXPERIMENTS.keys())
    if not args.experiment:
        raise SystemExit("Specify --experiment or --all.")
    unknown = [eid for eid in args.experiment if eid not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"Unknown experiment ids: {', '.join(unknown)}")
    return list(args.experiment)


def cmd_to_text(cmd: list[str] | str) -> str:
    if isinstance(cmd, str):
        return cmd
    return shlex.join(cmd)


def main() -> int:
    ap = argparse.ArgumentParser(description="List/check/run paper experiments.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list")
    check_ap = sub.add_parser("check")
    check_ap.add_argument("--experiment", action="append")
    check_ap.add_argument("--all", action="store_true")

    run_ap = sub.add_parser("run")
    run_ap.add_argument("--experiment", action="append")
    run_ap.add_argument("--all", action="store_true")
    run_ap.add_argument("--smoke", action="store_true", help="Run the smallest supported variant.")
    run_ap.add_argument("--dry-run", action="store_true")
    run_ap.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "paper_suite"))

    args = ap.parse_args()
    assets = discover_assets()

    if args.cmd == "list":
        for exp_id, meta in EXPERIMENTS.items():
            print(f"{exp_id}\t{meta['title']}\t{meta['paper_ref']}")
        return 0

    if args.cmd == "check":
        for exp_id in _selected_ids(args):
            meta = EXPERIMENTS[exp_id]
            missing = meta["check"](assets)  # type: ignore[index]
            status = "ok" if not missing else "blocked"
            print(f"[{status}] {exp_id}: {meta['title']}")
            if missing:
                for item in missing:
                    print(f"  - {item}")
            print(f"  command: {cmd_to_text(meta['build'](assets, True, Path('/tmp')))}")
        return 0

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for exp_id in _selected_ids(args):
        meta = EXPERIMENTS[exp_id]
        missing = meta["check"](assets)  # type: ignore[index]
        if missing:
            print(f"[blocked] {exp_id}: {'; '.join(missing)}")
            return 2
        cmd = meta["build"](assets, bool(args.smoke), out_dir)  # type: ignore[index]
        print(f"[run] {exp_id}")
        print(cmd_to_text(cmd))
        if args.dry_run:
            continue
        if not meta["run_supported"]:
            print(f"[manual] {exp_id} must be run manually.")
            continue
        assert isinstance(cmd, list)
        proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
        if proc.returncode != 0:
            return proc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
