#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files


def _iter_lvbench_zip_files() -> list[str]:
    files = list_repo_files("lmms-lab/LVBench", repo_type="dataset")
    zips = [f for f in files if f.startswith("video_chunks/") and f.endswith(".zip")]
    return sorted(zips)


def _extract_zip_mp4s(zip_path: Path, out_dir: Path) -> int:
    extracted = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if not name.lower().endswith(".mp4"):
                continue
            out_path = out_dir / Path(name).name
            if out_path.exists() and out_path.stat().st_size > 0:
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
            extracted += 1
    return extracted


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-cache-dir", default="/tmp/hf_cache_lvbench", help="Cache dir for HF downloads")
    ap.add_argument("--out-dir", default="/tmp/chenwei_video_cache/lvbench", help="Directory to place extracted .mp4 files")
    ap.add_argument("--max-files", type=int, default=0, help="Optional: limit number of zip chunks to download")
    args = ap.parse_args()

    hf_cache_dir = Path(args.hf_cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_files = _iter_lvbench_zip_files()
    if args.max_files and args.max_files > 0:
        zip_files = zip_files[: args.max_files]

    if not zip_files:
        raise SystemExit("No LVBench video_chunks/*.zip files found in repo.")

    total_extracted = 0
    for i, rel in enumerate(zip_files, start=1):
        print(f"[lvbench] download {i}/{len(zip_files)}: {rel}", flush=True)
        local = hf_hub_download(
            repo_id="lmms-lab/LVBench",
            repo_type="dataset",
            filename=rel,
            cache_dir=str(hf_cache_dir),
            resume_download=True,
        )
        zip_path = Path(local)
        print(f"[lvbench] extract mp4s: {zip_path.name}", flush=True)
        n = _extract_zip_mp4s(zip_path, out_dir)
        total_extracted += n
        print(f"[lvbench] extracted {n} mp4s (cumulative {total_extracted})", flush=True)

    # Best-effort cleanup of negative-cache markers from prior runs.
    failed_markers = list(out_dir.glob("*.mp4.failed"))
    if failed_markers:
        print(f"[lvbench] removing {len(failed_markers)} *.failed markers", flush=True)
        for p in failed_markers:
            try:
                p.unlink()
            except Exception:
                pass

    print(f"[lvbench] done. out_dir={out_dir} extracted={total_extracted}", flush=True)


if __name__ == "__main__":
    main()

