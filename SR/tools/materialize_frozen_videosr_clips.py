"""Decode the frozen VSR frame indices without reselecting or realigning clips."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from build_videosr_clips import HD_H, HD_W, PREVIEW_H, PREVIEW_W, decode_range


def source_video(root: Path, relative: str) -> Path:
    name = Path(relative)
    if name.is_absolute() or ".." in name.parts:
        raise ValueError(f"Missing video or invalid relative path: {relative}")
    path = root / name
    if not path.is_file():
        raise ValueError(f"Missing video or invalid relative path: {relative}")
    return path.resolve()


def materialize(row: dict, data_root: Path, output: Path) -> dict:
    scene, segment = row["scene"], row["segment"]
    if not all(value and Path(value).name == value and value not in (".", "..")
               for value in (scene, segment)):
        raise ValueError("Invalid scene or segment identifier")
    seq = f"{scene}__{segment}"
    if row["seq"] != seq:
        raise ValueError("Sequence directory does not match scene and segment")
    indices = row["lr_indices"]
    center = row["center_offset"]
    if (len(indices) != row["window"] or row["window"] != 15 or center != 7
            or indices != list(range(indices[0], indices[0] + 15))
            or indices[center] != row["center_index"]):
        raise ValueError("Expected the frozen continuous T=15 centre-frame window")
    preview = source_video(data_root, row["preview_video"])
    hd_video = source_video(data_root, row["hd_video"])
    lr = decode_range(str(preview), indices[0], indices[-1], PREVIEW_W, PREVIEW_H)
    gt = decode_range(str(hd_video), row["hd_index"], row["hd_index"], HD_W, HD_H)
    if lr.shape != (15, PREVIEW_H, PREVIEW_W, 3) or gt.shape != (1, HD_H, HD_W, 3):
        raise ValueError(f"Unexpected decoded frame shape for {row['clip_id']}")
    folder = output / seq
    for name in ("lr", "gt"):
        (folder / name).mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(lr):
        if not cv2.imwrite(str(folder / "lr" / f"{index:04d}.png"), frame):
            raise OSError(f"Could not write LR image for {row['clip_id']}")
    if not cv2.imwrite(str(folder / "gt" / f"{center:04d}.png"), gt[0]):
        raise OSError(f"Could not write GT image for {row['clip_id']}")
    return {**row, "preview_video": str(preview), "hd_video": str(hd_video)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).resolve().parents[1] / "manifests/clips_manifest.jsonl")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="Smoke check only; zero means all clips")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.limit:
        rows = rows[:args.limit]
    # A fresh output directory prevents mixing complete and partial manifests.
    args.out_root.mkdir(parents=True, exist_ok=False)
    with (args.out_root / "clips_manifest.jsonl").open("w") as stream:
        for row in rows:
            result = materialize(row, args.data_root, args.out_root)
            stream.write(json.dumps(result, sort_keys=True) + "\n")
            stream.flush()
            print(f"Decoded {row['clip_id']}", flush=True)
    print(f"Decoded {len(rows)} clips; no temporal or spatial correction was fitted")


if __name__ == "__main__":
    main()
