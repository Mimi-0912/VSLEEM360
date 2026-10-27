#!/usr/bin/env python3
"""Audit temporal and geometric alignment without changing source pixels."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("accepted"):
                yield row


def decode_at(video: str, pts: float, width: int, height: int) -> np.ndarray:
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-ss", f"{pts:.9f}", "-i", video,
        "-frames:v", "1", "-vf", f"scale={width}:{height}:flags=area",
        "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
    ]
    proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    expected = width * height * 3
    if len(proc.stdout) != expected:
        raise RuntimeError(f"decoded {len(proc.stdout)} bytes, expected {expected}: {video}@{pts}")
    return np.frombuffer(proc.stdout, np.uint8).reshape(height, width, 3)


def feature(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def compare(reference: np.ndarray, candidate: np.ndarray) -> dict:
    ref, cand = feature(reference), feature(candidate)
    window = cv2.createHanningWindow((ref.shape[1], ref.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(ref, cand, window)
    # Phase correlation wraps both axes. Horizontal wrap is physically correct
    # for ERP; large vertical wrap is rejected by the audit threshold.
    aligned = np.roll(candidate, (-round(dy), -round(dx)), axis=(0, 1))
    mae = float(np.mean(np.abs(reference.astype(np.float32) - aligned.astype(np.float32))) / 255.0)
    return {"dx_audit_px": float(dx), "dy_audit_px": float(dy), "response": float(response), "aligned_rgb_mae": mae}


def uniform(rows: list[dict], count: int) -> list[dict]:
    if len(rows) <= count:
        return rows
    return [rows[round(i * (len(rows) - 1) / (count - 1))] for i in range(count)] if count > 1 else [rows[len(rows)//2]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes-per-split", type=int, default=5)
    parser.add_argument("--samples-per-scene", type=int, default=3)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--temporal-radius", type=int, default=2)
    args = parser.parse_args()

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for path in args.pairs:
        for row in read_jsonl(path):
            grouped[(row["split"], row["scene"])].append(row)

    selected_scenes = []
    for split in sorted({key[0] for key in grouped}):
        scenes = sorted(scene for candidate_split, scene in grouped if candidate_split == split)
        selected_scenes.extend((split, scene) for scene in uniform(scenes, args.scenes_per_split))

    records = []
    for split, scene in selected_scenes:
        for row in uniform(grouped[(split, scene)], args.samples_per_scene):
            hd = decode_at(row["hd_video"], row["hd_pts"], args.width, args.height)
            candidates = []
            centre = len(row["preview_pts"]) // 2
            for offset in range(-args.temporal_radius, args.temporal_radius + 1):
                pos = centre + offset
                if not 0 <= pos < len(row["preview_pts"]):
                    continue
                preview = decode_at(row["preview_video"], row["preview_pts"][pos], args.width, args.height)
                candidates.append({"offset": offset, "preview_pts": row["preview_pts"][pos], **compare(hd, preview)})
            best = max(candidates, key=lambda x: x["response"])
            records.append({
                "id": row["id"], "split": split, "scene": scene, "hd_pts": row["hd_pts"],
                "pts_nearest_delta_ms": row["pts_delta_ms"], "best": best, "candidates": candidates,
            })
            print(f"{row['id']} best_dt={best['offset']:+d} dx={best['dx_audit_px']:.2f} dy={best['dy_audit_px']:.2f} response={best['response']:.3f}", flush=True)

    dx = np.array([r["best"]["dx_audit_px"] for r in records])
    dy = np.array([r["best"]["dy_audit_px"] for r in records])
    offsets = np.array([r["best"]["offset"] for r in records])
    summary = {
        "schema": 1, "audit_resolution": [args.width, args.height], "record_count": len(records),
        "selected_scenes": [f"{s}/{c}" for s, c in selected_scenes],
        "best_temporal_offset_counts": {str(int(x)): int(np.sum(offsets == x)) for x in sorted(set(offsets))},
        "median_dx_audit_px": float(np.median(dx)), "median_dy_audit_px": float(np.median(dy)),
        "median_abs_dx_audit_px": float(np.median(np.abs(dx))),
        "median_abs_dy_audit_px": float(np.median(np.abs(dy))),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
