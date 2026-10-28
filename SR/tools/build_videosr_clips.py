#!/usr/bin/env python3
"""Extract continuous video-SR clips (Option A) from the frozen test split.

The VPR frame set (query_preview/query_hd) is UNUSABLE for video-SR: its frames
are ~10 s apart, so feeding them to a temporal model fuses unrelated scenes. The
HD stream, meanwhile, is sparse keyframes (~0.48 fps), so no continuous HD
sequence exists to pair frame-for-frame. Option A (chosen) is the only red-line
compliant shape:

  * LR input  : T=15 CONTINUOUS native Preview frames (stride=1), centred on an
                accepted manifest row -> real temporal propagation for the model.
  * GT target : the SINGLE HD frame at that row's hd_index, which the frozen
                manifest already aligned (pts_delta <= 20 ms) to the Preview
                centre. The temporal centre of the T=15 window (offset 7) is
                exactly preview_center_index, so it stays HD-aligned for free.

One clip per scene (user decision: 每场景1段×15帧, 取视频中段). For each scene we
take the MIDDLE accepted row (by hd_index), extend its native 7-frame window to
+-7, verify the extended window stays in-range, and decode index-exact.

Frames are decoded with ffmpeg ``select='between(n,START,END)'`` from stream
start -- slower than seeking but provably index-exact (no pts/keyframe rounding),
which matters because a shifted window silently corrupts the temporal signal.

Layout (REDS-like, 序列名/帧号.png):
  <root>/<scene>/lr/0000.png .. 0014.png   (Preview LR, 1920x960)
  <root>/<scene>/gt/0007.png               (centre HD GT, 5760x2880)
A clips_manifest.jsonl records full provenance per clip.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

WINDOW = 15          # T: temporal window length
HALF = WINDOW // 2   # 7 -> center +-7
PREVIEW_W, PREVIEW_H = 1920, 960
HD_W, HD_H = 5760, 2880


def decode_range(video: str, start: int, end: int, w: int, h: int) -> np.ndarray:
    """Decode frames [start, end] (inclusive, by decode index n) as (N,h,w,3) BGR.

    Uses select-from-stream-start so N is counted by decode order -> the frame
    at index k is exactly the k-th decoded frame, independent of pts/keyframes.
    """
    count = end - start + 1
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-i", video,
        "-vf", f"select='between(n\\,{start}\\,{end})'",
        "-fps_mode", "passthrough", "-frames:v", str(count),
        "-an", "-sn", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1",
    ]
    proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    expected = count * w * h * 3
    if len(proc.stdout) != expected:
        raise RuntimeError(
            f"ffmpeg returned {len(proc.stdout)} bytes, expected {expected}: "
            f"{video} frames [{start},{end}] {w}x{h}")
    return np.frombuffer(proc.stdout, np.uint8).reshape(count, h, w, 3)


def probe_nb_frames(video: str) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames", "-of", "csv=p=0", video],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        encoding="utf-8").stdout.strip()
    return int(out.rstrip(","))


def pick_middle_row(rows: list[dict], preview_nb: int) -> dict | None:
    """Middle accepted row whose +-7 window fits in [0, preview_nb).

    Walk outward from the median so we stay near the video's middle segment
    while guaranteeing an in-range extended window.
    """
    order = sorted(rows, key=lambda r: r["hd_index"])
    mid = len(order) // 2
    for step in range(len(order)):
        for idx in (mid + step, mid - step):
            if 0 <= idx < len(order):
                c = order[idx]["preview_center_index"]
                if c - HALF >= 0 and c + HALF < preview_nb:
                    return order[idx]
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N scenes only (debug)")
    args = ap.parse_args()

    by_scene: dict[str, list[dict]] = defaultdict(list)
    with args.manifest.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("accepted"):
                by_scene[r["scene"]].append(r)

    scenes = sorted(by_scene)
    if args.limit:
        scenes = scenes[:args.limit]

    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_root / "clips_manifest.jsonl"
    clips = []
    with manifest_path.open("w", encoding="utf-8") as mf:
        for si, scene in enumerate(scenes, 1):
            rows = by_scene[scene]
            preview_video = rows[0]["preview_video"]
            hd_video = rows[0]["hd_video"]
            preview_nb = probe_nb_frames(preview_video)
            row = pick_middle_row(rows, preview_nb)
            if row is None:
                print(f"[skip] {scene}: no in-range +-{HALF} window", flush=True)
                continue
            center = row["preview_center_index"]
            start, end = center - HALF, center + HALF
            lr = decode_range(preview_video, start, end, PREVIEW_W, PREVIEW_H)
            if lr.shape != (WINDOW, PREVIEW_H, PREVIEW_W, 3):
                raise RuntimeError(f"{scene}: LR shape {lr.shape}")
            hd = decode_range(hd_video, row["hd_index"], row["hd_index"], HD_W, HD_H)[0]
            if hd.shape != (HD_H, HD_W, 3):
                raise RuntimeError(f"{scene}: GT shape {hd.shape}")

            seq_dir = args.out_root / scene
            (seq_dir / "lr").mkdir(parents=True, exist_ok=True)
            (seq_dir / "gt").mkdir(parents=True, exist_ok=True)
            for i in range(WINDOW):
                cv2.imwrite(str(seq_dir / "lr" / f"{i:04d}.png"), lr[i])
            cv2.imwrite(str(seq_dir / "gt" / f"{HALF:04d}.png"), hd)

            rec = {
                "scene": scene, "seq": scene,
                "center_index": center, "window": WINDOW,
                "lr_indices": list(range(start, end + 1)),
                "center_offset": HALF,
                "hd_index": row["hd_index"], "hd_pts": row["hd_pts"],
                "preview_center_pts": row["preview_center_pts"],
                "pts_delta_ms": row["pts_delta_ms"],
                "id": row["id"], "preview_video": preview_video, "hd_video": hd_video,
            }
            clips.append(rec)
            mf.write(json.dumps(rec, sort_keys=True) + "\n")
            mf.flush()
            print(f"[{si}/{len(scenes)}] {scene} center={center} "
                  f"win=[{start},{end}] hd={row['hd_index']}", flush=True)

    print(f"[done] {len(clips)} clips -> {args.out_root}")


if __name__ == "__main__":
    main()
