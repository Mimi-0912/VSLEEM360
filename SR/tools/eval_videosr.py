#!/usr/bin/env python3
"""Video-SR zero-shot inference + evaluation on continuous clips (Option A).

This is the SR MAIN-TABLE path. Unlike eval_zeroshot.py (single-frame T=1
inference of the same video models -> appendix), here each model receives the
full T=15 CONTINUOUS Preview window, so its temporal propagation is actually
exercised. The model's sr_fn returns the TEMPORAL-CENTRE frame (offset 7), which
build_videosr_clips.py guarantees is preview_center_index -- the frame the frozen
manifest aligned to the single HD GT. So we super-resolve the centre and score it
against that one HD frame.

Everything else is byte-identical to the frozen zero-shot convention: official x4
weights, ERP-tiled inference (tiled_sr), x4 output resized (bicubic, ERP wrap) to
the 5760x2880 x3 target, and the SAME five metrics with the SAME constants as
eval_zeroshot.py (PSNR/SSIM Y crop_border=3 via mmagic; LPIPS alex full RGB;
WS-PSNR/WS-SSIM cos-latitude Y; predictions rounded to uint8).

Clips come from build_videosr_clips.py: <root>/<scene>/lr/{0000..0014}.png +
<root>/<scene>/gt/0007.png, indexed by <root>/clips_manifest.jsonl.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zeroshot_infer import NATIVE_SCALE, build_sr_fn, resize_x4_to_target  # noqa: E402
from tiled_infer import tiled_sr  # noqa: E402
from metrics import to_y01, ws_psnr_y, ws_ssim_y  # noqa: E402
from mmagic.evaluation.metrics import psnr as mm_psnr, ssim as mm_ssim  # noqa: E402
from eval_zeroshot import (  # noqa: E402 -- single source of frozen conventions
    CROP_BORDER, CONVERT_TO, CHANNEL_ORDER, CONVENTIONS, LPIPSMetric)


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return np.ascontiguousarray(bgr[..., ::-1])


def load_clip(seq_dir: Path, window: int, center_off: int):
    """Return (lr_seq uint8 (T,H,W,3) RGB, gt uint8 (Hg,Wg,3) RGB)."""
    lr = np.stack([load_rgb(seq_dir / "lr" / f"{i:04d}.png") for i in range(window)])
    gt = load_rgb(seq_dir / "gt" / f"{center_off:04d}.png")
    return lr, gt


def sr_predict_video(sr_fn, lr_seq_u8, tile_h, tile_w, overlap, target_w, target_h):
    """T=15 continuous LR window -> centre-frame prediction uint8 RGB at target.

    tiled_sr feeds each spatial tile's full T window to sr_fn, which returns the
    temporal centre; the x4 centre frame is then resized to the x3 target."""
    hr_x4 = tiled_sr(lr_seq_u8, sr_fn, NATIVE_SCALE, tile_h, tile_w, overlap)
    hr = resize_x4_to_target(hr_x4, target_w, target_h)
    return np.round(hr * 255.0).clip(0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=["basicvsrpp", "realbasicvsr"])
    ap.add_argument("--clips-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--weights-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "weights")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--target-w", type=int, default=5760)
    ap.add_argument("--target-h", type=int, default=2880)
    ap.add_argument("--tile-h", type=int, default=960)
    ap.add_argument("--tile-w", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only-scenes", default="",
                    help="comma-separated scene names; evaluate ONLY these clips. "
                         "Use for figure-only re-runs; does NOT touch the frozen "
                         "main-table outputs (point --out-dir elsewhere).")
    ap.add_argument("--save-fig-scenes", default="",
                    help="comma-separated scene names whose centre prediction PNG to save")
    args = ap.parse_args()

    clips = [json.loads(l) for l in (args.clips_root / "clips_manifest.jsonl")
             .read_text(encoding="utf-8").splitlines() if l.strip()]
    only = set(x for x in args.only_scenes.split(",") if x)
    if only:
        clips = [c for c in clips if c["scene"] in only]
        missing = only - {c["scene"] for c in clips}
        if missing:
            raise SystemExit(f"--only-scenes not in manifest: {sorted(missing)}")
    if args.limit:
        clips = clips[:args.limit]
    save_scenes = set(x for x in args.save_fig_scenes.split(",") if x)

    sr_fn, meta = build_sr_fn(args.model, args.weights_dir, args.device)
    lpips_metric = LPIPSMetric(args.device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out_dir / "figures"
    if save_scenes:
        fig_dir.mkdir(exist_ok=True)

    fields = ["scene", "id", "psnr", "ssim", "lpips", "ws_psnr", "ws_ssim"]
    records = []
    started = time.perf_counter()
    per_frame_path = args.out_dir / "per_frame.csv"
    with per_frame_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for clip in clips:
            seq_dir = args.clips_root / clip["seq"]
            lr_seq, gt = load_clip(seq_dir, clip["window"], clip["center_offset"])
            if gt.shape != (args.target_h, args.target_w, 3):
                raise SystemExit(f"GT shape {gt.shape} != target for {clip['scene']}")
            pred = sr_predict_video(sr_fn, lr_seq, args.tile_h, args.tile_w,
                                    args.overlap, args.target_w, args.target_h)
            psnr_v = float(mm_psnr(pred, gt, crop_border=CROP_BORDER,
                                   convert_to=CONVERT_TO, channel_order=CHANNEL_ORDER))
            ssim_v = float(mm_ssim(pred, gt, crop_border=CROP_BORDER,
                                   convert_to=CONVERT_TO, channel_order=CHANNEL_ORDER))
            lpips_v = lpips_metric(pred, gt)
            pred_y, gt_y = to_y01(pred), to_y01(gt)
            wsp, wss = ws_psnr_y(pred_y, gt_y), ws_ssim_y(pred_y, gt_y)
            rec = {"scene": clip["scene"], "id": clip["id"], "psnr": psnr_v,
                   "ssim": ssim_v, "lpips": lpips_v, "ws_psnr": wsp, "ws_ssim": wss}
            records.append(rec)
            writer.writerow(rec)
            fh.flush()
            if clip["scene"] in save_scenes:
                cv2.imwrite(str(fig_dir / f"{clip['scene']}.png"), pred[..., ::-1])
            print(f"{clip['scene']} psnr={psnr_v:.2f} ssim={ssim_v:.4f} "
                  f"lpips={lpips_v:.4f} wsp={wsp:.2f} wss={wss:.4f}", flush=True)

    numeric = ["psnr", "ssim", "lpips", "ws_psnr", "ws_ssim"]
    summary = {
        "schema": 1,
        "task": "video-SR (T=15 continuous window, centre-frame eval)",
        "model": args.model,
        "clip_count": len(records),
        "aggregation": "arithmetic mean over clips (one centre frame per scene)",
        "metrics_mean": {k: float(np.mean([r[k] for r in records])) for k in numeric},
        "metrics_std": {k: float(np.std([r[k] for r in records])) for k in numeric},
        "conventions": CONVENTIONS,
        "window": clips[0]["window"] if clips else None,
        "tile": {"tile_h": args.tile_h, "tile_w": args.tile_w, "overlap": args.overlap},
        "target": [args.target_w, args.target_h],
        "native_scale": NATIVE_SCALE,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        **meta,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary["metrics_mean"], indent=2))
    print(f"[done] {len(records)} clips -> {args.out_dir}")


if __name__ == "__main__":
    main()
