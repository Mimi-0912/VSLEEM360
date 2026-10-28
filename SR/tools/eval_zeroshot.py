#!/usr/bin/env python3
"""Fused zero-shot inference + evaluation for VSLEEM360 image-SR baselines.

Single-pass by design: predictions for the full 5780-frame test split would be
~184 GB of PNGs (only 113 GB free), and zero-shot needs each prediction exactly
once. So each frame is: super-resolved in memory (official x4 model, ERP tiled)
-> resized to the 5760x2880 x3 target (ERP wrap) -> five metrics accumulated ->
prediction discarded. A few PNGs are optionally saved for paper figures.

Reuses the SINGLE source of truth in zeroshot_infer.py (build_sr_fn, tiled_sr,
resize_x4_to_target) so evaluation predictions are byte-identical to what a
standalone inference run would produce.

Metric conventions (SR task brief section 4 -- all fixed here as constants):
  PSNR / SSIM : official mmagic functions, range [0,255], Y channel (BT.601
                luma via mmcv), crop_border = CROP_BORDER pixels each edge.
  LPIPS       : official lpips AlexNet, spatial=False, inputs in [-1,1]; full
                RGB frame, no crop (perceptual metric, edge effect negligible).
  WS-PSNR /   : cos(latitude)-weighted, on the SAME Y channel, FULL frame (no
  WS-SSIM       crop -- latitude weighting + circular longitude handle borders).
Data range 255, values rounded to uint8 before all metrics (what a saved PNG
would contain, so numbers match released predictions).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zeroshot_infer import (  # noqa: E402
    NATIVE_SCALE, build_sr_fn, list_pairs, load_rgb, resize_x4_to_target)
from metrics import to_y01, ws_psnr_y, ws_ssim_y  # noqa: E402
from mmagic.evaluation.metrics import psnr as mm_psnr, ssim as mm_ssim  # noqa: E402

# --- Fixed evaluation conventions (documented in summary.json) ---
CROP_BORDER = 3          # SR convention: crop `scale` px; x3 target -> 3
CONVERT_TO = "Y"         # luma channel
CHANNEL_ORDER = "rgb"    # our arrays are RGB
DATA_RANGE = 255
LPIPS_NET = "alex"
CONVENTIONS = {
    "psnr_ssim_impl": "mmagic.evaluation.metrics.psnr/ssim (official)",
    "channel": "Y (BT.601 luma, mmcv rgb2ycbcr y_only)",
    "data_range": DATA_RANGE,
    "crop_border_px": CROP_BORDER,
    "rounded_to_uint8": True,
    "lpips": f"official lpips {LPIPS_NET}, spatial=False, inputs [-1,1], full RGB no-crop",
    "ws_metrics": "cos-latitude weighted on Y, full frame (no crop), ERP circular longitude",
    "scale_handling": "official x4 weights; x4 output resized (bicubic, ERP wrap) to 5760x2880 x3 target",
}


class LPIPSMetric:
    def __init__(self, device: str):
        import lpips
        self.device = torch.device(device)
        self.model = lpips.LPIPS(net=LPIPS_NET, spatial=False).eval().to(self.device)

    @torch.inference_mode()
    def __call__(self, pred_rgb_u8: np.ndarray, gt_rgb_u8: np.ndarray) -> float:
        def to_t(x):
            t = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).unsqueeze(0)
            return t.float().to(self.device).div_(127.5).sub_(1.0)
        return float(self.model(to_t(pred_rgb_u8), to_t(gt_rgb_u8)).item())


def sr_predict(sr_fn, lr_rgb_u8, tile_h, tile_w, overlap, target_w, target_h):
    """Full zero-shot prediction: LR uint8 RGB (960,1920,3) -> pred uint8 RGB
    (2880,5760,3). Identical to zeroshot_infer's per-frame path."""
    from tiled_infer import tiled_sr
    lr_seq = lr_rgb_u8[None]  # T=1
    hr_x4 = tiled_sr(lr_seq, sr_fn, NATIVE_SCALE, tile_h, tile_w, overlap)
    hr = resize_x4_to_target(hr_x4, target_w, target_h)
    return np.round(hr * 255.0).clip(0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["basicvsrpp", "realbasicvsr"])
    parser.add_argument("--preview-dir", type=Path, required=True)
    parser.add_argument("--hd-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "weights")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-w", type=int, default=5760)
    parser.add_argument("--target-h", type=int, default=2880)
    parser.add_argument("--tile-h", type=int, default=960)
    parser.add_argument("--tile-w", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save-fig-ids", default="",
                        help="comma-separated frame ids whose prediction PNG to save for figures")
    args = parser.parse_args()

    sr_fn, meta = build_sr_fn(args.model, args.weights_dir, args.device)
    lpips_metric = LPIPSMetric(args.device)
    pairs = list_pairs(args.preview_dir, args.hd_dir)
    if args.limit:
        pairs = pairs[:args.limit]
    save_ids = set(x for x in args.save_fig_ids.split(",") if x)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out_dir / "figures"
    if save_ids:
        fig_dir.mkdir(exist_ok=True)

    fields = ["id", "scene", "psnr", "ssim", "lpips", "ws_psnr", "ws_ssim"]
    records = []
    started = time.perf_counter()
    per_frame_path = args.out_dir / "per_frame.csv"
    with per_frame_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in pairs:
            lr = load_rgb(row["lr_path"])              # (960,1920,3) RGB u8
            gt = load_rgb(row["gt_path"])              # (2880,5760,3) RGB u8
            if gt.shape != (args.target_h, args.target_w, 3):
                raise SystemExit(f"GT shape {gt.shape} != target for {row['id']}")
            pred = sr_predict(sr_fn, lr, args.tile_h, args.tile_w, args.overlap,
                              args.target_w, args.target_h)
            # plain PSNR/SSIM: official mmagic, Y, crop_border
            psnr_v = float(mm_psnr(pred, gt, crop_border=CROP_BORDER,
                                   convert_to=CONVERT_TO, channel_order=CHANNEL_ORDER))
            ssim_v = float(mm_ssim(pred, gt, crop_border=CROP_BORDER,
                                   convert_to=CONVERT_TO, channel_order=CHANNEL_ORDER))
            lpips_v = lpips_metric(pred, gt)
            pred_y, gt_y = to_y01(pred), to_y01(gt)
            wsp = ws_psnr_y(pred_y, gt_y)
            wss = ws_ssim_y(pred_y, gt_y)
            rec = {"id": row["id"], "scene": row["scene"], "psnr": psnr_v,
                   "ssim": ssim_v, "lpips": lpips_v, "ws_psnr": wsp, "ws_ssim": wss}
            records.append(rec)
            writer.writerow(rec)
            fh.flush()
            if row["id"] in save_ids:
                cv2.imwrite(str(fig_dir / f"{row['id']}.png"), pred[..., ::-1])
            print(f"{row['id']} psnr={psnr_v:.2f} ssim={ssim_v:.4f} "
                  f"lpips={lpips_v:.4f} wsp={wsp:.2f} wss={wss:.4f}", flush=True)

    numeric = ["psnr", "ssim", "lpips", "ws_psnr", "ws_ssim"]
    by_scene = defaultdict(list)
    for r in records:
        by_scene[r["scene"]].append(r)
    scene_rows = [{"scene": s, "frames": len(rs),
                   **{k: float(np.mean([x[k] for x in rs])) for k in numeric}}
                  for s, rs in sorted(by_scene.items())]
    with (args.out_dir / "per_scene.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["scene", "frames", *numeric])
        w.writeheader(); w.writerows(scene_rows)

    summary = {
        "schema": 1,
        "model": args.model,
        "frame_count": len(records),
        "scene_count": len(scene_rows),
        "aggregation": "arithmetic mean over frames",
        "metrics_mean": {k: float(np.mean([r[k] for r in records])) for k in numeric},
        "metrics_std": {k: float(np.std([r[k] for r in records])) for k in numeric},
        "conventions": CONVENTIONS,
        "tile": {"tile_h": args.tile_h, "tile_w": args.tile_w, "overlap": args.overlap},
        "target": [args.target_w, args.target_h],
        "native_scale": NATIVE_SCALE,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        **meta,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary["metrics_mean"], indent=2))
    print(f"[done] {len(records)} frames -> {args.out_dir}")


if __name__ == "__main__":
    main()
