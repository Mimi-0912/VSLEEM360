#!/usr/bin/env python3
"""Zero-shot image-SR inference for VSLEEM360 with OFFICIAL BasicVSR++ /
RealBasicVSR (official mmagic implementation + official x4 weights, network
UNMODIFIED).

Pipeline (SR task brief §6, option a):
  Preview LR 1920x960 (real degradation)
    --official x4 model (tiled)-->  7680x3840
    --resize down (bicubic, ERP horizontal wrap-pad)-->  5760x2880  (x3-aligned)
No fine-tuning; no network change to fake x3 -- the x4 output is resized to the
x3 target and compared against the HD ground truth.

ERP tiling: the full latitude height (960) is processed in ONE vertical block so
no artificial pole seam is introduced; longitude (width) is tiled CIRCULARLY so
the 360-degree seam stays continuous (wrap). This reuses tiled_infer.tiled_sr.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from importlib.resources import files
from pathlib import Path

# Reduce allocator fragmentation so a large contiguous pixel_shuffle block can
# be served even when the desktop holds ~3 GiB. Must be set before torch inits
# CUDA, hence before importing torch below.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmagic.registry import MODELS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiled_infer import tiled_sr  # noqa: E402

MIM = os.environ.get("MMAGIC_CONFIG_ROOT", str(files("mmagic").joinpath(".mim", "configs")))
CONFIGS = {
    "basicvsrpp": (f"{MIM}/basicvsr_pp/basicvsr-pp_c64n7_8xb1-600k_reds4.py",
                   "basicvsrpp_x4_reds4.pth"),
    "realbasicvsr": (f"{MIM}/real_basicvsr/"
                     "realbasicvsr_c64b20-1x30x8_8xb1-lr5e-5-150k_reds.py",
                     "realbasicvsr_x4_reds.pth"),
}
NATIVE_SCALE = 4  # official weights are x4; verified by probe_forward.py


def build_sr_fn(model_key: str, weights_dir: Path, device: str):
    """Return (sr_fn, meta). sr_fn: uint8 (T,h,w,3) RGB -> float32 (4h,4w,3) in
    [0,1], the x4-super-resolved CENTRE frame. Uses the official mmagic config
    and official x4 weights with NO structural modification."""
    cfg_path, weight_name = CONFIGS[model_key]
    ckpt_path = weights_dir / weight_name
    init_default_scope("mmagic")
    cfg = Config.fromfile(cfg_path)
    model = MODELS.build(cfg.model)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    load_info = model.load_state_dict(state, strict=False)
    # Only step_counter (a training scalar, not a weight) may be unexpected.
    bad_unexpected = [k for k in load_info.unexpected_keys if k != "step_counter"]
    if load_info.missing_keys or bad_unexpected:
        raise RuntimeError(
            f"non-clean official weight load for {model_key}: "
            f"missing={load_info.missing_keys[:5]} "
            f"unexpected={bad_unexpected[:5]}")
    generator = model.generator.eval().to(device)

    @torch.inference_mode()
    def sr_fn(tile_rgb: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(tile_rgb)).float().div_(255.0)
        x = x.permute(0, 3, 1, 2).unsqueeze(0).to(device)  # (1,T,3,h,w)
        out = generator(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        center = out[:, out.shape[1] // 2]  # (1,3,H,W) centre frame
        center = center.clamp_(0, 1)[0].permute(1, 2, 0).contiguous()
        return center.cpu().numpy().astype(np.float32)

    meta = {"model_key": model_key, "config": str(Path(cfg_path).resolve()),
            "checkpoint": str(ckpt_path.resolve()),
            "unexpected_keys_ignored": load_info.unexpected_keys}
    return sr_fn, meta


def resize_x4_to_target(hr_x4: np.ndarray, out_w: int, out_h: int,
                        interp: int = cv2.INTER_CUBIC) -> np.ndarray:
    """Resize an x4 HR image (float32 [0,1], H4 x W4 x 3) down to (out_h,out_w)
    to align the x3 GT. Longitude is treated as CIRCULAR: the image is padded
    with wrapped columns before resizing and the margin is cropped afterwards,
    so sampling across the 360-degree seam uses true neighbours (not clamped
    edges). Latitude uses the native border (poles are real boundaries)."""
    h4, w4 = hr_x4.shape[:2]
    ratio = out_w / w4  # == out_h / h4 (both 0.75 for x4->x3)
    pad = 16  # source-side wrap margin (multiple of 4 so pad*ratio is integer)
    out_pad = int(round(pad * ratio))
    padded = np.concatenate(
        [hr_x4[:, w4 - pad:], hr_x4, hr_x4[:, :pad]], axis=1)
    resized = cv2.resize(padded, (out_w + 2 * out_pad, out_h),
                         interpolation=interp)
    return resized[:, out_pad:out_pad + out_w]


def scene_of(stem: str) -> str:
    """'XJ26E07185439SKR_f0010' -> 'XJ26E07185439SKR' (drop the _f###### tag)."""
    return stem.rsplit("_f", 1)[0]


def list_pairs(preview_dir: Path, hd_dir: Path) -> list[dict]:
    pairs = []
    for p in sorted(preview_dir.glob("*.jpg")):
        gt = hd_dir / p.name
        if not gt.exists():
            raise SystemExit(f"missing HD pair for {p.name}")
        pairs.append({"id": p.stem, "scene": scene_of(p.stem),
                      "lr_path": str(p), "gt_path": str(gt)})
    if not pairs:
        raise SystemExit(f"no *.jpg preview frames in {preview_dir}")
    return pairs


def load_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list(CONFIGS))
    parser.add_argument("--preview-dir", type=Path, required=True)
    parser.add_argument("--hd-dir", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "weights")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-w", type=int, default=5760)
    parser.add_argument("--target-h", type=int, default=2880)
    # Full-latitude single vertical block; longitude tiled circularly.
    parser.add_argument("--tile-h", type=int, default=960)
    parser.add_argument("--tile-w", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    sr_fn, meta = build_sr_fn(args.model, args.weights_dir, args.device)
    pairs = list_pairs(args.preview_dir, args.hd_dir)
    if args.limit:
        pairs = pairs[:args.limit]
    args.pred_dir.mkdir(parents=True, exist_ok=True)
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)

    records = []
    started = time.perf_counter()
    for row in pairs:
        lr = load_rgb(row["lr_path"])                 # (960,1920,3) uint8 RGB
        lr_seq = lr[None]                             # T=1 -> (1,960,1920,3)
        hr_x4 = tiled_sr(lr_seq, sr_fn, NATIVE_SCALE,
                         args.tile_h, args.tile_w, args.overlap)  # float [0,1]
        hr = resize_x4_to_target(hr_x4, args.target_w, args.target_h)
        hr_u8 = np.round(hr * 255.0).clip(0, 255).astype(np.uint8)
        out_png = args.pred_dir / f"{row['id']}.png"
        cv2.imwrite(str(out_png), hr_u8[..., ::-1])   # RGB->BGR
        records.append({"id": row["id"], "scene": row["scene"],
                        "prediction": str(out_png.resolve()),
                        "gt_path": str(Path(row["gt_path"]).resolve())})
        print(row["id"], flush=True)

    with args.out_manifest.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    summary = {"model": args.model, "predicted": len(records),
               "native_scale": NATIVE_SCALE,
               "target": [args.target_w, args.target_h],
               "tile": [args.tile_h, args.tile_w, args.overlap],
               "elapsed_seconds": round(time.perf_counter() - started, 1),
               **meta}
    print(json.dumps(summary, indent=2))
    (args.out_manifest.with_suffix(".run.json")).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

