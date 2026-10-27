#!/usr/bin/env python3
"""Extract VPR features with the OFFICIAL MixVPR / EigenPlaces implementations.

Why this exists
---------------
vpr_05_run_mixvpr.py and vpr_06_run_eigenplaces.py did not run the published
models: they built an ImageNet ResNet backbone, an average-pooling layer
(misleadingly named ``gem``), and a randomly initialised linear projection that
was never loaded from any checkpoint, then labelled the output "MixVPR" and
"EigenPlaces". See VPR/weights/SOURCES.txt. Every number those scripts produced
for those two baselines must be regenerated with this script.

Input protocols (deliberately different, and this difference is disclosed)
--------------------------------------------------------------------------
EigenPlaces
    native 224x448 and any other 2:1 size. Its GeM pooling is defined as a
    generalised mean over the whole feature map, so it is size- and
    aspect-ratio-agnostic; verified by forward passes at 224x448, 320x320 and
    512x1024 producing (N, 2048) descriptors.

MixVPR
    released 320x320 configuration. Its aggregator projects the flattened 20x20 feature
    grid (`row_proj = Linear(400, 4)`), so the tested 224x448/256x512/384x768/512x1024
    inputs are incompatible with that fixed projection. The paper reports
    MixVPR in a separate block at 320x320 below the common 224x448 protocol;
    no general claim is made about every possible factorization of 400 positions.

Outputs (VPR/features/)
-----------------------
  {eigenplaces_official,mixvpr_official}_features_{database,query}_{stream}{tag}.npy
  plus a .json sidecar with model identity, checkpoint digest and preprocessing.

Both files preserve the metadata row order, so downstream evaluation can join on
frame_name exactly as the existing scripts do.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(os.environ.get("VSLEEM_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
FRAMES = ROOT / "vpr_frames"
FEAT = ROOT / "VPR" / "features"
WEIGHTS = ROOT / "VPR" / "weights"
sys.path.insert(0, str(ROOT / "VPR" / "third_party"))

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------- data
class FrameDataset(Dataset):
    def __init__(self, image_dir: Path, metadata_csv: Path, size: tuple[int, int]):
        self.image_dir = image_dir
        with metadata_csv.open(encoding="utf-8") as fh:
            self.rows = list(csv.DictReader(fh))
        self.tf = T.Compose([
            T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        path = self.image_dir / f"{self.rows[i]['frame_name']}.jpg"
        img = Image.open(path).convert("RGB")
        return self.tf(img), i


# ---------------------------------------------------------------- models
def build_eigenplaces(checkpoint: Path):
    """Official GeoLocalizationNet_, ResNet50 + GeM + Linear(2048->2048)."""
    from eigenplaces_model.eigenplaces_network import GeoLocalizationNet_
    model = GeoLocalizationNet_("ResNet50", 2048)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval()


class OfficialMixVPR(torch.nn.Module):
    """Official MixVPR aggregator over an official torchvision ResNet-50 trunk.

    Mirrors models/backbones/resnet.py with layers_to_crop=[4] and
    models/aggregators/mixvpr.py, so the released 4096-D checkpoint loads with
    strict=True.
    """

    def __init__(self) -> None:
        super().__init__()
        import torchvision
        from mixvpr_model.aggregators.mixvpr import MixVPR

        # Mirror models/backbones/resnet.py: a torchvision ResNet with the
        # layer3 output exposed, layer4/avgpool/fc removed, and the whole thing
        # held under `backbone.model` so checkpoint keys match exactly
        # (backbone.model.conv1..., backbone.model.layer3.5.bn3...).
        resnet = torchvision.models.resnet50(weights=None)
        resnet.layer4 = None
        resnet.avgpool = None
        resnet.fc = None

        self.backbone = torch.nn.Module()
        self.backbone.model = resnet
        self.aggregator = MixVPR(in_channels=1024, in_h=20, in_w=20,
                                 out_channels=1024, mix_depth=4,
                                 mlp_ratio=1, out_rows=4)

    def forward(self, x):
        m = self.backbone.model
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        return self.aggregator(x)


def build_mixvpr(checkpoint: Path):
    model = OfficialMixVPR()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval()


MODELS = {
    "eigenplaces_official": {
        "build": build_eigenplaces,
        "ckpt": WEIGHTS / "ResNet50_2048_eigenplaces.pth",
        "size": (224, 448),          # (H, W) native 2:1
        "tag": "_224x448",
        "dim": 2048,
        "identity": "ResNet50 + GeM + Linear(2048,2048), SF-XL weights",
    },
    "mixvpr_official": {
        "build": build_mixvpr,
        "ckpt": WEIGHTS / "mixvpr_4096_official.ckpt",
        "size": (320, 320),          # official demo input; aggregator grid is fixed
        "tag": "_320x320",
        "dim": 4096,
        "identity": "ResNet50(layer3) + MixVPR(20x20 grid), GSV-Cities 4096-D",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- extraction
@torch.no_grad()
def extract(model, loader, device, dim: int) -> np.ndarray:
    out = np.zeros((len(loader.dataset), dim), dtype=np.float32)
    for imgs, idx in tqdm(loader, desc="  extracting", ncols=90):
        feats = model(imgs.to(device, non_blocking=True))
        feats = F.normalize(feats.float(), p=2, dim=1)
        out[idx.numpy()] = feats.cpu().numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--stream", required=True, choices=["hd", "preview"])
    ap.add_argument("--split", required=True, choices=["database", "query"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="overwrite existing output")
    ap.add_argument("--size", default="",
                    help="override input as HxW, e.g. 256x512 (resolution ablation)")
    args = ap.parse_args()

    spec = dict(MODELS[args.model])
    if args.size:
        h, w = (int(x) for x in args.size.lower().split("x"))
        spec["size"] = (h, w)
        spec["tag"] = f"_{h}x{w}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_path = FEAT / (f"{args.model}_features_{args.split}_{args.stream}"
                       f"{spec['tag']}.npy")
    if out_path.exists() and not args.force:
        print(f"[skip] {out_path.name} exists (use --force to overwrite)")
        return
    if not spec["ckpt"].exists():
        raise SystemExit(f"missing checkpoint: {spec['ckpt']}")

    print(f"model      {args.model}  ({spec['identity']})")
    print(f"checkpoint {spec['ckpt'].name}  sha256 {sha256(spec['ckpt'])[:16]}...")
    print(f"input      {spec['size'][0]}x{spec['size'][1]}")
    print(f"split      {args.split}/{args.stream}")
    print(f"device     {device}")

    ds = FrameDataset(FRAMES / f"{args.split}_{args.stream}",
                      FRAMES / f"{args.split}_metadata.csv",
                      spec["size"])
    if args.limit:
        ds.rows = ds.rows[:args.limit]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    model = spec["build"](spec["ckpt"]).to(device)
    t0 = time.perf_counter()
    feats = extract(model, loader, device, spec["dim"])
    elapsed = time.perf_counter() - t0

    assert feats.shape == (len(ds.rows), spec["dim"]), feats.shape
    FEAT.mkdir(parents=True, exist_ok=True)
    np.save(out_path, feats)

    sidecar = out_path.with_suffix(".json")
    sidecar.write_text(json.dumps({
        "model": args.model,
        "identity": spec["identity"],
        "checkpoint": spec["ckpt"].name,
        "checkpoint_sha256": sha256(spec["ckpt"]),
        "input_hw": list(spec["size"]),
        "split": args.split,
        "stream": args.stream,
        "frames": len(ds.rows),
        "dim": spec["dim"],
        "normalization": "L2, applied after the model's own output",
        "preprocess": {"resize": list(spec["size"]), "interpolation": "bicubic",
                       "mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "elapsed_s": round(elapsed, 1),
        "device": str(device),
    }, indent=2), encoding="utf-8")

    print(f"\n[written] {out_path.name}  shape {feats.shape}  "
          f"{elapsed/60:.1f} min")
    print(f"[written] {sidecar.name}")


if __name__ == "__main__":
    main()
