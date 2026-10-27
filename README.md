# VSLEEM360

**A Large-Scale 360-Degree Panoramic Video Dataset for Near-Ground Mobile Perception**

Accepted as a full paper at MMM 2027 (paper 472).

Authors: Ziyanran Wang, Donghui Li, Xiaoying Li, Hui Li, Hengyang Luo,
Hang Zheng, and Fei Cheng. Contact: Fei.Cheng@xjtlu.edu.cn.

## Release Status

This is a **private preparation snapshot**, prepared on 2026-10-27, in
https://github.com/Mimi-0912/VSLEEM360. It is not yet a
public dataset release or a complete installation of every baseline. The current
package contains original evaluation and diagnostic scripts, frozen splits,
reference result files, and public data indexes. Raw videos, frozen VPR images,
large feature caches, model weights, and third-party implementations are separate.

Dataset hosting is confirmed as **Google Drive**; the shared download URL,
available capacity, release version, and publication date remain to be finalized.
No placeholder URL is presented as a working download.

See `docs/REPRODUCTION_STATUS.md` for the distinction between verified offline
statistics and the model installation/inference steps still being prepared.

The data includes 421 sequences (approximately 78.3 hours), HD 5760x2880 and
Preview 1920x960 ERP streams, GPS/IMU records, and sequence attributes. The
frozen sequence split has 252 training, 84 validation, and 85 test sequences.

## Contents

| Path | Purpose |
|---|---|
| `VPR/code/` | Official MixVPR/EigenPlaces extraction wrapper, geographic protocol, and retrieval evaluator |
| `VPR/results/` | Current frozen results for 15 evaluated configurations |
| `VPR/weights/checkpoint_manifest.json` | Checkpoint filenames, sizes, and SHA-256 values; no weight files |
| `VO/code/` | Positional ATE/RPE evaluation from TUM trajectories and reference-speed sensitivity |
| `VO/results/` | Saved per-sequence metrics and sensitivity results |
| `SR/tools/` | Per-sequence statistical analysis and image alignment diagnostics |
| `SR/runs/` | Per-clip results for two models, sequence summaries, and segment identity join |
| `release_manifest/` | Sequence indexes, file hashes, attributes, split lists, and data card |
| `SOURCE_PROVENANCE.json` | Hashes of original project files copied without modification |

## Offline Result Analysis

Use a separate Python environment. `requirements-analysis.txt` covers the
lightweight statistics. It does not install the inference models or their CUDA
extensions.

```bash
python -m pip install -r requirements-analysis.txt
```

Recompute the VSR summary from the included 255 per-clip measurements:

```bash
python SR/tools/analyze_videosr_multi.py --root SR/runs --prefix videosr_3t_ --manifest SR/runs/clips_segments.jsonl --out analysis/vsr_summary.json
```

The primary paired test uses one mean for each of 85 sequences, not 255
independent observations. The segment file contains identity fields for joining
results; it is not an extraction manifest or a substitute for the actual clips.

## Benchmark Inputs and Model Execution

VPR evaluation additionally needs the frozen `vpr_frames/` metadata and JPEGs,
plus feature arrays in `VPR/features/`. Preserve the frozen JPEG bytes: decoding
new images from reencoded videos does not reproduce the same inputs.

After obtaining those external inputs and installing FAISS, the canonical
retrieval command is:

```bash
python VPR/code/vpr_19_evaluate_camera_ready.py --root . --cpu --out analysis/vpr_results.json
```

The extraction wrapper requires PyTorch, torchvision, Pillow, tqdm, the named
checkpoints, and official model files under `VPR/third_party/`. The third-party
files are deliberately not bundled without their upstream license notices.
The separate model environments, all preprocessing/inference wrappers, and 3D
reconstruction workflow still need packaging before claiming full reproduction
from videos in a clean environment. The supplied scripts are not a complete
one-command reproduction package.

Official model sources:

- MixVPR: https://github.com/amaralibey/MixVPR
- EigenPlaces: https://github.com/gmberton/EigenPlaces
- CosPlace: https://github.com/gmberton/CosPlace
- NetVLAD: https://github.com/Relja/netvlad
- ORB-SLAM3: https://github.com/UZ-SLAMLab/ORB_SLAM3
- DROID-SLAM: https://github.com/princeton-vl/DROID-SLAM
- 360DVO: https://github.com/chris1004336379/360DVO
- BasicVSR++: https://github.com/ckkelvinchan/BasicVSR_PlusPlus
- RealBasicVSR: https://github.com/ckkelvinchan/RealBasicVSR
- 3D Gaussian Splatting: https://github.com/graphdeco-inria/gaussian-splatting

## Protocol Limits

VPR uses float64 geographic coordinates and an inclusive 25 m positive radius;
the geographic split has 19,342 database frames and 11,457 queries, of which
10,622 have a positive. EigenPlaces uses 2:1 ERP inputs; MixVPR is reported
separately at 320x320 with its released fixed-grid aggregator.

Consumer GPS is a reference, not independently calibrated ground truth. Raw
GPS logs and camera-exported trajectories have different sampling rates. The
65-sequence sensitivity subset is not a certification of GPS accuracy, and it
does not replace the 421-sequence dataset or the 85-sequence test set.

The image diagnostic covered all 255 VSR clips: 246 peaks within one frame,
eight at two or three frames, and one at the search boundary. It does not
measure physical clock synchronization error. The boundary case remains
unresolved and is retained in the reported evaluation.

Selected videos are anonymized. The known three-sequence repair has checks
of frame counts, presentation timestamps, and sampled frames; no claim of
exhaustive human inspection of every frame is made. See the data card.

## Licenses and Citation

The **dataset** license is CC BY-NC 4.0; see `release_manifest/LICENSE.txt`.
Original project scripts use **PolyForm Noncommercial License 1.0.0**;
see `LICENSE.md` and `LICENSE_SCOPE.md`. Commercial use is not granted.
Third-party software and pretrained weights retain their original conditions.
This noncommercial source-available snapshot is not an unrestricted
open-source release.

Please cite the paper title and authors above. The proceedings DOI is not yet
available, and no DOI has been invented. Machine-readable metadata is provided
in `CITATION.cff`.
