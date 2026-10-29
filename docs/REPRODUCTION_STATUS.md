# Reproduction Status

Updated on 2026-10-29. This release contains the evaluation and diagnostic
files listed in the main README. Videos, images, and checkpoints are separate;
dataset v1.0 is distributed through the Google Drive link below.
See `REPRODUCING.md` for exact commands.

| Part | Included and checked | Additional requirements |
|---|---|---|
| VSR saved-result statistics | 255 per-clip measurements, 85 sequence identities, paired sequence-mean analysis; rerun matches the frozen summary | Python, NumPy, SciPy |
| VPR evaluation | Official extraction wrapper, geographic protocol, current result JSON, checkpoint hashes | Frozen JPEGs/metadata, FAISS, model environments, official implementations and checkpoints |
| VO evaluation | Positional ATE/RPE and GPS reference-sensitivity scripts with saved results | Predicted trajectories, camera-exported reference trajectories, environment and path setup |
| VSR model inference | Pinned environment, official configuration names, weight hashes, 255 relative-path clip records, frozen-frame decoder, tiler and inference/evaluation scripts are included | Download the exact official weights and dataset; a full fresh-environment inference rerun is not claimed |
| 3D reconstruction | No reconstruction implementation is bundled here | Separate environment, pose/point preparation and documented commands are still needed |

The verified offline command is in the main README. Its successful execution
does not establish that all baseline models can run in a clean environment.
The VSR decoder must use the saved indices; it does not repeat the cancelled
boundary investigation or alter any scored input. Diagnostics describe image
matching, not an independent physical synchronization calibration.

Dataset v1.0 (29 October 2026):
https://drive.google.com/drive/folders/1YpiFDGTrzWICMcrFoox61fHxbW8m3Rr5
(read-only). Preserve the frozen VPR
JPEG bytes and evaluation splits. Do not regenerate a different set of images
from reencoded videos and claim to reproduce the same inputs.

The dataset license is CC BY-NC 4.0. Original project scripts use PolyForm
Noncommercial License 1.0.0, as selected to implement the authors'
noncommercial requirement. See `LICENSE_SCOPE.md` for the separation from
dataset and third-party material. Third-party implementations and weights
retain their own conditions and are not redistributed in this snapshot.
