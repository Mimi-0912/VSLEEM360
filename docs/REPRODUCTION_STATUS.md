# Reproduction Status

Prepared on 2026-10-27. This private snapshot contains the evaluation and
diagnostic files listed in the main README. It is not a complete baseline
installation or a public release of videos, images, or checkpoints.

| Part | Included and checked | Additional requirements |
|---|---|---|
| VSR saved-result statistics | 255 per-clip measurements, 85 sequence identities, paired sequence-mean analysis; rerun matches the frozen summary | Python, NumPy, SciPy |
| VPR evaluation | Official extraction wrapper, geographic protocol, current result JSON, checkpoint hashes | Frozen JPEGs/metadata, FAISS, model environments, official implementations and checkpoints |
| VO evaluation | Positional ATE/RPE and GPS reference-sensitivity scripts with saved results | Predicted trajectories, camera-exported reference trajectories, environment and path setup |
| VSR model inference | Official model repository links and checkpoint separation are documented | Exact environment, checkpoint/configuration selection, clip extraction and inference commands are still being packaged |
| 3D reconstruction | No reconstruction implementation is bundled here | Separate environment, pose/point preparation and documented commands are still needed |

The verified offline command is in the main README. Its successful execution
does not establish that all baseline models can run in a clean environment.

Dataset access will be provided through Google Drive. Preserve the frozen VPR
JPEG bytes and evaluation splits. Do not regenerate a different set of images
from reencoded videos and claim to reproduce the same inputs.

The dataset license is CC BY-NC 4.0. Original project scripts use PolyForm
Noncommercial License 1.0.0, as selected to implement the authors'
noncommercial requirement. See `LICENSE_SCOPE.md` for the separation from
dataset and third-party material. Third-party implementations and weights
retain their own conditions and are not redistributed in this snapshot.
