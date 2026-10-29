# Reproduction Guide

Run commands from the repository root. Python 3.10 is used for VSR. Results
under `analysis/` are local outputs and do not replace the frozen reference
results. Data upload is in progress; commands requiring the full dataset are
provided for use after the verified download is available.

## Saved-result analysis

Only NumPy and SciPy are needed to repeat the main VSR statistical analysis:

```bash
python -m pip install -r requirements-analysis.txt
```

```bash
python SR/tools/analyze_videosr_multi.py --root SR/runs --prefix videosr_3t_ --manifest SR/runs/clips_segments.jsonl --out analysis/vsr_summary.json
```

This reads 255 measurements per model and reports paired tests on 85 sequence
means. It does not run either neural network. Compare with
`SR/runs/videosr_3t_summary.json`.

## Dataset layout

The downloaded dataset root contains `<scene>/<scene>_hd_process.mp4`,
`<scene>/<scene>_preview_process.mp4`, sensor records, and `vpr_frames/` with
the four frozen JPEG directories plus metadata and geographic split CSVs.
Keep the frozen JPEG bytes; do not regenerate them from reencoded videos.

Set the dataset location to your actual download directory:

```bash
export VSLEEM_DATA_ROOT=/path/to/VSLEEM360
```

For VPR scripts, link its frozen image directory into the code checkout:

```bash
ln -s "$VSLEEM_DATA_ROOT/vpr_frames" vpr_frames
```

The dataset manifest records all 421 sequences, split as 252/84/85. The
diagnostic 65-sequence GPS subset does not replace this split.

The attributes distinguish `hd_video_frames`, `preview_video_frames`, and
`trajectory_points`. The legacy `num_frames` column aliases `trajectory_points`,
not HD video frames. Raw GPS logs are nonempty in 409 sequences; both IMU logs
exist in 420 sequences. `XJ26E13091126DSC` lacks accelerometer/gyroscope logs.

## VSR environment and official weights

`SR/environment.yml` pins Python 3.10, PyTorch 2.1.2, torchvision 0.16.2,
CUDA 11.8, MMagic 1.2.0, MMCV 2.1.0 and MMEngine 0.10.3. The compiled MMCV
operations must match PyTorch and CUDA; follow the OpenMMLab wheel index in
that file. FFmpeg/ffprobe is needed for index-based video decoding.

```bash
conda env create -f SR/environment.yml
```

```bash
conda activate vsleem360-sr
```

Download the official weights into `SR/weights/` with these local names:

| Model | Local name | Official source |
|---|---|---|
| BasicVSR++ REDS4 x4 | `basicvsrpp_x4_reds4.pth` | [OpenMMLab checkpoint](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth) |
| RealBasicVSR REDS x4 | `realbasicvsr_x4_reds.pth` | [OpenMMLab checkpoint](https://download.openmmlab.com/mmediting/restorers/real_basicvsr/realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds_20211104-52f77c2c.pth) |

```bash
python tools/verify_checkpoints.py --task vsr
```

The loader locates configurations in the installed MMagic package, rather than
in an author's home directory. `MMAGIC_CONFIG_ROOT` can override that config
directory. The exact configs are:

- `basicvsr_pp/basicvsr-pp_c64n7_8xb1-600k_reds4.py`
- `real_basicvsr/realbasicvsr_c64b20-1x30x8_8xb1-lr5e-5-150k_reds.py`

Missing weight tensors are fatal. Only RealBasicVSR's non-weight training
scalar `step_counter` may be ignored. No architecture is changed to x3.
The official configs can also populate PyTorch's cache with SpyNet/VGG
initialization weights from their upstream URLs on first use, so model setup
may need internet access even when the main checkpoint is already present.

## Decode the frozen VSR clips

`SR/manifests/clips_manifest.jsonl` has the original 255 frame-index records
with paths relative to the dataset root. Decode into a new output directory:

```bash
python SR/tools/materialize_frozen_videosr_clips.py --data-root "$VSLEEM_DATA_ROOT" --out-root SR/clips/frozen_255
```

This writes each saved 15-frame LR window and its centre HD target, without
reselecting quantiles or fitting an alignment. For a first smoke check use
`--limit 1` and a different, new output directory. Decoding all 255 clips
requires substantial free space; a raw-RGB upper bound is about 34 GB before
PNG compression. Decoder versions can affect bytes, so exact frame identities
alone are not a promise of byte-identical redecoding on another machine.

## Run VSR inference and metrics

These are the settings used for the reported video experiment:

```bash
python SR/tools/eval_videosr.py --model basicvsrpp --clips-root SR/clips/frozen_255 --weights-dir SR/weights --tile-h 384 --tile-w 512 --overlap 64 --out-dir analysis/videosr_3t_basicvsrpp
```

```bash
python SR/tools/eval_videosr.py --model realbasicvsr --clips-root SR/clips/frozen_255 --weights-dir SR/weights --tile-h 384 --tile-w 512 --overlap 64 --out-dir analysis/videosr_3t_realbasicvsr
```

```bash
python SR/tools/analyze_videosr_multi.py --root analysis --prefix videosr_3t_ --manifest SR/runs/clips_segments.jsonl --out analysis/recomputed_vsr_summary.json
```

Both models infer at x4, then resize with bicubic interpolation and circular
horizontal padding to the 5760x2880 x3 target. PSNR/SSIM use BT.601 Y and a
3-pixel crop; LPIPS uses full-frame RGB/AlexNet; WS metrics use cosine-latitude
weights. The 255-clip image diagnostic has 246 peaks within one frame, eight
at two or three frames, and one unresolved boundary case. It is not physical
clock calibration. The reported scores retain all clips.

## VPR extraction and retrieval

VPR needs PyTorch/torchvision, Pillow, tqdm, NumPy and SciPy. Retrieval also
requires FAISS (CPU is supported). Official weights are listed in
`VPR/weights/checkpoint_manifest.json`; they are not bundled.

| Model | Weight source |
|---|---|
| MixVPR ResNet50 / 4096 | [Official weights table](https://github.com/amaralibey/MixVPR); choose the 4096-D row and save as `mixvpr_4096_official.ckpt` |
| EigenPlaces ResNet50 / 2048 | [Official release](https://github.com/gmberton/EigenPlaces/releases/download/v1.0/ResNet50_2048_eigenplaces.pth) |
| CosPlace ResNet18 / 512 | [Official release](https://github.com/gmberton/CosPlace/releases/download/v0.1/ResNet18_512_places.pth) |
| NetVLAD VGG16 / Pitts30K | [Pitts30K weights](https://cvg-data.inf.ethz.ch/hloc/netvlad/Pitts30K_struct.mat), locally named `VGG16-NetVLAD-Pitts30K.mat` |

The extraction wrapper imports `eigenplaces_model` and `mixvpr_model` under
`VPR/third_party/`. Obtain the implementations from the official repositories,
keeping their notices. The EigenPlaces package is `eigenplaces_model/` in
[EigenPlaces](https://github.com/gmberton/EigenPlaces); the MixVPR import maps
to `models/` in [MixVPR](https://github.com/amaralibey/MixVPR). Upstream commits
inspected for these paths on 2026-10-28 are EigenPlaces
`a2969f71d5ea31017443af490b15273ca4c50af1` and MixVPR
`4043915cef24818003ece1a8112bc8a24e69abe0`; these are not asserted to be the
original experiment checkout commits. The original local EigenPlaces loader
skipped upstream backbone initialization before strict full-checkpoint loading.

For a new checkout, prepare the import paths below. These commands fetch
third-party sources from their authors; those sources keep their own terms.

```bash
mkdir -p VPR/third_party
```

```bash
git clone https://github.com/gmberton/EigenPlaces.git VPR/third_party/EigenPlaces
```

```bash
git -C VPR/third_party/EigenPlaces checkout a2969f71d5ea31017443af490b15273ca4c50af1
```

```bash
ln -s EigenPlaces/eigenplaces_model VPR/third_party/eigenplaces_model
```

```bash
git clone https://github.com/amaralibey/MixVPR.git VPR/third_party/MixVPR
```

```bash
git -C VPR/third_party/MixVPR checkout 4043915cef24818003ece1a8112bc8a24e69abe0
```

```bash
ln -s MixVPR/models VPR/third_party/mixvpr_model
```

```bash
python VPR/code/vpr_17_extract_official.py --model eigenplaces_official --stream preview --split database --size 224x448
```

Repeat for `--split query`, both streams, and the desired reported size.
EigenPlaces Preview sizes are 224x448, 256x512, 384x768 and 512x1024. MixVPR uses
`--model mixvpr_official --size 320x320`; its fixed aggregator is incompatible
with the tested 2:1 sizes. Each output sidecar records preprocessing and the
checkpoint digest. Preserve metadata row order.

The retrieval command below expects **all 15 configurations' feature arrays**
named in `vpr_19_evaluate_camera_ready.py`, including separately generated
CosPlace/NetVLAD and GPS-course features. They are not generated by the
MixVPR/EigenPlaces wrapper or bundled as frozen caches in this repository.

```bash
python VPR/code/vpr_19_evaluate_camera_ready.py --root . --cpu --out analysis/vpr_results.json
```

This uses 19,342 database images and 11,457 queries: 10,622 answerable and 835
unanswerable, with inclusive distance <=25 m and float64 local geography.
The authoritative reference is `VPR/results/camera_ready_20261026.json`.

## VO evaluation and reference sensitivity

The metric evaluator needs `evo` and NumPy. Obtain predicted TUM-format files
under `VO/trajectories/<method>/<scene>.txt`, and camera-derived reference files
under `VO/reference/<scene>/trajectory_gt.txt`. Those are generated inputs,
not the raw GPS text logs. The evaluation code does not estimate trajectories.

```bash
python VO/code/vo_04_recompute_positional.py --batch --gt_root VO/reference --est_root VO/trajectories --out_csv analysis/vo_metrics_positional.csv
```

Use the `rpe_pos_time1s_m` column for the reported positional 1-second RPE.
The CLI's historical terminal summary also prints a frame-delta variant;
that variant must not be substituted for the time-based paper metric.

The saved-result sensitivity analysis additionally reads the released
`<scene>/<scene>_trajectory.csv` exports:

```bash
python VO/code/vo_08_reference_sensitivity.py --metrics VO/results/vo_metrics_positional.csv --data-root "$VSLEEM_DATA_ROOT" --out analysis/vo_reference_sensitivity.json
```

This diagnoses intervals faster than 20 m/s. The 84-to-65 subset changes the
median-ATE gap from 61.5% to 38.4%; it does not certify reference accuracy.

## External workflows and validation boundary

Full ORB-SLAM3, DROID-SLAM, 360DVO and 3DGS installations/runners are not
included. Use their official repositories and the configurations in the paper.
No fresh-environment all-model rerun is claimed. The supplied frozen results
are separated from newly generated outputs, and the data download remains
unavailable until cloud verification finishes. See `REPRODUCTION_STATUS.md`.
