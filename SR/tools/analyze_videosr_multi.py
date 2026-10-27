#!/usr/bin/env python3
"""Analyse multi-temporal video-SR results (3 clips per scene: early/middle/late).

Two questions this answers:
  1. Are the single-clip main-table numbers representative of the sequence?
     -> per-scene spread across the three temporal segments.
  2. Do the between-model conclusions survive the wider sampling?
     -> paired comparison per (scene, segment), clustered by scene.

Statistics note: the three segments of one scene are NOT independent samples --
they come from the same video, same camera, same lighting. Treating all 3N
clips as independent would understate the standard error. So the primary
summary is reported two ways:
    * mean over clips (comparable to the single-clip table), and
    * mean over scene means (each scene counts once), plus a paired t-test on
      scene-level means for within-model comparisons.

Reads <root>/<model>/per_frame.csv produced by eval_videosr.py on a multi-clip
manifest (columns: scene, id, psnr, ssim, lpips, ws_psnr, ws_ssim). The `seq`
directory name carries the segment suffix (scene__early), and eval_videosr.py
records `scene` only, so segment is recovered from `id` (scene:segment) when
present, otherwise from an optional --manifest join.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ["psnr", "ssim", "lpips", "ws_psnr", "ws_ssim"]
HIGHER_BETTER = {"psnr": True, "ssim": True, "lpips": False,
                 "ws_psnr": True, "ws_ssim": True}
SEGMENTS = ["early", "middle", "late"]


def load_per_frame(path: Path, manifest: Path | None) -> list[dict]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    # `scene` in per_frame.csv is clip["scene"] (no suffix). Recover the segment.
    seg_of: dict[str, str] = {}
    if manifest and manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            seg_of[r["id"]] = r.get("segment", "middle")
    for r in rows:
        r["segment"] = seg_of.get(r["id"], "middle")
        for m in METRICS:
            r[m] = float(r[m])
    return rows


def summarise(rows: list[dict], label: str) -> dict:
    scene_means = {}
    by_scene = defaultdict(list)
    for r in rows:
        by_scene[r["scene"]].append(r)
    for scene, rs in by_scene.items():
        scene_means[scene] = {m: float(np.mean([x[m] for x in rs])) for m in METRICS}

    out = {"label": label, "n_clips": len(rows), "n_scenes": len(by_scene),
           "mean_over_clips": {}, "std_over_clips": {},
           "mean_over_scene_means": {}, "per_scene": scene_means}
    for m in METRICS:
        vals = np.array([r[m] for r in rows])
        out["mean_over_clips"][m] = float(vals.mean())
        out["std_over_clips"][m] = float(vals.std(ddof=0))
        out["mean_over_scene_means"][m] = float(
            np.mean([scene_means[s][m] for s in scene_means]))
    return out


def temporal_spread(rows: list[dict]) -> dict:
    """Within-scene spread across the three segments (representativeness proxy)."""
    by_scene = defaultdict(dict)
    for r in rows:
        by_scene[r["scene"]][r["segment"]] = r
    out = {}
    for m in METRICS:
        rngs, stds = [], []
        for scene, segs in by_scene.items():
            vals = [segs[s][m] for s in SEGMENTS if s in segs]
            if len(vals) >= 2:
                rngs.append(max(vals) - min(vals))
                stds.append(float(np.std(vals, ddof=0)))
        out[m] = {"median_range_across_segments": float(np.median(rngs)) if rngs else None,
                  "p90_range": float(np.percentile(rngs, 90)) if rngs else None,
                  "median_std": float(np.median(stds)) if stds else None,
                  "n_scenes_with_2plus": len(rngs)}
    return out


def paired_models(a: list[dict], b: list[dict]) -> dict:
    """Paired comparison of two models on the same (scene, segment) clips."""
    idx_a = {(r["scene"], r["segment"]): r for r in a}
    idx_b = {(r["scene"], r["segment"]): r for r in b}
    common = sorted(set(idx_a) & set(idx_b))
    out = {"n_pairs": len(common), "metrics": {}}
    for m in METRICS:
        d = np.array([idx_a[k][m] - idx_b[k][m] for k in common])
        better = (d > 0) if HIGHER_BETTER[m] else (d < 0)
        try:
            from scipy import stats as st
            t, p = st.ttest_rel([idx_a[k][m] for k in common],
                                [idx_b[k][m] for k in common])
        except Exception:
            t, p = float("nan"), float("nan")
        out["metrics"][m] = {"mean_diff": float(d.mean()),
                             "t": float(t), "p": float(p),
                             "wins": int(better.sum()), "n": int(len(d))}
    return out


def paired_sequence_means(a: list[dict], b: list[dict]) -> dict:
    """Primary paired test: one equally weighted mean per shared sequence."""
    ga, gb = defaultdict(list), defaultdict(list)
    for r in a:
        ga[r['scene']].append(r)
    for r in b:
        gb[r['scene']].append(r)
    common = sorted(set(ga) & set(gb))
    if any(len(ga[s]) != 3 or len(gb[s]) != 3 for s in common):
        raise ValueError('Expected three clips per sequence for both methods')
    from scipy.stats import ttest_rel
    out = {'n_pairs': len(common), 'unit': 'sequence mean over three clips', 'metrics': {}}
    for m in METRICS:
        x = np.array([np.mean([r[m] for r in ga[s]]) for s in common])
        y = np.array([np.mean([r[m] for r in gb[s]]) for s in common])
        t, p = ttest_rel(x, y)
        out['metrics'][m] = {'mean_diff': float(np.mean(x-y)), 't': float(t),
            'p': float(p), 'wins': int(((x>y) if HIGHER_BETTER[m] else (x<y)).sum()), 'n': len(common)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True,
                    help="directory containing <model>/per_frame.csv")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="multi-clip clips_manifest.jsonl (recovers segment labels)")
    ap.add_argument("--models", default="basicvsrpp,realbasicvsr")
    ap.add_argument("--prefix", default="",
                    help="subdirectory prefix, e.g. 'videosr_3t_' for runs/videosr_3t_<model>/")
    ap.add_argument("--out", type=Path, default=None, help="write JSON summary here")
    args = ap.parse_args()

    models = [m for m in args.models.split(",") if m]
    data = {}
    for model in models:
        p = args.root / f"{args.prefix}{model}" / "per_frame.csv"
        if not p.exists():
            print(f"[skip] {model}: no {p}", file=sys.stderr)
            continue
        data[model] = load_per_frame(p, args.manifest)

    if not data:
        raise SystemExit("no per_frame.csv found")

    report = {"segments": SEGMENTS, "models": {}}
    for model, rows in data.items():
        s = summarise(rows, model)
        s["temporal_spread"] = temporal_spread(rows)
        report["models"][model] = s

    if len(data) >= 2:
        a, b = models[0], models[1]
        report["paired"] = {"a": a, "b": b, **paired_models(data[a], data[b])}
        report["paired_sequence_means"] = {"a": a, "b": b, **paired_sequence_means(data[a], data[b])}

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True),
                            encoding="utf-8")
        print(f"[written] {args.out}")

    # Human-readable table
    print(f"\n{'model':<16} {'n_clip':>6} {'n_scene':>7} "
          + " ".join(f"{m:>9}" for m in METRICS))
    for model, s in report["models"].items():
        print(f"{model:<16} {s['n_clips']:>6} {s['n_scenes']:>7} "
              + " ".join(f"{s['mean_over_clips'][m]:>9.4f}" for m in METRICS))
        print(f"{'  (scene-mean)':<16} {'':>6} {'':>7} "
              + " ".join(f"{s['mean_over_scene_means'][m]:>9.4f}" for m in METRICS))

    print("\nWithin-scene spread across the three segments "
          "(median range / p90 range):")
    for model, s in report["models"].items():
        ts = s["temporal_spread"]
        print(f"  {model}:")
        for m in METRICS:
            v = ts[m]
            if v["median_range_across_segments"] is None:
                continue
            print(f"    {m:<8} median {v['median_range_across_segments']:.4f}   "
                  f"p90 {v['p90_range']:.4f}   (n={v['n_scenes_with_2plus']})")

    if "paired" in report:
        p = report["paired"]
        print(f"\nPaired {p['a']} - {p['b']} over {p['n_pairs']} matched clips:")
        for m in METRICS:
            v = p["metrics"][m]
            print(f"  {m:<8} diff {v['mean_diff']:+.4f}  t {v['t']:+.2f}  "
                  f"p {v['p']:.2e}  wins {v['wins']}/{v['n']}")


if __name__ == "__main__":
    main()
