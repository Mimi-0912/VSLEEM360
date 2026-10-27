#!/usr/bin/env python3
"""
vo_04_recompute_positional.py — 重算 VO 表：以"位置相对误差"取代 SE(3) RPE。

背景（MMM2027 审稿 + 2026-10-20 审计）:
  原 vo_03_evaluate.py 直接把 GT 的单位四元数 (0,0,0,1) 喂给 evo 的
  PoseRelation.translation_part / rotation_angle_deg：
    * rotation_angle_deg 算的是相对虚构参考朝向的三维旋转角，不是航向精度；
    * SE(3) 的相对平移表达在前一相机坐标系中，参考朝向全为单位阵会让
      估计与参考的平移处于不同基准 —— 即使位置完全一致、仅朝向差 90°，
      RPE_t 也会报 1.69 m（见 audit_mmm2027_20261020/vo_identity_orientation_control.json）。
  因此本脚本只保留"位置"量，不涉及任何朝向假设。

指标定义:
  ATE      : Sim(3) Umeyama 对齐后逐帧位置误差 RMSE（与原表一致，未改）。
  RPE_pos  : Sim(3) 对齐后，在全局坐标系中取相隔 delta 帧的帧对 (i, j)，
             计算 || (p_est_j - p_est_i) - (p_gt_j - p_gt_i) ||，取 RMSE。
             纯位置量，无需参考旋转。Δt≈1s 的 delta 按各轨迹中位帧间隔换算，
             与原 RPE 的 delta 选取保持一致以便对照。

输出:
  <out_csv>  每序列每方法一行
  终端打印各方法中位数汇总，可直接抄进表4。
"""
import argparse
import copy
import csv
from pathlib import Path

import numpy as np

from evo.core import metrics, sync
from evo.tools import file_interface

DEFAULT_MAX_DIFF = 0.05


def load_tum(path):
    return file_interface.read_tum_trajectory_file(str(path))


def _sim3_align(traj_gt, traj_est):
    """Sim(3) Umeyama 对齐（含尺度），原地修改 traj_est。"""
    traj_est.align(traj_gt, correct_scale=True)
    return traj_est


def ate_rmse(traj_gt, traj_est):
    """ATE RMSE（Sim3 对齐后）。注意 est 会原地对齐。"""
    traj_est.align(traj_gt, correct_scale=True)
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((traj_gt, traj_est))
    return ape.get_statistic(metrics.StatisticsType.rmse)


def positional_relative_rmse(traj_gt, traj_est, delta):
    """全局坐标系中的位置相对误差 RMSE（不依赖任何朝向）。

    对每个 i，取 j = i + delta（若越界则跳过），
        e_i = || (p_est[j] - p_est[i]) - (p_gt[j] - p_gt[i]) ||
    返回 sqrt(mean(e_i^2))。delta 为**帧数**口径。
    """
    p_gt = traj_gt.positions_xyz
    p_est = traj_est.positions_xyz
    n = min(len(p_gt), len(p_est))
    if n <= delta:
        return float("nan")
    d_gt = p_gt[delta:n] - p_gt[:n - delta]
    d_est = p_est[delta:n] - p_est[:n - delta]
    err = np.linalg.norm(d_est - d_gt, axis=1)
    return float(np.sqrt(np.mean(err ** 2)))


def positional_relative_rmse_time(traj_gt, traj_est, target_span=1.0):
    """按**真实时间跨度**配对的位置相对误差（推荐主口径）。

    对每个 i，选取使 t_j - t_i 最接近 target_span 的 j（且必须落在
    [0.75, 1.25] * target_span 内，否则该 i 不参与）。这样三个方法
    即使关键帧密度不同（ORB 稀疏、DROID/360DVO 密集），配对的
    物理时间跨度也一致，指标可比。

    返回 (rmse, median_span, n_pairs)。
    """
    ts = np.asarray(traj_gt.timestamps)
    p_gt = traj_gt.positions_xyz
    p_est = traj_est.positions_xyz
    n = min(len(ts), len(p_gt), len(p_est))
    ts, p_gt, p_est = ts[:n], p_gt[:n], p_est[:n]

    lo, hi = 0.75 * target_span, 1.25 * target_span
    errs, spans = [], []
    j = 1
    for i in range(n):
        if j <= i:
            j = i + 1
        while j < n and ts[j] - ts[i] < target_span:
            j += 1
        # 比较 j-1 与 j，取更接近 target_span 者
        cand = [k for k in (j - 1, j) if i < k < n]
        if not cand:
            break
        k = min(cand, key=lambda q: abs((ts[q] - ts[i]) - target_span))
        span = ts[k] - ts[i]
        if lo <= span <= hi:
            errs.append(np.linalg.norm((p_est[k] - p_est[i]) - (p_gt[k] - p_gt[i])))
            spans.append(span)

    if not errs:
        return float("nan"), float("nan"), 0
    errs = np.asarray(errs)
    return (float(np.sqrt(np.mean(errs ** 2))),
            float(np.median(spans)), int(len(errs)))


def evaluate_pair(gt_path, est_path, max_diff=DEFAULT_MAX_DIFF):
    traj_gt = load_tum(gt_path)
    traj_est = load_tum(est_path)

    traj_gt_s, traj_est_s = sync.associate_trajectories(
        traj_gt, traj_est, max_diff=max_diff)

    res = {"n_gt": traj_gt.num_poses, "n_est": traj_est.num_poses,
           "n_matched": traj_est_s.num_poses}
    if traj_est_s.num_poses < 10:
        res["status"] = f"too_few_matches({traj_est_s.num_poses})"
        return res

    # ATE 与原实现保持一致
    res["ate_rmse_m"] = ate_rmse(traj_gt_s, copy.deepcopy(traj_est_s))

    # 对齐后的轨迹用于位置相对误差
    est_aligned = _sim3_align(traj_gt_s, copy.deepcopy(traj_est_s))

    ts = np.asarray(traj_gt_s.timestamps)
    med_dt = float(np.median(np.diff(ts))) if len(ts) > 1 else 0.1177
    delta_1s = max(1, round(1.0 / med_dt)) if med_dt > 0 else 8
    res["delta_1s_frames"] = delta_1s

    # 帧对时间跨度诊断（原稿件称严格 1s，实际并非）
    if len(ts) > delta_1s:
        spans = ts[delta_1s:] - ts[:len(ts) - delta_1s]
        res["pair_span_median_s"] = float(np.median(spans))
        res["pair_span_max_s"] = float(np.max(spans))
    else:
        res["pair_span_median_s"] = float("nan")
        res["pair_span_max_s"] = float("nan")

    res["rpe_pos_1s_m"] = positional_relative_rmse(traj_gt_s, est_aligned, delta_1s)
    res["rpe_pos_1f_m"] = positional_relative_rmse(traj_gt_s, est_aligned, 1)

    # 主口径：按真实时间跨度 1s 配对（三方法可比）
    rmse_t, span_t, npair_t = positional_relative_rmse_time(
        traj_gt_s, est_aligned, target_span=1.0)
    res["rpe_pos_time1s_m"] = rmse_t
    res["time_pair_span_median_s"] = span_t
    res["time_pair_n"] = npair_t
    res["status"] = "ok"
    return res


CSV_FIELDS = [
    "method", "scene", "status", "n_gt", "n_est", "n_matched",
    "ate_rmse_m", "delta_1s_frames", "pair_span_median_s", "pair_span_max_s",
    "rpe_pos_1s_m", "rpe_pos_1f_m",
    "rpe_pos_time1s_m", "time_pair_span_median_s", "time_pair_n",
]


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.6f}"
    return "" if v is None else v


def run_batch(gt_root, est_root, out_csv, max_diff=DEFAULT_MAX_DIFF):
    est_root = Path(est_root)
    rows = []
    for method_dir in sorted(d for d in est_root.iterdir() if d.is_dir()):
        method = method_dir.name
        for est_path in sorted(method_dir.glob("*.txt")):
            scene = est_path.stem
            gt_path = Path(gt_root) / scene / "trajectory_gt.txt"
            row = {"method": method, "scene": scene}
            if not gt_path.exists():
                row["status"] = "no_gt"
                rows.append(row)
                continue
            try:
                row.update(evaluate_pair(gt_path, est_path, max_diff))
            except Exception as exc:
                row["status"] = f"error:{type(exc).__name__}"
            rows.append(row)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in CSV_FIELDS})

    print(f"\n写出 {len(rows)} 行 → {out_csv}\n")
    print(f"{'method':<10} {'ok':>4} {'medATE':>9} {'medRPEpos1s':>12} "
          f"{'medSpan_s':>10} {'maxSpan_s':>10}")
    for method in sorted({r['method'] for r in rows}):
        sub = [r for r in rows if r['method'] == method and r.get('status') == 'ok']
        if not sub:
            continue
        ate = np.median([r['ate_rmse_m'] for r in sub])
        rpe = np.median([r['rpe_pos_1s_m'] for r in sub])
        span = np.median([r['pair_span_median_s'] for r in sub])
        mspan = np.max([r['pair_span_max_s'] for r in sub])
        print(f"{method:<10} {len(sub):>4} {ate:>9.4f} {rpe:>12.4f} "
              f"{span:>10.4f} {mspan:>10.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--gt_root")
    ap.add_argument("--est_root")
    ap.add_argument("--out_csv")
    ap.add_argument("--max_diff", type=float, default=DEFAULT_MAX_DIFF)
    args = ap.parse_args()

    if args.batch:
        run_batch(args.gt_root, args.est_root, args.out_csv, args.max_diff)
    else:
        ap.error("目前只支持 --batch")


if __name__ == "__main__":
    main()
