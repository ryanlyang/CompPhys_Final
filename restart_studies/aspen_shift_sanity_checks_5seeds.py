#!/usr/bin/env python3
from __future__ import annotations


import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


SHIFT_METRICS = [
    "prob_l1_drift",
    "top1_flip_rate",
    "class_js_divergence",
    "confidence_drop",
    "entropy_shift",
]
STRONG_METRICS = ["prob_l1_drift", "top1_flip_rate", "class_js_divergence"]
SHARED_REPLAY_METRICS = ["delta_acc", "class_js_divergence", "confidence_drop", "entropy_shift"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run sanity checks for Aspen shift eval over 5 seeds")
    p.add_argument(
        "--results_root",
        type=Path,
        default=Path("/home/ryreu/atlas/CompPhys_FinalProject/restart_studies/results"),
    )
    p.add_argument("--run_basename", type=str, default="prelim_reimpl_cluster")
    p.add_argument("--aspen_suffix", type=str, default="_aspen_shift_1M")
    p.add_argument("--seeds", type=str, default="41,52,63,74,85")
    p.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Default: <results_root>/<run_basename>_aspen_sanity_5seeds",
    )
    p.add_argument("--clean_acc_tol", type=float, default=0.002)
    p.add_argument("--clean_auc_tol", type=float, default=0.002)
    p.add_argument("--corruption_tol", type=float, default=0.005)
    p.add_argument("--mapping_r2_min_strong", type=float, default=0.80)
    p.add_argument("--mapping_intercept_tol", type=float, default=0.03)
    p.add_argument("--loocv_mae_warn", type=float, default=0.06)
    p.add_argument("--extrap_warn_ratio", type=float, default=1.25)
    p.add_argument("--extrap_severe_ratio", type=float, default=2.0)
    return p.parse_args()


def parse_seed_list(spec: str) -> List[int]:
    norm = spec.replace(" ", ",")
    toks = [t.strip() for t in norm.split(",") if t.strip()]
    seeds = [int(t) for t in toks]
    if len(seeds) != 5:
        raise ValueError(f"Expected exactly 5 seeds, got {len(seeds)} from '{spec}'")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Seeds must be unique, got {seeds}")
    return seeds


def read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def to_float(x: object, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def finite_stats(vals: List[float]) -> Dict[str, float]:
    arr = np.asarray([float(v) for v in vals if np.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0.0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "n": float(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1) if arr.size > 1 else 0.0),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size == 0:
        return float("nan"), float("nan")
    if x.size < 2 or np.allclose(x, x[0]):
        return 0.0, float(np.mean(y))
    a, b = np.polyfit(x, y, deg=1)
    return float(a), float(b)


def loocv_mae(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    n = x.size
    if n < 3:
        return float("nan"), float("nan")
    errs: List[float] = []
    for i in range(n):
        mask = np.ones((n,), dtype=bool)
        mask[i] = False
        a, b = fit_line(x[mask], y[mask])
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        pred = a * x[i] + b
        errs.append(abs(float(pred - y[i])))
    if not errs:
        return float("nan"), float("nan")
    mae = float(np.mean(errs))
    rmse = float(np.sqrt(np.mean(np.square(errs))))
    return mae, rmse


def close_bool(v: float, thr: float) -> bool:
    return bool(np.isfinite(v) and abs(v) <= thr)


def main() -> None:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    results_root = args.results_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (results_root / f"{args.run_basename}_aspen_sanity_5seeds").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows: List[Dict[str, object]] = []
    corr_diff_rows: List[Dict[str, object]] = []
    loocv_rows: List[Dict[str, object]] = []
    extrap_rows: List[Dict[str, object]] = []
    pred_rows: List[Dict[str, object]] = []

    for seed in seeds:
        model_run_name = f"{args.run_basename}_seed{seed}"
        aspen_run_name = f"{model_run_name}{args.aspen_suffix}"
        model_run_dir = results_root / model_run_name
        aspen_run_dir = results_root / aspen_run_name

        required_model = [
            "clean_metrics.json",
            "corruption_metrics.csv",
        ]
        required_aspen = [
            "summary.json",
            "clean_reference.json",
            "jetclass_calibration_points.csv",
            "metric_to_deltaacc_mapping.csv",
            "aspen_shift_metrics.json",
            "aspen_predicted_deltaacc_by_metric.csv",
        ]
        missing = []
        for fn in required_model:
            if not (model_run_dir / fn).is_file():
                missing.append(f"model:{fn}")
        for fn in required_aspen:
            if not (aspen_run_dir / fn).is_file():
                missing.append(f"aspen:{fn}")
        if missing:
            per_seed_rows.append(
                {
                    "seed": seed,
                    "model_run_name": model_run_name,
                    "aspen_run_name": aspen_run_name,
                    "status": "missing_files",
                    "missing_files": "|".join(missing),
                }
            )
            continue

        clean_metrics = read_json(model_run_dir / "clean_metrics.json")
        clean_ref = read_json(aspen_run_dir / "clean_reference.json")
        aspen_summary = read_json(aspen_run_dir / "summary.json")
        aspen_metrics = read_json(aspen_run_dir / "aspen_shift_metrics.json")
        mapping_rows = read_csv_rows(aspen_run_dir / "metric_to_deltaacc_mapping.csv")
        calib_rows = read_csv_rows(aspen_run_dir / "jetclass_calibration_points.csv")
        pred_metric_rows = read_csv_rows(aspen_run_dir / "aspen_predicted_deltaacc_by_metric.csv")
        orig_corr_rows = read_csv_rows(model_run_dir / "corruption_metrics.csv")

                                  
                                
                                  
        clean_acc_orig = to_float(clean_metrics.get("acc"))
        clean_auc_orig = to_float(clean_metrics.get("auc_macro_ovr"))
        clean_acc_replay = to_float(clean_ref.get("clean_acc"))
        clean_auc_replay = to_float(clean_ref.get("clean_auc_macro_ovr"))
        clean_acc_diff = clean_acc_replay - clean_acc_orig
        clean_auc_diff = clean_auc_replay - clean_auc_orig
        clean_acc_ok = close_bool(clean_acc_diff, args.clean_acc_tol)
        clean_auc_ok = close_bool(clean_auc_diff, args.clean_auc_tol)

                                              
                                          
                                              
                                              
        orig_map: Dict[Tuple[str, str], Dict[str, str]] = {}
        for r in orig_corr_rows:
            k = (str(r["corruption_kind"]), str(r["severity"]))
            orig_map[k] = r
        calib_map: Dict[Tuple[str, str], Dict[str, str]] = {}
        for r in calib_rows:
            k = (str(r["corruption_kind"]), str(r["severity"]))
            calib_map[k] = r

        replay_diffs: Dict[str, List[float]] = {m: [] for m in SHARED_REPLAY_METRICS}
        keys_common = sorted(set(orig_map.keys()) & set(calib_map.keys()))
        for key in keys_common:
            ro = orig_map[key]
            rc = calib_map[key]
            for m in SHARED_REPLAY_METRICS:
                vo = to_float(ro.get(m))
                vc = to_float(rc.get(m))
                d = abs(vc - vo) if np.isfinite(vo) and np.isfinite(vc) else float("nan")
                replay_diffs[m].append(d)
                corr_diff_rows.append(
                    {
                        "seed": seed,
                        "corruption_kind": key[0],
                        "severity": key[1],
                        "metric": m,
                        "original_value": vo,
                        "replay_value": vc,
                        "abs_diff": d,
                    }
                )
        replay_max_diff = {m: (max(v) if v else float("nan")) for m, v in replay_diffs.items()}
        replay_ok = True
        for m in SHARED_REPLAY_METRICS:
            if not (np.isfinite(replay_max_diff[m]) and replay_max_diff[m] <= args.corruption_tol):
                replay_ok = False
                break

                                  
                                
                                  
        mapping_by_metric: Dict[str, Dict[str, str]] = {str(r["metric"]): r for r in mapping_rows}
        strong_r2_ok = True
        strong_intercept_ok = True
        for m in STRONG_METRICS:
            r = mapping_by_metric.get(m)
            if r is None:
                strong_r2_ok = False
                strong_intercept_ok = False
                continue
            r2 = to_float(r.get("r2"))
            intercept = to_float(r.get("intercept"))
            if not (np.isfinite(r2) and r2 >= args.mapping_r2_min_strong):
                strong_r2_ok = False
            if not (np.isfinite(intercept) and abs(intercept) <= args.mapping_intercept_tol):
                strong_intercept_ok = False

                                      
        y = np.asarray([to_float(r.get("delta_acc")) for r in calib_rows], dtype=np.float64)
        loocv_metric_mae: Dict[str, float] = {}
        for m in SHIFT_METRICS:
            x = np.asarray([to_float(r.get(m)) for r in calib_rows], dtype=np.float64)
            mae, rmse = loocv_mae(x, y)
            loocv_metric_mae[m] = mae
            loocv_rows.append(
                {
                    "seed": seed,
                    "metric": m,
                    "loocv_mae": mae,
                    "loocv_rmse": rmse,
                }
            )
        strong_loocv_ok = True
        for m in STRONG_METRICS:
            mae = loocv_metric_mae.get(m, float("nan"))
            if not (np.isfinite(mae) and mae <= args.loocv_mae_warn):
                strong_loocv_ok = False

                                  
                            
                                  
        aspen_stats = aspen_metrics.get("aspen_stats", {})
        class_dist = np.asarray(aspen_stats.get("class_dist", []), dtype=np.float64)
        top1_hist = np.asarray(aspen_stats.get("top1_hist", []), dtype=np.float64)
        mean_conf = to_float(aspen_stats.get("mean_confidence"))
        mean_ent = to_float(aspen_stats.get("mean_entropy"))
        n_aspen = int(to_float(aspen_metrics.get("n_jets_used"), default=0))

        aspen_finite_ok = bool(
            class_dist.size > 0
            and top1_hist.size > 0
            and np.isfinite(class_dist).all()
            and np.isfinite(top1_hist).all()
            and np.isfinite(mean_conf)
            and np.isfinite(mean_ent)
        )
        aspen_norm_ok = bool(
            np.isfinite(class_dist.sum())
            and np.isfinite(top1_hist.sum())
            and abs(float(class_dist.sum()) - 1.0) <= 1e-6
            and abs(float(top1_hist.sum()) - 1.0) <= 1e-6
        )
        aspen_size_ok = n_aspen >= 1_000_000
        top1_max_share = float(np.max(top1_hist)) if top1_hist.size else float("nan")

                                  
                                 
                                  
        aspen_shift_metrics = aspen_summary.get("aspen_shift_metrics", {})
        severe_extrap_count = 0
        warn_extrap_count = 0
        for m in SHIFT_METRICS:
            calib_vals = np.asarray([to_float(r.get(m)) for r in calib_rows], dtype=np.float64)
            calib_vals = calib_vals[np.isfinite(calib_vals)]
            if calib_vals.size == 0:
                cmin = cmax = float("nan")
                ratio = float("nan")
            else:
                cmin = float(np.min(calib_vals))
                cmax = float(np.max(calib_vals))
                a = to_float(aspen_shift_metrics.get(m))
                ratio = float(a / cmax) if np.isfinite(a) and cmax > 0 else float("nan")
                if np.isfinite(ratio):
                    if ratio > args.extrap_severe_ratio:
                        severe_extrap_count += 1
                    elif ratio > args.extrap_warn_ratio:
                        warn_extrap_count += 1
            extrap_rows.append(
                {
                    "seed": seed,
                    "metric": m,
                    "aspen_value": to_float(aspen_shift_metrics.get(m)),
                    "calib_min": cmin,
                    "calib_max": cmax,
                    "aspen_over_calibmax_ratio": ratio,
                }
            )

                                  
                                     
                                  
        pred_by_metric = {str(r["metric"]): r for r in pred_metric_rows}
        clipped_count = 0
        strong_preds: List[float] = []
        for m, r in pred_by_metric.items():
            raw = to_float(r.get("predicted_delta_acc_raw"))
            clipped = to_float(r.get("predicted_delta_acc_clipped"))
            if np.isfinite(raw) and np.isfinite(clipped) and abs(raw - clipped) > 1e-9:
                clipped_count += 1
            if m in STRONG_METRICS and np.isfinite(clipped):
                strong_preds.append(clipped)
        strong_mean_pred = float(np.mean(strong_preds)) if strong_preds else float("nan")
        strong_expected_acc = float(np.clip(clean_acc_orig - strong_mean_pred, 0.0, 1.0)) if np.isfinite(strong_mean_pred) else float("nan")

        ensemble_pred_delta = to_float(aspen_summary.get("ensemble_predicted_delta_acc"))
        ensemble_pred_acc = to_float(aspen_summary.get("ensemble_predicted_expected_acc"))

        pred_rows.append(
            {
                "seed": seed,
                "ensemble_predicted_delta_acc": ensemble_pred_delta,
                "ensemble_predicted_expected_acc": ensemble_pred_acc,
                "strong_metric_mean_predicted_delta_acc": strong_mean_pred,
                "strong_metric_mean_expected_acc": strong_expected_acc,
                "num_metrics_clipped": clipped_count,
            }
        )

        status = "ok"
        if not all([clean_acc_ok, clean_auc_ok, replay_ok, aspen_finite_ok, aspen_norm_ok, aspen_size_ok]):
            status = "warning"

        per_seed_rows.append(
            {
                "seed": seed,
                "model_run_name": model_run_name,
                "aspen_run_name": aspen_run_name,
                "status": status,
                "missing_files": "",
                "clean_acc_orig": clean_acc_orig,
                "clean_acc_replay": clean_acc_replay,
                "clean_acc_diff": clean_acc_diff,
                "clean_acc_ok": clean_acc_ok,
                "clean_auc_orig": clean_auc_orig,
                "clean_auc_replay": clean_auc_replay,
                "clean_auc_diff": clean_auc_diff,
                "clean_auc_ok": clean_auc_ok,
                "corruption_replay_ok": replay_ok,
                "corruption_replay_common_rows": len(keys_common),
                "replay_max_abs_diff_delta_acc": replay_max_diff["delta_acc"],
                "replay_max_abs_diff_class_js_divergence": replay_max_diff["class_js_divergence"],
                "replay_max_abs_diff_confidence_drop": replay_max_diff["confidence_drop"],
                "replay_max_abs_diff_entropy_shift": replay_max_diff["entropy_shift"],
                "mapping_strong_r2_ok": strong_r2_ok,
                "mapping_strong_intercept_ok": strong_intercept_ok,
                "mapping_strong_loocv_ok": strong_loocv_ok,
                "aspen_n_jets_used": n_aspen,
                "aspen_finite_ok": aspen_finite_ok,
                "aspen_norm_ok": aspen_norm_ok,
                "aspen_size_ok": aspen_size_ok,
                "aspen_top1_max_share": top1_max_share,
                "warn_extrap_metric_count": warn_extrap_count,
                "severe_extrap_metric_count": severe_extrap_count,
                "ensemble_predicted_delta_acc": ensemble_pred_delta,
                "ensemble_predicted_expected_acc": ensemble_pred_acc,
                "strong_metric_mean_predicted_delta_acc": strong_mean_pred,
                "strong_metric_mean_expected_acc": strong_expected_acc,
                "num_metrics_clipped": clipped_count,
            }
        )

                        
    if per_seed_rows:
        write_csv(
            output_dir / "per_seed_checks.csv",
            rows=per_seed_rows,
            fieldnames=list(per_seed_rows[0].keys()),
        )
    if corr_diff_rows:
        write_csv(
            output_dir / "corruption_replay_diffs.csv",
            rows=corr_diff_rows,
            fieldnames=list(corr_diff_rows[0].keys()),
        )
    if loocv_rows:
        write_csv(
            output_dir / "mapping_loocv_by_seed_metric.csv",
            rows=loocv_rows,
            fieldnames=list(loocv_rows[0].keys()),
        )
    if extrap_rows:
        write_csv(
            output_dir / "extrapolation_by_seed_metric.csv",
            rows=extrap_rows,
            fieldnames=list(extrap_rows[0].keys()),
        )
    if pred_rows:
        write_csv(
            output_dir / "predicted_shift_by_seed.csv",
            rows=pred_rows,
            fieldnames=list(pred_rows[0].keys()),
        )

                        
    ok_rows = [r for r in per_seed_rows if r.get("status") != "missing_files"]
    aggregate: Dict[str, object] = {
        "results_root": str(results_root),
        "output_dir": str(output_dir),
        "run_basename": args.run_basename,
        "seeds_requested": seeds,
        "runs_found": len(ok_rows),
        "runs_missing": len([r for r in per_seed_rows if r.get("status") == "missing_files"]),
        "missing_seeds": [int(r["seed"]) for r in per_seed_rows if r.get("status") == "missing_files"],
    }
    if ok_rows:
        aggregate["clean_acc_diff_stats"] = finite_stats([to_float(r["clean_acc_diff"]) for r in ok_rows])
        aggregate["clean_auc_diff_stats"] = finite_stats([to_float(r["clean_auc_diff"]) for r in ok_rows])
        aggregate["ensemble_pred_delta_stats"] = finite_stats([to_float(r["ensemble_predicted_delta_acc"]) for r in ok_rows])
        aggregate["ensemble_pred_acc_stats"] = finite_stats([to_float(r["ensemble_predicted_expected_acc"]) for r in ok_rows])
        aggregate["strong_pred_delta_stats"] = finite_stats([to_float(r["strong_metric_mean_predicted_delta_acc"]) for r in ok_rows])
        aggregate["strong_pred_acc_stats"] = finite_stats([to_float(r["strong_metric_mean_expected_acc"]) for r in ok_rows])
        aggregate["top1_max_share_stats"] = finite_stats([to_float(r["aspen_top1_max_share"]) for r in ok_rows])
        aggregate["num_metrics_clipped_stats"] = finite_stats([to_float(r["num_metrics_clipped"]) for r in ok_rows])

        aggregate["checks_pass_count"] = {
            "clean_acc_ok": int(sum(bool(r.get("clean_acc_ok")) for r in ok_rows)),
            "clean_auc_ok": int(sum(bool(r.get("clean_auc_ok")) for r in ok_rows)),
            "corruption_replay_ok": int(sum(bool(r.get("corruption_replay_ok")) for r in ok_rows)),
            "mapping_strong_r2_ok": int(sum(bool(r.get("mapping_strong_r2_ok")) for r in ok_rows)),
            "mapping_strong_intercept_ok": int(sum(bool(r.get("mapping_strong_intercept_ok")) for r in ok_rows)),
            "mapping_strong_loocv_ok": int(sum(bool(r.get("mapping_strong_loocv_ok")) for r in ok_rows)),
            "aspen_finite_ok": int(sum(bool(r.get("aspen_finite_ok")) for r in ok_rows)),
            "aspen_norm_ok": int(sum(bool(r.get("aspen_norm_ok")) for r in ok_rows)),
            "aspen_size_ok": int(sum(bool(r.get("aspen_size_ok")) for r in ok_rows)),
        }
        aggregate["extrapolation_counts"] = {
            "warn_total": int(sum(int(to_float(r.get("warn_extrap_metric_count"), 0.0)) for r in ok_rows)),
            "severe_total": int(sum(int(to_float(r.get("severe_extrap_metric_count"), 0.0)) for r in ok_rows)),
        }

    with (output_dir / "aggregate_summary.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, sort_keys=True)

                                     
    lines: List[str] = []
    lines.append("# Aspen Shift Sanity Checks (5 Seeds)")
    lines.append("")
    lines.append(f"- Results root: `{results_root}`")
    lines.append(f"- Output dir: `{output_dir}`")
    lines.append(f"- Runs found: `{aggregate.get('runs_found', 0)}` / 5")
    lines.append("")
    if ok_rows:
        cpass = aggregate.get("checks_pass_count", {})
        lines.append("## Pass Counts")
        lines.append("")
        for k, v in cpass.items():
            lines.append(f"- {k}: {v}/5")
        lines.append("")
        lines.append("## Key Aggregate Stats")
        lines.append("")
        for k in [
            "clean_acc_diff_stats",
            "clean_auc_diff_stats",
            "ensemble_pred_delta_stats",
            "ensemble_pred_acc_stats",
            "strong_pred_delta_stats",
            "strong_pred_acc_stats",
            "top1_max_share_stats",
            "num_metrics_clipped_stats",
        ]:
            st = aggregate.get(k, {})
            if not isinstance(st, dict):
                continue
            lines.append(
                f"- {k}: mean={st.get('mean')}, std={st.get('std')}, "
                f"min={st.get('min')}, max={st.get('max')}"
            )
        lines.append("")
        ex = aggregate.get("extrapolation_counts", {})
        lines.append(
            f"- Extrapolation warnings: warn_total={ex.get('warn_total')}, "
            f"severe_total={ex.get('severe_total')}"
        )
        lines.append("")
        lines.append("## Per-Seed Snapshot")
        lines.append("")
        for r in ok_rows:
            lines.append(
                f"- seed {r['seed']}: status={r['status']}, "
                f"clean_acc_diff={r['clean_acc_diff']:.6f}, "
                f"replay_ok={r['corruption_replay_ok']}, "
                f"severe_extrap_metrics={r['severe_extrap_metric_count']}, "
                f"ensemble_pred_acc={r['ensemble_predicted_expected_acc']:.6f}, "
                f"strong_pred_acc={r['strong_metric_mean_expected_acc']:.6f}"
            )
    else:
        lines.append("No complete runs found for sanity aggregation.")

    (output_dir / "sanity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=" * 72)
    print("Aspen sanity checks complete")
    print(f"Output dir: {output_dir}")
    print(f"Runs found: {aggregate.get('runs_found', 0)} / 5")
    print("Wrote:")
    print(f"  - {output_dir / 'per_seed_checks.csv'}")
    print(f"  - {output_dir / 'corruption_replay_diffs.csv'}")
    print(f"  - {output_dir / 'mapping_loocv_by_seed_metric.csv'}")
    print(f"  - {output_dir / 'extrapolation_by_seed_metric.csv'}")
    print(f"  - {output_dir / 'predicted_shift_by_seed.csv'}")
    print(f"  - {output_dir / 'aggregate_summary.json'}")
    print(f"  - {output_dir / 'sanity_report.md'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
