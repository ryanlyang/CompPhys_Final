#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def finite_stats(values: List[float]) -> Dict[str, float]:
    vals = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    n = len(vals)
    mean = sum(vals) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in vals) / (n - 1)
        std = math.sqrt(max(var, 0.0))
    else:
        std = 0.0
    return {"n": n, "mean": mean, "std": std, "min": min(vals), "max": max(vals)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate 5-seed preliminary reimplementation outputs")
    parser.add_argument(
        "--results_root",
        type=Path,
        default=PROJECT_ROOT / "restart_studies" / "results",
        help="Directory containing per-seed run folders",
    )
    parser.add_argument("--run_basename", type=str, default="prelim_reimpl_cluster")
    parser.add_argument("--seeds", type=str, default="41,52,63,74,85")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Where to write aggregate outputs (default: <results_root>/<run_basename>_aggregate_5seeds)",
    )
    parser.add_argument("--strict", action="store_true", default=False, help="Fail if any expected seed run is missing")
    args = parser.parse_args()

    seeds = parse_seed_list(args.seeds)
    results_root = args.results_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (results_root / f"{args.run_basename}_aggregate_5seeds").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_files = [
        "summary.json",
        "clean_metrics.json",
        "correlations.csv",
        "method_effectiveness_summary.csv",
        "top_shift_metric_ranking.csv",
        "sanity_checks.json",
    ]

    clean_metric_fields = [
        "acc",
        "auc_macro_ovr",
        "signal_vs_bg_auc",
        "signal_vs_bg_fpr50",
        "target_vs_bg_ratio_auc",
        "target_vs_bg_ratio_fpr50",
        "mean_entropy",
        "mean_confidence",
        "best_epoch",
        "best_val_metric_seen",
        "val_metric_reloaded",
        "trainer_posthoc_metric_abs_diff",
    ]
    corr_fields = [
        "spearman_delta_auc",
        "pearson_delta_auc",
        "spearman_delta_acc",
        "pearson_delta_acc",
    ]
    method_fields = [
        "targeted_drop",
        "random_drop",
        "gap_target_minus_random",
        "auc_gap",
        "acc_gap",
    ]

    run_rows: List[Dict[str, object]] = []
    clean_rows: List[Dict[str, object]] = []
    corr_rows_long: List[Dict[str, object]] = []
    method_rows_long: List[Dict[str, object]] = []
    top1_counter: Counter[str] = Counter()

    missing_runs: List[str] = []

    for seed in seeds:
        run_name = f"{args.run_basename}_seed{seed}"
        run_dir = results_root / run_name
        missing_files = [fn for fn in expected_files if not (run_dir / fn).is_file()]

        status = "ok" if not missing_files else "missing_files"
        run_rows.append(
            {
                "seed": seed,
                "run_name": run_name,
                "run_dir": str(run_dir),
                "status": status,
                "missing_files": "|".join(missing_files),
            }
        )

        if missing_files:
            missing_runs.append(run_name)
            continue

        clean = read_json(run_dir / "clean_metrics.json")
        clean_row: Dict[str, object] = {"seed": seed, "run_name": run_name}
        for key in clean_metric_fields:
            val = clean.get(key, float("nan"))
            clean_row[key] = float(val) if isinstance(val, (int, float)) else float("nan")
        clean_rows.append(clean_row)

        for row in read_csv_rows(run_dir / "correlations.csv"):
            out: Dict[str, object] = {
                "seed": seed,
                "run_name": run_name,
                "metric": row["metric"],
            }
            for f in corr_fields:
                out[f] = float(row[f])
            corr_rows_long.append(out)

        for row in read_csv_rows(run_dir / "method_effectiveness_summary.csv"):
            out = {
                "seed": seed,
                "run_name": run_name,
                "method": row["method"],
            }
            for f in method_fields:
                out[f] = float(row[f])
            method_rows_long.append(out)

        ranking_rows = read_csv_rows(run_dir / "top_shift_metric_ranking.csv")
        for rr in ranking_rows:
            if rr.get("rank") == "1":
                top1_counter[rr.get("metric", "") or "<missing>"] += 1
                break

    if args.strict and missing_runs:
        raise SystemExit(f"Missing required runs/files for: {missing_runs}")

                                                     
    write_csv(
        output_dir / "seed_manifest.csv",
        rows=run_rows,
        fieldnames=["seed", "run_name", "run_dir", "status", "missing_files"],
    )

    if clean_rows:
        write_csv(
            output_dir / "clean_metrics_by_seed.csv",
            rows=clean_rows,
            fieldnames=["seed", "run_name", *clean_metric_fields],
        )

    if corr_rows_long:
        write_csv(
            output_dir / "shift_correlations_by_seed.csv",
            rows=corr_rows_long,
            fieldnames=["seed", "run_name", "metric", *corr_fields],
        )

    if method_rows_long:
        write_csv(
            output_dir / "method_effectiveness_by_seed.csv",
            rows=method_rows_long,
            fieldnames=["seed", "run_name", "method", *method_fields],
        )

                          
    clean_summary_rows: List[Dict[str, object]] = []
    for field in clean_metric_fields:
        vals = [float(r[field]) for r in clean_rows if field in r]
        st = finite_stats(vals)
        clean_summary_rows.append(
            {
                "metric": field,
                "n": st["n"],
                "mean": st["mean"],
                "std": st["std"],
                "min": st["min"],
                "max": st["max"],
            }
        )
    if clean_summary_rows:
        write_csv(
            output_dir / "clean_metrics_aggregate.csv",
            rows=clean_summary_rows,
            fieldnames=["metric", "n", "mean", "std", "min", "max"],
        )

    corr_grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in corr_rows_long:
        m = str(r["metric"])
        for f in corr_fields:
            corr_grouped[m][f].append(float(r[f]))

    corr_summary_rows: List[Dict[str, object]] = []
    for metric in sorted(corr_grouped.keys()):
        row: Dict[str, object] = {"metric": metric}
        for f in corr_fields:
            st = finite_stats(corr_grouped[metric][f])
            row[f"n_{f}"] = st["n"]
            row[f"mean_{f}"] = st["mean"]
            row[f"std_{f}"] = st["std"]
            row[f"min_{f}"] = st["min"]
            row[f"max_{f}"] = st["max"]
        corr_summary_rows.append(row)

    if corr_summary_rows:
        write_csv(
            output_dir / "shift_correlations_aggregate.csv",
            rows=corr_summary_rows,
            fieldnames=[
                "metric",
                *[f"n_{f}" for f in corr_fields],
                *[f"mean_{f}" for f in corr_fields],
                *[f"std_{f}" for f in corr_fields],
                *[f"min_{f}" for f in corr_fields],
                *[f"max_{f}" for f in corr_fields],
            ],
        )

    method_grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in method_rows_long:
        m = str(r["method"])
        for f in method_fields:
            method_grouped[m][f].append(float(r[f]))

    method_summary_rows: List[Dict[str, object]] = []
    for method in sorted(method_grouped.keys()):
        row = {"method": method}
        for f in method_fields:
            st = finite_stats(method_grouped[method][f])
            row[f"n_{f}"] = st["n"]
            row[f"mean_{f}"] = st["mean"]
            row[f"std_{f}"] = st["std"]
            row[f"min_{f}"] = st["min"]
            row[f"max_{f}"] = st["max"]
        method_summary_rows.append(row)

    if method_summary_rows:
        write_csv(
            output_dir / "method_effectiveness_aggregate.csv",
            rows=method_summary_rows,
            fieldnames=[
                "method",
                *[f"n_{f}" for f in method_fields],
                *[f"mean_{f}" for f in method_fields],
                *[f"std_{f}" for f in method_fields],
                *[f"min_{f}" for f in method_fields],
                *[f"max_{f}" for f in method_fields],
            ],
        )

    top_metric_rows = [
        {"metric": metric, "count_rank1": count}
        for metric, count in sorted(top1_counter.items(), key=lambda x: (-x[1], x[0]))
    ]
    if top_metric_rows:
        write_csv(
            output_dir / "top_shift_metric_rank1_frequency.csv",
            rows=top_metric_rows,
            fieldnames=["metric", "count_rank1"],
        )

    best_top_metric = top_metric_rows[0]["metric"] if top_metric_rows else None

    aggregate_summary = {
        "run_basename": args.run_basename,
        "seeds_requested": seeds,
        "results_root": str(results_root),
        "output_dir": str(output_dir),
        "runs_ok": int(sum(1 for r in run_rows if r["status"] == "ok")),
        "runs_missing": int(sum(1 for r in run_rows if r["status"] != "ok")),
        "missing_run_names": missing_runs,
        "top_rank1_metric_mode": best_top_metric,
    }

                                                                
    clean_lookup = {r["metric"]: r for r in clean_summary_rows}
    for key in ("acc", "auc_macro_ovr", "signal_vs_bg_auc", "target_vs_bg_ratio_auc"):
        if key in clean_lookup:
            aggregate_summary[f"{key}_mean"] = clean_lookup[key]["mean"]
            aggregate_summary[f"{key}_std"] = clean_lookup[key]["std"]

    with (output_dir / "aggregate_summary.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate_summary, f, indent=2, sort_keys=True)

    print("[done] wrote aggregate outputs:")
    print(f"  - {output_dir / 'aggregate_summary.json'}")
    print(f"  - {output_dir / 'seed_manifest.csv'}")
    if clean_rows:
        print(f"  - {output_dir / 'clean_metrics_by_seed.csv'}")
        print(f"  - {output_dir / 'clean_metrics_aggregate.csv'}")
    if corr_rows_long:
        print(f"  - {output_dir / 'shift_correlations_by_seed.csv'}")
        print(f"  - {output_dir / 'shift_correlations_aggregate.csv'}")
    if method_rows_long:
        print(f"  - {output_dir / 'method_effectiveness_by_seed.csv'}")
        print(f"  - {output_dir / 'method_effectiveness_aggregate.csv'}")
    if top_metric_rows:
        print(f"  - {output_dir / 'top_shift_metric_rank1_frequency.csv'}")


if __name__ == "__main__":
    main()
