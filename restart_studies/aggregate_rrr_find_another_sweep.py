#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def parse_csv_list(text: str) -> List[str]:
    out = [x.strip() for x in text.replace(" ", ",").split(",") if x.strip()]
    if not out:
        raise ValueError(f"Empty list from: {text!r}")
    return out


def to_tag(x: str) -> str:
    return x.replace("-", "m").replace(".", "p").replace("+", "")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate RRR/find-another 48-config sweep outputs.")
    p.add_argument(
        "--results_root",
        type=Path,
        default=Path("/home/ryreu/atlas/CompPhys_FinalProject/restart_studies/results"),
    )
    p.add_argument("--run_basename", type=str, default="rrr_findanother_seed52")
    p.add_argument("--a_sources", type=str, default="input_grad,integrated_gradients,smoothgrad")
    p.add_argument("--lambda_values", type=str, default="1,10,100,1000")
    p.add_argument("--mask_fracs", type=str, default="0.05,0.10,0.20,0.30")
    p.add_argument("--output_run_name", type=str, default="")
    return p.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def float_or_nan(x: object) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def int_or_default(x: object, default: int = -1) -> int:
    v = float_or_nan(x)
    if np.isfinite(v):
        return int(v)
    return int(default)


def summarize_mean_std(vals: List[float]) -> Tuple[float, float]:
    arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0


def main() -> None:
    args = parse_args()

    a_sources = parse_csv_list(args.a_sources)
    lambda_values = parse_csv_list(args.lambda_values)
    mask_fracs = parse_csv_list(args.mask_fracs)

    out_name = args.output_run_name.strip() or f"{args.run_basename}_aggregate_48cfg"
    out_dir = (args.results_root / out_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, object]] = []
    iter_rows_all: List[Dict[str, object]] = []
    best_rows: List[Dict[str, object]] = []

    for src in a_sources:
        for lam in lambda_values:
            for mfrac in mask_fracs:
                run_name = f"{args.run_basename}_{src}_lam{to_tag(lam)}_mask{to_tag(mfrac)}"
                run_dir = (args.results_root / run_name).resolve()
                iter_csv = run_dir / "iteration_summary.csv"
                summary_json = run_dir / "summary.json"
                status = "missing"
                n_iters = 0
                best_iter = float("nan")
                best_aspen = float("nan")
                best_auc = float("nan")
                best_acc = float("nan")

                if run_dir.is_dir() and iter_csv.is_file() and summary_json.is_file():
                    status = "ok"
                    it_rows = read_csv(iter_csv)
                    n_iters = len(it_rows)
                    for r in it_rows:
                        iter_rows_all.append(
                            {
                                "run_name": run_name,
                                "a_source": src,
                                "lambda_rrr": float_or_nan(lam),
                                "mask_frac": float_or_nan(mfrac),
                                "iteration": int_or_default(r.get("iteration", "nan"), default=-1),
                                "iteration_seed": int_or_default(r.get("iteration_seed", "nan"), default=-1),
                                "val_acc": float_or_nan(r.get("val_acc", "nan")),
                                "val_auc_macro_ovr": float_or_nan(r.get("val_auc_macro_ovr", "nan")),
                                "test_acc": float_or_nan(r.get("test_acc", "nan")),
                                "test_auc_macro_ovr": float_or_nan(r.get("test_auc_macro_ovr", "nan")),
                                "aspen_prob_l1_drift": float_or_nan(r.get("aspen_prob_l1_drift", "nan")),
                                "aspen_top1_flip_rate": float_or_nan(r.get("aspen_top1_flip_rate", "nan")),
                                "aspen_class_js_divergence": float_or_nan(r.get("aspen_class_js_divergence", "nan")),
                                "aspen_confidence_drop": float_or_nan(r.get("aspen_confidence_drop", "nan")),
                                "aspen_entropy_shift": float_or_nan(r.get("aspen_entropy_shift", "nan")),
                                "aspen_strong3_mean": float_or_nan(r.get("aspen_strong3_mean", "nan")),
                                "aspen_n_jets_used": float_or_nan(r.get("aspen_n_jets_used", "nan")),
                                "a_total_feat_frac_of_validxdim": float_or_nan(
                                    r.get("a_total_feat_frac_of_validxdim", "nan")
                                ),
                            }
                        )
                    if it_rows:
                        sorted_rows = sorted(
                            it_rows,
                            key=lambda r: (
                                float_or_nan(r.get("aspen_strong3_mean", "nan")),
                                -float_or_nan(r.get("test_auc_macro_ovr", "nan")),
                            ),
                        )
                        br = sorted_rows[0]
                        best_iter = float_or_nan(br.get("iteration", "nan"))
                        best_aspen = float_or_nan(br.get("aspen_strong3_mean", "nan"))
                        best_auc = float_or_nan(br.get("test_auc_macro_ovr", "nan"))
                        best_acc = float_or_nan(br.get("test_acc", "nan"))
                        best_rows.append(
                            {
                                "run_name": run_name,
                                "a_source": src,
                                "lambda_rrr": float_or_nan(lam),
                                "mask_frac": float_or_nan(mfrac),
                                "best_iteration": int_or_default(best_iter, default=-1),
                                "best_aspen_strong3_mean": best_aspen,
                                "best_test_auc_macro_ovr": best_auc,
                                "best_test_acc": best_acc,
                            }
                        )

                manifest_rows.append(
                    {
                        "run_name": run_name,
                        "a_source": src,
                        "lambda_rrr": float_or_nan(lam),
                        "mask_frac": float_or_nan(mfrac),
                        "status": status,
                        "n_iterations_found": int(n_iters),
                        "best_iteration": int_or_default(best_iter, default=-1) if np.isfinite(best_iter) else "",
                        "best_aspen_strong3_mean": best_aspen,
                        "best_test_auc_macro_ovr": best_auc,
                        "best_test_acc": best_acc,
                        "path": str(run_dir),
                    }
                )

    manifest_rows_sorted = sorted(
        manifest_rows,
        key=lambda r: (
            str(r["a_source"]),
            float_or_nan(r["lambda_rrr"]),
            float_or_nan(r["mask_frac"]),
        ),
    )
    write_csv(
        out_dir / "config_manifest.csv",
        rows=manifest_rows_sorted,
        fieldnames=[
            "run_name",
            "a_source",
            "lambda_rrr",
            "mask_frac",
            "status",
            "n_iterations_found",
            "best_iteration",
            "best_aspen_strong3_mean",
            "best_test_auc_macro_ovr",
            "best_test_acc",
            "path",
        ],
    )

    if iter_rows_all:
        iter_rows_all_sorted = sorted(
            iter_rows_all,
            key=lambda r: (
                str(r["a_source"]),
                float_or_nan(r["lambda_rrr"]),
                float_or_nan(r["mask_frac"]),
                int(r["iteration"]),
            ),
        )
        write_csv(
            out_dir / "iteration_metrics_all.csv",
            rows=iter_rows_all_sorted,
            fieldnames=list(iter_rows_all_sorted[0].keys()),
        )

    best_rows_sorted = sorted(
        best_rows,
        key=lambda r: (float_or_nan(r["best_aspen_strong3_mean"]), -float_or_nan(r["best_test_auc_macro_ovr"])),
    )
    if best_rows_sorted:
        write_csv(
            out_dir / "best_iteration_per_config.csv",
            rows=best_rows_sorted,
            fieldnames=list(best_rows_sorted[0].keys()),
        )
        write_csv(
            out_dir / "ranking_best_by_aspen.csv",
            rows=best_rows_sorted,
            fieldnames=list(best_rows_sorted[0].keys()),
        )

    by_src_rows: List[Dict[str, object]] = []
    for src in a_sources:
        rows = [r for r in best_rows_sorted if str(r["a_source"]) == src]
        mean_aspen, std_aspen = summarize_mean_std([float_or_nan(r["best_aspen_strong3_mean"]) for r in rows])
        mean_auc, std_auc = summarize_mean_std([float_or_nan(r["best_test_auc_macro_ovr"]) for r in rows])
        mean_acc, std_acc = summarize_mean_std([float_or_nan(r["best_test_acc"]) for r in rows])
        by_src_rows.append(
            {
                "a_source": src,
                "n_configs": int(len(rows)),
                "best_aspen_strong3_mean_mean": mean_aspen,
                "best_aspen_strong3_mean_std": std_aspen,
                "best_test_auc_macro_ovr_mean": mean_auc,
                "best_test_auc_macro_ovr_std": std_auc,
                "best_test_acc_mean": mean_acc,
                "best_test_acc_std": std_acc,
            }
        )
    write_csv(
        out_dir / "aggregate_by_asource.csv",
        rows=by_src_rows,
        fieldnames=list(by_src_rows[0].keys()) if by_src_rows else ["a_source"],
    )

    n_expected = int(len(a_sources) * len(lambda_values) * len(mask_fracs))
    n_ok = int(sum(1 for r in manifest_rows_sorted if str(r["status"]) == "ok"))
    top1 = best_rows_sorted[0] if best_rows_sorted else None
    summary = {
        "run_basename": args.run_basename,
        "output_run_name": out_name,
        "results_root": str(args.results_root.resolve()),
        "n_expected_configs": n_expected,
        "n_completed_configs": n_ok,
        "a_sources": a_sources,
        "lambda_values": [float_or_nan(x) for x in lambda_values],
        "mask_fracs": [float_or_nan(x) for x in mask_fracs],
        "top_config_by_aspen_strong3": top1,
    }
    with (out_dir / "aggregate_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("========================================================================")
    print("RRR/find-another sweep aggregation complete")
    print(f"Output dir: {out_dir}")
    print(f"Configs found: {n_ok} / {n_expected}")
    print("Wrote:")
    print(f"  - {out_dir / 'config_manifest.csv'}")
    if iter_rows_all:
        print(f"  - {out_dir / 'iteration_metrics_all.csv'}")
    if best_rows_sorted:
        print(f"  - {out_dir / 'best_iteration_per_config.csv'}")
        print(f"  - {out_dir / 'ranking_best_by_aspen.csv'}")
    print(f"  - {out_dir / 'aggregate_by_asource.csv'}")
    print(f"  - {out_dir / 'aggregate_summary.json'}")
    print("========================================================================")


if __name__ == "__main__":
    main()
