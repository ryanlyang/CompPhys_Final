#!/usr/bin/env python3

from __future__ import annotations

"""
Native (weaver-driven) corruption evaluation for JetClass checkpoints.

This script avoids the custom posthoc forward path by:
1) Building corrupted ROOT test files on disk.
2) Running weaver in predict mode with --load-model-weights.
3) Reading prediction ROOT outputs to compute metrics/correlations.
"""

import argparse
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import awkward as ak
except ModuleNotFoundError:
    ak = None

try:
    import uproot
except ModuleNotFoundError:
    uproot = None

from run_jetclass_part0_baseline_and_shift import (
    CLASS_NAMES,
    CLASS_TO_LABEL_INDEX,
    LABEL_NAMES,
    js_divergence,
    macro_auc,
    parse_index_spec,
    parse_float_list,
    pearson_corr,
    prepare_split_dirs,
    resolve_weaver_command,
    spearman_corr,
)


CORRUPTION_TYPES = ("gaussian_noise", "dropout", "masking", "eta_phi_jitter")


def write_csv(rows: List[Dict[str, float]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_probs(probs: np.ndarray) -> np.ndarray:
    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D probability array, got shape={arr.shape}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, None)
    denom = np.clip(arr.sum(axis=1, keepdims=True), 1e-12, None)
    arr = arr / denom
    return arr.astype(np.float32, copy=False)


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, n_classes: int) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    probs = normalize_probs(probs)
    y_pred = probs.argmax(axis=1)

    y_onehot = np.zeros((len(y_true), n_classes), dtype=np.int64)
    y_onehot[np.arange(len(y_true)), y_true] = 1

    acc = float((y_pred == y_true).mean())
    auc = float(macro_auc(y_onehot, probs))
    entropy = float((-probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=1).mean())
    conf = float(probs.max(axis=1).mean())
    return {
        "accuracy": acc,
        "macro_auc": auc,
        "mean_entropy": entropy,
        "mean_confidence": conf,
    }


def _tree_from_root(path: Path):
    f = uproot.open(path)
    for k in f.keys():
        obj = f[k]
        if isinstance(obj, uproot.behaviors.TTree.TTree):
            return obj
    raise RuntimeError(f"No TTree found in {path}")


def read_probs_from_pred_root(path: Path, n_classes: int) -> np.ndarray:
    tree = _tree_from_root(path)
    keys = set(tree.keys())

    if "softmax" in keys:
        arr = tree["softmax"].array(library="np")
        arr = np.asarray(arr)
        if arr.ndim == 2 and arr.shape[1] == n_classes:
            return normalize_probs(arr)

    score_branches = [f"score_{lbl}" for lbl in LABEL_NAMES]
    if all(k in keys for k in score_branches):
        arr = np.stack([tree[k].array(library="np") for k in score_branches], axis=1)
        return normalize_probs(arr)

    if "score" in keys:
        arr = np.asarray(tree["score"].array(library="np"))
        if arr.ndim == 2 and arr.shape[1] == n_classes:
            return normalize_probs(arr)

    candidates: List[str] = []
    for k in tree.keys():
        try:
            probe = np.asarray(tree[k].array(entry_stop=4, library="np"))
        except Exception:
            continue
        if probe.ndim == 2 and probe.shape[1] == n_classes:
            candidates.append(k)
    if candidates:
        arr = np.asarray(tree[candidates[0]].array(library="np"))
        return normalize_probs(arr)

    raise RuntimeError(f"Could not locate prediction probabilities in {path}")


def discover_pred_roots(base_pred_output: Path) -> List[Path]:
    if base_pred_output.exists():
        return [base_pred_output]
    stem = base_pred_output.with_suffix("").name
    roots = sorted(base_pred_output.parent.glob(f"{stem}*.root"))
    return roots


def build_jagged_bool_mask(
    counts: np.ndarray,
    rng: np.random.Generator,
    prob_true: float,
    force_at_least_one_true: bool = False,
) -> ak.Array:
    out: List[np.ndarray] = []
    p = float(np.clip(prob_true, 0.0, 1.0))
    for c in counts.tolist():
        n = int(c)
        if n <= 0:
            out.append(np.zeros((0,), dtype=np.bool_))
            continue
        row = (rng.random(n) < p)
        if force_at_least_one_true and not bool(row.any()):
            row[int(rng.integers(0, n))] = True
        out.append(row.astype(np.bool_))
    return ak.Array(out)


def build_jagged_normal(
    counts: np.ndarray,
    rng: np.random.Generator,
    sigma: float,
) -> ak.Array:
    out: List[np.ndarray] = []
    s = float(max(0.0, sigma))
    for c in counts.tolist():
        n = int(c)
        if n <= 0:
            out.append(np.zeros((0,), dtype=np.float32))
            continue
        row = rng.normal(0.0, s, size=n).astype(np.float32)
        out.append(row)
    return ak.Array(out)


def _all_part_fields(fields: Sequence[str]) -> List[str]:
    return [k for k in fields if k.startswith("part_")]


def corrupt_table(table: ak.Array, corruption: str, severity: float, rng: np.random.Generator) -> Dict[str, ak.Array]:
    out: Dict[str, ak.Array] = {k: table[k] for k in table.fields}
    part_fields = _all_part_fields(table.fields)
    counts = ak.to_numpy(ak.num(out["part_energy"], axis=1)).astype(np.int64, copy=False)

    if corruption == "dropout":
        keep = build_jagged_bool_mask(counts, rng, prob_true=(1.0 - severity), force_at_least_one_true=True)
        for k in part_fields:
            out[k] = out[k][keep]
        return out

    if corruption == "masking":
        drop = build_jagged_bool_mask(counts, rng, prob_true=severity, force_at_least_one_true=False)
        for k in part_fields:
            out[k] = ak.where(drop, 0, out[k])
        if "part_energy" in out:
            out["part_energy"] = ak.where(out["part_energy"] <= 1e-6, 1e-6, out["part_energy"])
        return out

    if corruption == "gaussian_noise":
        for k in ("part_px", "part_py", "part_pz"):
            if k in out:
                noise = build_jagged_normal(counts, rng, sigma=severity)
                out[k] = out[k] + noise

        if "part_energy" in out:
            eps = 1e-6
            mult = build_jagged_normal(counts, rng, sigma=severity)
            factor = ak.where((1.0 + mult) < 0.05, 0.05, 1.0 + mult)
            out["part_energy"] = out["part_energy"] * factor
            out["part_energy"] = ak.where(out["part_energy"] <= eps, eps, out["part_energy"])

        for k in ("part_deta", "part_dphi", "part_d0val", "part_dzval"):
            if k in out:
                noise = build_jagged_normal(counts, rng, sigma=severity)
                out[k] = out[k] + noise
        return out

    if corruption == "eta_phi_jitter":
        for k in ("part_deta", "part_dphi"):
            if k in out:
                noise = build_jagged_normal(counts, rng, sigma=severity)
                out[k] = out[k] + noise
        return out

    raise ValueError(f"Unsupported corruption type '{corruption}'")


def write_corrupted_test_files(
    src_files: Sequence[Path],
    dst_dir: Path,
    corruption: str,
    severity: float,
    seed: int,
) -> List[Path]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    out_files: List[Path] = []
    rng = np.random.default_rng(seed)

    for src in src_files:
        dst = dst_dir / src.name
        tree = uproot.open(src)["tree"]
        table = tree.arrays(library="ak")
        corr = corrupt_table(table, corruption=corruption, severity=severity, rng=rng)
        with uproot.recreate(dst) as fout:
            fout["tree"] = {k: corr[k] for k in table.fields}
        out_files.append(dst)
    return out_files


def run_weaver_predict(
    *,
    weaver_cmd: Sequence[str],
    split_paths: Dict[str, Path],
    test_files: Sequence[Path],
    data_config: Path,
    network_config: Path,
    checkpoint: Path,
    predict_output: Path,
    log_path: Path,
    gpus: str,
    batch_size: int,
    num_workers: int,
    fetch_step: float,
) -> None:
    cmd: List[str] = [*weaver_cmd, "--predict", "--data-train"]
    for cls in CLASS_NAMES:
        cmd.append(f"{cls}:{split_paths['train']}/{cls}_*.root")
    cmd.extend(["--data-val", f"{split_paths['val']}/*.root", "--data-test"])
    for f in test_files:
        cls = f.name.split("_", 1)[0]
        cmd.append(f"{cls}:{f}")

    model_prefix = predict_output.parent / "native_eval_model" / "net"
    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    predict_output.parent.mkdir(parents=True, exist_ok=True)

    cmd.extend(
        [
            "--data-config",
            str(data_config),
            "--network-config",
            str(network_config),
            "--model-prefix",
            str(model_prefix),
            "--load-model-weights",
            str(checkpoint),
            "--num-workers",
            str(num_workers),
            "--fetch-step",
            str(fetch_step),
            "--batch-size",
            str(batch_size),
            "--num-epochs",
            "1",
            "--samples-per-epoch",
            "1",
            "--samples-per-epoch-val",
            "1",
            "--gpus",
            gpus,
            "--predict-gpus",
            gpus,
            "--optimizer",
            "ranger",
            "--start-lr",
            "1e-3",
            "--log",
            str(log_path),
            "--predict-output",
            str(predict_output),
        ]
    )
    subprocess.run(cmd, check=True)


def load_pred_probs_and_truth(
    pred_base: Path,
    class_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    roots = discover_pred_roots(pred_base)
    if not roots:
        raise RuntimeError(f"No prediction ROOT files found for base output '{pred_base}'")

    by_cls: Dict[str, Path] = {}
    for p in roots:
        for cls in class_names:
            if p.name.endswith(f"_{cls}.root"):
                by_cls[cls] = p
                break

    if len(by_cls) == len(class_names):
        probs_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []
        for cls in class_names:
            probs_cls = read_probs_from_pred_root(by_cls[cls], n_classes=len(class_names))
            y_cls = np.full((len(probs_cls),), CLASS_TO_LABEL_INDEX[cls], dtype=np.int64)
            probs_list.append(probs_cls)
            y_list.append(y_cls)
        probs = np.concatenate(probs_list, axis=0)
        y_true = np.concatenate(y_list, axis=0)
        return probs, y_true

    # Fallback: single prediction file (already ordered by weaver data-test order).
    if len(roots) == 1:
        probs = read_probs_from_pred_root(roots[0], n_classes=len(class_names))
        raise RuntimeError(
            "Prediction output was not split per class; cannot infer y_true robustly in fallback mode. "
            "Use labeled data-test inputs (ClassName:file.root)."
        )

    raise RuntimeError(
        f"Could not map prediction files to all classes. Found={sorted([p.name for p in roots])}"
    )


def summarize_top_correlations(correlations: List[Dict[str, float]]) -> List[Dict[str, float]]:
    ranked = sorted(
        correlations,
        key=lambda x: (
            -abs(x.get("spearman_with_auc_drop", float("nan")))
            if not math.isnan(x.get("spearman_with_auc_drop", float("nan")))
            else float("-inf")
        ),
    )
    return ranked[:3]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Native weaver corruption evaluation for JetClass checkpoints.")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--feature-set", default="kinpid", choices=["kin", "kinpid", "full"])

    p.add_argument("--train-indices", default="0-7")
    p.add_argument("--val-indices", default="8")
    p.add_argument("--test-indices", default="9")

    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--fetch-step", type=float, default=0.01)
    p.add_argument("--gpus", default="0")

    p.add_argument("--noise-levels", default="0.01,0.03,0.05,0.1")
    p.add_argument("--dropout-levels", default="0.05,0.1,0.2,0.3")
    p.add_argument("--mask-levels", default="0.05,0.1,0.2,0.3")
    p.add_argument("--jitter-levels", default="0.01,0.03,0.05,0.1")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep-corrupted-files", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if ak is None or uproot is None:
        raise RuntimeError("Missing dependencies: awkward + uproot are required.")

    np.random.seed(args.seed)

    dataset_dir = Path(args.dataset_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    split_dir = output_dir / "splits"
    corr_files_root = output_dir / "corrupted_test_files"
    predict_root = output_dir / "predict_outputs"
    weaver_logs = output_dir / "weaver_logs"

    output_dir.mkdir(parents=True, exist_ok=True)
    predict_root.mkdir(parents=True, exist_ok=True)
    weaver_logs.mkdir(parents=True, exist_ok=True)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    split_paths = prepare_split_dirs(
        dataset_dir=dataset_dir,
        split_dir=split_dir,
        train_indices=parse_index_spec(args.train_indices),
        val_indices=parse_index_spec(args.val_indices),
        test_indices=parse_index_spec(args.test_indices),
    )
    test_files_clean = sorted(split_paths["test"].glob("*.root"))
    if not test_files_clean:
        raise FileNotFoundError(f"No test files found under {split_paths['test']}")

    data_config = Path(__file__).resolve().parent / "data" / "JetClass" / f"JetClass_{args.feature_set}.yaml"
    network_config = Path(__file__).resolve().parent / "networks" / "example_ParticleTransformer.py"
    if not data_config.exists():
        raise FileNotFoundError(f"Missing data config: {data_config}")
    if not network_config.exists():
        raise FileNotFoundError(f"Missing network config: {network_config}")

    weaver_cmd = resolve_weaver_command()

    # Clean baseline
    clean_pred_base = predict_root / "pred_clean.root"
    run_weaver_predict(
        weaver_cmd=weaver_cmd,
        split_paths=split_paths,
        test_files=test_files_clean,
        data_config=data_config,
        network_config=network_config,
        checkpoint=checkpoint,
        predict_output=clean_pred_base,
        log_path=weaver_logs / "clean.log",
        gpus=args.gpus,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        fetch_step=args.fetch_step,
    )
    clean_probs, clean_y = load_pred_probs_and_truth(clean_pred_base, class_names=CLASS_NAMES)
    clean_metrics = compute_metrics(clean_y, clean_probs, n_classes=len(CLASS_NAMES))
    clean_pred = clean_probs.argmax(axis=1)
    clean_class_dist = clean_probs.mean(axis=0)

    levels = {
        "gaussian_noise": parse_float_list(args.noise_levels),
        "dropout": parse_float_list(args.dropout_levels),
        "masking": parse_float_list(args.mask_levels),
        "eta_phi_jitter": parse_float_list(args.jitter_levels),
    }

    rows: List[Dict[str, float]] = []
    run_seed = int(args.seed)
    for corr in CORRUPTION_TYPES:
        for sev in levels[corr]:
            sev_tag = str(sev).replace(".", "p")
            corr_dir = corr_files_root / f"{corr}_{sev_tag}"
            corr_test_files = write_corrupted_test_files(
                src_files=test_files_clean,
                dst_dir=corr_dir,
                corruption=corr,
                severity=float(sev),
                seed=(run_seed + int(sev * 100000) + abs(hash(corr)) % 100000),
            )

            pred_base = predict_root / f"pred_{corr}_{sev_tag}.root"
            run_weaver_predict(
                weaver_cmd=weaver_cmd,
                split_paths=split_paths,
                test_files=corr_test_files,
                data_config=data_config,
                network_config=network_config,
                checkpoint=checkpoint,
                predict_output=pred_base,
                log_path=weaver_logs / f"{corr}_{sev_tag}.log",
                gpus=args.gpus,
                batch_size=args.eval_batch_size,
                num_workers=args.num_workers,
                fetch_step=args.fetch_step,
            )

            probs, y_true = load_pred_probs_and_truth(pred_base, class_names=CLASS_NAMES)
            if len(y_true) != len(clean_y):
                raise RuntimeError(
                    f"Prediction size mismatch for {corr}@{sev}: {len(y_true)} vs clean {len(clean_y)}"
                )
            if not np.array_equal(y_true, clean_y):
                # Keep going with clean ordering for drift metrics; supervised still valid.
                pass
            m = compute_metrics(y_true, probs, n_classes=len(CLASS_NAMES))
            pred = probs.argmax(axis=1)
            row = {
                "corruption": corr,
                "severity": float(sev),
                "accuracy": m["accuracy"],
                "macro_auc": m["macro_auc"],
                "auc_drop": clean_metrics["macro_auc"] - m["macro_auc"],
                "acc_drop": clean_metrics["accuracy"] - m["accuracy"],
                "mean_entropy": m["mean_entropy"],
                "entropy_shift": m["mean_entropy"] - clean_metrics["mean_entropy"],
                "mean_confidence": m["mean_confidence"],
                "confidence_drop": clean_metrics["mean_confidence"] - m["mean_confidence"],
                "prob_l1_drift": float(np.mean(np.sum(np.abs(probs - clean_probs), axis=1))),
                "top1_flip_rate": float(np.mean(pred != clean_pred)),
                "class_js_div": float(js_divergence(probs.mean(axis=0), clean_class_dist)),
            }
            rows.append(row)

    shift_metric_names = [
        "entropy_shift",
        "confidence_drop",
        "prob_l1_drift",
        "top1_flip_rate",
        "class_js_div",
    ]
    auc_drop = np.asarray([r["auc_drop"] for r in rows], dtype=np.float64)
    acc_drop = np.asarray([r["acc_drop"] for r in rows], dtype=np.float64)
    correlations: List[Dict[str, float]] = []
    for metric in shift_metric_names:
        vals = np.asarray([r[metric] for r in rows], dtype=np.float64)
        correlations.append(
            {
                "metric": metric,
                "pearson_with_auc_drop": pearson_corr(vals, auc_drop),
                "spearman_with_auc_drop": spearman_corr(vals, auc_drop),
                "pearson_with_acc_drop": pearson_corr(vals, acc_drop),
                "spearman_with_acc_drop": spearman_corr(vals, acc_drop),
            }
        )

    summary = {
        "checkpoint": str(checkpoint),
        "dataset_dir": str(dataset_dir),
        "feature_set": args.feature_set,
        "seed": args.seed,
        "clean_metrics": clean_metrics,
        "clean_pred_class_distribution": {
            cls: float(clean_class_dist[CLASS_TO_LABEL_INDEX[cls]]) for cls in CLASS_NAMES
        },
        "top_correlations_by_spearman_auc_drop": summarize_top_correlations(correlations),
        "num_corruption_points": len(rows),
        "weaver_logs_dir": str(weaver_logs),
        "predict_outputs_dir": str(predict_root),
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "corruption_metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    (output_dir / "correlations.json").write_text(json.dumps(correlations, indent=2) + "\n")
    write_csv(rows, output_dir / "corruption_metrics.csv")
    write_csv(correlations, output_dir / "correlations.csv")

    print("=== Clean test metrics (native weaver predict) ===")
    for k, v in clean_metrics.items():
        print(f"{k:>18s}: {v:.6f}")
    print("\n=== Top shift metrics by |Spearman with AUC drop| ===")
    for row in summarize_top_correlations(correlations):
        print(
            f"{row['metric']:>18s}: "
            f"spearman_auc={row['spearman_with_auc_drop']:.4f}, "
            f"pearson_auc={row['pearson_with_auc_drop']:.4f}, "
            f"spearman_acc={row['spearman_with_acc_drop']:.4f}"
        )
    print(f"\nWrote reports to: {output_dir}")

    if not args.keep_corrupted_files:
        shutil.rmtree(corr_files_root, ignore_errors=True)
        print("Removed temporary corrupted ROOT files.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

