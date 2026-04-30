#!/usr/bin/env python3
from __future__ import annotations


import argparse
import csv
import gc
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

try:
    import h5py
except ModuleNotFoundError:
    h5py = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

PRACTICETAGGING_ROOT = PROJECT_ROOT / "PracticeTagging"
if str(PRACTICETAGGING_ROOT) not in sys.path:
    sys.path.insert(0, str(PRACTICETAGGING_ROOT))

from evaluate_jetclass_hlt_teacher_baseline import (              
    IDX_CHARGE,
    IDX_D0,
    IDX_D0ERR,
    IDX_DZ,
    IDX_DZERR,
    IDX_E,
    IDX_ETA,
    IDX_PHI,
    IDX_PID0,
    IDX_PID1,
    IDX_PID2,
    IDX_PID3,
    IDX_PID4,
    IDX_PT,
    JetClassTransformer,
    collect_files_by_class,
    compute_features,
    get_mean_std,
    load_split,
    split_files_by_class,
    standardize,
)

from reimplement_preliminary_studies import (              
    apply_corruption_batch,
    evaluate_probs,
    jensen_shannon_divergence,
    mean_confidence,
    mean_entropy,
    parse_corruptions,
    safe_corr,
    set_seed,
)


SHIFT_METRICS = [
    "prob_l1_drift",
    "top1_flip_rate",
    "class_js_divergence",
    "confidence_drop",
    "entropy_shift",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-seed AspenOpenJets shift calibration/evaluation")
    p.add_argument(
        "--results_root",
        type=Path,
        default=Path("/home/ryreu/atlas/CompPhys_FinalProject/restart_studies/results"),
        help="Root directory containing per-seed preliminary run dirs.",
    )
    p.add_argument("--run_basename", type=str, default="prelim_reimpl_cluster")
    p.add_argument("--seed", type=int, default=52)
    p.add_argument(
        "--model_run_name",
        type=str,
        default="",
        help="Optional explicit run dir name (default: <run_basename>_seed<seed>).",
    )
    p.add_argument(
        "--jetclass_data_dir",
        type=Path,
        default=None,
        help="JetClass root dir override. If omitted, uses run config data_dir.",
    )
    p.add_argument(
        "--aspen_data_dir",
        type=Path,
        default=Path("/home/ryreu/atlas/CompPhys_Final/data/AspenOpenJets"),
    )
    p.add_argument("--aspen_glob", type=str, default="Run*.h5")
    p.add_argument("--aspen_n_jets", type=int, default=1_000_000)
    p.add_argument("--aspen_chunk_jets", type=int, default=50_000)

    p.add_argument(
        "--output_root",
        type=Path,
        default=PROJECT_ROOT / "restart_studies" / "results",
    )
    p.add_argument(
        "--output_run_name",
        type=str,
        default="",
        help="Optional explicit output run dir name.",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=-1, help="If <0, inherit from run config.")
    p.add_argument("--batch_size", type=int, default=-1, help="If <0, inherit from run config.")
    p.add_argument("--clip_delta_min", type=float, default=0.0)
    p.add_argument("--clip_delta_max", type=float, default=1.0)
    return p.parse_args()


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def probs_to_stats(probs: np.ndarray) -> Dict[str, np.ndarray | float]:
    probs = np.asarray(probs, dtype=np.float64)
    n = int(probs.shape[0])
    n_classes = int(probs.shape[1])
    pred = np.argmax(probs, axis=1)
    hist = np.bincount(pred, minlength=n_classes).astype(np.float64)
    hist = hist / max(1.0, float(n))
    class_dist = probs.mean(axis=0)
    class_dist = class_dist / max(class_dist.sum(), 1e-12)
    return {
        "class_dist": class_dist.astype(np.float64),
        "top1_hist": hist.astype(np.float64),
        "mean_confidence": float(mean_confidence(probs)),
        "mean_entropy": float(mean_entropy(probs)),
    }


def distributional_shift_metrics(
    clean_stats: Dict[str, np.ndarray | float],
    shift_stats: Dict[str, np.ndarray | float],
) -> Dict[str, float]:
    c_dist = np.asarray(clean_stats["class_dist"], dtype=np.float64)
    s_dist = np.asarray(shift_stats["class_dist"], dtype=np.float64)
    c_hist = np.asarray(clean_stats["top1_hist"], dtype=np.float64)
    s_hist = np.asarray(shift_stats["top1_hist"], dtype=np.float64)
    c_conf = float(clean_stats["mean_confidence"])
    s_conf = float(shift_stats["mean_confidence"])
    c_ent = float(clean_stats["mean_entropy"])
    s_ent = float(shift_stats["mean_entropy"])
    return {
                                                                                      
        "prob_l1_drift": float(np.abs(c_dist - s_dist).sum()),
                                                                                  
        "top1_flip_rate": float(0.5 * np.abs(c_hist - s_hist).sum()),
        "class_js_divergence": float(jensen_shannon_divergence(c_dist, s_dist)),
        "confidence_drop": float(c_conf - s_conf),
        "entropy_shift": float(s_ent - c_ent),
    }


def fit_linear_map(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size == 0:
        return {
            "n_points": 0.0,
            "slope": float("nan"),
            "intercept": float("nan"),
            "spearman_xy": float("nan"),
            "pearson_xy": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "r2": float("nan"),
        }
    if x.size < 2 or np.allclose(x, x[0]):
        slope = 0.0
        intercept = float(np.mean(y))
    else:
        slope, intercept = np.polyfit(x, y, deg=1)

    pred = slope * x + intercept
    mae = float(np.mean(np.abs(pred - y)))
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    var_y = float(np.var(y))
    if var_y <= 1e-12:
        r2 = float("nan")
    else:
        r2 = float(1.0 - np.mean((pred - y) ** 2) / var_y)
    sp, pr = safe_corr(x, y)
    return {
        "n_points": float(x.size),
        "slope": float(slope),
        "intercept": float(intercept),
        "spearman_xy": float(sp),
        "pearson_xy": float(pr),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def resolve_model_run(args: argparse.Namespace) -> Tuple[str, Path]:
    model_run_name = args.model_run_name.strip() if args.model_run_name else f"{args.run_basename}_seed{args.seed}"
    run_dir = (args.results_root / model_run_name).resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Run dir not found: {run_dir}")
    return model_run_name, run_dir


def resolve_output_run(args: argparse.Namespace, model_run_name: str) -> Tuple[str, Path]:
    run_name = args.output_run_name.strip() if args.output_run_name else f"{model_run_name}_aspen_shift_1M"
    run_dir = (args.output_root / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_name, run_dir


def sanitize_aoj_track_features(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float32).copy()
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32)


def aoj_pfcands_to_raw_tokens(pfcands: np.ndarray, max_constits: int) -> Tuple[np.ndarray, np.ndarray]:
                                                                                             
    arr = np.asarray(pfcands, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] < 10:
        raise ValueError(f"Expected PFCands shape [N,C,>=10], got {arr.shape}")

    n = int(arr.shape[0])
    c = min(int(arr.shape[1]), int(max_constits))
    arr = arr[:, :c, :]

    px = arr[:, :, 0]
    py = arr[:, :, 1]
    pz = arr[:, :, 2]
    ene = arr[:, :, 3]
    d0 = arr[:, :, 4]
    d0err = arr[:, :, 5]
    dz = arr[:, :, 6]
    dzerr = arr[:, :, 7]
    charge = arr[:, :, 8]
    pdgid = np.rint(arr[:, :, 9]).astype(np.int64)

    mask = (np.abs(px) + np.abs(py) + np.abs(pz) + np.abs(ene)) > 0.0

    pt = np.sqrt(np.maximum(px * px + py * py, 1e-12))
    p = np.sqrt(np.maximum(px * px + py * py + pz * pz, 1e-12))
    eta = 0.5 * np.log(np.clip((p + pz) / np.maximum(p - pz, 1e-8), 1e-8, 1e8))
    phi = np.arctan2(py, px)
    ene = np.maximum(ene, 1e-8)

    raw = np.zeros((n, c, 14), dtype=np.float32)
    raw[:, :, IDX_PT] = pt
    raw[:, :, IDX_ETA] = eta
    raw[:, :, IDX_PHI] = phi
    raw[:, :, IDX_E] = ene
    raw[:, :, IDX_CHARGE] = charge
    raw[:, :, IDX_D0] = sanitize_aoj_track_features(d0)
    raw[:, :, IDX_D0ERR] = sanitize_aoj_track_features(d0err)
    raw[:, :, IDX_DZ] = sanitize_aoj_track_features(dz)
    raw[:, :, IDX_DZERR] = sanitize_aoj_track_features(dzerr)

    abs_pid = np.abs(pdgid)
    is_ele = abs_pid == 11
    is_mu = abs_pid == 13
    is_pho = abs_pid == 22
    is_ch = (~is_ele) & (~is_mu) & (~is_pho) & (np.abs(charge) > 0.0)
    is_nh = (~is_ele) & (~is_mu) & (~is_pho) & (~is_ch)

    raw[:, :, IDX_PID0] = is_ch.astype(np.float32)
    raw[:, :, IDX_PID1] = is_nh.astype(np.float32)
    raw[:, :, IDX_PID2] = is_pho.astype(np.float32)
    raw[:, :, IDX_PID3] = is_ele.astype(np.float32)
    raw[:, :, IDX_PID4] = is_mu.astype(np.float32)

    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    raw[~mask] = 0.0
    return raw.astype(np.float32), mask.astype(bool)


def infer_probs_numpy(
    model: torch.nn.Module,
    feat: np.ndarray,
    mask: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    probs_parts: List[np.ndarray] = []
    with torch.no_grad():
        for s in range(0, feat.shape[0], batch_size):
            e = min(feat.shape[0], s + batch_size)
            x = torch.from_numpy(feat[s:e]).to(device, non_blocking=True)
            m = torch.from_numpy(mask[s:e]).to(device, non_blocking=True)
            logits = model(x, m)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs_parts.append(probs.astype(np.float64))
    return np.concatenate(probs_parts, axis=0)


def stream_aspen_stats(
    model: torch.nn.Module,
    aspen_data_dir: Path,
    glob_pattern: str,
    n_jets: int,
    chunk_jets: int,
    max_constits: int,
    feature_mode: str,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Dict[str, object]:
    files = sorted(p for p in aspen_data_dir.glob(glob_pattern) if p.is_file())
    if not files:
        raise RuntimeError(f"No Aspen files matched {aspen_data_dir}/{glob_pattern}")

    n_target = int(max(1, n_jets))
    n_seen = 0
    sum_probs: np.ndarray | None = None
    class_counts: np.ndarray | None = None
    sum_conf = 0.0
    sum_ent = 0.0
    file_rows: List[Dict[str, object]] = []

    for fp in files:
        if n_seen >= n_target:
            break
        with h5py.File(fp, "r") as f:
            if "PFCands" not in f:
                continue
            ds = f["PFCands"]
            n_file = int(ds.shape[0])
            read_file = 0
            start = 0
            while start < n_file and n_seen < n_target:
                take = int(min(chunk_jets, n_file - start, n_target - n_seen))
                arr = np.asarray(ds[start:start + take], dtype=np.float32)
                tok, mask = aoj_pfcands_to_raw_tokens(arr, max_constits=max_constits)
                feat = compute_features(tok, mask, feature_mode=feature_mode)
                feat = standardize(feat, mask, mean, std)

                probs = infer_probs_numpy(
                    model=model,
                    feat=feat,
                    mask=mask,
                    batch_size=batch_size,
                    device=device,
                )
                pred = np.argmax(probs, axis=1)
                cls_hist = np.bincount(pred, minlength=probs.shape[1]).astype(np.float64)

                if sum_probs is None:
                    sum_probs = np.zeros((probs.shape[1],), dtype=np.float64)
                    class_counts = np.zeros((probs.shape[1],), dtype=np.float64)

                sum_probs += probs.sum(axis=0)
                class_counts += cls_hist
                sum_conf += float(np.max(probs, axis=1).sum())
                p = np.clip(probs, 1e-12, 1.0)
                sum_ent += float((-(p * np.log(p)).sum(axis=1)).sum())

                n_chunk = int(probs.shape[0])
                n_seen += n_chunk
                read_file += n_chunk
                start += take

            file_rows.append(
                {
                    "file_name": fp.name,
                    "file": str(fp),
                    "n_jets_consumed": int(read_file),
                }
            )

    if n_seen <= 0 or sum_probs is None or class_counts is None:
        raise RuntimeError("No Aspen jets were processed.")

    mean_probs = sum_probs / float(n_seen)
    mean_probs = mean_probs / max(mean_probs.sum(), 1e-12)
    top1_hist = class_counts / float(n_seen)
    mean_conf = sum_conf / float(n_seen)
    mean_ent = sum_ent / float(n_seen)
    return {
        "n_jets_used": int(n_seen),
        "files_considered": len(files),
        "files_consumed": file_rows,
        "stats": {
            "class_dist": mean_probs.astype(np.float64),
            "top1_hist": top1_hist.astype(np.float64),
            "mean_confidence": float(mean_conf),
            "mean_entropy": float(mean_ent),
        },
    }


def main() -> None:
    args = parse_args()
    if h5py is None:
        raise SystemExit("Missing dependency: h5py. Install it in your runtime and rerun.")

    model_run_name, model_run_dir = resolve_model_run(args)
    output_run_name, output_run_dir = resolve_output_run(args, model_run_name)

    cfg_path = model_run_dir / "config.json"
    ckpt_path = model_run_dir / "clean_baseline_best.pt"
    if not cfg_path.is_file():
        raise SystemExit(f"Missing config: {cfg_path}")
    if not ckpt_path.is_file():
        raise SystemExit(f"Missing checkpoint: {ckpt_path}")

    cfg = read_json(cfg_path)
    seed = int(cfg["seed"])
    set_seed(seed)

    jetclass_data_dir = args.jetclass_data_dir.resolve() if args.jetclass_data_dir else Path(str(cfg["data_dir"])).resolve()
    if not jetclass_data_dir.is_dir():
        raise SystemExit(f"JetClass data dir not found: {jetclass_data_dir}")
    aspen_data_dir = args.aspen_data_dir.resolve()
    if not aspen_data_dir.is_dir():
        raise SystemExit(f"Aspen data dir not found: {aspen_data_dir}")

    batch_size = int(args.batch_size if args.batch_size > 0 else cfg["batch_size"])
    num_workers = int(args.num_workers if args.num_workers >= 0 else cfg["num_workers"])
    max_constits = int(cfg["max_constits"])
    feature_mode = str(cfg["feature_mode"])

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    print("[info] model_run_dir=", model_run_dir)
    print("[info] output_run_dir=", output_run_dir)
    print("[info] device=", device)
    print("[info] seed=", seed)
    print("[info] jetclass_data_dir=", jetclass_data_dir)
    print("[info] aspen_data_dir=", aspen_data_dir)

    files_by_class = collect_files_by_class(jetclass_data_dir)
    class_names = sorted(files_by_class.keys())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    tr_files, va_files, te_files = split_files_by_class(
        files_by_class,
        n_train=int(cfg["train_files_per_class"]),
        n_val=int(cfg["val_files_per_class"]),
        n_test=int(cfg["test_files_per_class"]),
        shuffle=bool(cfg["shuffle_files"]),
        seed=seed,
    )

    print("[info] loading JetClass train/test splits for calibration")
    tr_tok, tr_mask, tr_y = load_split(
        tr_files,
        n_total=int(cfg["n_train_jets"]),
        max_constits=max_constits,
        class_to_idx=class_to_idx,
        seed=seed + 101,
    )
    te_tok, te_mask, te_y = load_split(
        te_files,
        n_total=int(cfg["n_test_jets"]),
        max_constits=max_constits,
        class_to_idx=class_to_idx,
        seed=seed + 303,
    )
    print(f"[info] loaded JetClass jets train={len(tr_y)} test={len(te_y)}")

    tr_feat = compute_features(tr_tok, tr_mask, feature_mode=feature_mode)
    mean, std = get_mean_std(tr_feat, tr_mask, np.arange(len(tr_y)))
    del tr_feat, tr_tok, tr_mask, tr_y
    gc.collect()

    te_feat = compute_features(te_tok, te_mask, feature_mode=feature_mode)
    te_feat = standardize(te_feat, te_mask, mean, std)

    model = JetClassTransformer(
        input_dim=int(te_feat.shape[-1]),
        n_classes=int(len(class_names)),
        embed_dim=int(cfg["embed_dim"]),
        num_heads=int(cfg["num_heads"]),
        num_layers=int(cfg["num_layers"]),
        ff_dim=int(cfg["ff_dim"]),
        dropout=float(cfg["dropout"]),
    ).to(device)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    print("[info] evaluating clean JetClass reference")
    clean_pack = evaluate_probs(
        model=model,
        feat=te_feat,
        mask=te_mask,
        labels=te_y,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    clean_probs = np.asarray(clean_pack["probs"], dtype=np.float64)
    clean_stats = probs_to_stats(clean_probs)
    clean_acc = float(clean_pack["acc"])
    del te_feat
    gc.collect()

    print("[info] recomputing corruption calibration points")
    corruption_specs = parse_corruptions(str(cfg["corruptions"]))
    calib_rows: List[Dict[str, object]] = []
    for idx, (kind, sev) in enumerate(corruption_specs):
        rng = np.random.RandomState(seed + 5000 + idx * 97)
        c_tok, c_mask = apply_corruption_batch(te_tok, te_mask, kind=kind, severity=sev, rng=rng)
        c_feat = compute_features(c_tok, c_mask, feature_mode=feature_mode)
        c_feat = standardize(c_feat, c_mask, mean, std)
        c_pack = evaluate_probs(
            model=model,
            feat=c_feat,
            mask=c_mask,
            labels=te_y,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )
        c_probs = np.asarray(c_pack["probs"], dtype=np.float64)
        c_stats = probs_to_stats(c_probs)
        shifts = distributional_shift_metrics(clean_stats, c_stats)
        row = {
            "corruption_kind": kind,
            "severity": float(sev),
            "acc_corrupted": float(c_pack["acc"]),
            "delta_acc": float(clean_acc - float(c_pack["acc"])),
        }
        row.update({k: float(v) for k, v in shifts.items()})
        calib_rows.append(row)
        del c_tok, c_mask, c_feat, c_pack, c_probs
        gc.collect()

    write_csv(
        output_run_dir / "jetclass_calibration_points.csv",
        rows=calib_rows,
        fieldnames=["corruption_kind", "severity", "acc_corrupted", "delta_acc"] + SHIFT_METRICS,
    )
    del te_tok, te_mask, te_y
    gc.collect()

    mapping_rows: List[Dict[str, object]] = []
    mapping_dict: Dict[str, Dict[str, float]] = {}
    y = np.asarray([float(r["delta_acc"]) for r in calib_rows], dtype=np.float64)
    for m in SHIFT_METRICS:
        x = np.asarray([float(r[m]) for r in calib_rows], dtype=np.float64)
        fit = fit_linear_map(x, y)
        row = {"metric": m}
        row.update(fit)
        mapping_rows.append(row)
        mapping_dict[m] = {k: float(v) for k, v in fit.items()}
    write_csv(
        output_run_dir / "metric_to_deltaacc_mapping.csv",
        rows=mapping_rows,
        fieldnames=[
            "metric",
            "n_points",
            "slope",
            "intercept",
            "spearman_xy",
            "pearson_xy",
            "mae",
            "rmse",
            "r2",
        ],
    )

    print("[info] streaming Aspen inference")
    aspen = stream_aspen_stats(
        model=model,
        aspen_data_dir=aspen_data_dir,
        glob_pattern=args.aspen_glob,
        n_jets=int(args.aspen_n_jets),
        chunk_jets=int(args.aspen_chunk_jets),
        max_constits=max_constits,
        feature_mode=feature_mode,
        mean=mean,
        std=std,
        batch_size=batch_size,
        device=device,
    )
    aspen_stats = aspen["stats"]
    aspen_shifts = distributional_shift_metrics(clean_stats, aspen_stats)

    pred_rows: List[Dict[str, object]] = []
    pred_vals: List[float] = []
    pred_wts: List[float] = []
    for m in SHIFT_METRICS:
        x_val = float(aspen_shifts[m])
        fit = mapping_dict[m]
        slope = float(fit["slope"])
        intercept = float(fit["intercept"])
        raw_pred = slope * x_val + intercept
        clip_pred = float(np.clip(raw_pred, args.clip_delta_min, args.clip_delta_max))
        exp_acc = float(np.clip(clean_acc - clip_pred, 0.0, 1.0))
        wt = abs(float(fit["spearman_xy"])) if np.isfinite(float(fit["spearman_xy"])) else 0.0
        pred_vals.append(clip_pred)
        pred_wts.append(wt)
        pred_rows.append(
            {
                "metric": m,
                "aspen_metric_value": x_val,
                "slope": slope,
                "intercept": intercept,
                "predicted_delta_acc_raw": float(raw_pred),
                "predicted_delta_acc_clipped": clip_pred,
                "predicted_expected_acc": exp_acc,
                "mapping_weight_abs_spearman": wt,
            }
        )

    write_csv(
        output_run_dir / "aspen_predicted_deltaacc_by_metric.csv",
        rows=pred_rows,
        fieldnames=[
            "metric",
            "aspen_metric_value",
            "slope",
            "intercept",
            "predicted_delta_acc_raw",
            "predicted_delta_acc_clipped",
            "predicted_expected_acc",
            "mapping_weight_abs_spearman",
        ],
    )

    weights = np.asarray(pred_wts, dtype=np.float64)
    preds = np.asarray(pred_vals, dtype=np.float64)
    if np.all(weights <= 0):
        weights = np.ones_like(weights)
    ensemble_delta = float(np.sum(weights * preds) / max(np.sum(weights), 1e-12))
    ensemble_acc = float(np.clip(clean_acc - ensemble_delta, 0.0, 1.0))

    config_used = {
        "model_run_name": model_run_name,
        "model_run_dir": str(model_run_dir),
        "checkpoint_path": str(ckpt_path),
        "seed": seed,
        "jetclass_data_dir": str(jetclass_data_dir),
        "aspen_data_dir": str(aspen_data_dir),
        "aspen_glob": args.aspen_glob,
        "aspen_n_jets_target": int(args.aspen_n_jets),
        "aspen_chunk_jets": int(args.aspen_chunk_jets),
        "feature_mode": feature_mode,
        "max_constits": max_constits,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "device": str(device),
        "clip_delta_min": float(args.clip_delta_min),
        "clip_delta_max": float(args.clip_delta_max),
        "metric_definition_note": (
            "Shift metrics are distributional between clean JetClass reference and target set; "
            "top1_flip_rate/prob_l1_drift are distributional analogues."
        ),
    }
    with (output_run_dir / "config_used.json").open("w", encoding="utf-8") as f:
        json.dump(config_used, f, indent=2, sort_keys=True)

    clean_ref_json = {
        "clean_acc": clean_acc,
        "clean_auc_macro_ovr": float(clean_pack["auc_macro_ovr"]),
        "clean_stats": {
            "class_dist": [float(x) for x in np.asarray(clean_stats["class_dist"])],
            "top1_hist": [float(x) for x in np.asarray(clean_stats["top1_hist"])],
            "mean_confidence": float(clean_stats["mean_confidence"]),
            "mean_entropy": float(clean_stats["mean_entropy"]),
        },
    }
    with (output_run_dir / "clean_reference.json").open("w", encoding="utf-8") as f:
        json.dump(clean_ref_json, f, indent=2, sort_keys=True)

    aspen_json = {
        "n_jets_used": int(aspen["n_jets_used"]),
        "files_considered": int(aspen["files_considered"]),
        "files_consumed": aspen["files_consumed"],
        "aspen_stats": {
            "class_dist": [float(x) for x in np.asarray(aspen_stats["class_dist"])],
            "top1_hist": [float(x) for x in np.asarray(aspen_stats["top1_hist"])],
            "mean_confidence": float(aspen_stats["mean_confidence"]),
            "mean_entropy": float(aspen_stats["mean_entropy"]),
        },
        "aspen_shift_metrics": {k: float(v) for k, v in aspen_shifts.items()},
    }
    with (output_run_dir / "aspen_shift_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(aspen_json, f, indent=2, sort_keys=True)

    summary = {
        "model_run_name": model_run_name,
        "seed": seed,
        "clean_acc": clean_acc,
        "aspen_n_jets_used": int(aspen["n_jets_used"]),
        "aspen_shift_metrics": {k: float(v) for k, v in aspen_shifts.items()},
        "ensemble_predicted_delta_acc": ensemble_delta,
        "ensemble_predicted_expected_acc": ensemble_acc,
    }
    with (output_run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("[done] outputs:")
    print(" ", output_run_dir / "config_used.json")
    print(" ", output_run_dir / "clean_reference.json")
    print(" ", output_run_dir / "jetclass_calibration_points.csv")
    print(" ", output_run_dir / "metric_to_deltaacc_mapping.csv")
    print(" ", output_run_dir / "aspen_shift_metrics.json")
    print(" ", output_run_dir / "aspen_predicted_deltaacc_by_metric.csv")
    print(" ", output_run_dir / "summary.json")
    print(
        "[done] ensemble prediction: "
        f"delta_acc={ensemble_delta:.6f}, expected_acc={ensemble_acc:.6f}, "
        f"aspen_n_jets={int(aspen['n_jets_used'])}"
    )


if __name__ == "__main__":
    main()
