#!/usr/bin/env python3

from __future__ import annotations

"""
Evaluate attribution-method effectiveness for a JetClass-trained ParT model.

This script:
1) Loads a trained Particle Transformer checkpoint.
2) Computes clean supervised metrics on a JetClass holdout split.
3) Computes attributions for multiple gradient-based methods.
4) Runs targeted-vs-random masking sanity checks.
5) Writes report tables for method comparison.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

from run_jetclass_part0_baseline_and_shift import (
    FILENAME_CLASS_NAMES_BY_LABEL_INDEX,
    InputArrays,
    LABEL_NAMES,
    best_permutation_accuracy,
    build_model,
    compute_supervised_metrics,
    js_divergence,
    load_split,
    parse_index_spec,
    predict_probs,
    prepare_split_dirs,
)


ATTR_METHODS = ("input_gradients", "integrated_gradients", "smoothgrad", "grad_x_input")

FEATURE_NAMES = {
    "kin": [
        "part_pt_log",
        "part_e_log",
        "part_logptrel",
        "part_logerel",
        "part_deltaR",
        "part_deta",
        "part_dphi",
    ],
    "kinpid": [
        "part_pt_log",
        "part_e_log",
        "part_logptrel",
        "part_logerel",
        "part_deltaR",
        "part_charge",
        "part_isChargedHadron",
        "part_isNeutralHadron",
        "part_isPhoton",
        "part_isElectron",
        "part_isMuon",
        "part_deta",
        "part_dphi",
    ],
    "full": [
        "part_pt_log",
        "part_e_log",
        "part_logptrel",
        "part_logerel",
        "part_deltaR",
        "part_charge",
        "part_isChargedHadron",
        "part_isNeutralHadron",
        "part_isPhoton",
        "part_isElectron",
        "part_isMuon",
        "part_d0",
        "part_d0err",
        "part_dz",
        "part_dzerr",
        "part_deta",
        "part_dphi",
    ],
}


def parse_float_list(spec: str) -> List[float]:
    vals = [float(x.strip()) for x in spec.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"Expected non-empty float list, got '{spec}'")
    return vals


def parse_method_list(spec: str) -> List[str]:
    methods = [x.strip() for x in spec.split(",") if x.strip()]
    if not methods:
        raise ValueError("Expected at least one attribution method.")
    invalid = [m for m in methods if m not in ATTR_METHODS]
    if invalid:
        raise ValueError(f"Unsupported methods: {invalid}. Supported: {ATTR_METHODS}")
    return methods


def write_csv(rows: List[Dict[str, float]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _forward_probs(
    model: torch.nn.Module,
    points: torch.Tensor,
    features: torch.Tensor,
    vectors: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    points = torch.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
    features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    vectors = torch.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)
    mask = torch.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    empty = (mask.sum(dim=2) <= 0).squeeze(1)
    if bool(empty.any().item()):
        mask = mask.clone()
        points = points.clone()
        features = features.clone()
        vectors = vectors.clone()
        idx = torch.where(empty)[0]
        vec_nonzero = (vectors[idx].abs().sum(dim=1) > 0).to(mask.dtype)
        mask[idx, 0, :] = vec_nonzero
        still_empty = (mask[idx].sum(dim=2) <= 0).squeeze(1)
        if bool(still_empty.any().item()):
            idx2 = idx[still_empty]
            mask[idx2, 0, 0] = 1.0
            points[idx2, :, 0] = 0.0
            features[idx2, :, 0] = 0.0
            vectors[idx2, :, 0] = 0.0
    out = model(points, features, vectors, mask)
    if not torch.isfinite(out).all():
        out = torch.nan_to_num(out, nan=0.0, posinf=50.0, neginf=-50.0)
    probs = torch.softmax(out, dim=1).clamp_min(1e-12)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return probs


def _gather_target_logprob(probs: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
    gathered = probs.gather(1, target_idx[:, None]).squeeze(1)
    return torch.log(gathered.clamp_min(1e-12))


def _safe_grad(score: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    grad = torch.autograd.grad(
        score,
        x,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )[0]
    if grad is None:
        grad = torch.zeros_like(x)
    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    return grad


def _tensor_batch(
    inputs: InputArrays,
    start: int,
    end: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    points = torch.from_numpy(inputs.points[start:end]).to(device)
    features = torch.from_numpy(inputs.features[start:end]).to(device)
    vectors = torch.from_numpy(inputs.vectors[start:end]).to(device)
    mask = torch.from_numpy(inputs.mask[start:end]).to(device)
    return points, features, vectors, mask


def subset_inputs_by_indices(inputs: InputArrays, indices: np.ndarray) -> InputArrays:
    idx = np.asarray(indices, dtype=np.int64)
    return InputArrays(
        points=inputs.points[idx],
        features=inputs.features[idx],
        vectors=inputs.vectors[idx],
        mask=inputs.mask[idx],
        y_onehot=inputs.y_onehot[idx],
        y_index=inputs.y_index[idx],
    )


def select_explain_indices(
    y_index: np.ndarray,
    n_select: int,
    rng: np.random.Generator,
    mode: str,
) -> np.ndarray:
    n_total = len(y_index)
    if n_select <= 0 or n_select >= n_total:
        return np.arange(n_total, dtype=np.int64)

    if mode == "head":
        return np.arange(n_select, dtype=np.int64)
    if mode == "random":
        idx = rng.choice(n_total, size=n_select, replace=False)
        idx.sort()
        return idx.astype(np.int64)
    if mode != "stratified":
        raise ValueError(f"Unsupported explain-sampling mode '{mode}'")

    classes, counts = np.unique(y_index, return_counts=True)
    probs = counts.astype(np.float64) / float(n_total)
    raw = probs * float(n_select)
    alloc = np.floor(raw).astype(np.int64)
    alloc = np.minimum(alloc, counts)
    remaining = int(n_select - alloc.sum())

    if remaining > 0:
        frac = raw - np.floor(raw)
        while remaining > 0:
            avail = np.where(counts - alloc > 0)[0]
            if avail.size == 0:
                break
            order = avail[np.argsort(frac[avail])[::-1]]
            progressed = False
            for j in order:
                if remaining <= 0:
                    break
                if alloc[j] < counts[j]:
                    alloc[j] += 1
                    remaining -= 1
                    progressed = True
            if not progressed:
                break

    picked: List[np.ndarray] = []
    for cls, k in zip(classes, alloc):
        if k <= 0:
            continue
        cls_idx = np.flatnonzero(y_index == cls)
        pick = rng.choice(cls_idx, size=int(k), replace=False)
        picked.append(pick.astype(np.int64))
    if not picked:
        idx = rng.choice(n_total, size=n_select, replace=False)
        idx.sort()
        return idx.astype(np.int64)

    out = np.concatenate(picked, axis=0)
    rng.shuffle(out)
    return out.astype(np.int64)


def _attr_input_gradients(
    model: torch.nn.Module,
    points: torch.Tensor,
    features: torch.Tensor,
    vectors: torch.Tensor,
    mask: torch.Tensor,
    target_idx: torch.Tensor,
) -> torch.Tensor:
    x = features.detach().clone().requires_grad_(True)
    probs = _forward_probs(model, points, x, vectors, mask)
    score = _gather_target_logprob(probs, target_idx).sum()
    grad = _safe_grad(score, x)
    return grad


def _attr_grad_x_input(
    model: torch.nn.Module,
    points: torch.Tensor,
    features: torch.Tensor,
    vectors: torch.Tensor,
    mask: torch.Tensor,
    target_idx: torch.Tensor,
) -> torch.Tensor:
    x = features.detach().clone().requires_grad_(True)
    probs = _forward_probs(model, points, x, vectors, mask)
    score = _gather_target_logprob(probs, target_idx).sum()
    grad = _safe_grad(score, x)
    return grad * x


def _attr_integrated_gradients(
    model: torch.nn.Module,
    points: torch.Tensor,
    features: torch.Tensor,
    vectors: torch.Tensor,
    mask: torch.Tensor,
    target_idx: torch.Tensor,
    ig_steps: int,
) -> torch.Tensor:
    x = features.detach()
    baseline = torch.zeros_like(x)
    delta = x - baseline
    total_grad = torch.zeros_like(x)
    for step in range(1, ig_steps + 1):
        alpha = float(step) / float(ig_steps)
        x_step = (baseline + alpha * delta).detach().requires_grad_(True)
        probs = _forward_probs(model, points, x_step, vectors, mask)
        score = _gather_target_logprob(probs, target_idx).sum()
        grad = _safe_grad(score, x_step)
        total_grad += grad
    avg_grad = total_grad / float(ig_steps)
    return delta * avg_grad


def _attr_smoothgrad(
    model: torch.nn.Module,
    points: torch.Tensor,
    features: torch.Tensor,
    vectors: torch.Tensor,
    mask: torch.Tensor,
    target_idx: torch.Tensor,
    smoothgrad_samples: int,
    smoothgrad_sigma: float,
) -> torch.Tensor:
    x = features.detach()
    total_abs_grad = torch.zeros_like(x)
    for _ in range(smoothgrad_samples):
        noise = torch.randn_like(x) * smoothgrad_sigma
        x_noisy = (x + noise).detach().requires_grad_(True)
        probs = _forward_probs(model, points, x_noisy, vectors, mask)
        score = _gather_target_logprob(probs, target_idx).sum()
        grad = _safe_grad(score, x_noisy)
        total_abs_grad += grad.abs()
    return total_abs_grad / float(smoothgrad_samples)


def compute_particle_importance(
    model: torch.nn.Module,
    inputs: InputArrays,
    target_idx: np.ndarray,
    method: str,
    batch_size: int,
    device: torch.device,
    ig_steps: int,
    smoothgrad_samples: int,
    smoothgrad_sigma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    feature_dim = inputs.features.shape[1]
    feature_abs_sum = np.zeros(feature_dim, dtype=np.float64)
    active_particle_count = 0.0
    per_particle_importance: List[np.ndarray] = []

    model.eval()
    n = len(inputs.y_index)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        points, features, vectors, mask = _tensor_batch(inputs, start, end, device)
        target_t = torch.from_numpy(target_idx[start:end]).to(device=device, dtype=torch.long)

        if method == "input_gradients":
            attr = _attr_input_gradients(model, points, features, vectors, mask, target_t)
        elif method == "grad_x_input":
            attr = _attr_grad_x_input(model, points, features, vectors, mask, target_t)
        elif method == "integrated_gradients":
            attr = _attr_integrated_gradients(
                model, points, features, vectors, mask, target_t, ig_steps=ig_steps
            )
        elif method == "smoothgrad":
            attr = _attr_smoothgrad(
                model,
                points,
                features,
                vectors,
                mask,
                target_t,
                smoothgrad_samples=smoothgrad_samples,
                smoothgrad_sigma=smoothgrad_sigma,
            )
        else:
            raise ValueError(f"Unknown method '{method}'")

        abs_attr = attr.abs() * mask
        importance = abs_attr.mean(dim=1)
        per_particle_importance.append(importance.detach().cpu().numpy().astype(np.float32))

        feature_abs_sum += abs_attr.detach().sum(dim=(0, 2)).cpu().numpy()
        active_particle_count += mask.detach().sum().item()

        del points, features, vectors, mask, target_t, attr, abs_attr, importance

    imp = np.concatenate(per_particle_importance, axis=0)
    feature_abs_mean = feature_abs_sum / max(active_particle_count, 1.0)
    return imp, feature_abs_mean


def _build_removal_indices(
    importance: np.ndarray,
    valid_mask: np.ndarray,
    fraction: float,
    mode: str,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    n, _ = importance.shape
    out: List[np.ndarray] = []
    for i in range(n):
        valid = np.flatnonzero(valid_mask[i] > 0)
        if valid.size == 0 or fraction <= 0:
            out.append(np.array([], dtype=np.int64))
            continue
        k = int(round(fraction * valid.size))
        k = max(1, min(k, valid.size))
        if mode == "targeted":
            scores = importance[i, valid]
            order = valid[np.argsort(scores)[::-1]]
            idx = order[:k]
        elif mode == "random":
            idx = rng.choice(valid, size=k, replace=False)
        else:
            raise ValueError(f"Unknown removal mode '{mode}'")
        out.append(np.asarray(idx, dtype=np.int64))
    return out


def _apply_removals(clean: InputArrays, removal_indices: List[np.ndarray]) -> InputArrays:
    points = clean.points.copy()
    features = clean.features.copy()
    vectors = clean.vectors.copy()
    mask = clean.mask.copy()
    for i, idx in enumerate(removal_indices):
        if idx.size == 0:
            continue
        points[i, :, idx] = 0.0
        features[i, :, idx] = 0.0
        vectors[i, :, idx] = 0.0
        mask[i, 0, idx] = 0.0
    return InputArrays(
        points=points,
        features=features,
        vectors=vectors,
        mask=mask,
        y_onehot=clean.y_onehot,
        y_index=clean.y_index,
    )


def evaluate_masking_condition(
    model: torch.nn.Module,
    clean_inputs: InputArrays,
    clean_probs: np.ndarray,
    clean_metrics: Dict[str, float],
    target_idx: np.ndarray,
    clean_pred: np.ndarray,
    importance: np.ndarray,
    fraction: float,
    mode: str,
    eval_batch_size: int,
    device: torch.device,
    rng: np.random.Generator,
) -> Dict[str, float]:
    valid_mask = (clean_inputs.mask[:, 0, :] > 0).astype(np.int32)
    removal = _build_removal_indices(
        importance=importance,
        valid_mask=valid_mask,
        fraction=fraction,
        mode=mode,
        rng=rng,
    )
    perturbed = _apply_removals(clean_inputs, removal)
    probs = predict_probs(model=model, inputs=perturbed, batch_size=eval_batch_size, device=device)
    metrics = compute_supervised_metrics(perturbed, probs)

    target_prob_clean = clean_probs[np.arange(len(target_idx)), target_idx]
    target_prob_pert = probs[np.arange(len(target_idx)), target_idx]
    pred_pert = probs.argmax(axis=1)

    return {
        "fraction": float(fraction),
        "mode": mode,
        "accuracy": float(metrics["accuracy"]),
        "macro_auc": float(metrics["macro_auc"]),
        "mean_entropy": float(metrics["mean_entropy"]),
        "mean_confidence": float(metrics["mean_confidence"]),
        "acc_drop": float(clean_metrics["accuracy"] - metrics["accuracy"]),
        "auc_drop": float(clean_metrics["macro_auc"] - metrics["macro_auc"]),
        "target_prob_drop": float(np.mean(target_prob_clean - target_prob_pert)),
        "top1_flip_rate": float(np.mean(clean_pred != pred_pert)),
        "class_js_div": float(js_divergence(clean_probs.mean(axis=0), probs.mean(axis=0))),
    }


def build_method_summary(
    method: str,
    rows: List[Dict[str, float]],
    fractions: Sequence[float],
) -> Dict[str, float]:
    targeted = [r for r in rows if r["method"] == method and r["mode"] == "targeted"]
    random_rows = [r for r in rows if r["method"] == method and r["mode"] == "random"]

    random_mean_by_fraction: Dict[float, Dict[str, float]] = {}
    for frac in fractions:
        frac_rows = [r for r in random_rows if math.isclose(r["fraction"], frac)]
        if not frac_rows:
            continue
        keys = [
            "acc_drop",
            "auc_drop",
            "target_prob_drop",
            "top1_flip_rate",
            "class_js_div",
        ]
        random_mean_by_fraction[float(frac)] = {
            k: float(np.mean([x[k] for x in frac_rows])) for k in keys
        }

    def _mean(vals: List[float]) -> float:
        if not vals:
            return float("nan")
        return float(np.mean(vals))

    target_target_prob = _mean([r["target_prob_drop"] for r in targeted])
    target_auc_drop = _mean([r["auc_drop"] for r in targeted])
    target_acc_drop = _mean([r["acc_drop"] for r in targeted])
    target_flip = _mean([r["top1_flip_rate"] for r in targeted])
    target_js = _mean([r["class_js_div"] for r in targeted])

    rand_target_prob = _mean([v["target_prob_drop"] for v in random_mean_by_fraction.values()])
    rand_auc_drop = _mean([v["auc_drop"] for v in random_mean_by_fraction.values()])
    rand_acc_drop = _mean([v["acc_drop"] for v in random_mean_by_fraction.values()])
    rand_flip = _mean([v["top1_flip_rate"] for v in random_mean_by_fraction.values()])
    rand_js = _mean([v["class_js_div"] for v in random_mean_by_fraction.values()])

    return {
        "method": method,
        "aopc_target_prob_drop_targeted": target_target_prob,
        "aopc_target_prob_drop_random": rand_target_prob,
        "aopc_target_prob_drop_gap": target_target_prob - rand_target_prob,
        "aopc_auc_drop_targeted": target_auc_drop,
        "aopc_auc_drop_random": rand_auc_drop,
        "aopc_auc_drop_gap": target_auc_drop - rand_auc_drop,
        "aopc_acc_drop_targeted": target_acc_drop,
        "aopc_acc_drop_random": rand_acc_drop,
        "aopc_acc_drop_gap": target_acc_drop - rand_acc_drop,
        "aopc_top1_flip_targeted": target_flip,
        "aopc_top1_flip_random": rand_flip,
        "aopc_top1_flip_gap": target_flip - rand_flip,
        "aopc_class_js_targeted": target_js,
        "aopc_class_js_random": rand_js,
        "aopc_class_js_gap": target_js - rand_js,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark attribution-method faithfulness for JetClass ParT."
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to saved_model.pt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-set", default="kinpid", choices=["kin", "kinpid", "full"])
    parser.add_argument(
        "--label-source",
        default="filename",
        choices=["filename", "branch"],
        help=(
            "How to derive class labels for evaluation. "
            "`filename` matches the class indexing used by the training command in this project."
        ),
    )
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--max-num-particles", type=int, default=128)
    parser.add_argument("--test-indices", default="9")
    parser.add_argument("--train-indices", default="0-7")
    parser.add_argument("--val-indices", default="8")
    parser.add_argument("--max-eval-jets", type=int, default=120000)
    parser.add_argument("--max-explain-jets", type=int, default=20000)
    parser.add_argument(
        "--explain-sampling",
        default="stratified",
        choices=["stratified", "random", "head"],
        help="How to pick the explain subset from eval jets.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--attr-batch-size", type=int, default=96)

    parser.add_argument(
        "--methods",
        default="input_gradients,integrated_gradients,smoothgrad",
        help="Comma-separated subset of: input_gradients,integrated_gradients,smoothgrad,grad_x_input",
    )
    parser.add_argument("--target-mode", default="pred", choices=["pred", "true"])
    parser.add_argument("--mask-fractions", default="0.02,0.05,0.1,0.2")
    parser.add_argument("--random-repeats", type=int, default=3)
    parser.add_argument("--ig-steps", type=int, default=16)
    parser.add_argument("--smoothgrad-samples", type=int, default=16)
    parser.add_argument("--smoothgrad-sigma", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if torch is None:
        raise RuntimeError("Missing dependency: torch")

    methods = parse_method_list(args.methods)
    mask_fractions = parse_float_list(args.mask_fractions)
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_dir = Path(args.dataset_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    split_dir = output_dir / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)

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

    test_files = sorted(split_paths["test"].glob("*.root"))
    if not test_files:
        raise FileNotFoundError(f"No test files found under {split_paths['test']}")
    print(f"Loading evaluation split from {len(test_files)} files...")
    eval_inputs = load_split(
        files=test_files,
        feature_set=args.feature_set,
        max_num_particles=args.max_num_particles,
        max_jets=args.max_eval_jets,
        label_source=args.label_source,
    )
    print(f"Loaded {len(eval_inputs.y_index)} evaluation jets.")
    class_names = (
        FILENAME_CLASS_NAMES_BY_LABEL_INDEX if args.label_source == "filename" else LABEL_NAMES
    )
    n_classes = len(class_names)

    explain_n = len(eval_inputs.y_index)
    if args.max_explain_jets > 0:
        explain_n = min(explain_n, args.max_explain_jets)
    explain_idx = select_explain_indices(
        y_index=eval_inputs.y_index,
        n_select=explain_n,
        rng=rng,
        mode=args.explain_sampling,
    )
    explain_inputs = subset_inputs_by_indices(eval_inputs, explain_idx)
    print(f"Using {len(explain_inputs.y_index)} jets for attribution benchmarking.")
    uniq_e, cnt_e = np.unique(explain_inputs.y_index, return_counts=True)
    explain_label_dist = {class_names[int(k)]: int(v) for k, v in zip(uniq_e, cnt_e)}
    print(f"Explain subset label distribution: {explain_label_dist}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and args.device != "cpu":
        print("Warning: CUDA not available, using CPU.")

    model = build_model(
        feature_set=args.feature_set,
        num_classes=n_classes,
        checkpoint_path=checkpoint,
        device=device,
    )

    clean_probs_eval = predict_probs(model, eval_inputs, args.eval_batch_size, device)
    clean_metrics_eval = compute_supervised_metrics(eval_inputs, clean_probs_eval)
    clean_perm_eval = best_permutation_accuracy(
        y_true=eval_inputs.y_index,
        y_pred=clean_probs_eval.argmax(axis=1),
        n_classes=n_classes,
    )
    clean_probs = clean_probs_eval[:explain_n]
    clean_metrics = compute_supervised_metrics(explain_inputs, clean_probs)
    clean_perm_explain = best_permutation_accuracy(
        y_true=explain_inputs.y_index,
        y_pred=clean_probs.argmax(axis=1),
        n_classes=n_classes,
    )
    clean_pred = clean_probs.argmax(axis=1)

    if args.target_mode == "pred":
        target_idx = clean_pred.copy()
    else:
        target_idx = explain_inputs.y_index.copy()

    perturbation_rows: List[Dict[str, float]] = []
    feature_rows: List[Dict[str, float]] = []
    method_summary_rows: List[Dict[str, float]] = []

    feature_names = FEATURE_NAMES[args.feature_set]

    for method in methods:
        print(f"\nComputing attributions for method: {method}")
        importance, feature_abs_mean = compute_particle_importance(
            model=model,
            inputs=explain_inputs,
            target_idx=target_idx,
            method=method,
            batch_size=args.attr_batch_size,
            device=device,
            ig_steps=args.ig_steps,
            smoothgrad_samples=args.smoothgrad_samples,
            smoothgrad_sigma=args.smoothgrad_sigma,
        )

        for i, val in enumerate(feature_abs_mean.tolist()):
            feature_rows.append(
                {
                    "method": method,
                    "feature_index": i,
                    "feature_name": feature_names[i],
                    "mean_abs_attribution": float(val),
                }
            )

        for frac in mask_fractions:
            row_t = evaluate_masking_condition(
                model=model,
                clean_inputs=explain_inputs,
                clean_probs=clean_probs,
                clean_metrics=clean_metrics,
                target_idx=target_idx,
                clean_pred=clean_pred,
                importance=importance,
                fraction=frac,
                mode="targeted",
                eval_batch_size=args.eval_batch_size,
                device=device,
                rng=rng,
            )
            row_t["method"] = method
            row_t["repeat"] = 0
            perturbation_rows.append(row_t)

            for rep in range(1, args.random_repeats + 1):
                row_r = evaluate_masking_condition(
                    model=model,
                    clean_inputs=explain_inputs,
                    clean_probs=clean_probs,
                    clean_metrics=clean_metrics,
                    target_idx=target_idx,
                    clean_pred=clean_pred,
                    importance=importance,
                    fraction=frac,
                    mode="random",
                    eval_batch_size=args.eval_batch_size,
                    device=device,
                    rng=rng,
                )
                row_r["method"] = method
                row_r["repeat"] = rep
                perturbation_rows.append(row_r)

        method_summary_rows.append(
            build_method_summary(method=method, rows=perturbation_rows, fractions=mask_fractions)
        )

    method_summary_rows = sorted(
        method_summary_rows,
        key=lambda x: x["aopc_target_prob_drop_gap"],
        reverse=True,
    )

    summary = {
        "checkpoint": str(checkpoint),
        "dataset_dir": str(dataset_dir),
        "feature_set": args.feature_set,
        "label_source": args.label_source,
        "class_names": class_names,
        "target_mode": args.target_mode,
        "methods": methods,
        "mask_fractions": mask_fractions,
        "random_repeats": args.random_repeats,
        "ig_steps": args.ig_steps,
        "smoothgrad_samples": args.smoothgrad_samples,
        "smoothgrad_sigma": args.smoothgrad_sigma,
        "seed": args.seed,
        "num_eval_jets": int(len(eval_inputs.y_index)),
        "num_explain_jets": int(len(explain_inputs.y_index)),
        "explain_sampling": args.explain_sampling,
        "explain_label_distribution": explain_label_dist,
        "clean_metrics_eval_split": clean_metrics_eval,
        "label_order_permutation_diagnostic_eval_split": clean_perm_eval,
        "clean_metrics_explain_subset": clean_metrics,
        "label_order_permutation_diagnostic_explain_subset": clean_perm_explain,
        "best_method_by_target_prob_gap": method_summary_rows[0] if method_summary_rows else None,
    }

    (output_dir / "interpretability_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "feature_attribution_summary.json").write_text(
        json.dumps(feature_rows, indent=2) + "\n"
    )
    (output_dir / "masking_perturbation_results.json").write_text(
        json.dumps(perturbation_rows, indent=2) + "\n"
    )
    (output_dir / "method_effectiveness_summary.json").write_text(
        json.dumps(method_summary_rows, indent=2) + "\n"
    )

    write_csv(feature_rows, output_dir / "feature_attribution_summary.csv")
    write_csv(perturbation_rows, output_dir / "masking_perturbation_results.csv")
    write_csv(method_summary_rows, output_dir / "method_effectiveness_summary.csv")

    print("\n=== Clean metrics (eval split) ===")
    for k, v in clean_metrics_eval.items():
        print(f"{k:>18s}: {v:.6f}")
    if np.isfinite(float(clean_perm_eval.get("best_permutation_accuracy", float("nan")))):
        print(
            f"{'best_perm_acc':>18s}: "
            f"{clean_perm_eval['best_permutation_accuracy']:.6f} "
            f"(delta={clean_perm_eval['improvement_over_raw']:.6f})"
        )
    print("\n=== Method ranking (targeted-vs-random target-prob-drop gap) ===")
    for row in method_summary_rows:
        print(
            f"{row['method']:>22s}: "
            f"gap={row['aopc_target_prob_drop_gap']:.6f}, "
            f"auc_gap={row['aopc_auc_drop_gap']:.6f}, "
            f"acc_gap={row['aopc_acc_drop_gap']:.6f}"
        )
    print(f"\nWrote interpretability reports to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
