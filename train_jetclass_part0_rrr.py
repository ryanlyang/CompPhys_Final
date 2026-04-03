#!/usr/bin/env python3

from __future__ import annotations

"""
Train JetClass ParticleTransformer with "Right for the Right Reasons" (RRR)
input-gradient regularization.

Core objective:
  loss = CE + lambda_rrr * || A * d(score)/d(input_features) ||^2

where A can be:
  - all ones (global gradient smoothing), or
  - adaptive top-k feature-channel mask (discourages current shortcut features),
    inspired by "find another explanation" style training.
"""

import argparse
import gc
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from run_jetclass_part0_baseline_and_shift import (
    FILENAME_CLASS_NAMES_BY_LABEL_INDEX,
    FEATURE_DIMS,
    InputArrays,
    LABEL_NAMES,
    ParticleTransformerWrapper,
    compute_supervised_metrics,
    load_split,
    parse_index_spec,
    prepare_split_dirs,
)


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train JetClass ParT with RRR gradient regularization.")
    p.add_argument("--dataset-dir", required=True, help="Flat dir with class ROOT files.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--feature-set", default="kinpid", choices=["kin", "kinpid", "full"])
    p.add_argument("--label-source", default="filename", choices=["filename", "branch"])
    p.add_argument("--max-num-particles", type=int, default=128)

    p.add_argument("--train-indices", default="0-7")
    p.add_argument("--val-indices", default="8")
    p.add_argument("--test-indices", default="9")
    p.add_argument("--max-train-jets", type=int, default=50000)
    p.add_argument("--max-val-jets", type=int, default=10000)
    p.add_argument("--max-test-jets", type=int, default=50000)

    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--lambda-rrr", type=float, default=8.0, help="Weight for RRR penalty term.")
    p.add_argument("--rrr-start-epoch", type=int, default=2, help="Enable RRR from this epoch (1-based).")
    p.add_argument("--rrr-score-mode", default="sum_log_probs", choices=["sum_log_probs", "true_log_prob", "pred_log_prob"])
    p.add_argument("--rrr-mask-mode", default="adaptive_topk", choices=["all", "adaptive_topk"])
    p.add_argument("--adaptive-topk-features", type=int, default=5)
    p.add_argument(
        "--adaptive-mask-floor",
        type=float,
        default=0.15,
        help="Background mask weight for non-topk channels in adaptive mode.",
    )
    p.add_argument(
        "--adaptive-refresh-epochs",
        type=int,
        default=1,
        help="How often to recompute adaptive feature importance once RRR starts.",
    )
    p.add_argument(
        "--adaptive-probe-jets",
        type=int,
        default=20000,
        help="Jets used to estimate gradient-importance channels.",
    )
    p.add_argument("--save-every-epoch", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tensor_from_np(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(x).to(device=device, non_blocking=True)


def build_model(feature_set: str, num_classes: int, device: torch.device) -> torch.nn.Module:
    cfg = dict(
        input_dim=FEATURE_DIMS[feature_set],
        num_classes=num_classes,
        pair_input_dim=4,
        use_pre_activation_pair=False,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={"dropout": 0, "attn_dropout": 0, "activation_dropout": 0},
        fc_params=[],
        activation="gelu",
        trim=True,
        for_inference=False,
        use_amp=False,
    )
    model = ParticleTransformerWrapper(**cfg)
    if hasattr(model, "mod") and hasattr(model.mod, "use_amp"):
        model.mod.use_amp = False
    model.to(device)
    return model


def _score_from_logits(logits: torch.Tensor, y: torch.Tensor, mode: str) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    if mode == "sum_log_probs":
        return log_probs.sum(dim=1).mean()
    if mode == "true_log_prob":
        return log_probs.gather(1, y[:, None]).mean()
    if mode == "pred_log_prob":
        pred = logits.detach().argmax(dim=1)
        return log_probs.gather(1, pred[:, None]).mean()
    raise ValueError(f"Unsupported rrr score mode: {mode}")


def compute_rrr_penalty(
    logits: torch.Tensor,
    features: torch.Tensor,
    y: torch.Tensor,
    score_mode: str,
    feature_mask: torch.Tensor | None,
) -> torch.Tensor:
    score = _score_from_logits(logits, y, mode=score_mode)
    grad = torch.autograd.grad(
        score,
        features,
        retain_graph=True,
        create_graph=True,
        allow_unused=False,
    )[0]
    grad2 = grad.pow(2)
    if feature_mask is not None:
        fm = feature_mask.to(device=grad2.device, dtype=grad2.dtype).view(1, -1, 1)
        grad2 = grad2 * fm
        denom = torch.clamp(fm.sum(), min=1.0) * grad2.shape[2]
    else:
        denom = float(grad2.shape[1] * grad2.shape[2])
    return grad2.sum(dim=(1, 2)).mean() / denom


@torch.no_grad()
def predict_probs(
    model: torch.nn.Module,
    inputs: InputArrays,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    outs: List[np.ndarray] = []
    n = len(inputs.y_index)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        points = _tensor_from_np(inputs.points[start:end], device)
        features = _tensor_from_np(inputs.features[start:end], device)
        vectors = _tensor_from_np(inputs.vectors[start:end], device)
        mask = _tensor_from_np(inputs.mask[start:end], device)

        logits = model(points, features, vectors, (mask > 0.5))
        if not torch.isfinite(logits).all():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
        probs = torch.softmax(logits, dim=1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
        outs.append(probs.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outs, axis=0)


def estimate_feature_importance(
    model: torch.nn.Module,
    inputs: InputArrays,
    device: torch.device,
    batch_size: int,
    score_mode: str,
    probe_jets: int,
    seed: int,
) -> np.ndarray:
    model.eval()
    n_total = len(inputs.y_index)
    n_probe = n_total if probe_jets <= 0 else min(n_total, probe_jets)
    rng = np.random.default_rng(seed)
    probe_idx = np.arange(n_total, dtype=np.int64)
    if n_probe < n_total:
        probe_idx = rng.choice(probe_idx, size=n_probe, replace=False)
    probe_idx = np.asarray(probe_idx, dtype=np.int64)

    feat_dim = inputs.features.shape[1]
    agg = torch.zeros(feat_dim, dtype=torch.float64, device=device)
    seen = 0

    for start in range(0, len(probe_idx), batch_size):
        bidx = probe_idx[start : start + batch_size]
        points = _tensor_from_np(inputs.points[bidx], device)
        features = _tensor_from_np(inputs.features[bidx], device)
        vectors = _tensor_from_np(inputs.vectors[bidx], device)
        mask = _tensor_from_np(inputs.mask[bidx], device)
        y = torch.from_numpy(inputs.y_index[bidx]).to(device=device, dtype=torch.long)

        features.requires_grad_(True)
        logits = model(points, features, vectors, (mask > 0.5))
        if not torch.isfinite(logits).all():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
        score = _score_from_logits(logits, y, score_mode)
        grad = torch.autograd.grad(
            score,
            features,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        # Mean absolute gradient by feature channel.
        agg += grad.abs().mean(dim=(0, 2)).to(torch.float64) * len(bidx)
        seen += len(bidx)

    if seen == 0:
        return np.zeros(feat_dim, dtype=np.float64)
    return (agg / float(seen)).detach().cpu().numpy()


def build_feature_mask(
    feature_set: str,
    mode: str,
    adaptive_importance: np.ndarray | None,
    adaptive_topk_features: int,
    adaptive_mask_floor: float,
) -> tuple[torch.Tensor | None, Dict[str, object]]:
    feat_dim = FEATURE_DIMS[feature_set]
    if mode == "all":
        return torch.ones(feat_dim, dtype=torch.float32), {
            "mask_mode": "all",
            "selected_feature_indices": list(range(feat_dim)),
            "selected_feature_names": FEATURE_NAMES[feature_set],
        }

    if adaptive_importance is None:
        return torch.ones(feat_dim, dtype=torch.float32), {
            "mask_mode": "adaptive_topk",
            "selected_feature_indices": list(range(feat_dim)),
            "selected_feature_names": FEATURE_NAMES[feature_set],
            "note": "importance unavailable; fallback to all-feature mask",
        }

    k = int(max(1, min(adaptive_topk_features, feat_dim)))
    idx = np.argsort(-adaptive_importance)[:k]
    idx = np.asarray(sorted(idx.tolist()), dtype=np.int64)
    mask = np.full((feat_dim,), float(adaptive_mask_floor), dtype=np.float32)
    mask[idx] = 1.0
    names = FEATURE_NAMES[feature_set]
    return torch.from_numpy(mask), {
        "mask_mode": "adaptive_topk",
        "adaptive_topk_features": k,
        "adaptive_mask_floor": float(adaptive_mask_floor),
        "selected_feature_indices": idx.tolist(),
        "selected_feature_names": [names[int(i)] for i in idx.tolist()],
        "importance_by_feature": {names[i]: float(adaptive_importance[i]) for i in range(feat_dim)},
    }


def eval_split_metrics(
    model: torch.nn.Module,
    split_name: str,
    inputs: InputArrays,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    probs = predict_probs(model, inputs, batch_size=batch_size, device=device)
    m = compute_supervised_metrics(inputs, probs)
    print(
        f"[{split_name}] acc={m['accuracy']:.5f} "
        f"macro_auc={m['macro_auc']:.5f} "
        f"entropy={m['mean_entropy']:.5f} "
        f"conf={m['mean_confidence']:.5f}"
    )
    return m


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    dataset_dir = Path(args.dataset_dir).resolve()
    split_paths = prepare_split_dirs(
        dataset_dir=dataset_dir,
        split_dir=split_dir,
        train_indices=parse_index_spec(args.train_indices),
        val_indices=parse_index_spec(args.val_indices),
        test_indices=parse_index_spec(args.test_indices),
    )

    train_files = sorted(split_paths["train"].glob("*.root"))
    val_files = sorted(split_paths["val"].glob("*.root"))
    test_files = sorted(split_paths["test"].glob("*.root"))
    if not train_files or not val_files or not test_files:
        raise FileNotFoundError("One or more splits are empty. Check indices/data.")

    print(f"Loading train split from {len(train_files)} files...")
    train_inputs = load_split(
        files=train_files,
        feature_set=args.feature_set,
        max_num_particles=args.max_num_particles,
        max_jets=args.max_train_jets,
        label_source=args.label_source,
    )
    print(f"Loaded {len(train_inputs.y_index)} train jets.")

    print(f"Loading val split from {len(val_files)} files...")
    val_inputs = load_split(
        files=val_files,
        feature_set=args.feature_set,
        max_num_particles=args.max_num_particles,
        max_jets=args.max_val_jets,
        label_source=args.label_source,
    )
    print(f"Loaded {len(val_inputs.y_index)} val jets.")

    class_names = (
        FILENAME_CLASS_NAMES_BY_LABEL_INDEX if args.label_source == "filename" else LABEL_NAMES
    )
    n_classes = len(class_names)
    for split_name, y in [
        ("train", train_inputs.y_index),
        ("val", val_inputs.y_index),
    ]:
        uniq, counts = np.unique(y, return_counts=True)
        dist = {class_names[int(k)]: int(v) for k, v in zip(uniq, counts)}
        print(f"{split_name} class distribution: {dist}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and args.device != "cpu":
        print("Warning: CUDA not available, using CPU.")

    model = build_model(feature_set=args.feature_set, num_classes=n_classes, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    n_train = len(train_inputs.y_index)
    history_rows: List[Dict[str, float]] = []
    mask_updates: List[Dict[str, object]] = []

    feature_mask: torch.Tensor | None = None
    best_val_auc = -float("inf")
    best_state: Dict[str, torch.Tensor] | None = None
    best_epoch = -1

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()

        rrr_enabled = args.lambda_rrr > 0 and epoch >= args.rrr_start_epoch
        should_refresh = (
            rrr_enabled
            and args.rrr_mask_mode == "adaptive_topk"
            and (
                feature_mask is None
                or (epoch - args.rrr_start_epoch) % max(1, args.adaptive_refresh_epochs) == 0
            )
        )
        if should_refresh:
            importance = estimate_feature_importance(
                model=model,
                inputs=train_inputs,
                device=device,
                batch_size=args.eval_batch_size,
                score_mode=args.rrr_score_mode,
                probe_jets=args.adaptive_probe_jets,
                seed=args.seed + epoch,
            )
            feature_mask, meta = build_feature_mask(
                feature_set=args.feature_set,
                mode=args.rrr_mask_mode,
                adaptive_importance=importance,
                adaptive_topk_features=args.adaptive_topk_features,
                adaptive_mask_floor=args.adaptive_mask_floor,
            )
            meta["epoch"] = epoch
            mask_updates.append(meta)
            print(
                f"[epoch {epoch}] adaptive RRR selected features: "
                f"{meta.get('selected_feature_names', [])}"
            )
        elif feature_mask is None:
            feature_mask, meta = build_feature_mask(
                feature_set=args.feature_set,
                mode=args.rrr_mask_mode,
                adaptive_importance=None,
                adaptive_topk_features=args.adaptive_topk_features,
                adaptive_mask_floor=args.adaptive_mask_floor,
            )
            meta["epoch"] = epoch
            mask_updates.append(meta)

        order = np.random.permutation(n_train)
        sum_loss = 0.0
        sum_ce = 0.0
        sum_rrr = 0.0
        seen = 0

        for start in range(0, n_train, args.batch_size):
            bidx = order[start : start + args.batch_size]
            points = _tensor_from_np(train_inputs.points[bidx], device)
            features = _tensor_from_np(train_inputs.features[bidx], device)
            vectors = _tensor_from_np(train_inputs.vectors[bidx], device)
            mask = _tensor_from_np(train_inputs.mask[bidx], device)
            y = torch.from_numpy(train_inputs.y_index[bidx]).to(device=device, dtype=torch.long)

            features.requires_grad_(rrr_enabled)
            optimizer.zero_grad(set_to_none=True)

            logits = model(points, features, vectors, (mask > 0.5))
            if not torch.isfinite(logits).all():
                logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
            ce = F.cross_entropy(logits, y)
            if rrr_enabled:
                rrr = compute_rrr_penalty(
                    logits=logits,
                    features=features,
                    y=y,
                    score_mode=args.rrr_score_mode,
                    feature_mask=feature_mask,
                )
            else:
                rrr = torch.zeros((), device=device, dtype=ce.dtype)
            loss = ce + float(args.lambda_rrr) * rrr

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}.")

            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            bs = len(bidx)
            seen += bs
            sum_loss += float(loss.detach().item()) * bs
            sum_ce += float(ce.detach().item()) * bs
            sum_rrr += float(rrr.detach().item()) * bs

        train_loss = sum_loss / max(seen, 1)
        train_ce = sum_ce / max(seen, 1)
        train_rrr = sum_rrr / max(seen, 1)

        train_metrics = eval_split_metrics(
            model, "train", train_inputs, batch_size=args.eval_batch_size, device=device
        )
        val_metrics = eval_split_metrics(
            model, "val", val_inputs, batch_size=args.eval_batch_size, device=device
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_ce": train_ce,
            "train_rrr": train_rrr,
            "train_accuracy": float(train_metrics["accuracy"]),
            "train_macro_auc": float(train_metrics["macro_auc"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_auc": float(val_metrics["macro_auc"]),
            "val_entropy": float(val_metrics["mean_entropy"]),
            "val_confidence": float(val_metrics["mean_confidence"]),
            "rrr_enabled": 1.0 if rrr_enabled else 0.0,
        }
        history_rows.append(row)
        print(
            f"[epoch {epoch:02d}] "
            f"loss={train_loss:.6f} ce={train_ce:.6f} rrr={train_rrr:.6f} "
            f"val_acc={row['val_accuracy']:.5f} val_auc={row['val_macro_auc']:.5f}"
        )

        if val_metrics["macro_auc"] > best_val_auc:
            best_val_auc = float(val_metrics["macro_auc"])
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, output_dir / "saved_model.pt")
            print(f"[epoch {epoch:02d}] New best val macro AUC: {best_val_auc:.6f} (checkpoint saved)")

        if args.save_every_epoch:
            torch.save(model.state_dict(), output_dir / f"saved_model_epoch{epoch:03d}.pt")

    if best_state is None:
        raise RuntimeError("Training completed but no checkpoint was saved.")
    model.load_state_dict(best_state, strict=True)
    model.to(device)
    model.eval()

    print("\n=== Final best-checkpoint metrics ===")
    train_best = eval_split_metrics(model, "train(best)", train_inputs, args.eval_batch_size, device)
    val_best = eval_split_metrics(model, "val(best)", val_inputs, args.eval_batch_size, device)

    # Load test only after training to reduce peak host-memory usage.
    gc.collect()
    print(f"Loading test split from {len(test_files)} files...")
    test_inputs = load_split(
        files=test_files,
        feature_set=args.feature_set,
        max_num_particles=args.max_num_particles,
        max_jets=args.max_test_jets,
        label_source=args.label_source,
    )
    print(f"Loaded {len(test_inputs.y_index)} test jets.")
    uniq_t, cnt_t = np.unique(test_inputs.y_index, return_counts=True)
    test_dist = {class_names[int(k)]: int(v) for k, v in zip(uniq_t, cnt_t)}
    print(f"test class distribution: {test_dist}")
    test_best = eval_split_metrics(model, "test(best)", test_inputs, args.eval_batch_size, device)

    feature_mask_out = (
        feature_mask.detach().cpu().numpy().astype(np.float64).tolist() if feature_mask is not None else None
    )

    summary = {
        "best_epoch": int(best_epoch),
        "best_val_macro_auc": float(best_val_auc),
        "elapsed_seconds": float(time.time() - t0),
        "n_train": int(len(train_inputs.y_index)),
        "n_val": int(len(val_inputs.y_index)),
        "n_test": int(len(test_inputs.y_index)),
        "train_metrics_best": train_best,
        "val_metrics_best": val_best,
        "test_metrics_best": test_best,
        "class_names": class_names,
        "rrr": {
            "lambda_rrr": float(args.lambda_rrr),
            "rrr_start_epoch": int(args.rrr_start_epoch),
            "rrr_score_mode": args.rrr_score_mode,
            "rrr_mask_mode": args.rrr_mask_mode,
            "adaptive_topk_features": int(args.adaptive_topk_features),
            "adaptive_mask_floor": float(args.adaptive_mask_floor),
            "adaptive_refresh_epochs": int(args.adaptive_refresh_epochs),
            "adaptive_probe_jets": int(args.adaptive_probe_jets),
            "final_feature_mask": feature_mask_out,
            "mask_updates": mask_updates,
        },
        "saved_model_path": str((output_dir / "saved_model.pt").resolve()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "rrr_mask_updates.json").write_text(json.dumps(mask_updates, indent=2) + "\n")
    (output_dir / "history.json").write_text(json.dumps(history_rows, indent=2) + "\n")

    # Compact CSV for quick plotting.
    header = list(history_rows[0].keys()) if history_rows else []
    if header:
        lines = [",".join(header)]
        for row in history_rows:
            vals = []
            for k in header:
                v = row[k]
                if isinstance(v, float):
                    vals.append(f"{v:.10g}")
                else:
                    vals.append(str(v))
            lines.append(",".join(vals))
        (output_dir / "history.csv").write_text("\n".join(lines) + "\n")

    print("\nDone. Key outputs:")
    print(f"  {output_dir / 'saved_model.pt'}")
    print(f"  {output_dir / 'summary.json'}")
    print(f"  {output_dir / 'history.csv'}")
    print(f"  {output_dir / 'rrr_mask_updates.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
