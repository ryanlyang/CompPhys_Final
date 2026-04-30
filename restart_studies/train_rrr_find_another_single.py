#!/usr/bin/env python3
from __future__ import annotations


import argparse
import csv
import gc
import json
import math
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRACTICETAGGING_ROOT = PROJECT_ROOT / "PracticeTagging"
if str(PRACTICETAGGING_ROOT) not in sys.path:
    sys.path.insert(0, str(PRACTICETAGGING_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_jetclass_hlt_teacher_baseline import (              
    IDX_CHARGE,
    IDX_D0,
    IDX_D0ERR,
    IDX_DZ,
    IDX_DZERR,
    IDX_PT,
    JetClassTransformer,
    collect_files_by_class,
    compute_features,
    eval_metrics,
    get_mean_std,
    load_split,
    split_files_by_class,
    standardize,
)
from evaluate_aspen_shift_calibration import (              
    distributional_shift_metrics,
    probs_to_stats,
    stream_aspen_stats,
)
from reimplement_preliminary_studies import (              
    attribution_input_grad,
    attribution_integrated_gradients,
    attribution_smoothgrad,
    evaluate_probs,
    mean_confidence,
    mean_entropy,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train one RRR/find-another config and evaluate JetClass + Aspen")
    p.add_argument(
        "--data_dir",
        type=Path,
        default=Path("/home/ryreu/atlas/PracticeTagging/data/jetclass_part0"),
    )
    p.add_argument(
        "--aspen_data_dir",
        type=Path,
        default=Path("/home/ryreu/atlas/CompPhys_Final/data/AspenOpenJets"),
    )
    p.add_argument("--aspen_glob", type=str, default="Run*.h5")
    p.add_argument("--aspen_n_jets", type=int, default=1_000_000)
    p.add_argument("--aspen_chunk_jets", type=int, default=50_000)

    p.add_argument("--output_root", type=Path, default=PROJECT_ROOT / "restart_studies" / "results")
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--seed", type=int, default=52)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--feature_mode", type=str, default="full", choices=["kin", "kinpid", "full"])
    p.add_argument("--max_constits", type=int, default=128)
    p.add_argument("--train_files_per_class", type=int, default=8)
    p.add_argument("--val_files_per_class", type=int, default=1)
    p.add_argument("--test_files_per_class", type=int, default=1)
    p.add_argument("--shuffle_files", action="store_true", default=False)
    p.add_argument("--n_train_jets", type=int, default=150000)
    p.add_argument("--n_val_jets", type=int, default=50000)
    p.add_argument("--n_test_jets", type=int, default=150000)

    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--rrr_batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--ff_dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--target_class", type=str, default="HToBB")
    p.add_argument("--background_class", type=str, default="ZJetsToNuNu")

    p.add_argument("--a_source", type=str, default="input_grad", choices=["input_grad", "integrated_gradients", "smoothgrad"])
    p.add_argument("--lambda_rrr", type=float, required=True)
    p.add_argument("--mask_frac", type=float, required=True)
    p.add_argument("--max_iterations", type=int, default=5)

    p.add_argument("--attr_batch_size", type=int, default=128)
    p.add_argument("--ig_steps", type=int, default=16)
    p.add_argument("--smoothgrad_samples", type=int, default=16)
    p.add_argument("--smoothgrad_sigma", type=float, default=0.10)
    return p.parse_args()


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def domain_prior_feature_indices(feature_mode: str, feat_dim: int) -> List[int]:
                                                                           
    if feature_mode == "kin":
        return []
    if feat_dim <= 7:
        return []
    return list(range(7, feat_dim))


def build_topk_token_mask(valid_mask: np.ndarray, scores: np.ndarray, frac: float) -> np.ndarray:
    n, t = valid_mask.shape
    out = np.zeros((n, t), dtype=bool)
    f = float(np.clip(frac, 1e-6, 1.0))
    for i in range(n):
        idx = np.where(valid_mask[i])[0]
        if idx.size == 0:
            continue
        k = int(round(f * float(idx.size)))
        k = max(1, min(k, int(idx.size)))
        s = scores[i, idx]
        sel_local = np.argsort(-s)[:k]
        out[i, idx[sel_local]] = True
    return out


def build_feature_mask_from_tokens(
    token_mask: np.ndarray,
    feat_dim: int,
    feature_indices: Sequence[int] | None,
) -> np.ndarray:
    n, t = token_mask.shape
    out = np.zeros((n, t, feat_dim), dtype=bool)
    if feature_indices is None:
        out[token_mask] = True
    else:
        if len(feature_indices) == 0:
            return out
        out[:, :, list(feature_indices)] = token_mask[:, :, None]
    return out


def compute_token_attributions(
    source: str,
    model: torch.nn.Module,
    feat: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    device: torch.device,
    ig_steps: int,
    sg_samples: int,
    sg_sigma: float,
) -> np.ndarray:
    if source == "input_grad":
        return attribution_input_grad(model, feat, mask, labels, batch_size=batch_size, device=device)
    if source == "integrated_gradients":
        return attribution_integrated_gradients(
            model=model,
            feat=feat,
            mask=mask,
            labels=labels,
            batch_size=batch_size,
            ig_steps=ig_steps,
            device=device,
        )
    if source == "smoothgrad":
        return attribution_smoothgrad(
            model=model,
            feat=feat,
            mask=mask,
            labels=labels,
            batch_size=batch_size,
            sg_samples=sg_samples,
            sg_sigma=sg_sigma,
            device=device,
        )
    raise ValueError(f"Unknown a_source: {source}")


def train_one_iteration(
    *,
    feat_tr: np.ndarray,
    mask_tr: np.ndarray,
    y_tr: np.ndarray,
    a_mask_tr: np.ndarray,
    feat_va: np.ndarray,
    mask_va: np.ndarray,
    y_va: np.ndarray,
    class_names: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.nn.Module, List[Dict[str, float]], Dict[str, float]]:
    n_classes = int(len(class_names))
    model = JetClassTransformer(
        input_dim=int(feat_tr.shape[-1]),
        n_classes=n_classes,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def sched_lambda(ep: int) -> float:
        if ep < args.warmup_epochs:
            return float(ep + 1) / float(max(1, args.warmup_epochs))
        x = float(ep - args.warmup_epochs) / float(max(1, args.epochs - args.warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * x))

    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=sched_lambda)
    hist: List[Dict[str, float]] = []

    n = int(len(y_tr))
    idx_all = np.arange(n, dtype=np.int64)
    bs = int(max(1, args.rrr_batch_size))
    lam = float(args.lambda_rrr)

    for ep in range(1, int(args.epochs) + 1):
        np.random.shuffle(idx_all)
        model.train()
        tot_loss = 0.0
        tot_ce = 0.0
        tot_rrr = 0.0
        tot_n = 0

        for s in range(0, n, bs):
            e = min(n, s + bs)
            bid = idx_all[s:e]
            x = torch.tensor(feat_tr[bid], dtype=torch.float32, device=device, requires_grad=True)
            m = torch.tensor(mask_tr[bid], dtype=torch.bool, device=device)
            y = torch.tensor(y_tr[bid], dtype=torch.long, device=device)
            a = torch.tensor(a_mask_tr[bid], dtype=torch.float32, device=device)

            opt.zero_grad(set_to_none=True)
                                                                          
                                                                                    
            if device.type == "cuda" and hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
                sdp_ctx = torch.backends.cuda.sdp_kernel(
                    enable_flash=False,
                    enable_mem_efficient=False,
                    enable_math=True,
                )
            else:
                sdp_ctx = nullcontext()
            with sdp_ctx:
                logits = model(x, m)
                ce = F.cross_entropy(logits, y)
                tlog = logits.gather(1, y.view(-1, 1)).sum()
                gx = torch.autograd.grad(tlog, x, create_graph=True)[0]
                rrr_num = (gx.square() * a).sum()
                rrr_den = torch.clamp(a.sum(), min=1.0)
                rrr = rrr_num / rrr_den
                loss = ce + lam * rrr
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bsz = int(y.shape[0])
            tot_loss += float(loss.item()) * bsz
            tot_ce += float(ce.item()) * bsz
            tot_rrr += float(rrr.item()) * bsz
            tot_n += bsz

        sch.step()
        val_pack = evaluate_probs(
            model=model,
            feat=feat_va,
            mask=mask_va,
            labels=y_va,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        val_metrics = eval_metrics(
            y_true=val_pack["labels"],
            probs=val_pack["probs"],
            class_names=class_names,
            background_class=args.background_class,
            target_class=args.target_class,
        )
        row = {
            "epoch": float(ep),
            "train_loss": float(tot_loss / max(1, tot_n)),
            "train_ce": float(tot_ce / max(1, tot_n)),
            "train_rrr": float(tot_rrr / max(1, tot_n)),
            "val_acc": float(val_metrics["acc"]),
            "val_auc_macro_ovr": float(val_metrics["auc_macro_ovr"]),
        }
        hist.append(row)
        print(
            f"[iter-train] ep={ep} loss={row['train_loss']:.4f} "
            f"ce={row['train_ce']:.4f} rrr={row['train_rrr']:.4f} "
            f"val_acc={row['val_acc']:.4f} val_auc={row['val_auc_macro_ovr']:.4f}"
        )

    last_val = {
        "val_acc": float(hist[-1]["val_acc"]) if hist else float("nan"),
        "val_auc_macro_ovr": float(hist[-1]["val_auc_macro_ovr"]) if hist else float("nan"),
    }
    return model, hist, last_val


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    run_dir = (args.output_root / args.run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] run_dir={run_dir}")

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"[info] device={device}")

    files_by_class = collect_files_by_class(args.data_dir.resolve())
    class_names = sorted(files_by_class.keys())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    print("[info] classes:")
    for c in class_names:
        print(f"  - {c}: {len(files_by_class[c])} files")

    tr_files, va_files, te_files = split_files_by_class(
        files_by_class,
        n_train=args.train_files_per_class,
        n_val=args.val_files_per_class,
        n_test=args.test_files_per_class,
        shuffle=args.shuffle_files,
        seed=args.seed,
    )

    print("[info] loading train/val/test splits")
    tr_tok, tr_mask, tr_y = load_split(
        tr_files,
        n_total=args.n_train_jets,
        max_constits=args.max_constits,
        class_to_idx=class_to_idx,
        seed=args.seed + 101,
    )
    va_tok, va_mask, va_y = load_split(
        va_files,
        n_total=args.n_val_jets,
        max_constits=args.max_constits,
        class_to_idx=class_to_idx,
        seed=args.seed + 202,
    )
    te_tok, te_mask, te_y = load_split(
        te_files,
        n_total=args.n_test_jets,
        max_constits=args.max_constits,
        class_to_idx=class_to_idx,
        seed=args.seed + 303,
    )
    print(f"[info] loaded jets train={len(tr_y)} val={len(va_y)} test={len(te_y)}")

    tr_feat = compute_features(tr_tok, tr_mask, feature_mode=args.feature_mode)
    va_feat = compute_features(va_tok, va_mask, feature_mode=args.feature_mode)
    te_feat = compute_features(te_tok, te_mask, feature_mode=args.feature_mode)

    mean, std = get_mean_std(tr_feat, tr_mask, np.arange(len(tr_y)))
    tr_feat = standardize(tr_feat, tr_mask, mean, std)
    va_feat = standardize(va_feat, va_mask, mean, std)
    te_feat = standardize(te_feat, te_mask, mean, std)
    feat_dim = int(tr_feat.shape[-1])

                                                                     
    domain_token_scores = np.maximum(tr_tok[:, :, IDX_PT], 0.0).astype(np.float32)
    token_mask_prior = build_topk_token_mask(tr_mask, domain_token_scores, frac=args.mask_frac)
    prior_dims = domain_prior_feature_indices(args.feature_mode, feat_dim)
    a_total = build_feature_mask_from_tokens(token_mask_prior, feat_dim=feat_dim, feature_indices=prior_dims)
    print(
        "[info] domain-prior mask coverage "
        f"tokens={token_mask_prior.sum()}/{tr_mask.sum()} "
        f"features={a_total.sum()}/{tr_mask.sum()*feat_dim}"
    )

    iter_rows: List[Dict[str, object]] = []
    prev_model: torch.nn.Module | None = None
    saved_ckpts: List[str] = []

    for it in range(1, int(args.max_iterations) + 1):
        it_dir = run_dir / f"iter_{it:02d}"
        it_dir.mkdir(parents=True, exist_ok=True)
        iter_seed = int(args.seed + 1000 * it)
        set_seed(iter_seed)
        print("=" * 72)
        print(f"[iter] {it}/{args.max_iterations}")
        print(f"[iter] seed={iter_seed}")

        if it >= 2:
            if prev_model is None:
                raise RuntimeError("Previous model missing for attribution-driven mask update.")
            print(f"[iter] computing full-dataset attributions source={args.a_source}")
            attrs = compute_token_attributions(
                source=args.a_source,
                model=prev_model,
                feat=tr_feat,
                mask=tr_mask,
                labels=tr_y,
                batch_size=args.attr_batch_size,
                device=device,
                ig_steps=args.ig_steps,
                sg_samples=args.smoothgrad_samples,
                sg_sigma=args.smoothgrad_sigma,
            )
            token_mask_new = build_topk_token_mask(tr_mask, attrs, frac=args.mask_frac)
            a_new = build_feature_mask_from_tokens(token_mask_new, feat_dim=feat_dim, feature_indices=None)
            a_total = np.logical_or(a_total, a_new)
            with (it_dir / "attribution_mask_update_stats.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "iteration": it,
                        "a_source": args.a_source,
                        "token_mask_new_frac_of_valid": float(token_mask_new.sum() / max(1, tr_mask.sum())),
                        "a_total_feat_frac_of_validxdim": float(a_total.sum() / max(1, tr_mask.sum() * feat_dim)),
                    },
                    f,
                    indent=2,
                    sort_keys=True,
                )
            del attrs, token_mask_new, a_new
            gc.collect()

        model, hist, last_val = train_one_iteration(
            feat_tr=tr_feat,
            mask_tr=tr_mask,
            y_tr=tr_y,
            a_mask_tr=a_total,
            feat_va=va_feat,
            mask_va=va_mask,
            y_va=va_y,
            class_names=class_names,
            args=args,
            device=device,
        )
        prev_model = model

        ckpt = it_dir / "model.pt"
        torch.save(model.state_dict(), ckpt)
        saved_ckpts.append(str(ckpt))
        write_csv(
            it_dir / "train_history.csv",
            rows=hist,
            fieldnames=["epoch", "train_loss", "train_ce", "train_rrr", "val_acc", "val_auc_macro_ovr"],
        )

        test_pack = evaluate_probs(
            model=model,
            feat=te_feat,
            mask=te_mask,
            labels=te_y,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        test_metrics = eval_metrics(
            y_true=test_pack["labels"],
            probs=test_pack["probs"],
            class_names=class_names,
            background_class=args.background_class,
            target_class=args.target_class,
        )
        test_metrics["mean_entropy"] = mean_entropy(np.asarray(test_pack["probs"], dtype=np.float64))
        test_metrics["mean_confidence"] = mean_confidence(np.asarray(test_pack["probs"], dtype=np.float64))
        with (it_dir / "jetclass_test_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2, sort_keys=True)

        clean_stats = probs_to_stats(np.asarray(test_pack["probs"], dtype=np.float64))
        aspen = stream_aspen_stats(
            model=model,
            aspen_data_dir=args.aspen_data_dir.resolve(),
            glob_pattern=args.aspen_glob,
            n_jets=args.aspen_n_jets,
            chunk_jets=args.aspen_chunk_jets,
            max_constits=args.max_constits,
            feature_mode=args.feature_mode,
            mean=mean,
            std=std,
            batch_size=args.batch_size,
            device=device,
        )
        aspen_stats = aspen["stats"]
        aspen_shift = distributional_shift_metrics(clean_stats, aspen_stats)
        aspen_out = {
            "iteration": it,
            "n_jets_used": int(aspen["n_jets_used"]),
            "files_considered": int(aspen["files_considered"]),
            "files_consumed": aspen["files_consumed"],
            "aspen_shift_metrics": {k: float(v) for k, v in aspen_shift.items()},
            "aspen_stats": {
                "class_dist": [float(x) for x in np.asarray(aspen_stats["class_dist"])],
                "top1_hist": [float(x) for x in np.asarray(aspen_stats["top1_hist"])],
                "mean_confidence": float(aspen_stats["mean_confidence"]),
                "mean_entropy": float(aspen_stats["mean_entropy"]),
            },
        }
        with (it_dir / "aspen_shift_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(aspen_out, f, indent=2, sort_keys=True)

        strong3 = float(
            (
                float(aspen_shift["prob_l1_drift"])
                + float(aspen_shift["top1_flip_rate"])
                + float(aspen_shift["class_js_divergence"])
            )
            / 3.0
        )
        row = {
            "iteration": it,
            "iteration_seed": int(iter_seed),
            "val_acc": float(last_val["val_acc"]),
            "val_auc_macro_ovr": float(last_val["val_auc_macro_ovr"]),
            "test_acc": float(test_metrics["acc"]),
            "test_auc_macro_ovr": float(test_metrics["auc_macro_ovr"]),
            "aspen_prob_l1_drift": float(aspen_shift["prob_l1_drift"]),
            "aspen_top1_flip_rate": float(aspen_shift["top1_flip_rate"]),
            "aspen_class_js_divergence": float(aspen_shift["class_js_divergence"]),
            "aspen_confidence_drop": float(aspen_shift["confidence_drop"]),
            "aspen_entropy_shift": float(aspen_shift["entropy_shift"]),
            "aspen_strong3_mean": strong3,
            "aspen_n_jets_used": int(aspen["n_jets_used"]),
            "a_total_feat_frac_of_validxdim": float(a_total.sum() / max(1, tr_mask.sum() * feat_dim)),
        }
        iter_rows.append(row)
        print(
            f"[iter] done {it}: test_acc={row['test_acc']:.4f}, test_auc={row['test_auc_macro_ovr']:.4f}, "
            f"aspen_strong3={row['aspen_strong3_mean']:.4f}"
        )

                                                 
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(
        run_dir / "iteration_summary.csv",
        rows=iter_rows,
        fieldnames=list(iter_rows[0].keys()) if iter_rows else [],
    )

    best_idx = int(np.argmin([float(r["aspen_strong3_mean"]) for r in iter_rows])) if iter_rows else -1
    final_summary = {
        "run_name": args.run_name,
        "seed": args.seed,
        "a_source": args.a_source,
        "lambda_rrr": float(args.lambda_rrr),
        "mask_frac": float(args.mask_frac),
        "max_iterations": int(args.max_iterations),
        "best_iteration_by_aspen_strong3": int(iter_rows[best_idx]["iteration"]) if best_idx >= 0 else None,
        "best_iteration_record": iter_rows[best_idx] if best_idx >= 0 else None,
        "saved_checkpoints": saved_ckpts,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2, sort_keys=True)

    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True, default=str)

    print("[done] outputs:")
    print(" ", run_dir / "config.json")
    print(" ", run_dir / "iteration_summary.csv")
    print(" ", run_dir / "summary.json")
    for i in range(1, args.max_iterations + 1):
        print(" ", run_dir / f"iter_{i:02d}" / "jetclass_test_metrics.json")
        print(" ", run_dir / f"iter_{i:02d}" / "aspen_shift_metrics.json")


if __name__ == "__main__":
    main()
