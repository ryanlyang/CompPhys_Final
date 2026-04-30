#!/usr/bin/env python3
from __future__ import annotations


import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import confusion_matrix, roc_auc_score
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRACTICETAGGING_ROOT = PROJECT_ROOT / "PracticeTagging"
if str(PRACTICETAGGING_ROOT) not in sys.path:
    sys.path.insert(0, str(PRACTICETAGGING_ROOT))

from evaluate_jetclass_hlt_teacher_baseline import (              
    IDX_E,
    IDX_ETA,
    IDX_PHI,
    IDX_PT,
    JetClassTransformer,
    JetDataset,
    collect_files_by_class,
    compute_features,
    eval_metrics,
    get_mean_std,
    load_split,
    make_loader,
    split_files_by_class,
    standardize,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_corruptions(spec: str) -> List[Tuple[str, float]]:
    items: List[Tuple[str, float]] = []
    for tok in spec.split(","):
        t = tok.strip()
        if not t:
            continue
        if ":" not in t:
            raise ValueError(f"Invalid corruption token '{t}', expected kind:severity")
        kind, sev = t.split(":", 1)
        items.append((kind.strip(), float(sev)))
    if not items:
        raise ValueError("No valid corruptions parsed")
    return items


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def macro_auc_ovr(y_true: np.ndarray, probs: np.ndarray, n_classes: int) -> float:
    y_1h = np.eye(n_classes, dtype=np.int64)[y_true]
    try:
        return float(roc_auc_score(y_1h, probs, average="macro", multi_class="ovr"))
    except ValueError:
        return float("nan")


def mean_entropy(probs: np.ndarray) -> float:
    p = np.clip(probs, 1e-12, 1.0)
    ent = -(p * np.log(p)).sum(axis=1)
    return float(ent.mean())


def mean_confidence(probs: np.ndarray) -> float:
    return float(np.max(probs, axis=1).mean())


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0)
    q = np.clip(np.asarray(q, dtype=np.float64), 1e-12, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p) - np.log(m)))
    kl_qm = np.sum(q * (np.log(q) - np.log(m)))
    return float(0.5 * (kl_pm + kl_qm))


def safe_corr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) < 2 or len(y) < 2:
        return float("nan"), float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan"), float("nan")
    sp, _ = spearmanr(x, y)
    pr, _ = pearsonr(x, y)
    return float(sp), float(pr)


def _wrap_phi_np(x: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(x), np.cos(x))


def _merge_two_tokens(tok_a: np.ndarray, tok_b: np.ndarray) -> np.ndarray:
    pt1 = max(float(tok_a[IDX_PT]), 1e-8)
    pt2 = max(float(tok_b[IDX_PT]), 1e-8)
    w1 = pt1 / (pt1 + pt2)
    w2 = 1.0 - w1
    out = tok_a.copy()
    out[IDX_PT] = pt1 + pt2
    out[IDX_E] = max(float(tok_a[IDX_E] + tok_b[IDX_E]), 1e-8)
    out[IDX_ETA] = np.clip(w1 * float(tok_a[IDX_ETA]) + w2 * float(tok_b[IDX_ETA]), -5.0, 5.0)
    out[IDX_PHI] = math.atan2(
        w1 * math.sin(float(tok_a[IDX_PHI])) + w2 * math.sin(float(tok_b[IDX_PHI])),
        w1 * math.cos(float(tok_a[IDX_PHI])) + w2 * math.cos(float(tok_b[IDX_PHI])),
    )
    return out.astype(np.float32)


def apply_corruption_batch(
    tokens: np.ndarray,
    mask: np.ndarray,
    kind: str,
    severity: float,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray]:
    out_tok = tokens.copy()
    out_mask = mask.copy()
    n, t, _ = out_tok.shape
    sev = float(max(0.0, severity))

    if kind == "pt_noise":
        for i in range(n):
            idx = np.where(out_mask[i])[0]
            if idx.size == 0:
                continue
            noise = rng.normal(loc=1.0, scale=sev, size=idx.size).astype(np.float32)
            noise = np.clip(noise, 0.25, 2.0)
            out_tok[i, idx, IDX_PT] = np.maximum(out_tok[i, idx, IDX_PT] * noise, 1e-8)
            out_tok[i, idx, IDX_E] = np.maximum(out_tok[i, idx, IDX_E] * noise, 1e-8)
        return out_tok, out_mask

    if kind == "eta_phi_jitter":
        for i in range(n):
            idx = np.where(out_mask[i])[0]
            if idx.size == 0:
                continue
            out_tok[i, idx, IDX_ETA] = np.clip(
                out_tok[i, idx, IDX_ETA] + rng.normal(0.0, sev, size=idx.size).astype(np.float32),
                -5.0,
                5.0,
            )
            out_tok[i, idx, IDX_PHI] = _wrap_phi_np(
                out_tok[i, idx, IDX_PHI] + rng.normal(0.0, sev, size=idx.size).astype(np.float32)
            )
        return out_tok, out_mask

    if kind == "dropout":
        p_drop = np.clip(sev, 0.0, 0.95)
        for i in range(n):
            idx = np.where(out_mask[i])[0]
            if idx.size == 0:
                continue
            drop = rng.rand(idx.size) < p_drop
            if drop.all():
                keep_local = int(np.argmax(out_tok[i, idx, IDX_PT]))
                drop[keep_local] = False
            to_drop = idx[drop]
            out_mask[i, to_drop] = False
            out_tok[i, to_drop] = 0.0
        return out_tok, out_mask

    if kind == "merge":
        p_merge = np.clip(sev, 0.0, 0.9)
        for i in range(n):
            idx = np.where(out_mask[i])[0]
            if idx.size < 2:
                continue
            idx_list = list(idx.astype(int))
            rng.shuffle(idx_list)
            num_pairs = len(idx_list) // 2
            n_merge = rng.binomial(num_pairs, p_merge)
            for m in range(n_merge):
                a = idx_list[2 * m]
                b = idx_list[2 * m + 1]
                if not out_mask[i, a] or not out_mask[i, b]:
                    continue
                out_tok[i, a] = _merge_two_tokens(out_tok[i, a], out_tok[i, b])
                out_mask[i, b] = False
                out_tok[i, b] = 0.0
        return out_tok, out_mask

    if kind == "global_scale":
        for i in range(n):
            idx = np.where(out_mask[i])[0]
            if idx.size == 0:
                continue
            s = float(np.clip(1.0 + rng.normal(0.0, sev), 0.25, 2.0))
            out_tok[i, idx, IDX_PT] = np.maximum(out_tok[i, idx, IDX_PT] * s, 1e-8)
            out_tok[i, idx, IDX_E] = np.maximum(out_tok[i, idx, IDX_E] * s, 1e-8)
        return out_tok, out_mask

    raise ValueError(f"Unknown corruption kind: {kind}")


def evaluate_probs(
    model: torch.nn.Module,
    feat: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Dict[str, object]:
    ds = JetDataset(feat=feat, mask=mask, label=labels)
    dl = make_loader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    logits_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    with torch.no_grad():
        for b in dl:
            x = b["feat"].to(device, non_blocking=True)
            m = b["mask"].to(device, non_blocking=True)
            y = b["label"].to(device, non_blocking=True)
            logits = model(x, m)
            logits_all.append(logits.cpu().numpy())
            labels_all.append(y.cpu().numpy())
    logits_np = np.concatenate(logits_all, axis=0)
    y_np = np.concatenate(labels_all, axis=0)
    probs_np = torch.softmax(torch.tensor(logits_np), dim=1).numpy()
    pred_np = np.argmax(probs_np, axis=1)
    acc = float((pred_np == y_np).mean())
    auc = macro_auc_ovr(y_np, probs_np, probs_np.shape[1])
    return {
        "logits": logits_np,
        "probs": probs_np,
        "labels": y_np,
        "pred": pred_np,
        "acc": acc,
        "auc_macro_ovr": auc,
        "all_finite_logits": bool(np.isfinite(logits_np).all()),
        "all_finite_probs": bool(np.isfinite(probs_np).all()),
    }


def pick_stratified_subset(labels: np.ndarray, total: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    classes = sorted(np.unique(labels).tolist())
    n_classes = len(classes)
    if total <= 0:
        return np.arange(len(labels), dtype=np.int64)
    base = total // n_classes
    rem = total % n_classes
    chunks: List[np.ndarray] = []
    for i, c in enumerate(classes):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        take = min(len(idx), base + (1 if i < rem else 0))
        if take > 0:
            chunks.append(idx[:take])
    if not chunks:
        return np.arange(len(labels), dtype=np.int64)
    out = np.concatenate(chunks, axis=0)
    rng.shuffle(out)
    return out.astype(np.int64)


def _true_class_logit(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return logits.gather(1, y.view(-1, 1)).sum()


def attribution_input_grad(
    model: torch.nn.Module,
    feat: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    n = feat.shape[0]
    out = np.zeros((n, feat.shape[1]), dtype=np.float32)
    for s in tqdm(range(0, n, batch_size), desc="Attr/InputGrad"):
        e = min(n, s + batch_size)
        x = torch.tensor(feat[s:e], dtype=torch.float32, device=device, requires_grad=True)
        m = torch.tensor(mask[s:e], dtype=torch.bool, device=device)
        y = torch.tensor(labels[s:e], dtype=torch.long, device=device)
        logits = model(x, m)
        obj = _true_class_logit(logits, y)
        g = torch.autograd.grad(obj, x, retain_graph=False, create_graph=False)[0]
        score = g.abs().sum(dim=2)
        score = torch.where(m, score, torch.zeros_like(score))
        out[s:e] = score.detach().cpu().numpy().astype(np.float32)
    return out


def attribution_integrated_gradients(
    model: torch.nn.Module,
    feat: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    ig_steps: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    n = feat.shape[0]
    out = np.zeros((n, feat.shape[1]), dtype=np.float32)
    steps = max(2, int(ig_steps))
    alphas = torch.linspace(0.0, 1.0, steps=steps, device=device)
    for s in tqdm(range(0, n, batch_size), desc="Attr/IG"):
        e = min(n, s + batch_size)
        x = torch.tensor(feat[s:e], dtype=torch.float32, device=device)
        m = torch.tensor(mask[s:e], dtype=torch.bool, device=device)
        y = torch.tensor(labels[s:e], dtype=torch.long, device=device)
        base = torch.zeros_like(x)
        grad_sum = torch.zeros_like(x)
        for a in alphas:
            xa = (base + a * (x - base)).detach().requires_grad_(True)
            logits = model(xa, m)
            obj = _true_class_logit(logits, y)
            g = torch.autograd.grad(obj, xa, retain_graph=False, create_graph=False)[0]
            grad_sum = grad_sum + g
        ig = (x - base) * (grad_sum / float(steps))
        score = ig.abs().sum(dim=2)
        score = torch.where(m, score, torch.zeros_like(score))
        out[s:e] = score.detach().cpu().numpy().astype(np.float32)
    return out


def attribution_smoothgrad(
    model: torch.nn.Module,
    feat: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    sg_samples: int,
    sg_sigma: float,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    n = feat.shape[0]
    out = np.zeros((n, feat.shape[1]), dtype=np.float32)
    k = max(1, int(sg_samples))
    sigma = float(max(1e-6, sg_sigma))
    for s in tqdm(range(0, n, batch_size), desc="Attr/SmoothGrad"):
        e = min(n, s + batch_size)
        x0 = torch.tensor(feat[s:e], dtype=torch.float32, device=device)
        m = torch.tensor(mask[s:e], dtype=torch.bool, device=device)
        y = torch.tensor(labels[s:e], dtype=torch.long, device=device)
        grad_acc = torch.zeros_like(x0)
        for _ in range(k):
            noise = torch.randn_like(x0) * sigma
            x = (x0 + noise).detach().requires_grad_(True)
            logits = model(x, m)
            obj = _true_class_logit(logits, y)
            g = torch.autograd.grad(obj, x, retain_graph=False, create_graph=False)[0]
            grad_acc = grad_acc + g.abs()
        score = (grad_acc / float(k)).sum(dim=2)
        score = torch.where(m, score, torch.zeros_like(score))
        out[s:e] = score.detach().cpu().numpy().astype(np.float32)
    return out


def build_remove_mask(
    attr: np.ndarray,
    valid_mask: np.ndarray,
    frac: float,
    rng: np.random.RandomState,
    targeted: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    n, t = attr.shape
    remove = np.zeros((n, t), dtype=bool)
    k_used = np.zeros((n,), dtype=np.int64)
    for i in range(n):
        idx = np.where(valid_mask[i])[0]
        if idx.size == 0:
            continue
        k = int(round(float(frac) * float(idx.size)))
        k = max(1, min(k, idx.size))
        k_used[i] = k
        if targeted:
            sel_local = np.argsort(-attr[i, idx])[:k]
            sel = idx[sel_local]
        else:
            sel = rng.choice(idx, size=k, replace=False)
        remove[i, sel] = True
    return remove, k_used


def apply_remove_mask(feat: np.ndarray, mask: np.ndarray, remove: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    out_feat = feat.copy()
    out_mask = mask.copy()
    out_mask[remove] = False
    out_feat[remove] = 0.0
    out_feat[~out_mask] = 0.0
    return out_feat, out_mask


@dataclass
class TrainOutputs:
    model: torch.nn.Module
    history: List[Dict[str, float]]
    best_val_metric_seen: float
    best_epoch: int


def train_clean_baseline(
    feat_tr: np.ndarray,
    mask_tr: np.ndarray,
    y_tr: np.ndarray,
    feat_va: np.ndarray,
    mask_va: np.ndarray,
    y_va: np.ndarray,
    args: argparse.Namespace,
    class_names: Sequence[str],
    device: torch.device,
) -> TrainOutputs:
    ds_tr = JetDataset(feat=feat_tr, mask=mask_tr, label=y_tr)
    ds_va = JetDataset(feat=feat_va, mask=mask_va, label=y_va)
    dl_tr = make_loader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dl_va = make_loader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = JetClassTransformer(
        input_dim=int(feat_tr.shape[-1]),
        n_classes=int(len(class_names)),
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
    best_metric = float("-inf")
    best_epoch = 0
    wait = 0
    best_state: Dict[str, torch.Tensor] | None = None

    for ep in range(1, int(args.epochs) + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        for b in dl_tr:
            x = b["feat"].to(device, non_blocking=True)
            m = b["mask"].to(device, non_blocking=True)
            y = b["label"].to(device, non_blocking=True)
            opt.zero_grad()
            logits = model(x, m)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bs = int(y.shape[0])
            total_loss += float(loss.item()) * bs
            total_n += bs

        val_pack = evaluate_probs(
            model=model,
            feat=feat_va,
            mask=mask_va,
            labels=y_va,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        va = eval_metrics(
            y_true=val_pack["labels"],
            probs=val_pack["probs"],
            class_names=class_names,
            background_class=args.background_class,
            target_class=args.target_class,
        )
        metric = float(va["auc_macro_ovr"]) if np.isfinite(float(va["auc_macro_ovr"])) else float(va["acc"])
        improved = metric > best_metric
        if improved:
            best_metric = metric
            best_epoch = ep
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        sch.step()

        row = {
            "epoch": float(ep),
            "train_loss": float(total_loss / max(1, total_n)),
            "val_acc": float(va["acc"]),
            "val_auc_macro_ovr": float(va["auc_macro_ovr"]),
            "val_signal_vs_bg_fpr50": float(va["signal_vs_bg_fpr50"]),
            "val_target_vs_bg_ratio_fpr50": float(va["target_vs_bg_ratio_fpr50"]),
            "best_metric_so_far": float(best_metric),
        }
        hist.append(row)
        print(
            f"[train] ep={ep} train_loss={row['train_loss']:.4f} "
            f"val_acc={row['val_acc']:.4f} val_auc={row['val_auc_macro_ovr']:.4f} "
            f"best={best_metric:.4f}"
        )
        if wait >= int(args.patience):
            print(f"[train] early stopping at epoch {ep}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainOutputs(model=model, history=hist, best_val_metric_seen=float(best_metric), best_epoch=int(best_epoch))


def permutation_best_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    conf = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    cost = conf.max() - conf
    r, c = linear_sum_assignment(cost)
    best = conf[r, c].sum()
    return float(best / max(1, conf.sum()))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reimplement preliminary JetClass studies (single seed)")
    p.add_argument(
        "--data_dir",
        type=Path,
        default=Path("/home/ryan/ComputerScience/ATLAS/HLT_Reco/ATLAS-top-tagging-open-data/data/jetclass_part0"),
    )
    p.add_argument("--output_root", type=Path, default=PROJECT_ROOT / "restart_studies" / "results")
    p.add_argument("--run_name", type=str, default="prelim_reimpl_seed52")
    p.add_argument("--seed", type=int, default=52)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--num_workers", type=int, default=2)

    p.add_argument("--feature_mode", type=str, default="full", choices=["kin", "kinpid", "full"])
    p.add_argument("--max_constits", type=int, default=128)
    p.add_argument("--train_files_per_class", type=int, default=8)
    p.add_argument("--val_files_per_class", type=int, default=1)
    p.add_argument("--test_files_per_class", type=int, default=1)
    p.add_argument("--shuffle_files", action="store_true", default=False)

    p.add_argument("--n_train_jets", type=int, default=12000)
    p.add_argument("--n_val_jets", type=int, default=3000)
    p.add_argument("--n_test_jets", type=int, default=12000)

    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--embed_dim", type=int, default=96)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--ff_dim", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--target_class", type=str, default="HToBB")
    p.add_argument("--background_class", type=str, default="ZJetsToNuNu")

    p.add_argument(
        "--corruptions",
        type=str,
        default=(
            "pt_noise:0.03,pt_noise:0.06,"
            "eta_phi_jitter:0.02,eta_phi_jitter:0.05,"
            "dropout:0.05,dropout:0.10,"
            "merge:0.10,merge:0.20,"
            "global_scale:0.03"
        ),
    )

    p.add_argument("--explain_subset_size", type=int, default=2000)
    p.add_argument("--explain_batch_size", type=int, default=128)
    p.add_argument("--mask_fracs", type=str, default="0.02,0.05,0.10,0.20")
    p.add_argument("--ig_steps", type=int, default=12)
    p.add_argument("--smoothgrad_samples", type=int, default=8)
    p.add_argument("--smoothgrad_sigma", type=float, default=0.10)
    p.add_argument("--random_mask_repeats", type=int, default=3)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    run_dir = (args.output_root / args.run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True, default=str)

    device = torch.device(args.device if str(args.device).startswith("cpu") or torch.cuda.is_available() else "cpu")
    print(f"[info] run_dir={run_dir}")
    print(f"[info] device={device}")

    files_by_class = collect_files_by_class(args.data_dir.resolve())
    class_names = sorted(files_by_class.keys())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    n_classes = len(class_names)
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
    split_overlap = {
        "train_val_overlap": int(len(set(sum([[str(x) for x in v] for v in tr_files.values()], [])) &
                                     set(sum([[str(x) for x in v] for v in va_files.values()], [])))),
        "train_test_overlap": int(len(set(sum([[str(x) for x in v] for v in tr_files.values()], [])) &
                                      set(sum([[str(x) for x in v] for v in te_files.values()], [])))),
        "val_test_overlap": int(len(set(sum([[str(x) for x in v] for v in va_files.values()], [])) &
                                    set(sum([[str(x) for x in v] for v in te_files.values()], [])))),
    }

    print("[info] loading train split")
    tr_tok, tr_mask, tr_y = load_split(
        tr_files,
        n_total=args.n_train_jets,
        max_constits=args.max_constits,
        class_to_idx=class_to_idx,
        seed=args.seed + 101,
    )
    print("[info] loading val split")
    va_tok, va_mask, va_y = load_split(
        va_files,
        n_total=args.n_val_jets,
        max_constits=args.max_constits,
        class_to_idx=class_to_idx,
        seed=args.seed + 202,
    )
    print("[info] loading test split")
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

    train_out = train_clean_baseline(
        feat_tr=tr_feat,
        mask_tr=tr_mask,
        y_tr=tr_y,
        feat_va=va_feat,
        mask_va=va_mask,
        y_va=va_y,
        args=args,
        class_names=class_names,
        device=device,
    )
    model = train_out.model

    torch.save(model.state_dict(), run_dir / "clean_baseline_best.pt")
    write_csv(
        run_dir / "train_history.csv",
        rows=train_out.history,
        fieldnames=[
            "epoch",
            "train_loss",
            "val_acc",
            "val_auc_macro_ovr",
            "val_signal_vs_bg_fpr50",
            "val_target_vs_bg_ratio_fpr50",
            "best_metric_so_far",
        ],
    )

    clean_val_pack = evaluate_probs(model, va_feat, va_mask, va_y, args.batch_size, args.num_workers, device)
    clean_test_pack = evaluate_probs(model, te_feat, te_mask, te_y, args.batch_size, args.num_workers, device)

    clean_test_metrics = eval_metrics(
        y_true=clean_test_pack["labels"],
        probs=clean_test_pack["probs"],
        class_names=class_names,
        background_class=args.background_class,
        target_class=args.target_class,
    )
    clean_test_metrics["mean_entropy"] = mean_entropy(clean_test_pack["probs"])
    clean_test_metrics["mean_confidence"] = mean_confidence(clean_test_pack["probs"])
    clean_test_metrics["best_val_metric_seen"] = float(train_out.best_val_metric_seen)
    clean_test_metrics["best_epoch"] = float(train_out.best_epoch)
    clean_test_metrics["val_metric_reloaded"] = float(
        clean_val_pack["auc_macro_ovr"] if np.isfinite(clean_val_pack["auc_macro_ovr"]) else clean_val_pack["acc"]
    )
    clean_test_metrics["trainer_posthoc_metric_abs_diff"] = float(
        abs(clean_test_metrics["best_val_metric_seen"] - clean_test_metrics["val_metric_reloaded"])
    )

    with (run_dir / "clean_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(clean_test_metrics, f, indent=2, sort_keys=True)

    corruption_specs = parse_corruptions(args.corruptions)
    clean_probs = np.asarray(clean_test_pack["probs"], dtype=np.float64)
    clean_acc = float(clean_test_pack["acc"])
    clean_auc = float(clean_test_pack["auc_macro_ovr"])
    clean_entropy = mean_entropy(clean_probs)
    clean_conf = mean_confidence(clean_probs)
    clean_pred = np.argmax(clean_probs, axis=1)
    clean_class_dist = clean_probs.mean(axis=0)

    corruption_rows: List[Dict[str, object]] = []
    all_corruption_finite = True
    for idx, (kind, sev) in enumerate(corruption_specs):
        rng = np.random.RandomState(args.seed + 5000 + idx * 97)
        c_tok, c_mask = apply_corruption_batch(te_tok, te_mask, kind=kind, severity=sev, rng=rng)
        c_feat = compute_features(c_tok, c_mask, feature_mode=args.feature_mode)
        c_feat = standardize(c_feat, c_mask, mean, std)
        c_pack = evaluate_probs(model, c_feat, c_mask, te_y, args.batch_size, args.num_workers, device)
        c_probs = np.asarray(c_pack["probs"], dtype=np.float64)
        c_pred = np.argmax(c_probs, axis=1)
        c_acc = float(c_pack["acc"])
        c_auc = float(c_pack["auc_macro_ovr"])
        c_entropy = mean_entropy(c_probs)
        c_conf = mean_confidence(c_probs)
        c_class_dist = c_probs.mean(axis=0)
        row = {
            "corruption_kind": kind,
            "severity": float(sev),
            "acc_clean": float(clean_acc),
            "auc_clean": float(clean_auc),
            "acc_corrupted": float(c_acc),
            "auc_corrupted": float(c_auc),
            "delta_acc": float(clean_acc - c_acc),
            "delta_auc": float(clean_auc - c_auc),
            "entropy_shift": float(c_entropy - clean_entropy),
            "confidence_drop": float(clean_conf - c_conf),
            "top1_flip_rate": float((clean_pred != c_pred).mean()),
            "prob_l1_drift": float(np.abs(clean_probs - c_probs).sum(axis=1).mean()),
            "class_js_divergence": float(jensen_shannon_divergence(clean_class_dist, c_class_dist)),
            "all_finite_logits": bool(c_pack["all_finite_logits"]),
            "all_finite_probs": bool(c_pack["all_finite_probs"]),
        }
        all_corruption_finite = all_corruption_finite and bool(c_pack["all_finite_logits"]) and bool(c_pack["all_finite_probs"])
        corruption_rows.append(row)

    write_csv(
        run_dir / "corruption_metrics.csv",
        rows=corruption_rows,
        fieldnames=[
            "corruption_kind",
            "severity",
            "acc_clean",
            "auc_clean",
            "acc_corrupted",
            "auc_corrupted",
            "delta_acc",
            "delta_auc",
            "entropy_shift",
            "confidence_drop",
            "top1_flip_rate",
            "prob_l1_drift",
            "class_js_divergence",
            "all_finite_logits",
            "all_finite_probs",
        ],
    )

    metric_names = [
        "prob_l1_drift",
        "top1_flip_rate",
        "class_js_divergence",
        "confidence_drop",
        "entropy_shift",
    ]
    corr_rows: List[Dict[str, object]] = []
    arr_delta_auc = np.array([float(r["delta_auc"]) for r in corruption_rows], dtype=np.float64)
    arr_delta_acc = np.array([float(r["delta_acc"]) for r in corruption_rows], dtype=np.float64)
    for m in metric_names:
        arr_m = np.array([float(r[m]) for r in corruption_rows], dtype=np.float64)
        sp_auc, pr_auc = safe_corr(arr_m, arr_delta_auc)
        sp_acc, pr_acc = safe_corr(arr_m, arr_delta_acc)
        corr_rows.append(
            {
                "metric": m,
                "spearman_delta_auc": sp_auc,
                "pearson_delta_auc": pr_auc,
                "spearman_delta_acc": sp_acc,
                "pearson_delta_acc": pr_acc,
            }
        )

    write_csv(
        run_dir / "correlations.csv",
        rows=corr_rows,
        fieldnames=[
            "metric",
            "spearman_delta_auc",
            "pearson_delta_auc",
            "spearman_delta_acc",
            "pearson_delta_acc",
        ],
    )

    ranking_rows = sorted(
        [
            {
                "metric": r["metric"],
                "abs_spearman_delta_auc": float(abs(r["spearman_delta_auc"])) if np.isfinite(r["spearman_delta_auc"]) else float("nan"),
            }
            for r in corr_rows
        ],
        key=lambda x: (-x["abs_spearman_delta_auc"]) if np.isfinite(x["abs_spearman_delta_auc"]) else 1e9,
    )
    for i, rr in enumerate(ranking_rows, start=1):
        rr["rank"] = i
    write_csv(
        run_dir / "top_shift_metric_ranking.csv",
        rows=ranking_rows,
        fieldnames=["rank", "metric", "abs_spearman_delta_auc"],
    )

    explain_idx = pick_stratified_subset(te_y, total=args.explain_subset_size, seed=args.seed + 7000)
    ex_feat = te_feat[explain_idx]
    ex_mask = te_mask[explain_idx]
    ex_y = te_y[explain_idx]
    explain_label_counts = {class_names[i]: int((ex_y == i).sum()) for i in range(n_classes)}
    with (run_dir / "explain_subset_label_counts.json").open("w", encoding="utf-8") as f:
        json.dump(explain_label_counts, f, indent=2, sort_keys=True)

    ex_clean_pack = evaluate_probs(model, ex_feat, ex_mask, ex_y, args.batch_size, args.num_workers, device)
    ex_clean_probs = np.asarray(ex_clean_pack["probs"], dtype=np.float64)
    ex_clean_acc = float(ex_clean_pack["acc"])
    ex_clean_auc = float(ex_clean_pack["auc_macro_ovr"])

    attrs: Dict[str, np.ndarray] = {
        "input_gradients": attribution_input_grad(
            model=model,
            feat=ex_feat,
            mask=ex_mask,
            labels=ex_y,
            batch_size=args.explain_batch_size,
            device=device,
        ),
        "integrated_gradients": attribution_integrated_gradients(
            model=model,
            feat=ex_feat,
            mask=ex_mask,
            labels=ex_y,
            batch_size=args.explain_batch_size,
            ig_steps=args.ig_steps,
            device=device,
        ),
        "smoothgrad": attribution_smoothgrad(
            model=model,
            feat=ex_feat,
            mask=ex_mask,
            labels=ex_y,
            batch_size=args.explain_batch_size,
            sg_samples=args.smoothgrad_samples,
            sg_sigma=args.smoothgrad_sigma,
            device=device,
        ),
    }

    fracs = [float(x.strip()) for x in args.mask_fracs.split(",") if x.strip()]
    interpret_rows: List[Dict[str, object]] = []
    parity_rows: List[Dict[str, object]] = []
    rng_global = np.random.RandomState(args.seed + 8000)
    for method, attr in attrs.items():
        for frac in fracs:
            remove_t, k_t = build_remove_mask(attr, ex_mask, frac, rng_global, targeted=True)
            feat_t, mask_t = apply_remove_mask(ex_feat, ex_mask, remove_t)
            pack_t = evaluate_probs(model, feat_t, mask_t, ex_y, args.batch_size, args.num_workers, device)
            probs_t = np.asarray(pack_t["probs"], dtype=np.float64)

            targeted_prob_drop = float(
                (ex_clean_probs[np.arange(len(ex_y)), ex_y] - probs_t[np.arange(len(ex_y)), ex_y]).mean()
            )
            targeted_auc_drop = float(ex_clean_auc - float(pack_t["auc_macro_ovr"]))
            targeted_acc_drop = float(ex_clean_acc - float(pack_t["acc"]))

            rand_prob_drops: List[float] = []
            rand_auc_drops: List[float] = []
            rand_acc_drops: List[float] = []
            k_rand_means: List[float] = []
            for rr in range(int(args.random_mask_repeats)):
                rng_r = np.random.RandomState(args.seed + 8500 + rr * 97 + int(frac * 1000))
                remove_r, k_r = build_remove_mask(attr, ex_mask, frac, rng_r, targeted=False)
                feat_r, mask_r = apply_remove_mask(ex_feat, ex_mask, remove_r)
                pack_r = evaluate_probs(model, feat_r, mask_r, ex_y, args.batch_size, args.num_workers, device)
                probs_r = np.asarray(pack_r["probs"], dtype=np.float64)
                rand_prob_drops.append(
                    float((ex_clean_probs[np.arange(len(ex_y)), ex_y] - probs_r[np.arange(len(ex_y)), ex_y]).mean())
                )
                rand_auc_drops.append(float(ex_clean_auc - float(pack_r["auc_macro_ovr"])))
                rand_acc_drops.append(float(ex_clean_acc - float(pack_r["acc"])))
                k_rand_means.append(float(np.mean(k_r)))

            row = {
                "method": method,
                "mask_frac": float(frac),
                "targeted_prob_drop": float(targeted_prob_drop),
                "random_prob_drop": float(np.mean(rand_prob_drops)),
                "gap_prob_drop_target_minus_rand": float(targeted_prob_drop - np.mean(rand_prob_drops)),
                "targeted_auc_drop": float(targeted_auc_drop),
                "random_auc_drop": float(np.mean(rand_auc_drops)),
                "gap_auc_drop_target_minus_rand": float(targeted_auc_drop - np.mean(rand_auc_drops)),
                "targeted_acc_drop": float(targeted_acc_drop),
                "random_acc_drop": float(np.mean(rand_acc_drops)),
                "gap_acc_drop_target_minus_rand": float(targeted_acc_drop - np.mean(rand_acc_drops)),
                "targeted_mask_k_mean": float(np.mean(k_t)),
                "random_mask_k_mean": float(np.mean(k_rand_means)),
            }
            interpret_rows.append(row)
            parity_rows.append(
                {
                    "method": method,
                    "mask_frac": float(frac),
                    "targeted_mask_k_mean": float(np.mean(k_t)),
                    "random_mask_k_mean": float(np.mean(k_rand_means)),
                    "abs_diff": float(abs(np.mean(k_t) - np.mean(k_rand_means))),
                }
            )

    write_csv(
        run_dir / "interpretability_per_fraction.csv",
        rows=interpret_rows,
        fieldnames=[
            "method",
            "mask_frac",
            "targeted_prob_drop",
            "random_prob_drop",
            "gap_prob_drop_target_minus_rand",
            "targeted_auc_drop",
            "random_auc_drop",
            "gap_auc_drop_target_minus_rand",
            "targeted_acc_drop",
            "random_acc_drop",
            "gap_acc_drop_target_minus_rand",
            "targeted_mask_k_mean",
            "random_mask_k_mean",
        ],
    )
    write_csv(
        run_dir / "masking_parity.csv",
        rows=parity_rows,
        fieldnames=["method", "mask_frac", "targeted_mask_k_mean", "random_mask_k_mean", "abs_diff"],
    )

    method_summary_rows: List[Dict[str, object]] = []
    for method in sorted(set(r["method"] for r in interpret_rows)):
        sub = [r for r in interpret_rows if r["method"] == method]
        method_summary_rows.append(
            {
                "method": method,
                "targeted_drop": float(np.mean([float(r["targeted_prob_drop"]) for r in sub])),
                "random_drop": float(np.mean([float(r["random_prob_drop"]) for r in sub])),
                "gap_target_minus_random": float(np.mean([float(r["gap_prob_drop_target_minus_rand"]) for r in sub])),
                "auc_gap": float(np.mean([float(r["gap_auc_drop_target_minus_rand"]) for r in sub])),
                "acc_gap": float(np.mean([float(r["gap_acc_drop_target_minus_rand"]) for r in sub])),
            }
        )

    write_csv(
        run_dir / "method_effectiveness_summary.csv",
        rows=method_summary_rows,
        fieldnames=["method", "targeted_drop", "random_drop", "gap_target_minus_random", "auc_gap", "acc_gap"],
    )

    best_perm_acc = permutation_best_accuracy(clean_test_pack["labels"], clean_test_pack["pred"], n_classes)
    perm_delta = float(best_perm_acc - float(clean_test_pack["acc"]))
    class_balance_counts = {class_names[i]: int((te_y == i).sum()) for i in range(n_classes)}
    class_counts_arr = np.array(list(class_balance_counts.values()), dtype=np.int64)
    sanity = {
        "class_balance_eval_split": {
            "counts": class_balance_counts,
            "max_minus_min": int(class_counts_arr.max() - class_counts_arr.min()),
            "status": "pass" if int(class_counts_arr.max() - class_counts_arr.min()) <= 1 else "warn",
        },
        "disjoint_split_construction": {
            "overlap_counts": split_overlap,
            "status": "pass" if all(v == 0 for v in split_overlap.values()) else "fail",
        },
        "label_mapping_permutation_diagnostic": {
            "raw_accuracy": float(clean_test_pack["acc"]),
            "best_permutation_accuracy": float(best_perm_acc),
            "delta_perm": float(perm_delta),
            "status": "pass" if perm_delta <= 0.02 else "warn",
        },
        "targeted_vs_random_masking_parity": {
            "max_abs_k_mean_diff": float(max(float(r["abs_diff"]) for r in parity_rows) if parity_rows else 0.0),
            "status": "pass" if (max(float(r["abs_diff"]) for r in parity_rows) if parity_rows else 0.0) < 1e-6 else "warn",
        },
        "non_finite_output_checks": {
            "clean_logits_finite": bool(clean_test_pack["all_finite_logits"]),
            "clean_probs_finite": bool(clean_test_pack["all_finite_probs"]),
            "corruptions_all_finite": bool(all_corruption_finite),
            "status": "pass"
            if bool(clean_test_pack["all_finite_logits"]) and bool(clean_test_pack["all_finite_probs"]) and bool(all_corruption_finite)
            else "fail",
        },
        "trainer_vs_reloaded_val_consistency": {
            "best_val_metric_seen": float(train_out.best_val_metric_seen),
            "reloaded_val_metric": float(clean_test_metrics["val_metric_reloaded"]),
            "abs_diff": float(clean_test_metrics["trainer_posthoc_metric_abs_diff"]),
            "status": "pass" if float(clean_test_metrics["trainer_posthoc_metric_abs_diff"]) <= 1e-4 else "warn",
        },
    }
    with (run_dir / "sanity_checks.json").open("w", encoding="utf-8") as f:
        json.dump(sanity, f, indent=2, sort_keys=True)

    summary = {
        "run_name": args.run_name,
        "seed": int(args.seed),
        "n_train_jets": int(args.n_train_jets),
        "n_val_jets": int(args.n_val_jets),
        "n_test_jets": int(args.n_test_jets),
        "feature_mode": args.feature_mode,
        "clean_metrics": clean_test_metrics,
        "top_shift_metric_ranking": ranking_rows[:3],
        "method_effectiveness": method_summary_rows,
        "sanity_checks": sanity,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("[done] outputs written:")
    print(f"  - {run_dir / 'summary.json'}")
    print(f"  - {run_dir / 'clean_metrics.json'}")
    print(f"  - {run_dir / 'corruption_metrics.csv'}")
    print(f"  - {run_dir / 'correlations.csv'}")
    print(f"  - {run_dir / 'top_shift_metric_ranking.csv'}")
    print(f"  - {run_dir / 'interpretability_per_fraction.csv'}")
    print(f"  - {run_dir / 'method_effectiveness_summary.csv'}")
    print(f"  - {run_dir / 'sanity_checks.json'}")


if __name__ == "__main__":
    main()

