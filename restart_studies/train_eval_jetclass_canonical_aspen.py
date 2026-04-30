#!/usr/bin/env python3
from __future__ import annotations


import argparse
import csv
import importlib.util
import inspect
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

try:
    import h5py
except ModuleNotFoundError:
    h5py = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from reimplement_preliminary_studies import (              
    apply_remove_mask,
    attribution_input_grad,
    attribution_integrated_gradients,
    attribution_smoothgrad,
    build_remove_mask,
    evaluate_probs,
    mean_confidence,
    mean_entropy,
    pick_stratified_subset,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train/eval clean JetClass model with canonical loading + Aspen + interpretability."
    )
    p.add_argument(
        "--canonical_backend_py",
        type=Path,
        default=None,
        help="Path to canonical evaluate_jetclass_hlt_teacher_baseline.py (optional auto-discovery).",
    )

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
    p.add_argument(
        "--feature_preprocessing",
        type=str,
        default="canonical",
        choices=["canonical", "legacy"],
    )
    p.add_argument(
        "--class_assignment",
        type=str,
        default="canonical_labels",
        choices=["filename", "canonical_labels"],
    )
    p.add_argument("--max_constits", type=int, default=128)
    p.add_argument("--train_files_per_class", type=int, default=8)
    p.add_argument("--val_files_per_class", type=int, default=1)
    p.add_argument("--test_files_per_class", type=int, default=1)
    p.add_argument("--shuffle_files", action="store_true", default=False)
    p.add_argument("--n_train_jets", type=int, default=150000)
    p.add_argument("--n_val_jets", type=int, default=50000)
    p.add_argument("--n_test_jets", type=int, default=150000)

    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--ff_dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--target_class", type=str, default="Hbb")
    p.add_argument("--background_class", type=str, default="QCD")

    p.add_argument("--explain_subset_size", type=int, default=20000)
    p.add_argument("--explain_batch_size", type=int, default=128)
    p.add_argument("--mask_fracs", type=str, default="0.02,0.05,0.10,0.20")
    p.add_argument("--ig_steps", type=int, default=16)
    p.add_argument("--smoothgrad_samples", type=int, default=16)
    p.add_argument("--smoothgrad_sigma", type=float, default=0.10)
    p.add_argument("--random_mask_repeats", type=int, default=3)
    return p.parse_args()


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow(r)


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


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / np.clip(p.sum(), 1e-12, None)
    q = q / np.clip(q.sum(), 1e-12, None)
    m = 0.5 * (p + q)
    kl_pm = np.sum(np.where(p > 0, p * np.log(np.clip(p / np.clip(m, 1e-12, None), 1e-12, None)), 0.0))
    kl_qm = np.sum(np.where(q > 0, q * np.log(np.clip(q / np.clip(m, 1e-12, None), 1e-12, None)), 0.0))
    return float(0.5 * (kl_pm + kl_qm))


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


def sanitize_aoj_track_features(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float32).copy()
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32)


def aoj_pfcands_to_raw_tokens(
    pfcands: np.ndarray,
    max_constits: int,
    backend: ModuleType,
) -> tuple[np.ndarray, np.ndarray]:
                                                                                             
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
    raw[:, :, backend.IDX_PT] = pt
    raw[:, :, backend.IDX_ETA] = eta
    raw[:, :, backend.IDX_PHI] = phi
    raw[:, :, backend.IDX_E] = ene
    raw[:, :, backend.IDX_CHARGE] = charge
    raw[:, :, backend.IDX_D0] = sanitize_aoj_track_features(d0)
    raw[:, :, backend.IDX_D0ERR] = sanitize_aoj_track_features(d0err)
    raw[:, :, backend.IDX_DZ] = sanitize_aoj_track_features(dz)
    raw[:, :, backend.IDX_DZERR] = sanitize_aoj_track_features(dzerr)

    abs_pid = np.abs(pdgid)
    is_ele = abs_pid == 11
    is_mu = abs_pid == 13
    is_pho = abs_pid == 22
    is_ch = (~is_ele) & (~is_mu) & (~is_pho) & (np.abs(charge) > 0.0)
    is_nh = (~is_ele) & (~is_mu) & (~is_pho) & (~is_ch)

    raw[:, :, backend.IDX_PID0] = is_ch.astype(np.float32)
    raw[:, :, backend.IDX_PID1] = is_nh.astype(np.float32)
    raw[:, :, backend.IDX_PID2] = is_pho.astype(np.float32)
    raw[:, :, backend.IDX_PID3] = is_ele.astype(np.float32)
    raw[:, :, backend.IDX_PID4] = is_mu.astype(np.float32)

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


def stream_aspen_stats_canonical(
    model: torch.nn.Module,
    backend: ModuleType,
    aspen_data_dir: Path,
    glob_pattern: str,
    n_jets: int,
    chunk_jets: int,
    max_constits: int,
    feature_mode: str,
    feature_preprocessing: str,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Dict[str, object]:
    if h5py is None:
        raise RuntimeError("Missing dependency: h5py")

    files = sorted(p for p in aspen_data_dir.glob(glob_pattern) if p.is_file())
    if not files:
        raise RuntimeError(f"No Aspen files matched {aspen_data_dir}/{glob_pattern}")

    use_standardize = str(feature_preprocessing) != "canonical"

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
                tok, mask = aoj_pfcands_to_raw_tokens(arr, max_constits=max_constits, backend=backend)
                feat = backend.compute_features(
                    tok,
                    mask,
                    feature_mode=feature_mode,
                    feature_preprocessing=feature_preprocessing,
                )
                if use_standardize:
                    feat = backend.standardize(feat, mask, mean, std)

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


def _candidate_backend_paths(user_path: Path | None) -> List[Path]:
    cands: List[Path] = []
    if user_path is not None:
        cands.append(user_path.expanduser().resolve())
    cands.extend(
        [
            (PROJECT_ROOT / "restart_studies" / "backends" / "evaluate_jetclass_hlt_teacher_baseline_canonical.py").resolve(),
            (PROJECT_ROOT / "PracticeTagging" / "evaluate_jetclass_hlt_teacher_baseline.py").resolve(),
            (PROJECT_ROOT / "ATLAS-top-tagging-open-data" / "evaluate_jetclass_hlt_teacher_baseline.py").resolve(),
            Path("/home/ryreu/atlas/CompPhys_FinalProject/restart_studies/backends/evaluate_jetclass_hlt_teacher_baseline_canonical.py"),
            Path("/home/ryreu/atlas/CompPhys_FinalProject/PracticeTagging/evaluate_jetclass_hlt_teacher_baseline.py"),
            Path("/home/ryreu/atlas/PracticeTagging/evaluate_jetclass_hlt_teacher_baseline.py"),
            Path("/home/ryreu/atlas/ATLAS-top-tagging-open-data/evaluate_jetclass_hlt_teacher_baseline.py"),
            Path("/home/ryreu/atlas/CompPhys_FinalProject/ATLAS-top-tagging-open-data/evaluate_jetclass_hlt_teacher_baseline.py"),
            Path("/home/ryan/ComputerScience/ATLAS/HLT_Reco/ATLAS-top-tagging-open-data/evaluate_jetclass_hlt_teacher_baseline.py"),
        ]
    )
    return cands


def load_canonical_backend(user_path: Path | None) -> tuple[ModuleType, Path]:
    req_attrs = [
        "collect_files_by_class",
        "split_files_by_class",
        "load_split",
        "compute_features",
        "get_mean_std",
        "standardize",
        "JetClassTransformer",
        "eval_metrics",
        "CANONICAL_CLASS_ORDER",
        "CLASS_NAME_ALIASES",
        "IDX_PT",
        "IDX_ETA",
        "IDX_PHI",
        "IDX_E",
        "IDX_CHARGE",
        "IDX_PID0",
        "IDX_PID1",
        "IDX_PID2",
        "IDX_PID3",
        "IDX_PID4",
        "IDX_D0",
        "IDX_D0ERR",
        "IDX_DZ",
        "IDX_DZERR",
    ]

    last_error = "No candidate evaluated."
    for path in _candidate_backend_paths(user_path):
        if not path.is_file():
            continue
        try:
            module_name = f"canonical_eval_backend_{abs(hash(str(path)))}"
            spec = importlib.util.spec_from_file_location(module_name, str(path))
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
                                                                                
                                              
            sys.modules[module_name] = mod
            spec.loader.exec_module(mod)                            

            missing = [a for a in req_attrs if not hasattr(mod, a)]
            if missing:
                last_error = f"{path}: missing attrs {missing}"
                continue

            sig_load = inspect.signature(mod.load_split)
            if "class_assignment" not in sig_load.parameters:
                last_error = f"{path}: load_split lacks class_assignment keyword (not canonical-enabled)."
                continue
            sig_feat = inspect.signature(mod.compute_features)
            if "feature_preprocessing" not in sig_feat.parameters:
                last_error = f"{path}: compute_features lacks feature_preprocessing keyword."
                continue
            return mod, path
        except Exception as exc:                    
            try:
                sys.modules.pop(module_name, None)
            except Exception:
                pass
            last_error = f"{path}: {exc}"
            continue

    raise RuntimeError(
        "Could not load canonical JetClass backend. "
        "Pass --canonical_backend_py explicitly. "
        f"Last error: {last_error}"
    )


def normalize_class_name(name: str, backend: ModuleType) -> str:
    alias: Dict[str, str] = dict(getattr(backend, "CLASS_NAME_ALIASES", {}))
    return str(alias.get(name, name))


def warmup_cosine_lambda(epoch: int, warmup_epochs: int, total_epochs: int) -> float:
    if epoch < warmup_epochs:
        return float(epoch + 1) / float(max(1, warmup_epochs))
    x = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
    return 0.5 * (1.0 + math.cos(math.pi * x))


def train_clean_model(
    *,
    model: torch.nn.Module,
    feat_tr: np.ndarray,
    mask_tr: np.ndarray,
    y_tr: np.ndarray,
    feat_va: np.ndarray,
    mask_va: np.ndarray,
    y_va: np.ndarray,
    class_names: Sequence[str],
    target_class: str,
    background_class: str,
    args: argparse.Namespace,
    backend: ModuleType,
    device: torch.device,
) -> tuple[torch.nn.Module, List[Dict[str, float]], Dict[str, float]]:
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lr_lambda=lambda ep: warmup_cosine_lambda(ep, args.warmup_epochs, args.epochs),
    )

    n = int(len(y_tr))
    idx_all = np.arange(n, dtype=np.int64)
    best_state = None
    best_metric = float("-inf")
    best_epoch = -1
    wait = 0
    hist: List[Dict[str, float]] = []

    for ep in range(1, int(args.epochs) + 1):
        np.random.shuffle(idx_all)
        model.train()
        tot_loss = 0.0
        tot_n = 0
        bs = int(args.batch_size)
        for s in range(0, n, bs):
            e = min(n, s + bs)
            bid = idx_all[s:e]
            x = torch.tensor(feat_tr[bid], dtype=torch.float32, device=device)
            m = torch.tensor(mask_tr[bid], dtype=torch.bool, device=device)
            y = torch.tensor(y_tr[bid], dtype=torch.long, device=device)
            opt.zero_grad(set_to_none=True)
            logits = model(x, m)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bsz = int(y.shape[0])
            tot_loss += float(loss.item()) * bsz
            tot_n += bsz

        sch.step()

        va_pack = evaluate_probs(
            model=model,
            feat=feat_va,
            mask=mask_va,
            labels=y_va,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        va_metrics = backend.eval_metrics(
            y_true=va_pack["labels"],
            probs=va_pack["probs"],
            class_names=class_names,
            background_class=background_class,
            target_class=target_class,
        )
        val_auc = float(va_metrics["auc_macro_ovr"])
        val_acc = float(va_metrics["acc"])
        metric = val_auc if np.isfinite(val_auc) else val_acc

        row = {
            "epoch": float(ep),
            "train_loss": float(tot_loss / max(1, tot_n)),
            "val_acc": float(val_acc),
            "val_auc_macro_ovr": float(val_auc),
            "val_target_vs_bg_ratio_fpr50": float(va_metrics["target_vs_bg_ratio_fpr50"]),
        }
        hist.append(row)
        print(
            f"[train] ep={ep} train_loss={row['train_loss']:.4f} "
            f"val_acc={row['val_acc']:.4f} val_auc={row['val_auc_macro_ovr']:.4f}"
        )

        if np.isfinite(metric) and metric > best_metric:
            best_metric = float(metric)
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if wait >= int(args.patience):
            print(f"[train] early-stop at ep={ep} (patience={args.patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    best_info = {"best_epoch": float(best_epoch), "best_metric": float(best_metric)}
    return model, hist, best_info


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    run_dir = (args.output_root / args.run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] run_dir={run_dir}")

    backend, backend_path = load_canonical_backend(args.canonical_backend_py)
    print(f"[info] canonical_backend={backend_path}")

    target_class = normalize_class_name(str(args.target_class), backend)
    background_class = normalize_class_name(str(args.background_class), backend)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"[info] device={device}")

    files_by_class = backend.collect_files_by_class(args.data_dir.resolve())
    if str(args.class_assignment) == "canonical_labels":
        class_names = list(backend.CANONICAL_CLASS_ORDER)
    else:
        class_names = sorted(files_by_class.keys())
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    print("[info] classes:")
    for c in class_names:
        if c in files_by_class:
            print(f"  - {c}: {len(files_by_class[c])} files")
        else:
            print(f"  - {c}: label_* branch")

    tr_files, va_files, te_files = backend.split_files_by_class(
        files_by_class,
        n_train=int(args.train_files_per_class),
        n_val=int(args.val_files_per_class),
        n_test=int(args.test_files_per_class),
        shuffle=bool(args.shuffle_files),
        seed=int(args.seed),
    )

    print("[info] loading train split")
    tr_tok, tr_mask, tr_y = backend.load_split(
        tr_files,
        n_total=int(args.n_train_jets),
        max_constits=int(args.max_constits),
        class_to_idx=class_to_idx,
        seed=int(args.seed) + 101,
        class_assignment=str(args.class_assignment),
    )
    print("[info] loading val split")
    va_tok, va_mask, va_y = backend.load_split(
        va_files,
        n_total=int(args.n_val_jets),
        max_constits=int(args.max_constits),
        class_to_idx=class_to_idx,
        seed=int(args.seed) + 202,
        class_assignment=str(args.class_assignment),
    )
    print("[info] loading test split")
    te_tok, te_mask, te_y = backend.load_split(
        te_files,
        n_total=int(args.n_test_jets),
        max_constits=int(args.max_constits),
        class_to_idx=class_to_idx,
        seed=int(args.seed) + 303,
        class_assignment=str(args.class_assignment),
    )
    print(f"[info] loaded jets train={len(tr_y)} val={len(va_y)} test={len(te_y)}")

    tr_feat = backend.compute_features(
        tr_tok,
        tr_mask,
        feature_mode=args.feature_mode,
        feature_preprocessing=args.feature_preprocessing,
    )
    va_feat = backend.compute_features(
        va_tok,
        va_mask,
        feature_mode=args.feature_mode,
        feature_preprocessing=args.feature_preprocessing,
    )
    te_feat = backend.compute_features(
        te_tok,
        te_mask,
        feature_mode=args.feature_mode,
        feature_preprocessing=args.feature_preprocessing,
    )

    if str(args.feature_preprocessing) == "canonical":
        mean = np.zeros((tr_feat.shape[-1],), dtype=np.float32)
        std = np.ones((tr_feat.shape[-1],), dtype=np.float32)
        standardization_mode = "canonical_manual_fixed"
    else:
        mean, std = backend.get_mean_std(tr_feat, tr_mask, np.arange(len(tr_y)))
        tr_feat = backend.standardize(tr_feat, tr_mask, mean, std)
        va_feat = backend.standardize(va_feat, va_mask, mean, std)
        te_feat = backend.standardize(te_feat, te_mask, mean, std)
        standardization_mode = "learned_train_split"

    if target_class not in class_to_idx:
        raise ValueError(f"target_class='{target_class}' not in class_names={class_names}")
    if background_class not in class_to_idx:
        raise ValueError(f"background_class='{background_class}' not in class_names={class_names}")

    model = backend.JetClassTransformer(
        input_dim=int(tr_feat.shape[-1]),
        n_classes=int(len(class_names)),
        embed_dim=int(args.embed_dim),
        num_heads=int(args.num_heads),
        num_layers=int(args.num_layers),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
    ).to(device)

    model, train_hist, best_info = train_clean_model(
        model=model,
        feat_tr=tr_feat,
        mask_tr=tr_mask,
        y_tr=tr_y,
        feat_va=va_feat,
        mask_va=va_mask,
        y_va=va_y,
        class_names=class_names,
        target_class=target_class,
        background_class=background_class,
        args=args,
        backend=backend,
        device=device,
    )

    torch.save(model.state_dict(), run_dir / "clean_baseline_best.pt")
    write_csv(
        run_dir / "train_history.csv",
        rows=train_hist,
        fieldnames=["epoch", "train_loss", "val_acc", "val_auc_macro_ovr", "val_target_vs_bg_ratio_fpr50"],
    )

    clean_pack = evaluate_probs(
        model=model,
        feat=te_feat,
        mask=te_mask,
        labels=te_y,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    clean_metrics = backend.eval_metrics(
        y_true=clean_pack["labels"],
        probs=clean_pack["probs"],
        class_names=class_names,
        background_class=background_class,
        target_class=target_class,
    )
    clean_metrics["mean_entropy"] = mean_entropy(np.asarray(clean_pack["probs"], dtype=np.float64))
    clean_metrics["mean_confidence"] = mean_confidence(np.asarray(clean_pack["probs"], dtype=np.float64))
    clean_metrics["standardization_mode"] = standardization_mode
    with (run_dir / "clean_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(clean_metrics, f, indent=2, sort_keys=True)

    clean_stats = probs_to_stats(np.asarray(clean_pack["probs"], dtype=np.float64))
    aspen = stream_aspen_stats_canonical(
        model=model,
        backend=backend,
        aspen_data_dir=args.aspen_data_dir.resolve(),
        glob_pattern=args.aspen_glob,
        n_jets=int(args.aspen_n_jets),
        chunk_jets=int(args.aspen_chunk_jets),
        max_constits=int(args.max_constits),
        feature_mode=str(args.feature_mode),
        feature_preprocessing=str(args.feature_preprocessing),
        mean=mean,
        std=std,
        batch_size=int(args.batch_size),
        device=device,
    )
    aspen_stats = aspen["stats"]
    aspen_shift = distributional_shift_metrics(clean_stats, aspen_stats)
    with (run_dir / "aspen_shift_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
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
            },
            f,
            indent=2,
            sort_keys=True,
        )

    explain_idx = pick_stratified_subset(te_y, total=args.explain_subset_size, seed=args.seed + 7000)
    ex_feat = te_feat[explain_idx]
    ex_mask = te_mask[explain_idx]
    ex_y = te_y[explain_idx]
    explain_label_counts = {class_names[i]: int((ex_y == i).sum()) for i in range(len(class_names))}
    with (run_dir / "explain_subset_label_counts.json").open("w", encoding="utf-8") as f:
        json.dump(explain_label_counts, f, indent=2, sort_keys=True)

    ex_clean_pack = evaluate_probs(
        model=model,
        feat=ex_feat,
        mask=ex_mask,
        labels=ex_y,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
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

    fracs = [float(x.strip()) for x in str(args.mask_fracs).split(",") if x.strip()]
    interpret_rows: List[Dict[str, object]] = []
    method_summary_rows: List[Dict[str, object]] = []
    rng_global = np.random.RandomState(args.seed + 8000)

    for method, attr in attrs.items():
        method_rows: List[Dict[str, object]] = []
        for frac in fracs:
            remove_t, k_t = build_remove_mask(attr, ex_mask, frac, rng_global, targeted=True)
            feat_t, mask_t = apply_remove_mask(ex_feat, ex_mask, remove_t)
            pack_t = evaluate_probs(
                model=model,
                feat=feat_t,
                mask=mask_t,
                labels=ex_y,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
            )
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
                pack_r = evaluate_probs(
                    model=model,
                    feat=feat_r,
                    mask=mask_r,
                    labels=ex_y,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    device=device,
                )
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
            method_rows.append(row)

        method_summary_rows.append(
            {
                "method": method,
                "targeted_drop": float(np.mean([float(r["targeted_prob_drop"]) for r in method_rows])),
                "random_drop": float(np.mean([float(r["random_prob_drop"]) for r in method_rows])),
                "gap_target_minus_random": float(np.mean([float(r["gap_prob_drop_target_minus_rand"]) for r in method_rows])),
                "auc_gap": float(np.mean([float(r["gap_auc_drop_target_minus_rand"]) for r in method_rows])),
                "acc_gap": float(np.mean([float(r["gap_acc_drop_target_minus_rand"]) for r in method_rows])),
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
        run_dir / "method_effectiveness_summary.csv",
        rows=method_summary_rows,
        fieldnames=["method", "targeted_drop", "random_drop", "gap_target_minus_random", "auc_gap", "acc_gap"],
    )

    summary = {
        "run_name": str(args.run_name),
        "seed": int(args.seed),
        "canonical_backend": str(backend_path),
        "feature_mode": str(args.feature_mode),
        "feature_preprocessing": str(args.feature_preprocessing),
        "class_assignment": str(args.class_assignment),
        "class_names": list(class_names),
        "target_class": str(target_class),
        "background_class": str(background_class),
        "standardization_mode": standardization_mode,
        "best_epoch": int(best_info["best_epoch"]),
        "clean_test": {
            "acc": float(clean_metrics["acc"]),
            "auc_macro_ovr": float(clean_metrics["auc_macro_ovr"]),
            "target_vs_bg_ratio_fpr50": float(clean_metrics["target_vs_bg_ratio_fpr50"]),
        },
        "aspen_shift_metrics": {k: float(v) for k, v in aspen_shift.items()},
        "aspen_n_jets_used": int(aspen["n_jets_used"]),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with (run_dir / "config_used.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True, default=str)

    print("[done] outputs written:")
    print(" ", run_dir / "summary.json")
    print(" ", run_dir / "clean_metrics.json")
    print(" ", run_dir / "aspen_shift_metrics.json")
    print(" ", run_dir / "interpretability_per_fraction.csv")
    print(" ", run_dir / "method_effectiveness_summary.csv")
    print(" ", run_dir / "clean_baseline_best.pt")


if __name__ == "__main__":
    main()
