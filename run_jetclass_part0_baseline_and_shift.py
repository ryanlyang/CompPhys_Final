#!/usr/bin/env python3

from __future__ import annotations

"""
Train a Particle Transformer on a flat JetClass subset (e.g. jetclass_part0),
then evaluate clean + corrupted test metrics and correlation against macro AUC.

This script intentionally reuses `weaver` for training, then performs post-hoc
evaluation in Python so we can run controlled corruption studies.
"""

import argparse
import csv
import glob
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    import awkward as ak
except ModuleNotFoundError:
    ak = None

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    import vector
except ModuleNotFoundError:
    vector = None

try:
    import uproot  # noqa: F401
except ModuleNotFoundError:
    uproot = None

if vector is not None and ak is not None:
    vector.register_awkward()


CLASS_NAMES = [
    "HToBB",
    "HToCC",
    "HToGG",
    "HToWW2Q1L",
    "HToWW4Q",
    "TTBar",
    "TTBarLep",
    "WToQQ",
    "ZToQQ",
    "ZJetsToNuNu",
]

LABEL_NAMES = [
    "label_QCD",
    "label_Hbb",
    "label_Hcc",
    "label_Hgg",
    "label_H4q",
    "label_Hqql",
    "label_Zqq",
    "label_Wqq",
    "label_Tbqq",
    "label_Tbl",
]

FEATURE_DIMS = {"kin": 7, "kinpid": 13, "full": 17}
CORRUPTION_TYPES = ("gaussian_noise", "dropout", "masking", "eta_phi_jitter")


@dataclass
class InputArrays:
    points: np.ndarray
    features: np.ndarray
    vectors: np.ndarray
    mask: np.ndarray
    y_onehot: np.ndarray
    y_index: np.ndarray


class ParticleTransformerWrapper(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        from weaver.nn.model.ParticleTransformer import ParticleTransformer

        self.mod = ParticleTransformer(**kwargs)

    def forward(self, points, features, lorentz_vectors, mask):
        return self.mod(features, v=lorentz_vectors, mask=mask)


def parse_index_spec(spec: str) -> List[int]:
    out: List[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo_s, hi_s = chunk.split("-", 1)
            lo = int(lo_s)
            hi = int(hi_s)
            if hi < lo:
                raise ValueError(f"Invalid range '{chunk}' in index spec '{spec}'")
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(chunk))
    return sorted(set(out))


def parse_float_list(spec: str) -> List[float]:
    vals = [float(x.strip()) for x in spec.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"Expected a non-empty float list, got '{spec}'")
    return vals


def resolve_weaver_command() -> List[str]:
    weaver_bin = shutil.which("weaver")
    if weaver_bin is not None:
        return [weaver_bin]

    # Fallback when console scripts are not on PATH but module is installed.
    if importlib.util.find_spec("weaver.train") is not None:
        return [sys.executable, "-m", "weaver.train"]

    raise RuntimeError(
        "Cannot find `weaver` executable nor `weaver.train` module.\n"
        "Install it with:\n"
        "  pip install 'weaver-core>=0.4'\n"
        "and ensure either `weaver` is on PATH or the Python module is available."
    )


def _safe_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def prepare_split_dirs(
    dataset_dir: Path,
    split_dir: Path,
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    test_indices: Sequence[int],
) -> Dict[str, Path]:
    split_paths = {k: split_dir / k for k in ("train", "val", "test")}
    for p in split_paths.values():
        p.mkdir(parents=True, exist_ok=True)

    for cls in CLASS_NAMES:
        files = sorted(dataset_dir.glob(f"{cls}_*.root"))
        if not files:
            raise FileNotFoundError(f"No files found for class '{cls}' in {dataset_dir}")

        def _pick(indices: Sequence[int]) -> List[Path]:
            picked: List[Path] = []
            for idx in indices:
                if idx < 0 or idx >= len(files):
                    raise IndexError(
                        f"Class '{cls}' has {len(files)} files, but requested index {idx}"
                    )
                picked.append(files[idx])
            return picked

        picked = {
            "train": _pick(train_indices),
            "val": _pick(val_indices),
            "test": _pick(test_indices),
        }
        for split_name, split_files in picked.items():
            for src in split_files:
                _safe_symlink(src, split_paths[split_name] / src.name)
    return split_paths


def build_weaver_command(
    args: argparse.Namespace, split_paths: Dict[str, Path], weaver_cmd: Sequence[str]
) -> List[str]:
    cmd: List[str] = [*weaver_cmd, "--data-train"]
    for cls in CLASS_NAMES:
        cmd.append(f"{cls}:{split_paths['train']}/{cls}_*.root")

    cmd.extend(["--data-val", f"{split_paths['val']}/*.root"])
    if args.include_weaver_test:
        cmd.append("--data-test")
        for cls in CLASS_NAMES:
            cmd.append(f"{cls}:{split_paths['test']}/{cls}_*.root")

    model_prefix = Path(args.output_dir) / "training" / "net"
    model_prefix.parent.mkdir(parents=True, exist_ok=True)

    cmd.extend(
        [
            "--data-config",
            f"data/JetClass/JetClass_{args.feature_set}.yaml",
            "--network-config",
            "networks/example_ParticleTransformer.py",
            "--use-amp",
            "--model-prefix",
            str(model_prefix),
            "--num-workers",
            str(args.num_workers),
            "--fetch-step",
            str(args.fetch_step),
            "--batch-size",
            str(args.train_batch_size),
            "--samples-per-epoch",
            str(args.samples_per_epoch),
            "--samples-per-epoch-val",
            str(args.samples_per_epoch_val),
            "--num-epochs",
            str(args.epochs),
            "--gpus",
            args.gpus,
            "--start-lr",
            str(args.start_lr),
            "--optimizer",
            "ranger",
            "--log",
            str(Path(args.output_dir) / "train.log"),
            "--predict-output",
            str(Path(args.output_dir) / "pred_clean.root"),
            "--tensorboard",
            args.tensorboard_name,
        ]
    )
    if args.extra_weaver_args:
        cmd.extend(args.extra_weaver_args.split())
    return cmd


def find_checkpoint(search_root: Path) -> Path:
    candidates = list(search_root.rglob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files found under {search_root}")

    def score(p: Path) -> Tuple[int, float]:
        # prefer "best" checkpoints first, then latest mtime
        return (1 if "best" in p.name.lower() else 0, p.stat().st_mtime)

    return sorted(candidates, key=score, reverse=True)[0]


def _pad(a, maxlen: int, value: float = 0.0, dtype: str = "float32") -> ak.Array:
    if isinstance(a, ak.Array):
        if a.ndim == 1:
            a = ak.unflatten(a, 1)
        a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
        return ak.values_astype(a, dtype)
    raise TypeError(f"Expected awkward array, got {type(a)}")


def _delta_phi(phi1: ak.Array, phi2: ak.Array) -> ak.Array:
    dphi = phi1 - phi2
    return (dphi + np.pi) % (2.0 * np.pi) - np.pi


def _has_field(table: ak.Array, key: str) -> bool:
    # uproot returns an awkward record array; membership must use table.fields,
    # not `"key" in table`, which triggers element-wise comparisons.
    return key in set(getattr(table, "fields", []))


def _get_or_zeros(table: Dict[str, ak.Array], key: str, ref: ak.Array) -> ak.Array:
    if _has_field(table, key):
        return table[key]
    return ak.zeros_like(ref)


def _build_features(
    table: Dict[str, ak.Array],
    feature_set: str,
    eps: float = 1e-8,
) -> Tuple[List[ak.Array], List[ak.Array], List[ak.Array], List[ak.Array]]:
    p4 = vector.zip(
        {
            "px": table["part_px"],
            "py": table["part_py"],
            "pz": table["part_pz"],
            "energy": table["part_energy"],
        }
    )
    part_pt = p4.pt
    part_eta = p4.eta
    part_phi = p4.phi

    jet_pt = table["jet_pt"]
    jet_eta = table["jet_eta"]
    jet_phi = table["jet_phi"]
    jet_energy = table["jet_energy"]

    part_deta = table["part_deta"] if _has_field(table, "part_deta") else (part_eta - jet_eta)
    part_dphi = table["part_dphi"] if _has_field(table, "part_dphi") else _delta_phi(part_phi, jet_phi)
    part_deltaR = np.hypot(part_deta, part_dphi)

    part_pt_log = np.log(np.clip(part_pt, eps, None))
    part_e_log = np.log(np.clip(table["part_energy"], eps, None))
    part_logptrel = np.log(np.clip(part_pt / np.clip(jet_pt, eps, None), eps, None))
    part_logerel = np.log(
        np.clip(table["part_energy"] / np.clip(jet_energy, eps, None), eps, None)
    )

    # Common representation used by repo configs.
    pf_points = [part_deta, part_dphi]
    pf_vectors = [table["part_px"], table["part_py"], table["part_pz"], table["part_energy"]]
    pf_mask = [ak.ones_like(table["part_energy"])]

    if feature_set == "kin":
        pf_features = [
            part_pt_log,
            part_e_log,
            part_logptrel,
            part_logerel,
            part_deltaR,
            part_deta,
            part_dphi,
        ]
    elif feature_set == "kinpid":
        pf_features = [
            part_pt_log,
            part_e_log,
            part_logptrel,
            part_logerel,
            part_deltaR,
            _get_or_zeros(table, "part_charge", table["part_energy"]),
            _get_or_zeros(table, "part_isChargedHadron", table["part_energy"]),
            _get_or_zeros(table, "part_isNeutralHadron", table["part_energy"]),
            _get_or_zeros(table, "part_isPhoton", table["part_energy"]),
            _get_or_zeros(table, "part_isElectron", table["part_energy"]),
            _get_or_zeros(table, "part_isMuon", table["part_energy"]),
            part_deta,
            part_dphi,
        ]
    elif feature_set == "full":
        part_d0 = np.tanh(_get_or_zeros(table, "part_d0val", table["part_energy"]))
        part_dz = np.tanh(_get_or_zeros(table, "part_dzval", table["part_energy"]))
        pf_features = [
            part_pt_log,
            part_e_log,
            part_logptrel,
            part_logerel,
            part_deltaR,
            _get_or_zeros(table, "part_charge", table["part_energy"]),
            _get_or_zeros(table, "part_isChargedHadron", table["part_energy"]),
            _get_or_zeros(table, "part_isNeutralHadron", table["part_energy"]),
            _get_or_zeros(table, "part_isPhoton", table["part_energy"]),
            _get_or_zeros(table, "part_isElectron", table["part_energy"]),
            _get_or_zeros(table, "part_isMuon", table["part_energy"]),
            part_d0,
            _get_or_zeros(table, "part_d0err", table["part_energy"]),
            part_dz,
            _get_or_zeros(table, "part_dzerr", table["part_energy"]),
            part_deta,
            part_dphi,
        ]
    else:
        raise ValueError(f"Unsupported feature_set '{feature_set}'")

    return pf_points, pf_features, pf_vectors, pf_mask


def read_root_file(
    filepath: Path,
    feature_set: str,
    max_num_particles: int,
) -> InputArrays:
    import uproot

    required = set(
        [
            "part_px",
            "part_py",
            "part_pz",
            "part_energy",
            "jet_pt",
            "jet_eta",
            "jet_phi",
            "jet_energy",
            *LABEL_NAMES,
        ]
    )
    optional = set(
        [
            "part_deta",
            "part_dphi",
            "part_charge",
            "part_isChargedHadron",
            "part_isNeutralHadron",
            "part_isPhoton",
            "part_isElectron",
            "part_isMuon",
            "part_d0val",
            "part_d0err",
            "part_dzval",
            "part_dzerr",
        ]
    )
    tree = uproot.open(filepath)["tree"]
    available = set(tree.keys())
    missing = sorted(required - available)
    if missing:
        raise KeyError(f"Missing required branches in {filepath}: {missing}")
    load_branches = sorted(required | (optional & available))
    table = tree.arrays(load_branches)

    pf_points, pf_features, pf_vectors, pf_mask = _build_features(table, feature_set)
    x_points = np.stack(
        [ak.to_numpy(_pad(v, max_num_particles)) for v in pf_points], axis=1
    ).astype(np.float32)
    x_features = np.stack(
        [ak.to_numpy(_pad(v, max_num_particles)) for v in pf_features], axis=1
    ).astype(np.float32)
    x_vectors = np.stack(
        [ak.to_numpy(_pad(v, max_num_particles)) for v in pf_vectors], axis=1
    ).astype(np.float32)
    x_mask = np.stack([ak.to_numpy(_pad(v, max_num_particles)) for v in pf_mask], axis=1).astype(
        np.float32
    )
    y_onehot = np.stack([ak.to_numpy(table[n]).astype(np.int64) for n in LABEL_NAMES], axis=1)
    y_index = y_onehot.argmax(axis=1)
    return InputArrays(
        points=x_points,
        features=x_features,
        vectors=x_vectors,
        mask=x_mask,
        y_onehot=y_onehot,
        y_index=y_index,
    )


def concat_inputs(chunks: Sequence[InputArrays], max_jets: int = -1) -> InputArrays:
    points = np.concatenate([c.points for c in chunks], axis=0)
    features = np.concatenate([c.features for c in chunks], axis=0)
    vectors = np.concatenate([c.vectors for c in chunks], axis=0)
    mask = np.concatenate([c.mask for c in chunks], axis=0)
    y_onehot = np.concatenate([c.y_onehot for c in chunks], axis=0)
    y_index = np.concatenate([c.y_index for c in chunks], axis=0)

    if max_jets > 0 and len(y_index) > max_jets:
        points = points[:max_jets]
        features = features[:max_jets]
        vectors = vectors[:max_jets]
        mask = mask[:max_jets]
        y_onehot = y_onehot[:max_jets]
        y_index = y_index[:max_jets]

    return InputArrays(
        points=points,
        features=features,
        vectors=vectors,
        mask=mask,
        y_onehot=y_onehot,
        y_index=y_index,
    )


def slice_inputs(inputs: InputArrays, n: int) -> InputArrays:
    return InputArrays(
        points=inputs.points[:n],
        features=inputs.features[:n],
        vectors=inputs.vectors[:n],
        mask=inputs.mask[:n],
        y_onehot=inputs.y_onehot[:n],
        y_index=inputs.y_index[:n],
    )


def load_split(
    files: Sequence[Path],
    feature_set: str,
    max_num_particles: int,
    max_jets: int,
) -> InputArrays:
    chunks: List[InputArrays] = []
    per_file_cap = -1
    if max_jets > 0 and len(files) > 0:
        # Keep class coverage when test files are class-partitioned.
        per_file_cap = int(math.ceil(max_jets / len(files)))

    for fpath in files:
        chunk = read_root_file(fpath, feature_set=feature_set, max_num_particles=max_num_particles)
        if per_file_cap > 0 and len(chunk.y_index) > per_file_cap:
            chunk = slice_inputs(chunk, per_file_cap)
        chunks.append(chunk)

    merged = concat_inputs(chunks, max_jets=max_jets)
    return merged


def build_model(feature_set: str, num_classes: int, checkpoint_path: Path, device: torch.device):
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
        for_inference=True,
    )
    model = ParticleTransformerWrapper(**cfg)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def predict_probs(
    model: torch.nn.Module,
    inputs: InputArrays,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    outs: List[np.ndarray] = []
    n = len(inputs.y_index)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            points = torch.from_numpy(inputs.points[start:end]).to(device)
            features = torch.from_numpy(inputs.features[start:end]).to(device)
            vectors = torch.from_numpy(inputs.vectors[start:end]).to(device)
            # weaver uses float mask with shape (N, 1, P)
            mask = torch.from_numpy(inputs.mask[start:end]).to(device)
            pred = model(points, features, vectors, mask)
            pred = pred.detach().cpu().numpy()
            outs.append(pred)
    probs = np.concatenate(outs, axis=0)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
    return probs


def rankdata_avg_ties(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    x_sorted = x[order]
    ranks = np.empty_like(x_sorted, dtype=np.float64)
    i = 0
    n = len(x_sorted)
    while i < n:
        j = i + 1
        while j < n and x_sorted[j] == x_sorted[i]:
            j += 1
        # 1-based average rank for ties
        ranks[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty_like(ranks)
    out[order] = ranks
    return out


def binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata_avg_ties(y_score.astype(np.float64))
    sum_pos = ranks[y_true == 1].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def macro_auc(y_onehot: np.ndarray, probs: np.ndarray) -> float:
    per_class = []
    for c in range(y_onehot.shape[1]):
        auc_c = binary_auc(y_onehot[:, c], probs[:, c])
        if not math.isnan(auc_c):
            per_class.append(auc_c)
    if not per_class:
        return float("nan")
    return float(np.mean(per_class))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / np.clip(p.sum(), 1e-12, None)
    q = q / np.clip(q.sum(), 1e-12, None)
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log(np.clip(a[mask] / np.clip(b[mask], 1e-12, None), 1e-12, None)))

    return float(0.5 * _kl(p, m) + 0.5 * _kl(q, m))


def compute_supervised_metrics(inputs: InputArrays, probs: np.ndarray) -> Dict[str, float]:
    pred = probs.argmax(axis=1)
    acc = float((pred == inputs.y_index).mean())
    auc = macro_auc(inputs.y_onehot, probs)
    entropy = float((-probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=1).mean())
    conf = float(probs.max(axis=1).mean())
    return {
        "accuracy": acc,
        "macro_auc": auc,
        "mean_entropy": entropy,
        "mean_confidence": conf,
    }


def feature_indices_for_deta_dphi(feature_set: str) -> Tuple[int, int]:
    if feature_set == "kin":
        return (5, 6)
    if feature_set == "kinpid":
        return (11, 12)
    if feature_set == "full":
        return (15, 16)
    raise ValueError(f"Unsupported feature_set '{feature_set}'")


def apply_corruption(
    clean: InputArrays,
    corruption: str,
    severity: float,
    rng: np.random.Generator,
    feature_set: str,
) -> InputArrays:
    points = clean.points.copy()
    features = clean.features.copy()
    vectors = clean.vectors.copy()
    mask = clean.mask.copy()

    real = mask > 0

    if corruption == "gaussian_noise":
        points += (rng.normal(0.0, severity, size=points.shape).astype(np.float32) * real)
        features += (rng.normal(0.0, severity, size=features.shape).astype(np.float32) * real)
        vectors += (rng.normal(0.0, severity, size=vectors.shape).astype(np.float32) * real)
    elif corruption == "dropout":
        drop = (rng.random(size=mask.shape) < severity) & (real > 0)
        bdrop_p = np.broadcast_to(drop, points.shape)
        bdrop_f = np.broadcast_to(drop, features.shape)
        bdrop_v = np.broadcast_to(drop, vectors.shape)
        points[bdrop_p] = 0.0
        features[bdrop_f] = 0.0
        vectors[bdrop_v] = 0.0
        mask[drop] = 0.0
    elif corruption == "masking":
        mdrop = (rng.random(size=mask.shape) < severity) & (real > 0)
        bdrop_p = np.broadcast_to(mdrop, points.shape)
        bdrop_f = np.broadcast_to(mdrop, features.shape)
        bdrop_v = np.broadcast_to(mdrop, vectors.shape)
        points[bdrop_p] = 0.0
        features[bdrop_f] = 0.0
        vectors[bdrop_v] = 0.0
    elif corruption == "eta_phi_jitter":
        jitter = rng.normal(0.0, severity, size=points.shape).astype(np.float32) * real
        points += jitter
        deta_idx, dphi_idx = feature_indices_for_deta_dphi(feature_set)
        features[:, deta_idx : deta_idx + 1, :] += jitter[:, 0:1, :]
        features[:, dphi_idx : dphi_idx + 1, :] += jitter[:, 1:2, :]
    else:
        raise ValueError(f"Unsupported corruption type '{corruption}'")

    return InputArrays(
        points=points,
        features=features,
        vectors=vectors,
        mask=mask,
        y_onehot=clean.y_onehot,
        y_index=clean.y_index,
    )


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = rankdata_avg_ties(np.asarray(x, dtype=np.float64))
    y_rank = rankdata_avg_ties(np.asarray(y, dtype=np.float64))
    return pearson_corr(x_rank, y_rank)


def run_corruption_study(
    model: torch.nn.Module,
    clean_inputs: InputArrays,
    clean_probs: np.ndarray,
    clean_metrics: Dict[str, float],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    corruption_results: List[Dict[str, float]] = []
    rng = np.random.default_rng(args.seed)

    levels = {
        "gaussian_noise": parse_float_list(args.noise_levels),
        "dropout": parse_float_list(args.dropout_levels),
        "masking": parse_float_list(args.mask_levels),
        "eta_phi_jitter": parse_float_list(args.jitter_levels),
    }

    clean_entropy = clean_metrics["mean_entropy"]
    clean_confidence = clean_metrics["mean_confidence"]
    clean_pred_class_dist = clean_probs.mean(axis=0)
    clean_top1 = clean_probs.argmax(axis=1)

    for corruption in CORRUPTION_TYPES:
        for severity in levels[corruption]:
            corr_inputs = apply_corruption(
                clean=clean_inputs,
                corruption=corruption,
                severity=severity,
                rng=rng,
                feature_set=args.feature_set,
            )
            corr_probs = predict_probs(
                model=model, inputs=corr_inputs, batch_size=args.eval_batch_size, device=device
            )
            metrics = compute_supervised_metrics(corr_inputs, corr_probs)

            corr_entropy = metrics["mean_entropy"]
            corr_confidence = metrics["mean_confidence"]
            corr_pred_class_dist = corr_probs.mean(axis=0)
            corr_top1 = corr_probs.argmax(axis=1)

            row = {
                "corruption": corruption,
                "severity": float(severity),
                "accuracy": metrics["accuracy"],
                "macro_auc": metrics["macro_auc"],
                "auc_drop": clean_metrics["macro_auc"] - metrics["macro_auc"],
                "acc_drop": clean_metrics["accuracy"] - metrics["accuracy"],
                "mean_entropy": corr_entropy,
                "entropy_shift": corr_entropy - clean_entropy,
                "mean_confidence": corr_confidence,
                "confidence_drop": clean_confidence - corr_confidence,
                "prob_l1_drift": float(np.mean(np.sum(np.abs(corr_probs - clean_probs), axis=1))),
                "top1_flip_rate": float(np.mean(corr_top1 != clean_top1)),
                "class_js_div": js_divergence(corr_pred_class_dist, clean_pred_class_dist),
            }
            corruption_results.append(row)

    shift_metric_names = [
        "entropy_shift",
        "confidence_drop",
        "prob_l1_drift",
        "top1_flip_rate",
        "class_js_div",
    ]
    correlations: List[Dict[str, float]] = []
    auc_drop = np.array([r["auc_drop"] for r in corruption_results], dtype=np.float64)
    acc_drop = np.array([r["acc_drop"] for r in corruption_results], dtype=np.float64)
    for metric in shift_metric_names:
        vals = np.array([r[metric] for r in corruption_results], dtype=np.float64)
        correlations.append(
            {
                "metric": metric,
                "pearson_with_auc_drop": pearson_corr(vals, auc_drop),
                "spearman_with_auc_drop": spearman_corr(vals, auc_drop),
                "pearson_with_acc_drop": pearson_corr(vals, acc_drop),
                "spearman_with_acc_drop": spearman_corr(vals, acc_drop),
            }
        )
    return corruption_results, correlations


def write_csv(rows: List[Dict[str, float]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    parser = argparse.ArgumentParser(
        description="Train JetClass ParT baseline and evaluate corruption-shift correlations."
    )
    parser.add_argument("--dataset-dir", required=True, help="Flat dir with class ROOT files.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--feature-set", default="kin", choices=["kin", "kinpid", "full"])
    parser.add_argument("--max-num-particles", type=int, default=128)

    parser.add_argument("--train-indices", default="0-7")
    parser.add_argument("--val-indices", default="8")
    parser.add_argument("--test-indices", default="9")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-batch-size", type=int, default=512)
    parser.add_argument("--samples-per-epoch", type=int, default=131072)
    parser.add_argument("--samples-per-epoch-val", type=int, default=32768)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--fetch-step", type=float, default=0.01)
    parser.add_argument("--start-lr", type=float, default=1e-3)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--tensorboard-name", default="JetClass_part0_ParT_baseline")
    parser.add_argument("--extra-weaver-args", default="")
    parser.add_argument(
        "--include-weaver-test",
        action="store_true",
        help="If set, pass --data-test to weaver during training. "
        "Disabled by default because post-training evaluation is handled by this script.",
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--checkpoint", default="")

    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--max-test-jets", type=int, default=200000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--noise-levels", default="0.01,0.03,0.05,0.1")
    parser.add_argument("--dropout-levels", default="0.05,0.1,0.2,0.3")
    parser.add_argument("--mask-levels", default="0.05,0.1,0.2,0.3")
    parser.add_argument("--jitter-levels", default="0.01,0.03,0.05,0.1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    missing = []
    if ak is None:
        missing.append("awkward")
    if vector is None:
        missing.append("vector")
    if torch is None:
        missing.append("torch")
    if uproot is None:
        missing.append("uproot")
    if missing:
        raise RuntimeError(
            "Missing Python dependencies: "
            + ", ".join(missing)
            + ". Install them before running this script."
        )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    split_dir = output_dir / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_indices = parse_index_spec(args.train_indices)
    val_indices = parse_index_spec(args.val_indices)
    test_indices = parse_index_spec(args.test_indices)

    split_paths = prepare_split_dirs(
        dataset_dir=dataset_dir,
        split_dir=split_dir,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
    )

    # Persist run configuration for reproducibility.
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "dataset_dir": str(dataset_dir),
                "feature_set": args.feature_set,
                "train_indices": train_indices,
                "val_indices": val_indices,
                "test_indices": test_indices,
                "seed": args.seed,
                "epochs": args.epochs,
                "samples_per_epoch": args.samples_per_epoch,
                "samples_per_epoch_val": args.samples_per_epoch_val,
                "noise_levels": parse_float_list(args.noise_levels),
                "dropout_levels": parse_float_list(args.dropout_levels),
                "mask_levels": parse_float_list(args.mask_levels),
                "jitter_levels": parse_float_list(args.jitter_levels),
            },
            indent=2,
        )
        + "\n"
    )

    if args.skip_train:
        if not args.checkpoint:
            raise ValueError("--skip-train requires --checkpoint")
        checkpoint_path = Path(args.checkpoint).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    else:
        weaver_cmd = resolve_weaver_command()
        cmd = build_weaver_command(args, split_paths, weaver_cmd=weaver_cmd)
        print("Running training command:")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)

        found_ckpt = find_checkpoint(output_dir / "training")
        checkpoint_path = output_dir / "saved_model.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(found_ckpt, checkpoint_path)
        print(f"Saved stable model snapshot: {checkpoint_path}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and args.device != "cpu":
        print("Warning: CUDA not available, using CPU.", file=sys.stderr)

    test_files = sorted(split_paths["test"].glob("*.root"))
    if not test_files:
        raise FileNotFoundError(f"No test files found under {split_paths['test']}")
    print(f"Loading test split from {len(test_files)} files...")
    clean_inputs = load_split(
        files=test_files,
        feature_set=args.feature_set,
        max_num_particles=args.max_num_particles,
        max_jets=args.max_test_jets,
    )
    print(f"Loaded {len(clean_inputs.y_index)} test jets for evaluation.")
    uniq, counts = np.unique(clean_inputs.y_index, return_counts=True)
    class_counts = {LABEL_NAMES[int(k)]: int(v) for k, v in zip(uniq, counts)}
    print(f"Test label distribution: {class_counts}")
    if len(uniq) < 2:
        raise RuntimeError(
            "Loaded test sample has fewer than 2 classes; AUC/correlation metrics are not meaningful. "
            "Check split indices and input labels."
        )

    model = build_model(
        feature_set=args.feature_set,
        num_classes=len(LABEL_NAMES),
        checkpoint_path=checkpoint_path,
        device=device,
    )
    clean_probs = predict_probs(
        model=model, inputs=clean_inputs, batch_size=args.eval_batch_size, device=device
    )
    clean_metrics = compute_supervised_metrics(clean_inputs, clean_probs)
    clean_class_prob_dist = clean_probs.mean(axis=0)
    clean_top1 = clean_probs.argmax(axis=1)
    clean_top1_frac = np.bincount(clean_top1, minlength=len(LABEL_NAMES)).astype(np.float64)
    clean_top1_frac = clean_top1_frac / np.clip(clean_top1_frac.sum(), 1e-12, None)

    corruption_rows, corr_rows = run_corruption_study(
        model=model,
        clean_inputs=clean_inputs,
        clean_probs=clean_probs,
        clean_metrics=clean_metrics,
        args=args,
        device=device,
    )

    summary = {
        "clean_metrics": clean_metrics,
        "clean_pred_class_distribution": {
            LABEL_NAMES[i]: float(clean_class_prob_dist[i]) for i in range(len(LABEL_NAMES))
        },
        "clean_top1_fraction": {
            LABEL_NAMES[i]: float(clean_top1_frac[i]) for i in range(len(LABEL_NAMES))
        },
        "top_correlations_by_spearman_auc_drop": summarize_top_correlations(corr_rows),
        "num_corruption_points": len(corruption_rows),
        "saved_model_path": str(checkpoint_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "corruption_metrics.json").write_text(
        json.dumps(corruption_rows, indent=2) + "\n"
    )
    (output_dir / "correlations.json").write_text(json.dumps(corr_rows, indent=2) + "\n")
    write_csv(corruption_rows, output_dir / "corruption_metrics.csv")
    write_csv(corr_rows, output_dir / "correlations.csv")

    print("\n=== Clean test metrics ===")
    for k, v in clean_metrics.items():
        print(f"{k:>18s}: {v:.6f}")
    print("\n=== Top shift metrics by |Spearman with AUC drop| ===")
    for row in summarize_top_correlations(corr_rows):
        print(
            f"{row['metric']:>18s}: "
            f"spearman_auc={row['spearman_with_auc_drop']:.4f}, "
            f"pearson_auc={row['pearson_with_auc_drop']:.4f}, "
            f"spearman_acc={row['spearman_with_acc_drop']:.4f}"
        )
    print(f"\nWrote reports to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
