#!/usr/bin/env python3

from __future__ import annotations

"""
Evaluate a JetClass-trained Particle Transformer checkpoint on AspenOpenJets.

This script maps Aspen fixed-column arrays into JetClass-style ParT inputs,
computes unlabeled target-domain metrics, and runs corruption/stability tests.
"""

import argparse
import csv
import importlib.util
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    import h5py
except ModuleNotFoundError:
    h5py = None


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
class ModelInputs:
    points: np.ndarray
    features: np.ndarray
    vectors: np.ndarray
    mask: np.ndarray


class ParticleTransformerWrapper(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        from weaver.nn.model.ParticleTransformer import ParticleTransformer

        self.mod = ParticleTransformer(**kwargs)

    def forward(self, points, features, lorentz_vectors, mask):
        return self.mod(features, v=lorentz_vectors, mask=mask)


def parse_float_list(spec: str) -> List[float]:
    vals = [float(x.strip()) for x in spec.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"Expected a non-empty float list, got '{spec}'")
    return vals


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


def _wrap_delta_phi(dphi: np.ndarray) -> np.ndarray:
    return (dphi + np.pi) % (2.0 * np.pi) - np.pi


def _fit_particles(arr: np.ndarray, max_particles: int) -> np.ndarray:
    if arr.shape[1] == max_particles:
        return arr
    if arr.shape[1] > max_particles:
        return arr[:, :max_particles]
    pad = np.zeros((arr.shape[0], max_particles - arr.shape[1]), dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=1)


def _col_or_zeros(x3: np.ndarray, idx: int) -> np.ndarray:
    if x3.shape[2] > idx:
        return x3[:, :, idx].astype(np.float32, copy=False)
    return np.zeros((x3.shape[0], x3.shape[1]), dtype=np.float32)


def _derive_pid_flags(charge: np.ndarray, pdgid: np.ndarray) -> Dict[str, np.ndarray]:
    if pdgid is None:
        return {
            "is_charged_hadron": (charge != 0).astype(np.float32),
            "is_neutral_hadron": np.zeros_like(charge, dtype=np.float32),
            "is_photon": np.zeros_like(charge, dtype=np.float32),
            "is_electron": np.zeros_like(charge, dtype=np.float32),
            "is_muon": np.zeros_like(charge, dtype=np.float32),
        }

    absid = np.abs(pdgid.astype(np.int64))
    is_photon = (absid == 22)
    is_electron = (absid == 11)
    is_muon = (absid == 13)
    # Common neutral hadron IDs in PF-like inputs.
    neutral_ids = np.array([130, 2112, 310, 3122, 3112, 3222, 3312, 3334], dtype=np.int64)
    is_neutral_hadron = np.isin(absid, neutral_ids)
    is_charged_hadron = (charge != 0) & (~is_electron) & (~is_muon) & (~is_photon)
    return {
        "is_charged_hadron": is_charged_hadron.astype(np.float32),
        "is_neutral_hadron": is_neutral_hadron.astype(np.float32),
        "is_photon": is_photon.astype(np.float32),
        "is_electron": is_electron.astype(np.float32),
        "is_muon": is_muon.astype(np.float32),
    }


def aspen_batch_to_model_inputs(
    pf: np.ndarray,
    jet: np.ndarray,
    feature_set: str,
    max_num_particles: int,
    eps: float = 1e-8,
) -> ModelInputs:
    if pf.ndim != 3:
        raise ValueError(f"PFCands must be rank-3, got shape {pf.shape}")
    if jet.ndim != 2:
        raise ValueError(f"jet_kinematics must be rank-2, got shape {jet.shape}")
    if pf.shape[0] != jet.shape[0]:
        raise ValueError(f"Batch size mismatch: PFCands {pf.shape[0]} vs jet_kinematics {jet.shape[0]}")
    if pf.shape[2] < 4:
        raise ValueError(
            f"PFCands has only {pf.shape[2]} feature columns; need at least 4 (px,py,pz,energy)."
        )
    if jet.shape[1] < 3:
        raise ValueError(
            f"jet_kinematics has only {jet.shape[1]} columns; need at least 3 (pt,eta,phi)."
        )

    px = _fit_particles(_col_or_zeros(pf, 0), max_num_particles)
    py = _fit_particles(_col_or_zeros(pf, 1), max_num_particles)
    pz = _fit_particles(_col_or_zeros(pf, 2), max_num_particles)
    energy = _fit_particles(_col_or_zeros(pf, 3), max_num_particles)
    d0val = _fit_particles(_col_or_zeros(pf, 4), max_num_particles)
    d0err = _fit_particles(_col_or_zeros(pf, 5), max_num_particles)
    dzval = _fit_particles(_col_or_zeros(pf, 6), max_num_particles)
    dzerr = _fit_particles(_col_or_zeros(pf, 7), max_num_particles)
    charge = _fit_particles(_col_or_zeros(pf, 8), max_num_particles)
    pdgid = _fit_particles(_col_or_zeros(pf, 9), max_num_particles) if pf.shape[2] > 9 else None

    jet_pt = jet[:, 0].astype(np.float32, copy=False)
    jet_eta = jet[:, 1].astype(np.float32, copy=False)
    jet_phi = jet[:, 2].astype(np.float32, copy=False)
    if jet.shape[1] > 4:
        jet_energy = jet[:, 4].astype(np.float32, copy=False)
    elif jet.shape[1] > 3:
        jet_mass = np.clip(jet[:, 3].astype(np.float32, copy=False), 0.0, None)
        jet_energy = np.sqrt(np.square(jet_pt * np.cosh(jet_eta)) + np.square(jet_mass)).astype(np.float32)
    else:
        jet_energy = (jet_pt * np.cosh(jet_eta)).astype(np.float32)

    part_pt = np.hypot(px, py).astype(np.float32)
    part_eta = np.arcsinh(np.divide(pz, np.clip(part_pt, eps, None))).astype(np.float32)
    part_phi = np.arctan2(py, px).astype(np.float32)

    part_deta = (part_eta - jet_eta[:, None]).astype(np.float32)
    part_dphi = _wrap_delta_phi(part_phi - jet_phi[:, None]).astype(np.float32)
    part_deltaR = np.hypot(part_deta, part_dphi).astype(np.float32)

    part_pt_log = np.log(np.clip(part_pt, eps, None)).astype(np.float32)
    part_e_log = np.log(np.clip(energy, eps, None)).astype(np.float32)
    part_logptrel = np.log(np.clip(part_pt / np.clip(jet_pt[:, None], eps, None), eps, None)).astype(np.float32)
    part_logerel = np.log(
        np.clip(energy / np.clip(jet_energy[:, None], eps, None), eps, None)
    ).astype(np.float32)

    pid_flags = _derive_pid_flags(charge=charge, pdgid=pdgid)
    part_d0 = np.tanh(d0val).astype(np.float32)
    part_dz = np.tanh(dzval).astype(np.float32)

    points = np.stack([part_deta, part_dphi], axis=1).astype(np.float32)
    vectors = np.stack([px, py, pz, energy], axis=1).astype(np.float32)

    if feature_set == "kin":
        features = np.stack(
            [
                part_pt_log,
                part_e_log,
                part_logptrel,
                part_logerel,
                part_deltaR,
                part_deta,
                part_dphi,
            ],
            axis=1,
        ).astype(np.float32)
    elif feature_set == "kinpid":
        features = np.stack(
            [
                part_pt_log,
                part_e_log,
                part_logptrel,
                part_logerel,
                part_deltaR,
                charge.astype(np.float32),
                pid_flags["is_charged_hadron"],
                pid_flags["is_neutral_hadron"],
                pid_flags["is_photon"],
                pid_flags["is_electron"],
                pid_flags["is_muon"],
                part_deta,
                part_dphi,
            ],
            axis=1,
        ).astype(np.float32)
    elif feature_set == "full":
        features = np.stack(
            [
                part_pt_log,
                part_e_log,
                part_logptrel,
                part_logerel,
                part_deltaR,
                charge.astype(np.float32),
                pid_flags["is_charged_hadron"],
                pid_flags["is_neutral_hadron"],
                pid_flags["is_photon"],
                pid_flags["is_electron"],
                pid_flags["is_muon"],
                part_d0,
                d0err.astype(np.float32),
                part_dz,
                dzerr.astype(np.float32),
                part_deta,
                part_dphi,
            ],
            axis=1,
        ).astype(np.float32)
    else:
        raise ValueError(f"Unsupported feature_set '{feature_set}'")

    # Valid constituent mask: at least one non-zero momentum/energy component.
    mask2d = (
        (np.abs(px) + np.abs(py) + np.abs(pz) + np.abs(energy)) > 0.0
    ).astype(np.float32)
    mask = mask2d[:, None, :]

    return ModelInputs(points=points, features=features, vectors=vectors, mask=mask)


def feature_indices_for_deta_dphi(feature_set: str) -> Tuple[int, int]:
    if feature_set == "kin":
        return (5, 6)
    if feature_set == "kinpid":
        return (11, 12)
    if feature_set == "full":
        return (15, 16)
    raise ValueError(f"Unsupported feature_set '{feature_set}'")


def apply_corruption(
    clean: ModelInputs,
    corruption: str,
    severity: float,
    rng: np.random.Generator,
    feature_set: str,
) -> ModelInputs:
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
        points[np.broadcast_to(drop, points.shape)] = 0.0
        features[np.broadcast_to(drop, features.shape)] = 0.0
        vectors[np.broadcast_to(drop, vectors.shape)] = 0.0
        mask[drop] = 0.0
    elif corruption == "masking":
        mdrop = (rng.random(size=mask.shape) < severity) & (real > 0)
        points[np.broadcast_to(mdrop, points.shape)] = 0.0
        features[np.broadcast_to(mdrop, features.shape)] = 0.0
        vectors[np.broadcast_to(mdrop, vectors.shape)] = 0.0
    elif corruption == "eta_phi_jitter":
        jitter = rng.normal(0.0, severity, size=points.shape).astype(np.float32) * real
        points += jitter
        deta_idx, dphi_idx = feature_indices_for_deta_dphi(feature_set)
        features[:, deta_idx : deta_idx + 1, :] += jitter[:, 0:1, :]
        features[:, dphi_idx : dphi_idx + 1, :] += jitter[:, 1:2, :]
    else:
        raise ValueError(f"Unsupported corruption type '{corruption}'")

    return ModelInputs(points=points, features=features, vectors=vectors, mask=mask)


def build_model(feature_set: str, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    cfg = dict(
        input_dim=FEATURE_DIMS[feature_set],
        num_classes=len(LABEL_NAMES),
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


def predict_probs(model: torch.nn.Module, inputs: ModelInputs, batch_size: int, device: torch.device) -> np.ndarray:
    outs: List[np.ndarray] = []
    n = inputs.points.shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            points = torch.from_numpy(inputs.points[start:end]).to(device)
            features = torch.from_numpy(inputs.features[start:end]).to(device)
            vectors = torch.from_numpy(inputs.vectors[start:end]).to(device)
            mask = torch.from_numpy(inputs.mask[start:end]).to(device)
            pred = model(points, features, vectors, mask).detach().cpu().numpy()
            outs.append(pred)
    probs = np.concatenate(outs, axis=0)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
    return probs


def write_csv(rows: List[Dict[str, float]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate JetClass-trained ParT on AspenOpenJets.")
    parser.add_argument("--aspen-dir", required=True, help="Directory with AspenOpenJets *.h5 files.")
    parser.add_argument("--checkpoint", required=True, help="Path to saved_model.pt from JetClass run.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--feature-set", default="kinpid", choices=["kin", "kinpid", "full"])
    parser.add_argument("--max-num-particles", type=int, default=128)
    parser.add_argument("--max-jets", type=int, default=300000)
    parser.add_argument("--max-files", type=int, default=-1, help="Use first N Aspen files; -1 means all.")
    parser.add_argument("--read-chunk-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference-summary", default="", help="Optional JetClass summary.json path.")
    parser.add_argument("--skip-corruptions", action="store_true")

    parser.add_argument("--noise-levels", default="0.01,0.03,0.05,0.1")
    parser.add_argument("--dropout-levels", default="0.05,0.1,0.2,0.3")
    parser.add_argument("--mask-levels", default="0.05,0.1,0.2,0.3")
    parser.add_argument("--jitter-levels", default="0.01,0.03,0.05,0.1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    missing = []
    if torch is None:
        missing.append("torch")
    if h5py is None:
        missing.append("h5py")
    if importlib.util.find_spec("weaver.nn.model.ParticleTransformer") is None:
        missing.append("weaver-core")
    if missing:
        raise RuntimeError(
            "Missing dependencies: " + ", ".join(missing) + ". Install them before running."
        )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    aspen_dir = Path(args.aspen_dir).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not aspen_dir.exists():
        raise FileNotFoundError(f"Aspen dir not found: {aspen_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    files = sorted(aspen_dir.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {aspen_dir}")
    if args.max_files > 0:
        files = files[: args.max_files]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and args.device != "cpu":
        print("Warning: CUDA unavailable, falling back to CPU.")

    model = build_model(args.feature_set, checkpoint_path, device)
    rng = np.random.default_rng(args.seed)

    level_map = {
        "gaussian_noise": parse_float_list(args.noise_levels),
        "dropout": parse_float_list(args.dropout_levels),
        "masking": parse_float_list(args.mask_levels),
        "eta_phi_jitter": parse_float_list(args.jitter_levels),
    }
    corr_keys: List[Tuple[str, float]] = []
    if not args.skip_corruptions:
        for c in CORRUPTION_TYPES:
            for s in level_map[c]:
                corr_keys.append((c, float(s)))

    n_total = 0
    clean_entropy_sum = 0.0
    clean_conf_sum = 0.0
    clean_class_prob_sum = np.zeros((len(LABEL_NAMES),), dtype=np.float64)
    clean_top1_counts = np.zeros((len(LABEL_NAMES),), dtype=np.int64)

    corr_acc: Dict[Tuple[str, float], Dict[str, object]] = {}
    for key in corr_keys:
        corr_acc[key] = {
            "n": 0,
            "entropy_sum": 0.0,
            "conf_sum": 0.0,
            "prob_l1_sum": 0.0,
            "top1_flip_sum": 0.0,
            "class_prob_sum": np.zeros((len(LABEL_NAMES),), dtype=np.float64),
        }

    per_file_cap = -1
    if args.max_jets > 0:
        per_file_cap = int(math.ceil(args.max_jets / len(files)))
    remaining = args.max_jets if args.max_jets > 0 else -1

    for fpath in files:
        if remaining == 0:
            break
        with h5py.File(fpath, "r") as f:
            if "PFCands" not in f or "jet_kinematics" not in f:
                raise KeyError(f"{fpath} missing required datasets PFCands/jet_kinematics")
            pf_ds = f["PFCands"]
            jet_ds = f["jet_kinematics"]
            n_file = pf_ds.shape[0]
            take_file = n_file
            if per_file_cap > 0:
                take_file = min(take_file, per_file_cap)
            if remaining > 0:
                take_file = min(take_file, remaining)

            for start in range(0, take_file, args.read_chunk_size):
                end = min(start + args.read_chunk_size, take_file)
                pf = np.asarray(pf_ds[start:end], dtype=np.float32)
                jet = np.asarray(jet_ds[start:end], dtype=np.float32)
                batch_inputs = aspen_batch_to_model_inputs(
                    pf=pf,
                    jet=jet,
                    feature_set=args.feature_set,
                    max_num_particles=args.max_num_particles,
                )
                clean_probs = predict_probs(model, batch_inputs, args.eval_batch_size, device)
                clean_entropy = (-clean_probs * np.log(np.clip(clean_probs, 1e-12, None))).sum(axis=1)
                clean_conf = clean_probs.max(axis=1)
                clean_top1 = clean_probs.argmax(axis=1)

                bsz = clean_probs.shape[0]
                n_total += bsz
                clean_entropy_sum += float(clean_entropy.sum())
                clean_conf_sum += float(clean_conf.sum())
                clean_class_prob_sum += clean_probs.sum(axis=0)
                clean_top1_counts += np.bincount(clean_top1, minlength=len(LABEL_NAMES))

                for ctype, severity in corr_keys:
                    corr_inp = apply_corruption(
                        clean=batch_inputs,
                        corruption=ctype,
                        severity=severity,
                        rng=rng,
                        feature_set=args.feature_set,
                    )
                    corr_probs = predict_probs(model, corr_inp, args.eval_batch_size, device)
                    corr_entropy = (-corr_probs * np.log(np.clip(corr_probs, 1e-12, None))).sum(axis=1)
                    corr_conf = corr_probs.max(axis=1)
                    corr_top1 = corr_probs.argmax(axis=1)

                    acc = corr_acc[(ctype, severity)]
                    acc["n"] += bsz
                    acc["entropy_sum"] += float(corr_entropy.sum())
                    acc["conf_sum"] += float(corr_conf.sum())
                    acc["prob_l1_sum"] += float(np.abs(corr_probs - clean_probs).sum(axis=1).sum())
                    acc["top1_flip_sum"] += float((corr_top1 != clean_top1).sum())
                    acc["class_prob_sum"] += corr_probs.sum(axis=0)

            if remaining > 0:
                remaining -= take_file

    if n_total <= 0:
        raise RuntimeError("No Aspen jets were processed; check inputs and limits.")

    clean_class_prob_dist = clean_class_prob_sum / float(n_total)
    clean_top1_frac = clean_top1_counts.astype(np.float64)
    clean_top1_frac = clean_top1_frac / np.clip(clean_top1_frac.sum(), 1e-12, None)

    clean_metrics = {
        "num_jets": int(n_total),
        "mean_entropy": float(clean_entropy_sum / n_total),
        "mean_confidence": float(clean_conf_sum / n_total),
        "pred_class_distribution": {
            LABEL_NAMES[i]: float(clean_class_prob_dist[i]) for i in range(len(LABEL_NAMES))
        },
        "top1_fraction": {LABEL_NAMES[i]: float(clean_top1_frac[i]) for i in range(len(LABEL_NAMES))},
    }

    corr_rows: List[Dict[str, float]] = []
    for ctype, severity in corr_keys:
        acc = corr_acc[(ctype, severity)]
        n = int(acc["n"])
        if n == 0:
            continue
        corr_class_dist = acc["class_prob_sum"] / float(n)
        corr_mean_entropy = float(acc["entropy_sum"] / n)
        corr_mean_conf = float(acc["conf_sum"] / n)
        row = {
            "corruption": ctype,
            "severity": float(severity),
            "num_jets": n,
            "mean_entropy": corr_mean_entropy,
            "entropy_shift": corr_mean_entropy - clean_metrics["mean_entropy"],
            "mean_confidence": corr_mean_conf,
            "confidence_drop": clean_metrics["mean_confidence"] - corr_mean_conf,
            "prob_l1_drift": float(acc["prob_l1_sum"] / n),
            "top1_flip_rate": float(acc["top1_flip_sum"] / n),
            "class_js_div": js_divergence(corr_class_dist, clean_class_prob_dist),
        }
        corr_rows.append(row)

    reference = {}
    if args.reference_summary:
        ref_path = Path(args.reference_summary).resolve()
        if ref_path.exists():
            ref = json.loads(ref_path.read_text())
            reference["reference_summary_path"] = str(ref_path)
            ref_clean = ref.get("clean_metrics", {})
            ref_dist_map = ref.get("clean_pred_class_distribution", {})
            ref_dist = np.array([float(ref_dist_map.get(k, 0.0)) for k in LABEL_NAMES], dtype=np.float64)
            if ref_clean:
                reference["entropy_shift_vs_jetclass"] = (
                    clean_metrics["mean_entropy"] - float(ref_clean.get("mean_entropy", np.nan))
                )
                reference["confidence_drop_vs_jetclass"] = (
                    float(ref_clean.get("mean_confidence", np.nan)) - clean_metrics["mean_confidence"]
                )
            if np.isfinite(ref_dist).all() and ref_dist.sum() > 0:
                reference["class_js_div_vs_jetclass"] = js_divergence(clean_class_prob_dist, ref_dist)
            else:
                reference["class_js_div_vs_jetclass"] = None
            reference["jetclass_top_correlations"] = ref.get("top_correlations_by_spearman_auc_drop", [])
        else:
            reference["reference_summary_path"] = str(ref_path)
            reference["warning"] = "Reference summary path does not exist."

    run_config = {
        "aspen_dir": str(aspen_dir),
        "checkpoint": str(checkpoint_path),
        "feature_set": args.feature_set,
        "max_num_particles": int(args.max_num_particles),
        "max_jets": int(args.max_jets),
        "max_files": int(args.max_files),
        "read_chunk_size": int(args.read_chunk_size),
        "eval_batch_size": int(args.eval_batch_size),
        "seed": int(args.seed),
        "noise_levels": level_map["gaussian_noise"],
        "dropout_levels": level_map["dropout"],
        "mask_levels": level_map["masking"],
        "jitter_levels": level_map["eta_phi_jitter"],
        "skip_corruptions": bool(args.skip_corruptions),
        "device_requested": args.device,
        "device_used": str(device),
        "files_scanned": [str(f) for f in files],
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")

    summary = {
        "clean_metrics": clean_metrics,
        "reference_shift_metrics": reference,
        "num_corruption_points": len(corr_rows),
    }
    (out_dir / "aspen_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "aspen_corruption_metrics.json").write_text(json.dumps(corr_rows, indent=2) + "\n")
    write_csv(corr_rows, out_dir / "aspen_corruption_metrics.csv")

    print("=== Aspen clean metrics ===")
    print(json.dumps(clean_metrics, indent=2))
    if reference:
        print("\n=== Aspen vs JetClass reference shifts ===")
        print(json.dumps(reference, indent=2))
    print(f"\nWrote outputs to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
