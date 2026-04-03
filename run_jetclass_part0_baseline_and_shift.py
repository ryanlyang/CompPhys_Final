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
import re
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

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

try:
    from scipy.optimize import linear_sum_assignment
except ModuleNotFoundError:
    linear_sum_assignment = None

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

# JetClass canonical label ids used by weaver with data/JetClass/*.yaml.
# This mapping was validated from weaver test workers in train.log:
#   ZJetsToNuNu->0, HToBB->1, HToCC->2, HToGG->3, HToWW4Q->4,
#   HToWW2Q1L->5, ZToQQ->6, WToQQ->7, TTBar->8, TTBarLep->9
CLASS_TO_LABEL_INDEX = {
    "ZJetsToNuNu": 0,
    "HToBB": 1,
    "HToCC": 2,
    "HToGG": 3,
    "HToWW4Q": 4,
    "HToWW2Q1L": 5,
    "ZToQQ": 6,
    "WToQQ": 7,
    "TTBar": 8,
    "TTBarLep": 9,
}
FILENAME_CLASS_NAMES_BY_LABEL_INDEX = [None] * 10
for _k, _v in CLASS_TO_LABEL_INDEX.items():
    FILENAME_CLASS_NAMES_BY_LABEL_INDEX[_v] = _k

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
_INPUT_TRANSFORM_CACHE: Dict[str, Dict[str, Dict[str, object]]] = {}


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


def _ensure_nonempty_mask_np(
    points: np.ndarray,
    features: np.ndarray,
    vectors: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, int]:
    # Ensure each jet has at least one active constituent. All-masked jets can
    # trigger NaNs inside attention blocks.
    active = mask[:, 0, :].sum(axis=1)
    empty = active <= 0
    n_empty = int(empty.sum())
    if n_empty == 0:
        return mask, 0

    out = mask.copy()
    idx = np.flatnonzero(empty)
    for i in idx.tolist():
        # Try to infer valid particles from non-zero 4-vectors.
        nonzero = (np.abs(vectors[i]).sum(axis=0) > 0).astype(np.float32)
        if float(nonzero.sum()) > 0:
            out[i, 0, :] = nonzero
        else:
            out[i, 0, 0] = 1.0
            points[i, :, 0] = 0.0
            features[i, :, 0] = 0.0
            vectors[i, :, 0] = 0.0
    return out, n_empty


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


def _parse_first_float(line: str) -> float | None:
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def resolve_trainer_log_path(output_dir: Path, checkpoint_path: Path, explicit: str) -> Path | None:
    if explicit:
        p = Path(explicit).resolve()
        return p if p.exists() else None

    candidates = [
        output_dir / "train.log",
        checkpoint_path.parent / "train.log",
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    return None


def resolve_trainer_summary_path(output_dir: Path, checkpoint_path: Path, explicit: str) -> Path | None:
    if explicit:
        p = Path(explicit).resolve()
        return p if p.exists() else None

    candidates = [
        output_dir / "summary.json",
        checkpoint_path.parent / "summary.json",
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    return None


def parse_weaver_log_metrics(log_path: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    try:
        lines = log_path.read_text(errors="ignore").splitlines()
    except Exception:
        return metrics

    current_val = None
    best_val = None
    last_test_metric = None
    for line in lines:
        if "Current validation metric:" in line:
            m = re.search(
                r"Current validation metric:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*"
                r"\(best:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)",
                line,
            )
            if m:
                try:
                    current_val = float(m.group(1))
                    best_val = float(m.group(2))
                except ValueError:
                    pass
        elif "Test metric" in line:
            v = _parse_first_float(line.split("Test metric", 1)[-1])
            if v is not None:
                last_test_metric = v

    if current_val is not None:
        metrics["current_validation_metric_last"] = current_val
    if best_val is not None:
        metrics["best_validation_metric"] = best_val
    if last_test_metric is not None:
        metrics["last_test_metric"] = last_test_metric
    return metrics


def parse_trainer_summary_metrics(summary_path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        payload = json.loads(summary_path.read_text())
    except Exception:
        return out

    # RRR trainer format.
    test_best = payload.get("test_metrics_best")
    if isinstance(test_best, dict):
        for k in ("accuracy", "macro_auc", "mean_entropy", "mean_confidence"):
            v = test_best.get(k)
            if v is not None:
                try:
                    out[k] = float(v)
                except (TypeError, ValueError):
                    pass
        if out:
            return out

    # Baseline/eval summary format.
    clean = payload.get("clean_metrics")
    if isinstance(clean, dict):
        for k in ("accuracy", "macro_auc", "mean_entropy", "mean_confidence"):
            v = clean.get(k)
            if v is not None:
                try:
                    out[k] = float(v)
                except (TypeError, ValueError):
                    pass
    return out


def _build_wrap_index(ref: ak.Array, maxlen: int) -> Tuple[ak.Array, np.ndarray]:
    counts = ak.to_numpy(ak.num(ref, axis=1)).astype(np.int64, copy=False)
    n = int(counts.shape[0])
    base = np.broadcast_to(np.arange(maxlen, dtype=np.int64), (n, maxlen))
    counts_safe = np.where(counts > 0, counts, 1)[:, None]
    wrap_idx = base % counts_safe
    return ak.Array(wrap_idx), (counts == 0)


def _pad(
    a: ak.Array,
    maxlen: int,
    pad_mode: str = "constant",
    value: float = 0.0,
    dtype: str = "float32",
    wrap_idx: ak.Array | None = None,
    wrap_zero_rows: np.ndarray | None = None,
) -> ak.Array:
    if not isinstance(a, ak.Array):
        raise TypeError(f"Expected awkward array, got {type(a)}")
    if a.ndim == 1:
        a = ak.unflatten(a, 1)

    if pad_mode == "wrap":
        idx = wrap_idx
        zero_rows = wrap_zero_rows
        if idx is None or zero_rows is None:
            idx, zero_rows = _build_wrap_index(a, maxlen)
        out = a[idx]
        if bool(np.any(zero_rows)):
            zmask = ak.Array(np.broadcast_to(zero_rows[:, None], (len(zero_rows), maxlen)))
            out = ak.where(zmask, value, out)
    else:
        out = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
    return ak.values_astype(out, dtype)


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


def _clip_min_jagged(x: ak.Array, min_value: float) -> ak.Array:
    # Avoid np.clip on jagged awkward arrays; older awkward versions try to
    # convert to regular arrays and crash.
    return ak.where(x < min_value, min_value, x)


def _clip_max_jagged(x: ak.Array, max_value: float) -> ak.Array:
    return ak.where(x > max_value, max_value, x)


def _parse_var_spec(item) -> Tuple[str, float | None, float, float | None, float | None, float]:
    # [name, subtract, multiply, clip_min, clip_max, ...]
    if isinstance(item, str):
        return item, None, 1.0, -5.0, 5.0, 0.0
    if not isinstance(item, (list, tuple)) or len(item) < 1:
        raise ValueError(f"Invalid variable spec: {item}")
    name = str(item[0])
    sub = None if len(item) < 2 or item[1] is None else float(item[1])
    mul = 1.0 if len(item) < 3 or item[2] is None else float(item[2])
    cmin = -5.0 if len(item) < 4 or item[3] is None else float(item[3])
    cmax = 5.0 if len(item) < 5 or item[4] is None else float(item[4])
    padv = 0.0 if len(item) < 6 or item[5] is None else float(item[5])
    return name, sub, mul, cmin, cmax, padv


def _load_input_transforms(
    feature_set: str,
) -> Dict[str, Dict[str, object]]:
    cached = _INPUT_TRANSFORM_CACHE.get(feature_set)
    if cached is not None:
        return cached
    if yaml is None:
        raise RuntimeError("Missing dependency: PyYAML (yaml).")

    cfg_path = Path(__file__).resolve().parent / "data" / "JetClass" / f"JetClass_{feature_set}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"JetClass data config not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text())
    inputs_cfg = cfg.get("inputs", {})

    out: Dict[str, Dict[str, object]] = {}
    for group in ("pf_points", "pf_features", "pf_vectors", "pf_mask"):
        gcfg = inputs_cfg.get(group, {}) or {}
        var_specs = gcfg.get("vars", [])
        parsed_vars = [_parse_var_spec(v) for v in var_specs]
        out[group] = {
            "length": int(gcfg.get("length", 128)),
            "pad_mode": str(gcfg.get("pad_mode", "constant")),
            "vars": parsed_vars,
            "pad_value": float(parsed_vars[0][5]) if parsed_vars else 0.0,
        }
    _INPUT_TRANSFORM_CACHE[feature_set] = out
    return out


def _apply_input_transform(
    x: ak.Array,
    subtract_by: float | None,
    multiply_by: float,
    clip_min: float | None,
    clip_max: float | None,
) -> ak.Array:
    y = x
    if subtract_by is not None:
        y = y - subtract_by
    if multiply_by != 1.0:
        y = y * multiply_by
    if clip_min is not None:
        y = _clip_min_jagged(y, clip_min)
    if clip_max is not None:
        y = _clip_max_jagged(y, clip_max)
    return y


def _build_features(
    table: Dict[str, ak.Array],
    transforms: Dict[str, Dict[str, object]],
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

    part_energy = table["part_energy"]
    part_pt_safe = _clip_min_jagged(part_pt, eps)
    part_energy_safe = _clip_min_jagged(part_energy, eps)
    jet_pt_safe = _clip_min_jagged(jet_pt, eps)
    jet_energy_safe = _clip_min_jagged(jet_energy, eps)

    part_pt_log = np.log(part_pt_safe)
    part_e_log = np.log(part_energy_safe)
    part_logptrel = np.log(_clip_min_jagged(part_pt / jet_pt_safe, eps))
    part_logerel = np.log(_clip_min_jagged(part_energy / jet_energy_safe, eps))

    part_charge = _get_or_zeros(table, "part_charge", table["part_energy"])
    part_is_ch = _get_or_zeros(table, "part_isChargedHadron", table["part_energy"])
    part_is_nh = _get_or_zeros(table, "part_isNeutralHadron", table["part_energy"])
    part_is_ph = _get_or_zeros(table, "part_isPhoton", table["part_energy"])
    part_is_el = _get_or_zeros(table, "part_isElectron", table["part_energy"])
    part_is_mu = _get_or_zeros(table, "part_isMuon", table["part_energy"])
    part_d0 = np.tanh(_get_or_zeros(table, "part_d0val", table["part_energy"]))
    part_d0err = _get_or_zeros(table, "part_d0err", table["part_energy"])
    part_dz = np.tanh(_get_or_zeros(table, "part_dzval", table["part_energy"]))
    part_dzerr = _get_or_zeros(table, "part_dzerr", table["part_energy"])
    part_mask = ak.ones_like(table["part_energy"])

    var_map: Dict[str, ak.Array] = {
        "part_deta": part_deta,
        "part_dphi": part_dphi,
        "part_pt_log": part_pt_log,
        "part_e_log": part_e_log,
        "part_logptrel": part_logptrel,
        "part_logerel": part_logerel,
        "part_deltaR": part_deltaR,
        "part_charge": part_charge,
        "part_isChargedHadron": part_is_ch,
        "part_isNeutralHadron": part_is_nh,
        "part_isPhoton": part_is_ph,
        "part_isElectron": part_is_el,
        "part_isMuon": part_is_mu,
        "part_d0": part_d0,
        "part_d0err": part_d0err,
        "part_dz": part_dz,
        "part_dzerr": part_dzerr,
        "part_px": table["part_px"],
        "part_py": table["part_py"],
        "part_pz": table["part_pz"],
        "part_energy": table["part_energy"],
        "part_mask": part_mask,
    }

    def _assemble(group: str) -> List[ak.Array]:
        arrs: List[ak.Array] = []
        for name, sub, mul, cmin, cmax, _padv in transforms[group]["vars"]:
            if name not in var_map:
                raise KeyError(f"Variable '{name}' required by config missing in var_map")
            arrs.append(_apply_input_transform(var_map[name], sub, mul, cmin, cmax))
        return arrs

    pf_points = _assemble("pf_points")
    pf_features = _assemble("pf_features")
    pf_vectors = _assemble("pf_vectors")
    pf_mask = _assemble("pf_mask")
    return pf_points, pf_features, pf_vectors, pf_mask


def _stack_group(
    arrs: List[ak.Array],
    group_cfg: Dict[str, object],
    max_num_particles: int,
) -> np.ndarray:
    if not arrs:
        raise ValueError("Cannot stack empty input group.")
    cfg_len = int(group_cfg.get("length", 128))
    length = min(cfg_len, max_num_particles) if max_num_particles > 0 else cfg_len
    pad_mode = str(group_cfg.get("pad_mode", "constant"))
    pad_value = float(group_cfg.get("pad_value", 0.0))

    def _dense_from_jagged(v: ak.Array) -> np.ndarray:
        if not isinstance(v, ak.Array):
            raise TypeError(f"Expected awkward array in _stack_group, got {type(v)}")
        if v.ndim == 1:
            v = ak.unflatten(v, 1)

        counts = ak.to_numpy(ak.num(v, axis=1)).astype(np.int64, copy=False)
        n = int(counts.shape[0])
        out = np.full((n, length), pad_value, dtype=np.float32)
        if n == 0:
            return out

        # Flatten one jagged axis and rebuild dense rows using offsets/counts.
        # This avoids awkward RegularArray conversions that fail on older versions.
        flat = ak.to_numpy(ak.flatten(v, axis=1))
        if flat.size == 0:
            return out
        try:
            flat = flat.astype(np.float32, copy=False)
        except Exception as exc:
            raise ValueError(
                "Unable to cast flattened constituent array to float32. "
                "Input appears non-numeric or irregular beyond first jagged axis."
            ) from exc

        starts = np.zeros(n, dtype=np.int64)
        if n > 1:
            starts[1:] = np.cumsum(counts[:-1], dtype=np.int64)

        nonempty_rows = np.flatnonzero(counts > 0)
        base = np.arange(length, dtype=np.int64)
        for ridx in nonempty_rows.tolist():
            c = int(counts[ridx])
            s = int(starts[ridx])
            if pad_mode == "wrap":
                gather = s + (base % c)
                out[ridx, :] = flat[gather]
            else:
                take = min(c, length)
                out[ridx, :take] = flat[s : s + take]
        return out

    cols: List[np.ndarray] = [_dense_from_jagged(v) for v in arrs]

    stacked = np.stack(cols, axis=1).astype(np.float32, copy=False)
    return stacked


def read_root_file(
    filepath: Path,
    feature_set: str,
    max_num_particles: int,
    label_source: str,
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
        ]
    )
    if label_source == "branch":
        required.update(LABEL_NAMES)
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

    transforms = _load_input_transforms(feature_set)
    pf_points, pf_features, pf_vectors, pf_mask = _build_features(table, transforms)
    x_points = _stack_group(pf_points, transforms["pf_points"], max_num_particles)
    x_features = _stack_group(pf_features, transforms["pf_features"], max_num_particles)
    x_vectors = _stack_group(pf_vectors, transforms["pf_vectors"], max_num_particles)
    x_mask = _stack_group(pf_mask, transforms["pf_mask"], max_num_particles)
    x_mask, n_empty = _ensure_nonempty_mask_np(x_points, x_features, x_vectors, x_mask)
    if n_empty > 0:
        print(
            f"Warning: repaired {n_empty} all-masked jets while reading {filepath.name}.",
            file=sys.stderr,
        )
    n_events = int(x_points.shape[0])
    if label_source == "branch":
        y_onehot = np.stack([ak.to_numpy(table[n]).astype(np.int64) for n in LABEL_NAMES], axis=1)
        y_index = y_onehot.argmax(axis=1)
    elif label_source == "filename":
        cls_name = filepath.name.split("_", 1)[0]
        if cls_name not in CLASS_TO_LABEL_INDEX:
            raise ValueError(
                f"Cannot infer class from filename '{filepath.name}'. "
                f"Expected prefix in {sorted(CLASS_TO_LABEL_INDEX)}"
            )
        cls_idx = int(CLASS_TO_LABEL_INDEX[cls_name])
        y_index = np.full(n_events, cls_idx, dtype=np.int64)
        y_onehot = np.zeros((n_events, len(LABEL_NAMES)), dtype=np.int64)
        y_onehot[np.arange(n_events), y_index] = 1
    else:
        raise ValueError(f"Unsupported label_source '{label_source}'")
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
    label_source: str,
) -> InputArrays:
    chunks: List[InputArrays] = []
    per_file_cap = -1
    if max_jets > 0 and len(files) > 0:
        # Keep class coverage when test files are class-partitioned.
        per_file_cap = int(math.ceil(max_jets / len(files)))

    for fpath in files:
        chunk = read_root_file(
            fpath,
            feature_set=feature_set,
            max_num_particles=max_num_particles,
            label_source=label_source,
        )
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
        # Keep inference config aligned with training network config.
        # We explicitly apply softmax in predict_probs().
        for_inference=False,
        # Numerical stability for post-hoc evaluation/attribution.
        use_amp=False,
    )
    model = ParticleTransformerWrapper(**cfg)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    # Some checkpoints/classes still carry this flag; force fp32 inference.
    if hasattr(model, "mod") and hasattr(model.mod, "use_amp"):
        model.mod.use_amp = False
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
                # First try to infer from non-zero vectors.
                vec_nonzero = (vectors[idx].abs().sum(dim=1) > 0).to(mask.dtype)
                mask[idx, 0, :] = vec_nonzero
                still_empty = (mask[idx].sum(dim=2) <= 0).squeeze(1)
                if bool(still_empty.any().item()):
                    idx2 = idx[still_empty]
                    mask[idx2, 0, 0] = 1.0
                    points[idx2, :, 0] = 0.0
                    features[idx2, :, 0] = 0.0
                    vectors[idx2, :, 0] = 0.0
            # Match weaver data pipeline: use boolean constituent mask.
            pred = model(points, features, vectors, (mask > 0.5))
            pred = pred.detach().cpu().numpy()
            outs.append(pred)
    raw = np.concatenate(outs, axis=0)
    if not np.isfinite(raw).all():
        n_bad = int(np.size(raw) - np.isfinite(raw).sum())
        print(
            f"Warning: model output contains {n_bad} non-finite values; applying nan_to_num stabilization.",
            file=sys.stderr,
        )
        raw = np.nan_to_num(raw, nan=0.0, posinf=50.0, neginf=-50.0)
    # Model is loaded with for_inference=False, so output is logits.
    x = raw - np.max(raw, axis=1, keepdims=True)
    ex = np.exp(x)
    probs = ex / np.clip(ex.sum(axis=1, keepdims=True), 1e-12, None)
    probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
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
    if not np.isfinite(probs).all():
        n_bad = int(np.size(probs) - np.isfinite(probs).sum())
        raise RuntimeError(f"Non-finite probabilities encountered in metric computation: {n_bad}")
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


def compute_confusion_counts(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def best_permutation_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> Dict[str, object]:
    cm = compute_confusion_counts(y_true, y_pred, n_classes=n_classes)
    total = int(cm.sum())
    raw_acc = float(np.trace(cm) / max(total, 1))
    out: Dict[str, object] = {
        "raw_accuracy_from_cm": raw_acc,
        "best_permutation_accuracy": float("nan"),
        "improvement_over_raw": float("nan"),
        "row_to_col_mapping": {},
    }
    if linear_sum_assignment is None:
        out["note"] = "scipy not available; skipping permutation diagnostic."
        return out
    row_ind, col_ind = linear_sum_assignment(-cm)
    matched = int(cm[row_ind, col_ind].sum())
    best_acc = float(matched / max(total, 1))
    mapping = {int(r): int(c) for r, c in zip(row_ind.tolist(), col_ind.tolist())}
    out["best_permutation_accuracy"] = best_acc
    out["improvement_over_raw"] = float(best_acc - raw_acc)
    out["row_to_col_mapping"] = mapping
    return out


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
    parser.add_argument(
        "--label-source",
        default="filename",
        choices=["filename", "branch"],
        help=(
            "How to derive class labels for evaluation. "
            "`filename` matches the class indexing used by the weaver command in this script; "
            "`branch` reads label_* branches from ROOT."
        ),
    )
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
    parser.add_argument(
        "--trainer-log",
        default="",
        help=(
            "Optional path to weaver train.log. If set (or auto-detected), "
            "trainer-native accuracy metrics are extracted from this log."
        ),
    )
    parser.add_argument(
        "--trainer-summary",
        default="",
        help=(
            "Optional path to trainer summary.json (e.g. RRR output). If set "
            "(or auto-detected), clean metrics are taken from this file."
        ),
    )

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
                "label_source": args.label_source,
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

    trainer_log_path = resolve_trainer_log_path(
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        explicit=args.trainer_log,
    )
    weaver_log_metrics: Dict[str, float] = {}
    if trainer_log_path is not None:
        weaver_log_metrics = parse_weaver_log_metrics(trainer_log_path)
        if weaver_log_metrics:
            print(f"Detected trainer log metrics from: {trainer_log_path}")
    trainer_summary_path = resolve_trainer_summary_path(
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        explicit=args.trainer_summary,
    )
    trainer_summary_metrics: Dict[str, float] = {}
    if trainer_summary_path is not None:
        trainer_summary_metrics = parse_trainer_summary_metrics(trainer_summary_path)
        if trainer_summary_metrics:
            print(f"Detected trainer summary metrics from: {trainer_summary_path}")

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
        label_source=args.label_source,
    )
    print(f"Loaded {len(clean_inputs.y_index)} test jets for evaluation.")
    class_names = (
        FILENAME_CLASS_NAMES_BY_LABEL_INDEX if args.label_source == "filename" else LABEL_NAMES
    )
    uniq, counts = np.unique(clean_inputs.y_index, return_counts=True)
    class_counts = {class_names[int(k)]: int(v) for k, v in zip(uniq, counts)}
    print(f"Test label distribution: {class_counts}")
    if len(uniq) < 2:
        raise RuntimeError(
            "Loaded test sample has fewer than 2 classes; AUC/correlation metrics are not meaningful. "
            "Check split indices and input labels."
        )

    n_classes = len(class_names)
    model = build_model(
        feature_set=args.feature_set,
        num_classes=n_classes,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    clean_probs = predict_probs(
        model=model, inputs=clean_inputs, batch_size=args.eval_batch_size, device=device
    )
    clean_metrics = compute_supervised_metrics(clean_inputs, clean_probs)
    clean_metrics_posthoc = dict(clean_metrics)
    clean_metrics_report = dict(clean_metrics_posthoc)
    reported_clean_accuracy_source = "posthoc_forward"
    if trainer_summary_metrics:
        clean_metrics_report.update(trainer_summary_metrics)
        reported_clean_accuracy_source = "trainer_summary_json"
    elif "last_test_metric" in weaver_log_metrics:
        clean_metrics_report["accuracy"] = float(weaver_log_metrics["last_test_metric"])
        reported_clean_accuracy_source = "weaver_test_metric_from_log"
    elif "best_validation_metric" in weaver_log_metrics:
        clean_metrics_report["accuracy"] = float(weaver_log_metrics["best_validation_metric"])
        reported_clean_accuracy_source = "weaver_best_validation_metric_from_log"
    elif "current_validation_metric_last" in weaver_log_metrics:
        clean_metrics_report["accuracy"] = float(weaver_log_metrics["current_validation_metric_last"])
        reported_clean_accuracy_source = "weaver_last_validation_metric_from_log"
    clean_class_prob_dist = clean_probs.mean(axis=0)
    clean_top1 = clean_probs.argmax(axis=1)
    perm_diag = best_permutation_accuracy(
        y_true=clean_inputs.y_index, y_pred=clean_top1, n_classes=n_classes
    )
    clean_top1_frac = np.bincount(clean_top1, minlength=n_classes).astype(np.float64)
    clean_top1_frac = clean_top1_frac / np.clip(clean_top1_frac.sum(), 1e-12, None)
    if np.isfinite(float(perm_diag.get("best_permutation_accuracy", float("nan")))):
        print(
            "Permutation diagnostic: "
            f"raw_acc={perm_diag['raw_accuracy_from_cm']:.6f}, "
            f"best_perm_acc={perm_diag['best_permutation_accuracy']:.6f}, "
            f"delta={perm_diag['improvement_over_raw']:.6f}"
        )

    corruption_rows, corr_rows = run_corruption_study(
        model=model,
        clean_inputs=clean_inputs,
        clean_probs=clean_probs,
        clean_metrics=clean_metrics,
        args=args,
        device=device,
    )

    summary = {
        "label_source": args.label_source,
        "class_names": class_names,
        "clean_metrics": clean_metrics_report,
        "clean_metrics_posthoc": clean_metrics_posthoc,
        "reported_clean_accuracy_source": reported_clean_accuracy_source,
        "weaver_log_path": str(trainer_log_path) if trainer_log_path is not None else None,
        "weaver_log_metrics": weaver_log_metrics,
        "trainer_summary_path": (
            str(trainer_summary_path) if trainer_summary_path is not None else None
        ),
        "trainer_summary_metrics": trainer_summary_metrics,
        "clean_pred_class_distribution": {
            class_names[i]: float(clean_class_prob_dist[i]) for i in range(len(class_names))
        },
        "clean_top1_fraction": {
            class_names[i]: float(clean_top1_frac[i]) for i in range(len(class_names))
        },
        "label_order_permutation_diagnostic": perm_diag,
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

    print("\n=== Clean test metrics (used for report) ===")
    for k, v in clean_metrics_report.items():
        print(f"{k:>18s}: {v:.6f}")
    if reported_clean_accuracy_source != "posthoc_forward":
        print(
            f"{'accuracy_source':>18s}: {reported_clean_accuracy_source}\n"
            f"{'posthoc_accuracy':>18s}: {clean_metrics_posthoc['accuracy']:.6f}"
        )
        if "accuracy" in clean_metrics_report:
            print(f"{'reported_accuracy':>18s}: {clean_metrics_report['accuracy']:.6f}")
    if weaver_log_metrics:
        print("\n=== Weaver trainer-log metrics ===")
        for k, v in weaver_log_metrics.items():
            print(f"{k:>30s}: {v:.6f}")
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
