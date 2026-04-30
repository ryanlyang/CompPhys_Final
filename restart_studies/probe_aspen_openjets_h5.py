#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    import h5py
except ModuleNotFoundError:
    h5py = None

JET_KINEMATICS_NAMES = ["pt", "eta", "phi", "msoftdrop"]
JET_TAGGING_NAMES = [
    "nConstituents",
    "tau1",
    "tau2",
    "tau3",
    "tau4",
    "particleNet_H4qvsQCD",
    "particleNet_HbbvsQCD",
    "particleNet_HccvsQCD",
    "particleNet_QCD",
    "particleNet_TvsQCD",
    "particleNet_WvsQCD",
    "particleNet_ZvsQCD",
    "particleNet_mass",
]
PFCAND_NAMES = [
    "px",
    "py",
    "pz",
    "E",
    "d0",
    "d0Err",
    "dz",
    "dzErr",
    "charge",
    "pdgId",
    "puppiWeight",
]
EVENT_INFO_NAMES = ["run", "luminosityBlock", "event"]
LABEL_KEY_CANDIDATES = {"label", "labels", "class", "classes", "target", "targets", "y"}


@dataclass
class RunningFeatureStats:
    n_features: int
    count_rows: int = 0
    sum: np.ndarray = field(init=False)
    sum_sq: np.ndarray = field(init=False)
    min: np.ndarray = field(init=False)
    max: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.sum = np.zeros(self.n_features, dtype=np.float64)
        self.sum_sq = np.zeros(self.n_features, dtype=np.float64)
        self.min = np.full(self.n_features, np.inf, dtype=np.float64)
        self.max = np.full(self.n_features, -np.inf, dtype=np.float64)

    def update(self, rows: np.ndarray) -> None:
        if rows.size == 0:
            return
        rows = np.asarray(rows, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != self.n_features:
            raise ValueError(f"expected rows shape [N, {self.n_features}], got {rows.shape}")
        self.count_rows += int(rows.shape[0])
        self.sum += rows.sum(axis=0)
        self.sum_sq += np.square(rows).sum(axis=0)
        self.min = np.minimum(self.min, rows.min(axis=0))
        self.max = np.maximum(self.max, rows.max(axis=0))

    def to_summary(self, names: Sequence[str]) -> Dict[str, Dict[str, float]]:
        if self.count_rows == 0:
            return {}
        mean = self.sum / self.count_rows
        var = np.maximum(self.sum_sq / self.count_rows - np.square(mean), 0.0)
        std = np.sqrt(var)
        out: Dict[str, Dict[str, float]] = {}
        for i, name in enumerate(names):
            out[name] = {
                "mean": float(mean[i]),
                "std": float(std[i]),
                "min": float(self.min[i]),
                "max": float(self.max[i]),
            }
        return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe AspenOpenJets HDF5 files to build schema/stats/split manifest."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("/home/ryreu/atlas/CompPhys_Final/data/AspenOpenJets"),
        help="Directory containing AOJ .h5 files.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.h5",
        help="File glob pattern inside --data_dir.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="If >0, inspect at most this many files (sorted order).",
    )
    parser.add_argument(
        "--sample_jets_per_file",
        type=int,
        default=10000,
        help="Jets sampled per file for stats (linspace sampling).",
    )
    parser.add_argument(
        "--split_fractions",
        type=str,
        default="0.8,0.1,0.1",
        help="Train/val/test fractions, comma-separated.",
    )
    parser.add_argument("--seed", type=int, default=52, help="Seed for deterministic split assignment.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("restart_studies/results"),
        help="Parent output directory.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="aspen_probe",
        help="Subdirectory name inside --output_dir.",
    )
    return parser.parse_args()


def parse_split_fractions(raw: str) -> Tuple[float, float, float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"--split_fractions must have 3 values, got {parts}")
    vals = [float(p) for p in parts]
    if any(v <= 0 for v in vals):
        raise ValueError(f"--split_fractions must all be > 0, got {vals}")
    total = sum(vals)
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        vals = [v / total for v in vals]
    return vals[0], vals[1], vals[2]


def list_h5_files(data_dir: Path, pattern: str) -> List[Path]:
    files = sorted(p for p in data_dir.glob(pattern) if p.is_file())
    return files


def select_indices(n_items: int, n_keep: int) -> np.ndarray:
    if n_items <= 0:
        return np.empty((0,), dtype=np.int64)
    if n_keep <= 0 or n_keep >= n_items:
        return np.arange(n_items, dtype=np.int64)
    idx = np.linspace(0, n_items - 1, n_keep, dtype=np.int64)
    return np.unique(idx)


def infer_jet_count(h5f: h5py.File) -> int:
    for key in ("PFCands", "jet_kinematics", "jet_tagging", "event_info"):
        if key in h5f and len(h5f[key].shape) >= 1:
            return int(h5f[key].shape[0])
    for key in h5f.keys():
        ds = h5f[key]
        if len(ds.shape) >= 1:
            return int(ds.shape[0])
    return 0


def infer_pf_mask(pfcands: np.ndarray) -> np.ndarray:
    if pfcands.ndim != 3:
        raise ValueError(f"expected PFCands shape [N, C, F], got {pfcands.shape}")
    if pfcands.shape[-1] >= 4:
        momentum_abs = np.abs(pfcands[..., :4]).sum(axis=-1)
    else:
        momentum_abs = np.abs(pfcands).sum(axis=-1)
    return momentum_abs > 0


def stable_hash_int(text: str) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:15]
    return int(digest, 16)


def split_counts(n: int, fracs: Sequence[float]) -> Tuple[int, int, int]:
    raw = [n * f for f in fracs]
    base = [int(math.floor(v)) for v in raw]
    rem = n - sum(base)
    order = sorted(range(3), key=lambda i: (raw[i] - base[i]), reverse=True)
    for i in range(rem):
        base[order[i % 3]] += 1
    return base[0], base[1], base[2]


def assign_splits(file_paths: Sequence[Path], fracs: Sequence[float], seed: int) -> Dict[str, str]:
    keyed = []
    for path in file_paths:
        sort_key = stable_hash_int(f"{seed}:{path.name}")
        keyed.append((sort_key, path))
    keyed.sort(key=lambda t: (t[0], str(t[1])))
    n_train, n_val, n_test = split_counts(len(keyed), fracs)

    assignments: Dict[str, str] = {}
    for i, (_, path) in enumerate(keyed):
        if i < n_train:
            split = "train"
        elif i < n_train + n_val:
            split = "val"
        else:
            split = "test"
        assignments[str(path)] = split
    return assignments


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    if h5py is None:
        raise SystemExit(
            "Missing dependency: h5py. Install it in your runtime (e.g., pip install h5py) and rerun."
        )
    frac_train, frac_val, frac_test = parse_split_fractions(args.split_fractions)

    files = list_h5_files(args.data_dir, args.glob)
    if args.max_files > 0:
        files = files[: args.max_files]

    if not files:
        raise SystemExit(f"No files matched {args.data_dir}/{args.glob}")

    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    jet_kin_stats = RunningFeatureStats(len(JET_KINEMATICS_NAMES))
    jet_tag_stats = RunningFeatureStats(len(JET_TAGGING_NAMES))
    pf_stats = RunningFeatureStats(len(PFCAND_NAMES))
    event_stats = RunningFeatureStats(len(EVENT_INFO_NAMES))

    schema_by_key: Dict[str, set] = {}
    keys_union: set = set()
    label_keys_found: set = set()
    warnings: List[str] = []
    pf_count_samples: List[np.ndarray] = []
    file_rows: List[Dict[str, object]] = []
    total_jets = 0
    total_events_sampled = 0
    duplicate_events_in_sample = 0
    n_files_with_pf = 0
    n_files_with_jet_kin = 0

    for path in files:
        with h5py.File(path, "r") as h5f:
            keys = sorted(list(h5f.keys()))
            keys_union.update(keys)
            for key in keys:
                ds = h5f[key]
                schema_by_key.setdefault(key, set()).add((tuple(ds.shape[1:]), str(ds.dtype)))
            label_keys_found.update(k for k in keys if k.lower() in LABEL_KEY_CANDIDATES)

            n_jets = infer_jet_count(h5f)
            total_jets += n_jets
            idx = select_indices(n_jets, args.sample_jets_per_file)

            sampled = int(idx.size)
            file_size_bytes = path.stat().st_size
            row = {
                "file": str(path),
                "file_name": path.name,
                "file_size_mb": round(file_size_bytes / (1024.0 * 1024.0), 3),
                "n_jets": n_jets,
                "sampled_jets": sampled,
                "keys": "|".join(keys),
            }

            if "jet_kinematics" in h5f:
                n_files_with_jet_kin += 1
                jk = np.asarray(h5f["jet_kinematics"][idx], dtype=np.float64)
                if jk.ndim == 2 and jk.shape[1] == len(JET_KINEMATICS_NAMES):
                    jet_kin_stats.update(jk)
                else:
                    warnings.append(
                        f"{path.name}: jet_kinematics has unexpected shape {jk.shape}, expected [N,4]"
                    )

            if "jet_tagging" in h5f:
                jt = np.asarray(h5f["jet_tagging"][idx], dtype=np.float64)
                if jt.ndim == 2 and jt.shape[1] == len(JET_TAGGING_NAMES):
                    jet_tag_stats.update(jt)
                else:
                    warnings.append(
                        f"{path.name}: jet_tagging has unexpected shape {jt.shape}, expected [N,13]"
                    )

            if "PFCands" in h5f:
                n_files_with_pf += 1
                pf = np.asarray(h5f["PFCands"][idx], dtype=np.float64)
                if pf.ndim == 3 and pf.shape[2] == len(PFCAND_NAMES):
                    mask = infer_pf_mask(pf)
                    counts = mask.sum(axis=1).astype(np.int64)
                    pf_count_samples.append(counts)
                    valid_pf = pf[mask]
                    if valid_pf.ndim == 1:
                        valid_pf = valid_pf.reshape(1, -1)
                    pf_stats.update(valid_pf)
                    row["pf_max_constits"] = int(pf.shape[1])
                    row["pf_mean_constits_sample"] = float(counts.mean()) if counts.size else 0.0
                    row["pf_padding_frac_sample"] = float(1.0 - mask.mean()) if mask.size else 0.0
                else:
                    warnings.append(
                        f"{path.name}: PFCands has unexpected shape {pf.shape}, expected [N,C,11]"
                    )

            if "event_info" in h5f:
                ev = np.asarray(h5f["event_info"][idx], dtype=np.float64)
                if ev.ndim == 2 and ev.shape[1] == len(EVENT_INFO_NAMES):
                    event_stats.update(ev)
                    total_events_sampled += int(ev.shape[0])
                    unique_n = int(np.unique(ev.astype(np.int64), axis=0).shape[0])
                    duplicate_events_in_sample += int(ev.shape[0] - unique_n)
                    row["sample_duplicate_events"] = int(ev.shape[0] - unique_n)
                else:
                    warnings.append(
                        f"{path.name}: event_info has unexpected shape {ev.shape}, expected [N,3]"
                    )

            file_rows.append(row)

    split_assignments = assign_splits(files, (frac_train, frac_val, frac_test), args.seed)
    for row in file_rows:
        row["split"] = split_assignments[row["file"]]

    pf_counts_concat = np.concatenate(pf_count_samples) if pf_count_samples else np.empty((0,), dtype=np.int64)
    pf_count_summary = {}
    if pf_counts_concat.size:
        pf_count_summary = {
            "count": int(pf_counts_concat.size),
            "mean": float(np.mean(pf_counts_concat)),
            "std": float(np.std(pf_counts_concat)),
            "min": int(np.min(pf_counts_concat)),
            "p50": float(np.percentile(pf_counts_concat, 50)),
            "p90": float(np.percentile(pf_counts_concat, 90)),
            "p95": float(np.percentile(pf_counts_concat, 95)),
            "p99": float(np.percentile(pf_counts_concat, 99)),
            "max": int(np.max(pf_counts_concat)),
        }

    schema_summary = {}
    schema_consistent = True
    for key, sig_set in schema_by_key.items():
        sigs = sorted(
            [{"shape_tail": list(sig[0]), "dtype": sig[1]} for sig in sig_set],
            key=lambda x: (x["shape_tail"], x["dtype"]),
        )
        schema_summary[key] = sigs
        if len(sig_set) > 1:
            schema_consistent = False
            warnings.append(f"Inconsistent schema across files for key '{key}': {sigs}")

    split_jet_totals = {"train": 0, "val": 0, "test": 0}
    split_file_totals = {"train": 0, "val": 0, "test": 0}
    for row in file_rows:
        split = str(row["split"])
        split_jet_totals[split] += int(row["n_jets"])
        split_file_totals[split] += 1

    loader_recipe = {
        "status": {
            "has_explicit_class_labels": bool(label_keys_found),
            "label_keys_found": sorted(label_keys_found),
            "dataset_is_unlabeled_for_supervised_classification": not bool(label_keys_found),
        },
        "expected_keys_for_aojprocessing_output": ["event_info", "jet_kinematics", "jet_tagging", "PFCands"],
        "mask_recommendation": {
            "expression": "mask = (abs(PFCands[..., :4]).sum(axis=-1) > 0)",
            "reason": "AOJProcessing zero-pads constituent rows; non-zero 4-momentum indicates valid constituent.",
        },
        "feature_names": {
            "event_info": EVENT_INFO_NAMES,
            "jet_kinematics": JET_KINEMATICS_NAMES,
            "jet_tagging": JET_TAGGING_NAMES,
            "PFCands": PFCAND_NAMES,
        },
        "split_plan": {
            "seed": args.seed,
            "fractions": {"train": frac_train, "val": frac_val, "test": frac_test},
            "assignment_basis": "file-level deterministic hash on file name",
            "split_file_counts": split_file_totals,
            "split_jet_counts": split_jet_totals,
        },
    }

    summary = {
        "data_dir": str(args.data_dir),
        "glob": args.glob,
        "files_inspected": len(files),
        "total_jets": int(total_jets),
        "sample_jets_per_file": int(args.sample_jets_per_file),
        "keys_union": sorted(keys_union),
        "schema_consistent": schema_consistent,
        "schema_by_key": schema_summary,
        "loader_recipe": loader_recipe,
        "jet_kinematics_stats": jet_kin_stats.to_summary(JET_KINEMATICS_NAMES),
        "jet_tagging_stats": jet_tag_stats.to_summary(JET_TAGGING_NAMES),
        "pfcand_valid_stats": pf_stats.to_summary(PFCAND_NAMES),
        "pfcand_count_per_jet_summary": pf_count_summary,
        "event_info_stats": event_stats.to_summary(EVENT_INFO_NAMES),
        "duplicate_events_in_sample": int(duplicate_events_in_sample),
        "total_events_sampled": int(total_events_sampled),
        "duplicate_event_fraction_in_sample": (
            float(duplicate_events_in_sample / total_events_sampled) if total_events_sampled else 0.0
        ),
        "files_with_PFCands": n_files_with_pf,
        "files_with_jet_kinematics": n_files_with_jet_kin,
        "warnings": warnings,
    }

    summary_path = run_dir / "probe_summary.json"
    schema_path = run_dir / "schema_summary.json"
    recipe_path = run_dir / "loader_recipe.json"
    manifest_path = run_dir / "file_manifest.csv"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with schema_path.open("w", encoding="utf-8") as f:
        json.dump(schema_summary, f, indent=2, sort_keys=True)
    with recipe_path.open("w", encoding="utf-8") as f:
        json.dump(loader_recipe, f, indent=2, sort_keys=True)

    manifest_fields = [
        "split",
        "file_name",
        "file",
        "file_size_mb",
        "n_jets",
        "sampled_jets",
        "pf_max_constits",
        "pf_mean_constits_sample",
        "pf_padding_frac_sample",
        "sample_duplicate_events",
        "keys",
    ]
    for row in file_rows:
        for field in manifest_fields:
            row.setdefault(field, "")
    write_csv(manifest_path, manifest_fields, file_rows)

    print("=" * 72)
    print("AspenOpenJets probe complete")
    print(f"Run directory: {run_dir}")
    print(f"Files inspected: {len(files)}")
    print(f"Total jets (all inspected files): {total_jets}")
    print(f"Keys found: {', '.join(sorted(keys_union))}")
    print(
        "Explicit labels present: "
        f"{loader_recipe['status']['has_explicit_class_labels']} "
        f"(keys: {loader_recipe['status']['label_keys_found']})"
    )
    if pf_count_summary:
        print(
            "Constituent count/jet (sampled): "
            f"mean={pf_count_summary['mean']:.2f}, "
            f"p95={pf_count_summary['p95']:.2f}, "
            f"max={pf_count_summary['max']}"
        )
    print("Recommended mask: mask = (abs(PFCands[..., :4]).sum(axis=-1) > 0)")
    print("Wrote:")
    print(f"  - {summary_path}")
    print(f"  - {schema_path}")
    print(f"  - {recipe_path}")
    print(f"  - {manifest_path}")
    if warnings:
        print(f"Warnings: {len(warnings)} (see probe_summary.json)")
    print("=" * 72)


if __name__ == "__main__":
    main()
