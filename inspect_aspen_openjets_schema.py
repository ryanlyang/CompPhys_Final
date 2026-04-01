#!/usr/bin/env python3

"""
Inspect AspenOpenJets HDF5 schema and estimate compatibility with JetClass
ParticleTransformer input requirements (`kin`, `kinpid`, `full`).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import h5py
except ModuleNotFoundError:
    h5py = None


@dataclass
class DatasetEntry:
    file: str
    path: str
    shape: Tuple[int, ...]
    dtype: str
    fields: Tuple[str, ...]
    attr_keys: Tuple[str, ...]
    attr_hints: Tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect AspenOpenJets schema and compatibility.")
    parser.add_argument(
        "--aspen-dir",
        default="/home/ryreu/atlas/CompPhys_Final/data/AspenOpenJets",
        help="Directory containing AspenOpenJets *.h5 files.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/aspen_schema_probe",
        help="Output directory for schema reports.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=4,
        help="Maximum number of H5 files to scan.",
    )
    parser.add_argument(
        "--preview-elements",
        type=int,
        default=2048,
        help="Number of flattened elements to sample for lightweight min/max checks.",
    )
    parser.add_argument(
        "--max-key-preview",
        type=int,
        default=120,
        help="Number of key candidates to print in text report preview.",
    )
    return parser.parse_args()


def norm_key(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("-", "_").replace(".", "_").replace("/", "_")
    s = re.sub(r"__+", "_", s)
    return s


def collect_entries(file_path: Path) -> List[DatasetEntry]:
    out: List[DatasetEntry] = []

    def _decode_maybe_bytes(x):
        if isinstance(x, (bytes, bytearray)):
            try:
                return x.decode("utf-8", errors="ignore")
            except Exception:
                return str(x)
        return x

    def _attr_hint(name: str, val) -> str:
        name_l = str(name).lower()
        decoded = _decode_maybe_bytes(val)
        if isinstance(decoded, np.ndarray):
            flat = decoded.reshape(-1)
            sample = [_decode_maybe_bytes(v) for v in flat[:12].tolist()]
            if flat.size <= 12:
                return f"{name}={sample}"
            return f"{name}[:12]={sample} (len={int(flat.size)})"
        if isinstance(decoded, (list, tuple)):
            sample = [_decode_maybe_bytes(v) for v in decoded[:12]]
            if len(decoded) <= 12:
                return f"{name}={sample}"
            return f"{name}[:12]={sample} (len={len(decoded)})"
        if isinstance(decoded, (str, np.str_)):
            txt = str(decoded).strip()
            if len(txt) > 200:
                txt = txt[:200] + "..."
            return f"{name}={txt}"
        return f"{name}={decoded}"

    with h5py.File(file_path, "r") as f:
        def visitor(name: str, obj) -> None:
            if isinstance(obj, h5py.Dataset):
                fields: Tuple[str, ...] = tuple(obj.dtype.names) if obj.dtype.names else tuple()
                attr_keys = tuple(sorted(str(k) for k in obj.attrs.keys()))
                attr_hints: List[str] = []
                for ak in attr_keys:
                    al = ak.lower()
                    # Prioritize attribute names that commonly store column metadata.
                    if any(
                        tok in al
                        for tok in ("feature", "column", "name", "label", "schema", "field", "branch")
                    ):
                        try:
                            attr_hints.append(_attr_hint(ak, obj.attrs[ak]))
                        except Exception:
                            attr_hints.append(f"{ak}=<unreadable>")
                out.append(
                    DatasetEntry(
                        file=file_path.name,
                        path=name,
                        shape=tuple(obj.shape),
                        dtype=str(obj.dtype),
                        fields=fields,
                        attr_keys=attr_keys,
                        attr_hints=tuple(attr_hints),
                    )
                )
        f.visititems(visitor)
    return out


def find_candidates(
    all_keys: Sequence[str],
    aliases: Sequence[str],
) -> List[str]:
    alias_norm = [norm_key(a) for a in aliases]
    scored: List[Tuple[int, int, str]] = []
    for k in all_keys:
        nk = norm_key(k)
        best = 0
        for a in alias_norm:
            if nk == a:
                best = max(best, 3)
            elif nk.endswith("_" + a) or nk.startswith(a + "_") or nk.endswith(a):
                best = max(best, 2)
            elif a in nk:
                best = max(best, 1)
        if best > 0:
            scored.append((best, -len(k), k))
    scored.sort(reverse=True)
    return [k for _, _, k in scored]


def best_or_empty(cands: Sequence[str]) -> str:
    return cands[0] if cands else ""


def detect_mapping(all_keys: Sequence[str]) -> Dict[str, Dict[str, object]]:
    alias_map = {
        "part_px": ["part_px", "particles_px", "pf_px", "pfcands_px", "cand_px", "px"],
        "part_py": ["part_py", "particles_py", "pf_py", "pfcands_py", "cand_py", "py"],
        "part_pz": ["part_pz", "particles_pz", "pf_pz", "pfcands_pz", "cand_pz", "pz"],
        "part_energy": [
            "part_energy",
            "particles_energy",
            "pf_energy",
            "pfcands_energy",
            "cand_energy",
            "energy",
        ],
        "part_pt": ["part_pt", "particles_pt", "pf_pt", "pfcands_pt", "cand_pt", "pt"],
        "part_eta": ["part_eta", "particles_eta", "pf_eta", "pfcands_eta", "cand_eta", "eta"],
        "part_phi": ["part_phi", "particles_phi", "pf_phi", "pfcands_phi", "cand_phi", "phi"],
        "jet_pt": ["jet_pt", "jets_pt", "ak8_pt", "fatjet_pt"],
        "jet_eta": ["jet_eta", "jets_eta", "ak8_eta", "fatjet_eta"],
        "jet_phi": ["jet_phi", "jets_phi", "ak8_phi", "fatjet_phi"],
        "jet_energy": ["jet_energy", "jets_energy", "jet_e", "ak8_energy", "fatjet_energy"],
        "part_charge": ["part_charge", "particles_charge", "pf_charge", "pfcands_charge", "charge"],
        "part_isChargedHadron": [
            "part_ischargedhadron",
            "part_is_charged_hadron",
            "is_charged_hadron",
        ],
        "part_isNeutralHadron": [
            "part_isneutralhadron",
            "part_is_neutral_hadron",
            "is_neutral_hadron",
        ],
        "part_isPhoton": ["part_isphoton", "part_is_photon", "is_photon"],
        "part_isElectron": ["part_iselectron", "part_is_electron", "is_electron"],
        "part_isMuon": ["part_ismuon", "part_is_muon", "is_muon"],
        "part_d0val": ["part_d0val", "d0", "track_d0"],
        "part_d0err": ["part_d0err", "d0err", "track_d0err"],
        "part_dzval": ["part_dzval", "dz", "track_dz"],
        "part_dzerr": ["part_dzerr", "dzerr", "track_dzerr"],
        "part_pdgid": ["part_pdgid", "pdgid", "particle_id", "part_pid", "pid"],
    }

    mapping: Dict[str, Dict[str, object]] = {}
    for canonical, aliases in alias_map.items():
        cands = find_candidates(all_keys, aliases)
        mapping[canonical] = {
            "matched_key": best_or_empty(cands),
            "num_candidates": len(cands),
            "top_candidates": cands[:5],
        }
    return mapping


def summarize_shapes(entries: Sequence[DatasetEntry]) -> Dict[str, object]:
    rank_counts: Dict[str, int] = {}
    dtype_counts: Dict[str, int] = {}
    first_dim_values: List[int] = []
    for e in entries:
        rank_counts[str(len(e.shape))] = rank_counts.get(str(len(e.shape)), 0) + 1
        dtype_counts[e.dtype] = dtype_counts.get(e.dtype, 0) + 1
        if len(e.shape) >= 1 and e.shape[0] > 0:
            first_dim_values.append(int(e.shape[0]))
    return {
        "rank_counts": rank_counts,
        "dtype_counts": dtype_counts,
        "first_dim_min": int(min(first_dim_values)) if first_dim_values else None,
        "first_dim_max": int(max(first_dim_values)) if first_dim_values else None,
    }


def _resolve_dataset_view(f, key: str):
    # Direct dataset path
    if key in f and isinstance(f[key], h5py.Dataset):
        return f[key]

    # Structured field syntax: "<dataset_path>.<field>"
    if "." in key:
        ds_path, field = key.rsplit(".", 1)
        if ds_path in f and isinstance(f[ds_path], h5py.Dataset):
            ds = f[ds_path]
            if ds.dtype.names and field in ds.dtype.names:
                return ds[field]

    # Structured field syntax: "<dataset_path>/<field>"
    if "/" in key:
        ds_path, field = key.rsplit("/", 1)
        if ds_path in f and isinstance(f[ds_path], h5py.Dataset):
            ds = f[ds_path]
            if ds.dtype.names and field in ds.dtype.names:
                return ds[field]
    return None


def lightweight_value_checks(
    file_path: Path,
    key_paths: Sequence[str],
    preview_elements: int,
) -> Dict[str, Dict[str, object]]:
    checks: Dict[str, Dict[str, object]] = {}
    with h5py.File(file_path, "r") as f:
        for key in key_paths:
            if not key:
                continue
            ds = _resolve_dataset_view(f, key)
            if ds is None:
                checks[key] = {"missing_in_file": True}
                continue
            if ds.size == 0:
                checks[key] = {"size": int(ds.size), "empty": True}
                continue
            arr = ds.reshape(-1)
            take = min(preview_elements, arr.shape[0])
            vals = np.asarray(arr[:take])
            if np.issubdtype(vals.dtype, np.number):
                checks[key] = {
                    "size": int(ds.size),
                    "preview_n": int(take),
                    "min": float(np.nanmin(vals)),
                    "max": float(np.nanmax(vals)),
                    "mean": float(np.nanmean(vals)),
                    "nan_count": int(np.isnan(vals).sum()) if np.issubdtype(vals.dtype, np.floating) else 0,
                }
            else:
                checks[key] = {
                    "size": int(ds.size),
                    "preview_n": int(take),
                    "non_numeric": True,
                    "dtype": str(vals.dtype),
                }
    return checks


def readiness(mapping: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    base_direct = ["part_px", "part_py", "part_pz", "part_energy"]
    base_alt = ["part_pt", "part_eta", "part_phi", "part_energy"]
    jet_base = ["jet_pt", "jet_eta", "jet_phi", "jet_energy"]

    def has(field: str) -> bool:
        return bool(mapping[field]["matched_key"])

    kin_particle_ok = all(has(k) for k in base_direct) or all(has(k) for k in base_alt)
    kin_jet_ok = all(has(k) for k in jet_base)
    kin_missing = []
    if not kin_particle_ok:
        kin_missing.extend(base_direct)
        kin_missing.extend(base_alt)
    if not kin_jet_ok:
        kin_missing.extend([k for k in jet_base if not has(k)])

    kinpid_fields = [
        "part_charge",
        "part_isChargedHadron",
        "part_isNeutralHadron",
        "part_isPhoton",
        "part_isElectron",
        "part_isMuon",
    ]
    kinpid_direct_missing = [k for k in kinpid_fields if not has(k)]
    pdgid_available = has("part_pdgid")
    kinpid_derivable_from_pdgid = pdgid_available and len(kinpid_direct_missing) > 0
    kinpid_ok = kin_particle_ok and kin_jet_ok and (
        len(kinpid_direct_missing) == 0 or kinpid_derivable_from_pdgid
    )

    full_fields = ["part_d0val", "part_d0err", "part_dzval", "part_dzerr"]
    full_missing = [k for k in full_fields if not has(k)]
    full_ok = kinpid_ok and len(full_missing) == 0

    notes = []
    if kin_particle_ok and not all(has(k) for k in base_direct):
        notes.append("Particle 4-vectors appear derivable from pt/eta/phi/energy (no direct px/py/pz match).")
    if kinpid_derivable_from_pdgid:
        notes.append("kinpid PID flags appear derivable from a PDGID/PID field.")
    if not full_ok and len(full_missing) > 0:
        notes.append("full feature set likely unavailable without track displacement branches.")

    return {
        "kin_ready": bool(kin_particle_ok and kin_jet_ok),
        "kinpid_ready": bool(kinpid_ok),
        "full_ready": bool(full_ok),
        "kin_missing_or_conditions": sorted(set(kin_missing)),
        "kinpid_missing_direct": kinpid_direct_missing,
        "kinpid_derivable_from_pdgid": kinpid_derivable_from_pdgid,
        "full_missing": full_missing,
        "notes": notes,
    }


def write_key_table(entries: Sequence[DatasetEntry], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "path",
                "shape",
                "dtype",
                "fields",
                "attr_keys",
                "attr_hints",
            ],
        )
        writer.writeheader()
        for e in entries:
            writer.writerow(
                {
                    "file": e.file,
                    "path": e.path,
                    "shape": str(e.shape),
                    "dtype": e.dtype,
                    "fields": "|".join(e.fields),
                    "attr_keys": "|".join(e.attr_keys),
                    "attr_hints": " || ".join(e.attr_hints),
                }
            )


def main() -> int:
    args = parse_args()
    if h5py is None:
        raise RuntimeError("Missing dependency: h5py. Install it with `pip install h5py`.")

    aspen_dir = Path(args.aspen_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(aspen_dir.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {aspen_dir}")
    files = files[: args.max_files]

    all_entries: List[DatasetEntry] = []
    for fpath in files:
        all_entries.extend(collect_entries(fpath))

    if not all_entries:
        raise RuntimeError(f"No datasets found inside scanned files: {[f.name for f in files]}")

    candidate_keys: List[str] = []
    for e in all_entries:
        candidate_keys.append(e.path)
        candidate_keys.append(e.path.split("/")[-1])
        for fld in e.fields:
            candidate_keys.append(fld)
            candidate_keys.append(f"{e.path}.{fld}")
            candidate_keys.append(f"{e.path}/{fld}")
        for hint in e.attr_hints:
            # Pull out token-like words from attribute hints as additional key candidates.
            for tok in re.split(r"[^A-Za-z0-9_]+", hint):
                if len(tok) >= 3:
                    candidate_keys.append(tok)

    unique_paths = sorted(set(candidate_keys))
    mapping = detect_mapping(unique_paths)
    ready = readiness(mapping)
    shape_summary = summarize_shapes(all_entries)

    # Lightweight value checks for matched keys in first scanned file.
    mapped_keys = [v["matched_key"] for v in mapping.values() if v["matched_key"]]
    value_checks = lightweight_value_checks(files[0], sorted(set(mapped_keys)), args.preview_elements)

    report = {
        "aspen_dir": str(aspen_dir),
        "files_scanned": [str(f) for f in files],
        "num_dataset_entries": len(all_entries),
        "num_unique_dataset_paths": len(set(e.path for e in all_entries)),
        "num_unique_candidate_keys": len(unique_paths),
        "shape_summary": shape_summary,
        "mapping_candidates": mapping,
        "readiness": ready,
        "value_checks_first_file": value_checks,
        "recommendation": (
            "Use kin mapping first."
            if ready["kin_ready"] and not ready["kinpid_ready"]
            else "Use kinpid mapping."
            if ready["kinpid_ready"]
            else "Schema gaps detected; add custom Aspen->JetClass mapping adapter."
        ),
    }

    json_path = out_dir / "aspen_schema_report.json"
    txt_path = out_dir / "aspen_schema_report.txt"
    csv_path = out_dir / "aspen_dataset_keys.csv"

    json_path.write_text(json.dumps(report, indent=2) + "\n")
    write_key_table(all_entries, csv_path)

    lines = []
    lines.append("AspenOpenJets Schema Probe")
    lines.append("=" * 32)
    lines.append(f"Aspen dir: {aspen_dir}")
    lines.append(f"Files scanned ({len(files)}):")
    for f in files:
        lines.append(f"  - {f}")
    lines.append("")
    lines.append("Dataset Path Preview:")
    unique_dataset_paths = sorted(set(e.path for e in all_entries))
    for p in unique_dataset_paths[: args.max_key_preview]:
        lines.append(f"  {p}")
    if len(unique_dataset_paths) > args.max_key_preview:
        lines.append(f"  ... ({len(unique_dataset_paths) - args.max_key_preview} more)")
    lines.append("")
    lines.append("Attribute Hint Preview:")
    shown_hints = 0
    for e in all_entries:
        if not e.attr_hints:
            continue
        lines.append(f"  [{e.file}] {e.path}")
        for h in e.attr_hints[:6]:
            lines.append(f"    - {h}")
        shown_hints += 1
        if shown_hints >= 20:
            break
    if shown_hints == 0:
        lines.append("  [none detected]")
    lines.append("")
    lines.append("Readiness:")
    lines.append(f"  kin_ready:    {ready['kin_ready']}")
    lines.append(f"  kinpid_ready: {ready['kinpid_ready']}")
    lines.append(f"  full_ready:   {ready['full_ready']}")
    if ready["notes"]:
        lines.append("Notes:")
        for n in ready["notes"]:
            lines.append(f"  - {n}")
    lines.append("")
    lines.append("Recommended next step:")
    lines.append(f"  {report['recommendation']}")
    lines.append("")
    lines.append("Matched keys (top candidate per canonical field):")
    for k in sorted(mapping.keys()):
        mk = mapping[k]["matched_key"]
        lines.append(f"  {k:22s} -> {mk if mk else '[missing]'}")
    lines.append("")
    lines.append("Outputs:")
    lines.append(f"  {json_path}")
    lines.append(f"  {txt_path}")
    lines.append(f"  {csv_path}")
    txt_path.write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
