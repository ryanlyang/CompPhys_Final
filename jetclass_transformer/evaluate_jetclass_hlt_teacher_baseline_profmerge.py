#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""JetClass prof-merge experiment launcher for jetclass_transformer.

This script is self-contained: it loads the baseline evaluator from known
locations and applies monkey-patches for type-agnostic merging.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_base_module():
    here = Path(__file__).resolve().parent
    candidates = [
        here / "evaluate_jetclass_hlt_teacher_baseline.py",
        Path("/home/ryreu/atlas/PracticeTagging/evaluate_jetclass_hlt_teacher_baseline.py"),
        Path("/home/ryreu/atlas/PracticeTagging/ATLAS-top-tagging-open-data/evaluate_jetclass_hlt_teacher_baseline.py"),
        Path("/home/ryreu/atlas/ATLAS-top-tagging-open-data/evaluate_jetclass_hlt_teacher_baseline.py"),
        Path("/home/ryan/ComputerScience/ATLAS/HLT_Reco/ATLAS-top-tagging-open-data/evaluate_jetclass_hlt_teacher_baseline.py"),
        Path("/home/ryan/ComputerScience/ATLAS/ATLAS-top-tagging-open-data/evaluate_jetclass_hlt_teacher_baseline.py"),
    ]
    for p in candidates:
        if not p.is_file():
            continue
        spec = importlib.util.spec_from_file_location("jetclass_hlt_base", str(p))
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        sys.modules["jetclass_hlt_base"] = mod
        spec.loader.exec_module(mod)
        return mod
    raise FileNotFoundError(
        "Could not locate evaluate_jetclass_hlt_teacher_baseline.py. Checked: "
        + ", ".join(str(p) for p in candidates)
    )


def _patch_for_prof_merge(base):
    original_get_type_config = base.get_type_config

    def _allowed_merge_and_output_type_type_agnostic(ti: int, tj: int):
        # Allow any pair except both unknown. Output type follows dominant-energy
        # constituent by returning override=None.
        if ti == base.TYPE_UNK and tj == base.TYPE_UNK:
            return False, None
        return True, None

    def _get_type_config_type_agnostic():
        cfg = original_get_type_config()
        # Flatten pairwise merge settings to be type-agnostic.
        r = np.full_like(cfg["merge_radius"], 0.018, dtype=np.float64)
        p = np.full_like(cfg["merge_prob"], 0.28, dtype=np.float64)
        unk = int(base.TYPE_UNK)
        r[unk, :] = np.minimum(r[unk, :], 0.010)
        r[:, unk] = np.minimum(r[:, unk], 0.010)
        p[unk, :] = np.minimum(p[unk, :], 0.08)
        p[:, unk] = np.minimum(p[:, unk], 0.08)
        cfg["merge_radius"] = r
        cfg["merge_prob"] = p
        return cfg

    base.allowed_merge_and_output_type = _allowed_merge_and_output_type_type_agnostic
    base.get_type_config = _get_type_config_type_agnostic


def main() -> None:
    base = _load_base_module()
    _patch_for_prof_merge(base)
    args = base.parse_args()
    base.run_experiment(args)


if __name__ == "__main__":
    main()
