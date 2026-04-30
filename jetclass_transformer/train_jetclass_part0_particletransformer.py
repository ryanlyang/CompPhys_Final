#!/usr/bin/env python3
"""
Train ParticleTransformer on a JetClass subset directory (e.g. jetclass_part0)
by building train/val/test splits directly from the files in that folder.

Default split is per class by file count: 8 train / 1 val / 1 test.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import shlex
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path


FILE_RE = re.compile(r"^(?P<cls>[A-Za-z0-9]+)_(?P<idx>\d{3})\.root$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ParticleTransformer on jetclass_part0 with internal split")
    p.add_argument(
        "--data_dir",
        type=Path,
        default=Path("/home/ryan/ComputerScience/ATLAS/HLT_Reco/ATLAS-top-tagging-open-data/data/jetclass_part0"),
        help="Directory containing JetClass ROOT files like HToBB_000.root",
    )
    p.add_argument(
        "--jetclass_repo",
        type=Path,
        default=Path("/home/ryan/ComputerScience/ATLAS/HLT_Reco/jetclass_transformer"),
        help="Path to jetclass_transformer repo",
    )
    p.add_argument("--feature_type", type=str, default="full", choices=["kin", "kinpid", "full"])

    p.add_argument("--train_files_per_class", type=int, default=8)
    p.add_argument("--val_files_per_class", type=int, default=1)
    p.add_argument("--test_files_per_class", type=int, default=1)
    p.add_argument("--shuffle_files", action="store_true", help="Shuffle file order within each class before splitting")
    p.add_argument("--seed", type=int, default=52)

    p.add_argument("--num_epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--start_lr", type=float, default=1e-3)
    p.add_argument("--samples_per_epoch", type=int, default=200000)
    p.add_argument("--samples_per_epoch_val", type=int, default=50000)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--fetch_step", type=float, default=0.05)
    p.add_argument("--gpus", type=str, default="0")

    p.add_argument("--use_amp", action="store_true", default=True)
    p.add_argument("--no_use_amp", action="store_false", dest="use_amp")

    p.add_argument(
        "--save_root",
        type=Path,
        default=Path("checkpoints/jetclass_part0_part"),
        help="Root directory for model checkpoints",
    )
    p.add_argument(
        "--log_file",
        type=Path,
        default=Path("offline_reconstructor_logs/jetclass_part0_particletransformer.log"),
        help="Weaver training log path",
    )
    p.add_argument("--run_name", type=str, default="part0_pt")
    p.add_argument("--predict_output", type=Path, default=Path("pred_jetclass_part0.root"))
    p.add_argument("--tensorboard", type=str, default="JetClass_part0_ParT")

    p.add_argument("--dry_run", action="store_true")
    p.add_argument(
        "--extra_args",
        nargs=argparse.REMAINDER,
        help="Extra args passed to weaver (use as: --extra_args --flag value ...)",
    )
    return p.parse_args()


def collect_files(data_dir: Path) -> dict[str, list[tuple[int, Path]]]:
    by_class: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for p in sorted(data_dir.glob("*.root")):
        m = FILE_RE.match(p.name)
        if not m:
            continue
        cls = m.group("cls")
        idx = int(m.group("idx"))
        by_class[cls].append((idx, p.resolve()))

    if not by_class:
        raise RuntimeError(f"No matching ROOT files found in {data_dir}")

    for cls in by_class:
        by_class[cls].sort(key=lambda x: x[0])
    return by_class


def split_by_class(
    by_class: dict[str, list[tuple[int, Path]]],
    n_tr: int,
    n_va: int,
    n_te: int,
    shuffle: bool,
    seed: int,
) -> tuple[list[tuple[str, Path]], list[Path], list[tuple[str, Path]]]:
    rng = random.Random(seed)
    train_labeled: list[tuple[str, Path]] = []
    val_unlabeled: list[Path] = []
    test_labeled: list[tuple[str, Path]] = []

    need = n_tr + n_va + n_te
    for cls, items in sorted(by_class.items()):
        if len(items) < need:
            raise ValueError(
                f"Class {cls} has {len(items)} files, needs at least {need} for split {n_tr}/{n_va}/{n_te}."
            )

        paths = [p for _, p in items]
        if shuffle:
            rng.shuffle(paths)

        tr = paths[:n_tr]
        va = paths[n_tr : n_tr + n_va]
        te = paths[n_tr + n_va : n_tr + n_va + n_te]

        train_labeled.extend((cls, p) for p in tr)
        val_unlabeled.extend(va)
        test_labeled.extend((cls, p) for p in te)

    return train_labeled, val_unlabeled, test_labeled


def resolve_weaver_command() -> list[str]:
    weaver_env = os.environ.get("WEAVER_BIN")
    if weaver_env and Path(weaver_env).is_file() and os.access(weaver_env, os.X_OK):
        return [weaver_env]

    weaver_bin = shutil.which("weaver")
    if weaver_bin:
        return [weaver_bin]

    user_weaver = Path.home() / ".local" / "bin" / "weaver"
    if user_weaver.is_file() and os.access(str(user_weaver), os.X_OK):
        return [str(user_weaver)]

    raise FileNotFoundError(
        "Could not find 'weaver' executable. Install with 'python -m pip install --user weaver-core' "
        "and ensure ~/.local/bin is on PATH, or set WEAVER_BIN to the executable path."
    )


def main() -> None:
    args = parse_args()

    args.data_dir = args.data_dir.resolve()
    args.jetclass_repo = args.jetclass_repo.resolve()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {args.data_dir}")
    if not args.jetclass_repo.exists():
        raise FileNotFoundError(f"jetclass_repo not found: {args.jetclass_repo}")

    network_cfg = args.jetclass_repo / "networks" / "example_ParticleTransformer.py"
    data_cfg = args.jetclass_repo / "data" / "JetClass" / f"JetClass_{args.feature_type}.yaml"

    if not network_cfg.exists():
        raise FileNotFoundError(f"Missing network config: {network_cfg}")
    if not data_cfg.exists():
        raise FileNotFoundError(f"Missing data config: {data_cfg}")

    by_class = collect_files(args.data_dir)
    train_labeled, val_unlabeled, test_labeled = split_by_class(
        by_class=by_class,
        n_tr=int(args.train_files_per_class),
        n_va=int(args.val_files_per_class),
        n_te=int(args.test_files_per_class),
        shuffle=bool(args.shuffle_files),
        seed=int(args.seed),
    )

    print("Detected classes and file counts:")
    for cls, items in sorted(by_class.items()):
        print(f"  {cls:12s} : {len(items)} files")

    print("Split summary:")
    print(f"  Train files: {len(train_labeled)}")
    print(f"  Val files  : {len(val_unlabeled)}")
    print(f"  Test files : {len(test_labeled)}")

    model_prefix = (args.save_root / args.run_name / "net").resolve()
    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.log_file.parent.mkdir(parents=True, exist_ok=True)

    weaver_launcher = resolve_weaver_command()
    cmd: list[str] = [*weaver_launcher, "--data-train"]
    cmd.extend([f"{cls}:{path}" for cls, path in train_labeled])

    cmd.append("--data-val")
    cmd.extend([str(p) for p in val_unlabeled])

    cmd.append("--data-test")
    cmd.extend([f"{cls}:{path}" for cls, path in test_labeled])

    cmd.extend(
        [
            "--data-config",
            str(data_cfg),
            "--network-config",
            str(network_cfg),
            "--model-prefix",
            str(model_prefix),
            "--num-workers",
            str(args.num_workers),
            "--fetch-step",
            str(args.fetch_step),
            "--batch-size",
            str(args.batch_size),
            "--samples-per-epoch",
            str(args.samples_per_epoch),
            "--samples-per-epoch-val",
            str(args.samples_per_epoch_val),
            "--num-epochs",
            str(args.num_epochs),
            "--gpus",
            str(args.gpus),
            "--start-lr",
            str(args.start_lr),
            "--optimizer",
            "ranger",
            "--log",
            str(args.log_file),
            "--predict-output",
            str(args.predict_output),
            "--tensorboard",
            args.tensorboard,
        ]
    )

    if args.use_amp:
        cmd.append("--use-amp")

    if args.extra_args:
        extra = args.extra_args
        if extra and extra[0] == "--":
            extra = extra[1:]
        cmd.extend(extra)

    print(f"\nUsing weaver launcher: {' '.join(shlex.quote(x) for x in weaver_launcher)}")
    print("\nWeaver command:")
    print(" ".join(shlex.quote(x) for x in cmd))

    if args.dry_run:
        print("\nDry run requested, exiting without training.")
        return

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
