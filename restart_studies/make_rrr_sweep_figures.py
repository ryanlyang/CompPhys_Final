#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "download_results" / "results"
FIGURE_DIR = PROJECT_ROOT / "images"

BASELINE_ACC = 0.7425
BASELINE_STRONG3 = 0.6199
PLANNED_LAMBDAS = [1.0, 10.0]
PLANNED_MASKS = [0.05, 0.10, 0.20, 0.30]
SOURCES = ["input_grad", "integrated_gradients", "smoothgrad"]
SOURCE_LABELS = {
    "input_grad": "Input gradients",
    "integrated_gradients": "Integrated Gradients",
    "smoothgrad": "SmoothGrad",
}


def load_completed_runs() -> tuple[pd.DataFrame, pd.DataFrame]:
    best_rows: list[dict[str, object]] = []
    iter_rows: list[dict[str, object]] = []

    for run_dir in sorted(RESULTS_ROOT.glob("rrr_findanother_seed52_*_lam*_mask*")):
        summary_path = run_dir / "summary.json"
        iter_path = run_dir / "iteration_summary.csv"
        if not summary_path.is_file() or not iter_path.is_file():
            continue

        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        df = pd.read_csv(iter_path)
        if len(df) != 5:
            continue
        if not {"aspen_strong3_mean", "test_acc", "test_auc_macro_ovr"}.issubset(df.columns):
            continue

        best_record = summary.get("best_iteration_record") or {}
        if not best_record:
            idx = df["aspen_strong3_mean"].astype(float).idxmin()
            best_record = df.loc[idx].to_dict()

        a_source = str(summary["a_source"])
        lambda_rrr = float(summary["lambda_rrr"])
        mask_frac = float(summary["mask_frac"])
        best_iter = int(best_record["iteration"])
        if lambda_rrr not in PLANNED_LAMBDAS:
            continue

        best_rows.append(
            {
                "run_name": run_dir.name,
                "a_source": a_source,
                "a_source_label": SOURCE_LABELS.get(a_source, a_source),
                "lambda_rrr": lambda_rrr,
                "mask_frac": mask_frac,
                "best_iteration": best_iter,
                "jetclass_test_acc": float(best_record["test_acc"]),
                "jetclass_test_auc": float(best_record["test_auc_macro_ovr"]),
                "aspen_strong3": float(best_record["aspen_strong3_mean"]),
            }
        )

        for _, row in df.iterrows():
            iter_rows.append(
                {
                    "run_name": run_dir.name,
                    "a_source": a_source,
                    "a_source_label": SOURCE_LABELS.get(a_source, a_source),
                    "lambda_rrr": lambda_rrr,
                    "mask_frac": mask_frac,
                    "iteration": int(row["iteration"]),
                    "jetclass_test_acc": float(row["test_acc"]),
                    "jetclass_test_auc": float(row["test_auc_macro_ovr"]),
                    "aspen_strong3": float(row["aspen_strong3_mean"]),
                }
            )

    best_df = pd.DataFrame(best_rows)
    iter_df = pd.DataFrame(iter_rows)
    if best_df.empty or iter_df.empty:
        raise RuntimeError(f"No completed RRR runs found under {RESULTS_ROOT}")
    return best_df, iter_df


def save_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_tradeoff_scatter(best_df: pd.DataFrame) -> None:
    plot_df = best_df.copy()
    colors = {
        "input_grad": "#2b6cb0",
        "integrated_gradients": "#dd6b20",
        "smoothgrad": "#2f855a",
    }
    markers = {1.0: "o", 10.0: "s"}

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for source in SOURCES:
        sub = plot_df[plot_df["a_source"] == source]
        for lam, lam_sub in sub.groupby("lambda_rrr"):
            sizes = 65 + 360 * lam_sub["mask_frac"].astype(float)
            ax.scatter(
                lam_sub["jetclass_test_acc"],
                lam_sub["aspen_strong3"],
                s=sizes,
                c=colors.get(source, "tab:gray"),
                marker=markers.get(float(lam), "o"),
                edgecolor="black",
                linewidth=0.6,
                alpha=0.86,
                label=f"{SOURCE_LABELS.get(source, source)}, $\\lambda={lam:g}$",
            )

    ax.scatter(
        [BASELINE_ACC],
        [BASELINE_STRONG3],
        marker="*",
        s=240,
        c="black",
        edgecolor="white",
        linewidth=0.7,
        label="Baseline",
        zorder=5,
    )
    best = plot_df.loc[plot_df["aspen_strong3"].astype(float).idxmin()]
    ax.annotate(
        "best shift",
        xy=(best["jetclass_test_acc"], best["aspen_strong3"]),
        xytext=(18, 20),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "lw": 0.8},
        fontsize=9,
    )

    ax.axhline(BASELINE_STRONG3, color="black", linestyle="--", linewidth=0.9, alpha=0.55)
    ax.axvline(BASELINE_ACC, color="black", linestyle="--", linewidth=0.9, alpha=0.55)
    ax.set_xlabel("JetClass test accuracy")
    ax.set_ylabel("AspenOpenJets strong3 shift (lower is better)")
    ax.set_title("Focused RRR sweep: in-domain accuracy vs. target-domain shift")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0.66, 0.75)
    ax.set_ylim(0.44, max(0.76, float(plot_df["aspen_strong3"].max()) + 0.03))
    ax.legend(fontsize=7.2, ncol=2, frameon=True, loc="upper left")
    save_figure(fig, "rrr_sweep_tradeoff")


def make_heatmaps(best_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.1), sharey=True)
    vmin = min(0.46, float(best_df["aspen_strong3"].min()) - 0.01)
    vmax = max(0.72, float(best_df["aspen_strong3"].max()) + 0.01)
    cmap = plt.cm.viridis_r.copy()
    cmap.set_bad(color="#e8e8e8")

    image = None
    for ax, source in zip(axes, SOURCES):
        grid = np.full((len(PLANNED_LAMBDAS), len(PLANNED_MASKS)), np.nan, dtype=float)
        for i, lam in enumerate(PLANNED_LAMBDAS):
            for j, mask in enumerate(PLANNED_MASKS):
                sub = best_df[
                    (best_df["a_source"] == source)
                    & np.isclose(best_df["lambda_rrr"], lam)
                    & np.isclose(best_df["mask_frac"], mask)
                ]
                if len(sub):
                    grid[i, j] = float(sub.iloc[0]["aspen_strong3"])

        image = ax.imshow(np.ma.masked_invalid(grid), vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
        ax.set_title(SOURCE_LABELS[source])
        ax.set_xticks(range(len(PLANNED_MASKS)))
        ax.set_xticklabels([f"{m:.2f}" for m in PLANNED_MASKS])
        ax.set_yticks(range(len(PLANNED_LAMBDAS)))
        ax.set_yticklabels([f"{lam:g}" for lam in PLANNED_LAMBDAS])
        ax.set_xlabel("Mask fraction")
        for i in range(len(PLANNED_LAMBDAS)):
            for j in range(len(PLANNED_MASKS)):
                if np.isfinite(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=8, color="white")
                else:
                    ax.text(j, i, "--", ha="center", va="center", fontsize=8, color="#666666")
        ax.tick_params(length=0)
    axes[0].set_ylabel("$\\lambda_{\\mathrm{RRR}}$")
    fig.suptitle("Best Aspen strong3 by focused hyperparameter cell", y=1.02)
    cbar = fig.colorbar(image, ax=axes, fraction=0.035, pad=0.025)
    cbar.set_label("Aspen strong3 (lower is better)")
    save_figure(fig, "rrr_sweep_heatmaps")


def make_iteration_trajectories(iter_df: pd.DataFrame) -> None:
    selected = [
        ("rrr_findanother_seed52_smoothgrad_lam10_mask0p05", "SmoothGrad, $\\lambda=10$, mask $0.05$"),
        ("rrr_findanother_seed52_input_grad_lam1_mask0p30", "Input grad, $\\lambda=1$, mask $0.30$"),
        ("rrr_findanother_seed52_integrated_gradients_lam10_mask0p05", "IG, $\\lambda=10$, mask $0.05$"),
    ]
    colors = ["#2f855a", "#2b6cb0", "#dd6b20"]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), sharex=True)
    for (run_name, label), color in zip(selected, colors):
        sub = iter_df[iter_df["run_name"] == run_name].sort_values("iteration")
        if sub.empty:
            continue
        axes[0].plot(sub["iteration"], sub["aspen_strong3"], marker="o", color=color, label=label)
        axes[1].plot(sub["iteration"], sub["jetclass_test_acc"], marker="o", color=color, label=label)

    axes[0].axhline(BASELINE_STRONG3, color="black", linestyle="--", linewidth=0.9, alpha=0.6)
    axes[1].axhline(BASELINE_ACC, color="black", linestyle="--", linewidth=0.9, alpha=0.6)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Aspen strong3 (lower is better)")
    axes[1].set_ylabel("JetClass test accuracy")
    for ax in axes:
        ax.set_xlabel("Find-another iteration")
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.grid(True, alpha=0.25)
    axes[0].set_title("Target-domain shift")
    axes[1].set_title("In-domain accuracy")
    axes[0].legend(fontsize=8, loc="best")
    save_figure(fig, "rrr_iteration_trajectories")


def main() -> None:
    best_df, iter_df = load_completed_runs()
    save_csv(FIGURE_DIR / "rrr_sweep_completed_best_points.csv", best_df.sort_values("aspen_strong3"))
    save_csv(FIGURE_DIR / "rrr_sweep_completed_iterations.csv", iter_df)
    make_tradeoff_scatter(best_df)
    make_heatmaps(best_df)
    make_iteration_trajectories(iter_df)
    print(f"Completed configs: {len(best_df)}")
    print(f"Wrote figures to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
