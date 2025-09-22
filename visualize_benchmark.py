#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot loss-vs-iteration curves from the benchmark CSV.

Produces three kinds of figures:
  1) Best run per optimizer (one line per optimizer)
  2) All runs (each run_name is one line)
  3) One figure per optimizer (all its run_names)

Usage examples:
  python viz_curves.py --csv results/MNIST_runs.csv --dataset MNIST --split val_eval
  python viz_curves.py --csv results/benchmark_runs.csv --dataset shakespeare_char --outdir results/figs --smooth 7
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ---------- Utilities ----------

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # Required columns
    need = {"optimizer_name", "run_name", "split", "iter", "loss"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    # Basic dtypes
    for c in ["iter", "loss", "lr", "seed", "mfu_percent", "dt_ms"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# Filter out the training dataset and the testing dataset
def filter_df(df: pd.DataFrame, dataset: str | None, split: str) -> pd.DataFrame:
    out = df[df["split"] == split].copy()
    if dataset is not None and "dataset" in out.columns:
        out = out[out["dataset"] == dataset]
    return out


def rolling(y: pd.Series, k: int) -> pd.Series:
    if k <= 1:
        return y
    return y.rolling(window=k, min_periods=1, center=False).median()


def select_best_run_per_optimizer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Choose one run per optimizer: the run whose *final* loss is minimal.
    df: (filtered to a single split already)
    Returns a DataFrame with columns [optimizer_name, run_name, final_iter, final_loss].
    """
    # find last iter per (optimizer, run)
    last_iter = df.groupby(["optimizer_name", "run_name"])["iter"].max().reset_index(name="final_iter")
    tail = df.merge(last_iter, on=["optimizer_name", "run_name"])
    tail = tail[tail["iter"] == tail["final_iter"]][["optimizer_name", "run_name", "final_iter", "loss"]]
    tail = tail.rename(columns={"loss": "final_loss"})

    idx = tail.groupby("optimizer_name")["final_loss"].idxmin()
    return tail.loc[idx].sort_values("final_loss", ascending=True).reset_index(drop=True)


def _plot_one_group(ax, g: pd.DataFrame, label: str, smooth: int, lw: float = 1.2):
    gg = g.sort_values("iter")
    x = gg["iter"].to_numpy()
    y = rolling(gg["loss"], smooth).to_numpy()
    ax.plot(x, y, label=label, linewidth=lw)


def style_axes(ax, title: str):
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.grid(True, which="both", axis="both", linestyle="--", alpha=0.25)


# ---------- Plotters ----------

def fig_best_per_optimizer(df: pd.DataFrame, outdir: Path, dataset: str, split: str, smooth: int):
    best = select_best_run_per_optimizer(df)
    fig, ax = plt.subplots(figsize=(10, 5))
    for _, row in best.iterrows():
        sub = df[(df["optimizer_name"] == row["optimizer_name"]) & (df["run_name"] == row["run_name"])]
        label = f"{row['optimizer_name']} ({row['run_name']})"
        _plot_one_group(ax, sub, label, smooth=smooth)
    style_axes(ax, f"Best run per optimizer — [{dataset}] ({split})")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / f"{dataset}_{split}_curves_best_per_optimizer.png", dpi=200)
    plt.close(fig)


def fig_all_runs(df: pd.DataFrame, outdir: Path, dataset: str, split: str, smooth: int):
    fig, ax = plt.subplots(figsize=(11, 6))
    # Each (optimizer, run_name) is its own line
    for (opt, run), g in df.groupby(["optimizer_name", "run_name"], sort=False):
        label = f"{run}"
        _plot_one_group(ax, g, label, smooth=smooth, lw=1.0)
    style_axes(ax, f"All runs — [{dataset}] ({split})")
    # Legend can get huge; show outside if too many labels
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / f"{dataset}_{split}_curves_all_runs.png", dpi=200)
    plt.close(fig)


def figs_per_optimizer(df: pd.DataFrame, outdir: Path, dataset: str, split: str, smooth: int):
    for opt, gopt in df.groupby("optimizer_name", sort=False):
        fig, ax = plt.subplots(figsize=(11, 6))
        for run, grun in gopt.groupby("run_name", sort=False):
            _plot_one_group(ax, grun, run, smooth=smooth, lw=1.2)
        style_axes(ax, f"{opt} — all runs — [{dataset}] ({split})")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / f"{dataset}_{split}_curves_{opt}.png", dpi=200)
        plt.close(fig)


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to benchmark CSV (e.g. results/MNIST_runs.csv)")
    ap.add_argument("--outdir", default="results/figs", help="Directory to save figures")
    ap.add_argument("--dataset", default=None, help="Filter by dataset column (e.g. MNIST)")
    ap.add_argument("--split", default="val_eval", help="Which split to plot (val_eval or train)")
    ap.add_argument("--smooth", type=int, default=1, help="Rolling median window; 1 disables smoothing")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = load_csv(Path(args.csv))
    # Determine dataset tag for titles/filenames
    dataset_tag = args.dataset or (str(df["dataset"].dropna().unique()[0])
                                   if "dataset" in df.columns and df["dataset"].notna().any()
                                   else "all")
    dff = filter_df(df, dataset=args.dataset, split=args.split)
    if dff.empty:
        raise SystemExit(f"No rows after filtering split='{args.split}'"
                         + (f" and dataset='{args.dataset}'" if args.dataset else ""))

    # Remove runs with non-positive losses (log-scale friendly)
    dff = dff[dff["loss"] > 0].copy()
    # Generate the three figure sets
    fig_best_per_optimizer(dff, outdir, dataset_tag, args.split, smooth=args.smooth)
    fig_all_runs(dff, outdir, dataset_tag, args.split, smooth=args.smooth)
    figs_per_optimizer(dff, outdir, dataset_tag, args.split, smooth=args.smooth)

    print(f"Saved figures to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
