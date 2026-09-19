"""Side-by-side comparison table + training curves for models already trained
via `python -m recsys.training`. Reads the history/test-metrics JSON that
training.py writes to artifacts/, so it has no torch/torch_geometric import
of its own -- keeps matplotlib out of the core package (and out of serve.py's
dependency tree later).

    python scripts/compare_models.py --artifacts-dir artifacts
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

KS = (5, 10, 20)
MODEL_NAMES = ("lightgcn", "mf")
DISPLAY_NAMES = {"lightgcn": "LightGCN", "mf": "MF"}


def load_results(artifacts_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    history_rows = []
    test_metrics = {}
    for name in MODEL_NAMES:
        history_path = artifacts_dir / f"{name}_history.json"
        metrics_path = artifacts_dir / f"{name}_test_metrics.json"
        if history_path.exists():
            history_rows.extend(json.loads(history_path.read_text()))
        if metrics_path.exists():
            test_metrics[DISPLAY_NAMES[name]] = json.loads(metrics_path.read_text())

    hist_df = pd.DataFrame(history_rows)
    results_df = pd.DataFrame(test_metrics).T
    if not results_df.empty:
        results_df = results_df[[f"{m}@{k}" for m in ("recall", "ndcg") for k in KS if f"{m}@{k}" in results_df.columns]]
    return hist_df, results_df


def plot_training_curves(hist_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for name, group in hist_df.groupby("model"):
        axes[0].plot(group.epoch, group.loss, marker="o", label=name)
        axes[1].plot(group.epoch, group["ndcg@10"], marker="o", label=name)
        axes[2].plot(group.epoch, group["recall@10"], marker="o", label=name)
    for ax, title in zip(axes, ["Training loss", "Val NDCG@10", "Val Recall@10"]):
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


def plot_bar_comparison(results_df: pd.DataFrame, out_path: Path) -> None:
    results_df.T.plot(kind="bar", figsize=(10, 6))
    plt.title("Model Performance Comparison by Metric")
    plt.ylabel("Score")
    plt.xlabel("Metrics")
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    hist_df, results_df = load_results(args.artifacts_dir)
    if results_df.empty:
        raise SystemExit(f"No *_test_metrics.json found under {args.artifacts_dir} -- train both models first.")

    print(results_df)
    if not hist_df.empty:
        plot_training_curves(hist_df, args.artifacts_dir / "training_curves.png")
    plot_bar_comparison(results_df, args.artifacts_dir / "comparison_bar.png")


if __name__ == "__main__":
    main()
