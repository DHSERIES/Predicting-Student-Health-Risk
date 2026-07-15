"""Simple, reproducible exploratory data analysis for the health-risk data."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "data" / "train.csv"
TEST_PATH = ROOT / "data" / "test.csv"
OUTPUT_DIR = ROOT / "eda_outputs"


def save_target_plot(train: pd.DataFrame, target: str) -> None:
    counts = train[target].value_counts(dropna=False)
    percentages = counts.div(len(train)).mul(100)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index.astype(str), counts.values, color="#4C78A8")
    ax.set(title="Target distribution", xlabel=target, ylabel="Rows")
    ax.bar_label(
        bars,
        labels=[f"{value:,.0f}\n({pct:.1f}%)" for value, pct in zip(counts, percentages)],
        padding=3,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "target_distribution.png", dpi=150)
    plt.close(fig)


def save_missing_plot(train: pd.DataFrame) -> None:
    missing_pct = train.isna().mean().mul(100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(missing_pct.index[::-1], missing_pct.values[::-1], color="#F58518")
    ax.set(title="Missing values in training data", xlabel="Missing rows (%)")
    ax.set_xlim(0, max(5, missing_pct.max() * 1.15))
    for y, value in enumerate(missing_pct.values[::-1]):
        ax.text(value + 0.3, y, f"{value:.1f}%", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "missing_values.png", dpi=150)
    plt.close(fig)


def save_numeric_distributions(train: pd.DataFrame, numeric_columns: list[str]) -> None:
    plot_columns = [column for column in numeric_columns if column != "id"]
    sample = train[plot_columns].sample(n=min(100_000, len(train)), random_state=42)
    columns_per_row = 3
    rows = int(np.ceil(len(plot_columns) / columns_per_row))
    fig, axes = plt.subplots(rows, columns_per_row, figsize=(14, 3.5 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, column in zip(axes, plot_columns):
        ax.hist(sample[column].dropna(), bins=35, color="#54A24B", alpha=0.85)
        ax.set_title(column.replace("_", " ").title())
        ax.set_ylabel("Rows")
    for ax in axes[len(plot_columns) :]:
        ax.remove()

    fig.suptitle("Numeric feature distributions (sample of up to 100,000 rows)", y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "numeric_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_correlation_plot(train: pd.DataFrame, numeric_columns: list[str]) -> None:
    features = [column for column in numeric_columns if column != "id"]
    correlation = train[features].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(features)), labels=features, rotation=45, ha="right")
    ax.set_yticks(range(len(features)), labels=features)
    ax.set_title("Numeric feature correlations")
    for row in range(len(features)):
        for column in range(len(features)):
            ax.text(column, row, f"{correlation.iloc[row, column]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "correlation_matrix.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    target_candidates = [column for column in train.columns if column not in test.columns]
    if len(target_candidates) != 1:
        raise ValueError(f"Expected one train-only target column, found: {target_candidates}")
    target = target_candidates[0]

    numeric_columns = train.select_dtypes(include="number").columns.tolist()
    categorical_columns = train.select_dtypes(exclude="number").columns.tolist()

    missing = pd.DataFrame(
        {
            "train_missing_count": train.isna().sum(),
            "train_missing_pct": train.isna().mean().mul(100),
            "test_missing_count": test.isna().sum().reindex(train.columns),
            "test_missing_pct": test.isna().mean().mul(100).reindex(train.columns),
        }
    )
    missing["missing_pct_difference"] = missing["test_missing_pct"] - missing["train_missing_pct"]
    missing.sort_values("train_missing_pct", ascending=False).to_csv(OUTPUT_DIR / "missing_values.csv")

    train[numeric_columns].describe().T.to_csv(OUTPUT_DIR / "numeric_summary.csv")

    categorical_rows = []
    for column in categorical_columns:
        counts = train[column].value_counts(dropna=False)
        categorical_rows.append(
            {
                "column": column,
                "unique_non_null": train[column].nunique(dropna=True),
                "most_common": str(counts.index[0]),
                "most_common_count": int(counts.iloc[0]),
                "most_common_pct": counts.iloc[0] / len(train) * 100,
            }
        )
    pd.DataFrame(categorical_rows).to_csv(OUTPUT_DIR / "categorical_summary.csv", index=False)

    train[target].value_counts(dropna=False).rename("count").to_csv(
        OUTPUT_DIR / "target_distribution.csv"
    )

    summary = [
        "Student Health Risk — Simple EDA",
        f"Train shape: {train.shape[0]:,} rows x {train.shape[1]} columns",
        f"Test shape: {test.shape[0]:,} rows x {test.shape[1]} columns",
        f"Detected target: {target}",
        f"Numeric columns: {len(numeric_columns)}",
        f"Categorical columns: {len(categorical_columns)}",
        f"Train duplicate rows: {train.duplicated().sum():,}",
        f"Test duplicate rows: {test.duplicated().sum():,}",
        f"Train missing cells: {train.isna().sum().sum():,} ({train.isna().mean().mean() * 100:.2f}%)",
        f"Test missing cells: {test.isna().sum().sum():,} ({test.isna().mean().mean() * 100:.2f}%)",
    ]
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    save_target_plot(train, target)
    save_missing_plot(train)
    save_numeric_distributions(train, numeric_columns)
    save_correlation_plot(train, numeric_columns)

    print("\n".join(summary))
    print(f"EDA outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
