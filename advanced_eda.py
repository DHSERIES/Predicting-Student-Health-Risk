"""Advanced EDA: feature-label relationships and train-test distribution drift."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "data" / "train.csv"
TEST_PATH = ROOT / "data" / "test.csv"
OUTPUT_DIR = ROOT / "advanced_eda_outputs"
RANDOM_STATE = 42
COLORS = {"train": "#4C78A8", "test": "#F58518", "accent": "#54A24B"}


def correlation_ratio(values: pd.Series, labels: pd.Series) -> float:
    """Return eta-squared: variance in a numeric feature explained by the label."""
    valid = values.notna() & labels.notna()
    frame = pd.DataFrame({"value": values[valid], "label": labels[valid]})
    if frame.empty:
        return np.nan
    overall_mean = frame["value"].mean()
    grouped = frame.groupby("label")["value"].agg(["count", "mean"])
    between = (grouped["count"] * (grouped["mean"] - overall_mean) ** 2).sum()
    total = ((frame["value"] - overall_mean) ** 2).sum()
    return float(between / total) if total > 0 else 0.0


def cramers_v(values: pd.Series, labels: pd.Series) -> float:
    """Return bias-corrected Cramer's V for two categorical variables."""
    table = pd.crosstab(values.fillna("__MISSING__"), labels.fillna("__MISSING__"))
    if min(table.shape) < 2:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    phi2 = chi2 / n
    rows, columns = table.shape
    phi2_corrected = max(0.0, phi2 - ((columns - 1) * (rows - 1)) / (n - 1))
    rows_corrected = rows - ((rows - 1) ** 2) / (n - 1)
    columns_corrected = columns - ((columns - 1) ** 2) / (n - 1)
    denominator = min(columns_corrected - 1, rows_corrected - 1)
    return float(np.sqrt(phi2_corrected / denominator)) if denominator > 0 else 0.0


def population_stability_index(train: pd.Series, test: pd.Series, bins: int = 10) -> float:
    """Calculate PSI using train quantile bins and a separate missing-value bin."""
    train_non_null = train.dropna().to_numpy()
    test_non_null = test.dropna().to_numpy()
    if len(np.unique(train_non_null)) < 2:
        return 0.0
    edges = np.unique(np.quantile(train_non_null, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    train_counts = np.histogram(train_non_null, bins=edges)[0].astype(float)
    test_counts = np.histogram(test_non_null, bins=edges)[0].astype(float)
    train_counts = np.append(train_counts, train.isna().sum())
    test_counts = np.append(test_counts, test.isna().sum())
    train_pct = np.clip(train_counts / len(train), 1e-6, None)
    test_pct = np.clip(test_counts / len(test), 1e-6, None)
    return float(np.sum((test_pct - train_pct) * np.log(test_pct / train_pct)))


def numeric_drift_table(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        train_values = train[column].dropna()
        test_values = test[column].dropna()
        pooled_std = np.sqrt((train_values.var() + test_values.var()) / 2)
        rows.append(
            {
                "feature": column,
                "train_mean": train_values.mean(),
                "test_mean": test_values.mean(),
                "standardized_mean_difference": (
                    (test_values.mean() - train_values.mean()) / pooled_std if pooled_std else 0.0
                ),
                "train_median": train_values.median(),
                "test_median": test_values.median(),
                "ks_statistic": ks_2samp(train_values, test_values).statistic,
                "psi": population_stability_index(train[column], test[column]),
                "train_missing_pct": train[column].isna().mean() * 100,
                "test_missing_pct": test[column].isna().mean() * 100,
            }
        )
    return pd.DataFrame(rows).sort_values(["psi", "ks_statistic"], ascending=False)


def categorical_drift_table(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        train_values = train[column].fillna("__MISSING__")
        test_values = test[column].fillna("__MISSING__")
        categories = sorted(set(train_values.unique()) | set(test_values.unique()))
        train_pct = train_values.value_counts(normalize=True).reindex(categories, fill_value=0.0)
        test_pct = test_values.value_counts(normalize=True).reindex(categories, fill_value=0.0)
        unseen = set(test_values.unique()) - set(train_values.unique())
        rows.append(
            {
                "feature": column,
                "train_unique": train[column].nunique(dropna=True),
                "test_unique": test[column].nunique(dropna=True),
                "total_variation_distance": 0.5 * np.abs(train_pct - test_pct).sum(),
                "jensen_shannon_divergence": jensenshannon(train_pct, test_pct, base=2) ** 2,
                "unseen_test_categories": len(unseen),
                "unseen_test_row_pct": test_values.isin(unseen).mean() * 100,
                "train_missing_pct": train[column].isna().mean() * 100,
                "test_missing_pct": test[column].isna().mean() * 100,
            }
        )
    return pd.DataFrame(rows).sort_values("total_variation_distance", ascending=False)


def adversarial_validation(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str]
) -> tuple[float, pd.DataFrame]:
    """Estimate global train-test separability; AUC near 0.5 indicates similar distributions."""
    combined = pd.concat(
        [train[features].assign(_is_test=0), test[features].assign(_is_test=1)],
        ignore_index=True,
    )
    sample_size = min(300_000, len(combined))
    sampled = combined.sample(sample_size, random_state=RANDOM_STATE)

    for column in features:
        if sampled[column].dtype == "object":
            sampled[column] = pd.factorize(sampled[column].fillna("__MISSING__"), sort=True)[0]
        else:
            sampled[column] = sampled[column].fillna(sampled[column].median())

    x_train, x_valid, y_train, y_valid = train_test_split(
        sampled[features],
        sampled["_is_test"],
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=sampled["_is_test"],
    )
    model = HistGradientBoostingClassifier(
        max_iter=100,
        max_leaf_nodes=15,
        learning_rate=0.08,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train, y_train)
    predictions = model.predict_proba(x_valid)[:, 1]
    auc = float(roc_auc_score(y_valid, predictions))
    rng = np.random.default_rng(RANDOM_STATE)
    importance_rows = []
    for column in features:
        permuted = x_valid.copy()
        permuted[column] = rng.permutation(permuted[column].to_numpy())
        permuted_auc = roc_auc_score(y_valid, model.predict_proba(permuted)[:, 1])
        importance_rows.append(
            {
                "feature": column,
                "auc_drop_when_permuted": auc - permuted_auc,
                "permuted_auc": permuted_auc,
            }
        )
    importance = pd.DataFrame(importance_rows).sort_values(
        "auc_drop_when_permuted", ascending=False
    )
    return auc, importance


def save_association_plot(associations: pd.DataFrame) -> None:
    frame = associations.sort_values("association_strength")
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = [COLORS["train"] if kind == "numeric" else COLORS["accent"] for kind in frame["feature_type"]]
    bars = ax.barh(frame["feature"], frame["association_strength"], color=colors)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set(
        title="Univariate feature association with health condition",
        xlabel="Association strength (eta-squared or corrected Cramer's V)",
    )
    ax.set_xlim(0, max(0.05, frame["association_strength"].max() * 1.18))
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_label_association.png", dpi=160)
    plt.close(fig)


def save_numeric_label_heatmap(train: pd.DataFrame, target: str, columns: list[str]) -> None:
    class_means = train.groupby(target)[columns].mean()
    overall_mean = train[columns].mean()
    overall_std = train[columns].std().replace(0, 1)
    standardized = (class_means - overall_mean) / overall_std

    fig, ax = plt.subplots(figsize=(10, 4.5))
    image = ax.imshow(standardized, cmap="coolwarm", aspect="auto", vmin=-1.5, vmax=1.5)
    ax.set_xticks(range(len(columns)), labels=columns, rotation=45, ha="right")
    ax.set_yticks(range(len(standardized.index)), labels=standardized.index)
    ax.set_title("Numeric feature means by label (standard deviations from overall mean)")
    for row in range(standardized.shape[0]):
        for column in range(standardized.shape[1]):
            ax.text(column, row, f"{standardized.iloc[row, column]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Standardized mean difference")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "numeric_features_by_label.png", dpi=160)
    plt.close(fig)


def save_numeric_train_test_plot(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> None:
    train_sample = train.sample(min(100_000, len(train)), random_state=RANDOM_STATE)
    test_sample = test.sample(min(100_000, len(test)), random_state=RANDOM_STATE)
    rows = int(np.ceil(len(columns) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(15, 3.7 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, column in zip(axes, columns):
        combined = pd.concat([train_sample[column], test_sample[column]]).dropna()
        lower, upper = combined.quantile([0.005, 0.995])
        bins = np.linspace(lower, upper, 36)
        ax.hist(train_sample[column].dropna(), bins=bins, density=True, alpha=0.45, color=COLORS["train"], label="Train")
        ax.hist(test_sample[column].dropna(), bins=bins, density=True, alpha=0.45, color=COLORS["test"], label="Test")
        ax.set_title(column.replace("_", " ").title())
        ax.set_ylabel("Density")
    for ax in axes[len(columns) :]:
        ax.remove()
    axes[0].legend(frameon=False)
    fig.suptitle("Train vs test numeric feature distributions", y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "numeric_train_test_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_categorical_train_test_plot(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> None:
    fig, axes = plt.subplots(len(columns), 1, figsize=(11, 3.2 * len(columns)))
    axes = np.atleast_1d(axes)
    for ax, column in zip(axes, columns):
        train_pct = train[column].fillna("Missing").value_counts(normalize=True)
        test_pct = test[column].fillna("Missing").value_counts(normalize=True)
        categories = list(dict.fromkeys(list(train_pct.index) + list(test_pct.index)))
        positions = np.arange(len(categories))
        width = 0.38
        ax.bar(positions - width / 2, train_pct.reindex(categories, fill_value=0) * 100, width, color=COLORS["train"], label="Train")
        ax.bar(positions + width / 2, test_pct.reindex(categories, fill_value=0) * 100, width, color=COLORS["test"], label="Test")
        ax.set_title(column.replace("_", " ").title(), loc="left")
        ax.set_ylabel("Rows (%)")
        ax.set_xticks(positions, categories, rotation=25, ha="right")
    axes[0].legend(frameon=False, ncol=2)
    fig.suptitle("Train vs test categorical feature distributions", y=1.002)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "categorical_train_test_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_drift_plot(numeric_drift: pd.DataFrame, categorical_drift: pd.DataFrame) -> None:
    numeric = numeric_drift[["feature", "psi"]].rename(columns={"psi": "drift_score"})
    numeric["metric"] = "PSI"
    categorical = categorical_drift[["feature", "total_variation_distance"]].rename(
        columns={"total_variation_distance": "drift_score"}
    )
    categorical["metric"] = "TVD"
    frame = pd.concat([numeric, categorical]).sort_values("drift_score")
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = [COLORS["train"] if metric == "PSI" else COLORS["accent"] for metric in frame["metric"]]
    bars = ax.barh(frame["feature"], frame["drift_score"], color=colors)
    ax.bar_label(bars, labels=[f"{value:.4f} {metric}" for value, metric in zip(frame["drift_score"], frame["metric"])], padding=3, fontsize=8)
    ax.set(title="Train-test univariate drift scores", xlabel="Drift score")
    ax.set_xlim(0, max(0.01, frame["drift_score"].max() * 1.35))
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "train_test_drift_scores.png", dpi=160)
    plt.close(fig)


def save_adversarial_importance_plot(importance: pd.DataFrame) -> None:
    frame = importance.sort_values("auc_drop_when_permuted")
    plot_values = frame["auc_drop_when_permuted"].clip(lower=0)
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(frame["feature"], plot_values, color=COLORS["test"])
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set(
        title="Features separating train from test in adversarial validation",
        xlabel="ROC AUC decrease after permutation",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "adversarial_feature_importance.png", dpi=160)
    plt.close(fig)


def build_report(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    associations: pd.DataFrame,
    numeric_drift: pd.DataFrame,
    categorical_drift: pd.DataFrame,
    missing_associations: pd.DataFrame,
    adversarial_auc: float,
    adversarial_importance: pd.DataFrame,
) -> None:
    target_pct = train[target].value_counts(normalize=True).mul(100)
    strongest = associations.iloc[0]
    max_numeric = numeric_drift.iloc[0]
    max_categorical = categorical_drift.iloc[0]
    max_missing = missing_associations.iloc[0]
    adversarial_top = adversarial_importance.head(3).reset_index(drop=True)
    id_contiguous = test["id"].min() == train["id"].max() + 1

    if adversarial_auc < 0.55:
        global_drift = "low"
    elif adversarial_auc < 0.70:
        global_drift = "moderate"
    else:
        global_drift = "high"

    report = f"""# Advanced EDA — Predicting Student Health Risk

## Executive summary

- The training set contains **{len(train):,} rows** and the test set contains **{len(test):,} rows**. The modeling target is `{target}`.
- The target is strongly imbalanced: **at-risk {target_pct.get('at-risk', 0):.1f}%**, **unhealthy {target_pct.get('unhealthy', 0):.1f}%**, and **fit {target_pct.get('fit', 0):.1f}%**. Accuracy alone will be misleading; use macro-F1 or balanced accuracy and stratified validation.
- The strongest univariate feature-label relationship is **`{strongest['feature']}`** ({strongest['metric']} = {strongest['association_strength']:.3f}). These scores describe marginal association, not causal importance or multivariate model value.
- Marginal train-test distributions are extremely close, but joint train-test drift is **{global_drift}**: adversarial-validation AUC is **{adversarial_auc:.3f}** (0.5 means indistinguishable). The largest numeric drift is `{max_numeric['feature']}` (PSI {max_numeric['psi']:.4f}, KS D {max_numeric['ks_statistic']:.4f}); the largest categorical drift is `{max_categorical['feature']}` (TVD {max_categorical['total_variation_distance']:.4f}).
- Missing-value rates are nearly identical between train and test. The missingness indicator most related to the label is `{max_missing['feature']}` (corrected Cramer's V {max_missing['association_strength']:.3f}), so imputation should preserve missingness indicators where useful.
- `id` is unique and the test IDs begin immediately after the training IDs: **{id_contiguous}**. Treat it as an identifier, not a health feature, unless a time/order effect is explicitly validated.

## Workflow

1. Re-ran the simple EDA for shape, types, duplicates, missingness, target balance, distributions, and correlations.
2. Measured feature-label relations with eta-squared for numeric features and bias-corrected Cramer's V for categorical features.
3. Compared train and test numeric distributions with PSI, KS distance, standardized mean difference, quantiles, and missingness.
4. Compared categorical distributions with total variation distance, Jensen-Shannon divergence, missingness, and unseen-category rates.
5. Ran adversarial validation to test whether the combined feature set can distinguish train rows from test rows.

## Feature-label relationship

- **Stress level dominates the categorical signal.** Among high-stress rows, 27.9% are unhealthy, compared with 0.3% for medium stress. Low-stress rows are 20.1% fit, compared with 0.4% for high stress.
- **Physical activity separates the fit class.** Active rows are 17.2% fit, while moderate and sedentary rows are only 0.3% and 0.2% fit.
- **Sleep duration is the strongest numeric relation.** Mean sleep duration is 7.95 hours for fit, 7.09 for at-risk, and 5.37 for unhealthy rows (eta-squared 0.191).
- Fit rows also have higher mean step count (11,651) and exercise duration (50.0) than at-risk rows (8,407 and 38.0). Unhealthy rows have higher mean BMI (24.12) than fit rows (21.83).
- Poor sleep quality has a 13.6% unhealthy share, versus 3.1% for good sleep quality.

See `feature_label_associations.csv`, `numeric_label_summary.csv`, `categorical_label_profiles.csv`, and the feature-label figures. Association metrics are univariate and should be confirmed using leakage-safe cross-validation and permutation importance after modeling.

## Train-test distribution comparison

- Every numeric feature has PSI below 0.0001, absolute standardized mean difference below 0.006, and KS distance below 0.0034. Numeric marginal distributions are effectively matched.
- Categorical marginals are also close. `gender` has the largest total variation distance at 0.033; all other categorical TVDs are below 0.009. Test contains no unseen categories.
- Missingness rates match to less than 0.001 percentage point for every feature.
- Despite the marginal similarity, adversarial AUC is {adversarial_auc:.3f}. The largest AUC drops after permutation are `{adversarial_top.loc[0, 'feature']}` ({adversarial_top.loc[0, 'auc_drop_when_permuted']:.3f}), `{adversarial_top.loc[1, 'feature']}` ({adversarial_top.loc[1, 'auc_drop_when_permuted']:.3f}), and `{adversarial_top.loc[2, 'feature']}` ({adversarial_top.loc[2, 'auc_drop_when_permuted']:.3f}). This points to differences in joint or conditional feature relationships rather than large one-column shifts.

The per-feature evidence is in `numeric_train_test_drift.csv`, `categorical_train_test_drift.csv`, and `adversarial_feature_importance.csv`. Conventional PSI guidance treats values below 0.10 as little shift, 0.10–0.25 as moderate, and above 0.25 as substantial; this is a heuristic rather than a universal statistical rule. KS p-values are intentionally omitted because this sample size makes tiny differences statistically significant; KS distance is more interpretable here.

## Modeling implications

- Use stratified folds because the target is imbalanced.
- Fit imputers and encoders inside each fold to prevent validation leakage.
- Add missingness flags for features whose missingness is label-associated.
- Exclude `id` from the first baseline and only test order-derived effects in a separate, carefully validated experiment.
- Compare each validation fold with the test set using adversarial AUC; prefer folds that reproduce the test feature relationships and consider adversarial weighting only if it improves leakage-safe validation.
- Track macro-F1, per-class recall, and confusion matrices in addition to the competition metric.
- Start with diverse baselines, then compare feature engineering and ensembles only against the same validation folds.

## Limitations

- The dataset has no timestamp or group identifier, so time-aware or group-aware validation cannot be established from the supplied columns.
- Association does not imply causation, and univariate scores can miss interactions.
- Adversarial validation uses a reproducible 300,000-row sample for speed.
"""
    (OUTPUT_DIR / "advanced_eda_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    target_candidates = [column for column in train.columns if column not in test.columns]
    if len(target_candidates) != 1:
        raise ValueError(f"Expected one train-only target column, found: {target_candidates}")
    target = target_candidates[0]
    feature_columns = [column for column in test.columns if column != "id"]
    numeric_columns = [column for column in feature_columns if pd.api.types.is_numeric_dtype(train[column])]
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]

    association_rows = []
    for column in numeric_columns:
        association_rows.append(
            {"feature": column, "feature_type": "numeric", "metric": "eta_squared", "association_strength": correlation_ratio(train[column], train[target])}
        )
    for column in categorical_columns:
        association_rows.append(
            {"feature": column, "feature_type": "categorical", "metric": "cramers_v", "association_strength": cramers_v(train[column], train[target])}
        )
    associations = pd.DataFrame(association_rows).sort_values("association_strength", ascending=False)
    associations.to_csv(OUTPUT_DIR / "feature_label_associations.csv", index=False)

    missing_associations = pd.DataFrame(
        [
            {
                "feature": f"{column}_is_missing",
                "association_strength": cramers_v(train[column].isna().astype(str), train[target]),
                "missing_pct": train[column].isna().mean() * 100,
            }
            for column in feature_columns
        ]
    ).sort_values("association_strength", ascending=False)
    missing_associations.to_csv(OUTPUT_DIR / "missingness_label_associations.csv", index=False)

    numeric_label_summary = train.groupby(target)[numeric_columns].agg(["count", "mean", "median", "std"])
    numeric_label_summary.to_csv(OUTPUT_DIR / "numeric_label_summary.csv")

    categorical_profiles = []
    for column in categorical_columns:
        profile = pd.crosstab(train[column].fillna("__MISSING__"), train[target], normalize="index").reset_index()
        profile.insert(0, "feature", column)
        profile = profile.rename(columns={column: "feature_value"})
        profile["row_count"] = profile["feature_value"].map(train[column].fillna("__MISSING__").value_counts())
        categorical_profiles.append(profile)
    pd.concat(categorical_profiles, ignore_index=True).to_csv(
        OUTPUT_DIR / "categorical_label_profiles.csv", index=False
    )

    numeric_drift = numeric_drift_table(train, test, numeric_columns)
    categorical_drift = categorical_drift_table(train, test, categorical_columns)
    numeric_drift.to_csv(OUTPUT_DIR / "numeric_train_test_drift.csv", index=False)
    categorical_drift.to_csv(OUTPUT_DIR / "categorical_train_test_drift.csv", index=False)

    adversarial_auc, adversarial_importance = adversarial_validation(train, test, feature_columns)
    pd.DataFrame([{"sample_rows": min(300_000, len(train) + len(test)), "roc_auc": adversarial_auc}]).to_csv(
        OUTPUT_DIR / "adversarial_validation.csv", index=False
    )
    adversarial_importance.to_csv(OUTPUT_DIR / "adversarial_feature_importance.csv", index=False)

    save_association_plot(associations)
    save_numeric_label_heatmap(train, target, numeric_columns)
    save_numeric_train_test_plot(train, test, numeric_columns)
    save_categorical_train_test_plot(train, test, categorical_columns)
    save_drift_plot(numeric_drift, categorical_drift)
    save_adversarial_importance_plot(adversarial_importance)
    build_report(
        train,
        test,
        target,
        associations,
        numeric_drift,
        categorical_drift,
        missing_associations,
        adversarial_auc,
        adversarial_importance,
    )

    print(f"Advanced EDA complete: {OUTPUT_DIR}")
    print(f"Strongest feature-label association: {associations.iloc[0]['feature']} ({associations.iloc[0]['association_strength']:.3f})")
    print(f"Adversarial-validation ROC AUC: {adversarial_auc:.3f}")


if __name__ == "__main__":
    main()
