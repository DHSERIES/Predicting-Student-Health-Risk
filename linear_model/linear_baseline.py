"""Linear-model baselines with leakage-safe stratified 5-fold validation."""

import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "data" / "train.csv"
TEST_PATH = ROOT / "data" / "test.csv"
OUTPUT_DIR = ROOT / "linear_baseline_outputs"
TARGET = "health_condition"
ID_COLUMN = "id"
N_SPLITS = 5
RANDOM_STATE = 42


def build_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        sparse_threshold=1.0,
    )


def build_models() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=400,
            random_state=RANDOM_STATE,
        ),
        "ridge_classifier": RidgeClassifier(
            alpha=1.0,
            class_weight="balanced",
            solver="lsqr",
        ),
        "sgd_logistic": SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            class_weight="balanced",
            max_iter=2_000,
            early_stopping=False,
            average=True,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def evaluate_models(
    features: pd.DataFrame,
    labels: pd.Series,
    preprocessor: ColumnTransformer,
    models: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_rows = []
    report_rows = []
    predictions_by_model: dict[str, np.ndarray] = {}

    for model_name, estimator in models.items():
        print(f"\nEvaluating {model_name}", flush=True)
        out_of_fold = np.empty(len(features), dtype=object)
        for fold_number, (train_index, valid_index) in enumerate(folds.split(features, labels), start=1):
            pipeline = Pipeline(
                [
                    ("preprocessor", clone(preprocessor)),
                    ("model", clone(estimator)),
                ]
            )
            started = time.perf_counter()
            pipeline.fit(features.iloc[train_index], labels.iloc[train_index])
            predictions = pipeline.predict(features.iloc[valid_index])
            elapsed = time.perf_counter() - started
            score = balanced_accuracy_score(labels.iloc[valid_index], predictions)
            out_of_fold[valid_index] = predictions
            fold_rows.append(
                {
                    "model": model_name,
                    "fold": fold_number,
                    "balanced_accuracy": score,
                    "fit_and_predict_seconds": elapsed,
                    "train_rows": len(train_index),
                    "validation_rows": len(valid_index),
                }
            )
            print(
                f"  fold {fold_number}/{N_SPLITS}: balanced_accuracy={score:.6f}, time={elapsed:.1f}s",
                flush=True,
            )

        predictions_by_model[model_name] = out_of_fold.copy()
        class_report = classification_report(labels, out_of_fold, output_dict=True, zero_division=0)
        for class_name, values in class_report.items():
            if isinstance(values, dict):
                report_rows.append(
                    {
                        "model": model_name,
                        "class": class_name,
                        "precision": values["precision"],
                        "recall": values["recall"],
                        "f1_score": values["f1-score"],
                        "support": values["support"],
                    }
                )

    fold_scores = pd.DataFrame(fold_rows)
    summary = (
        fold_scores.groupby("model", as_index=False)
        .agg(
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            balanced_accuracy_min=("balanced_accuracy", "min"),
            balanced_accuracy_max=("balanced_accuracy", "max"),
            mean_fold_seconds=("fit_and_predict_seconds", "mean"),
        )
        .sort_values("balanced_accuracy_mean", ascending=False)
    )
    return fold_scores, summary, pd.DataFrame(report_rows), predictions_by_model


def save_fold_score_plot(fold_scores: pd.DataFrame, summary: pd.DataFrame) -> None:
    order = summary["model"].tolist()
    positions = np.arange(len(order))
    means = summary.set_index("model").loc[order, "balanced_accuracy_mean"]
    stds = summary.set_index("model").loc[order, "balanced_accuracy_std"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(positions, means, xerr=stds, color="#4C78A8", alpha=0.9, capsize=4)
    ax.set_yticks(positions, labels=order)
    ax.invert_yaxis()
    ax.set(title="Linear baseline: stratified 5-fold validation", xlabel="Mean balanced accuracy (error bar = fold SD)")
    ax.set_xlim(max(0, means.min() - 0.05), min(1.0, means.max() + 0.05))
    ax.bar_label(bars, labels=[f"{value:.4f}" for value in means], padding=4)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "linear_model_cv_scores.png", dpi=160)
    plt.close(fig)


def save_confusion_plot(labels: pd.Series, predictions: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    matrix = confusion_matrix(labels, predictions, labels=class_names)
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), labels=class_names, rotation=30, ha="right")
    ax.set_yticks(range(len(class_names)), labels=class_names)
    ax.set(title="Best linear model: normalized OOF confusion matrix", xlabel="Predicted", ylabel="Actual")
    for row in range(len(class_names)):
        for column in range(len(class_names)):
            ax.text(column, row, f"{normalized[row, column]:.1%}\n({matrix[row, column]:,})", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Share of actual class")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "best_linear_oof_confusion_matrix.png", dpi=160)
    plt.close(fig)

    return pd.DataFrame(matrix, index=[f"actual_{name}" for name in class_names], columns=[f"predicted_{name}" for name in class_names])


def save_coefficients(pipeline: Pipeline) -> None:
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    model = pipeline.named_steps["model"]
    coefficients = np.atleast_2d(model.coef_)
    classes = model.classes_
    rows = []
    for class_name, class_coefficients in zip(classes, coefficients):
        for feature_name, coefficient in zip(feature_names, class_coefficients):
            rows.append(
                {
                    "class": class_name,
                    "feature": feature_name,
                    "coefficient": coefficient,
                    "absolute_coefficient": abs(coefficient),
                }
            )
    pd.DataFrame(rows).sort_values(["class", "absolute_coefficient"], ascending=[True, False]).to_csv(
        OUTPUT_DIR / "best_linear_coefficients.csv", index=False
    )


def build_report(
    summary: pd.DataFrame,
    class_report: pd.DataFrame,
    best_model: str,
    target_distribution: pd.Series,
) -> None:
    best = summary.iloc[0]
    best_classes = class_report[(class_report["model"] == best_model) & class_report["class"].isin(target_distribution.index)]
    recall_text = ", ".join(
        f"{row['class']} {row['recall']:.3f}" for _, row in best_classes.sort_values("class").iterrows()
    )
    ranking = "\n".join(
        f"{index + 1}. `{row['model']}`: {row['balanced_accuracy_mean']:.6f} ± {row['balanced_accuracy_std']:.6f}"
        for index, (_, row) in enumerate(summary.iterrows())
    )
    report = f"""# Linear Baseline Report

## Result

The best linear baseline is **`{best_model}`** with mean 5-fold balanced accuracy **{best['balanced_accuracy_mean']:.6f} ± {best['balanced_accuracy_std']:.6f}**. Its fold range is {best['balanced_accuracy_min']:.6f}–{best['balanced_accuracy_max']:.6f}.

## Model ranking

{ranking}

## Validation design

- Stratified 5-fold cross-validation with shuffling and random seed {RANDOM_STATE}.
- Primary metric: `balanced_accuracy_score`, the unweighted mean recall across classes.
- `id` excluded from modeling.
- Numeric features: fold-local median imputation with missing indicators, then standardization.
- Categorical features: fold-local most-frequent imputation and one-hot encoding with unknown-category handling.
- All models use balanced class weights because the target distribution is highly imbalanced.
- The same folds and preprocessing design are used for all models.

## Best-model class behavior

Out-of-fold recall by class: {recall_text}. The normalized confusion matrix shows where minority-class errors occur.

## Models evaluated

- Logistic regression with L2 regularization.
- Ridge classifier with balanced class weights.
- SGD linear classifier with logistic loss and averaged weights.

## Artifacts

- `linear_baseline_summary.csv`: model-level comparison.
- `linear_baseline_fold_scores.csv`: every fold score and runtime.
- `linear_baseline_classification_report.csv`: out-of-fold precision, recall, and F1 by class.
- `best_linear_oof_predictions.csv`: best model's leakage-safe out-of-fold predictions.
- `best_linear_coefficients.csv`: full-data fitted coefficient table.
- `linear_baseline_submission.csv`: test predictions from the refitted best pipeline.
- `best_linear_pipeline.joblib`: serialized preprocessing and model pipeline.

## Caveat

The advanced EDA found moderate joint train-test separability (adversarial AUC 0.650). These random stratified folds are a consistent baseline, but test performance may differ if the joint distribution shift affects the label relationship.
"""
    (OUTPUT_DIR / "linear_baseline_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    feature_columns = [column for column in test.columns if column != ID_COLUMN]
    numeric_columns = [column for column in feature_columns if pd.api.types.is_numeric_dtype(train[column])]
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]
    features = train[feature_columns]
    labels = train[TARGET]

    preprocessor = build_preprocessor(numeric_columns, categorical_columns)
    models = build_models()
    fold_scores, summary, class_report, predictions_by_model = evaluate_models(
        features, labels, preprocessor, models
    )
    best_model_name = summary.iloc[0]["model"]
    best_oof_predictions = predictions_by_model[best_model_name]
    class_names = sorted(labels.unique())

    fold_scores.to_csv(OUTPUT_DIR / "linear_baseline_fold_scores.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "linear_baseline_summary.csv", index=False)
    class_report.to_csv(OUTPUT_DIR / "linear_baseline_classification_report.csv", index=False)
    pd.DataFrame(
        {ID_COLUMN: train[ID_COLUMN], "actual": labels, "oof_prediction": best_oof_predictions}
    ).to_csv(OUTPUT_DIR / "best_linear_oof_predictions.csv", index=False)

    save_fold_score_plot(fold_scores, summary)
    confusion = save_confusion_plot(labels, best_oof_predictions, class_names)
    confusion.to_csv(OUTPUT_DIR / "best_linear_oof_confusion_matrix.csv")

    print(f"\nRefitting best model on all training data: {best_model_name}", flush=True)
    best_pipeline = Pipeline(
        [
            ("preprocessor", clone(preprocessor)),
            ("model", clone(models[best_model_name])),
        ]
    )
    best_pipeline.fit(features, labels)
    test_predictions = best_pipeline.predict(test[feature_columns])
    pd.DataFrame({ID_COLUMN: test[ID_COLUMN], TARGET: test_predictions}).to_csv(
        OUTPUT_DIR / "linear_baseline_submission.csv", index=False
    )
    save_coefficients(best_pipeline)
    joblib.dump(best_pipeline, OUTPUT_DIR / "best_linear_pipeline.joblib", compress=3)
    build_report(summary, class_report, best_model_name, labels.value_counts())

    print("\nLinear baseline complete", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"Outputs: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
