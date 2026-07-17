# Linear Baseline Report

## Result

The best linear baseline is **`logistic_regression`** with mean 5-fold balanced accuracy **0.869810 ± 0.001281**. Its fold range is 0.868821–0.871878.

## Model ranking

1. `logistic_regression`: 0.869810 ± 0.001281
2. `sgd_logistic`: 0.850521 ± 0.001297
3. `ridge_classifier`: 0.780076 ± 0.000827

## Validation design

- Stratified 5-fold cross-validation with shuffling and random seed 42.
- Primary metric: `balanced_accuracy_score`, the unweighted mean recall across classes.
- `id` excluded from modeling.
- Numeric features: fold-local median imputation with missing indicators, then standardization.
- Categorical features: fold-local most-frequent imputation and one-hot encoding with unknown-category handling.
- All models use balanced class weights because the target distribution is highly imbalanced.
- The same folds and preprocessing design are used for all models.

## Best-model class behavior

Out-of-fold recall by class: at-risk 0.795, fit 0.900, unhealthy 0.915. The normalized confusion matrix shows where minority-class errors occur.

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