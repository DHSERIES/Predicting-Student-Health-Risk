# Advanced EDA — Predicting Student Health Risk

## Executive summary

- The training set contains **690,088 rows** and the test set contains **295,753 rows**. The modeling target is `health_condition`.
- The target is strongly imbalanced: **at-risk 85.9%**, **unhealthy 8.4%**, and **fit 5.8%**. Accuracy alone will be misleading; use macro-F1 or balanced accuracy and stratified validation.
- The strongest univariate feature-label relationship is **`stress_level`** (cramers_v = 0.387). These scores describe marginal association, not causal importance or multivariate model value.
- Marginal train-test distributions are extremely close, but joint train-test drift is **moderate**: adversarial-validation AUC is **0.650** (0.5 means indistinguishable). The largest numeric drift is `heart_rate` (PSI 0.0001, KS D 0.0020); the largest categorical drift is `gender` (TVD 0.0331).
- Missing-value rates are nearly identical between train and test. The missingness indicator most related to the label is `bmi_is_missing` (corrected Cramer's V 0.031), so imputation should preserve missingness indicators where useful.
- `id` is unique and the test IDs begin immediately after the training IDs: **True**. Treat it as an identifier, not a health feature, unless a time/order effect is explicitly validated.

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
- Despite the marginal similarity, adversarial AUC is 0.650. The largest AUC drops after permutation are `water_intake` (0.073), `physical_activity_level` (0.058), and `calorie_expenditure` (0.052). This points to differences in joint or conditional feature relationships rather than large one-column shifts.

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