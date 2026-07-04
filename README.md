# Pulp Brightness Prediction

Predicting pulp brightness after the first stage of ClO2 bleaching at a paper mill, using
statistical learning. Built for **Södra's** pulp mill in Värö, Sweden, as part of a
university course in statistical learning.

> **Note on data availability:** This project was completed using real production sensor
> data from Södra's mill. For confidentiality reasons, Södra asked that the underlying
> dataset not be published, so this repository contains only the final model notebook —
> not the data itself, nor the private `constants` / `functions` / `evaluation` helper
> modules it imports. The notebook is included for portfolio purposes and is **not
> directly runnable** without that proprietary environment. Everything below summarizes
> the full project report.

## Background

Södra's Värö mill produces ~2,000 tons of pulp per day. Bleaching consumes 60–70 kg of
chemicals per ton of pulp, costing roughly 300 million SEK/year. Being able to predict the
pulp's brightness after the first bleaching stage — instead of only measuring it afterward
— creates room to reduce chemical dosing without sacrificing quality.

**Goal:** predict `s1_brightness_out` (brightness after the first D-stage) from sensor
readings taken earlier in the process (chlorine dioxide feed, kappa number, flow, tank
levels, temperature, etc.).

Södra set two hard requirements for any model to be considered for production:

1. The model's output must be **monotonically increasing** in `s1_feed_clo2` (more
   bleaching chemical should never predict lower brightness).
2. `s1_feed_clo2` must rank among the **top 3 most important features** (by mean absolute
   SHAP value).

## Approach

Three models were compared, in increasing order of complexity:

| Model | Test R² | Test MSE | Test MAE | Meets requirements |
|---|---|---|---|---|
| Linear Regression | 0.229 | 1.878 | 1.002 | Yes |
| Ridge Regression | 0.232 | 1.873 | 1.013 | Yes |
| **HistGradientBoostingRegressor** | **0.613** | **0.943** | **0.597** | Yes |

The gradient boosting model (this repository's notebook, `HGBR.ipynb`) roughly doubled the
explained variance of the linear baselines and was the only one with strong enough
precision to be a realistic production candidate.

Key modeling steps:

- **Time alignment** — sensor data (1-minute resolution) was offset and aggregated to
  15-minute windows to match the brightness measurement cadence, accounting for the
  transport delay of pulp through the reactor.
- **Outlier handling** — production-stoppage rows were dropped, followed by z-score
  filtering (|z| > 3) on the training set.
- **Feature engineering** — a `prev_out` lag feature (average of the last two brightness
  readings) and degree-2 polynomial feature interactions.
- **Constrained forward selection** — features were added greedily by validation R²,
  under the constraint that `s1_feed_clo2`'s SHAP importance stay at least 50% of the top
  feature's, keeping the chemical-dosing signal from being crowded out.
- **Monotonicity constraint** — enforced directly via `HistGradientBoostingRegressor`'s
  `monotonic_cst` on `s1_feed_clo2`.
- **Hyperparameter search** — `GridSearchCV` over tree depth, leaf count, learning rate,
  and L2 regularization.
- Model performance and the two production requirements were validated with **SHAP**
  dependence plots; experiments were tracked with **MLflow**.

## Limitations

- Validated only on data from one mill and one bleaching stage — generalization to other
  plants isn't established.
- No denoising was applied to the (reportedly noisy) sensor data, which likely caps
  achievable accuracy.
- Traditional regression is used rather than a time-series model, even though the process
  is continuous and autocorrelated; the `prev_out` lag feature is a partial workaround.

## Repository contents

- `HGBR.ipynb` — the winning model: feature selection, hyperparameter search, training,
  evaluation, and SHAP-based requirement checks.

## Background & attribution

This started as a group project for the *Statistisk inlärning* (Statistical Learning)
course at Linnaeus University, together with Linus Ögren and Diart Hasani, comparing
linear regression, ridge regression, and gradient boosting for Södra. The gradient
boosting notebook in this repository — the best-performing of the three models — was
developed individually by me.

## Author

Jesper Bern Sanders
