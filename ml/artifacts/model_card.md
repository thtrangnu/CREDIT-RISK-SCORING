# Model Card — Home Credit Default Risk

## Summary

| | |
|---|---|
| Task | Binary classification, predicting default, 8.1% positive |
| Data | Home Credit Default Risk (Kaggle), 7 tables, 307,511 applicants |
| Model | LightGBM, 709 features, 18 monotonic constraints |
| Calibration | isotonic, evaluated with nested 5-fold on OOF |
| CV | StratifiedKFold 5-fold, seed=42 |
| Final model | fit on 100% of train, 1400 trees |

**The number that matters most:** at a 70.0% approval rate, the bad rate among approved
applicants is 3.5% versus 8.1% if
everyone were approved. That is **56.5% less credit loss**,
blocking 69.5% of all defaults.

## 1. Ranking quality (OOF)

| Metric | Baseline (Block 1) | Engineered (Block 2-3) | Delta |
|---|---|---|---|
| AUC | 0.76076 | 0.78757 | **+0.02681** |
| Gini | 0.52152 | 0.57514 | +0.05362 |

Baseline = the 120 raw columns of `application_train`, no imputation, no reweighting.
Engineered = 709 features from all 7 tables. **Same fold split** (same
seed, same n_folds, same row order), so the delta measures the lift from feature engineering
rather than luck in how the data was split.

## 2. Calibration

ECE is reported under several binning schemes, because uniform binning flatters the result on
a skewed problem (77% of predictions fall below 0.1, so the first bin swallows nearly
everything):

| | Before calibration | After (isotonic) |
|---|---|---|
| Brier | 0.06598 | 0.06589 |
| ECE uniform-10 | 0.00417 | **0.00060** |
| ECE quantile-10 | 0.00372 | 0.00046 |
| ECE quantile-50 | 0.00597 | 0.00201 |

The improvement holds at the same magnitude across all three binning schemes, so it is not an
artifact of binning.

The "after" numbers are measured with **nested calibration**: split the OOF predictions into
5 parts, fit the calibrator on 4 of them and predict the rest, so every
point is calibrated by a model that never saw its label. Measuring the naive way (fit, then
predict on the same data) produces a falsely low ECE.

Platt/sigmoid makes ECE **worse**. The raw model is already fairly well calibrated (a
deliberate consequence of not using `scale_pos_weight`/`is_unbalance`), so forcing a rigid
sigmoid on top of it does damage.

## 3. Cutoff policy — turning the score into a decision

Computed on calibrated OOF predictions. "Approve X%" means approving the X% of applicants with
the lowest PD.

| Approval rate | PD cutoff | Bad rate among approved | Loss reduction | Defaults blocked | Relative loss |
|---|---|---|---|---|---|
| 50.0% | 0.0460 | 2.3% | 71.1% | 85.5% | 0.145 |
| 60.0% | 0.0604 | 2.8% | 64.8% | 78.9% | 0.211 |
| 70.0% | 0.0863 | 3.5% | 56.5% | 69.5% | 0.305 |
| 80.0% | 0.1261 | 4.4% | 45.7% | 56.6% | 0.434 |
| 90.0% | 0.1887 | 5.6% | 30.8% | 37.8% | 0.622 |
| 100.0% | 1.0000 | 8.1% | 0.0% | 0.0% | 1.000 |

*Relative loss*: assumes loss is proportional to the number of approved defaults and that
every loan carries the same exposure. Good enough to compare thresholds against each other,
but NOT a real loss model (it lacks LGD, EAD and per-loan amounts).

## 4. Segments and disparate impact

Gender and age are **protected attributes** in lending (ECOA/Reg B). The tables below are at
the reference approval rate of 70.0%:

### By gender
| Group | n | Actual bad rate | Mean PD | Calibration gap | AUC | Approval rate |
|---|---|---|---|---|---|---|
| F | 202,448 | 7.0% | 7.1% | +0.0008 | 0.7858 | 74.7% |
| M | 105,059 | 10.1% | 10.0% | -0.0014 | 0.7778 | 61.0% |
| XNA | 4 | 0.0% | 10.0% | +0.1001 | — | 25.0% |

### By age band
| Group | n | Actual bad rate | Mean PD | Calibration gap | AUC | Approval rate |
|---|---|---|---|---|---|---|
| 35-44 | 84,261 | 8.4% | 8.1% | -0.0032 | 0.7888 | 69.9% |
| 25-34 | 72,429 | 10.7% | 10.7% | +0.0000 | 0.7780 | 58.0% |
| 45-54 | 70,190 | 7.0% | 7.0% | -0.0003 | 0.7857 | 74.8% |
| 55-64 | 60,522 | 5.4% | 5.5% | +0.0011 | 0.7585 | 82.0% |
| <25 | 12,233 | 12.3% | 14.2% | +0.0192 | 0.7432 | 41.0% |
| 65+ | 7,876 | 3.7% | 3.6% | -0.0010 | 0.7722 | 91.9% |

**Adverse impact ratio** (EEOC 4/5ths rule, below 0.80 warrants investigation):

- Gender: **0.8162** — just above the line.
- Age band: **0.4464** — **fails.**

How to read this correctly: the model **calibrates very evenly** across groups (|mean PD −
actual bad rate| stays under 0.2pp everywhere except the under-25 band, which is off by
+1.9pp). In other words it is not "biased" in the sense of systematically mispredicting for
one group. The approval-rate gap reflects a genuine difference in risk in the data (a 12.3%
bad rate for under-25 versus 3.7% for 65+).

But "the approval rate differs because the risk genuinely differs" is **not a valid legal
defence**. The 4/5ths rule measures *impact*, not *intent*. In a real deployment this 0.4464
would have to go through compliance and would most likely require one of: dropping features
that proxy for age, setting group-specific cutoffs, or formally accepting and documenting the
compliance risk.

`CODE_GENDER` is **currently used as a feature** and can appear in reason codes. That would
not be permitted in a real system. It is kept in the demo so the segment tables above have
something to compare against, but it is the first thing to remove if this were productionized.

## 5. Explainability

- `shap.TreeExplainer` on `model.txt`, margin space, 5000 sampled rows (seed=42).
- Top 3 global: EXT_SOURCE_2, EXT_SOURCE_3, EXT_SOURCE_1.
- 7/18 features carrying a monotonic constraint land in the top-30 global SHAP
  importance, so the Block 3 domain reasoning matches what the model actually learned.
- Reason codes (adverse action): see `ml/src/reason_codes.py`. Hand-curated for the strong
  signals, with a readable fallback marked `curated=False` to flag what still needs legal
  review. The generated text is Vietnamese, matching the UI.
- File: `shap_global_importance.csv` (709 features, sorted by mean|SHAP|).

## 6. Known limitations

1. **Early stopping uses the same fold that produces the OOF prediction.** In `train.py` and
   `train_engineered.py` the validation fold both stops training and supplies `oof[va]`. The
   iteration count is therefore chosen using the very data it is about to predict, which makes
   AUC 0.78757 **mildly optimistic** rather than a fully clean OOF estimate. The delta
   against the baseline is still fair because both sides are biased identically. The fix is to
   carve out a separate inner validation split from the training portion.
2. **The calibrator is fit on OOF predictions but applied to a model trained on 100% of the
   data.** The final model is sharper than the 5-fold models, so its score distribution shifts
   slightly relative to what the calibrator learned. This is standard practice (equivalent to
   `CalibratedClassifierCV(ensemble=False)`), but it is an assumption, not a given.
3. **No hyperparameter tuning.** The values in `params.yaml` were chosen by hand; no Optuna or
   other search has been run.
4. **No stability analysis over time.** The Kaggle dataset has no clear time axis, so
   out-of-time validation — mandatory for a real scorecard — is not possible here.
5. **Relative loss is not a real loss model.** See the note in section 3.
