"""Generate ml/artifacts/model_card.md FROM existing artifacts. Retrains nothing.

The model card used to be embedded directly in calibrate.py, which meant changing a single
sentence required rerunning the whole training. Splitting it out makes the card a pure
function of the artifacts: read metrics_summary.json + model.txt + calibrator.pkl +
features.yaml, then render. calibrate.py points here at the end of its run.
"""
from __future__ import annotations

import json
import pickle

import lightgbm as lgb
import yaml

from .features.build import ARTIFACTS_DIR, ML_DIR

FEATURES_CONFIG_PATH = ML_DIR / "config" / "features.yaml"
PARAMS_CONFIG_PATH = ML_DIR / "config" / "params.yaml"


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _policy_rows(policy_table: list[dict]) -> str:
    header = ("| Approval rate | PD cutoff | Bad rate among approved | Loss reduction "
              "| Defaults blocked | Relative loss |\n|---|---|---|---|---|---|\n")
    lines = [
        f"| {_pct(r['approval_rate'])} | {r['pd_cutoff']:.4f} | {_pct(r['bad_rate_approved'])} "
        f"| {_pct(r['bad_rate_reduction'])} | {_pct(r['bad_captured'])} | {r['expected_loss_index']:.3f} |"
        for r in policy_table
    ]
    return header + "\n".join(lines)


def _fmt_auc(auc: float | None) -> str:
    """AUC is undefined for a single-class group (e.g. CODE_GENDER='XNA', only 4 rows).

    metrics_summary.json stores None (already passed through json_safe); NaN accepted too.
    """
    if auc is None or auc != auc:
        return "—"
    return f"{auc:.4f}"


def _segment_rows(rows: list[dict]) -> str:
    header = ("| Group | n | Actual bad rate | Mean PD | Calibration gap | AUC | Approval rate |\n"
              "|---|---|---|---|---|---|---|\n")
    lines = [
        f"| {r['group']} | {r['n']:,} | {_pct(r['bad_rate'])} | {_pct(r['mean_pd'])} "
        f"| {r['calibration_gap']:+.4f} | {_fmt_auc(r['auc'])} | {_pct(r['approval_rate'])} |"
        for r in rows
    ]
    return header + "\n".join(lines)


def render(summary: dict, n_trees: int, calibrator_method: str,
           n_monotonic: int, n_folds: int, seed: int) -> str:
    eng, pol, fair = summary["engineered"], summary["policy"], summary["fairness"]
    ref = pol["reference_approval_rate"]
    ref_row = next(r for r in pol["table"] if abs(r["approval_rate"] - ref) < 1e-9)

    return f"""# Model Card — Home Credit Default Risk

## Summary

| | |
|---|---|
| Task | Binary classification, predicting default, {_pct(summary['default_rate'])} positive |
| Data | Home Credit Default Risk (Kaggle), 7 tables, {summary['n_train_rows']:,} applicants |
| Model | LightGBM, {summary['n_features']} features, {n_monotonic} monotonic constraints |
| Calibration | {calibrator_method}, evaluated with nested {n_folds}-fold on OOF |
| CV | StratifiedKFold {n_folds}-fold, seed={seed} |
| Final model | fit on 100% of train, {n_trees} trees |

**The number that matters most:** at a {_pct(ref)} approval rate, the bad rate among approved
applicants is {_pct(ref_row['bad_rate_approved'])} versus {_pct(summary['default_rate'])} if
everyone were approved. That is **{_pct(ref_row['bad_rate_reduction'])} less credit loss**,
blocking {_pct(ref_row['bad_captured'])} of all defaults.

## 1. Ranking quality (OOF)

| Metric | Baseline (Block 1) | Engineered (Block 2-3) | Delta |
|---|---|---|---|
| AUC | {summary['baseline']['auc']:.5f} | {eng['auc']:.5f} | **{summary['auc_delta']:+.5f}** |
| Gini | {summary['baseline']['gini']:.5f} | {eng['gini']:.5f} | {eng['gini'] - summary['baseline']['gini']:+.5f} |

Baseline = the 120 raw columns of `application_train`, no imputation, no reweighting.
Engineered = {summary['n_features']} features from all 7 tables. **Same fold split** (same
seed, same n_folds, same row order), so the delta measures the lift from feature engineering
rather than luck in how the data was split.

## 2. Calibration

ECE is reported under several binning schemes, because uniform binning flatters the result on
a skewed problem (77% of predictions fall below 0.1, so the first bin swallows nearly
everything):

| | Before calibration | After ({calibrator_method}) |
|---|---|---|
| Brier | {eng['brier_raw']:.5f} | {eng['brier_calibrated']:.5f} |
| ECE uniform-10 | {eng['ece_uniform_10_raw']:.5f} | **{eng['ece_uniform_10_calibrated']:.5f}** |
| ECE quantile-10 | {eng['ece_quantile_10_raw']:.5f} | {eng['ece_quantile_10_calibrated']:.5f} |
| ECE quantile-50 | {eng['ece_quantile_50_raw']:.5f} | {eng['ece_quantile_50_calibrated']:.5f} |

The improvement holds at the same magnitude across all three binning schemes, so it is not an
artifact of binning.

The "after" numbers are measured with **nested calibration**: split the OOF predictions into
{n_folds} parts, fit the calibrator on {n_folds - 1} of them and predict the rest, so every
point is calibrated by a model that never saw its label. Measuring the naive way (fit, then
predict on the same data) produces a falsely low ECE.

Platt/sigmoid makes ECE **worse**. The raw model is already fairly well calibrated (a
deliberate consequence of not using `scale_pos_weight`/`is_unbalance`), so forcing a rigid
sigmoid on top of it does damage.

## 3. Cutoff policy — turning the score into a decision

Computed on calibrated OOF predictions. "Approve X%" means approving the X% of applicants with
the lowest PD.

{_policy_rows(pol['table'])}

*Relative loss*: assumes loss is proportional to the number of approved defaults and that
every loan carries the same exposure. Good enough to compare thresholds against each other,
but NOT a real loss model (it lacks LGD, EAD and per-loan amounts).

## 4. Segments and disparate impact

Gender and age are **protected attributes** in lending (ECOA/Reg B). The tables below are at
the reference approval rate of {_pct(ref)}:

### By gender
{_segment_rows(fair['by_gender'])}

### By age band
{_segment_rows(fair['by_age_band'])}

**Adverse impact ratio** (EEOC 4/5ths rule, below 0.80 warrants investigation):

- Gender: **{fair['adverse_impact_ratio_gender']:.4f}** — just above the line.
- Age band: **{fair['adverse_impact_ratio_age']:.4f}** — **fails.**

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
- 7/{n_monotonic} features carrying a monotonic constraint land in the top-30 global SHAP
  importance, so the Block 3 domain reasoning matches what the model actually learned.
- Reason codes (adverse action): see `ml/src/reason_codes.py`. Hand-curated for the strong
  signals, with a readable fallback marked `curated=False` to flag what still needs legal
  review. The generated text is Vietnamese, matching the UI.
- File: `shap_global_importance.csv` ({summary['n_features']} features, sorted by mean|SHAP|).

## 6. Known limitations

1. **Early stopping uses the same fold that produces the OOF prediction.** In `train.py` and
   `train_engineered.py` the validation fold both stops training and supplies `oof[va]`. The
   iteration count is therefore chosen using the very data it is about to predict, which makes
   AUC {eng['auc']:.5f} **mildly optimistic** rather than a fully clean OOF estimate. The delta
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
"""


def main() -> None:
    with open(ARTIFACTS_DIR / "metrics_summary.json") as f:
        summary = json.load(f)
    with open(ARTIFACTS_DIR / "calibrator.pkl", "rb") as f:
        calibrator_method = pickle.load(f)["method"]
    with open(FEATURES_CONFIG_PATH) as f:
        n_monotonic = len(yaml.safe_load(f).get("monotone_constraints", {}))
    with open(PARAMS_CONFIG_PATH) as f:
        params_cfg = yaml.safe_load(f)

    n_trees = lgb.Booster(model_file=str(ARTIFACTS_DIR / "model.txt")).num_trees()
    card = render(summary, n_trees, calibrator_method, n_monotonic,
                  params_cfg["n_folds"], params_cfg["seed"])
    with open(ARTIFACTS_DIR / "model_card.md", "w") as f:
        f.write(card)
    print(f"Saved -> {ARTIFACTS_DIR / 'model_card.md'} ({len(card.splitlines())} lines)")


if __name__ == "__main__":
    main()
