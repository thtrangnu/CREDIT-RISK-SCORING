"""Block 9: turning the score into a DECISION — cutoff policy + segment comparison.

Why this block exists: AUC and ECE answer "how well does the model rank", not "what do
we get by using it". In lending the real question is: at a given approval rate, how much
does the default rate among approved applicants drop? That is the number you can take to
the business.

Every figure here is computed on CALIBRATED OOF predictions, not in-sample: each applicant
is scored by a model that never saw it during training, and calibrated by a model that
never saw its label (see calibrate.py).

Segments (`segment_report`) live here rather than in metrics.py because they answer a
POLICY question (who is affected and how), not a purely statistical one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import roc_auc

# Approval rates to sweep. Not a recommendation, just a range for reading the trade-off.
DEFAULT_APPROVAL_RATES = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

# Reference threshold used for segment reporting. 0.7 = approve the best 70% of applicants.
REFERENCE_APPROVAL_RATE = 0.7

# EEOC "4/5ths rule": if the least-favoured group's approval rate divided by the most-favoured
# group's falls below 0.8, that is a sign of adverse impact worth investigating.
ADVERSE_IMPACT_THRESHOLD = 0.8


def approve_mask(pd_score: np.ndarray, approval_rate: float) -> np.ndarray:
    """Approve the `approval_rate` share of applicants with the LOWEST PD.

    Cuts by quantile rather than by an absolute PD threshold. Holding the approval rate
    fixed is the fair way to compare models or versions, and it is closer to how a risk
    team actually operates (set the volume target first, derive the cutoff from it).
    """
    if not 0.0 < approval_rate <= 1.0:
        raise ValueError(f"approval_rate must be in (0, 1], got {approval_rate}")
    if approval_rate == 1.0:
        return np.ones(len(pd_score), dtype=bool)
    order = np.argsort(pd_score, kind="mergesort")
    k = int(round(len(pd_score) * approval_rate))
    mask = np.zeros(len(pd_score), dtype=bool)
    mask[order[:k]] = True
    return mask


def cutoff_table(
    y_true, pd_score, approval_rates: tuple[float, ...] = DEFAULT_APPROVAL_RATES
) -> pd.DataFrame:
    """Trade-off table: how much you approve vs how much defaults vs how many bads you block."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(pd_score, dtype=float)
    base_rate = float(y.mean())
    total_bad = int(y.sum())

    rows = []
    for rate in approval_rates:
        mask = approve_mask(p, rate)
        n_approved = int(mask.sum())
        bad_approved = int(y[mask].sum())
        bad_rate_approved = bad_approved / n_approved if n_approved else float("nan")
        rows.append({
            "approval_rate": round(rate, 4),
            "n_approved": n_approved,
            "pd_cutoff": float(p[mask].max()) if n_approved else float("nan"),
            "bad_rate_approved": bad_rate_approved,
            # How much the default rate drops versus approving everyone. This is the
            # figure that tells the story.
            "bad_rate_reduction": 1.0 - bad_rate_approved / base_rate,
            # Share of all defaults that get stopped at the door.
            "bad_captured": (total_bad - bad_approved) / total_bad if total_bad else float("nan"),
            # Relative expected loss, assuming loss scales with the number of approved
            # defaults and every loan carries the same exposure. Normalized so
            # 1.0 = approve everyone.
            "expected_loss_index": bad_approved / total_bad if total_bad else float("nan"),
        })
    return pd.DataFrame(rows)


def segment_report(
    y_true, pd_score, groups, approval_rate: float = REFERENCE_APPROVAL_RATE
) -> pd.DataFrame:
    """Model quality and policy impact per segment.

    How to read the columns:
      bad_rate vs mean_pd far apart -> the model is miscalibrated for that group
      auc noticeably lower          -> the model discriminates poorly on that group
      approval_rate far apart       -> the policy affects groups unevenly
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(pd_score, dtype=float)
    approved = approve_mask(p, approval_rate)

    df = pd.DataFrame({"y": y, "p": p, "approved": approved, "group": np.asarray(groups)})
    rows = []
    for name, sub in df.groupby("group", dropna=False):
        # AUC is meaningless for a single-class group; report NaN rather than a fake number.
        has_both = 0 < sub["y"].sum() < len(sub)
        rows.append({
            "group": name,
            "n": len(sub),
            "bad_rate": float(sub["y"].mean()),
            "mean_pd": float(sub["p"].mean()),
            "calibration_gap": float(sub["p"].mean() - sub["y"].mean()),
            "auc": roc_auc(sub["y"].values, sub["p"].values) if has_both else float("nan"),
            "approval_rate": float(sub["approved"].mean()),
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def adverse_impact_ratio(segments: pd.DataFrame, min_group_size: int = 1000) -> float:
    """min(approval_rate) / max(approval_rate), the EEOC "4/5ths rule".

    Groups that are too small (under 1000 by default) are skipped. A few dozen rows cannot
    support a conclusion, but they can drag the ratio down through pure noise (for example
    CODE_GENDER='XNA', which has 4 rows).
    """
    big = segments[segments["n"] >= min_group_size]
    if len(big) < 2:
        return float("nan")
    return float(big["approval_rate"].min() / big["approval_rate"].max())


def age_bands(days_birth) -> np.ndarray:
    """DAYS_BIRTH (negative, counted back from the application date) -> age band labels.

    Age is also a protected attribute in lending (ECOA), so it belongs in the segment
    report alongside gender rather than gender alone.
    """
    years = -np.asarray(days_birth, dtype=float) / 365.25
    bins = [0, 25, 35, 45, 55, 65, 200]
    labels = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]
    return pd.cut(years, bins=bins, labels=labels, right=False).astype(str)
