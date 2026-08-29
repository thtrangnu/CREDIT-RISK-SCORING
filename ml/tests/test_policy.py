import numpy as np
import pandas as pd
import pytest

from ml.src.policy import (adverse_impact_ratio, age_bands, approve_mask,
                           cutoff_table, segment_report)


@pytest.fixture
def separable():
    """200 rows where the score ranks risk PERFECTLY: the 20 bads have the highest PD."""
    y = np.r_[np.zeros(180, dtype=int), np.ones(20, dtype=int)]
    p = np.r_[np.linspace(0.01, 0.30, 180), np.linspace(0.60, 0.95, 20)]
    return y, p


def test_approve_mask_takes_the_right_share_and_the_lowest_pd_group(separable):
    _, p = separable
    mask = approve_mask(p, 0.7)
    assert mask.sum() == 140
    # No rejected application has a lower PD than an approved one.
    assert p[mask].max() <= p[~mask].min()


def test_approve_mask_at_100_percent_approves_everyone(separable):
    _, p = separable
    assert approve_mask(p, 1.0).all()


def test_approve_mask_rejects_out_of_range_rates():
    with pytest.raises(ValueError):
        approve_mask(np.array([0.1, 0.2]), 0.0)
    with pytest.raises(ValueError):
        approve_mask(np.array([0.1, 0.2]), 1.5)


def test_cutoff_table_approving_everyone_reduces_nothing(separable):
    y, p = separable
    row = cutoff_table(y, p, approval_rates=(1.0,)).iloc[0]
    assert row["bad_rate_approved"] == pytest.approx(y.mean())
    assert row["bad_rate_reduction"] == pytest.approx(0.0)
    assert row["expected_loss_index"] == pytest.approx(1.0)


def test_cutoff_table_tightening_lowers_bad_rate_monotonically(separable):
    y, p = separable
    tbl = cutoff_table(y, p, approval_rates=(0.5, 0.7, 0.9, 1.0))
    assert tbl["bad_rate_approved"].is_monotonic_increasing
    assert tbl["bad_captured"].is_monotonic_decreasing


def test_cutoff_table_perfect_model_blocks_every_bad(separable):
    y, p = separable
    row = cutoff_table(y, p, approval_rates=(0.9,)).iloc[0]  # 20/200 = 10% are bad
    assert row["bad_rate_approved"] == pytest.approx(0.0)
    assert row["bad_captured"] == pytest.approx(1.0)


def test_segment_report_splits_groups_and_uses_one_shared_threshold(separable):
    y, p = separable
    groups = np.array(["A"] * 100 + ["B"] * 100)
    rep = segment_report(y, p, groups, approval_rate=0.5)
    assert set(rep["group"]) == {"A", "B"}
    assert rep["n"].sum() == 200
    # Group A is all low PD so it is fully approved; group B holds every bad so fewer pass.
    assert rep.set_index("group").loc["A", "approval_rate"] > rep.set_index("group").loc["B", "approval_rate"]


def test_segment_report_auc_is_nan_for_single_class_group():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    rep = segment_report(y, p, np.array(["allgood"] * 3 + ["allbad"] * 3), approval_rate=0.5)
    assert rep["auc"].isna().all()


def test_adverse_impact_ratio_skips_groups_that_are_too_small():
    segments = pd.DataFrame({
        "group": ["big_a", "big_b", "tiny"],
        "n": [5000, 5000, 4],
        "approval_rate": [0.80, 0.60, 0.0],
    })
    # Including the 4-row group would give 0/0.8 = 0, pure noise. Skipping it gives 0.6/0.8.
    assert adverse_impact_ratio(segments, min_group_size=1000) == pytest.approx(0.75)


def test_adverse_impact_ratio_is_nan_without_two_large_groups():
    segments = pd.DataFrame({"group": ["a"], "n": [5000], "approval_rate": [0.7]})
    assert np.isnan(adverse_impact_ratio(segments))


def test_age_bands_maps_negative_days_birth_to_bands():
    # -25 * 365.25 days is exactly 25 years, which falls in the "25-34" band (right=False).
    days = np.array([-20 * 365.25, -25 * 365.25, -40 * 365.25, -70 * 365.25])
    assert list(age_bands(days)) == ["<25", "25-34", "35-44", "65+"]
