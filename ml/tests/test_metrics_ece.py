import numpy as np
import pytest

from ml.src.metrics import expected_calibration_error


def test_ece_is_zero_when_predictions_match_observed_frequency():
    # 1000 rows at p=0.2 with exactly 20% ones, 1000 rows at p=0.8 with exactly 80% ones.
    y = np.r_[np.zeros(800), np.ones(200), np.zeros(200), np.ones(800)]
    p = np.r_[np.full(1000, 0.2), np.full(1000, 0.8)]
    assert expected_calibration_error(y, p) == pytest.approx(0.0, abs=1e-12)
    assert expected_calibration_error(y, p, strategy="quantile") == pytest.approx(0.0, abs=1e-12)


def test_ece_detects_systematic_bias():
    # Predicting 0.5 everywhere while reality is 10% -> ECE = |0.5 - 0.1| = 0.4.
    y = np.r_[np.zeros(900), np.ones(100)]
    p = np.full(1000, 0.5)
    assert expected_calibration_error(y, p) == pytest.approx(0.4)


def test_quantile_binning_catches_error_that_uniform_bins_cancel_out():
    """The real weakness of uniform binning: errors in OPPOSITE directions inside one
    wide bin cancel each other out. Here every prediction is below 0.1, so uniform-10
    dumps them all into bin [0, 0.1): the p=0.02 group is over-predicted and the p=0.08
    group is under-predicted. Averaged together they nearly match, so uniform ECE is
    about 0 even though the model really is off. Quantile binning separates the two
    groups and sees the error.
    """
    p_low, p_high = np.full(10_000, 0.02), np.full(10_000, 0.08)
    y_low = np.r_[np.ones(0), np.zeros(10_000)]          # actually 0% (over-predicted by 0.02)
    y_high = np.r_[np.ones(1_000), np.zeros(9_000)]      # actually 10% (under-predicted by 0.02)
    p = np.r_[p_low, p_high]
    y = np.r_[y_low, y_high]

    uniform = expected_calibration_error(y, p, n_bins=10, strategy="uniform")
    quantile = expected_calibration_error(y, p, n_bins=10, strategy="quantile")
    assert uniform == pytest.approx(0.0, abs=1e-9)   # completely hidden
    assert quantile == pytest.approx(0.02, abs=1e-9)  # sees the true gap


def test_ece_rejects_an_invalid_strategy():
    with pytest.raises(ValueError):
        expected_calibration_error([0, 1], [0.1, 0.9], strategy="kmeans")
