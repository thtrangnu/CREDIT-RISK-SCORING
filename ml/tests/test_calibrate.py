import numpy as np
import pytest

from ml.src.calibrate import fit_calibrator, nested_calibrate, predict_calibrated
from ml.src.metrics import expected_calibration_error


@pytest.fixture
def miscalibrated_data():
    """Scores squashed toward zero (systematically underconfident) vs true probability -> high ECE."""
    rng = np.random.default_rng(42)
    n = 4000
    true_p = rng.uniform(0.05, 0.95, size=n)
    y = rng.binomial(1, true_p)
    raw_score = np.clip(true_p * 0.5, 0.0, 1.0)  # squashed toward 0 -> miscalibrated
    return raw_score, y


@pytest.mark.parametrize("method", ["isotonic", "sigmoid"])
def test_calibrated_predictions_are_bounded_probabilities(method, miscalibrated_data):
    score, y = miscalibrated_data
    model = fit_calibrator(score, y, method)
    calibrated = predict_calibrated(model, score, method)

    assert calibrated.min() >= 0.0
    assert calibrated.max() <= 1.0
    assert len(calibrated) == len(score)


def test_isotonic_calibration_is_monotonic_wrt_raw_score(miscalibrated_data):
    score, y = miscalibrated_data
    model = fit_calibrator(score, y, "isotonic")
    calibrated = predict_calibrated(model, score, "isotonic")

    order = np.argsort(score)
    assert np.all(np.diff(calibrated[order]) >= -1e-9)


@pytest.mark.parametrize("method", ["isotonic", "sigmoid"])
def test_nested_calibrate_improves_ece_on_systematically_miscalibrated_scores(method, miscalibrated_data):
    score, y = miscalibrated_data
    ece_before = expected_calibration_error(y, score)

    calibrated = nested_calibrate(score, y, method, n_folds=5, seed=0)
    ece_after = expected_calibration_error(y, calibrated)

    assert ece_after < ece_before
    assert len(calibrated) == len(score)


def test_nested_calibrate_never_lets_a_fold_see_its_own_label(miscalibrated_data):
    """Indirect sanity check: nested calibration on pure noise must not produce an
    unrealistically low ECE through leakage, compared with calibrating once on everything."""
    score, y = miscalibrated_data
    naive_model = fit_calibrator(score, y, "isotonic")
    naive_calibrated = predict_calibrated(naive_model, score, "isotonic")
    naive_ece = expected_calibration_error(y, naive_calibrated)

    nested = nested_calibrate(score, y, "isotonic", n_folds=5, seed=0)
    nested_ece = expected_calibration_error(y, nested)

    # nested (out-of-fold, honest) must not beat naive (in-sample, optimistic) by much
    assert nested_ece >= naive_ece - 1e-6
