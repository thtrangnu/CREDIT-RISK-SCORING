import numpy as np
import pytest

from ml.src.metrics import expected_calibration_error


def test_ece_bang_0_khi_prediction_khop_hoan_hao_tan_suat():
    # 1000 dòng ở p=0.2 với đúng 20% là 1, 1000 dòng ở p=0.8 với đúng 80% là 1.
    y = np.r_[np.zeros(800), np.ones(200), np.zeros(200), np.ones(800)]
    p = np.r_[np.full(1000, 0.2), np.full(1000, 0.8)]
    assert expected_calibration_error(y, p) == pytest.approx(0.0, abs=1e-12)
    assert expected_calibration_error(y, p, strategy="quantile") == pytest.approx(0.0, abs=1e-12)


def test_ece_bat_duoc_do_lech_co_he_thong():
    # Dự đoán 0.5 cho toàn bộ trong khi thực tế 10% -> ECE = |0.5 - 0.1| = 0.4.
    y = np.r_[np.zeros(900), np.ones(100)]
    p = np.full(1000, 0.5)
    assert expected_calibration_error(y, p) == pytest.approx(0.4)


def test_quantile_binning_bat_duoc_sai_lech_bi_uniform_bin_trung_hoa():
    """Điểm yếu thật của uniform binning: sai lệch NGƯỢC CHIỀU trong cùng 1 bin rộng
    triệt tiêu lẫn nhau. Ở đây toàn bộ prediction < 0.1 nên uniform-10 nhét hết vào
    bin [0, 0.1): nhóm p=0.02 bị dự đoán CAO hơn thực tế, nhóm p=0.08 bị dự đoán
    THẤP hơn — trung bình lại thì gần như khớp, ECE uniform ~ 0 dù model lệch thật.
    Quantile binning tách 2 nhóm ra nên nhìn thấy sai lệch.
    """
    p_low, p_high = np.full(10_000, 0.02), np.full(10_000, 0.08)
    y_low = np.r_[np.ones(0), np.zeros(10_000)]          # thực tế 0% (dự đoán thừa 0.02)
    y_high = np.r_[np.ones(1_000), np.zeros(9_000)]      # thực tế 10% (dự đoán thiếu 0.02)
    p = np.r_[p_low, p_high]
    y = np.r_[y_low, y_high]

    uniform = expected_calibration_error(y, p, n_bins=10, strategy="uniform")
    quantile = expected_calibration_error(y, p, n_bins=10, strategy="quantile")
    assert uniform == pytest.approx(0.0, abs=1e-9)   # bị che hoàn toàn
    assert quantile == pytest.approx(0.02, abs=1e-9)  # nhìn thấy đúng độ lệch


def test_ece_tu_choi_strategy_khong_hop_le():
    with pytest.raises(ValueError):
        expected_calibration_error([0, 1], [0.1, 0.9], strategy="kmeans")
