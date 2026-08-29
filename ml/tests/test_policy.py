import numpy as np
import pandas as pd
import pytest

from ml.src.policy import (adverse_impact_ratio, age_bands, approve_mask,
                           cutoff_table, segment_report)


@pytest.fixture
def separable():
    """200 dòng, score dự đoán ĐÚNG thứ hạng rủi ro: 20 bad có PD cao nhất."""
    y = np.r_[np.zeros(180, dtype=int), np.ones(20, dtype=int)]
    p = np.r_[np.linspace(0.01, 0.30, 180), np.linspace(0.60, 0.95, 20)]
    return y, p


def test_approve_mask_duyet_dung_ti_le_va_dung_nhom_pd_thap(separable):
    _, p = separable
    mask = approve_mask(p, 0.7)
    assert mask.sum() == 140
    # Không hồ sơ nào bị từ chối lại có PD thấp hơn hồ sơ được duyệt.
    assert p[mask].max() <= p[~mask].min()


def test_approve_mask_100_phan_tram_duyet_het(separable):
    _, p = separable
    assert approve_mask(p, 1.0).all()


def test_approve_mask_tu_choi_ti_le_ngoai_khoang():
    with pytest.raises(ValueError):
        approve_mask(np.array([0.1, 0.2]), 0.0)
    with pytest.raises(ValueError):
        approve_mask(np.array([0.1, 0.2]), 1.5)


def test_cutoff_table_duyet_het_khong_giam_rui_ro(separable):
    y, p = separable
    row = cutoff_table(y, p, approval_rates=(1.0,)).iloc[0]
    assert row["bad_rate_approved"] == pytest.approx(y.mean())
    assert row["bad_rate_reduction"] == pytest.approx(0.0)
    assert row["expected_loss_index"] == pytest.approx(1.0)


def test_cutoff_table_that_chat_thi_bad_rate_giam_don_dieu(separable):
    y, p = separable
    tbl = cutoff_table(y, p, approval_rates=(0.5, 0.7, 0.9, 1.0))
    assert tbl["bad_rate_approved"].is_monotonic_increasing
    assert tbl["bad_captured"].is_monotonic_decreasing


def test_cutoff_table_model_hoan_hao_chan_het_bad(separable):
    y, p = separable
    row = cutoff_table(y, p, approval_rates=(0.9,)).iloc[0]  # 20/200 = 10% là bad
    assert row["bad_rate_approved"] == pytest.approx(0.0)
    assert row["bad_captured"] == pytest.approx(1.0)


def test_segment_report_tach_dung_nhom_va_tinh_approval_theo_nguong_chung(separable):
    y, p = separable
    groups = np.array(["A"] * 100 + ["B"] * 100)
    rep = segment_report(y, p, groups, approval_rate=0.5)
    assert set(rep["group"]) == {"A", "B"}
    assert rep["n"].sum() == 200
    # Nhóm A toàn PD thấp -> được duyệt hết; nhóm B chứa toàn bộ bad -> duyệt ít hơn.
    assert rep.set_index("group").loc["A", "approval_rate"] > rep.set_index("group").loc["B", "approval_rate"]


def test_segment_report_auc_la_nan_khi_nhom_chi_co_1_lop():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    rep = segment_report(y, p, np.array(["allgood"] * 3 + ["allbad"] * 3), approval_rate=0.5)
    assert rep["auc"].isna().all()


def test_adverse_impact_ratio_bo_qua_nhom_qua_nho():
    segments = pd.DataFrame({
        "group": ["big_a", "big_b", "tiny"],
        "n": [5000, 5000, 4],
        "approval_rate": [0.80, 0.60, 0.0],
    })
    # Nếu tính cả nhóm 4 dòng, tỉ số = 0/0.8 = 0 (nhiễu). Bỏ qua -> 0.6/0.8.
    assert adverse_impact_ratio(segments, min_group_size=1000) == pytest.approx(0.75)


def test_adverse_impact_ratio_nan_khi_khong_du_2_nhom_lon():
    segments = pd.DataFrame({"group": ["a"], "n": [5000], "approval_rate": [0.7]})
    assert np.isnan(adverse_impact_ratio(segments))


def test_age_bands_doi_days_birth_am_sang_nhom_tuoi():
    # -25 * 365.25 ngày ~ 25 tuổi tròn -> rơi vào band "25-34" (right=False).
    days = np.array([-20 * 365.25, -25 * 365.25, -40 * 365.25, -70 * 365.25])
    assert list(age_bands(days)) == ["<25", "25-34", "35-44", "65+"]
