import numpy as np
import pandas as pd
import pytest

from ml.src.features.build import FeaturePipeline


def _synthetic_tables() -> dict[str, pd.DataFrame]:
    """3 applicants: #1 có đủ lịch sử 5 bảng phụ, #2 chỉ có 1 khoản bureau, #3 hoàn toàn mới."""
    application = pd.DataFrame({
        "SK_ID_CURR": [1, 2, 3],
        "TARGET": [0, 1, 0],
        "CODE_GENDER": ["M", "F", "M"],
        "AMT_INCOME_TOTAL": [100000.0, 200000.0, 150000.0],
        "DAYS_EMPLOYED": [-1000, 365243, -500],
    })
    bureau = pd.DataFrame({
        "SK_ID_CURR": [1, 1, 2],
        "SK_ID_BUREAU": [10, 11, 12],
        "CREDIT_ACTIVE": ["Active", "Closed", "Active"],
        "CREDIT_CURRENCY": ["currency 1"] * 3,
        "CREDIT_TYPE": ["Consumer credit", "Car loan", "Consumer credit"],
        "AMT_CREDIT_SUM": [1000.0, 500.0, 2000.0],
    })
    bureau_balance = pd.DataFrame({
        "SK_ID_BUREAU": [10, 10, 11],
        "MONTHS_BALANCE": [0, -1, 0],
        "STATUS": ["0", "1", "C"],
    })
    previous_application = pd.DataFrame({
        "SK_ID_CURR": [1],
        "SK_ID_PREV": [900],
        "NAME_CONTRACT_TYPE": ["Cash loans"],
        "NAME_CONTRACT_STATUS": ["Approved"],
        "NAME_CLIENT_TYPE": ["New"],
        "NAME_PORTFOLIO": ["POS"],
        "NAME_YIELD_GROUP": ["middle"],
        "CHANNEL_TYPE": ["Country-wide"],
        "FLAG_LAST_APPL_PER_CONTRACT": ["Y"],
        "AMT_CREDIT": [5000.0],
        "AMT_APPLICATION": [5000.0],
        "DAYS_FIRST_DRAWING": [365243],
        "DAYS_FIRST_DUE": [365243],
        "DAYS_LAST_DUE_1ST_VERSION": [365243],
        "DAYS_LAST_DUE": [365243],
        "DAYS_TERMINATION": [365243],
    })
    pos_cash = pd.DataFrame({
        "SK_ID_CURR": [1, 1],
        "SK_ID_PREV": [900, 900],
        "MONTHS_BALANCE": [-1, -2],
        "CNT_INSTALMENT": [12, 12],
        "CNT_INSTALMENT_FUTURE": [10, 11],
        "NAME_CONTRACT_STATUS": ["Active", "Active"],
        "SK_DPD": [0, 0],
        "SK_DPD_DEF": [0, 0],
    })
    installments = pd.DataFrame({
        "SK_ID_CURR": [1, 1],
        "SK_ID_PREV": [900, 900],
        "NUM_INSTALMENT_VERSION": [1, 1],
        "NUM_INSTALMENT_NUMBER": [1, 2],
        "DAYS_INSTALMENT": [-30, -1],
        "DAYS_ENTRY_PAYMENT": [-31, -1],
        "AMT_INSTALMENT": [500.0, 500.0],
        "AMT_PAYMENT": [500.0, 400.0],
    })
    credit_card = pd.DataFrame({
        "SK_ID_CURR": [1],
        "SK_ID_PREV": [901],
        "MONTHS_BALANCE": [-1],
        "AMT_BALANCE": [1000.0],
        "AMT_CREDIT_LIMIT_ACTUAL": [5000.0],
        "NAME_CONTRACT_STATUS": ["Active"],
        "SK_DPD": [0],
        "SK_DPD_DEF": [0],
    })
    return {
        "application": application,
        "bureau": bureau,
        "bureau_balance": bureau_balance,
        "previous_application": previous_application,
        "pos_cash": pos_cash,
        "installments": installments,
        "credit_card": credit_card,
    }


def test_pipeline_output_grain_is_one_row_per_applicant():
    tables = _synthetic_tables()
    pipeline = FeaturePipeline.fit(tables)
    flat = pipeline.transform(tables)

    assert len(flat) == 3
    assert not flat["SK_ID_CURR"].duplicated().any()


def test_applicant_with_no_history_gets_nan_not_zero():
    tables = _synthetic_tables()
    pipeline = FeaturePipeline.fit(tables)
    flat = pipeline.transform(tables)

    row3 = flat.loc[flat.SK_ID_CURR == 3].iloc[0]
    assert np.isnan(row3["BUREAU_AMT_CREDIT_SUM_SUM"])
    assert np.isnan(row3["PREV_AMT_CREDIT_MEAN"])


def test_days_employed_sentinel_cleaned_to_nan():
    """365243 = sentinel 'không áp dụng' của Home Credit (chủ yếu hưu trí), phải thành NaN."""
    tables = _synthetic_tables()
    pipeline = FeaturePipeline.fit(tables)
    flat = pipeline.transform(tables)

    row2 = flat.loc[flat.SK_ID_CURR == 2].iloc[0]
    row1 = flat.loc[flat.SK_ID_CURR == 1].iloc[0]
    assert np.isnan(row2["DAYS_EMPLOYED"])
    assert row1["DAYS_EMPLOYED"] == -1000


def test_installments_derived_features_computed_before_aggregation():
    tables = _synthetic_tables()
    pipeline = FeaturePipeline.fit(tables)
    flat = pipeline.transform(tables)

    row1 = flat.loc[flat.SK_ID_CURR == 1].iloc[0]
    # trả 1: đúng hạn & đủ tiền; trả 2: đúng hạn nhưng thiếu 100 -> tổng PAYMENT_DIFF = 100
    assert row1["INSTAL_PAYMENT_DIFF_SUM"] == pytest.approx(100.0)


def test_serving_single_applicant_keeps_same_columns_as_fit_train():
    """Chống train-serve skew: serve 1 applicant ít category hơn vẫn phải ra ĐÚNG bộ cột lúc fit."""
    tables = _synthetic_tables()
    pipeline = FeaturePipeline.fit(tables)
    flat_train = pipeline.transform(tables)

    serve_tables = {
        "application": tables["application"][tables["application"].SK_ID_CURR == 2].reset_index(drop=True),
        "bureau": tables["bureau"][tables["bureau"].SK_ID_CURR == 2].reset_index(drop=True),
        "bureau_balance": tables["bureau_balance"].iloc[0:0],
        "previous_application": tables["previous_application"].iloc[0:0],
        "pos_cash": tables["pos_cash"].iloc[0:0],
        "installments": tables["installments"].iloc[0:0],
        "credit_card": tables["credit_card"].iloc[0:0],
    }
    flat_serve = pipeline.transform(serve_tables)

    assert set(flat_serve.columns) == set(flat_train.columns)


def test_pipeline_save_load_roundtrip_produces_identical_transform(tmp_path):
    tables = _synthetic_tables()
    pipeline = FeaturePipeline.fit(tables)
    flat_before = pipeline.transform(tables)

    path = tmp_path / "pipeline.pkl"
    pipeline.save(path)
    reloaded = FeaturePipeline.load(path)
    flat_after = reloaded.transform(tables)

    pd.testing.assert_frame_equal(flat_before, flat_after)


def test_build_module_khong_co_main_guard():
    """Regression: `if __name__ == "__main__"` trong build.py làm hỏng pickle.

    Chạy `python -m ml.src.features.build` sẽ nạp module này AS `__main__`, khiến
    `FeaturePipeline.__module__ == "__main__"` lúc pickle -> feature_pipeline.pkl
    không unpickle được từ backend/pytest. Bug này đã xảy ra 2 lần (lần 2 là do
    khối guard bị thêm lại ngay dưới chính comment cảnh báo nó).
    Entry point đúng: `python -m ml.src.run_build_features`.
    """
    from pathlib import Path

    source = (Path(__file__).parent.parent / "src" / "features" / "build.py").read_text()
    executable_guards = [
        line for line in source.splitlines()
        if line.startswith("if __name__")  # trong comment thì luôn có tiền tố '#'
    ]
    assert executable_guards == [], f"build.py không được có __main__ guard: {executable_guards}"


def test_feature_pipeline_pickle_duoc_tu_module_dung():
    """FeaturePipeline phải mang __module__ thật để unpickle được ở mọi entry point."""
    from ml.src.features.build import FeaturePipeline

    assert FeaturePipeline.__module__ == "ml.src.features.build"
