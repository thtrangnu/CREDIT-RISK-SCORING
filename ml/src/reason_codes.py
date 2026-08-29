"""Feature -> câu giải thích tiếng Việt, dùng cho reason codes (adverse action).

Đây là "contract" thuần dữ liệu/hàm thuần (không train, không load model) nên
backend/ được phép import module này (xem ghi chú biên giới tầng trong
scorer.py) — giống cách backend import FeaturePipeline.

Trong hệ thống thật, thư viện câu chữ adverse-action phải được pháp chế duyệt
từng dòng. Vì vậy đây là bảng CURATE TAY cho feature tín hiệu mạnh nhất (khớp
18 feature có monotonic constraint ở features.yaml + vài feature SHAP quan
trọng khác), KHÔNG auto-generate cho mọi trường hợp. Feature ngoài danh sách
vẫn ra được câu đọc hiểu được (fallback), nhưng đánh dấu `curated=False` để
biết cần review pháp chế trước khi dùng thật.
"""
from __future__ import annotations

import numpy as np

TABLE_LABELS = {
    "BUREAU": "lịch sử tín dụng ở tổ chức khác",
    "PREV": "lịch sử vay tại Home Credit",
    "POS": "lịch sử vay POS/tiền mặt",
    "INSTAL": "lịch sử trả góp",
    "CC": "lịch sử thẻ tín dụng",
}

# Sắp theo độ dài giảm dần để match suffix dài nhất trước (NUNIQUE_PREV trước PREV).
AGG_LABELS = {
    "NUNIQUE_PREV": "số khoản vay trước có",
    "RECORDS": "số bản ghi lịch sử",
    "MEAN": "trung bình",
    "SUM": "tổng",
    "MAX": "cao nhất",
    "MIN": "thấp nhất",
    "STD": "độ biến động",
    "COUNT": "số lượng",
    "SHARE": "tỉ lệ",
}

RAW_COLUMN_LABELS = {
    "EXT_SOURCE_1": "điểm tín dụng ngoài #1",
    "EXT_SOURCE_2": "điểm tín dụng ngoài #2",
    "EXT_SOURCE_3": "điểm tín dụng ngoài #3",
    "DAYS_BIRTH": "tuổi",
    "DAYS_EMPLOYED": "thâm niên công việc hiện tại",
    "AMT_INCOME_TOTAL": "tổng thu nhập",
    "AMT_CREDIT": "số tiền vay",
    "AMT_ANNUITY": "khoản trả góp hàng năm",
    "REGION_RATING_CLIENT": "xếp hạng rủi ro khu vực sinh sống",
    "REGION_RATING_CLIENT_W_CITY": "xếp hạng rủi ro khu vực (kèm thành phố)",
    "CREDIT_DAY_OVERDUE": "số ngày quá hạn",
    "AMT_CREDIT_SUM_OVERDUE": "số tiền quá hạn",
    "AMT_CREDIT_SUM": "tổng dư nợ",
    "DAYS_LATE": "số ngày trả trễ",
    "PAYMENT_DIFF": "số tiền trả thiếu",
    "SK_DPD": "số ngày quá hạn (DPD)",
    "CNT_CHILDREN": "số con",
    "NAME_EDUCATION_TYPE": "trình độ học vấn",
    "NAME_FAMILY_STATUS": "tình trạng hôn nhân",
    "OWN_CAR_AGE": "tuổi xe sở hữu",
    "CODE_GENDER": "giới tính",
    "OCCUPATION_TYPE": "nghề nghiệp",
    "ORGANIZATION_TYPE": "loại hình tổ chức làm việc",
    "NAME_CONTRACT_TYPE": "loại hợp đồng vay",
}


def humanize(feature_name: str) -> tuple[str, bool]:
    """Câu giải thích tiếng Việt + cờ curated (True = đã curate tay, False = fallback)."""
    for prefix, table_label in TABLE_LABELS.items():
        if not feature_name.startswith(prefix + "_"):
            continue
        rest = feature_name[len(prefix) + 1:]
        for suffix, agg_label in AGG_LABELS.items():
            if rest.endswith("_" + suffix):
                raw = rest[: -(len(suffix) + 1)]
                raw_label = RAW_COLUMN_LABELS.get(raw)
                if raw_label:
                    return f"{agg_label} {raw_label} ({table_label})", True
                return f"{agg_label} {raw.replace('_', ' ').lower()} ({table_label})", False
        return f"{rest.replace('_', ' ').lower()} ({table_label})", False

    if feature_name in RAW_COLUMN_LABELS:
        return RAW_COLUMN_LABELS[feature_name], True
    return feature_name.replace("_", " ").title(), False


def direction_phrase(shap_value: float) -> str:
    return "làm TĂNG rủi ro" if shap_value > 0 else "làm GIẢM rủi ro"


def _to_json_safe(value):
    """Feature value có thể là số (numpy scalar), NaN, hoặc string (cột categorical
    gốc như CODE_GENDER/ORGANIZATION_TYPE — những cột này hoàn toàn có thể lọt top-K
    SHAP của 1 applicant cụ thể). KHÔNG được ép float() vô điều kiện — sẽ crash trên string.
    """
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def build_reason_codes(feature_names: list[str], shap_row, feature_values, top_k: int = 5) -> list[dict]:
    """Top-K feature ảnh hưởng mạnh nhất (|SHAP| lớn nhất) -> reason code có cấu trúc."""
    order = sorted(range(len(feature_names)), key=lambda i: -abs(shap_row[i]))[:top_k]
    reasons = []
    for i in order:
        label, curated = humanize(feature_names[i])
        reasons.append({
            "feature": feature_names[i],
            "label": label,
            "value": _to_json_safe(feature_values[i]),
            "shap": float(shap_row[i]),
            "direction": direction_phrase(shap_row[i]),
            "curated": curated,
        })
    return reasons
