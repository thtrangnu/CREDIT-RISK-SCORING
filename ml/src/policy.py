"""Block 9: dịch điểm số sang QUYẾT ĐỊNH — cutoff policy + đối chiếu phân khúc.

Vì sao cần block này: AUC/ECE trả lời "model xếp hạng tốt cỡ nào", KHÔNG trả
lời "dùng nó thì được gì". Trong lending, câu hỏi thật sự là: ở một tỉ lệ
duyệt cho trước, tỉ lệ vỡ nợ trong nhóm được duyệt giảm bao nhiêu — đó mới là
con số nói chuyện được với business.

Mọi con số ở đây tính trên OOF ĐÃ HIỆU CHỈNH (nested isotonic), không phải
in-sample: mỗi applicant được chấm bởi model KHÔNG nhìn thấy nó lúc train, và
được hiệu chỉnh bởi calibrator KHÔNG nhìn thấy label của nó (xem calibrate.py).

Phân khúc (`segment_report`) đứng ở đây chứ không ở metrics.py vì nó là câu hỏi
CHÍNH SÁCH (ai bị ảnh hưởng thế nào), không phải một metric thống kê thuần.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import roc_auc

# Tỉ lệ duyệt để quét. Không phải khuyến nghị — là dải để đọc trade-off.
DEFAULT_APPROVAL_RATES = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

# Ngưỡng tham chiếu dùng cho báo cáo phân khúc. 0.7 = duyệt 70% hồ sơ tốt nhất.
REFERENCE_APPROVAL_RATE = 0.7

# Ngưỡng "4/5ths rule" (EEOC): tỉ lệ duyệt của nhóm bất lợi nhất / nhóm thuận
# lợi nhất < 0.8 là dấu hiệu adverse impact cần điều tra.
ADVERSE_IMPACT_THRESHOLD = 0.8


def approve_mask(pd_score: np.ndarray, approval_rate: float) -> np.ndarray:
    """Duyệt `approval_rate` phần hồ sơ có PD THẤP nhất.

    Cắt theo phân vị thay vì theo một ngưỡng PD tuyệt đối: giữ tỉ lệ duyệt cố
    định là cách so sánh công bằng giữa các model/phiên bản, và cũng gần với
    cách bộ phận risk vận hành thật (định mức volume trước, suy ra cutoff sau).
    """
    if not 0.0 < approval_rate <= 1.0:
        raise ValueError(f"approval_rate phải trong (0, 1], nhận {approval_rate}")
    if approval_rate == 1.0:
        return np.ones(len(pd_score), dtype=bool)
    cutoff = float(np.quantile(pd_score, approval_rate))
    order = np.argsort(pd_score, kind="mergesort")
    k = int(round(len(pd_score) * approval_rate))
    mask = np.zeros(len(pd_score), dtype=bool)
    mask[order[:k]] = True
    return mask


def cutoff_table(
    y_true, pd_score, approval_rates: tuple[float, ...] = DEFAULT_APPROVAL_RATES
) -> pd.DataFrame:
    """Bảng trade-off: duyệt bao nhiêu <-> vỡ nợ bao nhiêu <-> chặn được bao nhiêu bad."""
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
            # Giảm bao nhiêu % tỉ lệ vỡ nợ so với "duyệt tất" — con số kể chuyện.
            "bad_rate_reduction": 1.0 - bad_rate_approved / base_rate,
            # % tổng số ca vỡ nợ bị chặn lại ở cửa.
            "bad_captured": (total_bad - bad_approved) / total_bad if total_bad else float("nan"),
            # Tổn thất kỳ vọng tương đối (giả định loss ~ số ca vỡ nợ được duyệt,
            # cùng exposure mỗi khoản). Chuẩn hoá về 1.0 = duyệt tất.
            "expected_loss_index": bad_approved / total_bad if total_bad else float("nan"),
        })
    return pd.DataFrame(rows)


def segment_report(
    y_true, pd_score, groups, approval_rate: float = REFERENCE_APPROVAL_RATE
) -> pd.DataFrame:
    """Chất lượng model + tác động chính sách theo từng phân khúc.

    Cột đọc thế nào:
      bad_rate / mean_pd  lệch nhau nhiều  -> model calibrate lệch cho nhóm đó
      auc                 thấp hơn hẳn     -> model phân biệt kém trên nhóm đó
      approval_rate       chênh nhau nhiều -> tác động chính sách không đồng đều
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(pd_score, dtype=float)
    approved = approve_mask(p, approval_rate)

    df = pd.DataFrame({"y": y, "p": p, "approved": approved, "group": np.asarray(groups)})
    rows = []
    for name, sub in df.groupby("group", dropna=False):
        # AUC vô nghĩa nếu nhóm chỉ có 1 lớp; báo NaN thay vì con số giả.
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
    """min(approval_rate) / max(approval_rate) theo "4/5ths rule" của EEOC.

    Bỏ qua nhóm quá nhỏ (mặc định <1000) — vài chục dòng không đủ để kết luận
    gì, mà lại kéo tỉ số xuống bằng nhiễu (vd CODE_GENDER='XNA' chỉ có 4 dòng).
    """
    big = segments[segments["n"] >= min_group_size]
    if len(big) < 2:
        return float("nan")
    return float(big["approval_rate"].min() / big["approval_rate"].max())


def age_bands(days_birth) -> np.ndarray:
    """DAYS_BIRTH (âm, tính từ ngày nộp đơn) -> nhãn nhóm tuổi.

    Tuổi cũng là thuộc tính được bảo vệ trong tín dụng (ECOA), nên nó cần có
    trong báo cáo phân khúc chứ không chỉ giới tính.
    """
    years = -np.asarray(days_birth, dtype=float) / 365.25
    bins = [0, 25, 35, 45, 55, 65, 200]
    labels = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]
    return pd.cut(years, bins=bins, labels=labels, right=False).astype(str)
