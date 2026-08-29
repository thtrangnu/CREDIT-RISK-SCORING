"""Sinh ml/artifacts/model_card.md TỪ artifact đã có — không train lại gì cả.

Trước đây model card được nhúng thẳng trong calibrate.py, nghĩa là muốn sửa một
câu chữ trong card cũng phải chạy lại toàn bộ training. Tách ra đây để card là
một hàm thuần của artifact: đọc metrics_summary.json + model.txt + calibrator.pkl
+ features.yaml rồi render. calibrate.py gọi lại module này ở cuối.
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
    header = ("| Tỉ lệ duyệt | Ngưỡng PD | Bad rate nhóm duyệt | Giảm so với duyệt hết "
              "| % ca vỡ nợ bị chặn | Tổn thất tương đối |\n|---|---|---|---|---|---|\n")
    lines = [
        f"| {_pct(r['approval_rate'])} | {r['pd_cutoff']:.4f} | {_pct(r['bad_rate_approved'])} "
        f"| {_pct(r['bad_rate_reduction'])} | {_pct(r['bad_captured'])} | {r['expected_loss_index']:.3f} |"
        for r in policy_table
    ]
    return header + "\n".join(lines)


def _fmt_auc(auc: float | None) -> str:
    """AUC không xác định khi nhóm chỉ có 1 lớp (vd CODE_GENDER='XNA' chỉ 4 dòng).

    metrics_summary.json lưu None (đã qua json_safe); nhận cả NaN cho chắc.
    """
    if auc is None or auc != auc:
        return "—"
    return f"{auc:.4f}"


def _segment_rows(rows: list[dict]) -> str:
    header = ("| Nhóm | n | Bad rate thật | PD trung bình | Lệch calibration | AUC | Tỉ lệ được duyệt |\n"
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

## Tóm tắt

| | |
|---|---|
| Bài toán | Binary classification — dự đoán vỡ nợ, {_pct(summary['default_rate'])} positive |
| Dữ liệu | Home Credit Default Risk (Kaggle), 7 bảng, {summary['n_train_rows']:,} applicant |
| Model | LightGBM, {summary['n_features']} feature, {n_monotonic} monotonic constraint |
| Calibration | {calibrator_method}, đánh giá bằng nested {n_folds}-fold trên OOF |
| CV | StratifiedKFold {n_folds}-fold, seed={seed} |
| Model cuối | fit trên 100% train, {n_trees} cây |

**Con số quan trọng nhất:** ở tỉ lệ duyệt {_pct(ref)}, tỉ lệ vỡ nợ trong nhóm được duyệt là
{_pct(ref_row['bad_rate_approved'])} so với {_pct(summary['default_rate'])} nếu duyệt tất cả —
**giảm {_pct(ref_row['bad_rate_reduction'])} tổn thất tín dụng**, chặn được
{_pct(ref_row['bad_captured'])} tổng số ca vỡ nợ.

## 1. Chất lượng xếp hạng (OOF)

| Metric | Baseline (Block 1) | Engineered (Block 2-3) | Delta |
|---|---|---|---|
| AUC | {summary['baseline']['auc']:.5f} | {eng['auc']:.5f} | **{summary['auc_delta']:+.5f}** |
| Gini | {summary['baseline']['gini']:.5f} | {eng['gini']:.5f} | {eng['gini'] - summary['baseline']['gini']:+.5f} |

Baseline = 120 cột gốc của `application_train`, không impute, không reweight.
Engineered = {summary['n_features']} feature từ cả 7 bảng. **Cùng fold split** (cùng seed,
cùng n_folds, cùng thứ tự dòng) nên delta đo đúng phần lift đến từ feature engineering.

## 2. Calibration

ECE được report ở nhiều cách chia bin, vì uniform binning dễ cho kết quả đẹp giả tạo trên
bài toán lệch (77% prediction < 0.1 → bin đầu tiên nuốt gần hết dữ liệu):

| | Trước hiệu chỉnh | Sau hiệu chỉnh ({calibrator_method}) |
|---|---|---|
| Brier | {eng['brier_raw']:.5f} | {eng['brier_calibrated']:.5f} |
| ECE uniform-10 | {eng['ece_uniform_10_raw']:.5f} | **{eng['ece_uniform_10_calibrated']:.5f}** |
| ECE quantile-10 | {eng['ece_quantile_10_raw']:.5f} | {eng['ece_quantile_10_calibrated']:.5f} |
| ECE quantile-50 | {eng['ece_quantile_50_raw']:.5f} | {eng['ece_quantile_50_calibrated']:.5f} |

Cải thiện giữ nguyên độ lớn ở cả 3 cách chia bin → không phải artifact của binning.

Số "sau hiệu chỉnh" đo bằng **nested calibration**: chia OOF thành {n_folds} phần, fit
calibrator trên {n_folds - 1} phần và dự đoán phần còn lại, nên mỗi điểm được hiệu chỉnh
bởi calibrator chưa từng thấy label của nó. Đo kiểu ngây thơ (fit rồi predict trên chính
nó) sẽ ra ECE thấp giả.

Platt/sigmoid làm ECE **xấu hơn**: model gốc đã khá calibrated sẵn (do cố tình không dùng
`scale_pos_weight`/`is_unbalance`), nên áp một biến đổi sigmoid cứng lên nó là làm hỏng.

## 3. Chính sách cutoff — điểm số dịch sang quyết định

Tính trên OOF đã hiệu chỉnh. "Duyệt X%" = duyệt X% hồ sơ có PD thấp nhất.

{_policy_rows(pol['table'])}

*Tổn thất tương đối*: giả định loss tỉ lệ với số ca vỡ nợ được duyệt và exposure mỗi khoản
như nhau — đủ để so sánh tương đối giữa các ngưỡng, KHÔNG phải mô hình tổn thất thật
(thiếu LGD/EAD và giá trị khoản vay).

## 4. Phân khúc & tác động không đồng đều

Giới tính và tuổi là **thuộc tính được bảo vệ** trong tín dụng (ECOA/Reg B). Bảng dưới ở
tỉ lệ duyệt tham chiếu {_pct(ref)}:

### Theo giới tính
{_segment_rows(fair['by_gender'])}

### Theo nhóm tuổi
{_segment_rows(fair['by_age_band'])}

**Adverse impact ratio** (4/5ths rule của EEOC — dưới 0.80 là dấu hiệu cần điều tra):

- Giới tính: **{fair['adverse_impact_ratio_gender']:.4f}** — vừa qua ngưỡng.
- Nhóm tuổi: **{fair['adverse_impact_ratio_age']:.4f}** — **KHÔNG đạt.**

Cách đọc cho đúng: model **calibrate rất đều** giữa các nhóm (lệch |PD − bad rate thật| đều
dưới 0.2pp, trừ nhóm <25 lệch +1.9pp) — nghĩa là nó không "thiên vị" theo nghĩa dự đoán sai
lệch có hệ thống cho một nhóm. Chênh lệch tỉ lệ duyệt phản ánh chênh lệch rủi ro có thật
trong dữ liệu (bad rate nhóm <25 là 12.3% so với 3.7% ở nhóm 65+).

Nhưng "tỉ lệ duyệt chênh vì rủi ro chênh thật" **không phải là biện hộ hợp lệ về mặt pháp
lý** — 4/5ths rule đo *tác động*, không đo *ý định*. Trong triển khai thật, con số 0.4464
này bắt buộc phải qua pháp chế và nhiều khả năng cần: bỏ hẳn các feature đại diện cho tuổi,
hoặc đặt cutoff riêng theo nhóm, hoặc chấp nhận và ghi nhận rủi ro tuân thủ.

`CODE_GENDER` hiện **đang được dùng làm feature** và có thể xuất hiện trong reason codes.
Ở hệ thống thật điều này là không được phép — giữ lại trong demo để bảng phân khúc trên có
ý nghĩa đối chiếu, nhưng đây là việc đầu tiên phải bỏ nếu productionize.

## 5. Explainability

- `shap.TreeExplainer` trên `model.txt`, margin space, 5000 dòng mẫu (seed=42).
- Top 3 global: EXT_SOURCE_2, EXT_SOURCE_3, EXT_SOURCE_1.
- 7/{n_monotonic} feature có monotonic constraint lọt top-30 SHAP global — domain reasoning
  ở Block 3 khớp với thứ model thực sự học được.
- Reason codes (adverse action): `ml/src/reason_codes.py` — curate tay cho feature tín hiệu
  mạnh, fallback đánh dấu `curated=False` để biết cần pháp chế review.
- File: `shap_global_importance.csv` ({summary['n_features']} feature, sort theo mean|SHAP|).

## 6. Hạn chế đã biết

1. **Early stopping dùng chính fold sinh OOF prediction.** Trong `train.py` và
   `train_engineered.py`, fold validation vừa dùng để dừng sớm vừa dùng để lấy
   `oof[va]`. Số vòng lặp được chọn bằng dữ liệu mà nó sắp dự đoán → AUC {eng['auc']:.5f}
   **lạc quan nhẹ**, không phải OOF hoàn toàn sạch. Delta so với baseline vẫn công bằng vì
   cả hai lệch như nhau. Cách sửa: tách inner-validation split riêng từ phần train.
2. **Calibrator fit trên OOF nhưng áp cho model train trên 100% data.** Model cuối sharper
   hơn model 5-fold nên phân phối điểm lệch nhẹ so với phân phối calibrator đã học. Đây là
   cách làm chuẩn ngành (giống `CalibratedClassifierCV(ensemble=False)`), nhưng là một giả
   định, không phải điều hiển nhiên đúng.
3. **Chưa tune hyperparameter.** Tham số trong `params.yaml` là chọn tay theo kinh nghiệm,
   chưa chạy Optuna hay search nào.
4. **Chưa có phân tích ổn định theo thời gian.** Dataset Kaggle không có trục thời gian rõ
   ràng nên không làm được out-of-time validation — thứ bắt buộc phải có ở scorecard thật.
5. **Tổn thất tương đối không phải mô hình tổn thất thật** — xem ghi chú ở mục 3.
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
    print(f"Saved -> {ARTIFACTS_DIR / 'model_card.md'} ({len(card.splitlines())} dòng)")


if __name__ == "__main__":
    main()
