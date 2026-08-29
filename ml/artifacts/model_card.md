# Model Card — Home Credit Default Risk

## Tóm tắt

| | |
|---|---|
| Bài toán | Binary classification — dự đoán vỡ nợ, 8.1% positive |
| Dữ liệu | Home Credit Default Risk (Kaggle), 7 bảng, 307,511 applicant |
| Model | LightGBM, 709 feature, 18 monotonic constraint |
| Calibration | isotonic, đánh giá bằng nested 5-fold trên OOF |
| CV | StratifiedKFold 5-fold, seed=42 |
| Model cuối | fit trên 100% train, 1400 cây |

**Con số quan trọng nhất:** ở tỉ lệ duyệt 70.0%, tỉ lệ vỡ nợ trong nhóm được duyệt là
3.5% so với 8.1% nếu duyệt tất cả —
**giảm 56.5% tổn thất tín dụng**, chặn được
69.5% tổng số ca vỡ nợ.

## 1. Chất lượng xếp hạng (OOF)

| Metric | Baseline (Block 1) | Engineered (Block 2-3) | Delta |
|---|---|---|---|
| AUC | 0.76076 | 0.78757 | **+0.02681** |
| Gini | 0.52152 | 0.57514 | +0.05362 |

Baseline = 120 cột gốc của `application_train`, không impute, không reweight.
Engineered = 709 feature từ cả 7 bảng. **Cùng fold split** (cùng seed,
cùng n_folds, cùng thứ tự dòng) nên delta đo đúng phần lift đến từ feature engineering.

## 2. Calibration

ECE được report ở nhiều cách chia bin, vì uniform binning dễ cho kết quả đẹp giả tạo trên
bài toán lệch (77% prediction < 0.1 → bin đầu tiên nuốt gần hết dữ liệu):

| | Trước hiệu chỉnh | Sau hiệu chỉnh (isotonic) |
|---|---|---|
| Brier | 0.06598 | 0.06589 |
| ECE uniform-10 | 0.00417 | **0.00060** |
| ECE quantile-10 | 0.00372 | 0.00046 |
| ECE quantile-50 | 0.00597 | 0.00201 |

Cải thiện giữ nguyên độ lớn ở cả 3 cách chia bin → không phải artifact của binning.

Số "sau hiệu chỉnh" đo bằng **nested calibration**: chia OOF thành 5 phần, fit
calibrator trên 4 phần và dự đoán phần còn lại, nên mỗi điểm được hiệu chỉnh
bởi calibrator chưa từng thấy label của nó. Đo kiểu ngây thơ (fit rồi predict trên chính
nó) sẽ ra ECE thấp giả.

Platt/sigmoid làm ECE **xấu hơn**: model gốc đã khá calibrated sẵn (do cố tình không dùng
`scale_pos_weight`/`is_unbalance`), nên áp một biến đổi sigmoid cứng lên nó là làm hỏng.

## 3. Chính sách cutoff — điểm số dịch sang quyết định

Tính trên OOF đã hiệu chỉnh. "Duyệt X%" = duyệt X% hồ sơ có PD thấp nhất.

| Tỉ lệ duyệt | Ngưỡng PD | Bad rate nhóm duyệt | Giảm so với duyệt hết | % ca vỡ nợ bị chặn | Tổn thất tương đối |
|---|---|---|---|---|---|
| 50.0% | 0.0460 | 2.3% | 71.1% | 85.5% | 0.145 |
| 60.0% | 0.0604 | 2.8% | 64.8% | 78.9% | 0.211 |
| 70.0% | 0.0863 | 3.5% | 56.5% | 69.5% | 0.305 |
| 80.0% | 0.1261 | 4.4% | 45.7% | 56.6% | 0.434 |
| 90.0% | 0.1887 | 5.6% | 30.8% | 37.8% | 0.622 |
| 100.0% | 1.0000 | 8.1% | 0.0% | 0.0% | 1.000 |

*Tổn thất tương đối*: giả định loss tỉ lệ với số ca vỡ nợ được duyệt và exposure mỗi khoản
như nhau — đủ để so sánh tương đối giữa các ngưỡng, KHÔNG phải mô hình tổn thất thật
(thiếu LGD/EAD và giá trị khoản vay).

## 4. Phân khúc & tác động không đồng đều

Giới tính và tuổi là **thuộc tính được bảo vệ** trong tín dụng (ECOA/Reg B). Bảng dưới ở
tỉ lệ duyệt tham chiếu 70.0%:

### Theo giới tính
| Nhóm | n | Bad rate thật | PD trung bình | Lệch calibration | AUC | Tỉ lệ được duyệt |
|---|---|---|---|---|---|---|
| F | 202,448 | 7.0% | 7.1% | +0.0008 | 0.7858 | 74.7% |
| M | 105,059 | 10.1% | 10.0% | -0.0014 | 0.7778 | 61.0% |
| XNA | 4 | 0.0% | 10.0% | +0.1001 | — | 25.0% |

### Theo nhóm tuổi
| Nhóm | n | Bad rate thật | PD trung bình | Lệch calibration | AUC | Tỉ lệ được duyệt |
|---|---|---|---|---|---|---|
| 35-44 | 84,261 | 8.4% | 8.1% | -0.0032 | 0.7888 | 69.9% |
| 25-34 | 72,429 | 10.7% | 10.7% | +0.0000 | 0.7780 | 58.0% |
| 45-54 | 70,190 | 7.0% | 7.0% | -0.0003 | 0.7857 | 74.8% |
| 55-64 | 60,522 | 5.4% | 5.5% | +0.0011 | 0.7585 | 82.0% |
| <25 | 12,233 | 12.3% | 14.2% | +0.0192 | 0.7432 | 41.0% |
| 65+ | 7,876 | 3.7% | 3.6% | -0.0010 | 0.7722 | 91.9% |

**Adverse impact ratio** (4/5ths rule của EEOC — dưới 0.80 là dấu hiệu cần điều tra):

- Giới tính: **0.8162** — vừa qua ngưỡng.
- Nhóm tuổi: **0.4464** — **KHÔNG đạt.**

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
- 7/18 feature có monotonic constraint lọt top-30 SHAP global — domain reasoning
  ở Block 3 khớp với thứ model thực sự học được.
- Reason codes (adverse action): `ml/src/reason_codes.py` — curate tay cho feature tín hiệu
  mạnh, fallback đánh dấu `curated=False` để biết cần pháp chế review.
- File: `shap_global_importance.csv` (709 feature, sort theo mean|SHAP|).

## 6. Hạn chế đã biết

1. **Early stopping dùng chính fold sinh OOF prediction.** Trong `train.py` và
   `train_engineered.py`, fold validation vừa dùng để dừng sớm vừa dùng để lấy
   `oof[va]`. Số vòng lặp được chọn bằng dữ liệu mà nó sắp dự đoán → AUC 0.78757
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
