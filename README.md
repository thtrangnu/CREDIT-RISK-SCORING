# Home Credit Default Risk — Credit Scoring end-to-end

Chấm điểm rủi ro vỡ nợ trên bộ [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
(Kaggle, 7 bảng quan hệ, 307.511 applicant), từ feature engineering đa bảng đến một API
chấm điểm có audit trail và một giao diện giải trình được từng quyết định.

> **Ở tỉ lệ duyệt 70%, tỉ lệ vỡ nợ trong nhóm được duyệt giảm từ 8.07% xuống 3.51%**
> — giảm **56.5%** tổn thất tín dụng, chặn được **69.5%** tổng số ca vỡ nợ.
> *(tính trên out-of-fold đã hiệu chỉnh, không phải in-sample)*

---

## Kết quả

| | Baseline | Sau feature engineering | |
|---|---|---|---|
| **AUC** (OOF) | 0.76076 | **0.78757** | **+0.02681** |
| **Gini** | 0.52152 | 0.57514 | +0.05362 |
| Số feature | 120 | 709 | 7 bảng |

Baseline và bản engineered dùng **cùng fold split** (cùng seed, cùng `n_folds`, cùng thứ tự
dòng), nên delta đo đúng phần lift đến từ feature engineering chứ không lẫn với may rủi của
lần chia fold khác.

### Calibration

Trong lending, thứ hạng đúng là chưa đủ — con số PD phải *đúng nghĩa xác suất* thì mới định
giá và đặt cutoff được.

| | Trước hiệu chỉnh | Sau isotonic |
|---|---|---|
| Brier | 0.06598 | 0.06589 |
| ECE (uniform-10 bin) | 0.00417 | **0.00060** |
| ECE (quantile-50 bin) | 0.00597 | 0.00201 |

Báo cáo ở nhiều cách chia bin có lý do: 77% prediction nằm dưới 0.1, nên uniform binning nhét
gần hết dữ liệu vào một bin và cho ECE đẹp giả tạo. Cải thiện **giữ nguyên độ lớn ở cả hai
cách chia** → không phải artifact của binning.

Con số "sau hiệu chỉnh" đo bằng **nested calibration**: chia OOF thành 5 phần, fit calibrator
trên 4 phần và dự đoán phần còn lại. Đo kiểu ngây thơ (fit rồi predict trên chính nó) sẽ ra
ECE thấp giả.

Platt/sigmoid làm ECE **xấu đi** — vì baseline cố tình không dùng `scale_pos_weight`, model
gốc đã khá calibrated sẵn, áp thêm một biến đổi sigmoid cứng lên nó là làm hỏng.

### Điểm số → quyết định

| Tỉ lệ duyệt | Ngưỡng PD | Bad rate nhóm duyệt | Giảm tổn thất | Ca vỡ nợ bị chặn |
|---|---|---|---|---|
| 50% | 0.0460 | 2.3% | −71.1% | 85.5% |
| **70%** | **0.0863** | **3.5%** | **−56.5%** | **69.5%** |
| 90% | 0.1887 | 5.6% | −30.8% | 37.8% |
| 100% | — | 8.1% | — | — |

### Tác động không đồng đều

Giới tính và tuổi là **thuộc tính được bảo vệ** theo ECOA. Ở ngưỡng duyệt 70%:

| Adverse impact ratio (4/5ths rule, ngưỡng 0.80) | |
|---|---|
| Theo giới tính | **0.816** — vừa qua ngưỡng |
| Theo nhóm tuổi | **0.446** — **không đạt** |

Model **calibrate rất đều** giữa các nhóm (lệch giữa PD trung bình và bad rate thật đều dưới
0.2pp, trừ nhóm dưới 25 tuổi lệch +1.9pp) — nó không thiên vị theo nghĩa dự đoán sai lệch có
hệ thống. Chênh lệch tỉ lệ duyệt phản ánh chênh lệch rủi ro có thật (bad rate nhóm <25 là
12.3%, nhóm 65+ là 3.7%).

Nhưng "chênh vì rủi ro chênh thật" **không phải biện hộ hợp lệ về pháp lý** — 4/5ths rule đo
*tác động*, không đo *ý định*. Chi tiết và hàm ý triển khai: [model card](ml/artifacts/model_card.md) mục 4.

---

## Kiến trúc

Ba tầng, biên giới cứng, mỗi tầng không biết gì về tầng sau nó:

```
ml/                     backend/                  frontend/
pipeline ML độc lập  →  FastAPI chỉ LOAD       →  React (Vite)
                        artifact, KHÔNG train
        └────────── ml/artifacts/ ──────────┘
              (biên giới duy nhất)
```

**Vì sao tách cứng:** backend serve 1 applicant phải chạy **đúng đường tính feature** như lúc
train. Nếu feature engineering nằm rải rác trong notebook thì không có cách nào đảm bảo điều
đó. Ở đây nó là một `FeaturePipeline` pickle được (`fit` trên train, `transform` dùng lại y
hệt lúc serve), nên training-serving skew bị chặn ở mức thiết kế chứ không phải mức kỷ luật.

`backend/` **không bao giờ** import `ml.src.train*` hay `ml.src.calibrate` (orchestration
training). Nó chỉ import `ml.src.features.build`, `ml.src.explain`, `ml.src.reason_codes` —
ba module thuần transform/hàm thuần, cần thiết để unpickle được artifact.

---

## Chạy thử

### 1. Chuẩn bị

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt   # đã include ml/requirements.txt
```

> `numpy<2.5` là pin **bắt buộc**, không phải cẩn thận thừa: `shap` phụ thuộc `numba`, và
> numba hiện chỉ hỗ trợ NumPy ≤ 2.4. Với numpy 2.5 thì `import shap` chết ngay.

Tải [dữ liệu Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data) và giải nén 7
file CSV vào `data/` (thư mục này được gitignore — ~2.5GB).

### 2. Chạy pipeline ML (theo đúng thứ tự)

```bash
python -m ml.src.run_build_features      # 7 bảng -> 709 feature + feature_pipeline.pkl
python -m ml.src.train                   # baseline OOF (Block 1)
python -m ml.src.train_engineered        # OOF engineered + monotonic (Block 2-3)
python -m ml.src.calibrate               # isotonic + model.txt (Block 4-5)
python -m ml.src.explain                 # SHAP global importance (Block 6)
python -m ml.src.export_metrics_summary  # metrics + cutoff policy + fairness (Block 9)
python -m ml.src.export_model_card       # model_card.md
```

> ⚠️ Đừng chạy `python -m ml.src.features.build`. Module đó **định nghĩa** `FeaturePipeline`;
> chạy trực tiếp sẽ nạp nó as `__main__`, ghi `__module__ == "__main__"` vào pickle, và
> `feature_pipeline.pkl` sẽ không unpickle được từ backend. Dùng `run_build_features` (chỉ
> import `main()`). Có test chặn regression này.

### 3. Backend + frontend

```bash
docker compose up -d mysql               # MySQL cho audit trail (port 3307)
cp .env.example .env                     # rồi sửa mật khẩu
alembic -c backend/alembic.ini upgrade head
uvicorn backend.app.main:app --reload    # startup load 7 bảng vào RAM, ~10-50s
```

```bash
cd frontend && npm install && npm run dev   # http://localhost:5173
```

### 4. Test

```bash
pytest ml/tests backend/tests -q
```

---

## Quyết định kỹ thuật đáng chú ý

**Metric tự viết tay.** `ml/src/metrics.py` không gọi `sklearn.metrics` cho các metric lõi:
AUC bằng Mann–Whitney rank O(n log n) với tie = rank trung bình, Gini, KS, PR-AUC step-sum,
Brier, ECE (uniform + quantile binning), partial AUC chuẩn hoá McClish, TPR@FPR, decile table
kiểu risk. Sklearn chỉ dùng trong test để assert kết quả khớp.

**Baseline sạch có chủ đích.** Không `scale_pos_weight`, không `is_unbalance`, không impute.
Đây không phải lười — reweighting làm hỏng calibration, mà calibration chính là thứ block sau
đo. Giữ baseline sạch để có mốc so sánh thật.

**Monotonic constraints có lý do domain, không phải rải đại.** 18 feature, mỗi cái kèm lý do
viết trong [`features.yaml`](ml/config/features.yaml). Cố tình **không** ràng buộc
`AMT_CREDIT`/`AMT_ANNUITY` vì quan hệ với rủi ro thực tế mơ hồ. Kiểm chứng chéo ở Block 6:
7/18 feature này lọt top-30 SHAP global — domain reasoning khớp thứ model thực sự học.

**Chống training-serving skew ở mức thiết kế.** Domain của mọi cột categorical được chốt lúc
`fit` và áp lại y hệt lúc `transform` — vì một applicant lẻ lúc serve gần như chắc chắn không
có đủ mọi category so với lúc train. `bureau_balance.STATUS` hardcode theo data dictionary
thay vì suy ra từ data. Có test chuyên bắt lớp bug này.

**2-level aggregation cho bureau.** `bureau_balance` chỉ có `SK_ID_BUREAU`, không có
`SK_ID_CURR` — phải agg theo tháng về từng khoản vay, merge vào `bureau`, rồi mới agg về
applicant. Backend lọc dữ liệu theo đúng chain đó khi serve.

**Feature derived phải tính ở mức dòng.** `DAYS_LATE` và `PAYMENT_DIFF` trong
`installments_payments` được tính trước khi agg — agg riêng từng cột gốc không tái tạo được
độ trễ/thiếu hụt của từng lần trả cụ thể. Cả hai đều lọt top-30 SHAP.

---

## Bug thật đã bắt được

Mấy cái này không xuất hiện trong test đơn giản, chỉ lộ ra khi chạy hệ thống thật:

1. **`Booster.predict()` khớp cột theo VỊ TRÍ, không theo tên.** Đảo thứ tự cột cho ra kết quả
   khác mà không báo lỗi. `model.feature_name()` cũng không đáng tin vì LightGBM tự sanitize
   tên cột có ký tự đặc biệt khi lưu `model.txt`. → mọi nơi dùng model đều reindex theo
   `feature_names.json` trước khi predict.
2. **Pickle ghi `__module__ = "__main__"`.** `feature_pipeline.pkl` build qua
   `python -m ml.src.features.build` không unpickle được từ bất kỳ entry point nào khác.
3. **`float()` vô điều kiện lên feature value.** Crash khi top-SHAP feature của một applicant
   là cột categorical (`CODE_GENDER`, `ORGANIZATION_TYPE`...).
4. **`json.dump` ghi `NaN` — không phải JSON hợp lệ.** `json.loads` của Python vẫn đọc được
   nên lỗi im lặng ở tầng ML, nhưng `JSON.parse` của trình duyệt thì ném lỗi. NaN xuất hiện
   thật: AUC của nhóm phân khúc chỉ có 1 lớp (`CODE_GENDER='XNA'`, 4 dòng).
5. **Label SHAP waterfall đè lên bar.** Label lấy nhầm cạnh bar (luôn lấy cạnh max thay vì
   cạnh đúng theo dấu); bar dài thì label tràn sang cột tên feature bên trái.

---

## Hạn chế đã biết

Ghi ra vì đây là những chỗ một reviewer sẽ hỏi, và câu trả lời "tôi biết" tốt hơn "tôi không
để ý":

1. **Early stopping dùng chính fold sinh OOF prediction.** Fold validation vừa để dừng sớm vừa
   để lấy `oof[va]` → **AUC 0.78757 lạc quan nhẹ**, không phải OOF hoàn toàn sạch. Delta so
   với baseline vẫn công bằng vì cả hai lệch như nhau. Sửa được bằng cách tách inner split.
2. **Calibrator fit trên OOF nhưng áp cho model train trên 100% data.** Chuẩn ngành (giống
   `CalibratedClassifierCV(ensemble=False)`) nhưng là một giả định, không hiển nhiên đúng.
3. **`CODE_GENDER` đang được dùng làm feature** và có thể lọt vào reason codes. Ở hệ thống
   thật là không được phép — giữ lại trong demo để bảng phân khúc có ý nghĩa đối chiếu.
4. **Chưa tune hyperparameter.** Tham số trong `params.yaml` chọn tay, chưa chạy Optuna.
5. **Không có out-of-time validation.** Dataset Kaggle không có trục thời gian rõ ràng, nên
   thiếu hẳn thứ bắt buộc phải có ở scorecard thật.
6. **"Tổn thất tương đối" không phải mô hình tổn thất thật** — giả định loss tỉ lệ với số ca
   vỡ nợ được duyệt và exposure mỗi khoản như nhau, thiếu LGD/EAD và giá trị khoản vay.

## Cố tình không làm

Đây là **portfolio project, phạm vi demo** — hình dáng như production ở những chỗ kể được
chuyện (tách tầng, chống skew, versioning, audit trail), nhưng chạy local, data tĩnh,
single-user. Những thứ dưới đây bị loại **có chủ đích**, không phải quên:

auth/JWT · monitoring/Prometheus · Kubernetes · CI/CD · drift detection · retraining tự động ·
LLM/chatbot · Playwright visual regression · Dockerfile cho backend/frontend

---

## Cấu trúc

```
ml/                              # TẦNG 1 — pipeline ML độc lập
├── config/{features,params}.yaml    # monotonic constraints + hyperparams
├── src/
│   ├── metrics.py                   # metric tự viết tay
│   ├── features/                    # agg tái dùng + FeaturePipeline
│   ├── train.py, train_engineered.py, calibrate.py
│   ├── explain.py, reason_codes.py  # SHAP + adverse action reason codes
│   ├── policy.py                    # cutoff policy + phân khúc/fairness
│   └── export_{metrics_summary,model_card}.py
├── artifacts/                   # ← BIÊN GIỚI sang backend
└── tests/
backend/                         # TẦNG 2 — FastAPI, chỉ load artifact
├── app/{scorer,schemas,config}.py
├── app/data/{source,assembler}.py   # kéo 7 bảng theo SK_ID_CURR
├── app/db/                          # MySQL — audit trail mỗi lần chấm
├── app/routers/{score,history,insights}.py
└── tests/
frontend/                        # TẦNG 3 — React, hướng "Risk Console"
└── src/{pages,components,styles}/
```

Chi tiết bối cảnh và các quyết định đã chốt: [ghi chú thiết kế](docs/NOTES.md).
Chi tiết model: [model card](ml/artifacts/model_card.md).
