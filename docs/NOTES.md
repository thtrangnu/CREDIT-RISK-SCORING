# Ghi chú thiết kế — Home Credit Default Risk Scoring

> Bối cảnh project, các quyết định đã chốt và trạng thái từng phần.
> Đọc file này trước khi sửa code để khỏi đề xuất ngược lại thứ đã cân nhắc rồi.

---

## 1. Định vị project (QUAN TRỌNG — đọc trước)

Đây là **portfolio project cho vị trí Data Scientist** (KHÔNG phải ML/AI Engineer,
KHÔNG phải LLM/GenAI). Mọi quyết định ưu tiên theo góc nhìn "một DS được chấm thế nào":
rigor thống kê, modeling, calibration, explainability, insight, và **con số impact đo được**.

Mục tiêu: một dòng mạnh trên CV + học được thứ thật. KHÔNG over-engineer.

Định hướng kỹ thuật: **"production-shaped demo"** — hình dáng như production ở chỗ
kể được chuyện (tách tầng sạch, chống skew, versioning, audit trail), nhưng phạm vi
là demo (chạy local/Docker, data Kaggle tĩnh, single-user, không scale).

**KHÔNG làm** (kể cả khi nghe hợp lý): auth/JWT, monitoring/Prometheus, k8s, CI/CD,
drift detection, retraining tự động, LLM/chatbot. Những thứ này để "Future work" trong README.

---

## 2. Bài toán

- Dataset: **Home Credit Default Risk** (Kaggle), 7 bảng quan hệ, ~307K applicant ở
  `application_train`.
- Bài toán: binary classification, dự đoán default. **Imbalanced** (~8% positive).
- Model chính: **LightGBM** + **SHAP** cho explainability.
- Domain: lending (regulated) → explainability không phải trang trí mà là yêu cầu
  pháp lý (adverse action reason codes).

### 7 bảng (grain + khóa)
| Bảng | Grain | Link |
|---|---|---|
| `application_{train,test}` | 1 dòng / applicant, chứa TARGET | SK_ID_CURR (PK) |
| `bureau` | nhiều dòng / applicant (tín dụng ở TCTD khác) | SK_ID_CURR; đẻ SK_ID_BUREAU |
| `bureau_balance` | snapshot tháng của mỗi khoản bureau | CHỈ có SK_ID_BUREAU → cần 2-level agg |
| `previous_application` | đơn vay HC trước đó | SK_ID_CURR; đẻ SK_ID_PREV |
| `POS_CASH_balance` | lịch sử POS/cash theo tháng | SK_ID_PREV (+ SK_ID_CURR) |
| `installments_payments` | từng lần trả (grain mịn nhất) | SK_ID_PREV (+ SK_ID_CURR) |
| `credit_card_balance` | sao kê thẻ theo tháng | SK_ID_PREV (+ SK_ID_CURR) |

Lưu ý: mọi cột `DAYS_*` là offset ÂM tính từ ngày nộp đơn.

---

## 3. Roadmap theo block + trạng thái

- [x] **Block 0** — EDA, framing. Hiểu imbalanced, tại sao không dùng accuracy.
- [x] **Block 1** — Baseline + metrics. ĐÃ CODE (xem mục 5). AUC baseline (chạy lại local) = 0.76076.
- [x] **Block 2** — Multi-table feature engineering. 709 features (từ 120). AUC OOF = 0.78757
  (delta = **+0.02681** so với baseline, cùng fold split để so sánh công bằng).
- [x] **Block 3** — Monotonic constraints trong `features.yaml` (18 feature có lý do domain rõ:
  EXT_SOURCE_*, DAYS_BIRTH/EMPLOYED, region rating, overdue/DPD bureau+installments+POS).
- [x] **Block 4–5** — Recalibration. Isotonic thắng Platt/sigmoid (đánh giá bằng nested 5-fold
  trên OOF, không lạc quan): ECE 0.00417 → **0.00060**, Brier 0.06598 → 0.06589.
- [x] **Block 6** — SHAP sâu → reason codes. Top global: EXT_SOURCE_2/3/1. 7/18 monotonic
  feature lọt top-30 SHAP (validate domain reasoning ở Block 3 khớp model học được).
- [x] **Block 7** — FastAPI backend + MySQL (docker-compose), verify chạy thật end-to-end
  (không chỉ unit test) — load 7 bảng thật, chấm applicant thật, ghi audit trail MySQL.
- [x] **Block 8** — React frontend, 3 trang (Score/Insights/History), verify qua browser
  thật (cả dark + light theme, cả mobile). Hướng thiết kế: "Risk Console" — Swiss grid +
  dark-luxury base + Fraunces/JetBrains Mono pairing + bento composition.
- [x] **Block 9** — Cutoff policy + fairness. Dịch điểm số sang quyết định: ở ngưỡng duyệt
  70%, bad rate nhóm được duyệt 8.07% → 3.51% (**giảm 56.5%** tổn thất, chặn 69.5% ca vỡ nợ).
  Phân khúc theo thuộc tính được bảo vệ (ECOA): adverse impact ratio giới tính 0.816 (đạt
  4/5ths), nhóm tuổi **0.446 (KHÔNG đạt)** — model calibrate đều giữa các nhóm nhưng tác động
  chính sách thì không. Surface lên model card + trang Insights.
- [x] **Block 10** — README + reproducibility. README đầy đủ (kết quả, kiến trúc, cách chạy,
  bug thật đã bắt, hạn chế đã biết). `ml/requirements.txt` + `backend/requirements.txt` pin
  version — `numpy<2.5` BẮT BUỘC (numba của shap chưa hỗ trợ numpy 2.5).
- [ ] **Future work** — auth, monitoring, CI/CD, drift detection, retraining tự động
  (cố tình KHÔNG làm — xem mục 1).

---

## 4. Quyết định thiết kế đã chốt (đừng đề xuất ngược lại)

1. **Baseline sạch có chủ đích**: KHÔNG dùng `scale_pos_weight` / `is_unbalance`,
   KHÔNG impute ở Block 1 — để giữ calibration sạch làm mốc đo. Reweighting/tuning
   để dành block sau. Đừng "sửa" baseline bằng cách thêm mấy cái này.
   (Cập nhật: hyperparameter tuning CHƯA làm và hiện không nằm trong scope — đã ghi vào
   "Hạn chế đã biết" ở README + model card thay vì để treo như một lời hứa.)
2. **CV scheme mặc định**: StratifiedKFold OOF. Tin CV hơn public LB.
3. **Metrics tự viết tay** trong `metrics.py`, KHÔNG gọi `sklearn.metrics` cho các
   metric lõi (đây là điểm nhấn CV — chứng minh hiểu metric). Có thể dùng sklearn để
   ASSERT test đúng, nhưng bản chạy chính là tự viết.
3. **Feature engineering phải là pipeline TÁI SỬ DỤNG** (không phải code rời trong
   notebook) — vì backend serve 1 applicant phải chạy ĐÚNG đường tính feature đó.
   → xuất ra `feature_pipeline.pkl` + `feature_names.json` (chống training-serving skew).
4. **Tách tầng cứng**: `ml/` độc lập, không biết gì về web. `backend/` chỉ LOAD
   artifact, KHÔNG train, KHÔNG `import` từ `ml/src/train.py`. Biên giới duy nhất giữa
   2 tầng là thư mục `ml/artifacts/`.
   → **Làm rõ khi code Block 7** (đã áp dụng trong `backend/app/scorer.py`): "KHÔNG train"
   nghĩa là backend không bao giờ import `ml.src.train`, `ml.src.train_engineered`,
   `ml.src.calibrate` (orchestration training thật). Backend ĐƯỢC PHÉP import
   `ml.src.features.build.FeaturePipeline`, `ml.src.explain` (SHAP inference), và
   `ml.src.reason_codes` — 3 module này là "feature/explain CONTRACT" thuần transform/hàm
   thuần, không train, cần thiết để unpickle `feature_pipeline.pkl` (pickle yêu cầu class
   definition import được ở nơi unpickle).
5. **DB = MySQL**, lưu audit trail (mỗi lần chấm → 1 dòng). Có `model_version` để
   governance. `pd_score` dùng DECIMAL không FLOAT.
6. Input serving: **chọn applicant có sẵn theo SK_ID_CURR** (backend tự kéo 7 bảng),
   KHÔNG bắt user điền tay 200 field. Upload file = optional sau.

---

## 5. Cấu trúc thư mục mục tiêu

```
home-credit-scoring/
├── ml/                          # TẦNG 1 — pipeline ML độc lập
│   ├── data/{raw,processed}/    # gitignore
│   ├── config/
│   │   ├── features.yaml        # khai báo feature + monotonic constraints
│   │   └── params.yaml          # hyperparams, seed, n_folds
│   ├── src/
│   │   ├── metrics.py           # [DONE] metric tự viết
│   │   ├── features/
│   │   │   ├── aggregations.py         # [DONE] hàm agg tái dùng + infer_categories (chống skew)
│   │   │   ├── bureau.py               # [DONE] 2-level agg (bureau_balance→bureau→curr)
│   │   │   ├── previous_application.py # [DONE]
│   │   │   ├── pos_cash.py             # [DONE]
│   │   │   ├── installments.py         # [DONE] + DAYS_LATE/PAYMENT_DIFF derived
│   │   │   ├── credit_card.py          # [DONE]
│   │   │   └── build.py                # [DONE] orchestrator + FeaturePipeline (fit/transform)
│   │   ├── train.py             # [DONE] OOF StratifiedKFold baseline
│   │   ├── train_engineered.py  # [DONE] Block 2+3: OOF trên engineered features + monotonic
│   │   ├── calibrate.py         # [DONE] Block 4-5: isotonic/Platt + final model.txt/calibrator.pkl
│   │   ├── explain.py           # [DONE] Block 6: SHAP global + local (explain_applicant)
│   │   ├── reason_codes.py      # [DONE] feature -> câu tiếng Việt (curated + fallback)
│   │   ├── policy.py            # [DONE] Block 9: cutoff policy + segment/fairness report
│   │   ├── export_metrics_summary.py  # [DONE] metrics + policy + fairness -> metrics_summary.json
│   │   ├── export_model_card.py # [DONE] model_card.md từ artifact (KHÔNG train lại)
│   │   └── run_build_features.py      # [DONE] entry point AN TOÀN cho build.py (xem ghi chú
│   │                                     trong features/build.py — KHÔNG chạy build.py bằng -m)
│   ├── artifacts/               # OUTPUT training = biên giới sang backend — TẤT CẢ [DONE]
│   │   ├── model.txt
│   │   ├── calibrator.pkl
│   │   ├── feature_pipeline.pkl
│   │   ├── feature_names.json
│   │   ├── shap_global_importance.csv
│   │   ├── metrics_summary.json
│   │   └── model_card.md
│   ├── notebooks/
│   ├── requirements.txt         # [DONE] pin version — numpy<2.5 BẮT BUỘC (numba/shap)
│   └── tests/                   # 55 test (Block 2-9)  [51 chạy được không cần shap]
├── backend/                     # TẦNG 2 — FastAPI, chỉ load artifact — [DONE]
│   ├── app/
│   │   ├── main.py              # lifespan: load RawTableStore + Scorer 1 lần lúc startup
│   │   ├── config.py            # đọc .env: DATABASE_URL, MODEL_VERSION, CORS_ORIGINS
│   │   ├── schemas.py           # Pydantic (ScoreResponse có base_value/raw_margin cho waterfall)
│   │   ├── scorer.py            # Scorer: pipeline.transform → predict → calibrate → SHAP
│   │   ├── reason_codes.py      # re-export ml.src.reason_codes (xem ghi chú biên giới tầng)
│   │   ├── dependencies.py      # get_store/get_scorer (FastAPI Depends, dễ override lúc test)
│   │   ├── data/{source.py,assembler.py}   # kéo 7 bảng theo sk_id_curr (applicant_train pool)
│   │   ├── db/{session.py,models.py,crud.py}  # SQLAlchemy + PyMySQL, ScoringHistory
│   │   └── routers/{score.py,history.py,insights.py}
│   ├── migrations/              # Alembic — 0001_create_scoring_history
│   └── tests/                   # 17 test (dùng data thật lọc theo 2 SK_ID_CURR, không mock)
├── frontend/                    # TẦNG 3 — React (Vite, JS/JSX) — [DONE]
│   └── src/
│       ├── pages/{ScorePage,InsightsPage,HistoryPage}.jsx
│       ├── components/
│       │   ├── cutoff-table/CutoffTable.jsx               # Block 9: bảng trade-off duyệt/rủi ro
│       │   ├── segment-table/SegmentTable.jsx             # Block 9: phân khúc + badge AIR
│       │   ├── applicant-selector/ApplicantSelector.jsx   # search-as-you-type
│       │   ├── score-gauge/ScoreGauge.jsx                 # radial gauge, animated, scale 5x base_rate
│       │   ├── reason-code-list/ReasonCodeList.jsx
│       │   ├── shap-waterfall/ShapWaterfall.jsx            # SVG tay, base→feature→"còn lại"→kết quả
│       │   ├── history-table/HistoryTable.jsx
│       │   ├── nav/NavBar.jsx                              # theme toggle dark/light
│       │   └── ui/{RiskBadge,StatTile}.jsx
│       ├── context/ScoreContext.jsx   # share applicant/scoreResult giữa Score ↔ Insights
│       ├── styles/{tokens.css,typography.css,global.css}   # design system: xem mục 8
│       └── api/client.js
├── docker-compose.yml           # mysql (backend/frontend chạy dev server trực tiếp, xem mục 8)
└── README.md
```

---

## 6. Trạng thái code hiện tại

### `ml/src/metrics.py` — ĐÃ CODE
Tự viết tay, 3 cụm:
- Ranking: AUC (Mann–Whitney rank, O(n log n), tie = rank trung bình), Gini (=2·AUC−1), KS (=max|TPR−FPR|).
- Calibration/prob: PR-AUC (step-sum / average precision), Brier (+ Murphy decomposition), ECE.
- Operational credit: partial AUC (chuẩn hóa McClish), TPR@FPR, decile table.

### `ml/src/train.py` — ĐÃ CODE
Baseline LightGBM, StratifiedKFold OOF, dùng categorical native của LGBM, KHÔNG
reweight, KHÔNG impute. In cả 2 họ metric. AUC OOF (chạy lại local, seed=42) = **0.76076**.
(Ghi chú "~0.74" trong bản trước là số cũ/khác môi trường — số 0.76076 này mới là mốc
so sánh chính thức vì cùng máy, cùng seed, cùng fold split với Block 2.)

### `ml/src/features/` — ĐÃ CODE (Block 2)
- `aggregations.py`: `aggregate_numeric`, `aggregate_categorical`, `group_size`, `merge_all`,
  `infer_categories`. Categorical agg nhận `categories` cố định lúc fit — bắt buộc để chống
  training-serving skew (1 applicant lẻ lúc serve có thể thiếu category so với lúc train).
- `bureau.py`: 2-level agg đúng như thiết kế — `bureau_balance` groupby SK_ID_BUREAU (level 1,
  domain STATUS hardcode vì đây là enum đóng theo data dictionary) → merge vào `bureau` →
  groupby SK_ID_CURR (level 2).
- `previous_application.py`, `pos_cash.py`, `credit_card.py`: agg trực tiếp về SK_ID_CURR
  (3 bảng này đã có sẵn cột SK_ID_CURR, không cần 2-level). Sentinel `365243` trong các cột
  DAYS_* của `previous_application` được dọn về NaN trước khi agg.
- `installments.py`: thêm 2 cột derived ở mức dòng TRƯỚC khi agg — `DAYS_LATE`,
  `PAYMENT_DIFF` — đây là tín hiệu hành vi trả nợ quan trọng nhất, không tái tạo được
  nếu chỉ agg riêng từng cột gốc.
- `build.py`: `FeaturePipeline` (`fit`/`transform`), pickle được — đây là artifact
  `feature_pipeline.pkl` mà backend sẽ load. `transform()` cũng dọn sentinel `DAYS_EMPLOYED
  == 365243` (~18% dòng, chủ yếu hưu trí) trong `application` về NaN.
- Kết quả: 709 features (từ 120 baseline). **AUC OOF = 0.78757** (delta **+0.02681**,
  cùng fold split với baseline).
- Test: `ml/tests/test_aggregations.py`, `test_bureau.py`, `test_build.py` (15 test) — có
  test chuyên bắt training-serving skew, đã bắt được 1 bug thật lúc code (domain STATUS của
  `bureau_balance` suy ra động thay vì cố định).

### `ml/src/train_engineered.py` — ĐÃ CODE (Block 2 retrain + Block 3)
Load feature engineered (cache `ml/artifacts/train_features.parquet`), map
`features.yaml.monotone_constraints` → mảng đúng thứ tự cột X cho LightGBM
(`build_monotone_constraints`, tự động ép về 0 nếu lỡ khai báo nhầm cho cột categorical).
So sánh trực tiếp AUC với `oof_baseline.npy` (cùng seed/n_folds → cùng fold split).

### `ml/config/features.yaml` — ĐÃ CODE (Block 3)
18 feature có monotonic constraint với lý do domain viết kèm (EXT_SOURCE_*, DAYS_BIRTH,
DAYS_EMPLOYED, AMT_INCOME_TOTAL, REGION_RATING_CLIENT[_W_CITY], bureau overdue,
installments late/underpay, POS DPD). Cố tình KHÔNG ràng buộc AMT_CREDIT/AMT_ANNUITY vì
quan hệ với rủi ro thực tế mơ hồ/phi tuyến.

### `ml/src/calibrate.py` — ĐÃ CODE (Block 4-5)
`nested_calibrate`: đánh giá isotonic/Platt bằng K-fold trên chính OOF (fit calibrator trên
K-1 phần, dự đoán phần còn lại) để không lạc quan ảo. Isotonic thắng: ECE 0.00417 → 0.00060,
Brier 0.06598 → 0.06589 (Platt/sigmoid làm ECE XẤU hơn — model gốc đã khá calibrated sẵn
vì không dùng scale_pos_weight, nên phép biến đổi sigmoid cứng không hợp).
Cũng train model cuối trên 100% data (`num_boost_round` chọn qua 1 split early-stopping
riêng) → lưu `model.txt`, `calibrator.pkl`, `model_card.md`.
Test: `ml/tests/test_calibrate.py` (6 test, gồm cả test chống leakage của nested calibration).

### `ml/src/explain.py` + `reason_codes.py` — ĐÃ CODE (Block 6)
- `explain_applicant(model, feature_names, row, top_k)`: tự reindex `row` theo `feature_names`
  TRƯỚC khi predict/explain — **bug thật đã bắt**: `Booster.predict()` khớp cột DataFrame
  THEO VỊ TRÍ chứ không theo tên (đảo thứ tự cột → predict ra kết quả khác, KHÔNG báo lỗi).
  `model.feature_name()` cũng không đáng tin (LightGBM tự sanitize tên cột có ký tự đặc biệt
  khi lưu `model.txt`). → mọi nơi dùng model (explain.py, backend/scorer.py) đều phải
  reindex theo đúng `feature_names.json`.
- `reason_codes.py`: curated tay ~25 feature tín hiệu mạnh (khớp 18 monotonic + top SHAP),
  fallback dễ đọc (đánh dấu `curated=False`) cho phần còn lại. `build_reason_codes` xử lý
  value là string (cột categorical gốc như CODE_GENDER/ORGANIZATION_TYPE hoàn toàn có thể
  lọt top-K SHAP của 1 applicant cụ thể) và NaN an toàn — **bug thật đã bắt**: ép `float()`
  vô điều kiện lên value sẽ crash khi top feature là categorical.
- Global: 7/18 monotonic feature lọt top-30 SHAP importance (EXT_SOURCE_1/2/3, DAYS_BIRTH,
  DAYS_EMPLOYED, INSTAL_DAYS_LATE_MAX, INSTAL_PAYMENT_DIFF_MEAN).
- Test: `test_explain.py`, `test_reason_codes.py` (11 test).

### `backend/` — ĐÃ CODE (Block 7)
- `scorer.py`: `Scorer.score()` trả cả `base_value`/`raw_margin` (không chỉ top-K reasons)
  để frontend vẽ waterfall cộng dồn khớp — "các yếu tố còn lại" = raw_margin − base_value −
  sum(top-K shap).
- `data/source.py`: load 7 bảng thật MỘT LẦN lúc startup (giữ RAM suốt vòng đời process,
  KHÔNG nạp vào MySQL — MySQL chỉ audit trail). Applicant pool = `application_train` (có
  TARGET thật để đối chiếu ở Insights).
- **Bug thật đã bắt (nghiêm trọng) lúc test**: `feature_pipeline.pkl` ban đầu pickle lúc
  `ml.src.features.build` chạy như `__main__` (qua `python -m ml.src.features.build`) →
  `FeaturePipeline.__module__` bị ghi thành `"__main__"` → KHÔNG unpickle được từ bất kỳ
  entry point nào khác (backend, pytest...). Fix: `build.py` KHÔNG còn `if __name__ ==
  "__main__"`; dùng `ml/src/run_build_features.py` (chỉ import `main()`, không định nghĩa
  class) để build lại artifact khi cần.
- Test (17, `backend/tests/`): dùng data THẬT lọc theo 2 SK_ID_CURR thật (100002 có lịch sử
  đầy đủ TARGET=1, 100006 không có bureau history TARGET=0) — **không mock**, vì tự dựng
  DataFrame tay 2 lần đều thiếu cột (categorical rồi numeric) so với schema pipeline thật
  fit — chuyển hẳn sang filter CSV thật cho chắc. `test_api.py` dùng SQLite in-memory qua
  `StaticPool` (mặc định mỗi connection mới vào `sqlite:///:memory:` là 1 DB rỗng riêng).
- Verify chạy thật (không chỉ unit test): `uvicorn backend.app.main:app`, load 2.5GB data
  ~10-50s, chấm applicant 100002 → PD 43.48%, risk "Cao", ghi MySQL đúng DECIMAL.

### `ml/src/policy.py` + `export_model_card.py` — ĐÃ CODE (Block 9)
- `policy.py`: `approve_mask` (cắt theo phân vị, không theo ngưỡng PD tuyệt đối — giữ tỉ lệ
  duyệt cố định là cách so sánh công bằng giữa các model, và gần cách risk vận hành thật),
  `cutoff_table`, `segment_report`, `adverse_impact_ratio` (4/5ths rule, bỏ qua nhóm <1000
  dòng để không bị nhiễu kéo), `age_bands`.
- `metrics.py` mở rộng: `expected_calibration_error(..., strategy="quantile")`. Lý do: 77%
  prediction < 0.1 nên uniform-10 nhét gần hết vào 1 bin; sai lệch NGƯỢC CHIỀU trong cùng bin
  triệt tiêu nhau. Có test dựng đúng tình huống đó (uniform ra 0, quantile ra 0.02).
- `export_model_card.py`: model card tách khỏi `calibrate.py`. Trước đây template nằm inline
  trong hàm training → sửa một câu chữ cũng phải train lại. Giờ card là hàm thuần của artifact.
- **Bug thật đã bắt**: `json.dump` ghi `NaN` (không phải JSON hợp lệ) khi AUC phân khúc không
  xác định — Python đọc lại được nên im lặng, nhưng `JSON.parse` của trình duyệt ném lỗi →
  trang Insights trắng. Fix: `json_safe()` NaN→None + `allow_nan=False`.
- Test: `test_policy.py` (11), `test_metrics_ece.py` (4).

### `frontend/` — ĐÃ CODE (Block 8)
- Hướng thiết kế **"Risk Console"**: Swiss/International grid + dark-luxury base (đã cân nhắc
  light-first nhưng dark hợp hơn cho "quant/risk desk" aesthetic) + Fraunces (display số/tiêu
  đề) + JetBrains Mono (data/nhãn) + bento composition ở Insights. Palette OKLCH, risk semantic
  3 màu (teal-green/amber/coral-red) tách biệt với accent gold (dùng cho action, không phải status).
  Cả 2 theme dark/light đều đầy đủ, không phải light là afterthought.
  ScoreGauge: scale hiển thị 0–40% (5x default rate 8.07%) thay vì 0–100% tuyến tính — vì hầu hết
  applicant PD thấp, scale tuyến tính làm gauge gần như rỗng, mất khả năng phân biệt. Vẫn có
  tick mốc TB quần thể để không gây hiểu lầm.
  ShapWaterfall: SVG tay (không dùng chart lib) — base_value → từng feature → "các yếu tố còn
  lại" (gộp) → kết quả, cộng dồn khớp chính xác với raw_margin.
- Verify qua browser thật (không chỉ code review): search applicant → score → SPA nav sang
  Insights giữ được state (React Context, mất khi full reload — đúng thiết kế) → waterfall
  đúng dữ liệu → History ghi đúng. Cả dark/light theme, cả mobile 375px.
- **Bug thật đã bắt + sửa lúc verify UI**: label giá trị SHAP trong waterfall dùng nhầm cạnh
  bar (luôn lấy x2/max thay vì đúng cạnh theo chiều dấu) → label đè lên bar; bar dài (vd
  "-1.007") label đè lên tên feature dòng bên trái → thêm logic đặt label NẰM TRONG bar khi
  đủ rộng.
- `docker-compose.yml` chỉ có `mysql` — backend/frontend chạy dev server trực tiếp
  (`uvicorn`, `npm run dev`) cho hot-reload nhanh lúc phát triển; Dockerfile cho 2 service
  này CHƯA viết (không cần thiết cho demo local, có thể thêm nếu cần "1 lệnh chạy hết").

---

## 8. Design system frontend (Block 8)

Tên hướng: **"Risk Console"**. Tham chiếu tinh thần: fintech data-console nghiêm túc
(kiểu Bloomberg terminal tinh chỉnh lại), KHÔNG phải dashboard-by-numbers chung chung.

- **Palette** (`frontend/src/styles/tokens.css`, OKLCH): nền ink ấm gần đen (dark) /
  kem ấm gần trắng (light); accent vàng gold cho action; 3 màu risk semantic tách biệt
  hue khỏi accent (teal-green thấp / amber trung bình / coral-red cao).
- **Typography**: Fraunces (serif, số lớn + heading) + JetBrains Mono (data/nhãn kỹ thuật,
  `tabular-nums`) + system-ui (body, không tính vào giới hạn vì không phải font tải riêng)
  — đúng giới hạn "tối đa 2 font import" của web/performance.md. Cả 2 load qua Google Fonts
  CDN (chấp nhận được cho demo local; self-host là việc có thể làm nếu productionize thật).
- **Component riêng, không dùng chart lib generic**: ScoreGauge (radial arc SVG tay,
  animate bằng `requestAnimationFrame` + easing), ShapWaterfall (waterfall SVG tay).
- **Motion**: compositor-friendly (transform/opacity), tôn trọng `prefers-reduced-motion`.
- Chưa làm: Playwright visual regression theo đúng chuẩn testing.md (KHÔNG có trong scope
  demo hiện tại — có thể thêm sau nếu cần).
