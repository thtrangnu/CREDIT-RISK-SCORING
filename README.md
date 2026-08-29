# Home Credit — chấm điểm rủi ro tín dụng

Project cá nhân làm trên bộ [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
của Kaggle: 7 bảng quan hệ, 307 nghìn hồ sơ vay, dự đoán ai sẽ vỡ nợ.

Mình làm full từ đầu đến cuối: gộp 7 bảng thành feature, train LightGBM, hiệu chỉnh xác suất,
giải thích bằng SHAP, rồi đóng gói thành API có ghi log và một cái web để bấm thử.

Con số mình quan tâm nhất không phải AUC mà là cái này: **nếu chỉ duyệt 70% hồ sơ tốt nhất thì
tỉ lệ vỡ nợ trong nhóm được duyệt tụt từ 8.07% xuống 3.51%**. Tức là giảm được hơn một nửa
tổn thất, và chặn được gần 70% số ca vỡ nợ ngay ở cửa.

## Kết quả

Sau khi gộp cả 7 bảng thay vì chỉ dùng bảng chính:

| | Chỉ `application_train` | Gộp 7 bảng | |
|---|---|---|---|
| AUC (OOF) | 0.76076 | 0.78757 | +0.02681 |
| Gini | 0.52152 | 0.57514 | +0.05362 |
| Số feature | 120 | 709 | |

Hai lần train dùng chung một cách chia fold (cùng seed, cùng thứ tự dòng), nên phần chênh
lệch đúng là do feature chứ không phải do hên xui lúc chia dữ liệu.

### Về calibration

Xếp hạng đúng thôi chưa đủ. Muốn đặt ngưỡng duyệt hay tính giá vốn rủi ro thì con số PD phải
đúng nghĩa xác suất, chứ không chỉ là điểm số để sort.

| | Trước | Sau isotonic |
|---|---|---|
| Brier | 0.06598 | 0.06589 |
| ECE (10 bin đều) | 0.00417 | 0.00060 |
| ECE (50 bin theo phân vị) | 0.00597 | 0.00201 |

Mình report hai kiểu chia bin vì kiểu chia đều dễ cho số đẹp giả. Có tới 77% prediction nằm
dưới 0.1, nên chia đều 10 bin thì một bin nuốt gần hết dữ liệu và mọi sai lệch bên trong nó
bị trung bình hóa mất. Chia theo phân vị thì mỗi bin đều có đủ mẫu. Kết quả là cải thiện vẫn
giữ nguyên độ lớn ở cả hai kiểu, nên mình tin nó là thật.

Số "sau hiệu chỉnh" đo bằng nested CV: cắt OOF thành 5 phần, fit calibrator trên 4 phần rồi
predict phần còn lại. Nếu fit rồi predict luôn trên chính nó thì ECE sẽ đẹp một cách vô nghĩa.

Một chi tiết thú vị: Platt scaling làm ECE **xấu đi**. Lý do là baseline mình cố ý không dùng
`scale_pos_weight`, nên model gốc vốn đã khá calibrated rồi. Ép thêm một hàm sigmoid cứng lên
nó chỉ tổ làm hỏng.

### Điểm số dịch sang quyết định

| Duyệt bao nhiêu | Ngưỡng PD | Vỡ nợ trong nhóm duyệt | Giảm tổn thất | Chặn được |
|---|---|---|---|---|
| 50% | 0.0460 | 2.3% | 71.1% | 85.5% ca |
| 70% | 0.0863 | 3.5% | 56.5% | 69.5% ca |
| 90% | 0.1887 | 5.6% | 30.8% | 37.8% ca |
| duyệt hết | | 8.1% | | |

### Chuyện fairness

Giới tính và tuổi là thuộc tính được bảo vệ theo ECOA, nên mình soi luôn. Ở ngưỡng duyệt 70%,
tính adverse impact ratio theo 4/5ths rule (dưới 0.80 là có vấn đề):

- Theo giới tính: 0.816, vừa đủ qua.
- Theo nhóm tuổi: **0.446, trượt hẳn.**

Đọc kỹ thì model không hề "thiên vị" theo nghĩa thống kê: chênh lệch giữa PD trung bình và bad
rate thật của từng nhóm đều dưới 0.2 điểm phần trăm, trừ nhóm dưới 25 tuổi lệch +1.9pp. Nó dự
đoán đúng mức rủi ro cho từng nhóm. Chênh lệch tỉ lệ duyệt đến từ chênh lệch rủi ro có thật:
nhóm dưới 25 có bad rate 12.3%, nhóm trên 65 chỉ 3.7%.

Vấn đề là "chênh vì rủi ro chênh thật" không cứu được về mặt pháp lý. 4/5ths rule đo tác động
chứ không đo ý định. Nếu đây là hệ thống thật thì con số 0.446 bắt buộc phải qua pháp chế, và
nhiều khả năng phải bỏ các feature đại diện cho tuổi hoặc đặt cutoff riêng theo nhóm. Mình
viết kỹ hơn trong [model card](ml/artifacts/model_card.md) mục 4.

## Cấu trúc

Ba tầng, cắt rõ ràng, tầng trước không biết gì về tầng sau:

```
ml/                     backend/                  frontend/
pipeline ML         →   FastAPI, chỉ load     →   React (Vite)
đứng độc lập            artifact, không train
        └────────── ml/artifacts/ ──────────┘
```

Lý do phải cắt cứng như vậy: khi backend chấm một hồ sơ lẻ, nó phải chạy đúng cái đường tính
feature như lúc train. Nếu feature engineering nằm rải rác trong notebook thì không có cách
nào đảm bảo được. Ở đây nó là một class `FeaturePipeline` pickle được, `fit` trên train và
`transform` dùng lại y nguyên lúc serve.

Backend không bao giờ import `ml.src.train*` hay `ml.src.calibrate`. Nó chỉ đụng vào
`features.build`, `explain` và `reason_codes`, ba module thuần transform, cần để unpickle
được artifact.

## Chạy thử

Cài đặt:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
```

Cái pin `numpy<2.5` trong requirements là bắt buộc chứ không phải cẩn thận thừa. `shap` kéo
theo `numba`, mà numba tới giờ vẫn chưa hỗ trợ numpy 2.5, nên `import shap` sẽ chết ngay.

Tải [data từ Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data) và giải nén 7
file CSV vào `data/`. Thư mục này gitignore, khoảng 2.5GB.

Chạy pipeline theo đúng thứ tự:

```bash
python -m ml.src.run_build_features      # 7 bảng -> 709 feature
python -m ml.src.train                   # baseline
python -m ml.src.train_engineered        # + monotonic constraints
python -m ml.src.calibrate               # isotonic + model cuối
python -m ml.src.explain                 # SHAP
python -m ml.src.export_metrics_summary  # metrics + cutoff + fairness
python -m ml.src.export_model_card
```

Đừng chạy `python -m ml.src.features.build` nhé. Module đó định nghĩa class `FeaturePipeline`,
chạy trực tiếp thì Python nạp nó thành `__main__` và pickle sẽ ghi `__module__ = "__main__"`,
sau đó backend không unpickle được nữa. Mình dính bug này một lần rồi nên giờ có test chặn.

Backend với frontend:

```bash
docker compose up -d mysql            # MySQL cho audit trail, port 3307
cp .env.example .env                  # nhớ sửa mật khẩu
alembic -c backend/alembic.ini upgrade head
uvicorn backend.app.main:app --reload # load 7 bảng vào RAM, đợi 10-50s
```

```bash
cd frontend && npm install && npm run dev
```

Test: `pytest ml/tests backend/tests -q` (55 test).

## Mấy chỗ mình nghĩ nhiều nhất

**Tự viết metric.** File `ml/src/metrics.py` không gọi `sklearn.metrics` cho các metric chính.
AUC viết bằng Mann-Whitney rank, tie thì lấy rank trung bình, chạy O(n log n). Rồi Gini, KS,
PR-AUC, Brier, ECE, partial AUC chuẩn hóa McClish, TPR@FPR, bảng decile kiểu risk. Sklearn chỉ
xuất hiện trong test để assert là mình viết đúng. Làm vậy vì mình muốn thực sự hiểu từng metric
chứ không phải gọi hàm rồi đọc số.

**Baseline để sạch.** Không `scale_pos_weight`, không `is_unbalance`, không impute gì cả.
Không phải lười. Reweighting làm hỏng calibration, mà calibration lại đúng là thứ block sau
đo. Muốn có mốc so sánh thật thì phải để baseline nguyên vẹn.

**Monotonic constraints chọn có lý do.** 18 feature, mỗi cái mình viết lý do domain kèm trong
[`features.yaml`](ml/config/features.yaml). Cố ý không ràng buộc `AMT_CREDIT` và `AMT_ANNUITY`
vì quan hệ của chúng với rủi ro khá mơ hồ, vay nhiều có thể là khách tốt được duyệt nhiều mà
cũng có thể là đang quá tải nợ. Về sau chạy SHAP thì thấy 7 trong 18 feature này lọt top 30,
tức là cái mình suy luận từ domain khớp với cái model thực sự học.

**Chống training-serving skew ngay từ thiết kế.** Domain của mọi cột categorical được chốt lúc
`fit` và áp lại y hệt lúc `transform`, vì một hồ sơ lẻ lúc serve gần như chắc chắn không có đủ
mọi giá trị category như lúc train. Riêng `bureau_balance.STATUS` mình hardcode theo data
dictionary thay vì suy từ data.

**Bảng bureau phải agg hai tầng.** `bureau_balance` chỉ có `SK_ID_BUREAU`, không có
`SK_ID_CURR`. Nên phải gom theo tháng về từng khoản vay trước, merge vào `bureau`, rồi mới gom
tiếp về từng người. Lúc serve backend cũng phải lọc dữ liệu theo đúng cái chain đó.

**Feature phái sinh phải tính ở mức dòng.** `DAYS_LATE` và `PAYMENT_DIFF` trong
`installments_payments` mình tính trước khi agg. Nếu agg riêng từng cột gốc rồi mới trừ nhau
thì mất hết thông tin về độ trễ của từng lần trả cụ thể. Cả hai đều lọt top 30 SHAP.

## Bug đã dính

Mấy cái này test đơn giản không bắt được, phải chạy thật mới lòi ra:

`Booster.predict()` khớp cột theo **vị trí** chứ không theo tên. Đảo thứ tự cột thì ra kết quả
khác mà chẳng báo lỗi gì. Định dùng `model.feature_name()` để reindex thì phát hiện nó cũng
không tin được, vì LightGBM tự sửa tên cột có ký tự đặc biệt khi lưu `model.txt`. Cuối cùng
mọi chỗ dùng model đều phải reindex theo `feature_names.json`.

Pickle ghi `__module__ = "__main__"` như nói ở trên, làm artifact không load được từ backend
lẫn pytest.

Ép `float()` lên giá trị feature mà không kiểm tra, crash ngay khi top SHAP của một người là
cột categorical như `CODE_GENDER`.

`json.dump` ghi ra `NaN`, mà `NaN` không phải JSON hợp lệ. Python đọc lại được nên ở tầng ML
không ai biết, nhưng `JSON.parse` của browser thì ném lỗi luôn, trang Insights trắng bốc. NaN
xuất hiện thật ở chỗ AUC của nhóm chỉ có một class (`CODE_GENDER = 'XNA'`, đúng 4 dòng).

Label trên biểu đồ SHAP waterfall đè lên bar, do lấy nhầm cạnh: luôn lấy cạnh max thay vì lấy
cạnh đúng theo dấu của giá trị.

## Chỗ còn yếu

Viết ra vì đây là những chỗ người ta sẽ hỏi, mà trả lời "mình biết" thì hơn "mình không để ý":

Early stopping đang dùng chính cái fold mà nó sắp predict. Fold validation vừa để dừng sớm vừa
để lấy `oof[va]`, nên số vòng lặp được chọn bằng đúng dữ liệu nó sắp chấm. Nghĩa là AUC 0.78757
lạc quan hơn thực tế một chút. Phần so sánh với baseline thì vẫn công bằng vì cả hai lệch như
nhau. Sửa được bằng cách tách một inner split riêng, mình chưa làm.

Calibrator fit trên OOF nhưng lại đem áp cho model train trên 100% data. Đây là cách chuẩn,
giống `CalibratedClassifierCV(ensemble=False)`, nhưng vẫn là một giả định chứ không hiển nhiên
đúng.

`CODE_GENDER` vẫn đang là feature và có thể lọt vào reason codes. Hệ thống thật thì không được
phép. Mình giữ lại trong demo để bảng phân khúc ở trên có cái mà đối chiếu.

Chưa tune hyperparameter, tham số trong `params.yaml` là chọn tay.

Không có out-of-time validation. Data Kaggle không có trục thời gian rõ ràng, mà scorecard thật
thì bắt buộc phải có cái này.

Cột "giảm tổn thất" giả định loss tỉ lệ với số ca vỡ nợ được duyệt và mọi khoản vay có exposure
như nhau. Đủ để so sánh giữa các ngưỡng với nhau, nhưng không phải mô hình tổn thất thật vì
thiếu LGD, EAD và giá trị từng khoản.

## Chỗ cố tình không làm

Đây là project portfolio, phạm vi demo. Nó có hình dáng production ở những chỗ kể được chuyện
(tách tầng, chống skew, versioning, audit trail) nhưng vẫn chạy local, data tĩnh, một người
dùng. Những thứ sau mình bỏ có chủ đích chứ không phải quên:

auth/JWT, monitoring, Kubernetes, CI/CD, drift detection, retraining tự động, LLM, Playwright
visual regression, Dockerfile cho backend và frontend.

## Cây thư mục

```
ml/                              tầng 1, pipeline ML độc lập
├── config/                          features.yaml + params.yaml
├── src/
│   ├── metrics.py                   metric tự viết
│   ├── features/                    agg tái dùng + FeaturePipeline
│   ├── train.py, train_engineered.py, calibrate.py
│   ├── explain.py, reason_codes.py  SHAP + reason codes
│   ├── policy.py                    cutoff + fairness
│   └── export_*.py
├── artifacts/                       ranh giới sang backend
└── tests/
backend/                         tầng 2, FastAPI chỉ load artifact
├── app/data/                        kéo 7 bảng theo SK_ID_CURR
├── app/db/                          MySQL, log mỗi lần chấm
├── app/routers/                     score, history, insights
└── tests/
frontend/                        tầng 3, React
└── src/
```

Bối cảnh và các quyết định đã chốt nằm trong [ghi chú thiết kế](docs/NOTES.md).
Chi tiết model trong [model card](ml/artifacts/model_card.md).
