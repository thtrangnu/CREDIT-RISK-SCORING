# Home Credit — Credit Risk Scoring

Chấm điểm rủi ro vỡ nợ cho hồ sơ vay tiêu dùng: từ 7 bảng dữ liệu thô ra một con số xác suất
có thể giải trình được từng lý do đứng sau nó.

Ở ngưỡng duyệt 70%, hệ thống kéo tỉ lệ vỡ nợ trong nhóm được duyệt từ 8.07% xuống 3.51%.

## Demo

Ba trang: **Score** (tìm hồ sơ, chấm điểm, xem top lý do), **Insights** (SHAP toàn cục,
waterfall từng hồ sơ, bảng cutoff, bảng fairness), **History** (audit trail mọi lần chấm).

<!-- TODO: chèn ảnh vào đây
![Score](docs/images/score.png)
![Insights](docs/images/insights.png)
-->

> Chưa có screenshot trong repo. Chạy `npm run dev` ở `frontend/` để xem.

## Mục lục

- [Giới thiệu](#giới-thiệu)
- [Tính năng](#tính-năng)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Cài đặt](#cài-đặt)
- [Cách chạy](#cách-chạy)
- [Dữ liệu](#dữ-liệu)
- [Phương pháp](#phương-pháp)
- [Kết quả](#kết-quả)
- [Mấy chỗ nghĩ nhiều nhất](#mấy-chỗ-nghĩ-nhiều-nhất)
- [Bug đã dính](#bug-đã-dính)
- [Tech stack](#tech-stack)
- [Roadmap](#roadmap)
- [License](#license)
- [Tác giả](#tác-giả)

## Giới thiệu

Đây là project cá nhân làm trên bộ [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
của Kaggle. Bài toán: 7 bảng quan hệ, 307 nghìn hồ sơ vay, dự đoán ai sẽ vỡ nợ. Dữ liệu lệch
nặng, chỉ khoảng 8% là positive.

Mình làm nó vì hai lý do. Một, muốn đi hết một pipeline ML thật thay vì dừng ở notebook có
điểm đẹp. Hai, lending là ngành có quản lý, nên explainability ở đây không phải trang trí mà
là yêu cầu pháp lý: từ chối ai thì phải nói được vì sao (adverse action reason codes).

Vì vậy mình quan tâm con số impact hơn con số metric. AUC 0.78 tự nó không nói lên điều gì với
người làm business. "Duyệt 70% hồ sơ tốt nhất thì cắt được hơn nửa tổn thất" thì có.

## Tính năng

- Gộp 7 bảng quan hệ thành 709 feature qua một pipeline tái sử dụng được (không phải code rời
  trong notebook)
- LightGBM với monotonic constraints trên 18 feature có lý do domain rõ ràng
- Hiệu chỉnh xác suất bằng isotonic, đánh giá bằng nested CV nên không lạc quan ảo
- SHAP cho cả toàn cục lẫn từng hồ sơ, dịch sang câu tiếng Việt đọc được
- Bảng cutoff: chọn tỉ lệ duyệt, thấy ngay tổn thất giảm bao nhiêu
- Phân tích fairness theo thuộc tính được bảo vệ (giới tính, tuổi) kèm adverse impact ratio
- API FastAPI ghi audit trail xuống MySQL, có `model_version` cho governance
- Web React xem điểm, waterfall SHAP và lịch sử chấm

## Cấu trúc thư mục

```
ml/                              tầng 1, pipeline ML đứng độc lập
├── config/
│   ├── features.yaml                monotonic constraints + lý do domain
│   └── params.yaml                  hyperparams, seed, n_folds
├── src/
│   ├── metrics.py                   metric tự viết tay
│   ├── features/                    hàm agg tái dùng + FeaturePipeline
│   ├── train.py                     baseline OOF
│   ├── train_engineered.py          OOF trên feature đã gộp + monotonic
│   ├── calibrate.py                 isotonic, model cuối
│   ├── explain.py, reason_codes.py  SHAP + reason codes
│   ├── policy.py                    cutoff policy + fairness
│   └── export_*.py                  metrics_summary.json, model_card.md
├── data/                            gitignore, tải từ Kaggle
├── artifacts/                       ranh giới sang backend
├── notebooks/                       EDA
└── tests/
backend/                         tầng 2, FastAPI chỉ load artifact
├── app/
│   ├── scorer.py                    transform -> predict -> calibrate -> SHAP
│   ├── data/                        kéo 7 bảng theo SK_ID_CURR
│   ├── db/                          MySQL, log mỗi lần chấm
│   └── routers/                     score, history, insights
├── migrations/                      Alembic
└── tests/
frontend/                        tầng 3, React + Vite
└── src/{pages,components,styles}/
```

## Cài đặt

```bash
git clone https://github.com/thtrangnu/Home-credit-scoring-.git
cd Home-credit-scoring-

python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt   # đã kéo theo ml/requirements.txt

cd frontend && npm install && cd ..
```

Cái pin `numpy<2.5` trong requirements là bắt buộc chứ không phải cẩn thận thừa. `shap` kéo
theo `numba`, mà numba tới giờ vẫn chưa hỗ trợ numpy 2.5, nên `import shap` sẽ chết ngay.

## Cách chạy

Chạy pipeline ML theo đúng thứ tự này:

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

Backend và frontend:

```bash
docker compose up -d mysql            # MySQL cho audit trail, port 3307
cp .env.example .env                  # nhớ sửa mật khẩu
alembic -c backend/alembic.ini upgrade head
uvicorn backend.app.main:app --reload # load 7 bảng vào RAM, đợi 10-50s
```

```bash
cd frontend && npm run dev            # http://localhost:5173
```

Test:

```bash
pytest ml/tests backend/tests -q      # 55 test
```

## Dữ liệu

Tải từ [trang competition](https://www.kaggle.com/c/home-credit-default-risk/data), cần tài
khoản Kaggle và bấm đồng ý điều khoản. Giải nén 7 file CSV vào `data/`, khoảng 2.5GB. Thư mục
này đã gitignore.

| Bảng | Một dòng là gì | Khóa |
|---|---|---|
| `application_train` | 1 hồ sơ vay, có `TARGET` | `SK_ID_CURR` |
| `bureau` | 1 khoản tín dụng ở tổ chức khác | `SK_ID_CURR`, sinh `SK_ID_BUREAU` |
| `bureau_balance` | snapshot theo tháng của khoản bureau | chỉ có `SK_ID_BUREAU` |
| `previous_application` | 1 đơn vay Home Credit trước đó | `SK_ID_CURR`, sinh `SK_ID_PREV` |
| `POS_CASH_balance` | lịch sử POS/cash theo tháng | `SK_ID_PREV` |
| `installments_payments` | từng lần trả góp | `SK_ID_PREV` |
| `credit_card_balance` | sao kê thẻ theo tháng | `SK_ID_PREV` |

Lưu ý mọi cột `DAYS_*` là số âm, tính lùi từ ngày nộp đơn. Riêng `DAYS_EMPLOYED` có sentinel
`365243` nghĩa là "không áp dụng", chiếm khoảng 18% dòng, chủ yếu người đã nghỉ hưu. Pipeline
đổi nó về NaN chứ không impute.

## Phương pháp

### Kiến trúc ba tầng

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

### Feature engineering

Bảng `bureau` phải agg hai tầng. `bureau_balance` chỉ có `SK_ID_BUREAU`, không có
`SK_ID_CURR`. Nên phải gom theo tháng về từng khoản vay trước, merge vào `bureau`, rồi mới gom
tiếp về từng người. Lúc serve backend cũng phải lọc dữ liệu theo đúng cái chain đó.

Feature phái sinh phải tính ở mức dòng. `DAYS_LATE` và `PAYMENT_DIFF` trong
`installments_payments` mình tính trước khi agg. Nếu agg riêng từng cột gốc rồi mới trừ nhau
thì mất hết thông tin về độ trễ của từng lần trả cụ thể. Cả hai đều lọt top 30 SHAP.

Domain của mọi cột categorical được chốt lúc `fit` và áp lại y hệt lúc `transform`, vì một hồ
sơ lẻ lúc serve gần như chắc chắn không có đủ mọi giá trị category như lúc train. Riêng
`bureau_balance.STATUS` mình hardcode theo data dictionary thay vì suy từ data.

### Model và hiệu chỉnh

LightGBM, StratifiedKFold 5 fold, tin CV hơn public LB. Categorical dùng native của LightGBM.

Monotonic constraints trên 18 feature, mỗi cái mình viết lý do domain kèm trong
[`features.yaml`](ml/config/features.yaml). Cố ý không ràng buộc `AMT_CREDIT` và `AMT_ANNUITY`
vì quan hệ của chúng với rủi ro khá mơ hồ, vay nhiều có thể là khách tốt được duyệt nhiều mà
cũng có thể là đang quá tải nợ.

Hiệu chỉnh bằng isotonic. Đánh giá bằng nested CV: cắt OOF thành 5 phần, fit calibrator trên 4
phần rồi predict phần còn lại. Nếu fit rồi predict luôn trên chính nó thì ECE sẽ đẹp một cách
vô nghĩa.

## Kết quả

Sau khi gộp cả 7 bảng thay vì chỉ dùng bảng chính:

| | Chỉ `application_train` | Gộp 7 bảng | |
|---|---|---|---|
| AUC (OOF) | 0.76076 | 0.78757 | +0.02681 |
| Gini | 0.52152 | 0.57514 | +0.05362 |
| Số feature | 120 | 709 | |

Hai lần train dùng chung một cách chia fold (cùng seed, cùng thứ tự dòng), nên phần chênh lệch
đúng là do feature chứ không phải do hên xui lúc chia dữ liệu.

### Calibration

Xếp hạng đúng thôi chưa đủ. Muốn đặt ngưỡng duyệt hay tính giá vốn rủi ro thì con số PD phải
đúng nghĩa xác suất, chứ không chỉ là điểm số để sort.

| | Trước | Sau isotonic |
|---|---|---|
| Brier | 0.06598 | 0.06589 |
| ECE (10 bin đều) | 0.00417 | 0.00060 |
| ECE (50 bin theo phân vị) | 0.00597 | 0.00201 |

Mình report hai kiểu chia bin vì kiểu chia đều dễ cho số đẹp giả. Có tới 77% prediction nằm
dưới 0.1, nên chia đều 10 bin thì một bin nuốt gần hết dữ liệu và mọi sai lệch bên trong nó bị
trung bình hóa mất. Chia theo phân vị thì mỗi bin đều có đủ mẫu. Kết quả là cải thiện vẫn giữ
nguyên độ lớn ở cả hai kiểu, nên mình tin nó là thật.

Một chi tiết thú vị: Platt scaling làm ECE xấu đi. Lý do là baseline mình cố ý không dùng
`scale_pos_weight`, nên model gốc vốn đã khá calibrated rồi. Ép thêm một hàm sigmoid cứng lên
nó chỉ tổ làm hỏng.

### Điểm số dịch sang quyết định

| Duyệt bao nhiêu | Ngưỡng PD | Vỡ nợ trong nhóm duyệt | Giảm tổn thất | Chặn được |
|---|---|---|---|---|
| 50% | 0.0460 | 2.3% | 71.1% | 85.5% ca |
| 70% | 0.0863 | 3.5% | 56.5% | 69.5% ca |
| 90% | 0.1887 | 5.6% | 30.8% | 37.8% ca |
| duyệt hết | | 8.1% | | |

### Fairness

Giới tính và tuổi là thuộc tính được bảo vệ theo ECOA, nên mình soi luôn. Ở ngưỡng duyệt 70%,
tính adverse impact ratio theo 4/5ths rule (dưới 0.80 là có vấn đề):

- Theo giới tính: 0.816, vừa đủ qua.
- Theo nhóm tuổi: 0.446, trượt hẳn.

Đọc kỹ thì model không hề thiên vị theo nghĩa thống kê: chênh lệch giữa PD trung bình và bad
rate thật của từng nhóm đều dưới 0.2 điểm phần trăm, trừ nhóm dưới 25 tuổi lệch +1.9pp. Nó dự
đoán đúng mức rủi ro cho từng nhóm. Chênh lệch tỉ lệ duyệt đến từ chênh lệch rủi ro có thật:
nhóm dưới 25 có bad rate 12.3%, nhóm trên 65 chỉ 3.7%.

Vấn đề là "chênh vì rủi ro chênh thật" không cứu được về mặt pháp lý. 4/5ths rule đo tác động
chứ không đo ý định. Nếu đây là hệ thống thật thì con số 0.446 bắt buộc phải qua pháp chế, và
nhiều khả năng phải bỏ các feature đại diện cho tuổi hoặc đặt cutoff riêng theo nhóm. Mình
viết kỹ hơn trong [model card](ml/artifacts/model_card.md) mục 4.

## Mấy chỗ nghĩ nhiều nhất

**Tự viết metric.** File `ml/src/metrics.py` không gọi `sklearn.metrics` cho các metric chính.
AUC viết bằng Mann-Whitney rank, tie thì lấy rank trung bình, chạy O(n log n). Rồi Gini, KS,
PR-AUC, Brier, ECE, partial AUC chuẩn hóa McClish, TPR@FPR, bảng decile kiểu risk. Sklearn chỉ
xuất hiện trong test để assert là mình viết đúng. Làm vậy vì mình muốn thực sự hiểu từng metric
chứ không phải gọi hàm rồi đọc số.

**Baseline để sạch.** Không `scale_pos_weight`, không `is_unbalance`, không impute gì cả.
Không phải lười. Reweighting làm hỏng calibration, mà calibration lại đúng là thứ block sau đo.
Muốn có mốc so sánh thật thì phải để baseline nguyên vẹn.

**Monotonic constraints khớp với SHAP.** Sau khi chạy SHAP thì thấy 7 trong 18 feature bị ràng
buộc lọt top 30 quan trọng nhất. Tức là cái mình suy luận từ domain khớp với cái model thực sự
học được, chứ không phải mình áp đặt lên nó.

## Bug đã dính

Mấy cái này test đơn giản không bắt được, phải chạy thật mới lòi ra:

`Booster.predict()` khớp cột theo **vị trí** chứ không theo tên. Đảo thứ tự cột thì ra kết quả
khác mà chẳng báo lỗi gì. Định dùng `model.feature_name()` để reindex thì phát hiện nó cũng
không tin được, vì LightGBM tự sửa tên cột có ký tự đặc biệt khi lưu `model.txt`. Cuối cùng
mọi chỗ dùng model đều phải reindex theo `feature_names.json`.

Pickle ghi `__module__ = "__main__"` như nói ở phần Cách chạy, làm artifact không load được từ
backend lẫn pytest.

Ép `float()` lên giá trị feature mà không kiểm tra, crash ngay khi top SHAP của một người là
cột categorical như `CODE_GENDER`.

`json.dump` ghi ra `NaN`, mà `NaN` không phải JSON hợp lệ. Python đọc lại được nên ở tầng ML
không ai biết, nhưng `JSON.parse` của browser thì ném lỗi luôn, trang Insights trắng bốc. NaN
xuất hiện thật ở chỗ AUC của nhóm chỉ có một class (`CODE_GENDER = 'XNA'`, đúng 4 dòng).

Label trên biểu đồ SHAP waterfall đè lên bar, do lấy nhầm cạnh: luôn lấy cạnh max thay vì lấy
cạnh đúng theo dấu của giá trị.

## Tech stack

**ML:** LightGBM, SHAP, scikit-learn (chỉ dùng cho isotonic/Platt và chia fold), pandas, numpy,
PyArrow
**Backend:** FastAPI, Pydantic v2, SQLAlchemy 2.0, Alembic, MySQL 8, PyMySQL
**Frontend:** React 19, Vite, React Router. Biểu đồ vẽ SVG tay, không dùng chart lib
**Khác:** Docker Compose, pytest

## Roadmap

### Chỗ còn yếu

Viết ra vì đây là những chỗ người ta sẽ hỏi, mà trả lời "mình biết" thì hơn "mình không để ý":

Early stopping đang dùng chính cái fold mà nó sắp predict. Fold validation vừa để dừng sớm vừa
để lấy `oof[va]`, nên số vòng lặp được chọn bằng đúng dữ liệu nó sắp chấm. Nghĩa là AUC 0.78757
lạc quan hơn thực tế một chút. Phần so sánh với baseline thì vẫn công bằng vì cả hai lệch như
nhau. Sửa được bằng cách tách một inner split riêng, mình chưa làm.

Calibrator fit trên OOF nhưng lại đem áp cho model train trên 100% data. Đây là cách chuẩn,
giống `CalibratedClassifierCV(ensemble=False)`, nhưng vẫn là một giả định chứ không hiển nhiên
đúng.

`CODE_GENDER` vẫn đang là feature và có thể lọt vào reason codes. Hệ thống thật thì không được
phép. Mình giữ lại trong demo để bảng fairness ở trên có cái mà đối chiếu.

Chưa tune hyperparameter, tham số trong `params.yaml` là chọn tay.

Không có out-of-time validation. Data Kaggle không có trục thời gian rõ ràng, mà scorecard thật
thì bắt buộc phải có cái này.

Cột "giảm tổn thất" giả định loss tỉ lệ với số ca vỡ nợ được duyệt và mọi khoản vay có exposure
như nhau. Đủ để so sánh giữa các ngưỡng với nhau, nhưng không phải mô hình tổn thất thật vì
thiếu LGD, EAD và giá trị từng khoản.

### Định làm tiếp

- [ ] Chèn screenshot vào phần Demo
- [ ] Tách inner split để có OOF sạch
- [ ] Tune hyperparameter bằng Optuna
- [ ] Bỏ `CODE_GENDER` khỏi feature, đo lại xem mất bao nhiêu AUC

### Cố tình không làm

Đây là project portfolio, phạm vi demo. Nó có hình dáng production ở những chỗ kể được chuyện
(tách tầng, chống skew, versioning, audit trail) nhưng vẫn chạy local, data tĩnh, một người
dùng. Những thứ sau mình bỏ có chủ đích chứ không phải quên: auth/JWT, monitoring, Kubernetes,
CI/CD, drift detection, retraining tự động, LLM, Playwright visual regression, Dockerfile cho
backend và frontend.

## License

Code trong repo này để học và làm portfolio, dùng thoải mái. Riêng dữ liệu thì theo điều khoản
của [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/rules) trên
Kaggle, mình không phân phối lại nên bạn cần tự tải.

## Tác giả

[@thtrangnu](https://github.com/thtrangnu)

Bối cảnh và các quyết định đã chốt nằm trong [ghi chú thiết kế](docs/NOTES.md).
Chi tiết model trong [model card](ml/artifacts/model_card.md).
