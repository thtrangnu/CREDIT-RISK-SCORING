# Home Credit — Credit Risk Scoring

Chấm điểm rủi ro vỡ nợ cho hồ sơ vay tiêu dùng. Input là 7 bảng dữ liệu thô, output là một
con số xác suất kèm lý do vì sao ra con số đó.

Duyệt 70% hồ sơ tốt nhất thì tỉ lệ vỡ nợ trong nhóm được duyệt tụt từ 8.07% xuống 3.51%.

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6-02569B?style=flat-square)
![SHAP](https://img.shields.io/badge/SHAP-explainability-8B5CF6?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.3-150458?style=flat-square&logo=pandas&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063?style=flat-square&logo=pydantic&logoColor=white)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0-D71F00?style=flat-square&logo=sqlalchemy&logoColor=white)
![MySQL](https://img.shields.io/badge/MySQL-8.0-4479A1?style=flat-square&logo=mysql&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?style=flat-square&logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-8-646CFF?style=flat-square&logo=vite&logoColor=white)
![Docker](https://img.shields.io/badge/Docker_Compose-2496ED?style=flat-square&logo=docker&logoColor=white)

## 🛠 Tech stack

| Tầng | Dùng gì |
|---|---|
| 🤖 **Model** | LightGBM, SHAP, scikit-learn (chỉ cho isotonic/Platt và chia fold) |
| 🐼 **Data** | pandas, numpy, PyArrow |
| ⚡ **API** | FastAPI, Pydantic v2, SQLAlchemy 2.0, Alembic |
| 🗄 **DB** | MySQL 8 (audit trail), PyMySQL |
| ⚛️ **Web** | React 19, Vite, React Router. Biểu đồ vẽ SVG tay, không dùng chart lib |
| 🧪 **Khác** | Docker Compose, pytest |


## 📸 Demo

Ba trang: Score (tìm hồ sơ, chấm điểm, xem top lý do), Insights (SHAP toàn cục, waterfall từng
hồ sơ, bảng cutoff, bảng fairness), History (log mọi lần chấm).

<!-- TODO: chèn ảnh vào đây
![Score](docs/images/score.png)
![Insights](docs/images/insights.png)
-->

> Chưa có screenshot. Chạy `npm run dev` ở `frontend/` để xem.

## 📑 Mục lục

- [Tech stack](#-tech-stack)
- [Demo](#-demo)
- [Giới thiệu](#-giới-thiệu)
- [Tính năng](#-tính-năng)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Cài đặt](#-cài-đặt)
- [Cách chạy](#-cách-chạy)
- [Dữ liệu](#-dữ-liệu)
- [Phương pháp](#-phương-pháp)
- [Kết quả](#-kết-quả)
- [Vài chỗ đáng nói](#-vài-chỗ-đáng-nói)
- [Bug đã dính](#-bug-đã-dính)
- [Roadmap](#-roadmap)
- [License](#-license)
- [Tác giả](#-tác-giả)

## 🎯 Giới thiệu

Bộ [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) trên Kaggle.
7 bảng quan hệ, 307 nghìn hồ sơ, khoảng 8% vỡ nợ. Bài toán binary classification kinh điển
nhưng data lệch và trải ra nhiều bảng nên phần khó nằm ở feature engineering.

Mình làm để đi hết một pipeline thật, không dừng ở notebook có điểm đẹp.

Lending là ngành có quản lý. Từ chối ai thì phải giải trình được vì sao (adverse action reason
codes), nên explainability ở đây là bắt buộc chứ không phải thêm cho vui. Đó cũng là lý do
mình đẩy SHAP ra tận UI thay vì để nó nằm trong notebook.

AUC 0.78 nói được gì với người làm business? Không nhiều. Nên phần cuối mình dịch điểm số
thành quyết định duyệt hay không, rồi đo xem cắt được bao nhiêu tổn thất.

## ✨ Tính năng

- Gộp 7 bảng thành 709 feature qua pipeline tái sử dụng được, không phải code rời trong notebook
- LightGBM + monotonic constraints trên 18 feature
- Hiệu chỉnh xác suất bằng isotonic, đánh giá bằng nested CV
- SHAP toàn cục và từng hồ sơ, dịch sang câu tiếng Việt đọc được
- Bảng cutoff: chọn tỉ lệ duyệt, thấy ngay tổn thất giảm bao nhiêu
- Phân tích fairness theo giới tính và tuổi, kèm adverse impact ratio
- API FastAPI ghi log xuống MySQL, có `model_version`
- Web React xem điểm, waterfall SHAP, lịch sử

## 📁 Cấu trúc thư mục

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

## 📦 Cài đặt

```bash
git clone https://github.com/thtrangnu/Home-credit-scoring-.git
cd Home-credit-scoring-

python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt   # đã kéo theo ml/requirements.txt

cd frontend && npm install && cd ..
```

Pin `numpy<2.5` là bắt buộc. shap kéo theo numba, numba chưa hỗ trợ numpy 2.5, `import shap`
chết ngay.

## ▶️ Cách chạy

Pipeline ML, đúng thứ tự này:

```bash
python -m ml.src.run_build_features      # 7 bảng -> 709 feature
python -m ml.src.train                   # baseline
python -m ml.src.train_engineered        # + monotonic constraints
python -m ml.src.calibrate               # isotonic + model cuối
python -m ml.src.explain                 # SHAP
python -m ml.src.export_metrics_summary  # metrics + cutoff + fairness
python -m ml.src.export_model_card
```

Đừng chạy `python -m ml.src.features.build`. Module đó định nghĩa class `FeaturePipeline`,
chạy trực tiếp thì Python nạp nó thành `__main__`, pickle ghi `__module__ = "__main__"`, và
backend không load lại được. Có test chặn rồi nhưng nói trước cho chắc.

Backend + frontend:

```bash
docker compose up -d mysql            # MySQL cho audit trail, port 3307
cp .env.example .env                  # nhớ sửa mật khẩu
alembic -c backend/alembic.ini upgrade head
uvicorn backend.app.main:app --reload # load 7 bảng vào RAM, đợi 10-50s
```

```bash
cd frontend && npm run dev            # http://localhost:5173
```

Test: `pytest ml/tests backend/tests -q`, 55 cái.

## 🗃 Dữ liệu

Tải ở [trang competition](https://www.kaggle.com/c/home-credit-default-risk/data), cần tài
khoản Kaggle và bấm đồng ý điều khoản. Giải nén 7 file CSV vào `data/`. Khoảng 2.5GB, đã
gitignore.

| Bảng | Một dòng là gì | Khóa |
|---|---|---|
| `application_train` | 1 hồ sơ vay, có `TARGET` | `SK_ID_CURR` |
| `bureau` | 1 khoản tín dụng ở tổ chức khác | `SK_ID_CURR`, sinh `SK_ID_BUREAU` |
| `bureau_balance` | snapshot theo tháng của khoản bureau | chỉ có `SK_ID_BUREAU` |
| `previous_application` | 1 đơn vay Home Credit trước đó | `SK_ID_CURR`, sinh `SK_ID_PREV` |
| `POS_CASH_balance` | lịch sử POS/cash theo tháng | `SK_ID_PREV` |
| `installments_payments` | từng lần trả góp | `SK_ID_PREV` |
| `credit_card_balance` | sao kê thẻ theo tháng | `SK_ID_PREV` |

Hai cái bẫy trong data: mọi cột `DAYS_*` là số âm tính lùi từ ngày nộp đơn, và `DAYS_EMPLOYED`
có sentinel `365243` nghĩa là "không áp dụng", chiếm 18% dòng, phần lớn là người nghỉ hưu.
Pipeline đổi nó về NaN.

## 🔬 Phương pháp

### Kiến trúc

```
ml/                     backend/                  frontend/
pipeline ML         →   FastAPI, chỉ load     →   React (Vite)
đứng độc lập            artifact, không train
        └────────── ml/artifacts/ ──────────┘
```

Backend chấm một hồ sơ lẻ thì phải chạy đúng cái đường tính feature như lúc train. Feature
engineering nằm rải rác trong notebook thì chịu, không đảm bảo được. Nên nó được gói thành
class `FeaturePipeline` pickle được: `fit` một lần trên train, `transform` dùng lại y nguyên
lúc serve.

Backend không import `ml.src.train*` hay `ml.src.calibrate`. Chỉ đụng `features.build`,
`explain`, `reason_codes`. Ba module này thuần transform, cần để unpickle artifact.

### Feature engineering

`bureau` phải agg hai tầng. `bureau_balance` chỉ có `SK_ID_BUREAU`, không có `SK_ID_CURR`. Gom
theo tháng về từng khoản vay, merge vào `bureau`, rồi mới gom về từng người. Backend lúc serve
cũng lọc theo đúng chain đó.

`DAYS_LATE` và `PAYMENT_DIFF` trong `installments_payments` tính ở mức dòng, trước khi agg. Agg
riêng từng cột gốc rồi trừ nhau thì mất sạch thông tin về từng lần trả cụ thể. Cả hai lọt top
30 SHAP.

Domain của cột categorical chốt lúc `fit`, áp lại y hệt lúc `transform`. Một hồ sơ lẻ thì gần
như chắc chắn không có đủ mọi giá trị category như lúc train. `bureau_balance.STATUS` thì
hardcode theo data dictionary luôn.

### Model

LightGBM, StratifiedKFold 5 fold. Tin CV hơn public LB. Categorical dùng native của LightGBM,
không one-hot.

Monotonic constraints trên 18 feature, lý do domain viết kèm trong
[`features.yaml`](ml/config/features.yaml). Không ràng buộc `AMT_CREDIT` với `AMT_ANNUITY`.
Vay nhiều có thể là khách tốt được duyệt nhiều, cũng có thể là đang quá tải nợ. Không có
hướng rõ ràng.

Hiệu chỉnh bằng isotonic, đo bằng nested CV: cắt OOF thành 5 phần, fit calibrator trên 4 phần
rồi predict phần còn lại. Fit rồi predict luôn trên chính nó thì ECE sẽ đẹp một cách vô nghĩa.

## 📊 Kết quả

Gộp cả 7 bảng so với chỉ dùng bảng chính:

| | Chỉ `application_train` | Gộp 7 bảng | |
|---|---|---|---|
| AUC (OOF) | 0.76076 | 0.78757 | +0.02681 |
| Gini | 0.52152 | 0.57514 | +0.05362 |
| Số feature | 120 | 709 | |

Hai lần train dùng chung cách chia fold (cùng seed, cùng thứ tự dòng). Phần chênh lệch là do
feature, không phải hên xui lúc chia data.

### Calibration

| | Trước | Sau isotonic |
|---|---|---|
| Brier | 0.06598 | 0.06589 |
| ECE (10 bin đều) | 0.00417 | 0.00060 |
| ECE (50 bin theo phân vị) | 0.00597 | 0.00201 |

Report hai kiểu chia bin vì chia đều dễ cho số đẹp giả. 77% prediction nằm dưới 0.1, chia đều
10 bin thì một bin nuốt gần hết data, sai lệch bên trong bị trung bình hóa mất. Chia theo phân
vị thì bin nào cũng đủ mẫu. Cải thiện giữ nguyên độ lớn ở cả hai kiểu nên mình tin nó thật.

Platt scaling làm ECE xấu đi. Baseline không dùng `scale_pos_weight` nên model gốc vốn đã khá
calibrated, ép thêm sigmoid cứng lên chỉ tổ hỏng.

### Điểm số dịch sang quyết định

| Duyệt bao nhiêu | Ngưỡng PD | Vỡ nợ trong nhóm duyệt | Giảm tổn thất | Chặn được |
|---|---|---|---|---|
| 50% | 0.0460 | 2.3% | 71.1% | 85.5% ca |
| 70% | 0.0863 | 3.5% | 56.5% | 69.5% ca |
| 90% | 0.1887 | 5.6% | 30.8% | 37.8% ca |
| duyệt hết | | 8.1% | | |

### Fairness

Giới tính và tuổi là thuộc tính được bảo vệ theo ECOA. Ở ngưỡng duyệt 70%, adverse impact
ratio theo 4/5ths rule (dưới 0.80 là có vấn đề):

- Giới tính: 0.816, vừa đủ qua
- Tuổi: 0.446, trượt hẳn

Model thì không thiên vị theo nghĩa thống kê. Chênh lệch giữa PD trung bình và bad rate thật
của từng nhóm đều dưới 0.2 điểm phần trăm, trừ nhóm dưới 25 tuổi lệch +1.9pp. Nó dự đoán đúng
mức rủi ro cho từng nhóm. Tỉ lệ duyệt chênh nhau là vì rủi ro chênh nhau thật: nhóm dưới 25
có bad rate 12.3%, nhóm trên 65 chỉ 3.7%.

Cái đó không cứu được về mặt pháp lý. 4/5ths rule đo tác động, không quan tâm ý định. Hệ thống
thật thì con số 0.446 phải qua pháp chế, nhiều khả năng phải bỏ các feature đại diện cho tuổi
hoặc đặt cutoff riêng theo nhóm. Viết kỹ hơn trong [model card](ml/artifacts/model_card.md)
mục 4.

## 💭 Vài chỗ đáng nói

**Metric tự viết.** `ml/src/metrics.py` không gọi `sklearn.metrics` cho metric chính. AUC viết
bằng Mann-Whitney rank, tie lấy rank trung bình, O(n log n). Rồi Gini, KS, PR-AUC, Brier, ECE,
partial AUC chuẩn hóa McClish, TPR@FPR, bảng decile kiểu risk. Sklearn chỉ xuất hiện trong test
để assert. Mất thời gian nhưng giờ mình hiểu từng cái làm gì.

**Baseline để sạch.** Không `scale_pos_weight`, không `is_unbalance`, không impute. Reweighting
làm hỏng calibration, mà calibration là thứ block sau đo, nên để nguyên.

**Monotonic constraints khớp SHAP.** 7 trong 18 feature bị ràng buộc lọt top 30 quan trọng
nhất. Cái suy từ domain khớp cái model học được, không phải mình áp đặt lên nó.

## 🐛 Bug đã dính

Test đơn giản không bắt được, phải chạy thật mới lòi:

`Booster.predict()` khớp cột theo **vị trí**, không theo tên. Đảo thứ tự cột thì ra kết quả
khác, không báo lỗi gì. Định dùng `model.feature_name()` để reindex thì hóa ra nó cũng không
tin được: LightGBM tự sửa tên cột có ký tự đặc biệt khi lưu `model.txt`. Cuối cùng mọi chỗ
dùng model đều reindex theo `feature_names.json`.

Pickle ghi `__module__ = "__main__"`, artifact không load được từ backend lẫn pytest. Đã nói ở
phần Cách chạy.

Ép `float()` lên giá trị feature mà không check kiểu. Crash khi top SHAP của một người là cột
categorical như `CODE_GENDER`.

`json.dump` ghi ra `NaN`. Mà `NaN` không phải JSON hợp lệ. Python đọc lại được nên ở tầng ML
chả ai biết, nhưng `JSON.parse` của browser thì ném lỗi luôn, trang Insights trắng bốc. NaN
đến từ AUC của nhóm chỉ có một class: `CODE_GENDER = 'XNA'`, đúng 4 dòng.

Label trên SHAP waterfall đè lên bar. Lấy nhầm cạnh, luôn lấy cạnh max thay vì cạnh đúng theo
dấu.

## 🗺 Roadmap

### Chỗ còn yếu

Liệt kê ra vì kiểu gì cũng có người hỏi:

Early stopping đang dùng chính cái fold nó sắp predict. Fold validation vừa để dừng sớm vừa để
lấy `oof[va]`, nên số vòng lặp được chọn bằng đúng data nó sắp chấm. AUC 0.78757 vì thế lạc
quan hơn thực tế một chút. Phần so với baseline vẫn công bằng vì cả hai lệch như nhau. Tách
một inner split là xong, mình chưa làm.

Calibrator fit trên OOF nhưng đem áp cho model train trên 100% data. Cách này chuẩn, giống
`CalibratedClassifierCV(ensemble=False)`, nhưng vẫn là giả định.

`CODE_GENDER` vẫn là feature và có thể lọt vào reason codes. Hệ thống thật thì không được
phép. Giữ lại trong demo để bảng fairness có cái đối chiếu.

Chưa tune hyperparameter. Tham số trong `params.yaml` chọn tay.

Không có out-of-time validation. Data Kaggle không có trục thời gian rõ ràng. Scorecard thật
thì bắt buộc phải có.

Cột "giảm tổn thất" giả định loss tỉ lệ với số ca vỡ nợ được duyệt, mọi khoản vay exposure như
nhau. So sánh giữa các ngưỡng thì được, gọi là mô hình tổn thất thì không. Thiếu LGD, EAD,
giá trị từng khoản.

### Định làm tiếp

- [ ] Chèn screenshot vào phần Demo
- [ ] Tách inner split để có OOF sạch
- [ ] Tune hyperparameter bằng Optuna
- [ ] Bỏ `CODE_GENDER` khỏi feature, đo lại xem mất bao nhiêu AUC

### Cố tình không làm

Project portfolio, phạm vi demo. Chạy local, data tĩnh, một người dùng. Mấy thứ sau bỏ có chủ
đích: auth/JWT, monitoring, Kubernetes, CI/CD, drift detection, retraining tự động, LLM,
Playwright visual regression, Dockerfile cho backend và frontend.

## 📄 License

Code dùng thoải mái. Data thì theo điều khoản của
[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/rules) trên
Kaggle, mình không phân phối lại nên bạn tự tải.

## 👤 Tác giả

[@thtrangnu](https://github.com/thtrangnu)

Bối cảnh và quyết định đã chốt: [ghi chú thiết kế](docs/NOTES.md). Chi tiết model:
[model card](ml/artifacts/model_card.md).
