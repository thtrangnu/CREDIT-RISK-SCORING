# Home Credit — Credit Risk Scoring

Default risk scoring for consumer loan applications. Seven raw tables go in, a probability
comes out, along with the reasons behind it.

Approve the best 70% of applications and the default rate among approved drops from 8.07%
to 3.51%.

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

| Layer | What |
|---|---|
| 🤖 **Model** | LightGBM, SHAP, scikit-learn (only for isotonic/Platt and fold splitting) |
| 🐼 **Data** | pandas, numpy, PyArrow |
| ⚡ **API** | FastAPI, Pydantic v2, SQLAlchemy 2.0, Alembic |
| 🗄 **DB** | MySQL 8 (audit trail), PyMySQL |
| ⚛️ **Web** | React 19, Vite, React Router. Charts are hand-written SVG, no chart library |
| 🧪 **Other** | Docker Compose, pytest |

## 📸 Demo

Three pages: Score (find an application, score it, see the top reasons), Insights (global
SHAP, per-application waterfall, cutoff table, fairness table), History (log of every scoring
call).

The UI is in Vietnamese, including the reason codes the model produces.

<!-- TODO: add images here
![Score](docs/images/score.png)
![Insights](docs/images/insights.png)
-->

> No screenshots yet. Run `npm run dev` in `frontend/` to see it.

## 📑 Contents

- [Tech stack](#-tech-stack)
- [Demo](#-demo)
- [Background](#-background)
- [Features](#-features)
- [Project structure](#-project-structure)
- [Installation](#-installation)
- [Running it](#-running-it)
- [Data](#-data)
- [Method](#-method)
- [Results](#-results)

## 🎯 Background

The [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset on
Kaggle. Seven relational tables, 307k applications, about 8% default. A textbook binary
classification problem, except the data is imbalanced and spread across tables, so the hard
part is feature engineering.

I built this to go through a full pipeline end to end instead of stopping at a notebook with
a nice score.

Lending is regulated. If you reject someone you have to be able to explain why (adverse action
reason codes), so explainability here is a requirement, not decoration. That is also why SHAP
made it all the way into the UI rather than staying in a notebook.

What does AUC 0.78 tell a business person? Not much. So the last part turns the score into an
approve/reject decision and measures how much loss that actually prevents.

## ✨ Features

- Joins 7 tables into 709 features through a reusable pipeline, not scattered notebook cells
- LightGBM with monotonic constraints on 18 features
- Probability calibration via isotonic regression, evaluated with nested CV
- SHAP both globally and per application, rendered as readable sentences
- Cutoff table: pick an approval rate, see how much loss it prevents
- Fairness analysis by gender and age with adverse impact ratios
- FastAPI service logging every call to MySQL, with `model_version`
- React app for the score, the SHAP waterfall and the history

## 📁 Project structure

```
ml/                              layer 1, standalone ML pipeline
├── config/
│   ├── features.yaml                monotonic constraints + domain rationale
│   └── params.yaml                  hyperparams, seed, n_folds
├── src/
│   ├── metrics.py                   hand-written metrics
│   ├── features/                    reusable aggregations + FeaturePipeline
│   ├── train.py                     baseline OOF
│   ├── train_engineered.py          OOF on joined features + monotonic
│   ├── calibrate.py                 isotonic, final model
│   ├── explain.py, reason_codes.py  SHAP + reason codes
│   ├── policy.py                    cutoff policy + fairness
│   └── export_*.py                  metrics_summary.json, model_card.md
├── data/                            gitignored, download from Kaggle
├── artifacts/                       the boundary with backend
├── notebooks/                       EDA
└── tests/
backend/                         layer 2, FastAPI, only loads artifacts
├── app/
│   ├── scorer.py                    transform -> predict -> calibrate -> SHAP
│   ├── data/                        pulls the 7 tables by SK_ID_CURR
│   ├── db/                          MySQL, one row per scoring call
│   └── routers/                     score, history, insights
├── migrations/                      Alembic
└── tests/
frontend/                        layer 3, React + Vite
└── src/{pages,components,styles}/
```

## 📦 Installation

```bash
git clone https://github.com/thtrangnu/Home-credit-scoring-.git
cd Home-credit-scoring-

python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt   # pulls in ml/requirements.txt

cd frontend && npm install && cd ..
```

The `numpy<2.5` pin is mandatory. shap depends on numba, numba does not support numpy 2.5 yet,
so `import shap` dies immediately.

## ▶️ Running it

The ML pipeline, in this order:

```bash
python -m ml.src.run_build_features      # 7 tables -> 709 features
python -m ml.src.train                   # baseline
python -m ml.src.train_engineered        # + monotonic constraints
python -m ml.src.calibrate               # isotonic + final model
python -m ml.src.explain                 # SHAP
python -m ml.src.export_metrics_summary  # metrics + cutoff + fairness
python -m ml.src.export_model_card
```

Do not run `python -m ml.src.features.build`. That module defines the `FeaturePipeline` class,
so running it directly makes Python load it as `__main__`, the pickle records
`__module__ = "__main__"`, and the backend can no longer load the artifact. There is a test
guarding this now, but worth saying anyway.

Backend and frontend:

```bash
docker compose up -d mysql            # MySQL for the audit trail, port 3307
cp .env.example .env                  # change the passwords
alembic -c backend/alembic.ini upgrade head
uvicorn backend.app.main:app --reload # loads 7 tables into RAM, takes 10-50s
```

```bash
cd frontend && npm run dev            # http://localhost:5173
```

Tests: `pytest ml/tests backend/tests -q`, 55 of them.

## 🗃 Data

Download from the [competition page](https://www.kaggle.com/c/home-credit-default-risk/data).
You need a Kaggle account and have to accept the rules. Unzip the 7 CSVs into `data/`. Around
2.5GB, already gitignored.

| Table | One row is | Key |
|---|---|---|
| `application_train` | one loan application, has `TARGET` | `SK_ID_CURR` |
| `bureau` | one credit line at another institution | `SK_ID_CURR`, emits `SK_ID_BUREAU` |
| `bureau_balance` | monthly snapshot of a bureau credit | only has `SK_ID_BUREAU` |
| `previous_application` | one earlier Home Credit application | `SK_ID_CURR`, emits `SK_ID_PREV` |
| `POS_CASH_balance` | monthly POS/cash history | `SK_ID_PREV` |
| `installments_payments` | one installment payment | `SK_ID_PREV` |
| `credit_card_balance` | monthly card statement | `SK_ID_PREV` |

Two traps in the data. Every `DAYS_*` column is negative, counting backwards from the
application date. And `DAYS_EMPLOYED` has a `365243` sentinel meaning "not applicable", which
covers 18% of rows, mostly retirees. The pipeline turns it into NaN.

## 🔬 Method

### Architecture

```
ml/                     backend/                  frontend/
standalone ML       →   FastAPI, only loads   →   React (Vite)
pipeline                artifacts, no training
        └────────── ml/artifacts/ ──────────┘
```

When the backend scores a single application it has to run the exact same feature path as
training. Feature engineering scattered across notebook cells cannot guarantee that. So it is
packaged as a picklable `FeaturePipeline`: `fit` once on train, `transform` reused verbatim at
serving time.

The backend never imports `ml.src.train*` or `ml.src.calibrate`. It only touches
`features.build`, `explain` and `reason_codes`. Those three are pure transforms, needed to
unpickle the artifact.

### Feature engineering

`bureau` needs two levels of aggregation. `bureau_balance` only has `SK_ID_BUREAU`, no
`SK_ID_CURR`. So you aggregate monthly history up to each credit line, merge into `bureau`,
then aggregate up to the applicant. The backend follows the same chain when it filters data at
serving time.

`DAYS_LATE` and `PAYMENT_DIFF` in `installments_payments` are computed at row level, before
aggregating. Aggregating the source columns separately and subtracting afterwards loses
everything about individual payments. Both land in the top 30 by SHAP.

Categorical domains are fixed at `fit` and reapplied identically at `transform`. A single
application almost certainly will not contain every category seen during training.
`bureau_balance.STATUS` is hardcoded straight from the data dictionary.

### Model

LightGBM, StratifiedKFold with 5 folds. I trust CV over the public leaderboard. Categoricals
use LightGBM's native handling, no one-hot.

Monotonic constraints on 18 features, each with its domain rationale written next to it in
[`features.yaml`](ml/config/features.yaml). No constraint on `AMT_CREDIT` or `AMT_ANNUITY`.
Borrowing a lot can mean a good customer who got approved for more, or someone drowning in
debt. No clear direction.

Calibration is isotonic, measured with nested CV: split the OOF predictions into 5 parts, fit
the calibrator on 4 and predict the rest. Fitting and predicting on the same data gives you a
meaninglessly good ECE.

## 📊 Results

Joining all 7 tables versus using only the main one:

| | `application_train` only | All 7 tables | |
|---|---|---|---|
| AUC (OOF) | 0.76076 | 0.78757 | +0.02681 |
| Gini | 0.52152 | 0.57514 | +0.05362 |
| Features | 120 | 709 | |

Both runs share the same fold split (same seed, same row order). The difference comes from the
features, not from luck in how the data got split.

### Calibration

| | Before | After isotonic |
|---|---|---|
| Brier | 0.06598 | 0.06589 |
| ECE (10 equal-width bins) | 0.00417 | 0.00060 |
| ECE (50 quantile bins) | 0.00597 | 0.00201 |

I report two binning schemes because equal-width bins flatter the result. 77% of predictions
sit below 0.1, so with 10 equal-width bins a single bin swallows most of the data and the
errors inside it average out. Quantile bins give every bin enough samples. The improvement
holds at the same magnitude under both, so I believe it is real.

Platt scaling made ECE worse. The baseline deliberately skips `scale_pos_weight`, so the raw
model was already fairly well calibrated, and forcing a rigid sigmoid on top only hurt.

### Turning the score into a decision

| Approval rate | PD cutoff | Defaults among approved | Loss reduction | Defaults blocked |
|---|---|---|---|---|
| 50% | 0.0460 | 2.3% | 71.1% | 85.5% |
| 70% | 0.0863 | 3.5% | 56.5% | 69.5% |
| 90% | 0.1887 | 5.6% | 30.8% | 37.8% |
| approve all | | 8.1% | | |

### Fairness

Gender and age are protected attributes under ECOA. At a 70% approval rate, adverse impact
ratio under the 4/5ths rule (below 0.80 is a problem):

- Gender: 0.816, just passes
- Age: 0.446, clearly fails

The model itself is not biased in the statistical sense. The gap between mean predicted PD and
actual bad rate stays under 0.2 percentage points for every group, except the under-25 group
which is off by +1.9pp. It predicts the right level of risk per group. Approval rates differ
because risk genuinely differs: under-25 has a 12.3% bad rate, over-65 only 3.7%.

That does not save you legally. The 4/5ths rule measures impact, not intent. In a real system
that 0.446 would have to go through compliance, and would likely mean dropping age proxies or
setting group-specific cutoffs. More detail in the [model card](ml/artifacts/model_card.md),
section 4.

---

Design notes and settled decisions: [docs/NOTES.md](docs/NOTES.md).
Model details: [model card](ml/artifacts/model_card.md).
