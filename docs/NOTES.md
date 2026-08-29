# Design notes — Home Credit Default Risk Scoring

> Project context, settled decisions, and the state of each part.
> Read this before changing code so you don't re-propose something already considered.

---

## 1. What this project is (IMPORTANT — read first)

This is a **portfolio project targeting a Data Scientist role** (NOT ML/AI Engineer, NOT
LLM/GenAI). Every decision is weighed from the angle of "how a DS gets evaluated": statistical
rigor, modeling, calibration, explainability, insight, and **a measurable impact number**.

Goal: one strong line on a CV, plus actually learning something. Do NOT over-engineer.

Technical direction: **"production-shaped demo"**. It has the shape of production wherever
that tells a story (clean layer separation, skew prevention, versioning, audit trail), but the
scope is a demo (runs local/Docker, static Kaggle data, single user, no scaling).

**Do NOT build** (even when it sounds reasonable): auth/JWT, monitoring/Prometheus, k8s,
CI/CD, drift detection, automated retraining, LLM/chatbot. These stay in "Future work".

---

## 2. The problem

- Dataset: **Home Credit Default Risk** (Kaggle), 7 relational tables, ~307K applicants in
  `application_train`.
- Task: binary classification, predict default. **Imbalanced** (~8% positive).
- Main model: **LightGBM** + **SHAP** for explainability.
- Domain: lending (regulated), so explainability is a legal requirement rather than
  decoration (adverse action reason codes).

### The 7 tables (grain + keys)
| Table | Grain | Link |
|---|---|---|
| `application_{train,test}` | 1 row / applicant, holds TARGET | SK_ID_CURR (PK) |
| `bureau` | many rows / applicant (credit at other institutions) | SK_ID_CURR; emits SK_ID_BUREAU |
| `bureau_balance` | monthly snapshot of each bureau credit | ONLY has SK_ID_BUREAU, needs 2-level agg |
| `previous_application` | earlier Home Credit applications | SK_ID_CURR; emits SK_ID_PREV |
| `POS_CASH_balance` | monthly POS/cash history | SK_ID_PREV (+ SK_ID_CURR) |
| `installments_payments` | one payment each (finest grain) | SK_ID_PREV (+ SK_ID_CURR) |
| `credit_card_balance` | monthly card statement | SK_ID_PREV (+ SK_ID_CURR) |

Note: every `DAYS_*` column is a NEGATIVE offset from the application date.

---

## 3. Roadmap by block + status

- [x] **Block 0** — EDA, framing. Understand the imbalance, why accuracy is useless here.
- [x] **Block 1** — Baseline + metrics. CODED (see section 6). Baseline AUC (rerun locally) = 0.76076.
- [x] **Block 2** — Multi-table feature engineering. 709 features (from 120). OOF AUC = 0.78757
  (delta = **+0.02681** vs baseline, same fold split for a fair comparison).
- [x] **Block 3** — Monotonic constraints in `features.yaml` (18 features with clear domain
  rationale: EXT_SOURCE_*, DAYS_BIRTH/EMPLOYED, region rating, overdue/DPD from
  bureau + installments + POS).
- [x] **Block 4–5** — Recalibration. Isotonic beats Platt/sigmoid (evaluated with nested
  5-fold on OOF, so not optimistic): ECE 0.00417 → **0.00060**, Brier 0.06598 → 0.06589.
- [x] **Block 6** — Deep SHAP → reason codes. Top global: EXT_SOURCE_2/3/1. 7/18 monotonic
  features land in the top-30 by SHAP (confirms the Block 3 domain reasoning matches what the
  model actually learned).
- [x] **Block 7** — FastAPI backend + MySQL (docker-compose), verified running end to end
  (not just unit tests): loads the real 7 tables, scores a real applicant, writes the audit
  trail to MySQL.
- [x] **Block 8** — React frontend, 3 pages (Score/Insights/History), verified in a real
  browser (dark + light theme, and mobile). Design direction: "Risk Console" — Swiss grid,
  dark-luxury base, Fraunces/JetBrains Mono pairing, bento composition.
- [x] **Block 9** — Cutoff policy + fairness. Turning the score into a decision: at a 70%
  approval rate, the bad rate among approved goes 8.07% → 3.51% (**56.5% less** loss, 69.5%
  of defaults blocked). Segments by protected attribute (ECOA): adverse impact ratio 0.816 for
  gender (passes 4/5ths), **0.446 for age (FAILS)**. The model calibrates evenly across groups
  but the policy impact does not. Surfaced in the model card and the Insights page.
- [x] **Block 10** — README + reproducibility. Full README (results, architecture, how to run,
  real bugs caught, known limitations). `ml/requirements.txt` + `backend/requirements.txt`
  pin versions — `numpy<2.5` is MANDATORY (shap's numba does not support numpy 2.5).
- [ ] **Future work** — auth, monitoring, CI/CD, drift detection, automated retraining
  (deliberately NOT built, see section 1).

---

## 4. Settled design decisions (don't propose the opposite)

1. **The baseline is deliberately clean**: NO `scale_pos_weight` / `is_unbalance`, NO
   imputation in Block 1, so calibration stays clean as a reference point. Reweighting and
   tuning are for later blocks. Don't "fix" the baseline by adding these.
   (Update: hyperparameter tuning has NOT been done and is currently out of scope. It is
   written up under "known limitations" in the README and model card rather than left hanging
   as a promise.)
2. **Default CV scheme**: StratifiedKFold OOF. Trust CV over the public leaderboard.
3. **Hand-written metrics** in `metrics.py`. Do NOT call `sklearn.metrics` for the core
   metrics — this is a CV talking point, it proves the metrics are understood. Sklearn may be
   used in tests to ASSERT correctness, but the production path is hand-written.
4. **Feature engineering must be a REUSABLE pipeline** (not loose notebook code), because the
   backend serving one applicant has to run the EXACT same feature path.
   → exports `feature_pipeline.pkl` + `feature_names.json` (prevents training-serving skew).
5. **Hard layer separation**: `ml/` is standalone and knows nothing about the web. `backend/`
   only LOADS artifacts, never trains, never imports from `ml/src/train.py`. The only boundary
   between the two layers is the `ml/artifacts/` directory.
   → **Clarified while coding Block 7** (applied in `backend/app/scorer.py`): "never trains"
   means the backend never imports `ml.src.train`, `ml.src.train_engineered`, or
   `ml.src.calibrate` (the real training orchestration). The backend IS ALLOWED to import
   `ml.src.features.build.FeaturePipeline`, `ml.src.explain` (SHAP inference), and
   `ml.src.reason_codes`. Those three are the "feature/explain CONTRACT": pure transforms and
   pure functions, no training, and required to unpickle `feature_pipeline.pkl` (pickle needs
   the class definition importable wherever you unpickle).
6. **DB = MySQL**, holding the audit trail (one row per scoring call). Has `model_version` for
   governance. `pd_score` uses DECIMAL, not FLOAT.
7. Serving input: **pick an existing applicant by SK_ID_CURR** (the backend pulls the 7 tables
   itself) rather than making the user fill in 200 fields by hand. File upload is optional
   later.

---

## 5. Target directory layout

```
home-credit-scoring/
├── ml/                          # LAYER 1 — standalone ML pipeline
│   ├── data/{raw,processed}/    # gitignored
│   ├── config/
│   │   ├── features.yaml        # feature declarations + monotonic constraints
│   │   └── params.yaml          # hyperparams, seed, n_folds
│   ├── src/
│   │   ├── metrics.py           # [DONE] hand-written metrics
│   │   ├── features/
│   │   │   ├── aggregations.py         # [DONE] reusable aggs + infer_categories (anti-skew)
│   │   │   ├── bureau.py               # [DONE] 2-level agg (bureau_balance→bureau→curr)
│   │   │   ├── previous_application.py # [DONE]
│   │   │   ├── pos_cash.py             # [DONE]
│   │   │   ├── installments.py         # [DONE] + DAYS_LATE/PAYMENT_DIFF derived
│   │   │   ├── credit_card.py          # [DONE]
│   │   │   └── build.py                # [DONE] orchestrator + FeaturePipeline (fit/transform)
│   │   ├── train.py             # [DONE] OOF StratifiedKFold baseline
│   │   ├── train_engineered.py  # [DONE] Block 2+3: OOF on engineered features + monotonic
│   │   ├── calibrate.py         # [DONE] Block 4-5: isotonic/Platt + final model.txt/calibrator.pkl
│   │   ├── explain.py           # [DONE] Block 6: SHAP global + local (explain_applicant)
│   │   ├── reason_codes.py      # [DONE] feature -> Vietnamese sentence (curated + fallback)
│   │   ├── policy.py            # [DONE] Block 9: cutoff policy + segment/fairness report
│   │   ├── export_metrics_summary.py  # [DONE] metrics + policy + fairness -> metrics_summary.json
│   │   ├── export_model_card.py # [DONE] model_card.md from artifacts (NO retraining)
│   │   └── run_build_features.py      # [DONE] SAFE entry point for build.py (see the note
│   │                                     in features/build.py — do NOT run build.py with -m)
│   ├── artifacts/               # TRAINING OUTPUT = the boundary with backend — ALL [DONE]
│   │   ├── model.txt
│   │   ├── calibrator.pkl
│   │   ├── feature_pipeline.pkl
│   │   ├── feature_names.json
│   │   ├── shap_global_importance.csv
│   │   ├── metrics_summary.json  # + policy/fairness block (Block 9)
│   │   └── model_card.md         # generated by export_model_card.py
│   ├── notebooks/
│   ├── requirements.txt         # [DONE] pinned — numpy<2.5 MANDATORY (numba/shap)
│   └── tests/                   # 55 tests (Block 2-9)  [51 run without shap installed]
├── backend/                     # LAYER 2 — FastAPI, only loads artifacts — [DONE]
│   ├── app/
│   │   ├── main.py              # lifespan: load RawTableStore + Scorer once at startup
│   │   ├── config.py            # reads .env: DATABASE_URL, MODEL_VERSION, CORS_ORIGINS
│   │   ├── schemas.py           # Pydantic (ScoreResponse carries base_value/raw_margin)
│   │   ├── scorer.py            # Scorer: pipeline.transform → predict → calibrate → SHAP
│   │   ├── reason_codes.py      # re-exports ml.src.reason_codes (see layer boundary note)
│   │   ├── dependencies.py      # get_store/get_scorer (FastAPI Depends, easy to override)
│   │   ├── data/{source.py,assembler.py}   # pull 7 tables by sk_id_curr (application_train pool)
│   │   ├── db/{session.py,models.py,crud.py}  # SQLAlchemy + PyMySQL, ScoringHistory
│   │   └── routers/{score.py,history.py,insights.py}
│   ├── migrations/              # Alembic — 0001_create_scoring_history
│   └── tests/                   # 17 tests (real data filtered to 2 SK_ID_CURR, no mocks)
├── frontend/                    # LAYER 3 — React (Vite, JS/JSX) — [DONE]
│   └── src/
│       ├── pages/{ScorePage,InsightsPage,HistoryPage}.jsx
│       ├── components/
│       │   ├── cutoff-table/CutoffTable.jsx               # Block 9: approval/risk trade-off
│       │   ├── segment-table/SegmentTable.jsx             # Block 9: segments + AIR badge
│       │   ├── applicant-selector/ApplicantSelector.jsx   # search-as-you-type
│       │   ├── score-gauge/ScoreGauge.jsx                 # radial gauge, animated, 5x base rate
│       │   ├── reason-code-list/ReasonCodeList.jsx
│       │   ├── shap-waterfall/ShapWaterfall.jsx           # hand SVG, base→feature→"rest"→result
│       │   ├── history-table/HistoryTable.jsx
│       │   ├── nav/NavBar.jsx                             # dark/light theme toggle
│       │   └── ui/{RiskBadge,StatTile}.jsx
│       ├── context/ScoreContext.jsx   # shares applicant/scoreResult between Score ↔ Insights
│       ├── styles/{tokens.css,typography.css,global.css}   # design system: see section 8
│       └── api/client.js
├── docker-compose.yml           # mysql only (backend/frontend run dev servers, see section 6)
└── README.md
```

Note: the UI and the reason codes are intentionally kept in Vietnamese. Docs and code comments
are in English.

---

## 6. Current state of the code

### `ml/src/metrics.py` — CODED
Hand-written, three groups:
- Ranking: AUC (Mann–Whitney rank, O(n log n), ties get the average rank), Gini (=2·AUC−1),
  KS (=max|TPR−FPR|).
- Calibration/probability: PR-AUC (step-sum / average precision), Brier (+ Murphy
  decomposition), ECE (uniform and quantile binning).
- Operational credit: partial AUC (McClish standardized), TPR@FPR, decile table.

### `ml/src/train.py` — CODED
LightGBM baseline, StratifiedKFold OOF, LightGBM's native categoricals, NO reweighting, NO
imputation. Prints both metric families. OOF AUC (rerun locally, seed=42) = **0.76076**.
(The "~0.74" in an earlier draft was a stale number from a different environment. 0.76076 is
the official reference because it comes from the same machine, seed, and fold split as Block 2.)

### `ml/src/features/` — CODED (Block 2)
- `aggregations.py`: `aggregate_numeric`, `aggregate_categorical`, `group_size`, `merge_all`,
  `infer_categories`. The categorical agg takes a fixed `categories` mapping from fit time,
  which is required to prevent training-serving skew (a single applicant at serving time may
  be missing categories seen during training).
- `bureau.py`: 2-level agg exactly as designed — `bureau_balance` groupby SK_ID_BUREAU
  (level 1, the STATUS domain is hardcoded because it's a closed enum per the data dictionary)
  → merge into `bureau` → groupby SK_ID_CURR (level 2).
- `previous_application.py`, `pos_cash.py`, `credit_card.py`: aggregate straight to
  SK_ID_CURR (these three already carry SK_ID_CURR, no 2-level needed). The `365243` sentinel
  in `previous_application`'s DAYS_* columns is cleaned to NaN before aggregating.
- `installments.py`: adds 2 derived columns at row level BEFORE aggregating — `DAYS_LATE` and
  `PAYMENT_DIFF`. These are the strongest repayment-behaviour signals and cannot be
  reconstructed by aggregating the source columns separately.
- `build.py`: `FeaturePipeline` (`fit`/`transform`), picklable. This is the
  `feature_pipeline.pkl` artifact the backend loads. `transform()` also cleans the
  `DAYS_EMPLOYED == 365243` sentinel (~18% of rows, mostly retirees) in `application` to NaN.
- Result: 709 features (from 120 baseline). **OOF AUC = 0.78757** (delta **+0.02681**, same
  fold split as the baseline).
- Tests: `ml/tests/test_aggregations.py`, `test_bureau.py`, `test_build.py` — including tests
  specifically for training-serving skew, which caught a real bug during development (the
  `bureau_balance` STATUS domain was being inferred dynamically instead of fixed).

### `ml/src/train_engineered.py` — CODED (Block 2 retrain + Block 3)
Loads the engineered features (cached at `ml/artifacts/train_features.parquet`), maps
`features.yaml.monotone_constraints` into an array matching X's column order for LightGBM
(`build_monotone_constraints`, which forces 0 if a constraint was accidentally declared for a
categorical column). Compares AUC directly against `oof_baseline.npy` (same seed/n_folds means
the same fold split).

### `ml/config/features.yaml` — CODED (Block 3)
18 features with monotonic constraints, each with its domain rationale written alongside
(EXT_SOURCE_*, DAYS_BIRTH, DAYS_EMPLOYED, AMT_INCOME_TOTAL, REGION_RATING_CLIENT[_W_CITY],
bureau overdue, installments late/underpay, POS DPD). AMT_CREDIT/AMT_ANNUITY are deliberately
left unconstrained because their real relationship with risk is ambiguous and non-monotonic.

### `ml/src/calibrate.py` — CODED (Block 4-5)
`nested_calibrate`: evaluates isotonic/Platt with K-fold over the OOF predictions themselves
(fit the calibrator on K-1 parts, predict the rest) so the estimate isn't falsely optimistic.
Isotonic wins: ECE 0.00417 → 0.00060, Brier 0.06598 → 0.06589. Platt/sigmoid makes ECE WORSE,
because the raw model is already fairly well calibrated (no `scale_pos_weight`), so forcing a
rigid sigmoid on top doesn't fit.
Also trains the final model on 100% of the data (`num_boost_round` chosen via a separate
early-stopping split) → saves `model.txt` and `calibrator.pkl`.
Tests: `ml/tests/test_calibrate.py`, including a leakage test for the nested calibration.

### `ml/src/explain.py` + `reason_codes.py` — CODED (Block 6)
- `explain_applicant(model, feature_names, row, top_k)` reindexes `row` by `feature_names`
  BEFORE predicting/explaining. **Real bug caught here**: `Booster.predict()` matches
  DataFrame columns BY POSITION, not by name (reorder the columns and you get a different
  prediction with NO error). `model.feature_name()` isn't trustworthy either, since LightGBM
  sanitizes column names containing special characters when saving `model.txt`. So every
  place that uses the model (explain.py, backend/scorer.py) must reindex by
  `feature_names.json`.
- `reason_codes.py`: ~25 strong-signal features curated by hand (matching the 18 monotonic
  ones plus top SHAP), with a readable fallback (marked `curated=False`) for the rest.
  `build_reason_codes` handles string values safely (raw categorical columns like
  CODE_GENDER/ORGANIZATION_TYPE can absolutely land in a given applicant's top-K SHAP) as well
  as NaN. **Real bug caught here**: calling `float()` unconditionally on the value crashes
  when the top feature is categorical.
- Global: 7/18 monotonic features land in the top-30 SHAP importance (EXT_SOURCE_1/2/3,
  DAYS_BIRTH, DAYS_EMPLOYED, INSTAL_DAYS_LATE_MAX, INSTAL_PAYMENT_DIFF_MEAN).
- Tests: `test_explain.py`, `test_reason_codes.py`.

### `backend/` — CODED (Block 7)
- `scorer.py`: `Scorer.score()` returns `base_value`/`raw_margin` alongside the top-K reasons,
  so the frontend waterfall adds up correctly — "everything else" = raw_margin − base_value −
  sum(top-K shap). The SHAP `TreeExplainer` is built once at startup, since walking ~1400
  trees is the expensive part and it used to be rebuilt on every request.
- `data/source.py`: loads the real 7 tables ONCE at startup and holds them in RAM for the
  process lifetime. They are NOT loaded into MySQL; MySQL is only the audit trail. The
  applicant pool is `application_train` (it has the real TARGET, useful for comparison on the
  Insights page).
- **Serious real bug caught during testing**: `feature_pipeline.pkl` was originally pickled
  while `ml.src.features.build` ran as `__main__` (via `python -m ml.src.features.build`), so
  `FeaturePipeline.__module__` was recorded as `"__main__"` and the artifact could NOT be
  unpickled from any other entry point (backend, pytest, ...). Fix: `build.py` no longer has
  an `if __name__ == "__main__"` block, and `ml/src/run_build_features.py` (which only imports
  `main()` and defines no classes) is the entry point for rebuilding the artifact. There is a
  regression test for this.
- Tests (`backend/tests/`): use REAL data filtered to 2 real SK_ID_CURR values (100002 has
  full history and TARGET=1; 100006 has no bureau history and TARGET=0). **No mocks** — two
  attempts at hand-building DataFrames both ended up missing columns (categorical first, then
  numeric) compared to the schema the real pipeline was fit on, so filtering the real CSVs was
  the safer route. `test_api.py` uses in-memory SQLite via `StaticPool` (by default every new
  connection to `sqlite:///:memory:` is a separate empty database).
- Verified actually running (not just unit tests): `uvicorn backend.app.main:app`, loading
  2.5GB of data in ~10-50s, scoring applicant 100002 → PD 43.48%, risk tier "Cao", written to
  MySQL with the right DECIMAL type.

### `ml/src/policy.py` + `export_model_card.py` — CODED (Block 9)
- `policy.py`: `approve_mask` (cuts by quantile rather than an absolute PD threshold — holding
  the approval rate fixed is the fair way to compare models, and it's close to how a risk team
  actually operates), `cutoff_table`, `segment_report`, `adverse_impact_ratio` (4/5ths rule,
  skipping groups under 1000 rows so noise doesn't drag the ratio), `age_bands`.
- `metrics.py` extended with `expected_calibration_error(..., strategy="quantile")`. Reason:
  77% of predictions are below 0.1, so uniform-10 dumps nearly everything into one bin and
  errors in OPPOSITE directions inside that bin cancel out. There's a test constructing
  exactly that situation (uniform reports 0, quantile reports 0.02).
- `export_model_card.py`: the model card is split out of `calibrate.py`. The template used to
  live inline inside the training function, so changing one sentence meant retraining. Now the
  card is a pure function of the artifacts.
- **Real bug caught**: `json.dump` writes `NaN`, which is not valid JSON, when a segment's AUC
  is undefined. Python reads it back fine so it failed silently in the ML layer, but the
  browser's `JSON.parse` throws and the Insights page goes blank. Fix: `json_safe()` maps
  NaN→None, plus `allow_nan=False`.
- Tests: `test_policy.py`, `test_metrics_ece.py`.

### `frontend/` — CODED (Block 8)
- Design direction **"Risk Console"**: Swiss/International grid, dark-luxury base (light-first
  was considered but dark suits a quant/risk desk aesthetic better), Fraunces (display numbers
  and headings) + JetBrains Mono (data/labels), bento composition on Insights. OKLCH palette,
  three semantic risk colours (teal-green / amber / coral-red) kept hue-separate from the gold
  accent (which is for actions, not status). Both dark and light themes are complete; light is
  not an afterthought.
  ScoreGauge displays a 0–40% scale (5x the 8.07% default rate) instead of a linear 0–100%,
  because most applicants have low PD and a linear scale leaves the gauge nearly empty with no
  discriminating power. A population-average tick is still shown so the scale isn't misleading.
  ShapWaterfall is hand-written SVG (no chart library): base_value → each feature →
  "everything else" (pooled) → result, summing exactly to raw_margin.
- Verified in a real browser (not just code review): search applicant → score → SPA nav to
  Insights preserves state (React Context, lost on a full reload, which is by design) →
  waterfall shows the right data → History records correctly. Both themes, and mobile at 375px.
- **Real bug caught and fixed during UI verification**: the SHAP value label in the waterfall
  used the wrong bar edge (always x2/max instead of the edge matching the sign), so labels
  overlapped the bar; long bars (e.g. "-1.007") pushed the label onto the feature name in the
  left column. Fixed by placing the label INSIDE the bar when it's wide enough.
- `docker-compose.yml` only defines `mysql`. Backend and frontend run their dev servers
  directly (`uvicorn`, `npm run dev`) for fast hot-reload during development; Dockerfiles for
  those two services are NOT written (unnecessary for a local demo, could be added if a
  "one command runs everything" setup is ever wanted).

---

## 7. Frontend design system (Block 8)

Direction name: **"Risk Console"**. Spiritual reference: a serious fintech data console
(a refined Bloomberg terminal), NOT a generic dashboard-by-numbers.

- **Palette** (`frontend/src/styles/tokens.css`, OKLCH): warm near-black ink background
  (dark) / warm near-white cream (light); gold accent for actions; three semantic risk colours
  with hues separated from the accent (teal-green low / amber medium / coral-red high).
- **Typography**: Fraunces (serif, large numbers + headings) + JetBrains Mono (data and
  technical labels, `tabular-nums`) + system-ui (body, which doesn't count against the limit
  since it isn't a downloaded font). Both are loaded from the Google Fonts CDN, acceptable for
  a local demo; self-hosting is the move if this is ever productionized.
- **Custom components rather than a generic chart library**: ScoreGauge (hand-written radial
  arc SVG, animated with `requestAnimationFrame` + easing), ShapWaterfall (hand-written
  waterfall SVG), CutoffTable, SegmentTable.
- **Motion**: compositor-friendly only (transform/opacity), respects `prefers-reduced-motion`.
- Not done: Playwright visual regression (NOT in scope for the current demo, could be added
  later).
