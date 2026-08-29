import json

import pandas as pd
import yaml
from fastapi import APIRouter

from ..config import ARTIFACTS_DIR, ROOT_DIR
from ..reason_codes import humanize
from ..schemas import (FairnessBlock, GlobalImportanceItem, InsightsResponse,
                       PolicyBlock)

router = APIRouter(prefix="/api", tags=["insights"])

FEATURES_CONFIG_PATH = ROOT_DIR / "ml" / "config" / "features.yaml"
TOP_N_GLOBAL = 25


@router.get("/insights", response_model=InsightsResponse)
def get_insights() -> InsightsResponse:
    importance_df = pd.read_csv(ARTIFACTS_DIR / "shap_global_importance.csv").head(TOP_N_GLOBAL)
    with open(ARTIFACTS_DIR / "metrics_summary.json") as f:
        metrics = json.load(f)
    with open(FEATURES_CONFIG_PATH) as f:
        monotonic_features = list(yaml.safe_load(f).get("monotone_constraints", {}))

    global_importance = []
    for row in importance_df.itertuples():
        label, curated = humanize(row.feature)
        global_importance.append(
            GlobalImportanceItem(
                feature=row.feature, label=label, mean_abs_shap=float(row.mean_abs_shap), curated=curated
            )
        )

    # policy/fairness only exist after running the newer ml/src/export_metrics_summary.py.
    # Use .get() so the backend still serves older artifacts instead of returning a 500.
    policy = metrics.get("policy")
    fairness = metrics.get("fairness")

    return InsightsResponse(
        global_importance=global_importance,
        monotonic_features=monotonic_features,
        model_metrics=metrics,
        policy=PolicyBlock(**policy) if policy else None,
        fairness=FairnessBlock(**fairness) if fairness else None,
    )
