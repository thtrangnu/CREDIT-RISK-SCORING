from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ApplicantSummary(BaseModel):
    sk_id_curr: int
    target: int | None
    code_gender: str
    amt_income_total: float
    amt_credit: float
    name_education_type: str
    name_family_status: str


class ReasonCode(BaseModel):
    feature: str
    label: str
    value: float | str | None
    shap: float
    direction: str
    curated: bool


class ScoreResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    sk_id_curr: int
    pd_uncalibrated: float
    pd_score: float
    risk_tier: str
    model_version: str
    reasons: list[ReasonCode]
    base_value: float
    raw_margin: float
    target_actual: int | None = None


class HistoryEntry(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: int
    sk_id_curr: int
    model_version: str
    pd_score: float
    risk_tier: str
    scored_at: datetime


class HistoryResponse(BaseModel):
    items: list[HistoryEntry]
    total: int
    limit: int
    offset: int


class GlobalImportanceItem(BaseModel):
    feature: str
    label: str
    mean_abs_shap: float
    curated: bool


class CutoffRow(BaseModel):
    """One row of the policy table: how much you approve vs how much defaults."""

    approval_rate: float
    n_approved: int
    pd_cutoff: float
    bad_rate_approved: float
    bad_rate_reduction: float
    bad_captured: float
    expected_loss_index: float


class SegmentRow(BaseModel):
    """One segment (gender / age band) at the reference approval rate."""

    group: str
    n: int
    bad_rate: float
    mean_pd: float
    calibration_gap: float
    auc: float | None = None  # None when the group has a single class, so AUC is undefined
    approval_rate: float


class PolicyBlock(BaseModel):
    reference_approval_rate: float
    table: list[CutoffRow]


class FairnessBlock(BaseModel):
    reference_approval_rate: float
    by_gender: list[SegmentRow]
    by_age_band: list[SegmentRow]
    adverse_impact_ratio_gender: float
    adverse_impact_ratio_age: float


class InsightsResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    global_importance: list[GlobalImportanceItem]
    monotonic_features: list[str]
    # The purely numeric metric block (AUC/Brier/the ECE variants) stays a passthrough
    # dict: it is a direct snapshot of metrics_summary.json and keeps gaining new ECE
    # variants. policy/fairness are TYPED instead, because they are the contract the
    # frontend renders tables from, and because OpenAPI should describe them.
    model_metrics: dict
    policy: PolicyBlock | None = None
    fairness: FairnessBlock | None = None
