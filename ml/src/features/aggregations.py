"""Reusable aggregation helpers for every satellite table (bureau, previous_application, ...).

Output column naming convention: {prefix}_{col}_{FUNC}, with the FUNC/value part always
upper-cased to avoid collisions when several tables get merged into application.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_FUNCS = ("count", "mean", "sum", "min", "max", "std")


def aggregate_numeric(
    df: pd.DataFrame,
    group_key: str,
    prefix: str,
    funcs: tuple[str, ...] = NUMERIC_FUNCS,
    exclude: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Aggregate every numeric column (except group_key/exclude) by group_key, one row per group."""
    num_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c != group_key and c not in exclude
    ]
    if not num_cols:
        return df[[group_key]].drop_duplicates().set_index(group_key)

    agg = df.groupby(group_key)[num_cols].agg(list(funcs))
    agg.columns = [f"{prefix}_{col}_{func.upper()}" for col, func in agg.columns]
    return agg


def infer_categories(df: pd.DataFrame, cat_cols: list[str]) -> dict[str, list]:
    """Capture the value domain of each categorical column AT FIT TIME (train).

    Reusing this domain at transform (serving) time is mandatory so pd.get_dummies always
    produces the same set of dummy columns. Without it you get training-serving skew, since
    a single applicant at serving time will not carry every category seen during training.
    """
    return {c: sorted(df[c].dropna().unique().tolist()) for c in cat_cols}


def aggregate_categorical(
    df: pd.DataFrame,
    group_key: str,
    prefix: str,
    cat_cols: list[str] | None = None,
    categories: dict[str, list] | None = None,
) -> pd.DataFrame:
    """Share of each category within a group (one-hot then mean = per-group share).

    `categories`: the domain fixed at fit time (see `infer_categories`). If omitted, the
    domain is inferred from the current `df`, which is only appropriate for exploration
    and tests.
    """
    if cat_cols is None:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        cat_cols = [c for c in cat_cols if c != group_key]
    if not cat_cols:
        return df[[group_key]].drop_duplicates().set_index(group_key)

    work = df[[group_key] + cat_cols].copy()
    for c in cat_cols:
        cats = categories[c] if categories and c in categories else sorted(work[c].dropna().unique().tolist())
        work[c] = pd.Categorical(work[c], categories=cats)

    dummies = pd.get_dummies(work, columns=cat_cols)
    agg = dummies.groupby(group_key).mean()
    agg.columns = [f"{prefix}_{col}_SHARE" for col in agg.columns]
    return agg


def group_size(df: pd.DataFrame, group_key: str, prefix: str, out_name: str = "COUNT") -> pd.DataFrame:
    """Row count per group (e.g. number of bureau credits, number of previous loans)."""
    s = df.groupby(group_key).size().rename(f"{prefix}_{out_name}")
    return s.to_frame()


def merge_all(base: pd.DataFrame, key: str, *others: pd.DataFrame) -> pd.DataFrame:
    """Left-join several aggregated tables (index=key) onto base, keeping one row per key."""
    out = base
    for other in others:
        out = out.merge(other, how="left", left_on=key, right_index=True)
    return out
