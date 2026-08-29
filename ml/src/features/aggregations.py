"""Hàm aggregation tái dùng cho mọi bảng phụ (bureau, previous_application, ...).

Quy ước đặt tên cột output: {prefix}_{col}_{FUNC}, luôn upper-case phần FUNC/value
để tránh đụng tên khi merge nhiều bảng vào application.
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
    """Agg mọi cột numeric (trừ group_key/exclude) theo group_key, 1 dòng/group."""
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
    """Chụp lại domain giá trị của mỗi cột categorical LÚC FIT (train).

    Bắt buộc dùng lại domain này ở transform (serve) để pd.get_dummies luôn
    sinh đúng bộ cột dummy — tránh training-serving skew khi 1 applicant lẻ
    lúc serve không có đủ mọi category so với lúc train.
    """
    return {c: sorted(df[c].dropna().unique().tolist()) for c in cat_cols}


def aggregate_categorical(
    df: pd.DataFrame,
    group_key: str,
    prefix: str,
    cat_cols: list[str] | None = None,
    categories: dict[str, list] | None = None,
) -> pd.DataFrame:
    """Tỉ lệ mỗi category trong group (one-hot rồi lấy mean = share theo group).

    `categories`: domain cố định lúc fit (xem `infer_categories`). Nếu không
    truyền, domain được suy ra ngay từ `df` hiện tại (chỉ dùng cho khám phá/test).
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
    """Số dòng mỗi group (vd: số khoản bureau, số lần vay trước)."""
    s = df.groupby(group_key).size().rename(f"{prefix}_{out_name}")
    return s.to_frame()


def merge_all(base: pd.DataFrame, key: str, *others: pd.DataFrame) -> pd.DataFrame:
    """Merge nhiều bảng đã agg (index=key) vào base theo left join, giữ 1 dòng/key."""
    out = base
    for other in others:
        out = out.merge(other, how="left", left_on=key, right_index=True)
    return out
