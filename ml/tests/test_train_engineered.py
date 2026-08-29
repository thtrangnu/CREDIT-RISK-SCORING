from ml.src.train_engineered import build_monotone_constraints


def test_declared_features_get_mapped_constraint_in_column_order():
    columns = ["A", "B", "C", "D"]
    declared = {"B": -1, "D": 1}
    out = build_monotone_constraints(columns, cat_cols=set(), declared=declared)
    assert out == [0, -1, 0, 1]


def test_undeclared_features_default_to_zero():
    columns = ["X", "Y"]
    out = build_monotone_constraints(columns, cat_cols=set(), declared={})
    assert out == [0, 0]


def test_categorical_feature_constraint_is_forced_to_zero():
    """LightGBM has no meaningful monotonic constraint for categorical splits."""
    columns = ["EXT_SOURCE_1", "NAME_CONTRACT_TYPE"]
    declared = {"EXT_SOURCE_1": -1, "NAME_CONTRACT_TYPE": 1}
    out = build_monotone_constraints(columns, cat_cols={"NAME_CONTRACT_TYPE"}, declared=declared)
    assert out == [-1, 0]
