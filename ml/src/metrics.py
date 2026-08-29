import numpy as np
import pandas as pd


def _rank_average(x):
    """1-based ranks; ties get the average rank (same as scipy.rankdata)."""
    x = np.asarray(x, dtype=float)
    sorter = np.argsort(x, kind="mergesort")
    x_sorted = x[sorter]
    obs = np.r_[True, x_sorted[1:] != x_sorted[:-1]]
    dense = obs.cumsum()
    bounds = np.r_[np.nonzero(obs)[0], len(x)]
    counts = np.diff(bounds)
    ends = np.cumsum(counts)
    starts = ends - counts + 1
    avg = (starts + ends) / 2.0
    ranks_sorted = avg[dense - 1]
    ranks = np.empty_like(ranks_sorted)
    ranks[sorter] = ranks_sorted
    return ranks


def roc_auc(y_true, y_score):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _rank_average(y_score)
    return float((r[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0)
                 / (n_pos * n_neg))


def gini(y_true, y_score):
    """Gini = 2*AUC - 1. The lingua franca of risk teams."""
    return 2.0 * roc_auc(y_true, y_score) - 1.0


def ks_statistic(y_true, y_score):
    """KS = max |TPR - FPR| sweeping the threshold from high score to low."""
    y_true = np.asarray(y_true, dtype=int)
    order = np.argsort(-np.asarray(y_score, dtype=float), kind="mergesort")
    y = y_true[order]
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    tpr = np.cumsum(y) / n_pos
    fpr = np.cumsum(1 - y) / n_neg
    return float(np.max(np.abs(tpr - fpr)))


def average_precision(y_true, y_score):
    """PR-AUC as a step sum: AP = sum (R_n - R_{n-1}) * P_n."""
    y_true = np.asarray(y_true, dtype=int)
    order = np.argsort(-np.asarray(y_score, dtype=float), kind="mergesort")
    y = y_true[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / (tp + fp)
    recall = tp / y.sum()
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def brier_score(y_true, y_prob):
    """MSE between probability and label. Captures calibration and sharpness."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(y_true, y_prob, n_bins=10, strategy="uniform"):
    """ECE = sum (|bin|/N) * |acc(bin) - conf(bin)|.

    `strategy`:
      "uniform"  - equal-width bins over [0, 1]. The most commonly reported form,
                   BUT misleading on a skewed problem: here 77% of predictions sit
                   below 0.1, so the first bin swallows most of the data and the
                   errors inside it average each other out.
      "quantile" - equal-frequency bins. Stricter on a skewed distribution because
                   every bin holds enough samples to compare accuracy vs confidence.

    Report both whenever drawing a conclusion about calibration (see model_card.md).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        edges = np.quantile(y_prob, np.linspace(0.0, 1.0, n_bins + 1))
    else:
        raise ValueError(f"strategy must be 'uniform' or 'quantile', got {strategy!r}")

    # digitize on the INNER edges -> bin index 0..n_bins-1; clip handles quantile
    # edges that coincide (many identical values).
    idx = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, n_bins - 1)
    N, ece = len(y_prob), 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        conf = y_prob[m].mean()
        acc = y_true[m].mean()
        ece += (m.sum() / N) * abs(acc - conf)
    return float(ece)


def _trapz(y, x):
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))


def partial_auc(y_true, y_score, max_fpr=0.2, standardized=True):
    """pAUC over FPR in [0, max_fpr], McClish-standardized to [0.5, 1]."""
    y_true = np.asarray(y_true, dtype=int)
    order = np.argsort(-np.asarray(y_score, dtype=float), kind="mergesort")
    y = y_true[order]
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    tpr = np.r_[0.0, np.cumsum(y) / n_pos]
    fpr = np.r_[0.0, np.cumsum(1 - y) / n_neg]
    stop = np.searchsorted(fpr, max_fpr, side="right")
    if stop < len(fpr):
        t = np.interp(max_fpr, fpr[stop - 1:stop + 1], tpr[stop - 1:stop + 1])
        fpr = np.r_[fpr[:stop], max_fpr]
        tpr = np.r_[tpr[:stop], t]
    pauc = _trapz(tpr, fpr)
    if not standardized:
        return pauc
    min_area = 0.5 * max_fpr ** 2
    max_area = max_fpr
    return 0.5 * (1.0 + (pauc - min_area) / (max_area - min_area))


def tpr_at_fpr(y_true, y_score, target_fpr=0.1):
    """What share of bads is caught if target_fpr of goods are wrongly rejected."""
    y_true = np.asarray(y_true, dtype=int)
    order = np.argsort(-np.asarray(y_score, dtype=float), kind="mergesort")
    y = y_true[order]
    tpr = np.r_[0.0, np.cumsum(y) / y.sum()]
    fpr = np.r_[0.0, np.cumsum(1 - y) / (len(y) - y.sum())]
    return float(np.interp(target_fpr, fpr, tpr))


def decile_table(y_true, y_score, n=10):
    """Risk-style gains table: lift, capture and KS per decile."""
    df = pd.DataFrame({"y": np.asarray(y_true, dtype=int),
                       "p": np.asarray(y_score, dtype=float)})
    df = df.sort_values("p", ascending=False).reset_index(drop=True)
    df["decile"] = (pd.Series(np.floor(np.arange(len(df)) / len(df) * n).astype(int) + 1)
                    .clip(upper=n).values)

    base_rate = df["y"].mean()
    tot_bad, tot_good = df["y"].sum(), (1 - df["y"]).sum()
    g = (df.groupby("decile")
           .agg(n=("y", "size"), bad=("y", "sum"),
                min_p=("p", "min"), max_p=("p", "max")))
    g["bad_rate"] = g["bad"] / g["n"]
    g["lift"] = g["bad_rate"] / base_rate
    g["cum_bad"] = g["bad"].cumsum()
    g["capture"] = g["cum_bad"] / tot_bad
    g["cum_good"] = (g["n"] - g["bad"]).cumsum()
    g["ks"] = (g["cum_bad"] / tot_bad - g["cum_good"] / tot_good).abs() * 100
    return g.reset_index().round(4)
