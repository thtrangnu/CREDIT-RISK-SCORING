import numpy as np
import pandas as pd


def _rank_average(x):
    """Rank 1-based, tie thì lấy rank trung bình (giống scipy.rankdata)."""
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
    """Gini = 2*AUC - 1. Ngôn ngữ chung của dân risk."""
    return 2.0 * roc_auc(y_true, y_score) - 1.0


def ks_statistic(y_true, y_score):
    """KS = max |TPR - FPR| khi quét threshold từ score cao xuống thấp."""
    y_true = np.asarray(y_true, dtype=int)
    order = np.argsort(-np.asarray(y_score, dtype=float), kind="mergesort")
    y = y_true[order]
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    tpr = np.cumsum(y) / n_pos
    fpr = np.cumsum(1 - y) / n_neg
    return float(np.max(np.abs(tpr - fpr)))


def average_precision(y_true, y_score):
    """PR-AUC dạng step-sum: AP = sum (R_n - R_{n-1}) * P_n."""
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
    """MSE giữa prob và nhãn — đo calibration + sharpness."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(y_true, y_prob, n_bins=10, strategy="uniform"):
    """ECE = sum (|bin|/N) * |acc(bin) - conf(bin)|.

    `strategy`:
      "uniform"  - bin rộng bằng nhau trên [0, 1]. Cách hay được report nhất,
                   NHƯNG dễ gây ảo giác trên bài toán lệch: ở đây 77% prediction
                   nằm dưới 0.1, nên bin đầu tiên nuốt gần hết dữ liệu và các
                   sai lệch bên trong nó bị trung bình hoá mất.
      "quantile" - bin bằng nhau về SỐ LƯỢNG mẫu (equal-frequency). Chặt hơn cho
                   phân phối lệch vì mọi bin đều có đủ mẫu để so acc vs conf.

    Luôn report cả hai khi kết luận về calibration (xem model_card.md).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        edges = np.quantile(y_prob, np.linspace(0.0, 1.0, n_bins + 1))
    else:
        raise ValueError(f"strategy phải là 'uniform' hoặc 'quantile', nhận {strategy!r}")

    # digitize trên các cạnh TRONG -> index bin 0..n_bins-1; clip cho trường hợp
    # quantile có cạnh trùng nhau (nhiều giá trị giống hệt).
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
    """pAUC vùng FPR in [0, max_fpr], chuẩn hoá McClish -> [0.5, 1]."""
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
    """Bắt được bao nhiêu % bad khi chấp nhận target_fpr % good bị từ chối oan."""
    y_true = np.asarray(y_true, dtype=int)
    order = np.argsort(-np.asarray(y_score, dtype=float), kind="mergesort")
    y = y_true[order]
    tpr = np.r_[0.0, np.cumsum(y) / y.sum()]
    fpr = np.r_[0.0, np.cumsum(1 - y) / (len(y) - y.sum())]
    return float(np.interp(target_fpr, fpr, tpr))


def decile_table(y_true, y_score, n=10):
    """Bảng gains/decile kiểu risk: lift, capture, KS theo từng decile."""
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
