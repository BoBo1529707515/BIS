# -*- coding: utf-8 -*-
"""
Upgraded Behavioral Decoder for touch start (hazard / logistic decoder)

Integrates TWO required upgrades:
(1) Event-stratified cross-validation (no fold with 0 positives):
    - Split POSITIVE event-bins (touch-start labels) into K folds
    - Each fold test set = NONTOUCH bins in [-WIN_PRE_S, 0] around the fold's events
    - Train set = all other NONTOUCH bins
(2) Peri-event averaging (PETH sanity check) for DV_hat and p_hat:
    - Align DV_hat(t) and p_hat(t) to touch-start label times
    - Plot mean ± 95% CI over events

Also outputs:
- ROC and PR curves (all-fit and event-CV)
- Metrics: ROC-AUC, PR-AUC, log loss, Brier
- Shuffle-label sanity check AUC distribution (quick)

Outputs under:
  figs/.../Behavior_decoder__touch_start_hazard__eventCV_plus_PETH/
    - decoder_coefficients.csv
    - decoder_metrics.csv
    - predictions_timeseries.csv
    - roc_allfit.png, roc_eventCV.png
    - pr_allfit.png,  pr_eventCV.png
    - peth_DV_hat.png, peth_p_hat.png
    - auc_shuffle_sanity.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib.util
from scipy.optimize import minimize

# =========================
# 0) CHANGE THIS PATH
# =========================
BASE_SCRIPT_PATH = r"F:\工作文件\RA\python\吸引子\NT-axis vs TOUCH-axis 独立投影 + 全套对比分析（含 touch 背景）.py"

# =========================
# 1) SETTINGS
# =========================
OUT_SUBDIR = "Behavior_decoder__touch_start_hazard__eventCV_plus_PETH"

# label at last NONTOUCH bin just before touch start:
EVENT_SHIFT_S = 0.5     # in units of dt (0.5 means ts - 0.5*dt)
USE_QUADRATIC = True    # include TSLE^2 and TSR^2

# regularized logistic regression
L2_LAMBDA = 1.0
SEED = 0

# Event-stratified CV window (test bins around each touch-start label time)
K_FOLDS = 5
WIN_PRE_S = 10.0        # use nontouch bins in [-10s, 0] before touch start label
WIN_POST_S = 0.0        # usually 0 for "imminent touch"; keep 0 unless you want symmetric windows

# PETH plotting range
PETH_PRE_S = 10.0
PETH_POST_S = 2.0

# sanity check
N_SHUFFLE = 200         # shuffle y labels and recompute all-fit AUC quickly

# numerical stability
DT_FALLBACK = 0.1
EPS = 1e-12

# =========================


def load_base_module(py_path: str):
    spec = importlib.util.spec_from_file_location("base_analysis_module", py_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def extract_touch_bouts(social_intervals):
    starts = np.array([s for (s, _) in social_intervals], dtype=float)
    ends   = np.array([e for (_, e) in social_intervals], dtype=float)
    keep = ends > starts
    return starts[keep], ends[keep]


def build_touch_mask(t, touch_starts, touch_ends):
    m = np.zeros_like(t, dtype=bool)
    for s, e in zip(touch_starts, touch_ends):
        m |= (t >= s) & (t < e)
    return m


def time_since_last_event(t, event_times, t0=None):
    """For each t[i], compute time since last event <= t[i]."""
    event_times = np.sort(np.array(event_times, dtype=float))
    out = np.full_like(t, np.nan, dtype=float)
    j = -1
    prev = float(t0) if (t0 is not None) else np.nan
    for i, ti in enumerate(t):
        while j + 1 < len(event_times) and event_times[j + 1] <= ti:
            j += 1
            prev = event_times[j]
        if np.isfinite(prev):
            out[i] = ti - prev
    return out


def standardize_fit(X):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd == 0] = 1.0
    return mu, sd


def standardize_apply(X, mu, sd):
    return (X - mu) / sd


def sigmoid(z):
    z = np.clip(z, -60, 60)
    return 1.0 / (1.0 + np.exp(-z))


def neg_log_lik_and_grad(w, X, y, l2_lambda):
    z = X @ w
    p = sigmoid(z)
    nll = -np.sum(y * np.log(p + EPS) + (1 - y) * np.log(1 - p + EPS))
    nll += 0.5 * l2_lambda * np.sum(w[1:] ** 2)  # exclude bias

    g = X.T @ (p - y)
    g[1:] += l2_lambda * w[1:]
    return nll, g


def fit_logistic_l2(X, y, l2_lambda=1.0):
    d = X.shape[1]
    w0 = np.zeros(d, dtype=float)

    def f(w):
        nll, _ = neg_log_lik_and_grad(w, X, y, l2_lambda)
        return nll

    def g(w):
        _, grad = neg_log_lik_and_grad(w, X, y, l2_lambda)
        return grad

    res = minimize(f, w0, jac=g, method="L-BFGS-B")
    return res.x, res.success, res.message


# -------- metrics (ROC/PR/logloss) --------

def auc_rank(y_true, scores):
    """ROC-AUC via rank (Mann-Whitney)."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    ok = np.isfinite(scores)
    y_true = y_true[ok]
    scores = scores[ok]

    n1 = int(np.sum(y_true == 1))
    n0 = int(np.sum(y_true == 0))
    if n1 < 2 or n0 < 2:
        return np.nan

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)

    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i + 1
        while j < len(s_sorted) and s_sorted[j] == s_sorted[i]:
            j += 1
        if j - i > 1:
            avg = np.mean(ranks[order[i:j]])
            ranks[order[i:j]] = avg
        i = j

    r1 = np.sum(ranks[y_true == 1])
    auc = (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return float(auc)


def roc_curve_points(y_true, scores):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    ok = np.isfinite(scores)
    y_true = y_true[ok]
    scores = scores[ok]

    uniq = np.unique(scores)
    thr = uniq[::-1]

    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)
    if P == 0 or N == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    tpr, fpr = [], []
    for th in thr:
        pred1 = scores >= th
        TP = np.sum(pred1 & (y_true == 1))
        FP = np.sum(pred1 & (y_true == 0))
        tpr.append(TP / P)
        fpr.append(FP / N)

    fpr = np.array([0.0] + fpr + [1.0])
    tpr = np.array([0.0] + tpr + [1.0])
    return fpr, tpr


def pr_curve_points(y_true, scores):
    """
    Precision-Recall curve points.
    Returns recall, precision (both include endpoints).
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    ok = np.isfinite(scores)
    y_true = y_true[ok]
    scores = scores[ok]

    P = np.sum(y_true == 1)
    if P == 0:
        return np.array([0.0, 1.0]), np.array([1.0, 0.0])

    thr = np.unique(scores)[::-1]
    precision = []
    recall = []

    for th in thr:
        pred1 = scores >= th
        TP = np.sum(pred1 & (y_true == 1))
        FP = np.sum(pred1 & (y_true == 0))
        prec = TP / max(TP + FP, 1)
        rec = TP / P
        precision.append(prec)
        recall.append(rec)

    # add endpoints
    recall = np.array([0.0] + recall + [1.0])
    # baseline precision at recall=0 can be 1 by convention
    precision = np.array([1.0] + precision + [P / len(y_true)])
    return recall, precision


def pr_auc(recall, precision):
    """Area under PR curve using trapezoid on recall axis."""
    order = np.argsort(recall)
    r = recall[order]
    p = precision[order]
    return float(np.trapz(p, r))


def log_loss(y_true, p_hat):
    y_true = np.asarray(y_true).astype(int)
    p_hat = np.asarray(p_hat).astype(float)
    ok = np.isfinite(p_hat)
    y_true = y_true[ok]
    p_hat = np.clip(p_hat[ok], EPS, 1 - EPS)
    return float(-np.mean(y_true * np.log(p_hat) + (1 - y_true) * np.log(1 - p_hat)))


def brier_score(y_true, p_hat):
    y_true = np.asarray(y_true).astype(int)
    p_hat = np.asarray(p_hat).astype(float)
    ok = np.isfinite(p_hat)
    y_true = y_true[ok]
    p_hat = p_hat[ok]
    return float(np.mean((p_hat - y_true) ** 2))


# -------- Event-stratified CV fold builder --------

def build_event_stratified_folds(t_use, y_use, k=5, win_pre_s=10.0, win_post_s=0.0, seed=0):
    """
    Use positive-labeled bins as 'events'.
    Split positive indices into k folds.
    For each fold:
      test_mask = union of bins in [event_time - win_pre_s, event_time + win_post_s]
      train_mask = complement
    """
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_use == 1)[0]
    pos_idx = np.unique(pos_idx)

    perm = rng.permutation(len(pos_idx))
    chunks = np.array_split(perm, k)

    folds = []
    for f in range(k):
        test_pos_idx = pos_idx[chunks[f]]
        test_mask = np.zeros_like(y_use, dtype=bool)
        for pi in test_pos_idx:
            et = t_use[pi]
            test_mask |= (t_use >= (et - win_pre_s)) & (t_use <= (et + win_post_s))
        train_mask = ~test_mask
        folds.append((train_mask, test_mask, test_pos_idx))
    return folds


# -------- PETH utilities --------

def make_rel_grid(dt, pre_s, post_s):
    return np.arange(-pre_s, post_s + 1e-9, dt, dtype=float)


def align_to_events(t, series, event_times, rel_grid):
    """
    Interpolate series(t) onto event_times + rel_grid for each event.
    Returns M (n_events, len(rel_grid))
    """
    series = np.asarray(series, dtype=float)
    M = np.full((len(event_times), len(rel_grid)), np.nan, dtype=float)
    for i, et in enumerate(event_times):
        tgt = et + rel_grid
        ok = (tgt >= t[0]) & (tgt <= t[-1])
        if np.sum(ok) < 5:
            continue
        M[i, ok] = np.interp(tgt[ok], t, series)
    return M


def mean_ci95(M):
    n = np.sum(np.isfinite(M), axis=0)
    mean = np.nanmean(M, axis=0)
    sd = np.nanstd(M, axis=0, ddof=1)
    sem = sd / np.sqrt(np.maximum(n, 1))
    lo = mean - 1.96 * sem
    hi = mean + 1.96 * sem
    lo[n < 2] = np.nan
    hi[n < 2] = np.nan
    return mean, lo, hi, n


def plot_peth(rel, M, title, ylabel, out_png):
    mean, lo, hi, n = mean_ci95(M)
    plt.figure(figsize=(8.5, 3.2))
    plt.axvline(0, linestyle="--", linewidth=1.2)
    ok = np.isfinite(mean)
    plt.plot(rel[ok], mean[ok], linewidth=2.0)
    plt.fill_between(rel[ok], lo[ok], hi[ok], alpha=0.2)
    plt.title(title)
    plt.xlabel("Time from touch start label (s)")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_roc(y_true, scores, title, out_png):
    fpr, tpr = roc_curve_points(y_true, scores)
    aucv = auc_rank(y_true, scores)
    plt.figure(figsize=(5.2, 5.2))
    plt.plot(fpr, tpr, linewidth=2.0, label=f"AUC={aucv:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_pr(y_true, scores, title, out_png):
    recall, precision = pr_curve_points(y_true, scores)
    aucv = pr_auc(recall, precision)
    plt.figure(figsize=(5.2, 5.2))
    plt.plot(recall, precision, linewidth=2.0, label=f"PR-AUC={aucv:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def main():
    rng = np.random.default_rng(SEED)

    base = load_base_module(BASE_SCRIPT_PATH)
    outdir = os.path.join(base.OUTDIR, OUT_SUBDIR)
    os.makedirs(outdir, exist_ok=True)

    # Use neural timebase as binning grid (behavior-only decoder)
    Yz, t, dt = base.load_neural(base.DATA_NPZ)
    if not np.isfinite(dt) or dt <= 0:
        dt = DT_FALLBACK

    social_intervals = base.load_behavior(base.BEH_XLSX, base.REUNION_ABS)
    touch_starts, touch_ends = extract_touch_bouts(social_intervals)

    is_touch = build_touch_mask(t, touch_starts, touch_ends)
    is_nontouch = ~is_touch

    # features
    tsle = time_since_last_event(t, touch_ends, t0=float(base.REUNION_ABS))  # time since last touch end
    tsr = t - float(base.REUNION_ABS)                                       # time since reunion

    # labels y on full grid: 1 at last NONTOUCH bin just before touch start
    y = np.zeros_like(t, dtype=int)
    shift = EVENT_SHIFT_S * dt
    for ts in touch_starts:
        target = float(ts) - shift
        idx = int(np.argmin(np.abs(t - target)))
        if 0 <= idx < len(t) and is_nontouch[idx]:
            y[idx] = 1

    # dataset only on nontouch with finite features
    base_mask = is_nontouch & np.isfinite(tsle) & np.isfinite(tsr)
    idx_use = np.where(base_mask)[0]
    t_use = t[idx_use]
    y_use = y[idx_use]

    f1 = tsle[idx_use]
    f2 = np.log1p(np.maximum(f1, 0.0))
    f3 = tsr[idx_use]

    feats = [f1, f2, f3]
    names = ["TSLE", "log1p_TSLE", "TSR"]

    if USE_QUADRATIC:
        feats += [f1**2, f3**2]
        names += ["TSLE2", "TSR2"]

    X_raw = np.column_stack(feats)

    # standardize (fit on all usable bins; OK for now; later you can do fold-wise standardize)
    mu, sd = standardize_fit(X_raw)
    Xs = standardize_apply(X_raw, mu, sd)

    X = np.column_stack([np.ones(len(Xs)), Xs])  # bias + standardized features
    colnames = ["bias"] + names

    n_pos = int(np.sum(y_use == 1))
    n_neg = int(np.sum(y_use == 0))

    print("=== Behavior decoder dataset ===")
    print(f"dt={dt}, total touch bouts={len(touch_starts)}")
    print(f"nontouch bins used={len(y_use)} | positives={n_pos} | negatives={n_neg}")

    # ---------- Fit on all data ----------
    w, ok, msg = fit_logistic_l2(X, y_use, l2_lambda=L2_LAMBDA)
    print("\n=== Fit (all data) ===")
    print("success:", ok, "|", msg)

    dv_hat_use = X @ w
    p_hat_use = sigmoid(dv_hat_use)

    auc_all = auc_rank(y_use, p_hat_use)
    pr_all = pr_auc(*pr_curve_points(y_use, p_hat_use))
    ll_all = log_loss(y_use, p_hat_use)
    bs_all = brier_score(y_use, p_hat_use)

    print(f"AUC (all-fit): {auc_all:.6f}")
    print(f"PR-AUC (all-fit): {pr_all:.6f}")
    print(f"log loss (all-fit): {ll_all:.6f} | brier: {bs_all:.6f}")

    # ---------- Event-stratified CV ----------
    folds = build_event_stratified_folds(
        t_use, y_use, k=K_FOLDS,
        win_pre_s=WIN_PRE_S, win_post_s=WIN_POST_S,
        seed=SEED
    )

    p_cv_use = np.full_like(p_hat_use, np.nan, dtype=float)
    auc_folds = []
    pr_folds = []
    ll_folds = []
    bs_folds = []

    print("\n=== Event-stratified CV ===")
    for fi, (train_mask, test_mask, test_pos_idx) in enumerate(folds):
        X_tr, y_tr = X[train_mask], y_use[train_mask]
        X_te, y_te = X[test_mask], y_use[test_mask]

        # Fit on train
        w_f, ok_f, _ = fit_logistic_l2(X_tr, y_tr, l2_lambda=L2_LAMBDA)
        p_te = sigmoid(X_te @ w_f)
        p_cv_use[test_mask] = p_te

        auc_f = auc_rank(y_te, p_te)
        pr_f = pr_auc(*pr_curve_points(y_te, p_te))
        ll_f = log_loss(y_te, p_te)
        bs_f = brier_score(y_te, p_te)

        auc_folds.append(auc_f)
        pr_folds.append(pr_f)
        ll_folds.append(ll_f)
        bs_folds.append(bs_f)

        print(f"Fold {fi+1}/{K_FOLDS}: test n={int(test_mask.sum())} pos={int(np.sum(y_te==1))} "
              f"AUC={auc_f:.4f} PR-AUC={pr_f:.4f} logloss={ll_f:.4f}")

    # pooled CV metrics on bins that were actually predicted (union of test windows)
    ok_cv = np.isfinite(p_cv_use)
    y_cv = y_use[ok_cv]
    p_cv = p_cv_use[ok_cv]

    auc_cv = auc_rank(y_cv, p_cv)
    pr_cv = pr_auc(*pr_curve_points(y_cv, p_cv))
    ll_cv = log_loss(y_cv, p_cv)
    bs_cv = brier_score(y_cv, p_cv)

    print("\nCV (event-stratified) pooled:")
    print(f"  AUC={auc_cv:.6f} | PR-AUC={pr_cv:.6f} | logloss={ll_cv:.6f} | brier={bs_cv:.6f}")
    print(f"  coverage: predicted bins {ok_cv.sum()} / {len(y_use)} (only test windows get CV predictions)")

    # ---------- Shuffle sanity check (all-fit AUC should go ~0.5) ----------
    sh_aucs = []
    for _ in range(N_SHUFFLE):
        y_sh = rng.permutation(y_use)
        sh_aucs.append(auc_rank(y_sh, p_hat_use))
    sh_aucs = np.array(sh_aucs, dtype=float)
    df_sh = pd.DataFrame({"auc_shuffle": sh_aucs})
    df_sh.to_csv(os.path.join(outdir, "auc_shuffle_sanity.csv"), index=False, encoding="utf-8-sig")

    print("\nShuffle sanity (all-fit):")
    print(f"  mean={np.nanmean(sh_aucs):.3f} | std={np.nanstd(sh_aucs):.3f} | "
          f"p95={np.nanpercentile(sh_aucs,95):.3f}")

    # ---------- Save coefficients & metrics ----------
    df_coef = pd.DataFrame({"feature": colnames, "weight": w})
    df_coef.to_csv(os.path.join(outdir, "decoder_coefficients.csv"), index=False, encoding="utf-8-sig")

    df_metrics = pd.DataFrame([{
        "dt": dt,
        "n_nontouch_bins_used": len(y_use),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "pos_rate": n_pos / max(len(y_use), 1),
        "L2_LAMBDA": L2_LAMBDA,
        "USE_QUADRATIC": USE_QUADRATIC,
        "EVENT_SHIFT_S": EVENT_SHIFT_S,
        "K_FOLDS": K_FOLDS,
        "WIN_PRE_S": WIN_PRE_S,
        "WIN_POST_S": WIN_POST_S,
        "AUC_allfit": auc_all,
        "PRAUC_allfit": pr_all,
        "logloss_allfit": ll_all,
        "brier_allfit": bs_all,
        "AUC_eventCV_pooled": auc_cv,
        "PRAUC_eventCV_pooled": pr_cv,
        "logloss_eventCV_pooled": ll_cv,
        "brier_eventCV_pooled": bs_cv,
        "AUC_eventCV_folds": ",".join([f"{a:.4f}" for a in auc_folds]),
        "PRAUC_eventCV_folds": ",".join([f"{a:.4f}" for a in pr_folds]),
    }])
    df_metrics.to_csv(os.path.join(outdir, "decoder_metrics.csv"), index=False, encoding="utf-8-sig")

    # ---------- Plots: ROC & PR ----------
    plot_roc(y_use, p_hat_use, "ROC (all-fit) | touch-start hazard decoder", os.path.join(outdir, "roc_allfit.png"))
    plot_pr(y_use, p_hat_use,  "PR (all-fit)  | touch-start hazard decoder", os.path.join(outdir, "pr_allfit.png"))

    plot_roc(y_cv, p_cv, "ROC (event-CV) | touch-start hazard decoder", os.path.join(outdir, "roc_eventCV.png"))
    plot_pr(y_cv, p_cv,  "PR (event-CV)  | touch-start hazard decoder", os.path.join(outdir, "pr_eventCV.png"))

    # ---------- Map predictions back to full time grid ----------
    dv_full = np.full_like(t, np.nan, dtype=float)
    p_full = np.full_like(t, np.nan, dtype=float)
    p_eventcv_full = np.full_like(t, np.nan, dtype=float)

    # fill all-fit on idx_use
    dv_full[idx_use] = dv_hat_use
    p_full[idx_use] = p_hat_use

    # fill event-CV only for bins predicted (subset of idx_use)
    # ok_cv is in "use-space"; map back to absolute indices
    idx_use_pred = idx_use[ok_cv]
    p_eventcv_full[idx_use_pred] = p_cv_use[ok_cv]

    # ---------- PETH sanity check: DV_hat and p_hat aligned to touch-start LABEL time ----------
    # event times = t at y==1 on full grid (these are the label bins)
    ev_idx_full = np.where(y == 1)[0]
    ev_times = t[ev_idx_full]

    rel = make_rel_grid(dt, PETH_PRE_S, PETH_POST_S)

    # Use all-fit traces for PETH (stable; CV trace is sparse by design)
    M_dv = align_to_events(t, dv_full, ev_times, rel)
    M_p  = align_to_events(t, p_full,  ev_times, rel)

    plot_peth(rel, M_dv,
              title=f"DV_hat PETH aligned to touch start label (n={len(ev_times)})",
              ylabel="DV_hat (logit hazard)",
              out_png=os.path.join(outdir, "peth_DV_hat.png"))

    plot_peth(rel, M_p,
              title=f"p_hat PETH aligned to touch start label (n={len(ev_times)})",
              ylabel="p(touch start) hat",
              out_png=os.path.join(outdir, "peth_p_hat.png"))

    # ---------- Export time series ----------
    df_ts = pd.DataFrame({
        "t_abs": t,
        "is_touch": is_touch.astype(int),
        "is_nontouch": is_nontouch.astype(int),
        "touch_start_label_on_nontouch": y.astype(int),
        "TSLE": tsle,
        "TSR": tsr,
        "DV_hat_allfit": dv_full,
        "p_hat_allfit": p_full,
        "p_hat_eventCV": p_eventcv_full,
    })
    df_ts.to_csv(os.path.join(outdir, "predictions_timeseries.csv"), index=False, encoding="utf-8-sig")

    print("\n✅ Saved to:", outdir)
    print("Key outputs:")
    print(" - decoder_coefficients.csv")
    print(" - decoder_metrics.csv")
    print(" - predictions_timeseries.csv")
    print(" - roc_allfit.png / roc_eventCV.png")
    print(" - pr_allfit.png  / pr_eventCV.png")
    print(" - peth_DV_hat.png / peth_p_hat.png")
    print(" - auc_shuffle_sanity.csv")


if __name__ == "__main__":
    main()
