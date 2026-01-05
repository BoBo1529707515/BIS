# -*- coding: utf-8 -*-
"""
ROC analysis for short-time changes around transitions:
1) touch END:  touch -> nontouch (align touch bout end)
2) touch START: nontouch -> touch (align touch bout start)

Two neuron groups (same as your current pipeline):
- non-touch decreasing neurons (from NPZ_NT dec mask)
- touch decreasing neurons (from NPZ_T  dec mask)

ROC setup (per window):
- class 0: pre-window mean
- class 1: post-window mean
AUC < 0.5 means post < pre (a "drop" after transition)

Key rigor:
- touch END baseline is truncated by touch bout duration:
    pre = [-min(win, touch_dur), 0)
- touch START baseline is truncated by preceding NONTOUCH gap:
    pre = [-min(win, prev_nontouch_gap), 0)
- require baseline length >= PRE_MIN_S (default 0.2s)

Outputs:
- CSV summary with AUC, bootstrap CI, permutation p-value
- ROC curve plots (both groups overlaid) for each transition & window
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib.util

# =========================
# 0) CHANGE THIS PATH
# =========================
BASE_SCRIPT_PATH = r"F:\工作文件\RA\python\吸引子\NT-axis vs TOUCH-axis 独立投影 + 全套对比分析（含 touch 背景）.py"

# =========================
# 1) SETTINGS
# =========================
OUT_SUBDIR = "ROC_short_windows__touch_start_and_end"

REL_MIN, REL_MAX = -3.0, 3.0   # we only need ±3s context for windowing/interp
WINDOWS_S = [0.2, 0.5, 1.0, 3.0]

PRE_MIN_S = 0.2       # allow short bouts/gaps; you can set 0.3 or 0.5 if you want stricter
MIN_FRAC = 0.6        # require >=60% samples present in each window
N_BOOT = 2000         # bootstrap resamples for CI
N_PERM = 5000         # permutation swaps per bout for p-value
SEED = 0

# =========================


def load_base_module(py_path: str):
    spec = importlib.util.spec_from_file_location("base_analysis_module", py_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def extract_touch_bouts(social_intervals):
    """social_intervals: list of (start_abs, end_abs) for touch bouts"""
    starts = np.array([s for (s, _) in social_intervals], dtype=float)
    ends   = np.array([e for (_, e) in social_intervals], dtype=float)
    keep = ends > starts
    return starts[keep], ends[keep]


def compute_prev_nontouch_gaps(touch_starts, touch_ends, t_start_ref=None):
    """
    For each touch start, compute preceding NONTOUCH gap:
      gap_i = touch_start_i - prev_touch_end
    For the first bout:
      if t_start_ref is provided, gap_0 = touch_start_0 - t_start_ref
      else gap_0 = NaN
    """
    order = np.argsort(touch_starts)
    ts = touch_starts[order]
    te = touch_ends[order]

    gaps = np.full_like(ts, np.nan, dtype=float)
    for i in range(len(ts)):
        if i == 0:
            if t_start_ref is not None:
                gaps[i] = float(ts[i] - t_start_ref)
        else:
            gaps[i] = float(ts[i] - te[i - 1])

    gaps_back = np.full_like(gaps, np.nan, dtype=float)
    gaps_back[order] = gaps
    return gaps_back


def make_rel_grid(dt, rel_min, rel_max):
    return np.arange(rel_min, rel_max + 1e-9, dt, dtype=float)


def interp_trace_to_grid(t_abs, y, e_abs, rel_grid):
    tgt_t = e_abs + rel_grid
    inside = (tgt_t >= t_abs[0]) & (tgt_t <= t_abs[-1])
    out = np.full_like(rel_grid, np.nan, dtype=float)
    if inside.sum() < 5:
        return out
    out[inside] = np.interp(tgt_t[inside], t_abs, y)
    return out


def window_mean(row, rel_grid, a, b, min_frac=0.6):
    mask = (rel_grid >= a) & (rel_grid < b)
    vals = row[mask]
    n_have = np.sum(np.isfinite(vals))
    n_need = int(np.ceil(min_frac * mask.sum()))
    if n_have < max(5, n_need):
        return np.nan
    return float(np.nanmean(vals))


# ---------- ROC utils (no sklearn dependency) ----------

def auc_rank(labels01, scores):
    """
    AUC via rank statistic (equivalent to Mann-Whitney U).
    labels01: 0/1 array
    scores: float array
    """
    labels01 = np.asarray(labels01).astype(int)
    scores = np.asarray(scores).astype(float)

    ok = np.isfinite(scores) & np.isfinite(labels01)
    labels01 = labels01[ok]
    scores = scores[ok]

    n1 = int(np.sum(labels01 == 1))
    n0 = int(np.sum(labels01 == 0))
    if n1 < 2 or n0 < 2:
        return np.nan

    # ranks with tie handling: average rank
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)

    # tie correction: average ranks for equal scores
    # (simple pass)
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

    r1 = np.sum(ranks[labels01 == 1])
    auc = (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return float(auc)


def roc_curve_points(labels01, scores):
    """
    Manual ROC curve points (FPR, TPR) by sweeping thresholds.
    Returns arrays fpr, tpr
    """
    labels01 = np.asarray(labels01).astype(int)
    scores = np.asarray(scores).astype(float)
    ok = np.isfinite(scores)
    labels01 = labels01[ok]
    scores = scores[ok]

    # thresholds: unique scores descending
    uniq = np.unique(scores)
    thr = uniq[::-1]

    P = np.sum(labels01 == 1)
    N = np.sum(labels01 == 0)
    if P == 0 or N == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    tpr = []
    fpr = []
    for th in thr:
        pred1 = scores >= th
        TP = np.sum(pred1 & (labels01 == 1))
        FP = np.sum(pred1 & (labels01 == 0))
        tpr.append(TP / P)
        fpr.append(FP / N)

    # add (0,0) and (1,1)
    fpr = np.array([0.0] + fpr + [1.0])
    tpr = np.array([0.0] + tpr + [1.0])
    return fpr, tpr


def bootstrap_ci_auc(pre_vals, post_vals, n_boot=2000, seed=0):
    """
    Bootstrap over bouts: resample paired (pre, post) with replacement,
    then compute AUC on the pooled pre/post samples.
    """
    rng = np.random.default_rng(seed)
    pre_vals = np.asarray(pre_vals, dtype=float)
    post_vals = np.asarray(post_vals, dtype=float)

    ok = np.isfinite(pre_vals) & np.isfinite(post_vals)
    pre_vals = pre_vals[ok]
    post_vals = post_vals[ok]
    n = len(pre_vals)
    if n < 6:
        return np.nan, np.nan

    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        pre_b = pre_vals[idx]
        post_b = post_vals[idx]
        labels = np.concatenate([np.zeros_like(pre_b), np.ones_like(post_b)])
        scores = np.concatenate([pre_b, post_b])
        aucs.append(auc_rank(labels, scores))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


def permutation_p_auc(pre_vals, post_vals, n_perm=5000, seed=0):
    """
    Permutation (paired label-swap within bout):
      For each bout, swap (pre, post) with p=0.5 to form null.
    Two-sided p-value on |AUC-0.5|.
    """
    rng = np.random.default_rng(seed)
    pre_vals = np.asarray(pre_vals, dtype=float)
    post_vals = np.asarray(post_vals, dtype=float)

    ok = np.isfinite(pre_vals) & np.isfinite(post_vals)
    pre_vals = pre_vals[ok]
    post_vals = post_vals[ok]
    n = len(pre_vals)
    if n < 6:
        return np.nan

    labels_obs = np.concatenate([np.zeros_like(pre_vals), np.ones_like(post_vals)])
    scores_obs = np.concatenate([pre_vals, post_vals])
    auc_obs = auc_rank(labels_obs, scores_obs)
    if not np.isfinite(auc_obs):
        return np.nan

    dev_obs = abs(auc_obs - 0.5)
    cnt = 0
    for _ in range(n_perm):
        swap = rng.random(n) < 0.5
        pre_p = pre_vals.copy()
        post_p = post_vals.copy()
        pre_p[swap], post_p[swap] = post_p[swap], pre_p[swap]
        labels = np.concatenate([np.zeros_like(pre_p), np.ones_like(post_p)])
        scores = np.concatenate([pre_p, post_p])
        auc_p = auc_rank(labels, scores)
        if abs(auc_p - 0.5) >= dev_obs - 1e-12:
            cnt += 1
    return float((cnt + 1) / (n_perm + 1))


# ---------- compute pre/post values per transition ----------

def collect_pre_post_touch_end(t_abs, y, touch_starts, touch_ends, rel_grid, win_s):
    """
    Align to touch end (touch -> nontouch).
    pre window is within touch bout: [-min(win, dur), 0)
    post window is [0, win)
    """
    pre = []
    post = []
    kept_bout_idx = []

    for i in range(len(touch_ends)):
        s = float(touch_starts[i])
        e = float(touch_ends[i])
        dur = e - s
        if dur <= 0:
            continue
        pre_len = min(win_s, dur)
        if pre_len < PRE_MIN_S:
            continue

        tr = interp_trace_to_grid(t_abs, y, e, rel_grid)
        m_pre = window_mean(tr, rel_grid, -pre_len, 0.0, min_frac=MIN_FRAC)
        m_post = window_mean(tr, rel_grid, 0.0, win_s, min_frac=MIN_FRAC)
        if np.isfinite(m_pre) and np.isfinite(m_post):
            pre.append(m_pre)
            post.append(m_post)
            kept_bout_idx.append(i)

    return np.array(pre), np.array(post), np.array(kept_bout_idx, dtype=int)


def collect_pre_post_touch_start(t_abs, y, touch_starts, prev_gaps, rel_grid, win_s):
    """
    Align to touch start (nontouch -> touch).
    pre window is within preceding NONTOUCH gap: [-min(win, gap), 0)
    post window is [0, win)
    """
    pre = []
    post = []
    kept_bout_idx = []

    for i in range(len(touch_starts)):
        ts = float(touch_starts[i])
        gap = float(prev_gaps[i]) if np.isfinite(prev_gaps[i]) else np.nan
        if not np.isfinite(gap) or gap <= 0:
            continue
        pre_len = min(win_s, gap)
        if pre_len < PRE_MIN_S:
            continue

        tr = interp_trace_to_grid(t_abs, y, ts, rel_grid)
        m_pre = window_mean(tr, rel_grid, -pre_len, 0.0, min_frac=MIN_FRAC)
        m_post = window_mean(tr, rel_grid, 0.0, win_s, min_frac=MIN_FRAC)
        if np.isfinite(m_pre) and np.isfinite(m_post):
            pre.append(m_pre)
            post.append(m_post)
            kept_bout_idx.append(i)

    return np.array(pre), np.array(post), np.array(kept_bout_idx, dtype=int)


def plot_roc_overlay(pre_a, post_a, label_a, pre_b, post_b, label_b, title, out_png):
    labels_a = np.concatenate([np.zeros_like(pre_a), np.ones_like(post_a)])
    scores_a = np.concatenate([pre_a, post_a])

    labels_b = np.concatenate([np.zeros_like(pre_b), np.ones_like(post_b)])
    scores_b = np.concatenate([pre_b, post_b])

    fpr_a, tpr_a = roc_curve_points(labels_a, scores_a)
    fpr_b, tpr_b = roc_curve_points(labels_b, scores_b)

    auc_a = auc_rank(labels_a, scores_a)
    auc_b = auc_rank(labels_b, scores_b)

    plt.figure(figsize=(5.2, 5.2))
    plt.plot(fpr_a, tpr_a, linewidth=1.8, label=f"{label_a} (AUC={auc_a:.3f})")
    plt.plot(fpr_b, tpr_b, linewidth=1.8, label=f"{label_b} (AUC={auc_b:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
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

    # load data
    Yz, t, dt = base.load_neural(base.DATA_NPZ)
    social_intervals = base.load_behavior(base.BEH_XLSX, base.REUNION_ABS)

    touch_starts, touch_ends = extract_touch_bouts(social_intervals)
    prev_gaps = compute_prev_nontouch_gaps(touch_starts, touch_ends, t_start_ref=float(base.REUNION_ABS))

    # neuron groups
    _, _, dec_nt = base.load_glm_npz(base.NPZ_NT)
    _, _, dec_t  = base.load_glm_npz(base.NPZ_T)

    idx_ntdec = np.where(np.array(dec_nt).astype(bool))[0]
    idx_tdec  = np.where(np.array(dec_t).astype(bool))[0]

    if len(idx_ntdec) == 0 or len(idx_tdec) == 0:
        print("ERROR: dec mask produced empty neuron group.")
        print(f"  nontouch-decreasing neurons: {len(idx_ntdec)}")
        print(f"  touch-decreasing neurons:    {len(idx_tdec)}")
        return

    # group mean traces (across neurons)
    g_ntdec = np.nanmean(Yz[:, idx_ntdec], axis=1)
    g_tdec  = np.nanmean(Yz[:, idx_tdec],  axis=1)

    rel_grid = make_rel_grid(dt, REL_MIN, REL_MAX)

    rows = []

    for win in WINDOWS_S:
        # -------- touch END ROC --------
        pre_nt, post_nt, kept_nt = collect_pre_post_touch_end(t, g_ntdec, touch_starts, touch_ends, rel_grid, win)
        pre_t,  post_t,  kept_t  = collect_pre_post_touch_end(t, g_tdec,  touch_starts, touch_ends, rel_grid, win)

        # fair: keep only intersection of bouts (same bout indices)
        keep_set = set(kept_nt.tolist()).intersection(set(kept_t.tolist()))
        keep_list = np.array(sorted(list(keep_set)), dtype=int)

        def filter_by_keep(pre, post, kept_idx, keep_list):
            m = {int(k): j for j, k in enumerate(kept_idx.tolist())}
            sel = [m[int(k)] for k in keep_list if int(k) in m]
            return pre[sel], post[sel]

        pre_nt2, post_nt2 = filter_by_keep(pre_nt, post_nt, kept_nt, keep_list)
        pre_t2,  post_t2  = filter_by_keep(pre_t,  post_t,  kept_t,  keep_list)

        if len(pre_nt2) >= 6:
            labels = np.concatenate([np.zeros_like(pre_nt2), np.ones_like(post_nt2)])
            scores = np.concatenate([pre_nt2, post_nt2])
            aucv = auc_rank(labels, scores)
            lo, hi = bootstrap_ci_auc(pre_nt2, post_nt2, n_boot=N_BOOT, seed=int(rng.integers(1e9)))
            p = permutation_p_auc(pre_nt2, post_nt2, n_perm=N_PERM, seed=int(rng.integers(1e9)))
            rows.append(dict(
                transition="touch_end (touch→nontouch)",
                window_s=win,
                group="non-touch↓ (group mean)",
                n_bouts=len(pre_nt2),
                auc=aucv,
                auc_minus_0p5=aucv - 0.5,
                ci95_lo=lo, ci95_hi=hi,
                perm_p_two_sided=p
            ))

        if len(pre_t2) >= 6:
            labels = np.concatenate([np.zeros_like(pre_t2), np.ones_like(post_t2)])
            scores = np.concatenate([pre_t2, post_t2])
            aucv = auc_rank(labels, scores)
            lo, hi = bootstrap_ci_auc(pre_t2, post_t2, n_boot=N_BOOT, seed=int(rng.integers(1e9)))
            p = permutation_p_auc(pre_t2, post_t2, n_perm=N_PERM, seed=int(rng.integers(1e9)))
            rows.append(dict(
                transition="touch_end (touch→nontouch)",
                window_s=win,
                group="touch↓ (group mean)",
                n_bouts=len(pre_t2),
                auc=aucv,
                auc_minus_0p5=aucv - 0.5,
                ci95_lo=lo, ci95_hi=hi,
                perm_p_two_sided=p
            ))

        if len(pre_nt2) >= 6 and len(pre_t2) >= 6:
            plot_roc_overlay(
                pre_nt2, post_nt2, "non-touch↓",
                pre_t2,  post_t2,  "touch↓",
                title=f"ROC: touch end | pre[-,0) vs post[0,{win:g}) | n_bouts={len(pre_nt2)}",
                out_png=os.path.join(outdir, f"ROC_touch_end__win_{win:g}s.png")
            )

        # -------- touch START ROC --------
        pre_nt, post_nt, kept_nt = collect_pre_post_touch_start(t, g_ntdec, touch_starts, prev_gaps, rel_grid, win)
        pre_t,  post_t,  kept_t  = collect_pre_post_touch_start(t, g_tdec,  touch_starts, prev_gaps, rel_grid, win)

        keep_set = set(kept_nt.tolist()).intersection(set(kept_t.tolist()))
        keep_list = np.array(sorted(list(keep_set)), dtype=int)

        pre_nt2, post_nt2 = filter_by_keep(pre_nt, post_nt, kept_nt, keep_list)
        pre_t2,  post_t2  = filter_by_keep(pre_t,  post_t,  kept_t,  keep_list)

        if len(pre_nt2) >= 6:
            labels = np.concatenate([np.zeros_like(pre_nt2), np.ones_like(post_nt2)])
            scores = np.concatenate([pre_nt2, post_nt2])
            aucv = auc_rank(labels, scores)
            lo, hi = bootstrap_ci_auc(pre_nt2, post_nt2, n_boot=N_BOOT, seed=int(rng.integers(1e9)))
            p = permutation_p_auc(pre_nt2, post_nt2, n_perm=N_PERM, seed=int(rng.integers(1e9)))
            rows.append(dict(
                transition="touch_start (nontouch→touch)",
                window_s=win,
                group="non-touch↓ (group mean)",
                n_bouts=len(pre_nt2),
                auc=aucv,
                auc_minus_0p5=aucv - 0.5,
                ci95_lo=lo, ci95_hi=hi,
                perm_p_two_sided=p
            ))

        if len(pre_t2) >= 6:
            labels = np.concatenate([np.zeros_like(pre_t2), np.ones_like(post_t2)])
            scores = np.concatenate([pre_t2, post_t2])
            aucv = auc_rank(labels, scores)
            lo, hi = bootstrap_ci_auc(pre_t2, post_t2, n_boot=N_BOOT, seed=int(rng.integers(1e9)))
            p = permutation_p_auc(pre_t2, post_t2, n_perm=N_PERM, seed=int(rng.integers(1e9)))
            rows.append(dict(
                transition="touch_start (nontouch→touch)",
                window_s=win,
                group="touch↓ (group mean)",
                n_bouts=len(pre_t2),
                auc=aucv,
                auc_minus_0p5=aucv - 0.5,
                ci95_lo=lo, ci95_hi=hi,
                perm_p_two_sided=p
            ))

        if len(pre_nt2) >= 6 and len(pre_t2) >= 6:
            plot_roc_overlay(
                pre_nt2, post_nt2, "non-touch↓",
                pre_t2,  post_t2,  "touch↓",
                title=f"ROC: touch start | pre[-,0) vs post[0,{win:g}) | n_bouts={len(pre_nt2)}",
                out_png=os.path.join(outdir, f"ROC_touch_start__win_{win:g}s.png")
            )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "ROC_summary_short_windows.csv"), index=False, encoding="utf-8-sig")

    print("\n=== ROC short-window summary saved ===")
    print(outdir)
    if len(df) > 0:
        # display a compact view
        show = df.copy()
        show["auc"] = show["auc"].map(lambda x: f"{x:.3f}" if np.isfinite(x) else "nan")
        show["auc_minus_0p5"] = show["auc_minus_0p5"].map(lambda x: f"{x:+.3f}" if np.isfinite(x) else "nan")
        show["perm_p_two_sided"] = show["perm_p_two_sided"].map(lambda x: f"{x:.4g}" if np.isfinite(x) else "nan")
        print("\n", show.sort_values(["transition", "window_s", "group"]).to_string(index=False))
    else:
        print("No valid rows (too few bouts). Consider lowering PRE_MIN_S or MIN_FRAC.")


if __name__ == "__main__":
    main()
