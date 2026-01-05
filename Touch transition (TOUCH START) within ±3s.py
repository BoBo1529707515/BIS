# -*- coding: utf-8 -*-
"""
NonTouch -> Touch transition (TOUCH START) within ±3s
Compare two neuron groups:
  - non-touch decreasing neurons (from NPZ_NT dec mask)
  - touch decreasing neurons (from NPZ_T  dec mask)

RIGOROUS baseline (per-bout, truncated):
  For each touch start at time ts_i:
    prev nontouch duration gap_i = ts_i - prev_touch_end
    baseline window = [-min(PRE_MAX_S, gap_i), 0)  (i.e., within the preceding NONTOUCH gap)
    require baseline length >= PRE_MIN_S, else skip that bout

Then compute:
  - PSTH overlay (mean ± 95% CI) for both groups, per-bout normalized
  - Per-bout post means (0..1s and 0..3s) and summary stats
  - Paired group difference per bout: (touch↓ - nontouch↓)

Outputs:
  figs/.../NonTouch2Touch_transition__3s__neuron_groups__truncated_pre/
    - psth_overlay__nontouch_to_touch__3s__groupmeans__truncated_pre.png
    - per_bout_post__groupmeans__truncated_pre.csv
    - summary_post__groupmeans__truncated_pre.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib.util
from scipy.stats import t as student_t
from scipy.stats import ttest_1samp, wilcoxon

# =========================
# 0) CHANGE THIS PATH
# =========================
BASE_SCRIPT_PATH = r"F:\工作文件\RA\python\吸引子\NT-axis vs TOUCH-axis 独立投影 + 全套对比分析（含 touch 背景）.py"

# =========================
# 1) SETTINGS
# =========================
OUT_SUBDIR = "NonTouch2Touch_transition__3s__neuron_groups__truncated_pre"

REL_MIN, REL_MAX = -3.0, 3.0

# baseline per bout uses preceding NONTOUCH gap: [-min(PRE_MAX_S, gap_i), 0)
PRE_MAX_S = 3.0      # use up to 3s
PRE_MIN_S = 1.0      # require at least 1s baseline; change to 0.5 if you want keep more bouts

POST_WINS = [
    ("post_0_1s", 0.0, 1.0),
    ("post_0_3s", 0.0, 3.0),
]

MIN_FRAC = 0.6  # window needs >=60% valid samples

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


def compute_prev_nontouch_gaps(touch_starts, touch_ends, t_start_ref=None):
    """
    For each touch start, compute preceding NONTOUCH gap:
      gap_i = touch_start_i - prev_touch_end
    For the first bout:
      if t_start_ref is provided, gap_0 = touch_start_0 - t_start_ref
      else gap_0 = NaN (will likely be skipped)
    Returns:
      gaps (n_bouts,) in seconds
      prev_end (n_bouts,) prev touch end time used
    """
    order = np.argsort(touch_starts)
    ts = touch_starts[order]
    te = touch_ends[order]

    gaps = np.full_like(ts, np.nan, dtype=float)
    prev_end = np.full_like(ts, np.nan, dtype=float)

    for i in range(len(ts)):
        if i == 0:
            if t_start_ref is not None:
                prev_end[i] = float(t_start_ref)
                gaps[i] = float(ts[i] - t_start_ref)
        else:
            prev_end[i] = float(te[i - 1])
            gaps[i] = float(ts[i] - te[i - 1])

    # map back to original order
    gaps_back = np.full_like(gaps, np.nan, dtype=float)
    prev_back = np.full_like(prev_end, np.nan, dtype=float)
    gaps_back[order] = gaps
    prev_back[order] = prev_end
    return gaps_back, prev_back


def build_event_matrix_norm_truncated_pre_by_gap(t_abs, y, event_times_abs, pre_gaps_s, rel_grid):
    """
    Align to each event (touch start).
    Baseline uses preceding NONTOUCH gap:
      pre_len_i = min(PRE_MAX_S, gap_i)
      require pre_len_i >= PRE_MIN_S
      baseline window = [-pre_len_i, 0)

    Returns:
      M_norm: (n_events, T)
      pre_used: (n_events,)
      keep: (n_events,) bool
    """
    n = len(event_times_abs)
    T = len(rel_grid)
    M_norm = np.full((n, T), np.nan, dtype=float)
    pre_used = np.full(n, np.nan, dtype=float)
    keep = np.zeros(n, dtype=bool)

    for i in range(n):
        e = float(event_times_abs[i])
        gap = float(pre_gaps_s[i]) if np.isfinite(pre_gaps_s[i]) else np.nan
        if not np.isfinite(gap) or gap <= 0:
            continue

        pre_len = min(PRE_MAX_S, gap)
        if pre_len < PRE_MIN_S:
            continue

        tr = interp_trace_to_grid(t_abs, y, e, rel_grid)
        base = window_mean(tr, rel_grid, -pre_len, 0.0, min_frac=MIN_FRAC)
        if np.isfinite(base):
            M_norm[i, :] = tr - base
            pre_used[i] = pre_len
            keep[i] = True

    return M_norm, pre_used, keep


def mean_ci95(M):
    n_eff = np.sum(np.isfinite(M), axis=0)
    mean = np.nanmean(M, axis=0)
    sd = np.nanstd(M, axis=0, ddof=1)
    sem = sd / np.sqrt(np.maximum(n_eff, 1))

    lo = np.full_like(mean, np.nan, dtype=float)
    hi = np.full_like(mean, np.nan, dtype=float)
    for k in range(len(mean)):
        n = int(n_eff[k])
        if n >= 2 and np.isfinite(sem[k]):
            tcrit = student_t.ppf(0.975, df=n - 1)
            lo[k] = mean[k] - tcrit * sem[k]
            hi[k] = mean[k] + tcrit * sem[k]
    return mean, lo, hi, n_eff


def plot_overlay(rel_grid, M_a, label_a, M_b, label_b, title, out_png):
    ma, la, ha, na = mean_ci95(M_a)
    mb, lb, hb, nb = mean_ci95(M_b)
    ok_a = na > 0
    ok_b = nb > 0

    plt.figure(figsize=(9.2, 4.3))
    plt.axvline(0, linestyle="--", linewidth=1.2)

    plt.plot(rel_grid[ok_a], ma[ok_a], linewidth=1.8, label=label_a)
    plt.fill_between(rel_grid[ok_a], la[ok_a], ha[ok_a], alpha=0.18)

    plt.plot(rel_grid[ok_b], mb[ok_b], linewidth=1.8, label=label_b)
    plt.fill_between(rel_grid[ok_b], lb[ok_b], hb[ok_b], alpha=0.18)

    plt.xlabel("Time from nontouch→touch transition (s)  [touch start]")
    plt.ylabel("Activity (z), per-bout normalized (pre within preceding NONTOUCH gap)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.xlim(rel_grid[0], rel_grid[-1])
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def stats_1samp(x):
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan
    t_p = ttest_1samp(x, 0.0, nan_policy="omit").pvalue
    try:
        w_p = wilcoxon(x).pvalue
    except Exception:
        w_p = np.nan
    return float(t_p), float(w_p)


def main():
    base = load_base_module(BASE_SCRIPT_PATH)
    outdir = os.path.join(base.OUTDIR, OUT_SUBDIR)
    os.makedirs(outdir, exist_ok=True)

    # Load neural + behavior
    Yz, t, dt = base.load_neural(base.DATA_NPZ)
    social_intervals = base.load_behavior(base.BEH_XLSX, base.REUNION_ABS)

    # Touch bouts
    touch_starts, touch_ends = extract_touch_bouts(social_intervals)

    # Event times: touch START = nontouch -> touch
    event_starts = touch_starts.copy()

    # Preceding NONTOUCH gap for each touch start
    # Use base.REUNION_ABS as reference for the very first bout's "previous end"
    gaps, prev_end = compute_prev_nontouch_gaps(touch_starts, touch_ends, t_start_ref=float(base.REUNION_ABS))

    # Load GLM masks: dec arrays define the two neuron groups
    beta_nt, _, dec_nt = base.load_glm_npz(base.NPZ_NT)
    beta_t,  _, dec_t  = base.load_glm_npz(base.NPZ_T)

    idx_ntdec = np.where(np.array(dec_nt).astype(bool))[0]
    idx_tdec  = np.where(np.array(dec_t).astype(bool))[0]

    if len(idx_ntdec) == 0 or len(idx_tdec) == 0:
        print("ERROR: dec mask produced empty neuron group.")
        print(f"  nontouch-decreasing neurons: {len(idx_ntdec)}")
        print(f"  touch-decreasing neurons:    {len(idx_tdec)}")
        return

    # Group mean time series (mean across neurons in each group)
    g_ntdec = np.nanmean(Yz[:, idx_ntdec], axis=1)
    g_tdec  = np.nanmean(Yz[:, idx_tdec],  axis=1)

    rel_grid = make_rel_grid(dt, REL_MIN, REL_MAX)

    # Build per-bout normalized matrices with baseline truncated by preceding NONTOUCH gap
    M_nt, pre_used_nt, keep_nt = build_event_matrix_norm_truncated_pre_by_gap(t, g_ntdec, event_starts, gaps, rel_grid)
    M_t,  pre_used_t,  keep_t  = build_event_matrix_norm_truncated_pre_by_gap(t, g_tdec,  event_starts, gaps, rel_grid)

    # Fair comparison: keep only bouts valid in BOTH groups
    keep_both = keep_nt & keep_t
    M_nt2 = M_nt[keep_both, :]
    M_t2  = M_t[keep_both, :]
    pre_used = pre_used_nt[keep_both]
    gaps_kept = gaps[keep_both]

    # Overlay plot
    plot_overlay(
        rel_grid,
        M_nt2, f"non-touch↓ neurons (group mean), bouts={M_nt2.shape[0]}",
        M_t2,  f"touch↓ neurons (group mean), bouts={M_t2.shape[0]}",
        title=f"NonTouch→Touch (touch start) ±3s | baseline truncated by preceding NONTOUCH gap (max {PRE_MAX_S:.1f}s, min {PRE_MIN_S:.1f}s)",
        out_png=os.path.join(outdir, "psth_overlay__nontouch_to_touch__3s__groupmeans__truncated_pre.png"),
    )

    # Per-bout post means (since pre normalized to 0, diff = mean(post))
    per_bout = {
        "touch_bout_index": np.where(keep_both)[0],
        "prev_nontouch_gap_s": gaps_kept,
        "pre_used_s": pre_used,
        "touch_dur_s": (touch_ends[keep_both] - touch_starts[keep_both]),
    }

    summary_rows = []
    for name, a, b in POST_WINS:
        post_nt = np.array([window_mean(M_nt2[i, :], rel_grid, a, b, MIN_FRAC) for i in range(M_nt2.shape[0])])
        post_t  = np.array([window_mean(M_t2[i,  :], rel_grid, a, b, MIN_FRAC) for i in range(M_t2.shape[0])])
        diff    = post_t - post_nt  # paired difference per bout

        per_bout[f"{name}__nontouch_dec_group"] = post_nt
        per_bout[f"{name}__touch_dec_group"] = post_t
        per_bout[f"{name}__paired_diff_touch_minus_nontouch"] = diff

        t_p_nt, w_p_nt = stats_1samp(post_nt)
        t_p_t,  w_p_t  = stats_1samp(post_t)
        t_p_d,  w_p_d  = stats_1samp(diff)

        summary_rows.extend([
            {"window": name, "group": "non-touch↓ (group mean)", "n_bouts": int(np.sum(np.isfinite(post_nt))),
             "mean": float(np.nanmean(post_nt)), "median": float(np.nanmedian(post_nt)), "ttest_p": t_p_nt, "wilcoxon_p": w_p_nt},
            {"window": name, "group": "touch↓ (group mean)", "n_bouts": int(np.sum(np.isfinite(post_t))),
             "mean": float(np.nanmean(post_t)), "median": float(np.nanmedian(post_t)), "ttest_p": t_p_t, "wilcoxon_p": w_p_t},
            {"window": name, "group": "paired diff (touch - nontouch)", "n_bouts": int(np.sum(np.isfinite(diff))),
             "mean": float(np.nanmean(diff)), "median": float(np.nanmedian(diff)), "ttest_p": t_p_d, "wilcoxon_p": w_p_d},
        ])

    df_long = pd.DataFrame(per_bout)
    df_sum = pd.DataFrame(summary_rows)

    df_long.to_csv(os.path.join(outdir, "per_bout_post__groupmeans__truncated_pre.csv"),
                   index=False, encoding="utf-8-sig")
    df_sum.to_csv(os.path.join(outdir, "summary_post__groupmeans__truncated_pre.csv"),
                  index=False, encoding="utf-8-sig")

    print("\n=== NonTouch→Touch transition within ±3s (touch start; truncated pre by preceding NONTOUCH gap) ===")
    print(f"touch bouts total: {len(touch_starts)}")
    print(f"kept bouts (both groups valid pre): {int(keep_both.sum())}")
    print(f"dt = {dt}")
    print(f"nontouch↓ neurons: {len(idx_ntdec)}")
    print(f"touch↓ neurons:    {len(idx_tdec)}")
    print("\nBaseline per bout: pre = [-min(PRE_MAX_S, prev_nontouch_gap), 0), require >= PRE_MIN_S")
    print(f"PRE_MAX_S={PRE_MAX_S}, PRE_MIN_S={PRE_MIN_S}")
    print("\nSummary:")
    print(df_sum.to_string(index=False))

    print("\n✅ Outputs saved to:")
    print(outdir)


if __name__ == "__main__":
    main()
