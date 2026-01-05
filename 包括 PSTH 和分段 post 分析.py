# -*- coding: utf-8 -*-
"""
Touch -> NonTouch transition (touch bout END) within ±3s
Compare two neuron groups:
  - non-touch decreasing neurons (from NPZ_NT dec mask)
  - touch decreasing neurons (from NPZ_T  dec mask)

Key fix:
  - Per-bout baseline uses the last min(PRE_MAX_S, bout_duration) seconds of the TOUCH bout:
      baseline window = [-min(PRE_MAX_S, dur_i), 0)
  - Optionally require baseline window length >= PRE_MIN_S; otherwise skip that bout.

Outputs:
  - overlay PSTH (mean ± 95%CI) for both groups (per-bout normalized)
  - per-bout post(0..3) means + summary stats (t-test / Wilcoxon vs 0)
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
OUT_SUBDIR = "Touch2NonTouch_transition__3s__neuron_groups__truncated_pre"

REL_MIN, REL_MAX = -3.0, 3.0

# baseline window for each bout: [-min(PRE_MAX_S, dur_i), 0)
PRE_MAX_S = 3.0      # “最多用 3s”
PRE_MIN_S = 1.0      # bout 太短时：至少要有 1s 的 pre 才纳入（你也可以改成 0.5）

POST_WIN = (0.0, 3.0)  # 你现在关心的“变换后 3s”
MIN_FRAC = 0.6         # 窗口内至少有 60% 采样点有效

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


def build_event_matrix_norm_truncated_pre(t_abs, y, bout_starts, bout_ends, rel_grid):
    """
    Align to each touch bout END.
    Per-bout baseline: [-min(PRE_MAX_S, dur_i), 0)
    Require baseline length >= PRE_MIN_S; otherwise skip bout.

    Returns:
      M_norm: (n_bouts, T) per-bout normalized traces
      pre_len_used: (n_bouts,) actual seconds used for baseline (NaN if skipped)
      keep: boolean mask for bouts kept (baseline valid)
    """
    n = len(bout_ends)
    T = len(rel_grid)
    M_norm = np.full((n, T), np.nan, dtype=float)
    pre_len_used = np.full(n, np.nan, dtype=float)
    keep = np.zeros(n, dtype=bool)

    for i in range(n):
        s = float(bout_starts[i])
        e = float(bout_ends[i])
        dur = e - s
        if dur <= 0:
            continue

        pre_len = min(PRE_MAX_S, dur)
        if pre_len < PRE_MIN_S:
            continue  # bout 太短，跳过

        tr = interp_trace_to_grid(t_abs, y, e, rel_grid)

        a = -pre_len
        b = 0.0
        base = window_mean(tr, rel_grid, a, b, min_frac=MIN_FRAC)
        if np.isfinite(base):
            M_norm[i, :] = tr - base
            pre_len_used[i] = pre_len
            keep[i] = True

    return M_norm, pre_len_used, keep


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

    plt.plot(rel_grid[ok_a], ma[ok_a], linewidth=1.8, label=f"{label_a}")
    plt.fill_between(rel_grid[ok_a], la[ok_a], ha[ok_a], alpha=0.18)

    plt.plot(rel_grid[ok_b], mb[ok_b], linewidth=1.8, label=f"{label_b}")
    plt.fill_between(rel_grid[ok_b], lb[ok_b], hb[ok_b], alpha=0.18)

    plt.xlabel("Time from touch→nontouch transition (s)  [touch bout end]")
    plt.ylabel("Activity (z), per-bout normalized (pre truncated to bout)")
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

    # Touch bouts and events (touch->nontouch transitions)
    touch_starts, touch_ends = extract_touch_bouts(social_intervals)

    # Load GLM masks: dec arrays define "decreasing neurons" for each model
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

    # Build per-bout normalized matrices with truncated pre
    M_nt, pre_used_nt, keep_nt = build_event_matrix_norm_truncated_pre(t, g_ntdec, touch_starts, touch_ends, rel_grid)
    M_t,  pre_used_t,  keep_t  = build_event_matrix_norm_truncated_pre(t, g_tdec,  touch_starts, touch_ends, rel_grid)

    # For fair comparison, keep only bouts that are valid for BOTH groups
    keep_both = keep_nt & keep_t
    M_nt2 = M_nt[keep_both, :]
    M_t2  = M_t[keep_both, :]
    pre_used = pre_used_nt[keep_both]  # same bout set; lengths may differ slightly only if NaNs differed, but keep_both avoids that

    # Plot overlay
    plot_overlay(
        rel_grid, M_nt2, f"non-touch↓ neurons (mean across neurons), bouts={M_nt2.shape[0]}",
        M_t2,  f"touch↓ neurons (mean across neurons), bouts={M_t2.shape[0]}",
        title=f"Touch→NonTouch transition (±3s), per-bout baseline truncated (max {PRE_MAX_S:.1f}s, min {PRE_MIN_S:.1f}s)",
        out_png=os.path.join(outdir, "psth_overlay__touch_to_nontouch__3s__groupmeans__truncated_pre.png"),
    )

    # Per-bout post(0..3) mean (since pre normalized to 0)
    post_nt = np.array([window_mean(M_nt2[i, :], rel_grid, POST_WIN[0], POST_WIN[1], MIN_FRAC) for i in range(M_nt2.shape[0])])
    post_t  = np.array([window_mean(M_t2[i,  :], rel_grid, POST_WIN[0], POST_WIN[1], MIN_FRAC) for i in range(M_t2.shape[0])])

    t_p_nt, w_p_nt = stats_1samp(post_nt)
    t_p_t,  w_p_t  = stats_1samp(post_t)

    df_long = pd.DataFrame({
        "bout_index_in_touch_list": np.where(keep_both)[0],
        "touch_dur_s": (touch_ends[keep_both] - touch_starts[keep_both]),
        "pre_used_s": pre_used,
        "post0_3_mean__nontouch_dec_group": post_nt,
        "post0_3_mean__touch_dec_group": post_t,
    })
    df_long.to_csv(os.path.join(outdir, "per_bout_post0_3s__groupmeans__truncated_pre.csv"),
                   index=False, encoding="utf-8-sig")

    df_sum = pd.DataFrame([
        {
            "group": "non-touch↓ neurons (group mean)",
            "n_bouts": int(np.sum(np.isfinite(post_nt))),
            "mean_post0_3": float(np.nanmean(post_nt)),
            "median_post0_3": float(np.nanmedian(post_nt)),
            "ttest_p": t_p_nt,
            "wilcoxon_p": w_p_nt,
        },
        {
            "group": "touch↓ neurons (group mean)",
            "n_bouts": int(np.sum(np.isfinite(post_t))),
            "mean_post0_3": float(np.nanmean(post_t)),
            "median_post0_3": float(np.nanmedian(post_t)),
            "ttest_p": t_p_t,
            "wilcoxon_p": w_p_t,
        }
    ])
    df_sum.to_csv(os.path.join(outdir, "summary_post0_3s__groupmeans__truncated_pre.csv"),
                  index=False, encoding="utf-8-sig")

    print("\n=== Touch→NonTouch transition within ±3s (truncated pre) ===")
    print(f"touch bouts total: {len(touch_ends)}")
    print(f"kept bouts (both groups have valid pre): {int(keep_both.sum())}")
    print(f"dt = {dt}")
    print(f"nontouch↓ neurons: {len(idx_ntdec)}")
    print(f"touch↓ neurons:    {len(idx_tdec)}")
    print("\nBaseline used per bout: pre = [-min(3s, dur_i), 0), require >= PRE_MIN_S")
    print(f"PRE_MAX_S={PRE_MAX_S}, PRE_MIN_S={PRE_MIN_S}")
    print("\nSummary (post[0,3) vs truncated pre baseline):")
    print(df_sum.to_string(index=False))

    print("\n✅ Outputs saved to:")
    print(outdir)


if __name__ == "__main__":
    main()
