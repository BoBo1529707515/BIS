# -*- coding: utf-8 -*-
"""
Integrated analysis:
1) Pre(-300~0) vs Post(639~end) per-neuron change
   - 1s bin means
   - Mann–Whitney U (two-sided) -> p
   - BH-FDR -> q
   - classify Up / Down / NS
   - plot 3 group figures (heatmap + mean±SEM)

2) Within NS, find "U-shape" neurons:
   - Pre high, Mid(0~639) low, Post high
   - export list + stats + plot

3) Compute Coding Direction (CD) for Pre vs Post in population space:
   cd = (mu_post - mu_pre) / ||mu_post - mu_pre||
   projection: s(t) = cd^T (y(t) - ref), ref = (mu_pre + mu_post)/2
   - export cd weights
   - plot CD projection trajectory
   - plot heatmap sorted by cd weights

Assumptions:
- Base script provides: load_neural(), DATA_NPZ, OUTDIR, REUNION_ABS
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib.util

from scipy.stats import mannwhitneyu

# =========================
# 0) CHANGE THIS PATH
# =========================
BASE_SCRIPT_PATH = r"F:\工作文件\RA\python\吸引子\NT-axis vs TOUCH-axis 独立投影 + 全套对比分析（含 touch 背景）.py"

# =========================
# 1) PARAMETERS YOU MAY TUNE
# =========================
OUT_SUBDIR = "PreReunion_vs_Post639s_change"

# Windows defined in "time from reunion (s)"
PRE_WIN = (-300.0, 0.0)       # 重聚前窗口
POST_START_S = 639.0          # 重聚后从 639s 开始（到录制结束，自动）
MID_WIN = (0.0, POST_START_S) # 用于U-shape：中间段

# Visualization
PLOT_LEFT_S = -300.0          # 热图左侧显示起点（通常与 PRE_WIN 起点一致）
SHOW_VLINES = True
VLINES = [0.0, 639.0]         # 竖线：重聚=0；对比窗口起点=639

# Stats settings
BIN_SIZE_S = 1.0              # 窗口内按 1s 做 bin mean
ALPHA_FDR = 0.05              # BH-FDR q阈值
MIN_BINS_PER_WIN = 3          # 每个窗口至少多少bin才检验

# Heatmap display range for z-scored activity
VMIN, VMAX = (-1.0, 3.0)

# U-shape thresholds (tune if needed)
U_THRESH = 0.5                # min(pre,post) - mid > U_THRESH
SYM_TOL = 0.5                 # |pre - post| < SYM_TOL

# CD plotting
CD_SMOOTH_BIN_S = 1.0          # 对 s(t) 再做 1s 平滑（看起来更像论文轨迹）
# =========================


def load_base_module(py_path: str):
    spec = importlib.util.spec_from_file_location("base_analysis_module", py_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def bh_fdr(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR: return reject(bool), qvals"""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    qvals = np.empty_like(q)
    qvals[order] = q
    reject = qvals <= alpha
    return reject, qvals


def window_mask(t_rel, win):
    a, b = win
    return (t_rel >= a) & (t_rel < b)


def bin_means(x, t_rel, win, bin_size_s):
    """Compute per-bin mean of x within win -> 1D array of bin means."""
    a, b = win
    if b <= a:
        return np.array([], dtype=float)
    edges = np.arange(a, b + 1e-9, bin_size_s)
    if len(edges) < 2:
        return np.array([], dtype=float)

    out = []
    for i in range(len(edges) - 1):
        m = (t_rel >= edges[i]) & (t_rel < edges[i + 1])
        if np.any(m):
            out.append(np.nanmean(x[m]))
    return np.asarray(out, dtype=float)


def ensure_Yz_neurons_by_time(Yz, t_abs):
    """Ensure Yz is (n_neurons, n_timepoints). Accept (T,N) or (N,T)."""
    Yz = np.asarray(Yz, dtype=float)
    t_abs = np.asarray(t_abs, dtype=float).ravel()
    T = len(t_abs)

    if Yz.ndim != 2:
        raise ValueError(f"Yz must be 2D, got shape {Yz.shape}")

    if Yz.shape[0] == T and Yz.shape[1] != T:
        Yz = Yz.T
    elif Yz.shape[1] == T and Yz.shape[0] != T:
        pass
    elif Yz.shape[0] == T and Yz.shape[1] == T:
        raise ValueError(f"Ambiguous square Yz shape {Yz.shape}; cannot infer time axis.")
    else:
        raise ValueError(
            f"Yz shape {Yz.shape} does not match time length {T}. Expected (N,T) or (T,N)."
        )
    return Yz, t_abs


def plot_group_heatmap_and_mean(
    Y_plot, t_rel_plot, group_indices, sort_values, title, out_png, vlines=None
):
    """Heatmap (sorted by sort_values) + mean±SEM."""
    vlines = vlines or []

    if len(group_indices) == 0:
        fig = plt.figure(figsize=(10, 4), constrained_layout=True)
        ax = fig.add_subplot(111)
        ax.set_title(title + " (n=0)")
        ax.axis("off")
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
        return

    gi = np.asarray(group_indices, dtype=int)
    sv = np.asarray(sort_values, dtype=float)
    d = sv[gi]
    sort_idx = np.argsort(d)[::-1]
    gi_sorted = gi[sort_idx]
    Yg = Y_plot[gi_sorted, :]

    mean = np.nanmean(Yg, axis=0)
    n = np.sum(np.isfinite(Yg), axis=0)
    sd = np.nanstd(Yg, axis=0, ddof=1)
    sem = sd / np.sqrt(np.maximum(n, 1))

    fig = plt.figure(figsize=(10.8, 6.9), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.2])

    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(
        Yg,
        aspect="auto",
        interpolation="nearest",
        extent=[t_rel_plot[0], t_rel_plot[-1], Yg.shape[0], 0],
        vmin=VMIN,
        vmax=VMAX,
    )
    ax0.set_title(f"{title} (n={len(gi_sorted)})")
    ax0.set_ylabel("Neurons (sorted)")
    for xv in vlines:
        ax0.axvline(xv, color="w", linewidth=2.0)

    ax0.text(
        0.02, 0.95,
        f"{len(gi_sorted)}/{Y_plot.shape[0]} neurons",
        transform=ax0.transAxes,
        color="w",
        fontsize=11,
        va="top",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.35, edgecolor="none")
    )

    cbar = fig.colorbar(im, ax=ax0, fraction=0.02, pad=0.02)
    cbar.set_label("Activity (z)")

    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax1.plot(t_rel_plot, mean, linewidth=2.0)
    ax1.fill_between(t_rel_plot, mean - sem, mean + sem, alpha=0.2)
    for xv in vlines:
        ax1.axvline(xv, color="k", linestyle="--", linewidth=1.5)
    ax1.set_xlabel("Time from reunion (s)")
    ax1.set_ylabel("Mean ± SEM")
    ax1.grid(alpha=0.3)

    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def compute_cd(Yz, t_rel, pre_win, post_win, center="mid"):
    """
    CD coding direction:
      mu_pre = mean_t y(t) in pre
      mu_post = mean_t y(t) in post
      cd = (mu_post - mu_pre)/||...||
    Projection:
      s(t) = cd^T ( y(t) - ref ), ref = (mu_pre+mu_post)/2 if center='mid'
    """
    pre_mask = window_mask(t_rel, pre_win)
    post_mask = window_mask(t_rel, post_win)
    if np.sum(pre_mask) < 5 or np.sum(post_mask) < 5:
        raise RuntimeError("Too few samples in PRE or POST for CD computation.")

    mu_pre = np.nanmean(Yz[:, pre_mask], axis=1)    # (N,)
    mu_post = np.nanmean(Yz[:, post_mask], axis=1)  # (N,)

    d = mu_post - mu_pre
    norm = float(np.linalg.norm(d))
    if (not np.isfinite(norm)) or norm == 0:
        raise RuntimeError("CD undefined: ||mu_post - mu_pre|| is 0 or NaN.")

    cd = d / norm  # (N,)

    if center == "mid":
        ref = 0.5 * (mu_pre + mu_post)
    elif center == "pre":
        ref = mu_pre
    else:
        raise ValueError("center must be 'mid' or 'pre'")

    S = cd @ (Yz - ref[:, None])  # (T,)
    return cd, S, mu_pre, mu_post


def smooth_by_time_bins(x, t_rel, bin_size_s):
    """Return (t_centers, x_binmean) with simple time-binning."""
    a, b = float(np.min(t_rel)), float(np.max(t_rel))
    edges = np.arange(a, b + 1e-9, bin_size_s)
    if len(edges) < 2:
        return t_rel, x

    tc, xb = [], []
    for i in range(len(edges) - 1):
        m = (t_rel >= edges[i]) & (t_rel < edges[i+1])
        if np.any(m):
            tc.append(0.5*(edges[i] + edges[i+1]))
            xb.append(float(np.nanmean(x[m])))
    return np.asarray(tc), np.asarray(xb)


def plot_cd_projection(t_rel, S, pre_win, mid_win, post_win, out_png, vlines=None):
    vlines = vlines or []

    # smooth
    t_sm, s_sm = smooth_by_time_bins(S, t_rel, CD_SMOOTH_BIN_S)

    fig = plt.figure(figsize=(10.8, 4.6), constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.plot(t_sm, s_sm, linewidth=2.0)

    # window shading
    ax.axvspan(pre_win[0], pre_win[1], alpha=0.12)
    ax.axvspan(mid_win[0], mid_win[1], alpha=0.08)
    ax.axvspan(post_win[0], post_win[1], alpha=0.12)

    for xv in vlines:
        ax.axvline(xv, linestyle="--", linewidth=1.5)

    ax.set_xlabel("Time from reunion (s)")
    ax.set_ylabel("CD projection  s(t)")
    ax.set_title("Coding Direction (Pre vs Post639s+) projection")
    ax.grid(alpha=0.25)

    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_cd_sorted_heatmap(Y_plot, t_rel_plot, cd, out_png, vlines=None):
    vlines = vlines or []
    order = np.argsort(cd)[::-1]  # high weight on top
    Y_sorted = Y_plot[order, :]

    fig = plt.figure(figsize=(10.8, 5.8), constrained_layout=True)
    ax = fig.add_subplot(111)
    im = ax.imshow(
        Y_sorted,
        aspect="auto",
        interpolation="nearest",
        extent=[t_rel_plot[0], t_rel_plot[-1], Y_sorted.shape[0], 0],
        vmin=VMIN, vmax=VMAX
    )
    ax.set_title("Heatmap sorted by CD weights (high -> low)")
    ax.set_xlabel("Time from reunion (s)")
    ax.set_ylabel("Neurons (sorted by CD weight)")
    for xv in vlines:
        ax.axvline(xv, color="w", linewidth=2.0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Activity (z)")
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main():
    base = load_base_module(BASE_SCRIPT_PATH)
    outdir = os.path.join(base.OUTDIR, OUT_SUBDIR)
    os.makedirs(outdir, exist_ok=True)

    # Load neural
    Yz, t_abs, dt = base.load_neural(base.DATA_NPZ)
    Yz, t_abs = ensure_Yz_neurons_by_time(Yz, t_abs)
    print(f"[OK] Yz shape = {Yz.shape} (neurons, time), t_abs len = {len(t_abs)}")

    reunion_abs = float(base.REUNION_ABS)
    t_rel = t_abs - reunion_abs
    print("t_rel range:", float(np.min(t_rel)), float(np.max(t_rel)))

    # POST window to recording end
    t_rel_end = float(np.max(t_rel))
    if t_rel_end <= POST_START_S:
        raise RuntimeError(
            f"Recording ends at t_rel={t_rel_end:.1f}s, but POST_START_S={POST_START_S}. "
            f"Check REUNION_ABS alignment or recording length."
        )
    POST_WIN = (POST_START_S, t_rel_end)

    # vlines defined ONCE (fix your previous error)
    vlines = VLINES if SHOW_VLINES else []

    # Plot range
    plot_left = float(PLOT_LEFT_S)
    plot_right = t_rel_end
    m_plot = (t_rel >= plot_left) & (t_rel <= plot_right)
    if np.sum(m_plot) < 10:
        raise RuntimeError("Plot range does not overlap time axis. Check PLOT_LEFT_S/REUNION_ABS.")
    t_rel_plot = t_rel[m_plot]
    Y_plot = Yz[:, m_plot]

    # Masks
    m_pre = window_mask(t_rel, PRE_WIN)
    m_post = window_mask(t_rel, POST_WIN)
    if np.sum(m_pre) < 10 or np.sum(m_post) < 10:
        raise RuntimeError("PRE_WIN or POST_WIN too short or out of range.")

    n_neurons = Yz.shape[0]
    pvals = np.full(n_neurons, np.nan, dtype=float)
    delta = np.full(n_neurons, np.nan, dtype=float)
    mu_pre = np.full(n_neurons, np.nan, dtype=float)
    mu_post = np.full(n_neurons, np.nan, dtype=float)

    # ---- Per-neuron stats (1s bins + MWU) ----
    for i in range(n_neurons):
        x = Yz[i, :]

        pre_bins = bin_means(x, t_rel, PRE_WIN, BIN_SIZE_S)
        post_bins = bin_means(x, t_rel, POST_WIN, BIN_SIZE_S)

        mu_pre[i] = float(np.nanmean(pre_bins)) if len(pre_bins) else np.nan
        mu_post[i] = float(np.nanmean(post_bins)) if len(post_bins) else np.nan
        delta[i] = mu_post[i] - mu_pre[i]

        if len(pre_bins) < MIN_BINS_PER_WIN or len(post_bins) < MIN_BINS_PER_WIN:
            pvals[i] = np.nan
            continue

        try:
            _, p = mannwhitneyu(post_bins, pre_bins, alternative="two-sided")
        except Exception:
            p = np.nan
        pvals[i] = p

    # FDR
    p_clean = np.where(np.isfinite(pvals), pvals, 1.0)
    reject, qvals = bh_fdr(p_clean, alpha=ALPHA_FDR)

    up = np.where(reject & (delta > 0))[0]
    down = np.where(reject & (delta < 0))[0]
    ns = np.where(~reject)[0]

    # Save lists
    pd.Series(up, name="neuron_index").to_csv(os.path.join(outdir, "neurons_up.csv"), index=False, encoding="utf-8-sig")
    pd.Series(down, name="neuron_index").to_csv(os.path.join(outdir, "neurons_down.csv"), index=False, encoding="utf-8-sig")
    pd.Series(ns, name="neuron_index").to_csv(os.path.join(outdir, "neurons_ns.csv"), index=False, encoding="utf-8-sig")

    # Save per-neuron stats
    df_stats = pd.DataFrame({
        "neuron_index": np.arange(n_neurons),
        "mu_pre": mu_pre,
        "mu_post_639plus": mu_post,
        "delta_post_minus_pre": delta,
        "pval_mannwhitney": pvals,
        "qval_fdr": qvals,
        "sig_fdr": reject.astype(int),
        "direction": np.where(reject & (delta > 0), "up",
                     np.where(reject & (delta < 0), "down", "ns"))
    })
    df_stats.to_csv(os.path.join(outdir, "stats_all_neurons.csv"), index=False, encoding="utf-8-sig")

    # Print summary
    print("\n=== Pre-reunion vs Post-639s neuron selection ===")
    print(f"PRE_WIN={PRE_WIN}")
    print(f"POST_WIN={POST_WIN}  (auto to end)")
    print(f"BIN_SIZE_S={BIN_SIZE_S}, FDR alpha={ALPHA_FDR}")
    print(f"Total neurons: {n_neurons}")
    print(f"Up (post>pre): {len(up)}")
    print(f"Down (post<pre): {len(down)}")
    print(f"NS: {len(ns)}")
    print("\nUp indices:", up.tolist())
    print("Down indices:", down.tolist())

    # Plot 3 panels
    plot_group_heatmap_and_mean(
        Y_plot, t_rel_plot, up, delta,
        title="Up neurons (post 639s+ > pre reunion)",
        out_png=os.path.join(outdir, "Up_heatmap_mean.png"),
        vlines=vlines
    )
    plot_group_heatmap_and_mean(
        Y_plot, t_rel_plot, down, delta,
        title="Down neurons (post 639s+ < pre reunion)",
        out_png=os.path.join(outdir, "Down_heatmap_mean.png"),
        vlines=vlines
    )
    plot_group_heatmap_and_mean(
        Y_plot, t_rel_plot, ns, delta,
        title="NS neurons (FDR not significant)",
        out_png=os.path.join(outdir, "NS_heatmap_mean.png"),
        vlines=vlines
    )

    # =========================
    # EXTRA: find "U-shape" neurons inside NS
    # =========================
    mu_mid = np.full(n_neurons, np.nan, dtype=float)
    U_index = np.full(n_neurons, np.nan, dtype=float)

    for i in range(n_neurons):
        x = Yz[i, :]
        mid_bins = bin_means(x, t_rel, MID_WIN, BIN_SIZE_S)
        mu_mid[i] = float(np.nanmean(mid_bins)) if len(mid_bins) else np.nan
        if np.isfinite(mu_pre[i]) and np.isfinite(mu_post[i]) and np.isfinite(mu_mid[i]):
            U_index[i] = min(mu_pre[i], mu_post[i]) - mu_mid[i]

    ns_u = []
    for i in ns:
        if not np.isfinite(U_index[i]):
            continue
        if (U_index[i] > U_THRESH) and (abs(mu_pre[i] - mu_post[i]) < SYM_TOL):
            ns_u.append(int(i))
    ns_u = np.array(ns_u, dtype=int)

    pd.Series(ns_u, name="neuron_index").to_csv(
        os.path.join(outdir, "neurons_ns_Ushape.csv"),
        index=False, encoding="utf-8-sig"
    )

    df_u = pd.DataFrame({
        "neuron_index": ns_u,
        "mu_pre": mu_pre[ns_u] if len(ns_u) else [],
        "mu_mid_0to639": mu_mid[ns_u] if len(ns_u) else [],
        "mu_post_639plus": mu_post[ns_u] if len(ns_u) else [],
        "U_index_minEdge_minus_mid": U_index[ns_u] if len(ns_u) else [],
        "abs_pre_minus_post": np.abs(mu_pre[ns_u] - mu_post[ns_u]) if len(ns_u) else [],
    }).sort_values("U_index_minEdge_minus_mid", ascending=False) if len(ns_u) else pd.DataFrame()

    df_u.to_csv(os.path.join(outdir, "stats_ns_Ushape.csv"),
                index=False, encoding="utf-8-sig")

    print(f"\n[NS U-shape] found {len(ns_u)} neurons within NS.")
    if len(ns_u) > 0:
        print("U-shape indices:", ns_u.tolist())

    plot_group_heatmap_and_mean(
        Y_plot, t_rel_plot, ns_u, U_index,   # use U_index for sorting
        title="NS U-shape neurons (PRE & POST high, MID low)",
        out_png=os.path.join(outdir, "NS_Ushape_heatmap_mean.png"),
        vlines=vlines
    )

    # =========================
    # CD coding direction (Pre vs Post)
    # =========================
    cd, S, mu_pre_vec, mu_post_vec = compute_cd(
        Yz=Yz, t_rel=t_rel, pre_win=PRE_WIN, post_win=POST_WIN, center="mid"
    )

    # Export CD weights with meta
    df_cd = pd.DataFrame({
        "neuron_index": np.arange(n_neurons),
        "cd_weight": cd,
        "mu_pre": mu_pre,
        "mu_post_639plus": mu_post,
        "delta_post_minus_pre": delta,
        "qval_fdr": qvals,
        "sig_fdr": reject.astype(int),
        "direction": np.where(reject & (delta > 0), "up",
                     np.where(reject & (delta < 0), "down", "ns"))
    }).sort_values("cd_weight", ascending=False)

    df_cd.to_csv(os.path.join(outdir, "cd_weights.csv"), index=False, encoding="utf-8-sig")

    # Plot CD projection trajectory
    plot_cd_projection(
        t_rel=t_rel, S=S,
        pre_win=PRE_WIN, mid_win=MID_WIN, post_win=POST_WIN,
        out_png=os.path.join(outdir, "CD_projection.png"),
        vlines=vlines
    )

    # Plot heatmap sorted by CD weights
    plot_cd_sorted_heatmap(
        Y_plot=Y_plot, t_rel_plot=t_rel_plot, cd=cd,
        out_png=os.path.join(outdir, "CD_sorted_heatmap.png"),
        vlines=vlines
    )

    print("\n✅ Saved to:", outdir)
    print("Key outputs:")
    print(" - stats_all_neurons.csv")
    print(" - neurons_up.csv / neurons_down.csv / neurons_ns.csv")
    print(" - Up_heatmap_mean.png / Down_heatmap_mean.png / NS_heatmap_mean.png")
    print(" - neurons_ns_Ushape.csv / stats_ns_Ushape.csv / NS_Ushape_heatmap_mean.png")
    print(" - cd_weights.csv / CD_projection.png / CD_sorted_heatmap.png")


if __name__ == "__main__":
    main()
