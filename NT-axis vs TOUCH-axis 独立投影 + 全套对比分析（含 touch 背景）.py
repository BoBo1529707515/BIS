# -*- coding: utf-8 -*-
"""
Cross-condition cell alignment + dual-axis projection (4-state version)
- Cell alignment: overlap/Jaccard/hypergeom, beta correlations, sign agreement
- Axis-level: build NT-axis from state1 dec cells; TOUCH-axis from state2 dec cells
- 4 states:
    0 pre-reunion isolation
    2 reunion TOUCH (within social bouts)
    1 reunion NONTOUCH before last_social_end
    3 re-isolation NONTOUCH after last_social_end
- Global step test: within non-touch (states 1+3), test step into state3 (control time trend)
- Per-bout step test: around each bout end, test step at t=0 (control time trend), summarize across bouts
- Plots: time series with touch bouts shaded; phase bars for 4 states; per-bout gamma distribution
Prints key results to console and saves csv/png.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore, hypergeom, pearsonr, spearmanr, ttest_1samp, wilcoxon
import statsmodels.api as sm

# ===================== PATHS =====================
BASE_FIG = r"F:\工作文件\RA\python\吸引子\figs"
DATA_NPZ = r"F:\工作文件\RA\python\吸引子\Mouse1_for_ssm.npz"
BEH_XLSX = r"F:\工作文件\RA\数据集\时间戳\FVB_Panneuronal_Reunion_Isolation_Day3_Mouse#1.xlsx"

NPZ_NT = os.path.join(BASE_FIG, "time_slope_non_touch", "time_slope_non_touch_results.npz")       # state1
NPZ_T  = os.path.join(BASE_FIG, "time_slope_social_only", "time_slope_social_only_results.npz")   # state2
NPZ_AR = os.path.join(BASE_FIG, "time_slope_all_reunion", "time_slope_all_reunion_results.npz")   # state1+2

OUTDIR = os.path.join(BASE_FIG, "cross_condition__alignment_plus_dual_axes__4state")
os.makedirs(OUTDIR, exist_ok=True)

# ===================== CONSTANTS =====================
REUNION_ABS = 903.0
PRE_REUNION_WINDOW_S = 200.0

ALPHA = 0.05
THR = 0.10

USE_HAC = True
HAC_LAG_S = 2.0

BASELINE_MODES = ("state0", "non_touch")

# per-bout step window
BOUT_PRE_S = 60.0
BOUT_POST_S = 180.0
MIN_PRE_FRAMES = 50
MIN_POST_FRAMES = 50
# --- per-bout windows to scan (seconds) ---
BOUT_WINDOWS = [
    ("inst_3s",   3.0,   3.0),   # [-3, +3]
    ("inst_5s",   5.0,   5.0),   # [-5, +5]
    ("short",    10.0,  30.0),   # [-10, +30]
    ("mid",      30.0,  60.0),   # [-30, +60]
    ("long",     60.0, 180.0),   # [-60, +180] 你现在这套
]

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ===================== HELPERS =====================
def print_block(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def load_neural(npz_path):
    dat = np.load(npz_path, allow_pickle=True)
    Y = np.asarray(dat["Y"])
    t = np.asarray(dat["t"])
    dt = float(dat["dt"].item())
    Yz = zscore(Y, axis=0)
    return Yz, t, dt

def load_behavior(beh_path, reunion_abs):
    beh_df = pd.read_excel(beh_path, header=None).dropna(how="all")
    beh_df[0] = pd.to_numeric(beh_df[0], errors="coerce")
    beh_df[1] = beh_df[1].astype(str).str.strip().str.lower()

    reunion_rows = beh_df[beh_df[1].str.contains("重聚期开始", na=False)]
    reunion_rel = float(reunion_rows.iloc[0, 0]) if len(reunion_rows) > 0 else 0.0

    starts = beh_df[beh_df[1].str.contains("社交开始", na=False)][0].values
    ends = beh_df[beh_df[1].str.contains("社交结束", na=False)][0].values

    social_intervals = [
        (reunion_abs + (s - reunion_rel), reunion_abs + (e - reunion_rel))
        for s, e in zip(starts, ends)
    ]
    return social_intervals

def build_social_flag(t, intervals):
    flag = np.zeros_like(t, dtype=int)
    for s, e in intervals:
        flag[(t >= s) & (t <= e)] = 1
    return flag

def build_four_state_labels(t, social_intervals, reunion_abs, pre_reunion_window=200.0):
    """
    labels4:
      0 pre-reunion isolation window
      2 touch bouts
      1 reunion non-touch before last_social_end
      3 re-isolation non-touch after last_social_end
    """
    labels = np.full(len(t), -1, dtype=int)
    social_flag = build_social_flag(t, social_intervals)

    # state0
    mask0 = (t >= (reunion_abs - pre_reunion_window)) & (t < reunion_abs)
    labels[mask0] = 0

    # state2 touch
    labels[social_flag == 1] = 2

    # last social end
    last_end = max(e for _, e in social_intervals) if len(social_intervals) else reunion_abs

    # non-touch after reunion: split into 1 vs 3
    mask_after = (t >= reunion_abs) & (social_flag == 0)
    labels[mask_after & (t < last_end)] = 1
    labels[mask_after & (t >= last_end)] = 3

    return labels, float(last_end)

def shade_touch_background(ax, social_intervals, reunion_abs, alpha=0.15):
    for s_abs, e_abs in social_intervals:
        ax.axvspan(s_abs - reunion_abs, e_abs - reunion_abs, alpha=alpha)

def load_glm_npz(path):
    z = np.load(path, allow_pickle=True)
    beta = np.asarray(z["beta_time"])
    p_fdr = np.asarray(z["p_fdr"])
    dec_strict = np.asarray(z["dec_mask_strict"]).astype(bool)
    return beta, p_fdr, dec_strict

def make_w(beta, dec_mask):
    w = np.zeros_like(beta, dtype=float)
    w[dec_mask] = -beta[dec_mask]  # beta<0 -> -beta>0
    return w

def cosine(u, v):
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return np.nan
    return float(np.dot(u, v) / (nu * nv))

def sign_label(beta, p_fdr):
    lab = np.array(["ns"] * len(beta), dtype=object)
    lab[(beta < -THR) & (p_fdr < ALPHA)] = "dec"
    lab[(beta >  THR) & (p_fdr < ALPHA)] = "inc"
    return lab

def jaccard(a, b):
    inter = np.sum(a & b)
    uni = np.sum(a | b)
    return float(inter / uni) if uni > 0 else np.nan

def overlap_hypergeom(a, b):
    N = len(a)
    K = int(np.sum(a))
    M = int(np.sum(b))
    x = int(np.sum(a & b))
    if K == 0 or M == 0:
        return np.nan, (N, K, M, x)
    rv = hypergeom(N, M, K)
    p = rv.sf(x - 1)
    return float(p), (N, K, M, x)

def zscore_by_baseline(S_raw, labels4, baseline_mode):
    if baseline_mode == "state0":
        base = (labels4 == 0)
        base_def = "state0 (pre-reunion isolation)"
    elif baseline_mode == "non_touch":
        base = np.isin(labels4, [0, 1, 3])
        base_def = "non_touch (states 0+1+3)"
    else:
        raise ValueError(baseline_mode)

    mu = float(np.nanmean(S_raw[base])) if base.sum() else float(np.nanmean(S_raw))
    sd = float(np.nanstd(S_raw[base])) if base.sum() else float(np.nanstd(S_raw))
    sd = sd + 1e-6
    return (S_raw - mu) / sd, mu, sd, base_def

def summarize_4states(S_raw, S_z, labels4):
    """
    Mean±SEM in four disjoint states: 0,2,1,3 (ordered as story-friendly)
    """
    order = [
        ("state0_pre_reunion_isolation", 0),
        ("state2_touch", 2),
        ("state1_reunion_non_touch", 1),
        ("state3_re_isolation_non_touch", 3),
    ]
    rows = []
    for name, sid in order:
        m = (labels4 == sid)
        n = int(m.sum())
        if n == 0:
            rows.append(dict(state=name, id=sid, n_frames=0, mean_z=np.nan, sem_z=np.nan,
                             mean_raw=np.nan, std_raw=np.nan))
            continue
        rows.append(dict(
            state=name, id=sid, n_frames=n,
            mean_z=float(np.nanmean(S_z[m])),
            sem_z=float(np.nanstd(S_z[m]) / np.sqrt(n)),
            mean_raw=float(np.nanmean(S_raw[m])),
            std_raw=float(np.nanstd(S_raw[m])),
        ))
    return pd.DataFrame(rows)

def global_step_test_nontouch(labels4, t, S_z, reunion_abs, last_end_abs, dt):
    """
    Within non-touch AFTER reunion (states 1 and 3 only):
      y = a + b*time_z + g*I(state==3) + eps
    """
    m = np.isin(labels4, [1, 3]) & (t >= reunion_abs)
    if m.sum() < 50:
        return None

    t1 = t[m]
    y = S_z[m]
    t_rel = t1 - reunion_abs
    time_z = (t_rel - t_rel.mean()) / (t_rel.std() + 1e-6)

    step = (t1 >= last_end_abs).astype(float)  # equivalently: I(state==3)

    X = np.column_stack([np.ones_like(time_z), time_z, step])
    model = sm.OLS(y, X)

    cov_type = "nonrobust"
    cov_kwds = None
    if USE_HAC:
        maxlags = int(np.clip(round(HAC_LAG_S / max(dt, 1e-6)), 1, 200))
        cov_type = "HAC"
        cov_kwds = {"maxlags": maxlags}

    res = model.fit(cov_type=cov_type, cov_kwds=cov_kwds)
    a, b, g = res.params
    out = dict(
        cov_type=cov_type, maxlags=(cov_kwds["maxlags"] if cov_kwds else np.nan),
        n=int(len(y)), n_pre=int((step == 0).sum()), n_post=int((step == 1).sum()),
        alpha=float(a), beta_time=float(b), gamma_step=float(g),
        t_gamma=float(res.tvalues[2]), p_gamma=float(res.pvalues[2]),
        p_beta_time=float(res.pvalues[1]),
    )
    return out

def per_bout_step_tests(t, S_z, social_intervals, dt, reunion_abs,
                        axis_tag, baseline_tag,
                        win_tag, pre_s, post_s,
                        min_pre_frames=None, min_post_frames=None,
                        hac_lag_s=2.0):
    """
    For each bout end e:
      window [e-pre_s, e+post_s]
      y = a + b*time_z + g*I(rel_t>=0) + eps
    """

    # 自动设置最小帧数：至少覆盖窗口的 ~60%（并且至少 10 帧）
    if min_pre_frames is None:
        min_pre_frames = max(10, int(np.ceil(0.6 * pre_s / dt)))
    if min_post_frames is None:
        min_post_frames = max(10, int(np.ceil(0.6 * post_s / dt)))

    rows = []
    for bi, (_, e) in enumerate(social_intervals):
        t0 = e - pre_s
        t1 = e + post_s
        i0 = int(np.searchsorted(t, t0, side="left"))
        i1 = int(np.searchsorted(t, t1, side="right")) - 1
        if i1 <= i0:
            continue

        tt = t[i0:i1+1]
        yy = S_z[i0:i1+1]
        rel = tt - e

        pre = rel < 0
        post = rel >= 0
        if pre.sum() < min_pre_frames or post.sum() < min_post_frames:
            continue

        time_z = (rel - rel.mean()) / (rel.std() + 1e-6)
        step = (rel >= 0).astype(float)

        X = np.column_stack([np.ones_like(time_z), time_z, step])
        model = sm.OLS(yy, X)

        cov_type = "nonrobust"
        cov_kwds = None
        if USE_HAC:
            # HAC lag 不要超过窗口长度的一半，避免“lag 过大”
            eff_hac = min(hac_lag_s, 0.5 * (pre_s + post_s))
            maxlags = int(np.clip(round(eff_hac / max(dt, 1e-6)), 1, 200))
            cov_type = "HAC"
            cov_kwds = {"maxlags": maxlags}

        res = model.fit(cov_type=cov_type, cov_kwds=cov_kwds)
        a, b, g = res.params

        rows.append(dict(
            win_tag=win_tag, pre_s=pre_s, post_s=post_s,
            bout_index=bi,
            bout_end_abs=float(e),
            bout_end_rel=float(e - reunion_abs),
            n=int(len(yy)), n_pre=int(pre.sum()), n_post=int(post.sum()),
            alpha=float(a), beta_time=float(b), gamma_step=float(g),
            t_gamma=float(res.tvalues[2]), p_gamma=float(res.pvalues[2]),
            cov_type=cov_type, maxlags=(cov_kwds["maxlags"] if cov_kwds else np.nan)
        ))

    df = pd.DataFrame(rows)

    # 输出：每窗一份
    csv_path = os.path.join(
        OUTDIR, f"per_bout_step_{axis_tag}__baseline_{baseline_tag}__{win_tag}.csv"
    )
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 分布图：每窗一份
    if len(df) > 0:
        gam = df["gamma_step"].values

        plt.figure(figsize=(7.4, 4.2))
        plt.hist(gam, bins=12)
        plt.axvline(0, linestyle="--", linewidth=1.2)
        plt.xlabel("Per-bout gamma (step at bout end)")
        plt.ylabel("Count")
        plt.title(f"{axis_tag} | {baseline_tag} | {win_tag}  [-{pre_s:.0f},+{post_s:.0f}]s | n={len(df)}")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(
            OUTDIR, f"per_bout_gamma_hist_{axis_tag}__baseline_{baseline_tag}__{win_tag}.png"
        ), dpi=300)
        plt.close()

    return df, csv_path


# ===================== MAIN =====================
def main():
    # Load data
    Yz, t, dt = load_neural(DATA_NPZ)
    social_intervals = load_behavior(BEH_XLSX, REUNION_ABS)
    labels4, last_end_abs = build_four_state_labels(t, social_intervals, REUNION_ABS, PRE_REUNION_WINDOW_S)
    t_rel = t - REUNION_ABS

    print_block("4-STATE CHECK")
    counts = {sid: int((labels4 == sid).sum()) for sid in [0,1,2,3,-1]}
    print("Frame counts:", counts)
    print(f"last_social_end_abs={last_end_abs:.3f}  (rel_to_reunion={last_end_abs-REUNION_ABS:.3f}s)")
    print(f"n_social_bouts={len(social_intervals)}")

    # Load GLM results (cell alignment)
    beta_nt, fdr_nt, dec_nt = load_glm_npz(NPZ_NT)
    beta_t,  fdr_t,  dec_t  = load_glm_npz(NPZ_T)
    beta_ar, fdr_ar, dec_ar = load_glm_npz(NPZ_AR)
    N = len(beta_nt)

    # ---------------- CELL ALIGNMENT ----------------
    print_block("CELL ALIGNMENT (STRICT dec cells)")
    pairs = [
        ("non_touch(state1)", dec_nt, "touch(state2)", dec_t),
        ("non_touch(state1)", dec_nt, "all_reunion(state1+2)", dec_ar),
        ("touch(state2)", dec_t, "all_reunion(state1+2)", dec_ar),
    ]
    overlap_rows = []
    for nameA, A, nameB, B in pairs:
        p_hyp, info = overlap_hypergeom(A, B)
        N0, K, M, x = info
        overlap_rows.append(dict(
            pair=f"{nameA} vs {nameB}",
            N=N0, A_size=K, B_size=M, overlap=x,
            jaccard=jaccard(A, B),
            hypergeom_p_ge_overlap=p_hyp
        ))
    df_overlap = pd.DataFrame(overlap_rows)
    print(df_overlap.to_string(index=False))
    df_overlap.to_csv(os.path.join(OUTDIR, "cell_alignment__overlap_strict.csv"), index=False, encoding="utf-8-sig")

    def corr_report(x, y, label):
        m = (~np.isnan(x)) & (~np.isnan(y))
        pr, pp = pearsonr(x[m], y[m])
        sr, sp = spearmanr(x[m], y[m])
        return dict(pair=label, pearson_r=pr, pearson_p=pp, spearman_rho=sr, spearman_p=sp)

    df_corr = pd.DataFrame([
        corr_report(beta_nt, beta_t,  "beta: non_touch vs touch"),
        corr_report(beta_nt, beta_ar, "beta: non_touch vs all_reunion"),
        corr_report(beta_t,  beta_ar, "beta: touch vs all_reunion"),
    ])
    print("\nBeta correlations:")
    print(df_corr.to_string(index=False))
    df_corr.to_csv(os.path.join(OUTDIR, "cell_alignment__beta_correlations.csv"), index=False, encoding="utf-8-sig")

    print_block("SIGN AGREEMENT (dec/inc/ns) 3x3 CROSSTABS")
    lab_nt = sign_label(beta_nt, fdr_nt)
    lab_t  = sign_label(beta_t,  fdr_t)
    lab_ar = sign_label(beta_ar, fdr_ar)

    ct_nt_t = pd.crosstab(pd.Series(lab_nt, name="non_touch"), pd.Series(lab_t, name="touch"))
    ct_nt_ar = pd.crosstab(pd.Series(lab_nt, name="non_touch"), pd.Series(lab_ar, name="all_reunion"))
    ct_t_ar = pd.crosstab(pd.Series(lab_t, name="touch"), pd.Series(lab_ar, name="all_reunion"))

    print("\nnon_touch vs touch:\n", ct_nt_t)
    print("\nnon_touch vs all_reunion:\n", ct_nt_ar)
    print("\ntouch vs all_reunion:\n", ct_t_ar)

    ct_nt_t.to_csv(os.path.join(OUTDIR, "sign_crosstab__non_touch_vs_touch.csv"), encoding="utf-8-sig")
    ct_nt_ar.to_csv(os.path.join(OUTDIR, "sign_crosstab__non_touch_vs_all_reunion.csv"), encoding="utf-8-sig")
    ct_t_ar.to_csv(os.path.join(OUTDIR, "sign_crosstab__touch_vs_all_reunion.csv"), encoding="utf-8-sig")

    shared_nt_t = np.where(dec_nt & dec_t)[0]
    print_block("KEY SETS (STRICT dec)")
    print(f"N={N}")
    print(f"non_touch dec_strict: {int(dec_nt.sum())}")
    print(f"touch     dec_strict: {int(dec_t.sum())}")
    print(f"shared(dec in both):  {len(shared_nt_t)}")
    print("shared indices:", shared_nt_t.tolist())

    # ---------------- AXIS LEVEL ----------------
    print_block("AXIS-LEVEL (4-state): NT-axis vs TOUCH-axis projections")
    w_nt = make_w(beta_nt, dec_nt)
    w_t  = make_w(beta_t,  dec_t)
    cos_w = cosine(w_nt, w_t)
    print(f"Cosine similarity cos(w_nt, w_touch) = {cos_w:.6g}")

    S_raw_nt = Yz @ w_nt
    S_raw_t  = Yz @ w_t

    pd.DataFrame([{
        "n_neurons": N,
        "n_dec_nt_strict": int(dec_nt.sum()),
        "n_dec_touch_strict": int(dec_t.sum()),
        "cosine_w_nt_vs_w_touch": cos_w,
        "last_social_end_rel_s": float(last_end_abs - REUNION_ABS),
    }]).to_csv(os.path.join(OUTDIR, "axis_summary.csv"), index=False, encoding="utf-8-sig")

    key_rows = []
    for baseline_mode in BASELINE_MODES:
        print_block(f"BASELINE MODE = {baseline_mode}")

        S_nt, mu_nt, sd_nt, base_def = zscore_by_baseline(S_raw_nt, labels4, baseline_mode)
        S_tz, mu_t,  sd_t,  _        = zscore_by_baseline(S_raw_t,  labels4, baseline_mode)

        # ---- timeseries dual axes (touch shaded) ----
        fig_path = os.path.join(OUTDIR, f"timeseries_dual_axes__baseline_{baseline_mode}.png")
        plt.figure(figsize=(12, 4.6))
        ax = plt.gca()
        ax.plot(t_rel, S_nt, linewidth=1.1, label=f"NT-axis (state1 dec cells; n={int(dec_nt.sum())})")
        ax.plot(t_rel, S_tz, linewidth=1.1, label=f"TOUCH-axis (state2 dec cells; n={int(dec_t.sum())})")
        ax.axvline(0.0, linestyle="--", linewidth=1.2, label="reunion")
        ax.axvline(last_end_abs - REUNION_ABS, linestyle="--", linewidth=1.2, label="last social end")
        ax.set_xlabel("Time relative to reunion (s)")
        ax.set_ylabel("Component (z)")
        ax.set_title(f"Dual-axis projection (4-state labels) | baseline={base_def}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
        ymin, ymax = ax.get_ylim()
        shade_touch_background(ax, social_intervals, REUNION_ABS, alpha=0.15)
        ax.set_ylim(ymin, ymax)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print("Saved:", fig_path)

        # ---- 4-state bar summaries (each axis) ----
        df4_nt = summarize_4states(S_raw_nt, S_nt, labels4)
        df4_t  = summarize_4states(S_raw_t,  S_tz, labels4)

        df4_nt.to_csv(os.path.join(OUTDIR, f"state4_summary__NT_axis__baseline_{baseline_mode}.csv"),
                      index=False, encoding="utf-8-sig")
        df4_t.to_csv(os.path.join(OUTDIR, f"state4_summary__TOUCH_axis__baseline_{baseline_mode}.csv"),
                     index=False, encoding="utf-8-sig")

        def barplot(df, title, out_png):
            plt.figure(figsize=(8.3, 4.2))
            x = np.arange(len(df))
            plt.bar(x, df["mean_z"].values, yerr=df["sem_z"].values, capsize=4)
            plt.xticks(x, df["state"].values, rotation=15, ha="right")
            plt.ylabel("Mean component (z)")
            plt.title(title)
            plt.grid(alpha=0.3, axis="y")
            plt.tight_layout()
            plt.savefig(out_png, dpi=300)
            plt.close()

        barplot(df4_nt, f"4-state means | NT-axis | baseline={base_def}",
                os.path.join(OUTDIR, f"bar_state4__NT_axis__baseline_{baseline_mode}.png"))
        barplot(df4_t, f"4-state means | TOUCH-axis | baseline={base_def}",
                os.path.join(OUTDIR, f"bar_state4__TOUCH_axis__baseline_{baseline_mode}.png"))

        print("\n4-state means (z): NT-axis")
        print(df4_nt[["state","n_frames","mean_z","sem_z"]].to_string(index=False))
        print("\n4-state means (z): TOUCH-axis")
        print(df4_t[["state","n_frames","mean_z","sem_z"]].to_string(index=False))

        # ---- Global step test within non-touch (states 1+3) ----
        st_nt = global_step_test_nontouch(labels4, t, S_nt, REUNION_ABS, last_end_abs, dt)
        st_t  = global_step_test_nontouch(labels4, t, S_tz, REUNION_ABS, last_end_abs, dt)

        print("\nGlobal step test within non-touch (states 1+3):  y = a + b*time_z + g*I(state3)")
        def show(tag, d):
            print(f"  {tag}: gamma={d['gamma_step']:.6g}, t={d['t_gamma']:.3g}, p={d['p_gamma']:.3g}, "
                  f"beta_p={d['p_beta_time']:.3g}, n={d['n']} (HAC maxlags={d['maxlags']})")
        show("NT-axis", st_nt)
        show("TOUCH-axis", st_t)

        pd.DataFrame([
            dict(axis="NT-axis", baseline=baseline_mode, **st_nt),
            dict(axis="TOUCH-axis", baseline=baseline_mode, **st_t),
        ]).to_csv(os.path.join(OUTDIR, f"global_step_nontouch__baseline_{baseline_mode}.csv"),
                  index=False, encoding="utf-8-sig")

        # ---- Per-bout step tests at each bout end (touch->non-touch transition) ----
        # ---- Per-bout window sensitivity scan ----
        sens_rows = []

        def bout_summary(df, axis_name):
            if len(df) == 0:
                return dict(axis=axis_name, n_bouts=0, mean_gamma=np.nan, median_gamma=np.nan,
                            ttest_p=np.nan, wilcoxon_p=np.nan)
            gam = df["gamma_step"].values
            t_p = ttest_1samp(gam, 0.0, nan_policy="omit").pvalue if len(gam) >= 3 else np.nan
            try:
                w_p = wilcoxon(gam).pvalue
            except Exception:
                w_p = np.nan
            return dict(axis=axis_name, n_bouts=len(gam),
                        mean_gamma=float(np.nanmean(gam)),
                        median_gamma=float(np.nanmedian(gam)),
                        ttest_p=float(t_p) if np.isfinite(t_p) else np.nan,
                        wilcoxon_p=float(w_p) if np.isfinite(w_p) else np.nan)

        print("\nPer-bout window scan @ bout end:")
        for win_tag, pre_s, post_s in BOUT_WINDOWS:
            # short windows: shorter HAC lag; long windows: keep 2s
            hac_lag = 0.5 if (pre_s + post_s) <= 20 else 2.0

            df_bout_nt, _ = per_bout_step_tests(
                t, S_nt, social_intervals, dt, REUNION_ABS,
                axis_tag="NT_axis", baseline_tag=baseline_mode,
                win_tag=win_tag, pre_s=pre_s, post_s=post_s,
                hac_lag_s=hac_lag
            )
            df_bout_t, _ = per_bout_step_tests(
                t, S_tz, social_intervals, dt, REUNION_ABS,
                axis_tag="TOUCH_axis", baseline_tag=baseline_mode,
                win_tag=win_tag, pre_s=pre_s, post_s=post_s,
                hac_lag_s=hac_lag
            )

            bs_nt = bout_summary(df_bout_nt, "NT-axis")
            bs_t = bout_summary(df_bout_t, "TOUCH-axis")

            sens_rows.append(dict(baseline=baseline_mode, win_tag=win_tag, pre_s=pre_s, post_s=post_s, **bs_nt))
            sens_rows.append(dict(baseline=baseline_mode, win_tag=win_tag, pre_s=pre_s, post_s=post_s, **bs_t))

            print(f"  [{win_tag:8s}] NT-axis    n={bs_nt['n_bouts']:2d} mean_g={bs_nt['mean_gamma']:+.3f} "
                  f"t_p={bs_nt['ttest_p']:.3g} w_p={bs_nt['wilcoxon_p']:.3g}")
            print(f"  [{win_tag:8s}] TOUCH-axis n={bs_t['n_bouts']:2d} mean_g={bs_t['mean_gamma']:+.3f} "
                  f"t_p={bs_t['ttest_p']:.3g} w_p={bs_t['wilcoxon_p']:.3g}")

        df_sens = pd.DataFrame(sens_rows)
        df_sens.to_csv(
            os.path.join(OUTDIR, f"per_bout_window_sensitivity__baseline_{baseline_mode}.csv"),
            index=False, encoding="utf-8-sig"
        )

        # ---- one-line key summary for copy/paste ----
        key_rows.append(dict(
            baseline=baseline_mode,
            cos_w_nt_vs_w_touch=cos_w,
            global_gamma_nontouch__NT=st_nt["gamma_step"],
            global_p_nontouch__NT=st_nt["p_gamma"],
            global_gamma_nontouch__TOUCH=st_t["gamma_step"],
            global_p_nontouch__TOUCH=st_t["p_gamma"],
            per_bout_mean_gamma__NT=bs_nt["mean_gamma"],
            per_bout_ttest_p__NT=bs_nt["ttest_p"],
            per_bout_mean_gamma__TOUCH=bs_t["mean_gamma"],
            per_bout_ttest_p__TOUCH=bs_t["ttest_p"],
        ))

    print_block("FINAL KEY SUMMARY (4-state)")
    df_key = pd.DataFrame(key_rows)
    print(df_key.to_string(index=False))
    df_key.to_csv(os.path.join(OUTDIR, "KEY_SUMMARY_TABLE__4state.csv"), index=False, encoding="utf-8-sig")

    print("\n✅ Done. Outputs saved to:")
    print(OUTDIR)

# tiny helper to avoid a stray f-string usage above (kept harmless)
def axis_name_path(x): return x

if __name__ == "__main__":
    main()
