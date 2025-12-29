# -*- coding: utf-8 -*-
"""
随时间逐步下降的神经元群体活动分析（raw data 版）

功能：
1) 用 raw（z-scored）数据，在不同条件下做时间斜率 GLM + FDR + effect size：
   - non_touch  : 重聚后 state1（有同伴无触摸）
   - social_only: 重聚后 state2（触摸）
   - all_reunion: 重聚后 state1+2 全部
2) 对 non_touch 条件：
   - 画代表性下降 neuron 的轨迹（只画 state1，touch 段空着）
   - 构造 non-touch ramp-down population axis，并只在 state1 画
   - 看这个轴在 isolation / reunion / re-isolation 三段中的时间轨迹
   - 量化三个阶段的平均强度（raw + z），输出 CSV + 柱状图（disjoint phases）
3) （为“故事A：适应/满足/耗竭”加证据）新增控制分析：
   A) 控制时间趋势后的 step 检验（state1 内）：
        S(t) = α + β·time_z + γ·I(t>=onset)
      其中 onset 默认=last_social_end（若用户提供 REISO_START/END，则 onset=REISO_START）。
      输出：step_test__baseline_{mode}.csv + piecewise_fit__baseline_{mode}.png
   B) 围绕每次 social bout “结束点”做 event-aligned 平均：
      输出：event_aligned_social_end__baseline_{mode}.png + .csv + .npz

更新：
- dual baseline: baseline_mode in {"state0", "non_touch"}
- disjoint phases:
    * isolation_state0
    * reunion_non_touch_pre_last_social_state1
    * re_isolation_after_last_social_state1

注意：
- 不再用高斯滤波平滑原始数据，只做 z-score。
- frame-wise SEM 会因时间自相关而偏乐观；主要用于直观对比。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import zscore
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

# ======================= 参数设置 =======================
base_out_dir = r"F:\工作文件\RA\python\吸引子\figs"
os.makedirs(base_out_dir, exist_ok=True)

data_path = r"F:\工作文件\RA\python\吸引子\Mouse1_for_ssm.npz"
beh_path = r"F:\工作文件\RA\数据集\时间戳\FVB_Panneuronal_Reunion_Isolation_Day3_Mouse#1.xlsx"

# Reunion 绝对时间（秒）
reunion_abs = 903.0

# 是否用 OASIS 去卷积（默认不用）
USE_OASIS = False
if USE_OASIS:
    try:
        from oasis.functions import deconvolve
    except ImportError:
        print("⚠️ 未找到 OASIS，自动关闭 USE_OASIS")
        USE_OASIS = False

# 可选：手动指定 re-isolation 时间段（绝对时间，秒）
# 如果都为 None，则默认从“最后一次 social 结束”到实验结束
REISO_START = None  # 例如 1600.0
REISO_END = None    # 例如 1900.0

# event-aligned 参数（围绕 social 结束）
ALIGN_PRE_S = 60.0
ALIGN_POST_S = 180.0
ALIGN_MIN_VALID_POST_S = 30.0  # 至少要有这么长的 post（避免末端 bouts 太短）

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ======================= 数据加载 & 状态构造 =======================
def load_neural_data(npz_path):
    """
    读取 npz: 需要包含 Y (T, N), t (T,), dt
    只做 z-score，不做滤波。
    """
    print("=" * 60)
    print("LOADING NEURAL DATA (raw z-scored, no smoothing)")
    print("=" * 60)
    dat = np.load(npz_path, allow_pickle=True)
    Y = np.asarray(dat["Y"])     # (T, N)
    t = np.asarray(dat["t"])     # (T,)
    dt = float(dat["dt"].item())

    Y_z = zscore(Y, axis=0)      # raw -> z-score
    print(f"✓ T={len(t)}, N={Y.shape[1]}, dt={dt:.3f}s")
    return Y_z, t, dt


def load_behavior(beh_path, reunion_abs):
    """
    从 Excel 读出 “重聚期开始”、“社交开始/结束” 时刻。
    返回 social_intervals（绝对时间）和 reunion_rel（Excel 内的重聚相对时间）。
    """
    print("\n" + "=" * 60)
    print("LOADING BEHAVIOR DATA")
    print("=" * 60)
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

    print(f"✓ Found {len(social_intervals)} social bouts")
    return social_intervals, reunion_rel


def build_social_flag(t, social_intervals):
    flag = np.zeros_like(t, dtype=int)
    for (start, end) in social_intervals:
        flag[(t >= start) & (t <= end)] = 1
    return flag


def build_three_state_labels(t, social_intervals, reunion_abs, pre_reunion_window=200.0):
    """
    state0: 重聚前 pre_reunion_window 秒内独处
    state1: 重聚后，非触摸（social_flag=0）
    state2: 重聚后，触摸（social_flag=1）
    其他时间点设为 -1。
    """
    labels = np.full(len(t), -1, dtype=int)
    social_flag = build_social_flag(t, social_intervals)

    labels[social_flag == 1] = 2
    mask_state0 = (t >= (reunion_abs - pre_reunion_window)) & (t < reunion_abs)
    labels[mask_state0] = 0
    mask_after_reunion = (t >= reunion_abs)
    mask_state1 = mask_after_reunion & (social_flag == 0)
    labels[mask_state1] = 1
    return labels


def extract_reunion_epoch(Y, t, reunion_abs, social_intervals, extra_after=60.0):
    """
    取重聚后的整段（直到最后一次社交结束 + buffer）。
    """
    print("\n" + "=" * 60)
    print("EXTRACTING REUNION EPOCH")
    print("=" * 60)

    is_after = t >= reunion_abs
    if len(social_intervals) > 0:
        end_all = max(e for (_, e) in social_intervals) + extra_after
        is_before_end = t <= end_all
        mask = is_after & is_before_end
    else:
        mask = is_after

    Y_epoch = Y[mask]
    t_epoch = t[mask]
    duration = t_epoch[-1] - t_epoch[0]
    print(f"✓ Extracted {len(t_epoch)} frames ({duration:.1f}s)")
    return Y_epoch, t_epoch, mask


def calcium_to_spikes(Y):
    """
    如 USE_OASIS=False，则直接用 Y（z-scored）当作活动。
    """
    if not USE_OASIS:
        print("\n" + "=" * 60)
        print("SKIP OASIS: using Y (z-scored) as 'spikes'")
        print("=" * 60)
        return Y

    print("\n" + "=" * 60)
    print("RUNNING OASIS DECONVOLUTION")
    print("=" * 60)
    T, N = Y.shape
    spikes = np.zeros_like(Y)
    for i in range(N):
        F_trace = Y[:, i]
        c, s, *_ = deconvolve(F_trace, penalty=1)
        spikes[:, i] = s
    spikes_z = zscore(spikes, axis=0)
    print("✓ OASIS complete")
    return spikes_z


# ======================= 时间斜率 GLM + FDR + effect size =======================
def run_time_slope_glm(spikes,
                       t,
                       labels_3state,
                       reunion_abs,
                       out_dir,
                       target_states,
                       condition_name,
                       time_ref="reunion",
                       fdr_alpha=0.05,
                       effect_thr=0.10):
    """
    通用时间斜率 GLM：
      activity ~ β0 + β_time * time_z
    """
    os.makedirs(out_dir, exist_ok=True)

    mask_valid = np.isin(labels_3state, list(target_states))
    t_cond = t[mask_valid]
    spikes_cond = spikes[mask_valid]

    if t_cond.size == 0:
        print(f"⚠️ 条件 {condition_name} 下没有时间点，跳过")
        return None

    if time_ref == "reunion":
        t_rel = t_cond - reunion_abs
    elif time_ref == "segment_min":
        t_rel = t_cond - t_cond.min()
    else:
        raise ValueError(f"Unknown time_ref: {time_ref}")

    t_rel_z = (t_rel - t_rel.mean()) / (t_rel.std() + 1e-6)

    X = np.column_stack([
        np.ones_like(t_rel_z),
        t_rel_z
    ])

    T_cond, N = spikes_cond.shape
    betas = np.zeros((N, 2))
    pvals = np.zeros((N, 2))

    print("\n" + "=" * 60)
    print(f"RUNNING TIME-SLOPE GLM [{condition_name}]")
    print("=" * 60)
    print(f"  target_states = {target_states}, T = {T_cond}, N = {N}")
    print(f"  FDR alpha = {fdr_alpha}, effect_thr = {effect_thr}")

    for i in range(N):
        y = spikes_cond[:, i]
        if np.allclose(y, y[0]):
            betas[i, :] = np.nan
            pvals[i, :] = np.nan
            continue

        model = sm.GLM(y, X, family=sm.families.Gaussian())
        try:
            res = model.fit()
            betas[i, :] = res.params
            pvals[i, :] = res.pvalues
        except Exception as e:
            print(f"  Neuron {i} GLM failed: {e}")
            betas[i, :] = np.nan
            pvals[i, :] = np.nan

    beta_time = betas[:, 1]
    p_time = pvals[:, 1]

    # FDR 校正
    p_fdr = np.full_like(p_time, np.nan)
    valid = ~np.isnan(p_time)
    if valid.sum() > 0:
        _, p_fdr_valid, _, _ = multipletests(
            p_time[valid],
            alpha=fdr_alpha,
            method="fdr_bh"
        )
        p_fdr[valid] = p_fdr_valid

    # 宽松 & 严格筛选
    dec_mask_loose = (beta_time < 0) & (p_time < 0.05)
    dec_idx_loose = np.where(dec_mask_loose)[0]

    dec_mask_strict = (beta_time < -effect_thr) & (p_fdr < fdr_alpha)
    dec_idx_strict = np.where(dec_mask_strict)[0]

    print(f"\n[{condition_name}] LOOSE  : β_time < 0 & p_time < 0.05 → {dec_idx_loose.size} neurons")
    print(" indices:", dec_idx_loose)
    print(f"[{condition_name}] STRICT : β_time < -{effect_thr} & FDR < {fdr_alpha} → {dec_idx_strict.size} neurons")
    print(" indices:", dec_idx_strict)

    # 直方图：所有 neuron vs 严格筛选
    finite_beta = beta_time[~np.isnan(beta_time)]
    if finite_beta.size > 0:
        bins = np.linspace(finite_beta.min(), finite_beta.max(), 25)
        plt.figure(figsize=(7, 4))
        plt.hist(finite_beta, bins=bins, alpha=0.4, label="All neurons")
        if dec_idx_strict.size > 0:
            plt.hist(beta_time[dec_mask_strict], bins=bins, alpha=0.8,
                     label="Strict: β_time < -thr & FDR<α")
        plt.axvline(0, color="red", linestyle="--", label="β_time = 0")
        plt.xlabel("β_time (Gaussian GLM)")
        plt.ylabel("Neuron count")
        plt.title(f"β_time in {condition_name} period\n(all vs strictly decreasing)")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8)

        hist_path = os.path.join(out_dir, f"beta_time_hist_{condition_name}_FDR_effect.png")
        plt.tight_layout()
        plt.savefig(hist_path, dpi=300)
        plt.close()
        print(f"✓ Saved: {hist_path}")

    out_npz = os.path.join(out_dir, f"time_slope_{condition_name}_results.npz")
    np.savez(
        out_npz,
        betas=betas,
        pvals=pvals,
        beta_time=beta_time,
        p_time=p_time,
        p_fdr=p_fdr,
        dec_mask_loose=dec_mask_loose,
        dec_idx_loose=dec_idx_loose,
        dec_mask_strict=dec_mask_strict,
        dec_idx_strict=dec_idx_strict,
        target_states=np.array(target_states),
        time_ref=time_ref,
        fdr_alpha=fdr_alpha,
        effect_thr=effect_thr
    )
    print(f"✓ Saved: {out_npz}")

    return {
        "betas": betas,
        "pvals": pvals,
        "beta_time": beta_time,
        "p_time": p_time,
        "p_fdr": p_fdr,
        "dec_mask": dec_mask_loose,      # 兼容老字段
        "dec_idx": dec_idx_loose,
        "dec_mask_strict": dec_mask_strict,
        "dec_idx_strict": dec_idx_strict,
        "t_cond": t_cond,
        "spikes_cond": spikes_cond,
        "t_rel": t_rel,
        "t_rel_z": t_rel_z,
        "condition_name": condition_name
    }


# ======================= A. 单 neuron 轨迹（touch 段空） =======================
def plot_example_neurons_non_touch(glm_res,
                                   t_epoch,
                                   spikes_epoch,
                                   labels_3state_epoch,
                                   reunion_abs,
                                   out_dir,
                                   n_examples=6):
    """
    用 non-touch 条件的 GLM 结果，画若干代表性下降 neuron：
    - 横轴：完整 reunion epoch 的时间（相对重聚）
    - 只在 state1（非触摸）画线，touch 等 state 用 NaN 掩掉
    """
    betas = glm_res["betas"]
    beta_time = glm_res["beta_time"]
    dec_mask_strict = glm_res["dec_mask_strict"]
    if dec_mask_strict.sum() == 0:
        print("⚠️ non-touch 条件下严格筛选没有 neuron，改用宽松 dec_mask")
        dec_mask_strict = glm_res["dec_mask"]

    t_rel_full = t_epoch - reunion_abs
    mask_state1 = (labels_3state_epoch == 1)

    candidate_idx = np.where(dec_mask_strict)[0]
    if candidate_idx.size == 0:
        print("⚠️ 仍然没有可用 neuron，跳过单细胞绘图")
        return

    sorted_idx = candidate_idx[np.argsort(beta_time[candidate_idx])]
    example_idx = sorted_idx[:min(n_examples, sorted_idx.size)]
    print(f"\n绘制代表性下降 neuron 轨迹（touch 段空），indices = {example_idx}")

    for i in example_idx:
        y_full = spikes_epoch[:, i]
        y_plot = np.where(mask_state1, y_full, np.nan)  # 只画 state1

        plt.figure(figsize=(7, 4))
        plt.plot(t_rel_full, y_plot, alpha=0.8, label="activity (state1 only)")
        plt.scatter(t_rel_full[mask_state1],
                    y_full[mask_state1],
                    s=4, alpha=0.3, label="raw (state1)")

        beta0, btime = betas[i]
        t_non = glm_res["t_rel"]
        t_non_z = glm_res["t_rel_z"]
        y_fit = beta0 + btime * t_non_z
        plt.plot(t_non, y_fit, "r", linewidth=2,
                 label=f"GLM fit (β_time={btime:.3g})")

        plt.xlabel("Time relative to reunion (s)")
        plt.ylabel("Activity (z-scored)")
        plt.title(f"Neuron {i} in non-touch state (state1)\n(touch intervals blank)")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8)

        fname = os.path.join(out_dir, f"example_neuron_{i}_non_touch_trace.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=300)
        plt.close()
        print(f"✓ Saved: {fname}")


# ======================= B. non-touch 群体轴（只画 state1） =======================
def build_and_plot_population_ramp_non_touch(glm_res,
                                             t_epoch,
                                             spikes_epoch,
                                             labels_3state_epoch,
                                             reunion_abs,
                                             out_dir):
    """
    构造 non-touch ramp-down axis：
      w_i = -β_time_i（只对严格筛选下降 neuron，其他为 0）

    在 reunion epoch 上计算 S_full(t) = spikes_epoch @ w。
    为避免 touch 段影响尺度：本函数绘图时用 state1 点的 mean/std 做 z-score（仅用于展示）。
    """
    beta_time = glm_res["beta_time"]
    dec_mask_strict = glm_res["dec_mask_strict"]
    if dec_mask_strict.sum() == 0:
        print("⚠️ non-touch 条件下严格筛选没有 neuron，改用宽松 dec_mask")
        dec_mask_strict = glm_res["dec_mask"]

    N = spikes_epoch.shape[1]
    w = np.zeros(N)
    w[dec_mask_strict] = -beta_time[dec_mask_strict]

    if np.allclose(w, 0):
        print("⚠️ non-touch ramp-down axis 为 0，跳过绘制")
        return None

    t_rel_full = t_epoch - reunion_abs
    S_raw = spikes_epoch @ w

    mask_state1 = (labels_3state_epoch == 1)
    if mask_state1.sum() == 0:
        print("⚠️ reunion epoch 中没有 state1 点，无法绘制 state1-only")
        return w

    mu = S_raw[mask_state1].mean()
    sd = S_raw[mask_state1].std() + 1e-6
    S_full = (S_raw - mu) / sd

    S_plot = np.where(mask_state1, S_full, np.nan)

    plt.figure(figsize=(8, 4))
    plt.plot(t_rel_full, S_plot, linewidth=1.0,
             label="Population ramp-down signal (state1 only; z by state1)")
    plt.xlabel("Time relative to reunion (s)")
    plt.ylabel("Population signal (z)")
    plt.title("Non-touch ramp-down population axis\n(touch intervals blank)")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)

    fname = os.path.join(out_dir, "population_non_touch_ramp_down_signal.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"✓ Saved: {fname}")

    out_npz = os.path.join(out_dir, "population_non_touch_ramp_down_signal.npz")
    np.savez(
        out_npz,
        t_rel_full=t_rel_full,
        S_raw=S_raw,
        S_full=S_full,
        S_plot=S_plot,
        w=w,
        dec_mask_strict=dec_mask_strict,
        beta_time=beta_time,
        display_norm="state1",
        display_mu=float(mu),
        display_sd=float(sd)
    )
    print(f"✓ Saved: {out_npz}")

    return w


# ======================= C. baseline 归一化工具 =======================
def zscore_by_baseline(S_raw, labels_3state_full, baseline_mode):
    """
    对 1D 时间序列 S_raw 做 z-score，但 mean/std 只在 baseline_mask 上计算。
    baseline_mode:
      - "state0"   : labels==0
      - "non_touch": labels in {0,1}
    """
    if baseline_mode == "state0":
        baseline_mask = (labels_3state_full == 0)
        baseline_def = "state0 (pre-reunion isolation)"
    elif baseline_mode == "non_touch":
        baseline_mask = np.isin(labels_3state_full, [0, 1])
        baseline_def = "non_touch (states 0+1)"
    else:
        raise ValueError(f"Unknown baseline_mode: {baseline_mode}")

    if baseline_mask.sum() == 0:
        print(f"⚠️ baseline_mask 为空（{baseline_mode}），fallback 用全程 mean/std（不推荐）")
        mu = float(np.nanmean(S_raw))
        sd = float(np.nanstd(S_raw) + 1e-6)
    else:
        mu = float(np.nanmean(S_raw[baseline_mask]))
        sd = float(np.nanstd(S_raw[baseline_mask]) + 1e-6)

    S_z = (S_raw - mu) / sd
    return S_z, mu, sd, baseline_def


def summarize_three_phases(t_rel_all,
                           S_all_raw,
                           S_all_z,
                           iso_mask,
                           reunion_nt_mask,
                           reiso_mask,
                           out_dir,
                           tag):
    """
    量化三个阶段的平均强度：输出 CSV + bar plot。

    注意：SEM/STD 目前把每个 frame 当作独立样本（时间自相关会让 SEM 偏乐观）。
          主要用于直观对比/汇报；严格统计可后续做 bin 或 block bootstrap。
    """
    os.makedirs(out_dir, exist_ok=True)

    phases = [
        ("isolation_state0", iso_mask),
        ("reunion_non_touch_pre_last_social_state1", reunion_nt_mask),
        ("re_isolation_after_last_social_state1", reiso_mask),
    ]

    rows = []
    for name, m in phases:
        n = int(m.sum())
        if n == 0:
            rows.append({
                "phase": name,
                "n_frames": 0,
                "mean_raw": np.nan,
                "std_raw": np.nan,
                "mean_z": np.nan,
                "std_z": np.nan,
                "sem_z": np.nan,
                "t_min_s": np.nan,
                "t_max_s": np.nan,
            })
            continue

        raw = S_all_raw[m]
        z = S_all_z[m]
        rows.append({
            "phase": name,
            "n_frames": n,
            "mean_raw": float(np.nanmean(raw)),
            "std_raw": float(np.nanstd(raw)),
            "mean_z": float(np.nanmean(z)),
            "std_z": float(np.nanstd(z)),
            "sem_z": float(np.nanstd(z) / np.sqrt(n)),
            "t_min_s": float(np.nanmin(t_rel_all[m])),
            "t_max_s": float(np.nanmax(t_rel_all[m])),
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, f"component_phase_summary__{tag}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✓ Saved: {csv_path}")
    print(df)

    # bar plot（z）
    plt.figure(figsize=(7.6, 4))
    x = np.arange(len(df))
    means = df["mean_z"].values
    errs = df["sem_z"].values
    plt.bar(x, means, yerr=errs, capsize=4)
    plt.xticks(x, df["phase"].values, rotation=20, ha="right")
    plt.ylabel("Mean component (z)")
    plt.title("Mean component by phase (±SEM, frame-wise)")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"component_phase_mean_bar__{tag}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"✓ Saved: {fig_path}")

    return df


# ======================= D. 故事A控制：time + step 检验（state1 内） =======================
def piecewise_step_test_state1(t_full,
                               S_all_z,
                               labels_3state_full,
                               reunion_abs,
                               onset_abs,
                               dt,
                               out_dir,
                               tag,
                               use_hac=True):
    """
    在 state1 内测试：控制时间趋势后，onset 后是否还有额外下降（gamma）。

    模型： y = α + β*time_z + γ*step + ε
      - y: S_all_z（baseline 归一化后的 component）
      - time_z: (t_rel - mean)/std
      - step: I(t >= onset_abs)

    统计：默认用 HAC(Newey-West) 来缓解自相关影响（更保守）。
    输出：step_test__{tag}.csv 以及拟合图 piecewise_fit__{tag}.png
    """
    os.makedirs(out_dir, exist_ok=True)

    state1_mask = (labels_3state_full == 1) & (t_full >= reunion_abs)
    if state1_mask.sum() < 10:
        print(f"⚠️ step test: state1 点太少（n={state1_mask.sum()}），跳过")
        return None

    t1 = t_full[state1_mask]
    y = S_all_z[state1_mask]

    t_rel = t1 - reunion_abs
    time_z = (t_rel - t_rel.mean()) / (t_rel.std() + 1e-6)

    step = (t1 >= onset_abs).astype(float)

    X = np.column_stack([
        np.ones_like(time_z),
        time_z,
        step
    ])

    model = sm.OLS(y, X)

    cov_type = "nonrobust"
    cov_kwds = None
    if use_hac:
        # 取一个温和的 maxlags：约 2 秒的滞后（对 dt=0.1 -> 20 lags）
        maxlags = int(np.clip(round(2.0 / max(dt, 1e-6)), 1, 200))
        cov_type = "HAC"
        cov_kwds = {"maxlags": maxlags}

    res = model.fit(cov_type=cov_type, cov_kwds=cov_kwds)

    alpha, beta_t, gamma = res.params
    se_alpha, se_beta, se_gamma = res.bse
    p_alpha, p_beta, p_gamma = res.pvalues
    t_alpha, t_beta, t_gamma = res.tvalues

    n = int(len(y))
    n_pre = int((step == 0).sum())
    n_post = int((step == 1).sum())

    # 保存表
    df = pd.DataFrame([{
        "tag": tag,
        "cov_type": cov_type,
        "maxlags": cov_kwds["maxlags"] if cov_kwds else np.nan,
        "n_state1": n,
        "n_pre": n_pre,
        "n_post": n_post,
        "reunion_abs": float(reunion_abs),
        "onset_abs": float(onset_abs),
        "alpha": float(alpha),
        "beta_time": float(beta_t),
        "gamma_step": float(gamma),
        "se_gamma": float(se_gamma),
        "t_gamma": float(t_gamma),
        "p_gamma": float(p_gamma),
        "p_beta_time": float(p_beta)
    }])
    csv_path = os.path.join(out_dir, f"step_test__{tag}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✓ Saved: {csv_path}")
    print(df)

    # 画拟合图：对 state1 做 5s bin 的均值（更好看）
    bin_s = 5.0
    bin_w = max(int(round(bin_s / dt)), 1)

    t_rel_sorted = t_rel.copy()
    order = np.argsort(t_rel_sorted)
    t_rel_sorted = t_rel_sorted[order]
    y_sorted = y[order]
    step_sorted = step[order]
    time_z_sorted = time_z[order]

    # bin
    nb = int(np.ceil(len(y_sorted) / bin_w))
    t_bin = np.zeros(nb)
    y_bin = np.zeros(nb)
    step_bin = np.zeros(nb)
    for k in range(nb):
        sl = slice(k * bin_w, min((k + 1) * bin_w, len(y_sorted)))
        t_bin[k] = float(np.mean(t_rel_sorted[sl]))
        y_bin[k] = float(np.mean(y_sorted[sl]))
        step_bin[k] = float(np.mean(step_sorted[sl]) > 0.5)  # majority

    # 预测
    # 注意：time_z 在 bin 上重新算更合理
    time_z_bin = (t_bin - t_rel.mean()) / (t_rel.std() + 1e-6)
    Xb = np.column_stack([np.ones_like(time_z_bin), time_z_bin, step_bin])
    yhat = Xb @ res.params

    plt.figure(figsize=(9.2, 4.2))
    plt.plot(t_bin, y_bin, linewidth=1.5, label="Binned mean (state1)")
    plt.plot(t_bin, yhat, linewidth=2.0, label="Fit: α+β·time_z+γ·step")

    plt.axvline(onset_abs - reunion_abs, linestyle="--", linewidth=1.5, label="onset (step)")
    plt.xlabel("Time relative to reunion (s)")
    plt.ylabel("Component (z)")
    title = f"State1 piecewise test ({tag})\nγ(step)={gamma:.3g}, p={p_gamma:.3g} ({cov_type})"
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"piecewise_fit__{tag}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"✓ Saved: {fig_path}")

    return df


# ======================= E. 故事A佐证：围绕 social end 的 event-aligned 平均 =======================
def event_aligned_social_end(t_full,
                             S_all_z,
                             labels_3state_full,
                             social_intervals,
                             reunion_abs,
                             dt,
                             out_dir,
                             tag,
                             pre_s=60.0,
                             post_s=180.0,
                             min_valid_post_s=30.0):
    """
    对齐到每次 social bout 的“结束时刻”（touch 结束），取 [-pre_s, +post_s] 的窗口。
    输出：
      - event_aligned_social_end__{tag}.png
      - event_aligned_social_end__{tag}.csv (mean±sem, all bouts; 以及 state1-only mean)
      - event_aligned_social_end__{tag}.npz (raw aligned matrix)

    解释口径（故事A）：如果 bout 后（进入 non-touch）component 系统性下降/更低，
    那 post 的 state1-only 平均会明显更低或呈下降。
    """
    os.makedirs(out_dir, exist_ok=True)

    if len(social_intervals) == 0:
        print("⚠️ event-aligned: 没有 social bouts，跳过")
        return None

    ends = np.array([e for (_, e) in social_intervals], dtype=float)

    n_pre = int(round(pre_s / dt))
    n_post = int(round(post_s / dt))
    L = n_pre + n_post + 1
    rel_t = (np.arange(L) - n_pre) * dt  # 0 对齐到 social end

    aligned = []
    aligned_state = []

    t_min = t_full.min()
    t_max = t_full.max()

    for e in ends:
        if (e - pre_s) < t_min or (e + min_valid_post_s) > t_max:
            continue

        t0 = e - pre_s
        t1 = e + post_s

        # 用最近采样点索引
        i0 = int(np.searchsorted(t_full, t0, side="left"))
        i1 = int(np.searchsorted(t_full, t1, side="right")) - 1
        if i0 < 0 or i1 >= len(t_full) or i1 <= i0:
            continue

        # 目标长度 L：尽量通过插值到统一 rel_t
        t_seg = t_full[i0:i1+1] - e
        y_seg = S_all_z[i0:i1+1]
        st_seg = labels_3state_full[i0:i1+1]

        # 插值 y（线性）到 rel_t
        y_interp = np.interp(rel_t, t_seg, y_seg, left=np.nan, right=np.nan)

        # state 用 nearest（按最邻近的时间点）
        # 构造 nearest index
        idx_near = np.searchsorted(t_seg, rel_t, side="left")
        idx_near = np.clip(idx_near, 0, len(t_seg)-1)
        # 左右比较更近者
        left_idx = np.clip(idx_near - 1, 0, len(t_seg)-1)
        choose_left = np.abs(t_seg[left_idx] - rel_t) <= np.abs(t_seg[idx_near] - rel_t)
        idx_final = np.where(choose_left, left_idx, idx_near)
        st_near = st_seg[idx_final]

        aligned.append(y_interp)
        aligned_state.append(st_near)

    if len(aligned) < 3:
        print(f"⚠️ event-aligned: 可用 bouts 太少（n={len(aligned)}），跳过")
        return None

    A = np.vstack(aligned)             # (B, L)
    S = np.vstack(aligned_state)       # (B, L) labels

    # 总体均值
    mean_all = np.nanmean(A, axis=0)
    sem_all = np.nanstd(A, axis=0) / np.sqrt(np.sum(~np.isnan(A), axis=0).clip(min=1))

    # 只看 post 的 state1（non-touch）
    A_state1 = A.copy()
    A_state1[S != 1] = np.nan
    mean_state1 = np.nanmean(A_state1, axis=0)
    sem_state1 = np.nanstd(A_state1, axis=0) / np.sqrt(np.sum(~np.isnan(A_state1), axis=0).clip(min=1))

    # 输出 CSV
    df = pd.DataFrame({
        "rel_time_s": rel_t,
        "mean_all": mean_all,
        "sem_all": sem_all,
        "mean_state1_only": mean_state1,
        "sem_state1_only": sem_state1
    })
    csv_path = os.path.join(out_dir, f"event_aligned_social_end__{tag}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✓ Saved: {csv_path} (bouts={A.shape[0]})")

    # 图：画 all + state1-only，并标注 0 时刻
    plt.figure(figsize=(9.2, 4.2))
    plt.plot(rel_t, mean_all, linewidth=2.0, label="Mean (all states)")
    plt.fill_between(rel_t, mean_all - sem_all, mean_all + sem_all, alpha=0.2)

    plt.plot(rel_t, mean_state1, linewidth=2.0, label="Mean (state1 only)")
    plt.fill_between(rel_t, mean_state1 - sem_state1, mean_state1 + sem_state1, alpha=0.2)

    plt.axvline(0.0, linestyle="--", linewidth=1.5, label="social end (touch->non-touch)")
    plt.axhline(0.0, linestyle=":", linewidth=1.0)
    plt.xlabel("Time relative to social end (s)")
    plt.ylabel("Component (z)")
    plt.title(f"Event-aligned to social end ({tag}) | bouts={A.shape[0]}")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"event_aligned_social_end__{tag}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"✓ Saved: {fig_path}")

    # 保存 npz
    npz_path = os.path.join(out_dir, f"event_aligned_social_end__{tag}.npz")
    np.savez(
        npz_path,
        rel_t=rel_t,
        aligned=A,
        aligned_state=S,
        ends_used=len(aligned)
    )
    print(f"✓ Saved: {npz_path}")

    return df


# ======================= F. 这个成分在 isolation / reunion / re-isolation =======================
def plot_component_isolation_reiso(w,
                                   spikes_full,
                                   t_full,
                                   labels_3state_full,
                                   social_intervals,
                                   reunion_abs,
                                   dt,
                                   out_dir,
                                   baseline_mode="state0"):
    """
    用 non-touch 的群体轴 w，在整段 recording 上算：
       S_all_raw(t) = spikes_full @ w

    然后按 baseline_mode 做 z-score：
      baseline_mode="state0"   : 用重聚前 isolation（labels==0）
      baseline_mode="non_touch": 用全程 non-touch（labels in {0,1}）
    """
    if w is None or np.allclose(w, 0):
        print("⚠️ 输入的 w 为 0，无法做 isolation/re-isolation 分析")
        return None

    os.makedirs(out_dir, exist_ok=True)

    S_all_raw = spikes_full @ w
    S_all, mu, sd, baseline_def = zscore_by_baseline(S_all_raw, labels_3state_full, baseline_mode)
    t_rel_all = t_full - reunion_abs

    # --- phase masks (disjoint) ---
    iso_mask = (labels_3state_full == 0)

    last_end = max(e for (_, e) in social_intervals) if len(social_intervals) > 0 else None

    # reunion non-touch BEFORE last social end (state1 only)
    if last_end is None:
        reunion_nt_mask = (labels_3state_full == 1) & (t_full >= reunion_abs)
    else:
        reunion_nt_mask = (labels_3state_full == 1) & (t_full >= reunion_abs) & (t_full < last_end)

    # re-isolation AFTER last social end (state1 only), prefer user range if provided
    if REISO_START is not None and REISO_END is not None:
        reiso_mask = (labels_3state_full == 1) & (t_full >= REISO_START) & (t_full <= REISO_END)
        reiso_def = f"user range [{REISO_START}, {REISO_END}]"
        onset_abs = float(REISO_START)
        onset_def = "REISO_START"
    else:
        if last_end is not None:
            reiso_mask = (labels_3state_full == 1) & (t_full >= last_end)
            reiso_def = "after last social (state1 only)"
            onset_abs = float(last_end)
            onset_def = "last_social_end"
        else:
            reiso_mask = np.zeros_like(t_full, dtype=bool)
            reiso_def = "no social bouts"
            onset_abs = float(reunion_abs)  # fallback
            onset_def = "fallback_reunion_abs"

    # --- plot time series in three panels ---
    plt.figure(figsize=(10, 6))

    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(t_rel_all[iso_mask], S_all[iso_mask], linewidth=1)
    ax1.set_title(f"Isolation (pre-reunion, state0) | baseline={baseline_def}")
    ax1.set_ylabel("Component (z)")
    ax1.grid(alpha=0.3)

    ax2 = plt.subplot(3, 1, 2, sharey=ax1)
    ax2.plot(t_rel_all[reunion_nt_mask], S_all[reunion_nt_mask], linewidth=1,
             color="tab:orange")
    ax2.set_title("Reunion non-touch (state1, pre-last-social)")
    ax2.set_ylabel("Component (z)")
    ax2.grid(alpha=0.3)

    ax3 = plt.subplot(3, 1, 3, sharey=ax1)
    ax3.plot(t_rel_all[reiso_mask], S_all[reiso_mask], linewidth=1,
             color="tab:green")
    ax3.set_title(f"Re-isolation ({reiso_def})")
    ax3.set_xlabel("Time relative to reunion (s)")
    ax3.set_ylabel("Component (z)")
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(out_dir, f"population_component_iso_reiso__baseline_{baseline_mode}.png")
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"✓ Saved: {fname}")

    out_npz = os.path.join(out_dir, f"population_component_iso_reiso__baseline_{baseline_mode}.npz")
    np.savez(
        out_npz,
        t_rel_all=t_rel_all,
        S_all=S_all,
        S_all_raw=S_all_raw,
        iso_mask=iso_mask,
        reunion_nt_mask=reunion_nt_mask,
        reiso_mask=reiso_mask,
        baseline_mode=baseline_mode,
        baseline_def=baseline_def,
        baseline_mu=float(mu),
        baseline_sd=float(sd),
        last_social_end=float(last_end) if last_end is not None else np.nan,
        reiso_def=reiso_def,
        onset_abs=float(onset_abs),
        onset_def=onset_def
    )
    print(f"✓ Saved: {out_npz}")

    # --- summarize phase means ---
    tag = f"baseline_{baseline_mode}"
    summarize_three_phases(
        t_rel_all=t_rel_all,
        S_all_raw=S_all_raw,
        S_all_z=S_all,
        iso_mask=iso_mask,
        reunion_nt_mask=reunion_nt_mask,
        reiso_mask=reiso_mask,
        out_dir=out_dir,
        tag=tag
    )

    # --- StoryA controls: (A) piecewise step test within state1 ---
    piecewise_step_test_state1(
        t_full=t_full,
        S_all_z=S_all,
        labels_3state_full=labels_3state_full,
        reunion_abs=reunion_abs,
        onset_abs=onset_abs,
        dt=dt,
        out_dir=out_dir,
        tag=tag,
        use_hac=True
    )

    # --- StoryA controls: (B) event aligned to each social end ---
    event_aligned_social_end(
        t_full=t_full,
        S_all_z=S_all,
        labels_3state_full=labels_3state_full,
        social_intervals=social_intervals,
        reunion_abs=reunion_abs,
        dt=dt,
        out_dir=out_dir,
        tag=tag,
        pre_s=ALIGN_PRE_S,
        post_s=ALIGN_POST_S,
        min_valid_post_s=ALIGN_MIN_VALID_POST_S
    )

    return {
        "t_rel_all": t_rel_all,
        "S_all_raw": S_all_raw,
        "S_all": S_all,
        "iso_mask": iso_mask,
        "reunion_nt_mask": reunion_nt_mask,
        "reiso_mask": reiso_mask,
        "baseline_mode": baseline_mode,
        "baseline_def": baseline_def,
        "baseline_mu": mu,
        "baseline_sd": sd,
        "last_social_end": last_end,
        "reiso_def": reiso_def,
        "onset_abs": onset_abs,
        "onset_def": onset_def
    }


def plot_component_isolation_reiso_dual(w,
                                        spikes_full,
                                        t_full,
                                        labels_3state_full,
                                        social_intervals,
                                        reunion_abs,
                                        dt,
                                        out_dir,
                                        baseline_modes=("state0", "non_touch")):
    """
    一次性输出两套 baseline（默认：state0 + non_touch）。
    """
    results = {}
    for mode in baseline_modes:
        results[mode] = plot_component_isolation_reiso(
            w=w,
            spikes_full=spikes_full,
            t_full=t_full,
            labels_3state_full=labels_3state_full,
            social_intervals=social_intervals,
            reunion_abs=reunion_abs,
            dt=dt,
            out_dir=out_dir,
            baseline_mode=mode
        )
    return results


# ======================= 主流程 =======================
def main():
    print("\n" + "=" * 70)
    print("TIME-SLOPE ANALYSIS (raw data) + POPULATION COMPONENTS (storyA controls)")
    print("=" * 70)

    # 1. 全程神经 & 行为
    Y, t, dt = load_neural_data(data_path)
    social_intervals, _ = load_behavior(beh_path, reunion_abs)
    spikes_full = calcium_to_spikes(Y)

    # 2. 重聚后 epoch
    _, t_epoch, mask_epoch = extract_reunion_epoch(Y, t, reunion_abs, social_intervals)
    spikes_epoch = spikes_full[mask_epoch]

    # 3. 三状态标签（full + epoch）
    labels_3state_full = build_three_state_labels(t, social_intervals, reunion_abs)
    labels_3state_epoch = labels_3state_full[mask_epoch]

    # 4. 三种条件下的时间斜率 GLM（只对 epoch 做）
    glm_results = {}
    conds = [
        ("non_touch",  (1,)),
        ("social_only", (2,)),
        ("all_reunion", (1, 2))
    ]

    for cond_name, states in conds:
        out_dir_cond = os.path.join(base_out_dir, f"time_slope_{cond_name}")
        glm_results[cond_name] = run_time_slope_glm(
            spikes_epoch,
            t_epoch,
            labels_3state_epoch,
            reunion_abs=reunion_abs,
            out_dir=out_dir_cond,
            target_states=states,
            condition_name=cond_name,
            time_ref="reunion",
            fdr_alpha=0.05,
            effect_thr=0.10
        )

    # 5. non-touch 条件：单 neuron 轨迹 + 群体轴 + iso/reiso + storyA 控制分析
    glm_non = glm_results["non_touch"]
    if glm_non is None:
        print("⚠️ non-touch 条件 GLM 失败，后续群体分析跳过")
        return

    non_dir = os.path.join(base_out_dir, "time_slope_non_touch")
    print(f"\n✓ non-touch 详细图输出目录: {non_dir}")

    # A) 单 neuron 轨迹
    plot_example_neurons_non_touch(
        glm_non,
        t_epoch,
        spikes_epoch,
        labels_3state_epoch,
        reunion_abs,
        non_dir,
        n_examples=6
    )

    # B) non-touch 群体轴（只画 state1），返回 w
    w = build_and_plot_population_ramp_non_touch(
        glm_non,
        t_epoch,
        spikes_epoch,
        labels_3state_epoch,
        reunion_abs,
        non_dir
    )

    # C) 这个成分在 isolation / reunion / re-isolation（整段）——dual baseline + controls
    plot_component_isolation_reiso_dual(
        w=w,
        spikes_full=spikes_full,
        t_full=t,
        labels_3state_full=labels_3state_full,
        social_intervals=social_intervals,
        reunion_abs=reunion_abs,
        dt=dt,
        out_dir=non_dir,
        baseline_modes=("state0", "non_touch")
    )

    print("\n✅ All analysis complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
