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
   - 看这个轴在 isolation / reunion / re-isolation 三段中的时间轨迹（raw）

注意：不再用高斯滤波平滑原始数据，只做 z-score。
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
    T = len(t)
    labels = np.full(T, -1, dtype=int)
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

    target_states: 元组，如 (1,) / (2,) / (1,2)
    condition_name: 用于打印和文件命名（"non_touch" / "social_only" / "all_reunion"）

    返回 glm_res 字典（供后面 non-touch 使用）。
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
        plt.hist(finite_beta,
                 bins=bins,
                 alpha=0.4,
                 label="All neurons")
        if dec_idx_strict.size > 0:
            plt.hist(beta_time[dec_mask_strict],
                     bins=bins,
                     alpha=0.8,
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

    在 reunion epoch 上计算 S_full(t) = spikes_epoch @ w，
    但只在 state1 时间点显示，其他状态 NaN → 图上空。
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
        return

    t_rel_full = t_epoch - reunion_abs
    S_full = spikes_epoch @ w
    S_full = (S_full - S_full.mean()) / (S_full.std() + 1e-6)

    mask_state1 = (labels_3state_epoch == 1)
    S_plot = np.where(mask_state1, S_full, np.nan)

    plt.figure(figsize=(8, 4))
    plt.plot(t_rel_full, S_plot, linewidth=1.0,
             label="Population ramp-down signal (state1 only)")
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
        S_full=S_full,
        S_plot=S_plot,
        w=w,
        dec_mask_strict=dec_mask_strict,
        beta_time=beta_time
    )
    print(f"✓ Saved: {out_npz}")

    return w


# ======================= C. 这个成分在 isolation / re-isolation 的情况 =======================
def plot_component_isolation_reiso(w,
                                   spikes_full,
                                   t_full,
                                   labels_3state_full,
                                   social_intervals,
                                   reunion_abs,
                                   out_dir):
    """
    用 non-touch 的群体轴 w，在整段 recording 上算：
       S_all(t) = spikes_full @ w   （z-score）

    然后分别看：
      - isolation：state0（重聚前 200 s）
      - reunion non-touch：state1
      - re-isolation：默认从最后一次 social 结束到结尾，或者用户指定的 REISO_START/END
    """
    if w is None or np.allclose(w, 0):
        print("⚠️ 输入的 w 为 0，无法做 isolation/re-isolation 分析")
        return

    S_all = spikes_full @ w
    S_all = (S_all - S_all.mean()) / (S_all.std() + 1e-6)
    t_rel_all = t_full - reunion_abs

    # isolation = state0
    iso_mask = (labels_3state_full == 0)

    # non-touch during reunion = state1
    non_touch_mask = (labels_3state_full == 1)

    # re-isolation：优先用用户给的时间，否则从最后一次 social 结束到最后
    if REISO_START is not None and REISO_END is not None:
        reiso_mask = (t_full >= REISO_START) & (t_full <= REISO_END)
    else:
        if len(social_intervals) > 0:
            last_end = max(e for (_, e) in social_intervals)
            reiso_mask = (t_full >= last_end)
        else:
            reiso_mask = np.zeros_like(t_full, dtype=bool)

    plt.figure(figsize=(10, 6))

    # 3 个子图：iso / non-touch / reiso
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(t_rel_all[iso_mask], S_all[iso_mask], linewidth=1)
    ax1.set_title("Isolation (pre-reunion, state0)")
    ax1.set_ylabel("Component (z)")
    ax1.grid(alpha=0.3)

    ax2 = plt.subplot(3, 1, 2, sharey=ax1)
    ax2.plot(t_rel_all[non_touch_mask], S_all[non_touch_mask], linewidth=1,
             color="tab:orange")
    ax2.set_title("Reunion non-touch (state1)")
    ax2.set_ylabel("Component (z)")
    ax2.grid(alpha=0.3)

    ax3 = plt.subplot(3, 1, 3, sharey=ax1)
    ax3.plot(t_rel_all[reiso_mask], S_all[reiso_mask], linewidth=1,
             color="tab:green")
    ax3.set_title("Re-isolation (after last social)")
    ax3.set_xlabel("Time relative to reunion (s)")
    ax3.set_ylabel("Component (z)")
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(out_dir, "population_component_iso_reiso.png")
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"✓ Saved: {fname}")

    out_npz = os.path.join(out_dir, "population_component_iso_reiso.npz")
    np.savez(
        out_npz,
        t_rel_all=t_rel_all,
        S_all=S_all,
        iso_mask=iso_mask,
        non_touch_mask=non_touch_mask,
        reiso_mask=reiso_mask
    )
    print(f"✓ Saved: {out_npz}")


# ======================= 主流程 =======================
def main():
    print("\n" + "=" * 70)
    print("TIME-SLOPE ANALYSIS (raw data) + POPULATION COMPONENTS")
    print("=" * 70)

    # 1. 全程神经 & 行为
    Y, t, dt = load_neural_data(data_path)
    social_intervals, reunion_rel = load_behavior(beh_path, reunion_abs)
    spikes_full = calcium_to_spikes(Y)

    # 2. 重聚后 epoch
    Y_epoch, t_epoch, mask_epoch = extract_reunion_epoch(Y, t, reunion_abs, social_intervals)
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

    # 5. non-touch 条件：单 neuron 轨迹 + 群体轴 + iso/reiso 表现
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

    # C) 这个成分在 isolation / re-isolation 的情况（整段）
    plot_component_isolation_reiso(
        w,
        spikes_full,
        t,
        labels_3state_full,
        social_intervals,
        reunion_abs,
        non_dir
    )

    print("\n✅ All analysis complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
