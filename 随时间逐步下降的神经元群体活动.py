# -*- coding: utf-8 -*-
"""
目标：
在“非触摸状态”下，用 GLM 找出随时间逐步下降的神经元，
并且同时做两件事：
  1）画出若干代表性 neuron 在非触摸期的时间轨迹 + GLM 拟合线
  2）构造一个“非触摸衰减群体轴”，画出群体信号随时间的变化

具体流程：
1) 读取 miniscope 神经数据 (Mouse1_for_ssm.npz) 和行为 Excel
2) 预处理（z-score、clip、平滑）
3) 构造三状态标签：
      state0：重聚前 200 s 独处
      state1：重聚后有同伴、无触摸（非触摸）
      state2：触摸期
4) 在 state1 上对每个 neuron 做 GLM：
      activity ~ β0 + β_time * time
   找出 β_time < 0 且显著的 neuron
5) 画：
   - 若干代表性下降 neuron 的时间轨迹
   - “非触摸衰减群体轴”的群体信号轨迹
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

# ======================= 参数设置 =======================
base_out_dir = r"F:\工作文件\RA\python\吸引子\figs"
os.makedirs(base_out_dir, exist_ok=True)

data_path = r"F:\工作文件\RA\python\吸引子\Mouse1_for_ssm.npz"
beh_path = r"F:\工作文件\RA\数据集\时间戳\FVB_Panneuronal_Reunion_Isolation_Day3_Mouse#1.xlsx"

# Reunion 在绝对时间轴上的时间（秒）
reunion_abs = 903.0

# 是否用 OASIS 去卷积（目前默认不用）
USE_OASIS = False
if USE_OASIS:
    try:
        from oasis.functions import deconvolve
    except ImportError:
        print("⚠️ 未找到 OASIS，自动关闭 USE_OASIS")
        USE_OASIS = False

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ======================= 数据加载 & 预处理 =======================
def load_neural_data(npz_path):
    """
    读取 npz: 需要包含 Y (T, N), t (T,), dt
    并做 z-score + 截断 + 一点平滑
    """
    print("=" * 60)
    print("LOADING NEURAL DATA")
    print("=" * 60)
    dat = np.load(npz_path, allow_pickle=True)
    Y = np.asarray(dat["Y"])     # (T, N)
    t = np.asarray(dat["t"])     # (T,)
    dt = float(dat["dt"].item())

    # 基本预处理：z-score + 截断 + 高斯平滑
    Y_z = (Y - Y.mean(axis=0)) / (Y.std(axis=0) + 1e-6)
    Y_z = np.clip(Y_z, -3, 3)
    Y_z = gaussian_filter1d(Y_z, sigma=1, axis=0)

    print(f"✓ T={len(t)}, N={Y.shape[1]}, dt={dt:.3f}s")
    return Y_z, t, dt


def load_behavior(beh_path, reunion_abs):
    """
    从 Excel 里读出 “重聚期开始”、“社交开始/结束” 时刻。
    返回：
      - social_intervals: [(start_abs, end_abs), ...]（绝对时间）
      - reunion_rel: Excel 里的重聚时间（相对）
    """
    print("\n" + "=" * 60)
    print("LOADING BEHAVIOR DATA")
    print("=" * 60)
    beh_df = pd.read_excel(beh_path, header=None).dropna(how="all")
    beh_df[0] = pd.to_numeric(beh_df[0], errors="coerce")
    beh_df[1] = beh_df[1].astype(str).str.strip().str.lower()

    # 找“重聚期开始”
    reunion_rows = beh_df[beh_df[1].str.contains("重聚期开始", na=False)]
    reunion_rel = float(reunion_rows.iloc[0, 0]) if len(reunion_rows) > 0 else 0.0

    # 社交开始/结束（相对时间）
    starts = beh_df[beh_df[1].str.contains("社交开始", na=False)][0].values
    ends = beh_df[beh_df[1].str.contains("社交结束", na=False)][0].values

    # 转成绝对时间
    social_intervals = [
        (reunion_abs + (s - reunion_rel), reunion_abs + (e - reunion_rel))
        for s, e in zip(starts, ends)
    ]

    print(f"✓ Found {len(social_intervals)} social bouts")
    return social_intervals, reunion_rel


def build_social_flag(t, social_intervals):
    """
    给每一个时间点打一个 0/1 标记：
      1 = 处于社交触摸区间
      0 = 不在社交触摸区间
    """
    social_flag = np.zeros_like(t, dtype=int)
    for (start, end) in social_intervals:
        social_flag[(t >= start) & (t <= end)] = 1
    return social_flag


def build_three_state_labels(t, social_intervals, reunion_abs, pre_reunion_window=200.0):
    """
    构造三状态标签：
      state0 = 重聚前 pre_reunion_window 秒内的独处期（只有一只鼠，无接触）
      state1 = 重聚后、且不在任何社交区间内（有同伴无触摸）
      state2 = 处于社交触摸区间内

    其余时间点标为 -1。
    """
    T = len(t)
    labels = np.full(T, -1, dtype=int)

    social_flag = build_social_flag(t, social_intervals)

    # state2: 社交触摸
    labels[social_flag == 1] = 2

    # state0: 重聚前 pre_reunion_window 秒内
    mask_state0 = (t >= (reunion_abs - pre_reunion_window)) & (t < reunion_abs)
    labels[mask_state0] = 0

    # state1: 重聚后且不在社交区间内
    mask_after_reunion = (t >= reunion_abs)
    mask_state1 = mask_after_reunion & (social_flag == 0)
    labels[mask_state1] = 1

    return labels


def extract_reunion_epoch(Y, t, reunion_abs, social_intervals, extra_after=60.0):
    """
    取重聚后的整段（直到最后一次社交结束 + buffer）
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


# ======================= (可选) 钙 → spike-like 活动 =======================
def calcium_to_spikes(Y):
    """
    用 OASIS 做钙去卷积，返回 z-scored spike-like 活动。
    如果 USE_OASIS=False，就直接用预处理后的 Y。
    """
    if not USE_OASIS:
        print("\n" + "=" * 60)
        print("SKIP OASIS: using preprocessed Y as 'spikes'")
        print("=" * 60)
        spikes = zscore(Y, axis=0)
        return spikes

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


# ======================= 非触摸 GLM：随时间下降 =======================
def run_non_touch_decay_glm(spikes, t, labels_3state, reunion_abs, out_dir,
                            target_states=(1,), time_ref="reunion"):
    """
    在指定的“非触摸状态”(默认 state1) 上，对每个 neuron 拟合 GLM：
        activity ~ β0 + β_time * time

    返回一个 dict，包括：
      betas, pvals, beta_time, p_time, dec_mask, dec_idx,
      t_nt, spikes_nt, t_rel
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) 只取目标的“非触摸状态”，比如 state1
    mask_valid = np.isin(labels_3state, list(target_states))
    t_nt = t[mask_valid]
    spikes_nt = spikes[mask_valid]   # (T_nt, N)

    if t_nt.size == 0:
        print("⚠️ 非触摸状态内没有时间点，检查 labels_3state / target_states")
        return None

    # 2) 定义时间变量：相对于重聚时间 或 相对于这段的最小 t
    if time_ref == "reunion":
        t_rel = t_nt - reunion_abs
    elif time_ref == "segment_min":
        t_rel = t_nt - t_nt.min()
    else:
        raise ValueError(f"Unknown time_ref: {time_ref}")

    # z-score 时间
    t_rel_z = (t_rel - t_rel.mean()) / (t_rel.std() + 1e-6)

    # 设计矩阵 X: [常数项, 时间项]
    X = np.column_stack([
        np.ones_like(t_rel_z),  # β0
        t_rel_z                 # β_time
    ])

    T_nt, N = spikes_nt.shape
    betas = np.zeros((N, 2))   # [β0, β_time]
    pvals = np.zeros((N, 2))

    print("\n" + "=" * 60)
    print("RUNNING NON-TOUCH DECAY GLM")
    print("=" * 60)
    print(f"  使用状态: {target_states}, T_non_touch = {T_nt}, N = {N}")
    print(f"  time_ref = {time_ref}")

    for i in range(N):
        y = spikes_nt[:, i]

        # 如果几乎没变化，跳过
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

    alpha = 0.05
    dec_mask = (beta_time < 0) & (p_time < alpha)
    dec_idx = np.where(dec_mask)[0]

    print(f"\n✓ Found {dec_idx.size} neurons with decreasing activity in non-touch period.")
    print("  Indices:", dec_idx)

    # β_time 分布直方图
    plt.figure(figsize=(6, 4))
    plt.hist(beta_time[~np.isnan(beta_time)], bins=30, alpha=0.7)
    plt.axvline(0, color="red", linestyle="--")
    plt.xlabel("β_time (Gaussian GLM)")
    plt.ylabel("Neuron count")
    plt.title("Distribution of β_time in non-touch period")
    plt.grid(alpha=0.3)
    hist_path = os.path.join(out_dir, "beta_time_hist_non_touch.png")
    plt.tight_layout()
    plt.savefig(hist_path, dpi=300)
    plt.close()
    print(f"✓ Saved: {hist_path}")

    # 保存数值结果
    out_npz = os.path.join(out_dir, "non_touch_decay_glm_results.npz")
    np.savez(
        out_npz,
        betas=betas,
        pvals=pvals,
        beta_time=beta_time,
        p_time=p_time,
        dec_mask=dec_mask,
        dec_idx=dec_idx,
        target_states=np.array(target_states),
        time_ref=time_ref
    )
    print(f"✓ Saved: {out_npz}")

    return {
        "betas": betas,
        "pvals": pvals,
        "beta_time": beta_time,
        "p_time": p_time,
        "dec_mask": dec_mask,
        "dec_idx": dec_idx,
        "t_nt": t_nt,
        "spikes_nt": spikes_nt,
        "t_rel": t_rel,
        "t_rel_z": t_rel_z
    }


# ======================= A. 画代表性 neuron 轨迹 =======================
def plot_example_neurons_non_touch(glm_res,
                                   t_epoch,
                                   spikes_epoch,
                                   labels_3state_epoch,
                                   reunion_abs,
                                   out_dir,
                                   n_examples=6):
    """
    画若干代表性“随时间下降”的 neuron：
      - 横轴：完整的 epoch 时间（相对重聚）
      - 只在 state1（非触摸）时间点画曲线，state2 (touch) 等地方用 NaN 掩掉 → 图上空白
    """
    betas = glm_res["betas"]
    beta_time = glm_res["beta_time"]
    dec_mask = glm_res["dec_mask"]

    # 完整时间轴（重聚对齐）
    t_rel_full = t_epoch - reunion_abs
    mask_state1 = (labels_3state_epoch == 1)

    # 只在显著下降 neuron 中选 β_time 最负的 n_examples 个
    candidate_idx = np.where(dec_mask)[0]
    if candidate_idx.size == 0:
        print("⚠️ 没有显著下降的 neuron，跳过绘制单细胞轨迹")
        return

    sorted_idx = candidate_idx[np.argsort(beta_time[candidate_idx])]  # 越负越前
    example_idx = sorted_idx[:min(n_examples, sorted_idx.size)]

    print(f"\n绘制代表性下降 neuron 轨迹（touch 段留空），indices = {example_idx}")

    for i in example_idx:
        y_full = spikes_epoch[:, i]

        # 平滑一下，只在 state1 画线，其它地方设为 NaN
        y_smooth = gaussian_filter1d(y_full, sigma=2)
        y_plot = np.where(mask_state1, y_smooth, np.nan)

        plt.figure(figsize=(7, 4))
        # 曲线：只有 non-touch 有值，touch 段自动断开
        plt.plot(t_rel_full, y_plot, label="activity (smoothed, state1)")

        # raw 点只画 non-touch
        plt.scatter(t_rel_full[mask_state1],
                    y_full[mask_state1],
                    s=5, alpha=0.3, label="raw (state1)")

        # GLM 拟合线：只在 non-touch 时间点画
        beta0, btime = betas[i]
        t_non = glm_res["t_rel"]          # 非触摸的相对时间（重聚对齐）
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



# ======================= B. 群体“非触摸衰减轴” =======================
def build_and_plot_population_ramp(glm_res,
                                   t_epoch,
                                   spikes_epoch,
                                   labels_3state_epoch,
                                   reunion_abs,
                                   out_dir):
    """
    构造一个“非触摸衰减轴”：
      w_i = -β_time_i（仅对显著下降 neuron，其余为 0）

    在完整 epoch 上计算群体信号：
      S_full(t) = spikes_epoch @ w

    但绘图时只在 state1（非触摸）位置显示，touch 等状态设为 NaN → 图上空白。
    """
    beta_time = glm_res["beta_time"]
    dec_mask = glm_res["dec_mask"]

    N = spikes_epoch.shape[1]
    w = np.zeros(N)
    w[dec_mask] = -beta_time[dec_mask]  # β_time<0 → -β_time>0

    if np.allclose(w, 0):
        print("⚠️ 没有下降 neuron，群体轴为 0，跳过绘制")
        return

    # 完整时间轴
    t_rel_full = t_epoch - reunion_abs

    # 完整群体信号（所有状态上都算出来）
    S_full = spikes_epoch @ w
    S_full = (S_full - S_full.mean()) / (S_full.std() + 1e-6)

    # 只在 non-touch (state1) 点显示，其他状态设为 NaN
    mask_state1 = (labels_3state_epoch == 1)
    S_plot = np.where(mask_state1, S_full, np.nan)

    # 再平滑一下（只在非 NaN 段上有效）
    S_smooth = gaussian_filter1d(np.nan_to_num(S_plot, nan=0.0), sigma=3)
    S_smooth = np.where(mask_state1, S_smooth, np.nan)

    plt.figure(figsize=(8, 4))
    plt.plot(t_rel_full, S_plot, alpha=0.3, label="Population signal (raw, state1)")
    plt.plot(t_rel_full, S_smooth, linewidth=2,
             label="Population signal (smoothed, state1)")
    plt.xlabel("Time relative to reunion (s)")
    plt.ylabel("Population ramp-down signal (z)")
    plt.title("Non-touch ramp-down population axis\n(touch intervals blank)")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)

    fname = os.path.join(out_dir, "population_non_touch_ramp_down_signal.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"✓ Saved: {fname}")

    # 保存群体轨迹
    out_npz = os.path.join(out_dir, "population_non_touch_ramp_down_signal.npz")
    np.savez(
        out_npz,
        t_rel_full=t_rel_full,
        S_full=S_full,
        S_plot=S_plot,
        S_smooth=S_smooth,
        w=w,
        dec_mask=dec_mask,
        beta_time=beta_time,
        labels_3state_epoch=labels_3state_epoch
    )
    print(f"✓ Saved: {out_npz}")


# ======================= 主流程 =======================
def main():
    print("\n" + "=" * 70)
    print("NON-TOUCH DECAY GLM ANALYSIS + EXAMPLES + POPULATION AXIS")
    print("=" * 70)

    # 1. 神经 & 行为数据
    Y, t, dt = load_neural_data(data_path)
    social_intervals, reunion_rel = load_behavior(beh_path, reunion_abs)

    # 2. (可选) 钙 → spike-like 活动
    spikes = calcium_to_spikes(Y)   # (T, N)

    # 3. 重聚后 epoch（和之前脚本保持一致）
    Y_epoch, t_epoch, mask_epoch = extract_reunion_epoch(
        Y, t, reunion_abs, social_intervals
    )
    spikes_epoch = spikes[mask_epoch]

    # 4. 三状态标签
    labels_3state_full = build_three_state_labels(t, social_intervals, reunion_abs)
    labels_3state_epoch = labels_3state_full[mask_epoch]

    # 5. 非触摸期间 GLM + 返回结果
    decay_out_dir = os.path.join(base_out_dir, "non_touch_decay_glm")
    print(f"\n✓ Output dir: {decay_out_dir}")

    glm_res = run_non_touch_decay_glm(
        spikes_epoch,
        t_epoch,
        labels_3state_epoch,
        reunion_abs=reunion_abs,
        out_dir=decay_out_dir,
        target_states=(1,),
        time_ref="reunion"
    )

    if glm_res is None:
        print("⚠️ GLM 未成功运行，终止后续分析")
        return

    # A) 单 neuron 轨迹（touch 段空）
    plot_example_neurons_non_touch(
        glm_res,
        t_epoch,
        spikes_epoch,
        labels_3state_epoch,
        reunion_abs,
        decay_out_dir,
        n_examples=6
    )

    # B) 群体衰减轴（touch 段空）
    build_and_plot_population_ramp(
        glm_res,
        t_epoch,
        spikes_epoch,
        labels_3state_epoch,
        reunion_abs,
        decay_out_dir
    )

    print("\n✅ Non-touch decay GLM + examples + population axis complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
