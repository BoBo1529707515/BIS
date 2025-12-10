# 主导分类神经元绘图_独立版.py
# -*- coding: utf-8 -*-
"""
功能：
1) 读取 miniscope 神经数据 & 行为时间戳（Excel，格式与 mouse_switching_ramping_analysis.py 相同）
2) 取重聚后的 epoch（从 reunion_abs 到最后一次社交结束 + buffer）
3) 用“社交 vs 非社交”做 GLM2，得到动机维度 β_motor
4) 按 |β_motor| 排序神经元，输出贡献最大的神经元编号
5) 绘制：
   - Top20 β_motor 权重条形图
   - 重聚 epoch 时间轴上的社交触摸 + 前10个贡献最大神经元的活动轨迹
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore

# ----------------- Matplotlib 中文支持（可选） -----------------
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ======================= 参数设置 =======================
base_out_dir = r"F:\工作文件\RA\python\吸引子\figs"
os.makedirs(base_out_dir, exist_ok=True)

data_path = r"F:\工作文件\RA\python\吸引子\Mouse1_for_ssm.npz"
beh_path = r"F:\工作文件\RA\数据集\时间戳\FVB_Panneuronal_Reunion_Isolation_Day3_Mouse#1.xlsx"

# Reunion 在绝对时间轴上的时间（秒），与你 HMM 脚本保持一致
reunion_abs = 903.0

# GLM2 的时间窗长度（秒）
BIN_SIZE = 1.0

# 重聚 epoch 的尾部额外 buffer（秒）
EXTRA_AFTER = 60.0

# 输出相关
TOP_K_PRINT = 30     # 终端打印前多少个贡献最大的神经元
TOP_K_SAVE = 80      # CSV 里保存前多少个
NUM_TO_PLOT = 10     # 画前多少个贡献神经元的活动轨迹
# =======================================================


# ======================= 数据加载 & 预处理 =======================
def load_neural_data(npz_path):
    """
    读取 npz: 需要包含 Y (T, N), t (T,), dt
    并做 z-score + clip + 一点平滑
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

    print(f"✓ T={len(t)}, N={Y.shape[1]}, dt≈{dt:.3f}s")
    return Y_z, t, dt


def load_behavior(beh_path, reunion_abs):
    """
    读取 Excel（格式与 mouse_switching_ramping_analysis.py 一致）：
      第 0 列：时间（相对时间）
      第 1 列：中文事件标签（如 “重聚期开始”、“社交开始”、“社交结束”）

    返回：
      - social_intervals: 列表 [(start_abs, end_abs), ...] 绝对时间
      - reunion_rel: Excel 中的重聚时间（相对）
    """
    print("\n" + "=" * 60)
    print("LOADING BEHAVIOR DATA")
    print("=" * 60)
    beh_df = pd.read_excel(beh_path, header=None).dropna(how="all")
    beh_df[0] = pd.to_numeric(beh_df[0], errors="coerce")
    beh_df[1] = beh_df[1].astype(str).str.strip().str.lower()

    # 找“重聚期开始”
    reunion_rows = beh_df[beh_df[1].str.contains("重聚期开始", na=False)]
    if len(reunion_rows) == 0:
        print("⚠️ Excel 中未找到 '重聚期开始' 行，默认 reunion_rel=0")
        reunion_rel = 0.0
    else:
        reunion_rel = float(reunion_rows.iloc[0, 0])

    # 找“社交开始 / 社交结束”（相对时间）
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
      1 = 处于社交接触区间
      0 = 非社交
    """
    social_flag = np.zeros_like(t, dtype=int)
    for (start, end) in social_intervals:
        social_flag[(t >= start) & (t <= end)] = 1
    return social_flag


def extract_reunion_epoch(Y, t, reunion_abs, social_intervals, extra_after=60.0):
    """
    取重聚后的整段 epoch：
      从 reunion_abs 开始，到最后一次社交结束 + extra_after 秒
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
    简单处理：把钙信号 z-score 一下，当作 spike-like 活动。
    （这里不做 OASIS 去卷积，以和现有 HMM 脚本一致）
    """
    print("\n" + "=" * 60)
    print("SKIP OASIS: using preprocessed Y as 'spikes'")
    print("=" * 60)
    spikes = zscore(Y, axis=0)
    return spikes


# ======================= 时间窗 binning & GLM2 =======================
def bin_data(spikes_epoch, t_epoch, social_flag_epoch, bin_size=1.0):
    """
    把 spike 活动按时间窗汇总，并给每个窗一个 0/1 的动机标签：
      1 = >50% 时间处于社交区间（高动机）
      0 = 否则（低动机）
    """
    dt = np.median(np.diff(t_epoch))
    frames_per_bin = max(1, int(round(bin_size / dt)))

    T = len(t_epoch)
    n_bins = T // frames_per_bin

    X_bins = []
    y_bins = []
    t_bins = []

    for b in range(n_bins):
        start = b * frames_per_bin
        end = start + frames_per_bin
        window_spikes = spikes_epoch[start:end]
        window_social = social_flag_epoch[start:end]

        X_bin = window_spikes.mean(axis=0)  # 每个 neuron 的平均活动
        y_bin = int(window_social.mean() > 0.5)
        t_bin = t_epoch[start:end].mean()

        X_bins.append(X_bin)
        y_bins.append(y_bin)
        t_bins.append(t_bin)

    X_bins = np.vstack(X_bins)
    y_bins = np.array(y_bins)
    t_bins = np.array(t_bins)

    print(f"\nBinning:")
    print(f"  dt ≈ {dt:.3f}s, frames_per_bin={frames_per_bin}, n_bins={n_bins}")
    print(f"  高动机窗口比例: {y_bins.mean():.3f}")
    return X_bins, y_bins, t_bins


def fit_glm2_and_get_motivation_axis(X_bins, y_bins):
    """
    GLM2: 行为(高/低动机) ~ 神经活动
    这里只放神经，不额外加条件变量。
    β_motor 即为动机维度方向。
    """
    print("\n" + "=" * 60)
    print("FITTING GLM2 (MOTIVATION DIMENSION)")
    print("=" * 60)

    clf = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=1000
    )
    clf.fit(X_bins, y_bins)
    acc = clf.score(X_bins, y_bins)
    beta_motor = clf.coef_[0]  # shape: (N_cells,)
    intercept = clf.intercept_[0]

    print(f"✓ GLM2 training accuracy: {acc:.3f}")
    print(f"✓ β_motor shape: {beta_motor.shape}")
    return beta_motor, intercept


# ======================= 排序 & 可视化 =======================
def rank_neurons(beta_motor):
    abs_w = np.abs(beta_motor)
    idx_sorted = np.argsort(abs_w)[::-1]   # |β| 从大到小
    sorted_w = beta_motor[idx_sorted]
    sorted_abs = abs_w[idx_sorted]
    return idx_sorted, sorted_w, sorted_abs


def save_ranking_csv(idx_sorted, sorted_w, out_dir, top_k=80):
    out_path = os.path.join(out_dir, "beta_motor_top_neurons_ranking.csv")
    k = min(top_k, len(idx_sorted))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("rank,neuron_index,beta_motor,abs_beta\n")
        for r in range(k):
            i = int(idx_sorted[r])
            w = float(sorted_w[r])
            f.write(f"{r+1},{i},{w:.8f},{abs(w):.8f}\n")

    print(f"✓ Saved ranking CSV: {out_path}")
    return out_path


def plot_top_weights_bar(idx_sorted, sorted_w, out_dir, top_n=20):
    n = min(top_n, len(idx_sorted))
    sel_idx = idx_sorted[:n]
    sel_w = sorted_w[:n]

    plt.figure(figsize=(12, 4))
    plt.bar(np.arange(n), sel_w)
    plt.xticks(np.arange(n), [str(int(i)) for i in sel_idx])
    plt.xlabel("Neuron index")
    plt.ylabel("β_motor weight")
    plt.title(f"Top {n} neurons by |β_motor| (signed)")
    plt.tight_layout()

    out_path = os.path.join(out_dir, "beta_motor_top_weights_bar.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✓ Saved: {out_path}")
    return out_path


def plot_touch_and_top_traces(
    spikes_epoch, t_epoch, social_flag_epoch,
    idx_sorted, sorted_w, out_dir, num_to_plot=10, reunion_abs=None
):
    num_to_plot = min(num_to_plot, len(idx_sorted))
    plot_idx = idx_sorted[:num_to_plot]

    # 画相对重聚的时间轴会更直观（0=重聚）
    if reunion_abs is not None:
        t_plot = t_epoch - reunion_abs
        x_label = "Time from reunion (s)"
    else:
        t_plot = t_epoch
        x_label = "Time (s)"

    fig, axes = plt.subplots(
        num_to_plot + 1, 1,
        figsize=(14, 2.0 * (num_to_plot + 1)),
        sharex=True
    )

    # Row 0: 社交触摸时间轴
    ax0 = axes[0]
    s = social_flag_epoch.astype(float)
    ax0.plot(t_plot, s, linewidth=1.0)
    ax0.fill_between(t_plot, 0, s, alpha=0.25)
    ax0.set_ylabel("Social")
    ax0.set_yticks([0, 1])
    ax0.set_yticklabels(["no", "yes"])
    ax0.set_title("Social touch timeline + Top β_motor neuron activity")

    # Rows 1..N: 前 num_to_plot 个神经元活动
    for ax, neuron_idx in zip(axes[1:], plot_idx):
        trace = spikes_epoch[:, neuron_idx]
        # 从 sorted_w 里找该 neuron 的 β weight
        w = sorted_w[np.where(idx_sorted == neuron_idx)[0][0]]
        ax.plot(t_plot, trace, linewidth=0.7)
        ax.set_ylabel(f"Cell {int(neuron_idx)}\nβ={w:+.3f}", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)

    axes[-1].set_xlabel(x_label)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "touch_timeline_top10_neurons.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"✓ Saved: {out_path}")
    return out_path


# ======================= 主流程 =======================
def main():
    print("\n" + "=" * 70)
    print("TOUCH-RELATED GLM2 & TOP NEURONS")
    print("=" * 70)

    out_dir = os.path.join(base_out_dir, "touch_glm2")
    os.makedirs(out_dir, exist_ok=True)
    print(f"✓ Output dir: {out_dir}")

    # 1. 载入神经数据 & 行为时间戳
    Y, t, dt = load_neural_data(data_path)
    social_intervals, reunion_rel = load_behavior(beh_path, reunion_abs)

    # 2. 简单 spike 估计
    spikes = calcium_to_spikes(Y)   # (T, N)

    # 3. 构造社交标签 + 抽取重聚后 epoch
    social_flag_full = build_social_flag(t, social_intervals)
    Y_epoch, t_epoch, mask_epoch = extract_reunion_epoch(Y, t, reunion_abs, social_intervals, extra_after=EXTRA_AFTER)
    spikes_epoch = spikes[mask_epoch]
    social_flag_epoch = social_flag_full[mask_epoch]

    # 4. 时间窗 binning + GLM2 -> β_motor
    X_bins, y_bins, t_bins = bin_data(spikes_epoch, t_epoch, social_flag_epoch, bin_size=BIN_SIZE)
    beta_motor, glm_intercept = fit_glm2_and_get_motivation_axis(X_bins, y_bins)

    # 5. 按 |β_motor| 排序神经元
    idx_sorted, sorted_beta, sorted_abs = rank_neurons(beta_motor)

    k_print = min(TOP_K_PRINT, len(idx_sorted))
    print(f"\nTop {k_print} neurons by |β_motor|:")
    for r in range(k_print):
        i = int(idx_sorted[r])
        b = float(sorted_beta[r])
        print(f"#{r+1:2d}  neuron {i:3d}  β_motor = {b:+.6f}")

    # 保存排名 CSV
    save_ranking_csv(idx_sorted, sorted_beta, out_dir, top_k=TOP_K_SAVE)

    # 额外保存一个 npz，方便以后复用
    np.savez(
        os.path.join(out_dir, "touch_glm2_outputs.npz"),
        beta_motor=beta_motor,
        idx_sorted=idx_sorted,
        spikes_epoch=spikes_epoch,
        t_epoch=t_epoch,
        social_flag_epoch=social_flag_epoch,
        dt=dt,
        reunion_abs=reunion_abs
    )
    print(f"✓ Saved: {os.path.join(out_dir, 'touch_glm2_outputs.npz')}")

    # 6. 画图：Top β 条形图 + 触摸时间轴 + 前10个神经元活动
    plot_top_weights_bar(idx_sorted, sorted_beta, out_dir, top_n=20)
    plot_touch_and_top_traces(
        spikes_epoch, t_epoch, social_flag_epoch,
        idx_sorted, sorted_beta, out_dir,
        num_to_plot=NUM_TO_PLOT,
        reunion_abs=reunion_abs
    )

    print("\n✅ Touch GLM2 neuron report complete.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
