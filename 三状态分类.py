# -*- coding: utf-8 -*-
"""
三状态 GLM 分析：
  state 0: 重聚前 200s（只有一只鼠）
  state 1: 重聚后，无社交触摸（对方在，但没触摸）
  state 2: 社交触摸区间

流程：
  1) 读取 Mouse1_for_ssm.npz 和 行为 Excel（格式同 mouse_switching_ramping_analysis.py）
  2) 定义长 epoch: [reunion_abs-200s, 最后一次社交结束 + buffer]
  3) 将每一帧标为三种状态之一
  4) 1s bin，多分类 logistic (softmax)，特征 = 神经元平均活动
  5) 输出三类的 Top 贡献神经元列表，并保存 CSV
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore

# ----------------- Matplotlib 中文支持（可选） -----------------
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ======================= 路径 & 参数 =======================
base_out_dir = r"F:\工作文件\RA\python\吸引子\figs"
os.makedirs(base_out_dir, exist_ok=True)

data_path = r"F:\工作文件\RA\python\吸引子\Mouse1_for_ssm.npz"
beh_path = r"F:\工作文件\RA\数据集\时间戳\FVB_Panneuronal_Reunion_Isolation_Day3_Mouse#1.xlsx"

# 重聚在绝对时间轴上的时间（秒）
reunion_abs = 903.0

# 向前扩的时间（秒）：重聚前 PRE_BEFORE 秒，没有另一只鼠
PRE_BEFORE = 200.0

# 重聚期尾部 buffer（秒）
EXTRA_AFTER = 60.0

# GLM 的 bin 宽（秒）
BIN_SIZE = 1.0

TOP_K_PER_STATE = 20   # 每个状态打印前多少个关键神经元
TOP_K_CSV = 80         # CSV 里保存前多少个（按每个状态的 |beta| 排序）

# ===========================================================


# ======================= 数据加载 & 预处理 =======================
def load_neural_data(npz_path):
    dat = np.load(npz_path, allow_pickle=True)
    Y = np.asarray(dat["Y"])     # (T, N)
    t = np.asarray(dat["t"])     # (T,)
    dt = float(dat["dt"].item())

    # z-score + clip + 一点平滑
    Y_z = (Y - Y.mean(axis=0)) / (Y.std(axis=0) + 1e-6)
    Y_z = np.clip(Y_z, -3, 3)
    Y_z = gaussian_filter1d(Y_z, sigma=1, axis=0)

    print(f"✓ Loaded neural: T={len(t)}, N={Y.shape[1]}, dt≈{dt:.3f}s")
    return Y_z, t, dt


def load_behavior_like_hmm_script(beh_path, reunion_abs):
    """
    完全照你 HMM 脚本的方式读 Excel：
      第0列：时间（相对）
      第1列：文本，含“重聚期开始”、“社交开始”、“社交结束”
    """
    print("\n" + "=" * 60)
    print("LOADING BEHAVIOR DATA (three-state)")
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


def calcium_to_spikes(Y):
    """
    简单处理：zscore 后当作 spike-like 活动（与之前脚本一致）
    """
    print("\n" + "=" * 60)
    print("SKIP OASIS: using z-scored Y as 'spikes'")
    print("=" * 60)
    spikes = zscore(Y, axis=0)
    return spikes


# ======================= 三状态标签构造 =======================
def build_three_state_labels(t, social_intervals, reunion_abs):
    """
    对整条时间轴上每一帧打 label：
      0 = 重聚前（t < reunion_abs）
      1 = 重聚后未触摸（t >= reunion_abs 且不在任何 social interval）
      2 = 社交触摸期（t 在social interval里）
    """
    labels = np.zeros_like(t, dtype=int)  # 默认 0

    # 先标记“重聚后的时间”为 1
    labels[t >= reunion_abs] = 1

    # 再把社交区间覆盖成 2
    for (start, end) in social_intervals:
        mask = (t >= start) & (t <= end)
        labels[mask] = 2

    return labels


def extract_extended_epoch(Y, t, labels, social_intervals,
                           reunion_abs, pre_before=200.0, extra_after=60.0):
    """
    epoch 范围：
      [max(t_min, reunion_abs - pre_before), min(t_max, last_social_end + extra_after)]
    """
    print("\n" + "=" * 60)
    print("EXTRACTING EXTENDED EPOCH (pre + reunion)")
    print("=" * 60)

    t_min = float(t.min())
    t_max = float(t.max())

    epoch_start = max(t_min, reunion_abs - pre_before)
    if len(social_intervals) > 0:
        last_end = max(e for (_, e) in social_intervals)
        epoch_end = min(t_max, last_end + extra_after)
    else:
        epoch_end = min(t_max, reunion_abs + extra_after)

    mask = (t >= epoch_start) & (t <= epoch_end)

    Y_epoch = Y[mask]
    t_epoch = t[mask]
    labels_epoch = labels[mask]

    duration = t_epoch[-1] - t_epoch[0]
    print(f"✓ Epoch: [{epoch_start:.1f}, {epoch_end:.1f}] s  "
          f"(duration≈{duration:.1f}s, frames={len(t_epoch)})")

    return Y_epoch, t_epoch, labels_epoch, epoch_start, epoch_end


# ======================= binning & 多分类 GLM =======================
def bin_multiclass(spikes_epoch, t_epoch, labels_epoch, bin_size=1.0):
    """
    1s bin：
      - 特征 = 每个 neuron 在 bin 内的平均活动
      - 标签 = bin 内三种状态的多数票（0/1/2）
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
        window_labels = labels_epoch[start:end]

        if len(window_spikes) == 0:
            continue

        X_bin = window_spikes.mean(axis=0)  # (N,)
        # 多数票
        counts = np.bincount(window_labels, minlength=3)
        y_bin = int(np.argmax(counts))
        t_bin = t_epoch[start:end].mean()

        X_bins.append(X_bin)
        y_bins.append(y_bin)
        t_bins.append(t_bin)

    X_bins = np.vstack(X_bins)
    y_bins = np.array(y_bins)
    t_bins = np.array(t_bins)

    print("\nBinning for 3-state GLM:")
    print(f"  dt≈{dt:.3f}s, frames_per_bin={frames_per_bin}, n_bins={n_bins}")
    for s in [0, 1, 2]:
        frac = (y_bins == s).mean()
        print(f"  state {s}: fraction={frac:.3f}")

    return X_bins, y_bins, t_bins


def fit_three_state_glm(X_bins, y_bins):
    """
    多分类 logistic（softmax）：
      y ∈ {0,1,2}
      coef_.shape = (3, N_neurons)
    """
    print("\n" + "=" * 60)
    print("FITTING 3-STATE GLM (multinomial)")
    print("=" * 60)

    clf = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        multi_class="multinomial",
        max_iter=5000
    )
    clf.fit(X_bins, y_bins)

    y_hat = clf.predict(X_bins)
    acc = accuracy_score(y_bins, y_hat)
    cm = confusion_matrix(y_bins, y_hat, labels=[0, 1, 2])

    print(f"✓ Training accuracy (3-state): {acc:.3f}")
    print("✓ Confusion matrix (rows=true, cols=pred, order=[0,1,2]):")
    print(cm)

    betas = clf.coef_   # (3, N)
    intercepts = clf.intercept_  # (3,)

    print(f"✓ betas shape: {betas.shape} (n_states x n_neurons)")
    return betas, intercepts, clf


# ======================= 排序 & 输出 =======================
def print_top_neurons_per_state(betas, top_k=20):
    """
    对每个状态 k，按 |β_k| 排序，打印前 top_k 个神经元
    """
    n_states, n_neurons = betas.shape
    for k in range(n_states):
        w = betas[k]
        abs_w = np.abs(w)
        idx_sorted = np.argsort(abs_w)[::-1]

        print(f"\n=== State {k} top {top_k} neurons (by |β|) ===")
        print("  (state解释: 0=重聚前alone, 1=重聚后非社交, 2=社交触摸)")
        for rank in range(min(top_k, n_neurons)):
            i = int(idx_sorted[rank])
            wi = float(w[i])
            print(f"#{rank+1:2d}  neuron {i:3d}  β_{k} = {wi:+.6f}")


def save_betas_csv(betas, out_dir, top_k_each=80):
    """
    保存一个表：每个 neuron 对各状态的β值；以及对每个状态的Top列表
    """
    n_states, n_neurons = betas.shape
    neuron_idx = np.arange(n_neurons)
    cols = ["neuron_index"] + [f"beta_state{k}" for k in range(n_states)]
    data = np.zeros((n_neurons, len(cols)))
    data[:, 0] = neuron_idx
    for k in range(n_states):
        data[:, k+1] = betas[k]

    df = pd.DataFrame(data, columns=cols)
    csv_all = os.path.join(out_dir, "three_state_glm_betas_all_neurons.csv")
    df.to_csv(csv_all, index=False)
    print(f"✓ Saved all-neuron betas CSV: {csv_all}")

    # 每个状态单独的 Top 列表
    for k in range(n_states):
        w = betas[k]
        abs_w = np.abs(w)
        idx_sorted = np.argsort(abs_w)[::-1]
        k_use = min(top_k_each, n_neurons)

        rows = []
        for rank in range(k_use):
            i = int(idx_sorted[rank])
            wi = float(w[i])
            rows.append([rank+1, i, wi, abs(wi)])

        df_k = pd.DataFrame(
            rows,
            columns=["rank", "neuron_index", f"beta_state{k}", "abs_beta"]
        )
        csv_k = os.path.join(out_dir, f"three_state_top_neurons_state{k}.csv")
        df_k.to_csv(csv_k, index=False)
        print(f"✓ Saved top list for state {k}: {csv_k}")


# ======================= 主流程 =======================
def main():
    print("\n" + "=" * 70)
    print("3-STATE GLM: pre-alone vs reunion-non-social vs social-touch")
    print("=" * 70)

    out_dir = os.path.join(base_out_dir, "three_state_glm")
    os.makedirs(out_dir, exist_ok=True)
    print(f"✓ Output dir: {out_dir}")

    # 1) 神经数据
    Y, t, dt = load_neural_data(data_path)
    spikes = calcium_to_spikes(Y)

    # 2) 行为：social bouts
    social_intervals, reunion_rel = load_behavior_like_hmm_script(beh_path, reunion_abs)

    # 3) 整条 trace 上逐帧三状态标签
    labels_full = build_three_state_labels(t, social_intervals, reunion_abs)

    # 4) 扩展 epoch：[reunion_abs - PRE_BEFORE, last_social_end + EXTRA_AFTER]
    spikes_epoch, t_epoch, labels_epoch, epoch_start, epoch_end = extract_extended_epoch(
        spikes, t, labels_full, social_intervals,
        reunion_abs, pre_before=PRE_BEFORE, extra_after=EXTRA_AFTER
    )

    # 5) 1s bin，多分类 GLM
    X_bins, y_bins, t_bins = bin_multiclass(
        spikes_epoch, t_epoch, labels_epoch, bin_size=BIN_SIZE
    )
    betas, intercepts, clf = fit_three_state_glm(X_bins, y_bins)

    # 6) 输出每个状态的关键神经元
    print_top_neurons_per_state(betas, top_k=TOP_K_PER_STATE)

    # 7) 保存 CSV
    save_betas_csv(betas, out_dir, top_k_each=TOP_K_CSV)

    # 8) 顺便保存一下结果包
    np.savez(
        os.path.join(out_dir, "three_state_glm_outputs.npz"),
        betas=betas,
        intercepts=intercepts,
        spikes_epoch=spikes_epoch,
        t_epoch=t_epoch,
        labels_epoch=labels_epoch,
        t_bins=t_bins,
        y_bins=y_bins,
        dt=dt,
        reunion_abs=reunion_abs,
        epoch_start=epoch_start,
        epoch_end=epoch_end
    )
    print(f"✓ Saved NPZ: {os.path.join(out_dir, 'three_state_glm_outputs.npz')}")

    print("\n✅ 3-state GLM analysis complete.")
    print("  state 0: 重聚前 200s（完全alone）")
    print("  state 1: 重聚后，但当前无社交触摸")
    print("  state 2: 社交触摸期\n")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
