# -*- coding: utf-8 -*-
"""
社交需求神经元解码分析 (History-Dependent Decoding / Satiety Accumulation)

功能：
1) 加载神经数据与行为时间戳。
2) 计算“累积社交接触时长”作为目标变量 Y。
3) 仅提取“社交接触 (Social Touch)”期间的数据 X。
4) 使用 LASSO 回归训练模型：Y ~ X * w。
5) 筛选出权重为负（Negative Weights）的神经元 -> "Social Need Neurons"。
6) 绘制模型预测性能及核心神经元的活动轨迹。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ======================= 参数设置 =======================
# 请根据实际情况修改路径
base_out_dir = r"F:\工作文件\RA\python\吸引子\figs\Lasso_Decoding"
data_path = r"F:\工作文件\RA\python\吸引子\Mouse1_for_ssm.npz"
beh_path = r"F:\工作文件\RA\数据集\时间戳\FVB_Panneuronal_Reunion_Isolation_Day3_Mouse#1.xlsx"

reunion_abs = 903.0  # 重聚开始的绝对时间

# 分析参数
MIN_SOCIAL_DURATION = 1.0  # 忽略短于1秒的接触片段
TRAIN_TEST_SPLIT_RATIO = 0.2
LASSO_CV_FOLDS = 5
TOP_N_NEURONS = 9  # 绘图时展示前N个核心神经元

# 绘图设置
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
os.makedirs(base_out_dir, exist_ok=True)


# ======================= 1. 数据加载模块 (复用原有逻辑) =======================

def load_neural_data(npz_path):
    print("=" * 60)
    print("LOADING NEURAL DATA")
    dat = np.load(npz_path, allow_pickle=True)
    Y = np.asarray(dat["Y"])  # (T, N)
    t = np.asarray(dat["t"])  # (T,)
    dt = float(dat["dt"].item())

    # 简单的 z-score，不进行过度平滑，保留原始动态
    Y_z = zscore(Y, axis=0)
    print(f"✓ Data loaded: T={len(t)}, N={Y.shape[1]}, dt={dt:.3f}s")
    return Y_z, t, dt


def load_behavior(beh_path, reunion_abs):
    print("\nLOADING BEHAVIOR DATA")
    beh_df = pd.read_excel(beh_path, header=None).dropna(how="all")
    beh_df[0] = pd.to_numeric(beh_df[0], errors="coerce")
    beh_df[1] = beh_df[1].astype(str).str.strip().str.lower()

    reunion_rows = beh_df[beh_df[1].str.contains("重聚期开始", na=False)]
    reunion_rel = float(reunion_rows.iloc[0, 0]) if len(reunion_rows) > 0 else 0.0

    starts = beh_df[beh_df[1].str.contains("社交开始", na=False)][0].values
    ends = beh_df[beh_df[1].str.contains("社交结束", na=False)][0].values

    social_intervals = []
    for s, e in zip(starts, ends):
        start_abs = reunion_abs + (s - reunion_rel)
        end_abs = reunion_abs + (e - reunion_rel)
        if end_abs - start_abs >= MIN_SOCIAL_DURATION:
            social_intervals.append((start_abs, end_abs))

    print(f"✓ Found {len(social_intervals)} valid social bouts (> {MIN_SOCIAL_DURATION}s)")
    return social_intervals


# ======================= 2. 特征工程：构建累积接触时间 =======================

def build_cumulative_social_time(t, social_intervals):
    """
    计算每个时间点的“累积社交接触时长”。
    规则：
    - 在非社交期间：值保持不变（记忆保持）或为0（如果只关心社交时刻）。
    - 在社交期间：值随时间线性增加。
    """
    cumulative_time = np.zeros_like(t)
    is_social = np.zeros_like(t, dtype=bool)

    current_cum = 0.0
    dt = t[1] - t[0]

    # 简单的遍历标记
    # 为了效率，先生成mask
    for (start, end) in social_intervals:
        mask = (t >= start) & (t <= end)
        is_social[mask] = True

    # 累积计算
    # 注意：这里我们计算的是"直到当前时刻的总社交时间"
    # 如果需要在非社交期间保持数值，以便观察"需求是否反弹"，这里可以不做清零。
    # 但LASSO训练只用 is_social=True 的帧。

    temp_cum = np.cumsum(is_social.astype(float) * dt)
    cumulative_time = temp_cum

    return cumulative_time, is_social


# ======================= 3. 解码分析核心逻辑 =======================

def run_lasso_decoding(X, y, feature_names=None):
    """
    运行 LASSO 回归，预测累积时间。
    X: 神经活动 (Samples, Neurons)
    y: 累积时间 (Samples,)
    """
    print("\n" + "=" * 60)
    print("RUNNING LASSO DECODING (History-Dependent)")
    print("=" * 60)

    # 划分训练/测试集 (不打乱顺序，以测试模型的泛化能力 - 例如预测后段的时间)
    # 但考虑到社交Bout可能是断续的，为了稳健性，这里使用随机Shuffle split
    # 也可以选择按Bout划分 (更严格)，此处先用标准随机划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TRAIN_TEST_SPLIT_RATIO, shuffle=True, random_state=42
    )

    print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # 定义 LASSO 模型 (自动交叉验证选择 alpha)
    # n_jobs=-1 使用所有CPU核心
    model = LassoCV(cv=LASSO_CV_FOLDS, random_state=42, n_jobs=-1, max_iter=2000)

    print("Fitting model...")
    model.fit(X_train, y_train)

    # 预测与评估
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    print(f"✓ Model Performance:")
    print(f"  Train R^2: {r2_train:.3f}")
    print(f"  Test  R^2: {r2_test:.3f}")
    print(f"  Best Alpha: {model.alpha_:.6f}")

    # 提取权重
    weights = model.coef_
    intercept = model.intercept_

    return {
        "model": model,
        "weights": weights,
        "intercept": intercept,
        "r2_test": r2_test,
        "y_test": y_test,
        "y_pred_test": y_pred_test
    }


# ======================= 4. 绘图与可视化 =======================

def plot_model_performance(results, out_dir):
    """绘制实际值 vs 预测值"""
    plt.figure(figsize=(6, 6))
    plt.scatter(results["y_test"], results["y_pred_test"], alpha=0.3, s=10, color='purple')

    # 画对角线
    min_val = min(results["y_test"].min(), results["y_pred_test"].min())
    max_val = max(results["y_test"].max(), results["y_pred_test"].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)

    plt.title(f"Decoding Cumulative Social Time\nR² = {results['r2_test']:.3f}")
    plt.xlabel("Actual Cumulative Time (s)")
    plt.ylabel("Predicted Cumulative Time (s)")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "1_prediction_performance.png"), dpi=300)
    plt.close()


def plot_top_neurons(X_full, t_full, is_social_mask, weights, social_intervals, out_dir):
    """
    绘制权重最大的前N个负权重神经元（Need Neurons）。
    展示它们在整个 Reuion 过程中的活动，并高亮社交时刻。
    """
    # 筛选负权重（表征随时间下降/被满足）
    # 权重越负，说明累积时间越长，活动越低 -> 符合 Social Drive 定义
    neg_indices = np.where(weights < -1e-4)[0]

    if len(neg_indices) == 0:
        print("⚠️ No negative weight neurons found!")
        return

    # 按权重绝对值排序 (越负越重要)
    sorted_indices = neg_indices[np.argsort(weights[neg_indices])][:TOP_N_NEURONS]

    print(f"\nTop {len(sorted_indices)} 'Social Need' neurons (Negative Weights):")
    print(sorted_indices)

    # 绘图
    n_cols = 3
    n_rows = (len(sorted_indices) + n_cols - 1) // n_cols

    plt.figure(figsize=(15, 3 * n_rows))

    for i, neuron_idx in enumerate(sorted_indices):
        ax = plt.subplot(n_rows, n_cols, i + 1)

        # 提取该神经元活动
        trace = X_full[:, neuron_idx]

        # 绘制整条曲线
        # 将 t 转换为相对重聚开始的时间
        t_rel = t_full - reunion_abs

        # 绘制背景灰色曲线 (全部)
        ax.plot(t_rel, trace, color='lightgray', alpha=0.5, linewidth=0.5)

        # 高亮社交片段 (Social Bouts)
        # 为了不连线，我们将非社交时刻设为nan
        social_trace = np.where(is_social_mask, trace, np.nan)
        ax.plot(t_rel, social_trace, color='#d62728', linewidth=1.2, label='Social Contact')

        # 标注重聚开始
        ax.axvline(0, color='k', linestyle='--', alpha=0.5)

        # 标注权重
        w_val = weights[neuron_idx]
        ax.set_title(f"Neuron #{neuron_idx}\nWeight: {w_val:.4f}", fontsize=10)

        if i == 0:
            ax.legend(loc='upper right', fontsize=8)

    plt.suptitle("Top 'Social Need' Neurons (Activity decreases as social time accumulates)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "2_top_negative_neurons_traces.png"), dpi=300)
    plt.close()


def plot_population_need_state(X_full, t_full, weights, reunion_abs, out_dir):
    """
    利用模型权重，将整个群体的活动投影到 'Satiety Axis'。
    Projection = X @ weights
    如果权重多为负，这个投影值应该随社交进行而下降。
    """
    # 投影
    projection = X_full @ weights

    # 平滑一下以便看清趋势 (比如 2秒窗口)
    # window_size = int(2.0 / (t_full[1] - t_full[0]))
    # projection_smooth = np.convolve(projection, np.ones(window_size)/window_size, mode='same')

    t_rel = t_full - reunion_abs

    plt.figure(figsize=(12, 5))
    plt.plot(t_rel, projection, color='#1f77b4', alpha=0.8, linewidth=1)

    plt.axvline(0, color='k', linestyle='--', label='Reunion Start')
    plt.ylabel("Predicted Accumulation (Model Output)")
    plt.xlabel("Time relative to Reunion (s)")
    plt.title("Population 'Social Satiety' State Trajectory\n(Projection of all neurons onto decoded axis)")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "3_population_satiety_trajectory.png"), dpi=300)
    plt.close()


# ======================= 主程序 =======================

def main():
    # 1. 加载数据
    Y_neural, t, dt = load_neural_data(data_path)
    social_intervals = load_behavior(beh_path, reunion_abs)

    # 2. 构建目标变量 (累积接触时间)
    cumulative_time, is_social_mask = build_cumulative_social_time(t, social_intervals)

    # 3. 数据筛选
    # 我们只用 "Social Contact" 期间的数据来训练
    # 假设：同样的接触动作，在实验初期和末期激发的神经活动不同，这种差异就是"需求"
    X_train_set = Y_neural[is_social_mask]
    y_train_set = cumulative_time[is_social_mask]

    if len(y_train_set) < 50:
        print("❌ Error: Not enough social data points for training!")
        return

    # 4. 运行解码
    results = run_lasso_decoding(X_train_set, y_train_set)

    # 如果模型效果太差，可能说明没有这种神经元，或者线性假设不成立
    if results['r2_test'] < 0.1:
        print("⚠️ Warning: Model predictive power is low. Results may be noisy.")

    # 5. 可视化
    print("\nGenerating Plots...")

    # A. 预测性能
    plot_model_performance(results, base_out_dir)

    # B. 核心神经元轨迹 (Top Negative Weights)
    plot_top_neurons(Y_neural, t, is_social_mask, results['weights'], social_intervals, base_out_dir)

    # C. 群体状态轨迹
    plot_population_need_state(Y_neural, t, results['weights'], reunion_abs, base_out_dir)

    # 6. 保存结果
    np.savez(
        os.path.join(base_out_dir, "lasso_results.npz"),
        weights=results['weights'],
        intercept=results['intercept'],
        r2=results['r2_test'],
        top_neurons=np.where(results['weights'] < -1e-4)[0]
    )

    print("\n✅ Analysis Complete. Results saved to:", base_out_dir)


if __name__ == "__main__":
    main()
