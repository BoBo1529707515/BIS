# mouse_rslds_analysis.py
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.stats import entropy

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 核心：使用 SSM 库的 rSLDS
import ssm
from ssm.util import find_permutation

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ======================= 参数设置 =======================
# ----- 你要求的项目位置 -----
base_out_dir = r"F:\工作文件\RA\python\吸引子\figs"
os.makedirs(base_out_dir, exist_ok=True)

data_path = r"F:\工作文件\RA\python\吸引子\Mouse1_for_ssm.npz"
beh_path = r"F:\工作文件\RA\数据集\时间戳\FVB_Panneuronal_Reunion_Isolation_Day3_Mouse#1.xlsx"

# 重聚时间（绝对秒，和前面脚本一致）
reunion_abs = 903.0

# rSLDS 超参数
K_states = 4  # 离散状态数
D_latent = 3  # 连续隐变量维度
downsample = 5  # 下采样
num_iters = 50  # EM 迭代次数


# ======================= 1. 数据加载 =======================
def load_neural_data(npz_path):
    """从 Mouse1_for_ssm.npz 读取神经数据"""
    print("=" * 60)
    print("LOADING NEURAL DATA (.npz)")
    print("=" * 60)

    dat = np.load(npz_path, allow_pickle=True)
    Y = np.asarray(dat["Y"])
    t = np.asarray(dat["t"])
    dt = float(dat["dt"].item())
    T, N = Y.shape

    # Z-score
    Y_z = (Y - Y.mean(axis=0)) / (Y.std(axis=0) + 1e-6)

    print(f"✓ Data shape: T={T} timesteps, N={N} neurons")
    print(f"✓ Sampling dt: {dt:.3f}s  ({1.0 / dt:.2f} Hz)")
    print(f"✓ Total duration: {t[-1]:.1f}s ({t[-1] / 60:.1f} min)")

    return Y_z, t, dt


def load_behavior(beh_path, reunion_abs):
    """加载行为数据并提取社交接触时间段"""
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

    print(f"✓ Found {len(social_intervals)} social interaction bouts")
    print(f"✓ Reunion time (abs): {reunion_abs:.1f}s (rel in Excel: {reunion_rel:.1f}s)")

    return social_intervals, reunion_rel


def extract_reunion_contact_epoch(Y, t, reunion_abs, social_intervals):
    """
    提取"重聚后有接触"的时间段数据

    Parameters:
    -----------
    Y : array (T, N)
        神经数据
    t : array (T,)
        时间戳
    reunion_abs : float
        重聚时间（绝对秒）
    social_intervals : list of (start, end)
        社交接触时间段列表

    Returns:
    --------
    Y_epoch : array (T_epoch, N)
        重聚后有接触时段的神经数据
    t_epoch : array (T_epoch,)
        对应的时间戳
    mask : array (T,) bool
        标记哪些帧属于重聚后接触时段
    """
    print("\n" + "=" * 60)
    print("EXTRACTING REUNION+CONTACT EPOCH")
    print("=" * 60)

    # 1. 找到重聚之后的所有时间点
    is_after_reunion = t >= reunion_abs

    # 2. 在重聚之后，找到有接触的时间段
    is_contact = np.zeros_like(t, dtype=bool)
    for (start, end) in social_intervals:
        # 只要是重聚之后的接触
        if end > reunion_abs:
            effective_start = max(start, reunion_abs)
            in_contact = (t >= effective_start) & (t <= end)
            is_contact |= in_contact

    # 3. 合并条件：重聚后 AND 有接触
    mask = is_after_reunion & is_contact

    if mask.sum() == 0:
        raise ValueError("No data found in 'reunion + contact' epoch!")

    Y_epoch = Y[mask]
    t_epoch = t[mask]

    duration = t_epoch[-1] - t_epoch[0]
    print(f"✓ Reunion+Contact epoch: {len(t_epoch)} frames")
    print(f"✓ Duration: {duration:.1f}s ({duration / 60:.1f} min)")
    print(f"✓ Time range: {t_epoch[0]:.1f}s to {t_epoch[-1]:.1f}s")
    print(f"✓ Percentage of total: {mask.mean() * 100:.1f}%")

    return Y_epoch, t_epoch, mask


def build_behavior_mask(t, reunion_abs, social_intervals):
    """构建行为标签（用于后续分析）"""
    is_social = np.zeros_like(t, dtype=bool)
    for (s, e) in social_intervals:
        is_social |= (t >= s) & (t <= e)

    is_after_reunion = t >= reunion_abs

    return is_social, is_after_reunion


def compare_K_values(Y, t, is_social, is_after_reunion,
                     K_range=[2, 3, 4], out_dir=None):
    """比较不同 K 值的模型性能"""
    print("\n" + "=" * 70)
    print(" " * 20 + "MODEL COMPARISON")
    print("=" * 70)
    print(f"\nTesting K ∈ {K_range}")

    results = {}

    for K in K_range:
        print(f"\n{'─' * 60}")
        print(f"Fitting rSLDS with K={K} states...")

        try:
            rslds, z_map, x_map, lls = fit_rslds(
                Y, downsample=downsample, K=K,
                D=D_latent, num_iters=num_iters
            )

            final_ll = lls[-1]
            T_ds = len(z_map)

            num_dynamics_params = K * (D_latent ** 2 + D_latent)
            num_transition_params = K * (K - 1)
            num_emission_params = K * Y.shape[1] * D_latent
            num_params = num_dynamics_params + num_transition_params + num_emission_params

            aic = -2 * final_ll + 2 * num_params
            bic = -2 * final_ll + np.log(T_ds) * num_params

            state_occupancy = np.bincount(z_map, minlength=K) / len(z_map)
            min_occupancy = state_occupancy.min()

            results[K] = {
                'll': final_ll, 'aic': aic, 'bic': bic,
                'num_params': num_params, 'min_occupancy': min_occupancy,
                'rslds': rslds, 'z_map': z_map, 'x_map': x_map, 'lls': lls
            }

            print(f"✓ LL={final_ll:.1f}, AIC={aic:.1f}, BIC={bic:.1f}")

        except Exception as e:
            print(f"✗ Failed: {e}")
            results[K] = None

    valid_results = {k: v for k, v in results.items() if v is not None}
    if len(valid_results) == 0:
        raise RuntimeError("All models failed!")

    best_K_bic = min(valid_results, key=lambda k: valid_results[k]['bic'])

    print(f"\n✓ Best K by BIC: {best_K_bic}")

    if out_dir:
        plot_model_comparison(valid_results, best_K_bic, out_dir)

    return best_K_bic, valid_results


def plot_model_comparison(results, best_K, out_dir):
    """绘制模型比较曲线"""
    K_list = sorted(results.keys())
    aic_list = [results[K]['aic'] for K in K_list]
    bic_list = [results[K]['bic'] for K in K_list]
    ll_list = [results[K]['ll'] for K in K_list]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(K_list, ll_list, 'o-', linewidth=2.5, markersize=10, color='steelblue')
    axes[0].set_xlabel('K', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Log-Likelihood', fontsize=13)
    axes[0].set_title('Model Fit', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)

    axes[1].plot(K_list, aic_list, 's-', linewidth=2.5, markersize=10, color='coral')
    axes[1].set_xlabel('K', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('AIC', fontsize=13)
    axes[1].set_title('AIC', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)

    axes[2].plot(K_list, bic_list, '^-', linewidth=2.5, markersize=10, color='mediumseagreen')
    axes[2].axvline(best_K, linestyle='--', color='red', linewidth=2)
    axes[2].set_xlabel('K', fontsize=13, fontweight='bold')
    axes[2].set_ylabel('BIC', fontsize=13)
    axes[2].set_title('BIC (Best)', fontsize=14, fontweight='bold')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def fit_rslds(Y, downsample=5, K=4, D=3, num_iters=50):
    """使用 SSM 库拟合 rSLDS"""
    print("\n" + "=" * 60)
    print("FITTING rSLDS")
    print("=" * 60)

    Y_ds = Y[::downsample]
    T, N = Y_ds.shape
    print(f"Input: T={T}, N={N}, K={K}, D={D}")

    rslds = ssm.SLDS(N=N, K=K, D=D, emissions="gaussian",
                     transitions="recurrent", dynamics="gaussian")
    rslds.initialize(Y_ds)

    print(f"Running Laplace-EM ({num_iters} iters)...")
    results = rslds.fit(datas=[Y_ds], method="laplace_em",
                        variational_posterior="structured_meanfield",
                        num_iters=num_iters, initialize=False)

    # 提取后验
    elbos = None
    q = None

    if isinstance(results, (tuple, list)):
        candidates = list(results)
    else:
        candidates = [results]

    for obj in candidates:
        if isinstance(obj, (np.ndarray, list)):
            arr = np.asarray(obj)
            if np.issubdtype(arr.dtype, np.number) and arr.ndim == 1:
                elbos = arr
                continue
        if any(hasattr(obj, attr) for attr in ["mean_continuous_states", "most_likely_states"]):
            q = obj

    if q is None:
        q = results
    if elbos is None and hasattr(q, "elbos"):
        elbos = np.asarray(q.elbos)
    if elbos is None:
        elbos = np.full(num_iters, np.nan)

    print(f"✓ Final ELBO: {elbos[-1]:.2f}")

    # 提取 z 和 x
    def _first_seq(arr):
        if isinstance(arr, (list, tuple)):
            arr = arr[0]
            if isinstance(arr, (list, tuple)):
                return _first_seq(arr)
        arr = np.asarray(arr)
        return arr[0] if arr.ndim == 3 else arr

    def _get_attr(obj, names):
        for name in names:
            if hasattr(obj, name):
                val = getattr(obj, name)
                try:
                    return val() if callable(val) else val
                except:
                    continue
        return None

    z_arr = _get_attr(q, ["most_likely_states", "discrete_states", "z", "mean_discrete_states"])
    if z_arr is None:
        raise RuntimeError("无法获取离散状态")
    z_arr = _first_seq(z_arr)
    z_map = z_arr.argmax(axis=-1) if z_arr.ndim == 2 else z_arr.astype(int)

    x_arr = _get_attr(q, ["mean_continuous_states", "continuous_states", "xs"])
    if x_arr is None:
        raise RuntimeError("无法获取连续状态")
    x_map = _first_seq(x_arr)

    print(f"✓ z_map: {z_map.shape}, x_map: {x_map.shape}")
    return rslds, z_map.astype(int), x_map, list(elbos)


def analyze_rslds_parameters(rslds, K, D):
    """分析模型参数"""
    print("\n" + "=" * 60)
    print("PARAMETERS ANALYSIS")
    print("=" * 60)

    # 转移矩阵
    if callable(rslds.transitions.transition_matrix):
        trans_matrix = rslds.transitions.transition_matrix(np.zeros(D))
    else:
        trans_matrix = rslds.transitions.transition_matrix

    print("\nTransition Matrix:")
    print(trans_matrix)

    return trans_matrix


def compute_state_occupancy(z_map, K, is_social, is_after_reunion):
    """计算状态占用率"""
    print("\n" + "=" * 60)
    print("STATE OCCUPANCY")
    print("=" * 60)

    ds = len(is_social) // len(z_map)
    is_social_ds = is_social[::ds][:len(z_map)]
    is_after_ds = is_after_reunion[::ds][:len(z_map)]

    occupancy = {}
    occupancy['Overall'] = np.bincount(z_map, minlength=K) / len(z_map)

    if is_social_ds.sum() > 0:
        occupancy['Social'] = np.bincount(z_map[is_social_ds], minlength=K) / is_social_ds.sum()

    for cond, occ in occupancy.items():
        print(f"{cond}: {occ}")

    return occupancy


def compute_dwell_times(z_map, K, dt_fit):
    """计算状态停留时间"""
    print("\n" + "=" * 60)
    print("DWELL TIMES")
    print("=" * 60)

    dwell_times = {k: [] for k in range(K)}
    current = z_map[0]
    duration = 1

    for i in range(1, len(z_map)):
        if z_map[i] == current:
            duration += 1
        else:
            dwell_times[current].append(duration * dt_fit)
            current = z_map[i]
            duration = 1
    dwell_times[current].append(duration * dt_fit)

    for k in range(K):
        if len(dwell_times[k]) > 0:
            print(f"State {k + 1}: mean={np.mean(dwell_times[k]):.2f}s")

    return dwell_times


def analyze_latent_trajectories(x_map, z_map, K):
    """分析隐空间轨迹"""
    print("\n" + "=" * 60)
    print("LATENT TRAJECTORIES")
    print("=" * 60)

    metrics = {}
    for k in range(K):
        x_k = x_map[z_map == k]
        if len(x_k) > 1:
            speed = np.linalg.norm(np.diff(x_k, axis=0), axis=1).mean()
            metrics[k] = {'n': len(x_k), 'speed': speed}
            print(f"State {k + 1}: n={len(x_k)}, speed={speed:.3f}")

    return metrics


# 可视化函数（简化版）
def plot_learning_curve(lls, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(lls, 'o-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Log-Likelihood')
    ax.set_title('Learning Curve')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'learning_curve.png'), dpi=300)
    plt.close()


def plot_state_timeline(z_map, t_ds, reunion_abs, social_intervals, K, out_dir):
    fig, ax = plt.subplots(figsize=(14, 4))
    t_rel = t_ds - reunion_abs
    colors = plt.cm.Set2(np.linspace(0, 1, K))

    for k in range(K):
        mask = z_map == k
        ax.scatter(t_rel[mask], np.full(mask.sum(), k + 1), c=[colors[k]], s=2, alpha=0.6)

    ax.axvline(0, linestyle='--', color='red', linewidth=2, label='Reunion')
    ax.set_xlabel('Time from Reunion (s)')
    ax.set_ylabel('State')
    ax.set_title('State Timeline')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'state_timeline.png'), dpi=300)
    plt.close()


def plot_latent_3d(x_map, z_map, K, out_dir):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.Set2(np.linspace(0, 1, K))

    for k in range(K):
        mask = z_map == k
        ax.scatter(x_map[mask, 0], x_map[mask, 1], x_map[mask, 2],
                   c=[colors[k]], s=5, alpha=0.5, label=f'S{k + 1}')

    ax.set_xlabel('Latent 1')
    ax.set_ylabel('Latent 2')
    ax.set_zlabel('Latent 3')
    ax.set_title('3D Latent Space')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'latent_3d.png'), dpi=300)
    plt.close()


# 主流程
def main():
    print("\n" + "=" * 70)
    print("rSLDS ANALYSIS: REUNION+CONTACT EPOCH ONLY")
    print("=" * 70)

    # 1. 加载数据
    Y, t, dt = load_neural_data(data_path)
    social_intervals, _ = load_behavior(beh_path, reunion_abs)

    # 2. 提取重聚后接触时段
    Y_epoch, t_epoch, mask_epoch = extract_reunion_contact_epoch(
        Y, t, reunion_abs, social_intervals
    )

    # 3. 创建输出目录
    epoch_out_dir = os.path.join(base_out_dir, "rslds_reunion_contact")
    os.makedirs(epoch_out_dir, exist_ok=True)
    print(f"\n✓ Output: {epoch_out_dir}")

    # 保存mask
    np.save(os.path.join(epoch_out_dir, "mask.npy"), mask_epoch)
    np.save(os.path.join(epoch_out_dir, "t_epoch.npy"), t_epoch)

    # 构建行为标签
    is_social = np.ones_like(t_epoch, dtype=bool)
    is_after = np.ones_like(t_epoch, dtype=bool)

    # 4. 模型比较（可选）
    ENABLE_COMPARISON = True

    if ENABLE_COMPARISON:
        best_K, comparison = compare_K_values(
            Y_epoch, t_epoch, is_social, is_after,
            K_range=[2, 3, 4, 5, 6], out_dir=epoch_out_dir
        )
        K_final = best_K
        rslds = comparison[K_final]['rslds']
        z_map = comparison[K_final]['z_map']
        x_map = comparison[K_final]['x_map']
        lls = comparison[K_final]['lls']
    else:
        K_final = K_states
        rslds, z_map, x_map, lls = fit_rslds(Y_epoch, K=K_final, D=D_latent)

    # 5. 时间对齐
    t_ds = t_epoch[::downsample][:len(z_map)]
    dt_fit = dt * downsample

    # 6. 分析
    trans_matrix = analyze_rslds_parameters(rslds, K_final, D_latent)
    occupancy = compute_state_occupancy(z_map, K_final, is_social, is_after)
    dwell_times = compute_dwell_times(z_map, K_final, dt_fit)
    metrics = analyze_latent_trajectories(x_map, z_map, K_final)

    # 7. 可视化
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    plot_learning_curve(lls, epoch_out_dir)
    plot_state_timeline(z_map, t_ds, reunion_abs, social_intervals, K_final, epoch_out_dir)
    plot_latent_3d(x_map, z_map, K_final, epoch_out_dir)

    # 8. 保存结果
    np.savez(
        os.path.join(epoch_out_dir, 'results.npz'),
        z_map=z_map, x_map=x_map, t_ds=t_ds,
        trans_matrix=trans_matrix,
        K_final=K_final
    )

    print(f"\n✓ All saved to: {epoch_out_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
