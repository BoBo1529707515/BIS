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
out_dir = r"F:\工作文件\RA\python\吸引子\figs"
os.makedirs(out_dir, exist_ok=True)

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
    """
    从 Mouse1_for_ssm.npz 读取神经数据。
    要求 npz 里面有键：Y, t, dt
    Y: (T, N) 神经元活动（已预处理）
    t: (T,) 时间戳（秒）
    dt: float，采样间隔（秒）
    """
    print("=" * 60)
    print("LOADING NEURAL DATA (.npz)")
    print("=" * 60)

    dat = np.load(npz_path, allow_pickle=True)

    Y = np.asarray(dat["Y"])  # (T, N)
    t = np.asarray(dat["t"])  # (T,)
    dt = float(dat["dt"].item())  # 标量

    T, N = Y.shape

    # Z-score（保证和原脚本一致）
    Y_z = (Y - Y.mean(axis=0)) / (Y.std(axis=0) + 1e-6)

    print(f"✓ Data shape: T={T} timesteps, N={N} neurons")
    print(f"✓ Sampling dt: {dt:.3f}s  ({1.0 / dt:.2f} Hz)")
    print(f"✓ Total duration: {t[-1]:.1f}s ({t[-1] / 60:.1f} min)")

    return Y_z, t, dt


def load_behavior(beh_path, reunion_abs):
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


def build_behavior_mask(t, reunion_abs, social_intervals):
    is_social = np.zeros_like(t, dtype=bool)
    for (s, e) in social_intervals:
        is_social |= (t >= s) & (t <= e)

    is_after_reunion = t >= reunion_abs

    print(f"✓ Social time: {is_social.sum() * (t[1] - t[0]):.1f}s "
          f"({is_social.mean() * 100:.1f}%)")
    print(f"✓ After reunion: {is_after_reunion.sum() * (t[1] - t[0]):.1f}s")

    return is_social, is_after_reunion


def compare_K_values(Y, t, is_social, is_after_reunion,
                     K_range=[2, 3, 4, 5, 6], out_dir=None):
    """
    比较不同 K 值的模型性能，使用 AIC/BIC 准则

    Parameters:
    -----------
    Y : array (T, N)
        神经数据
    t : array (T,)
        时间戳
    is_social, is_after_reunion : array (T,)
        行为标签
    K_range : list
        要测试的状态数范围
    out_dir : str
        保存路径

    Returns:
    --------
    best_K : int
        BIC 最优的状态数
    results : dict
        每个 K 的详细结果
    """
    print("\n" + "=" * 70)
    print(" " * 20 + "MODEL COMPARISON")
    print("=" * 70)
    print(f"\nTesting K ∈ {K_range}")
    print(f"Criteria: AIC (Akaike) and BIC (Bayesian)\n")

    results = {}

    for K in K_range:
        print(f"\n{'─' * 60}")
        print(f"Fitting rSLDS with K={K} states...")
        print(f"{'─' * 60}")

        try:
            # 拟合模型（使用和主流程相同的参数）
            rslds, z_map, x_map, lls = fit_rslds(
                Y,
                downsample=downsample,
                K=K,
                D=D_latent,
                num_iters=num_iters
            )

            # 获取最终 log-likelihood
            final_ll = lls[-1]
            T_ds = len(z_map)  # 下采样后的时间点数

            # 估计参数数量
            # 1. 动力学参数: K * (D^2 + D) [A_k 矩阵 + b_k 向量]
            # 2. 转移矩阵: K * (K-1) [每行有 K-1 个自由参数]
            # 3. 发射参数: K * N * D [简化估计]
            num_dynamics_params = K * (D_latent ** 2 + D_latent)
            num_transition_params = K * (K - 1)
            num_emission_params = K * Y.shape[1] * D_latent
            num_params = num_dynamics_params + num_transition_params + num_emission_params

            # 计算信息准则
            aic = -2 * final_ll + 2 * num_params
            bic = -2 * final_ll + np.log(T_ds) * num_params

            # 计算状态占比（检查是否有"浪费"的状态）
            state_occupancy = np.bincount(z_map, minlength=K) / len(z_map)
            min_occupancy = state_occupancy.min()

            results[K] = {
                'll': final_ll,
                'aic': aic,
                'bic': bic,
                'num_params': num_params,
                'min_occupancy': min_occupancy,
                'rslds': rslds,
                'z_map': z_map,
                'x_map': x_map,
                'lls': lls
            }

            print(f"✓ LL = {final_ll:.1f}")
            print(f"✓ AIC = {aic:.1f}")
            print(f"✓ BIC = {bic:.1f}")
            print(f"✓ Params = {num_params}")
            print(f"✓ Min state occupancy = {min_occupancy:.3f}")

        except Exception as e:
            print(f"✗ Failed: {e}")
            results[K] = None

    # 过滤失败的模型
    valid_results = {k: v for k, v in results.items() if v is not None}

    if len(valid_results) == 0:
        raise RuntimeError("All models failed to fit!")

    # 选择最优 K
    best_K_aic = min(valid_results, key=lambda k: valid_results[k]['aic'])
    best_K_bic = min(valid_results, key=lambda k: valid_results[k]['bic'])

    print("\n" + "=" * 70)
    print("MODEL COMPARISON RESULTS")
    print("=" * 70)
    print(f"\n{'K':<5} {'LL':<12} {'AIC':<12} {'BIC':<12} {'Params':<10} {'MinOcc':<10}")
    print("-" * 70)

    for K in sorted(valid_results.keys()):
        res = valid_results[K]
        marker = ""
        if K == best_K_aic:
            marker += " (AIC*)"
        if K == best_K_bic:
            marker += " (BIC*)"

        print(f"{K:<5} {res['ll']:<12.1f} {res['aic']:<12.1f} "
              f"{res['bic']:<12.1f} {res['num_params']:<10} "
              f"{res['min_occupancy']:<10.3f} {marker}")

    print(f"\n✓ Best K by AIC: {best_K_aic}")
    print(f"✓ Best K by BIC: {best_K_bic}")

    # 可视化
    if out_dir is not None:
        plot_model_comparison(valid_results, best_K_bic, out_dir)

    return best_K_bic, valid_results


def plot_model_comparison(results, best_K, out_dir):
    """绘制模型比较曲线"""
    print("\n✓ Generating model comparison plot...")

    K_list = sorted(results.keys())
    aic_list = [results[K]['aic'] for K in K_list]
    bic_list = [results[K]['bic'] for K in K_list]
    ll_list = [results[K]['ll'] for K in K_list]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Log-Likelihood
    ax = axes[0]
    ax.plot(K_list, ll_list, 'o-', linewidth=2.5, markersize=10,
            color='steelblue', label='Log-Likelihood')
    ax.axvline(best_K, linestyle='--', color='red', linewidth=2,
               alpha=0.7, label=f'Best K={best_K}')
    ax.set_xlabel('Number of States (K)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Log-Likelihood', fontsize=13, fontweight='bold')
    ax.set_title('Model Fit Quality', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xticks(K_list)

    # 2. AIC
    ax = axes[1]
    ax.plot(K_list, aic_list, 's-', linewidth=2.5, markersize=10,
            color='coral', label='AIC')
    best_K_aic = min(results, key=lambda k: results[k]['aic'])
    ax.axvline(best_K_aic, linestyle='--', color='red', linewidth=2,
               alpha=0.7, label=f'Best K={best_K_aic}')
    ax.set_xlabel('Number of States (K)', fontsize=13, fontweight='bold')
    ax.set_ylabel('AIC (lower is better)', fontsize=13, fontweight='bold')
    ax.set_title('Akaike Information Criterion', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xticks(K_list)

    # 3. BIC
    ax = axes[2]
    ax.plot(K_list, bic_list, '^-', linewidth=2.5, markersize=10,
            color='mediumseagreen', label='BIC')
    ax.axvline(best_K, linestyle='--', color='red', linewidth=2,
               alpha=0.7, label=f'Best K={best_K}')
    ax.scatter([best_K], [results[best_K]['bic']], s=300,
               color='red', edgecolor='black', linewidth=2,
               zorder=10, marker='*', label='Selected')
    ax.set_xlabel('Number of States (K)', fontsize=13, fontweight='bold')
    ax.set_ylabel('BIC (lower is better)', fontsize=13, fontweight='bold')
    ax.set_title('Bayesian Information Criterion', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xticks(K_list)

    plt.suptitle('rSLDS Model Selection', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'model_comparison.png'),
                dpi=300, bbox_inches='tight')
    print("✓ Saved: model_comparison.png")
    plt.close()


# ======================= 2. 拟合 rSLDS =======================
def fit_rslds(Y, downsample=5, K=4, D=3, num_iters=50):
    """
    使用 SSM 库拟合 recurrent SLDS（rSLDS）

    重要设计：
    - 不再手写 EM 循环
    - 不再调用 rslds.most_likely_states / rslds.smooth（这些在不同版本的 ssm 里接口差异太大）
    - 只调用 rslds.fit(...)，然后从变分后验 q 里面读取 z_map / x_map

    参数
    ----
    Y : ndarray, shape (T, N)
        原始神经活动（时间 x 神经元）
    downsample : int
        下采样步长（比如 5 表示每 5 帧取一帧）
    K : int
        离散状态数
    D : int
        连续隐变量维度
    num_iters : int
        Laplace-EM 的迭代次数

    返回
    ----
    rslds : ssm.SLDS
        拟合好的模型
    z_map : ndarray, shape (T_ds,)
        MAP 离散状态序列（下采样后的时间轴）
    x_map : ndarray, shape (T_ds, D)
        连续隐变量轨迹（下采样后的时间轴）
    lls : list[float]
        每次 EM 迭代的 ELBO / log-likelihood（如无法获取则为 NaN）
    """
    print("\n" + "=" * 60)
    print("FITTING rSLDS (Recurrent Switching LDS)")
    print("=" * 60)

    # ---------------------- 下采样数据 ----------------------
    Y_ds = Y[::downsample]
    T, N = Y_ds.shape
    print(f"Input: T={T} timesteps, N={N} neurons")
    print(f"Target: K={K} discrete states, D={D} continuous latent dims")

    # ---------------------- 初始化模型 ----------------------
    print("\nInitializing rSLDS model...")
    rslds = ssm.SLDS(
        N=N,
        K=K,
        D=D,
        emissions="gaussian",
        transitions="recurrent",
        dynamics="gaussian",
    )

    print("Initializing parameters from data...")
    rslds.initialize(Y_ds)

    # ---------------------- 运行 Laplace-EM ----------------------
    print(f"\nRunning Laplace-EM (num_iters={num_iters})...")
    # 不自己写循环，交给 ssm 内部去做
    results = rslds.fit(
        datas=[Y_ds],
        method="laplace_em",
        variational_posterior="structured_meanfield",
        num_iters=num_iters,
        initialize=False,
    )

    # ---------------------- 从 results 里抽取 elbos 和 q ----------------------
    elbos = None
    q = None

    # 先把 results 变成一个 list，方便遍历
    if isinstance(results, (tuple, list)):
        candidates = list(results)
    else:
        candidates = [results]

    # 挑出 "像后验" 的那个对象，和 "像 ELBO 序列" 的那个对象
    for obj in candidates:
        # 1. 数值一维数组 / 列表 -> 可能是 ELBO 序列
        if isinstance(obj, (np.ndarray, list)):
            arr = np.asarray(obj)
            if np.issubdtype(arr.dtype, np.number) and arr.ndim == 1:
                elbos = arr
                continue

        # 2. 含有连续 / 离散状态相关属性的 -> 认为是 variational posterior q
        if any(
                hasattr(obj, attr)
                for attr in [
                    "mean_continuous_states",
                    "continuous_states",
                    "mean_continuous_state",
                    "most_likely_states",
                    "discrete_states",
                    "mean_discrete_states",
                ]
        ):
            q = obj

    # 如果上面没找到 q，就默认整个 results 就是 q
    if q is None:
        q = results

    # 尝试从 q 里拿 elbos（有些版本把 ELBO 存在 q.elbos 里）
    if elbos is None and hasattr(q, "elbos"):
        elbos = np.asarray(q.elbos)

    # 再不行，就用 NaN 填充一条假 learning curve，防止后面画图报错
    if elbos is None:
        elbos = np.full(num_iters, np.nan, dtype=float)

    print(f"\n✓ Finished EM. Final ELBO: {elbos[-1]:.2f}")

    # ---------------------- 一些小工具函数 ----------------------
    def _first_sequence(arr):
        """
        兼容以下几种情况：
        1. (T,) 或 (T, D) 直接就是单序列
        2. (num_seqs, T) 或 (num_seqs, T, D) 取第一条
        3. tuple/list of arrays - 取第一个，递归展开
        """
        # 如果是 list / tuple，先展平
        if isinstance(arr, (list, tuple)):
            if len(arr) == 0:
                raise ValueError("Empty sequence list from variational posterior.")

            # 取第一个元素（通常是最主要的推断结果）
            arr = arr[0]

            # 如果取出来的还是 list/tuple，递归处理
            if isinstance(arr, (list, tuple)):
                return _first_sequence(arr)

        # 现在 arr 应该是单个 array 了
        arr = np.asarray(arr)

        if arr.ndim == 3:
            return arr[0]  # (num_seqs, T, D) -> (T, D)
        elif arr.ndim in (1, 2):
            return arr
        else:
            raise ValueError(f"Unexpected array shape: {arr.shape}")

    def _get_attr_maybe_callable(obj, name_list):
        """
        从 obj 里尝试按顺序取若干属性，如果属性是可调用的（method），就调用它。
        找到第一个成功的就返回，否则返回 None。
        """
        for name in name_list:
            if hasattr(obj, name):
                val = getattr(obj, name)
                try:
                    val = val() if callable(val) else val
                except TypeError:
                    # 有些 method 需要参数，直接跳过
                    continue
                return val
        return None

    # ---------------------- 从 q 里取离散状态 z_map ----------------------
    # 优先使用"最可能的路径"；如果只有每个 state 的概率，就 argmax 一下
    z_arr = _get_attr_maybe_callable(
        q,
        [
            "most_likely_states",  # 常见：直接给 z_t
            "discrete_states",  # 可能是 z 或 (batch, T)
            "z", "zs",
            "mean_discrete_states",  # 可能是 T x K 概率
        ],
    )

    if z_arr is None:
        raise RuntimeError(
            "无法从 variational posterior 中找到离散状态：\n"
            "尝试的属性有 most_likely_states / discrete_states / z / zs / mean_discrete_states。\n"
            "建议在交互环境里 print(type(q), dir(q)) 看看后验对象里状态是怎么命名的。"
        )

    z_arr = _first_sequence(z_arr)

    # 如果是 (T, K) 概率，就取 argmax
    if z_arr.ndim == 2 and z_arr.shape[1] > 1:
        z_map = z_arr.argmax(axis=-1)
    else:
        z_map = z_arr.astype(int)

    # ---------------------- 从 q 里取连续隐变量 x_map ----------------------
    x_arr = _get_attr_maybe_callable(
        q,
        [
            "mean_continuous_states",  # 常见：T x D 或 (batch, T, D)
            "continuous_states",  # 有些版本这么叫
            "xs",
            "mean_continuous_state",  # 命名差异
        ],
    )

    if x_arr is None:
        raise RuntimeError(
            "无法从 variational posterior 中找到连续隐状态：\n"
            "尝试的属性有 mean_continuous_states / continuous_states / xs / mean_continuous_state。\n"
            "建议在交互环境里 print(type(q), dir(q)) 看看后验对象里 latent 是怎么命名的。"
        )

    x_map = _first_sequence(x_arr)

    print(f"\n✓ Discrete states shape: {z_map.shape}")
    print(f"✓ Continuous latents shape: {x_map.shape}")

    # 返回时把 elbos 转成 list，方便你后面画 learning curve
    return rslds, z_map.astype(int), x_map, list(elbos)


# ======================= 3. 动力学分析 =======================
def analyze_rslds_parameters(rslds, K, D):
    print("\n" + "=" * 60)
    print("rSLDS PARAMETERS ANALYSIS")
    print("=" * 60)

    # 初始状态概率（兼容不同版本的属性名）
    print("\n1. Initial State Probabilities:")
    if hasattr(rslds.init_state_distn, 'initial_state_distn'):
        init_probs = rslds.init_state_distn.initial_state_distn
    elif hasattr(rslds.init_state_distn, 'init_state_distn'):
        init_probs = rslds.init_state_distn.init_state_distn
    else:
        init_probs = np.asarray(rslds.init_state_distn)

    for k in range(K):
        print(f"   State {k + 1}: {init_probs[k]:.3f}")

    # 在原点处的转移矩阵（兼容不同版本）
    print("\n2. Transition Matrix at x=0:")
    if callable(rslds.transitions.transition_matrix):
        # 如果是函数（recurrent transitions），传入 x=0
        trans_matrix = rslds.transitions.transition_matrix(np.zeros(D))
    else:
        # 如果是数组（standard transitions），直接用
        trans_matrix = rslds.transitions.transition_matrix

    print("   " + "".join([f"S{i + 1:2d}    " for i in range(K)]))
    for i in range(K):
        print(f"S{i + 1} " + "  ".join([f"{trans_matrix[i, j]:.3f}" for j in range(K)]))

    # 各状态动力学
    print("\n3. State-Specific Dynamics:")
    for k in range(K):
        A_k = rslds.dynamics.As[k]
        b_k = rslds.dynamics.bs[k]

        print(f"\n   State {k + 1}:")
        print(f"   - Dynamics matrix A_{k + 1}:")
        for row in A_k:
            print(f"     [{', '.join(f'{x:+.3f}' for x in row)}]")
        print(f"   - Bias b_{k + 1}: [{', '.join(f'{x:+.2f}' for x in b_k)}]")

        eigvals = np.linalg.eigvals(A_k)
        max_eigval = np.max(np.abs(eigvals))
        print(f"   - Max |eigenvalue|: {max_eigval:.3f} "
              f"({'stable' if max_eigval < 1 else 'unstable'})")

    return trans_matrix


def compute_state_occupancy(z_map, K, is_social, is_after_reunion):
    print("\n" + "=" * 60)
    print("STATE OCCUPANCY ANALYSIS")
    print("=" * 60)

    # 把行为 mask 下采样到和 z_map 对齐
    ds = len(is_social) // len(z_map)
    is_social_ds = is_social[::ds][:len(z_map)]
    is_after_ds = is_after_reunion[::ds][:len(z_map)]

    occupancy = {}
    occupancy['Overall'] = np.bincount(z_map, minlength=K) / len(z_map)

    mask_before = ~is_after_ds
    if mask_before.sum() > 0:
        occupancy['Before Reunion'] = (np.bincount(z_map[mask_before], minlength=K)
                                       / mask_before.sum())

    mask_after_nonsocial = is_after_ds & ~is_social_ds
    if mask_after_nonsocial.sum() > 0:
        occupancy['After (Non-social)'] = (np.bincount(z_map[mask_after_nonsocial], minlength=K)
                                           / mask_after_nonsocial.sum())

    if is_social_ds.sum() > 0:
        occupancy['Social Interaction'] = (np.bincount(z_map[is_social_ds], minlength=K)
                                           / is_social_ds.sum())

    print("\n" + " " * 25 + "".join([f"S{i + 1:2d}     " for i in range(K)]))
    for cond, occ in occupancy.items():
        print(f"{cond:24s} " + "  ".join([f"{x:.3f}" for x in occ]))

    return occupancy


def compute_dwell_times(z_map, K, dt_fit):
    print("\n" + "=" * 60)
    print("STATE DWELL TIME ANALYSIS")
    print("=" * 60)

    dwell_times = {k: [] for k in range(K)}
    current_state = z_map[0]
    duration = 1

    for i in range(1, len(z_map)):
        if z_map[i] == current_state:
            duration += 1
        else:
            dwell_times[current_state].append(duration * dt_fit)
            current_state = z_map[i]
            duration = 1
    dwell_times[current_state].append(duration * dt_fit)

    print(f"\n{'State':<8} {'Count':<8} {'Mean (s)':<12} "
          f"{'Median (s)':<12} {'Max (s)':<10}")
    print("-" * 60)

    for k in range(K):
        if len(dwell_times[k]) > 0:
            dt_k = np.array(dwell_times[k])
            print(f"S{k + 1:<7} {len(dt_k):<8} {dt_k.mean():<12.2f} "
                  f"{np.median(dt_k):<12.2f} {dt_k.max():<10.2f}")

    return dwell_times


def analyze_latent_trajectories(x_map, z_map, K):
    print("\n" + "=" * 60)
    print("CONTINUOUS LATENT TRAJECTORY ANALYSIS")
    print("=" * 60)

    print(f"\n{'State':<8} {'N':<8} {'Speed (mean±std)':<20} "
          f"{'Volume':<12} {'Entropy':<10}")
    print("-" * 70)

    metrics = {}
    for k in range(K):
        mask_k = z_map == k
        x_k = x_map[mask_k]

        if len(x_k) < 2:
            print(f"S{k + 1:<7} {len(x_k):<8} (insufficient data)")
            continue

        velocity = np.diff(x_k, axis=0)
        speed = np.linalg.norm(velocity, axis=1)

        try:
            if len(x_k) >= x_k.shape[1] + 1:
                hull = ConvexHull(x_k)
                volume = hull.volume
            else:
                volume = 0
        except:
            volume = 0

        try:
            hist, _ = np.histogramdd(x_k, bins=10)
            prob = hist / (hist.sum() + 1e-10)
            traj_entropy = entropy(prob.flatten() + 1e-10)
        except:
            traj_entropy = 0

        metrics[k] = {
            'n_samples': len(x_k),
            'mean_speed': speed.mean(),
            'std_speed': speed.std(),
            'volume': volume,
            'entropy': traj_entropy,
            'centroid': x_k.mean(axis=0)
        }

        speed_str = f"{speed.mean():.3f}±{speed.std():.3f}"
        print(f"S{k + 1:<7} {len(x_k):<8} {speed_str:<20} "
              f"{volume:<12.3f} {traj_entropy:<10.3f}")

    return metrics


# ======================= 4. 可视化函数 =======================

def plot_learning_curve(lls, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(lls, linewidth=2, color='steelblue', marker='o', markersize=4)
    ax.set_xlabel('EM Iteration', fontsize=13, fontweight='bold')
    ax.set_ylabel('Log-Likelihood', fontsize=13, fontweight='bold')
    ax.set_title('rSLDS Learning Curve', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'learning_curve.png'),
                dpi=300, bbox_inches='tight')
    print("✓ Saved: learning_curve.png")
    plt.close()


def plot_state_timeline(z_map, t_ds, reunion_abs, social_intervals, K, out_dir):
    fig, ax = plt.subplots(figsize=(16, 5))
    t_rel = t_ds - reunion_abs

    colors = plt.cm.Set2(np.linspace(0, 1, K))
    state_names = ['Transition', 'Special', 'Social', 'Baseline']

    for k in range(K):
        mask_k = z_map == k
        segments = []
        start = None
        for i, is_k in enumerate(mask_k):
            if is_k and start is None:
                start = i
            elif not is_k and start is not None:
                segments.append((start, i - 1))
                start = None
        if start is not None:
            segments.append((start, len(mask_k) - 1))

        for idx, (s_idx, e_idx) in enumerate(segments):
            ax.fill_between(
                [t_rel[s_idx], t_rel[e_idx]],
                0, k + 1,
                color=colors[k],
                alpha=0.7,
                label=f'State {k + 1}: {state_names[k]}' if idx == 0 else ''
            )

    ax.plot(t_rel, z_map + 1, linewidth=0.8, color='black', alpha=0.5, zorder=10)

    for i, (s, e) in enumerate(social_intervals):
        ax.axvspan(s - reunion_abs, e - reunion_abs,
                   alpha=0.15, color='red', zorder=0,
                   label='Social Bout' if i == 0 else '')

    ax.axvline(0, linestyle='--', color='black', linewidth=2,
               label='Reunion Event', zorder=15)

    ax.set_xlabel('Time from Reunion (s)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Hidden State', fontsize=14, fontweight='bold')
    ax.set_title('rSLDS State Sequence Timeline', fontsize=16, fontweight='bold')
    ax.set_ylim(0.3, K + 0.7)
    ax.set_yticks(np.arange(1, K + 1))
    ax.set_yticklabels([f'State {k + 1}' for k in range(K)])

    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
              fontsize=11, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'state_timeline.png'),
                dpi=300, bbox_inches='tight')
    print("✓ Saved: state_timeline.png")
    plt.close()


def plot_latent_trajectories_3d(x_map, z_map, K, out_dir):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(14, 10))
    colors = plt.cm.Set2(np.linspace(0, 1, K))
    state_names = ['Transition', 'Special', 'Social', 'Baseline']

    ax = fig.add_subplot(221, projection='3d')
    for k in range(K):
        mask_k = z_map == k
        ax.scatter(x_map[mask_k, 0], x_map[mask_k, 1], x_map[mask_k, 2],
                   c=[colors[k]], s=8, alpha=0.6,
                   label=f'S{k + 1} ({state_names[k]})')
    ax.set_xlabel('Latent 1', fontsize=11)
    ax.set_ylabel('Latent 2', fontsize=11)
    ax.set_zlabel('Latent 3', fontsize=11)
    ax.set_title('3D Latent Trajectory (rSLDS)', fontsize=13, fontweight='bold')
    ax.legend(markerscale=2, fontsize=9)

    ax = fig.add_subplot(222)
    for k in range(K):
        mask_k = z_map == k
        ax.scatter(x_map[mask_k, 0], x_map[mask_k, 1],
                   c=[colors[k]], s=5, alpha=0.6, label=f'S{k + 1}')
    ax.set_xlabel('Latent 1')
    ax.set_ylabel('Latent 2')
    ax.set_title('Latent 1 vs 2')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(223)
    for k in range(K):
        mask_k = z_map == k
        ax.scatter(x_map[mask_k, 0], x_map[mask_k, 2],
                   c=[colors[k]], s=5, alpha=0.6)
    ax.set_xlabel('Latent 1')
    ax.set_ylabel('Latent 3')
    ax.set_title('Latent 1 vs 3')
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(224)
    for k in range(K):
        mask_k = z_map == k
        ax.scatter(x_map[mask_k, 1], x_map[mask_k, 2],
                   c=[colors[k]], s=5, alpha=0.6)
    ax.set_xlabel('Latent 2')
    ax.set_ylabel('Latent 3')
    ax.set_title('Latent 2 vs 3')
    ax.grid(alpha=0.3)

    plt.suptitle('Continuous Latent Space (rSLDS)',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'latent_trajectories_3d.png'),
                dpi=300, bbox_inches='tight')
    print("✓ Saved: latent_trajectories_3d.png")
    plt.close()


def plot_dynamics_vector_field(rslds, K, out_dir):
    fig, axes = plt.subplots(1, K, figsize=(5 * K, 4))
    if K == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, K))
    state_names = ['Transition', 'Special', 'Social', 'Baseline']

    x_range = np.linspace(-3, 3, 15)
    y_range = np.linspace(-3, 3, 15)
    X, Y = np.meshgrid(x_range, y_range)

    for k in range(K):
        ax = axes[k]
        A_k = rslds.dynamics.As[k]
        b_k = rslds.dynamics.bs[k]

        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x_curr = np.array([X[i, j], Y[i, j], 0])
                x_next = A_k @ x_curr + b_k
                U[i, j] = x_next[0] - x_curr[0]
                V[i, j] = x_next[1] - x_curr[1]

        ax.quiver(X, Y, U, V, alpha=0.6, color=colors[k])
        ax.set_xlabel('Latent 1', fontsize=11)
        ax.set_ylabel('Latent 2', fontsize=11)
        ax.set_title(f'State {k + 1}: {state_names[k]}',
                     fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_aspect('equal')

    plt.suptitle('State-Specific Dynamics (Vector Fields)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'dynamics_vector_fields.png'),
                dpi=300, bbox_inches='tight')
    print("✓ Saved: dynamics_vector_fields.png")
    plt.close()


def plot_transition_matrix(trans_matrix, K, out_dir):
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        trans_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
        xticklabels=[f'S{i + 1}' for i in range(K)],
        yticklabels=[f'S{i + 1}' for i in range(K)],
        cbar_kws={'label': 'Transition Probability'},
        ax=ax, vmin=0, vmax=1, linewidths=1, linecolor='gray'
    )
    ax.set_xlabel('To State', fontsize=13, fontweight='bold')
    ax.set_ylabel('From State', fontsize=13, fontweight='bold')
    ax.set_title('State Transition Matrix (rSLDS)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'transition_matrix.png'),
                dpi=300, bbox_inches='tight')
    print("✓ Saved: transition_matrix.png")
    plt.close()


def plot_occupancy_comparison(occupancy, K, out_dir):
    fig, ax = plt.subplots(figsize=(12, 6))
    conditions = list(occupancy.keys())
    x = np.arange(K)
    width = 0.8 / len(conditions)
    colors_cond = plt.cm.Pastel1(np.linspace(0, 1, len(conditions)))

    for i, cond in enumerate(conditions):
        offset = (i - len(conditions) / 2 + 0.5) * width
        bars = ax.bar(x + offset, occupancy[cond], width,
                      label=cond, alpha=0.85,
                      color=colors_cond[i],
                      edgecolor='black', linewidth=0.8)
        for bar in bars:
            h = bar.get_height()
            if h > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, h,
                        f'{h:.2f}', ha='center', va='bottom',
                        fontsize=8, fontweight='bold')

    ax.set_xlabel('State', fontsize=13, fontweight='bold')
    ax.set_ylabel('Occupancy Rate', fontsize=13, fontweight='bold')
    ax.set_title('State Occupancy Across Conditions (rSLDS)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'State {i + 1}' for i in range(K)])
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'occupancy_comparison.png'),
                dpi=300, bbox_inches='tight')
    print("✓ Saved: occupancy_comparison.png")
    plt.close()


def plot_dwell_time_distributions(dwell_times, K, out_dir):
    fig, axes = plt.subplots(1, K, figsize=(5 * K, 4), sharey=True)
    if K == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, K))
    for k in range(K):
        if len(dwell_times[k]) > 0:
            dt_k = np.array(dwell_times[k])
            axes[k].hist(dt_k, bins=30, edgecolor='black',
                         alpha=0.75, color=colors[k])
            axes[k].axvline(dt_k.mean(), color='red', linestyle='--',
                            linewidth=2, label=f'Mean={dt_k.mean():.1f}s')
            axes[k].set_title(f'State {k + 1}', fontsize=12, fontweight='bold')
            axes[k].set_xlabel('Dwell Time (s)', fontsize=11)
            axes[k].legend(fontsize=9)
            axes[k].grid(alpha=0.3)
        if k == 0:
            axes[k].set_ylabel('Count', fontsize=11)

    plt.suptitle('State Dwell Time Distributions (rSLDS)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'dwell_time_distributions.png'),
                dpi=300, bbox_inches='tight')
    print("✓ Saved: dwell_time_distributions.png")
    plt.close()


def plot_energy_landscape(trans_matrix, K, out_dir):
    """
    基于转移矩阵计算能量景观（稳态概率的负对数）

    能量定义：E_i = -log(π_i)，其中 π 是稳态分布
    """
    print("\n" + "=" * 60)
    print("COMPUTING ENERGY LANDSCAPE")
    print("=" * 60)

    # 计算稳态分布（转移矩阵的主特征向量）
    eigvals, eigvecs = np.linalg.eig(trans_matrix.T)
    steady_idx = np.argmax(np.abs(eigvals))
    steady_state = np.abs(eigvecs[:, steady_idx])
    steady_state /= steady_state.sum()

    # 能量 = -log(概率)
    energy = -np.log(steady_state + 1e-10)
    energy -= energy.min()  # 归一化，最低能量为 0

    print("\nSteady-state probabilities:")
    for k in range(K):
        print(f"  State {k + 1}: π = {steady_state[k]:.4f}, Energy = {energy[k]:.3f}")

    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, K))
    state_names = ['Transition', 'Special', 'Social', 'Baseline']

    # 左图：能量柱状图
    bars = ax1.bar(range(K), energy, color=colors,
                   edgecolor='black', linewidth=2, alpha=0.8)
    for i, (bar, e) in enumerate(zip(bars, energy)):
        ax1.text(i, e + 0.05, f'{e:.2f}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.set_xlabel('State', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Free Energy (a.u.)', fontsize=13, fontweight='bold')
    ax1.set_title('Energy Landscape', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(K))
    ax1.set_xticklabels([f'S{i + 1}\n{state_names[i]}' for i in range(K)])
    ax1.grid(axis='y', alpha=0.3)

    # 右图：稳态概率（反映"吸引力"）
    bars = ax2.bar(range(K), steady_state, color=colors,
                   edgecolor='black', linewidth=2, alpha=0.8)
    for i, (bar, p) in enumerate(zip(bars, steady_state)):
        ax2.text(i, p + 0.01, f'{p:.3f}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax2.set_xlabel('State', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Steady-state Probability', fontsize=13, fontweight='bold')
    ax2.set_title('Attractor Strength', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(K))
    ax2.set_xticklabels([f'S{i + 1}\n{state_names[i]}' for i in range(K)])
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'energy_landscape.png'),
                dpi=300, bbox_inches='tight')
    print("✓ Saved: energy_landscape.png")
    plt.close()

    return steady_state, energy


def plot_latent_vector_field(x_map, z_map, rslds, K, out_dir):
    """
    在隐空间中绘制实际轨迹点，并叠加动力学向量

    - 每个点用颜色标识所属状态
    - 箭头显示该点处的动力学方向 (x_{t+1} - x_t)
    """
    print("\n" + "=" * 60)
    print("GENERATING LATENT SPACE VECTOR FIELD")
    print("=" * 60)

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(16, 12))
    colors = plt.cm.Set2(np.linspace(0, 1, K))
    state_names = ['Transition', 'Special', 'Social', 'Baseline']

    # ===== 1. 3D 视图：轨迹 + 向量场 =====
    ax = fig.add_subplot(221, projection='3d')

    # 稀疏采样（避免画太多箭头）
    subsample = 50
    indices = np.arange(0, len(x_map) - 1, subsample)

    for k in range(K):
        mask_k = z_map == k
        # 绘制轨迹点
        ax.scatter(x_map[mask_k, 0], x_map[mask_k, 1], x_map[mask_k, 2],
                   c=[colors[k]], s=3, alpha=0.4, label=f'S{k + 1} ({state_names[k]})')

        # 绘制向量（只画属于该状态的）
        for idx in indices:
            if z_map[idx] == k:
                x_curr = x_map[idx]
                A_k = rslds.dynamics.As[k]
                b_k = rslds.dynamics.bs[k]
                x_next = A_k @ x_curr + b_k

                # 向量从 x_curr 指向 x_next
                dx = x_next - x_curr
                ax.quiver(x_curr[0], x_curr[1], x_curr[2],
                          dx[0], dx[1], dx[2],
                          color=colors[k], alpha=0.6,
                          arrow_length_ratio=0.3, linewidth=1.5)

    ax.set_xlabel('Latent 1', fontsize=11, fontweight='bold')
    ax.set_ylabel('Latent 2', fontsize=11, fontweight='bold')
    ax.set_zlabel('Latent 3', fontsize=11, fontweight='bold')
    ax.set_title('3D Latent Trajectory + Dynamics', fontsize=13, fontweight='bold')
    ax.legend(markerscale=3, fontsize=9)

    # ===== 2-4. 三个 2D 投影 =====
    projections = [
        (222, 0, 1, 'Latent 1', 'Latent 2'),
        (223, 0, 2, 'Latent 1', 'Latent 3'),
        (224, 1, 2, 'Latent 2', 'Latent 3'),
    ]

    for subplot_idx, dim1, dim2, xlabel, ylabel in projections:
        ax = fig.add_subplot(subplot_idx)

        for k in range(K):
            mask_k = z_map == k
            ax.scatter(x_map[mask_k, dim1], x_map[mask_k, dim2],
                       c=[colors[k]], s=5, alpha=0.5)

            # 画向量
            for idx in indices:
                if z_map[idx] == k:
                    x_curr = x_map[idx]
                    A_k = rslds.dynamics.As[k]
                    b_k = rslds.dynamics.bs[k]
                    x_next = A_k @ x_curr + b_k

                    ax.arrow(x_curr[dim1], x_curr[dim2],
                             x_next[dim1] - x_curr[dim1],
                             x_next[dim2] - x_curr[dim2],
                             color=colors[k], alpha=0.5,
                             width=0.01, head_width=0.05)

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'{xlabel} vs {ylabel}', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)

    plt.suptitle('Latent Space Trajectories with Dynamics Vectors',
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'latent_vector_field.png'),
                dpi=300, bbox_inches='tight')
    print("✓ Saved: latent_vector_field.png")
    plt.close()


# ======================= 5. 主流程 =======================
def main():
    print("\n" + "=" * 70)
    print(" " * 15 + "rSLDS FULL ANALYSIS PIPELINE")
    print("=" * 70)

    # ==================== 1. 加载数据 ====================
    Y, t, dt = load_neural_data(data_path)
    social_intervals, reunion_rel = load_behavior(beh_path, reunion_abs)
    is_social, is_after_reunion = build_behavior_mask(t, reunion_abs, social_intervals)

    # 时间截断
    cutoff_time = 300.0  # 秒
    valid_idx = t >= cutoff_time

    Y = Y[valid_idx]
    t = t[valid_idx]
    is_social = is_social[valid_idx]
    is_after_reunion = is_after_reunion[valid_idx]

    print(f"\n⚠️  Discarded first {cutoff_time:.0f}s")
    print(f"✓ Remaining: T={len(t)} timesteps ({t[-1] - t[0]:.1f}s)")

    # ==================== 2. 模型比较（新增） ====================
    # 设置开关：是否进行模型比较
    ENABLE_MODEL_COMPARISON = False  # 改成 False 则跳过比较，直接用 K_states

    if ENABLE_MODEL_COMPARISON:
        print("\n" + "🔍" * 35)
        print("Starting model comparison across different K values...")
        print("🔍" * 35)

        # 调用模型比较函数
        best_K, comparison_results = compare_K_values(
            Y, t, is_social, is_after_reunion,
            K_range=[2, 3, 4, 5, 6],  # 可自定义要测试的 K 范围
            out_dir=out_dir
        )

        # 使用最优的 K
        K_states_final = best_K
        print(f"\n✅ Selected K={K_states_final} based on BIC criterion")

        # 从比较结果中提取对应的模型（避免重复拟合）
        rslds = comparison_results[K_states_final]['rslds']
        z_map = comparison_results[K_states_final]['z_map']
        x_map = comparison_results[K_states_final]['x_map']
        lls = comparison_results[K_states_final]['lls']

    else:
        # 直接使用预设的 K_states（原始逻辑）
        K_states_final = K_states
        print(f"\n⚠️  Skipping model comparison, using K={K_states_final}")

        rslds, z_map, x_map, lls = fit_rslds(
            Y, downsample=downsample, K=K_states_final,
            D=D_latent, num_iters=num_iters
        )

    # ==================== 3. 时间对齐 ====================
    t_ds = t[::downsample][:len(z_map)]
    dt_fit = dt * downsample

    # ==================== 4. 动力学分析 ====================
    trans_matrix = analyze_rslds_parameters(rslds, K_states_final, D_latent)
    occupancy = compute_state_occupancy(z_map, K_states_final, is_social, is_after_reunion)
    dwell_times = compute_dwell_times(z_map, K_states_final, dt_fit)
    metrics = analyze_latent_trajectories(x_map, z_map, K_states_final)

    # ==================== 5. 可视化 ====================
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70 + "\n")

    plot_learning_curve(lls, out_dir)
    plot_state_timeline(z_map, t_ds, reunion_abs, social_intervals, K_states_final, out_dir)
    plot_latent_trajectories_3d(x_map, z_map, K_states_final, out_dir)
    plot_dynamics_vector_field(rslds, K_states_final, out_dir)
    plot_transition_matrix(trans_matrix, K_states_final, out_dir)
    plot_occupancy_comparison(occupancy, K_states_final, out_dir)
    plot_dwell_time_distributions(dwell_times, K_states_final, out_dir)

    # 新增的两个图
    steady_state, energy = plot_energy_landscape(trans_matrix, K_states_final, out_dir)
    plot_latent_vector_field(x_map, z_map, rslds, K_states_final, out_dir)

    # ==================== 6. 保存结果 ====================
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # 保存数值结果
    save_dict = {
        'z_map': z_map,
        'x_map': x_map,
        't_ds': t_ds,
        'trans_matrix': trans_matrix,
        'occupancy': occupancy,
        'dwell_times': dwell_times,
        'learning_curve': lls,
        'dynamics_matrices': rslds.dynamics.As,
        'dynamics_biases': rslds.dynamics.bs,
        'steady_state': steady_state,
        'energy': energy,
        'K_final': K_states_final  # 保存最终使用的 K
    }

    # 如果进行了模型比较，保存比较结果
    if ENABLE_MODEL_COMPARISON:
        save_dict['model_comparison'] = {
            k: {
                'll': v['ll'],
                'aic': v['aic'],
                'bic': v['bic'],
                'num_params': v['num_params'],
                'min_occupancy': v['min_occupancy']
            }
            for k, v in comparison_results.items()
        }

    np.savez(
        os.path.join(out_dir, 'rslds_results.npz'),
        **save_dict
    )

    # 保存文本报告
    with open(os.path.join(out_dir, 'rslds_report.txt'), 'w', encoding='utf-8') as f:
        f.write("rSLDS DYNAMICS ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: Recurrent Switching Linear Dynamical System\n")
        f.write(f"Discrete States: K={K_states_final}\n")
        f.write(f"Continuous Latent Dims: D={D_latent}\n")
        f.write(f"Data cutoff: First {cutoff_time:.0f}s discarded\n")
        f.write(f"Final Log-Likelihood: {lls[-1]:.2f}\n\n")

        # 如果进行了模型比较，记录比较结果
        if ENABLE_MODEL_COMPARISON:
            f.write("MODEL SELECTION:\n")
            f.write(f"  Tested K ∈ {list(comparison_results.keys())}\n")
            f.write(f"  Selected K={K_states_final} (BIC criterion)\n")
            f.write("\n  Comparison table:\n")
            f.write(f"  {'K':<5} {'LL':<12} {'AIC':<12} {'BIC':<12}\n")
            f.write("  " + "-" * 45 + "\n")
            for k in sorted(comparison_results.keys()):
                res = comparison_results[k]
                marker = " *" if k == K_states_final else ""
                f.write(f"  {k:<5} {res['ll']:<12.1f} {res['aic']:<12.1f} "
                        f"{res['bic']:<12.1f}{marker}\n")
            f.write("\n")

        f.write("OCCUPANCY RATES:\n")
        for cond, occ in occupancy.items():
            f.write(f"  {cond}:\n")
            for k in range(K_states_final):
                f.write(f"    State {k + 1}: {occ[k]:.3f}\n")

        f.write("\nENERGY LANDSCAPE:\n")
        for k in range(K_states_final):
            f.write(f"  State {k + 1}: Energy = {energy[k]:.3f}, "
                    f"π = {steady_state[k]:.4f}\n")

    print(f"✓ All results saved to: {out_dir}")
    print("\n" + "=" * 70)
    print(" " * 20 + "ANALYSIS COMPLETE!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
