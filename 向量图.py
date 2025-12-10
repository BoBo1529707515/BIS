# mouse_rslds_analysis.py
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d

import ssm

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ======================= 参数设置 =======================
base_out_dir = r"F:\工作文件\RA\python\吸引子\figs"
os.makedirs(base_out_dir, exist_ok=True)

data_path = r"F:\工作文件\RA\python\吸引子\Mouse1_for_ssm.npz"
beh_path = r"F:\工作文件\RA\数据集\时间戳\FVB_Panneuronal_Reunion_Isolation_Day3_Mouse#1.xlsx"

reunion_abs = 903.0

# 超参数
K_TRY = [2, 3, 4]
D_TRY = [2, 3]
downsample = 15
num_iters = 25


# ======================= 数据加载 =======================
def load_neural_data(npz_path):
    print("=" * 60)
    print("LOADING NEURAL DATA")
    print("=" * 60)
    dat = np.load(npz_path, allow_pickle=True)
    Y = np.asarray(dat["Y"])
    t = np.asarray(dat["t"])
    dt = float(dat["dt"].item())

    # 增强预处理
    Y_z = (Y - Y.mean(axis=0)) / (Y.std(axis=0) + 1e-6)
    Y_z = np.clip(Y_z, -3, 3)
    Y_z = gaussian_filter1d(Y_z, sigma=1, axis=0)

    print(f"✓ T={len(t)}, N={Y.shape[1]}, dt={dt:.3f}s")
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
    print(f"✓ Found {len(social_intervals)} social bouts")
    return social_intervals, reunion_rel


def extract_reunion_contact_epoch(Y, t, reunion_abs, social_intervals):
    print("\n" + "=" * 60)
    print("EXTRACTING REUNION+CONTACT EPOCH")
    print("=" * 60)

    is_after = t >= reunion_abs
    is_contact = np.zeros_like(t, dtype=bool)
    for (start, end) in social_intervals:
        if end > reunion_abs:
            effective_start = max(start, reunion_abs)
            is_contact |= (t >= effective_start) & (t <= end)

    mask = is_after & is_contact
    if mask.sum() == 0:
        raise ValueError("No reunion+contact data!")

    Y_epoch = Y[mask]
    t_epoch = t[mask]
    duration = t_epoch[-1] - t_epoch[0]

    print(f"✓ Extracted {len(t_epoch)} frames ({duration:.1f}s)")
    return Y_epoch, t_epoch, mask


# ======================= 🔧 修复的状态提取（版本兼容）=======================
def _extract_states_from_rslds_fixed(rslds, Y_ds, K, D):
    """修复版状态提取（处理 ssm 版本差异）"""
    print("  [Fixed Extract] 从模型提取状态（兼容版）...")
    T = len(Y_ds)

    # ===== 步骤1：提取离散状态（修复 log_likelihoods）=====
    try:
        print("    尝试 Viterbi...")

        # 🔧 修复：检查 log_likelihoods 的返回值
        log_likes_result = rslds.emissions.log_likelihoods(Y_ds)

        if log_likes_result is None:
            raise ValueError("log_likelihoods returned None")

        # 处理不同返回格式
        if isinstance(log_likes_result, (list, tuple)):
            log_likes = np.asarray(log_likes_result[0])
        else:
            log_likes = np.asarray(log_likes_result)

        # 确保形状正确 (T, K)
        if log_likes.ndim == 1:
            log_likes = log_likes[:, None]

        # 获取初始分布和转移矩阵
        log_pi0 = rslds.init_state_distn.log_initial_state_distn
        log_Ps = rslds.transitions.log_transition_matrices(None, None, None, [T])

        if isinstance(log_Ps, (list, tuple)):
            log_Ps = log_Ps[0]

        # Viterbi 算法
        delta = log_pi0 + log_likes[0]
        psi = np.zeros((T, K), dtype=int)

        for t in range(1, T):
            delta_new = np.zeros(K)
            for k in range(K):
                vals = delta + log_Ps[t - 1, :, k]
                psi[t, k] = vals.argmax()
                delta_new[k] = vals.max() + log_likes[t, k]
            delta = delta_new

        # 回溯
        z_map = np.zeros(T, dtype=int)
        z_map[-1] = delta.argmax()
        for t in range(T - 2, -1, -1):
            z_map[t] = psi[t + 1, z_map[t + 1]]

        print(f"    ✓ Viterbi 成功: z_map.shape={z_map.shape}")

    except Exception as e:
        print(f"    × Viterbi failed: {type(e).__name__}: {str(e)[:80]}")
        # 兜底：用转移矩阵的最大后验
        try:
            z_map = np.zeros(T, dtype=int)
            z_map[0] = rslds.init_state_distn.log_initial_state_distn.argmax()
            for t in range(1, T):
                z_map[t] = np.random.choice(K, p=np.ones(K) / K)
            print(f"    ⚠️  使用简化离散状态")
        except:
            z_map = np.random.randint(0, K, size=T)
            print(f"    ⚠️  使用随机离散状态")

    # ===== 步骤2：提取连续状态（修复 Vs 属性）=====
    try:
        print("    尝试 Kalman 平滑...")
        x_map = _run_kalman_smoother_fixed(rslds, Y_ds, z_map, T, D, K)
        print(f"    ✓ Kalman 成功: x_map.shape={x_map.shape}")
        return z_map, x_map

    except Exception as e:
        print(f"    × Kalman failed: {type(e).__name__}: {str(e)[:80]}")

    # 兜底：PCA
    print("    [Fallback] 使用 PCA")
    pca = PCA(n_components=D)
    x_map = pca.fit_transform(Y_ds)
    print(f"    ⚠️  PCA 解释方差: {pca.explained_variance_ratio_.sum():.2%}")

    return z_map, x_map


def _run_kalman_smoother_fixed(rslds, Y, z_map, T, D, K):
    """修复版 Kalman 平滑（处理 Vs 属性问题）"""
    N = Y.shape[1]

    # 获取动力学参数
    As = rslds.dynamics.As  # (K, D, D)
    bs = rslds.dynamics.bs  # (K, D)

    # 🔧 修复：Vs 可能叫 Sigmas 或 _sqrt_Sigmas
    if hasattr(rslds.dynamics, 'Vs'):
        Qs = rslds.dynamics.Vs
    elif hasattr(rslds.dynamics, 'Sigmas'):
        Qs = rslds.dynamics.Sigmas
    elif hasattr(rslds.dynamics, '_sqrt_Sigmas'):
        sqrt_Qs = rslds.dynamics._sqrt_Sigmas
        Qs = np.array([sq @ sq.T for sq in sqrt_Qs])
    else:
        # 兜底：单位协方差
        Qs = np.tile(np.eye(D)[None, :, :], (K, 1, 1))
        print("    ⚠️  动力学协方差未找到，使用单位矩阵")

    # 获取观测参数
    Cs = rslds.emissions.Cs  # (K, N, D) 或 (N, D)
    ds = rslds.emissions.ds  # (K, N) 或 (N,)

    # 🔧 修复：观测协方差的多种命名
    if hasattr(rslds.emissions, 'Vs'):
        Rs = rslds.emissions.Vs
    elif hasattr(rslds.emissions, 'Sigmas'):
        Rs = rslds.emissions.Sigmas
    elif hasattr(rslds.emissions, '_sqrt_Sigmas'):
        sqrt_Rs = rslds.emissions._sqrt_Sigmas
        Rs = np.array([sq @ sq.T for sq in sqrt_Rs])
    elif hasattr(rslds.emissions, 'inv_etas'):
        # GaussianEmissions 用 inv_etas (对角协方差的逆)
        inv_etas = rslds.emissions.inv_etas
        Rs = np.array([np.diag(1.0 / np.exp(ie)) for ie in inv_etas])
        print("    ✓ 使用 inv_etas 计算观测协方差")
    else:
        Rs = np.tile(np.eye(N)[None, :, :], (K, 1, 1))
        print("    ⚠️  观测协方差未找到，使用单位矩阵")

    # 处理维度
    if Cs.ndim == 2:
        Cs = np.tile(Cs[None, :, :], (K, 1, 1))
    if ds.ndim == 1:
        ds = np.tile(ds[None, :], (K, 1))
    if Rs.ndim == 2:
        Rs = np.tile(Rs[None, :, :], (K, 1, 1))
    if Qs.ndim == 2:
        Qs = np.tile(Qs[None, :, :], (K, 1, 1))

    # Kalman 滤波和平滑
    x0 = np.zeros(D)
    P0 = np.eye(D) * 10.0

    xs_filt = np.zeros((T, D))
    Ps_filt = np.zeros((T, D, D))

    x_pred = x0
    P_pred = P0

    for t in range(T):
        k = z_map[t]
        A = As[k]
        b = bs[k]
        Q = Qs[k]
        C = Cs[k]
        d = ds[k]
        R = Rs[k]

        # Update
        y = Y[t]
        innovation = y - C @ x_pred - d
        S = C @ P_pred @ C.T + R

        try:
            K_gain = P_pred @ C.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K_gain = P_pred @ C.T @ np.linalg.pinv(S)

        x_filt = x_pred + K_gain @ innovation
        P_filt = (np.eye(D) - K_gain @ C) @ P_pred

        xs_filt[t] = x_filt
        Ps_filt[t] = P_filt

        # Predict
        if t < T - 1:
            k_next = z_map[t + 1]
            A_next = As[k_next]
            b_next = bs[k_next]
            Q_next = Qs[k_next]

            x_pred = A_next @ x_filt + b_next
            P_pred = A_next @ P_filt @ A_next.T + Q_next

    # RTS 平滑
    xs_smooth = np.copy(xs_filt)

    for t in range(T - 2, -1, -1):
        k = z_map[t]
        k_next = z_map[t + 1]

        A = As[k_next]
        b = bs[k_next]
        Q = Qs[k_next]

        P_pred = A @ Ps_filt[t] @ A.T + Q

        try:
            J = Ps_filt[t] @ A.T @ np.linalg.inv(P_pred)
        except np.linalg.LinAlgError:
            J = Ps_filt[t] @ A.T @ np.linalg.pinv(P_pred)

        xs_smooth[t] = xs_filt[t] + J @ (xs_smooth[t + 1] - A @ xs_filt[t] - b)

    return xs_smooth


# ======================= 拟合函数（修复 elbos）=======================
def fit_rslds_with_fallback(Y, downsample=15, K_try=[2, 3, 4], D_try=[2, 3], num_iters=25):
    print("\n" + "=" * 60)
    print("FITTING rSLDS (AUTO-FALLBACK)")
    print("=" * 60)

    T_raw = len(Y)
    T_ds = T_raw // downsample

    print(f"原始数据: T={T_raw}, 下采样后: T={T_ds}")

    if T_ds < 50:
        downsample = max(5, T_raw // 80)
        T_ds = T_raw // downsample
        print(f"调整下采样率: {downsample}, T={T_ds}")

    for K in K_try:
        for D in D_try:
            print(f"\n[Attempt] K={K}, D={D}")
            try:
                rslds, z_map, x_map, lls = _fit_single(
                    Y, downsample=downsample, K=K, D=D, num_iters=num_iters
                )
                print(f"✅ SUCCESS with K={K}, D={D}")
                return rslds, z_map, x_map, lls, K, D
            except Exception as e:
                print(f"✗ Failed: {type(e).__name__}: {str(e)[:100]}")
                continue

    print("\n[Fallback] All rSLDS failed, using HMM...")
    return _fit_hmm_fallback(Y, downsample=downsample, K=2)


def _fit_single(Y, downsample=15, K=2, D=2, num_iters=25):
    """单次rSLDS拟合（修复版）"""
    Y_ds = Y[::downsample]
    T, N = Y_ds.shape
    print(f"  Input: T={T}, N={N}, K={K}, D={D}")

    if T < 30:
        raise ValueError(f"Too few timesteps: {T}")

    rslds = ssm.SLDS(
        N=N, K=K, D=D,
        emissions="gaussian",
        transitions="standard",
        dynamics="gaussian"
    )

    rslds.initialize(Y_ds)

    print(f"  Running Laplace-EM...")
    results = rslds.fit(
        datas=[Y_ds],
        method="laplace_em",
        variational_posterior="structured_meanfield",
        num_iters=num_iters,
        initialize=False
    )

    # 解析结果
    q, elbos = _parse_fit_results(results, num_iters)

    # 提取状态（修复版）
    print("  提取状态...")
    z_map, x_map = _extract_states_from_rslds_fixed(rslds, Y_ds, K, D)

    # 🔧 修复：确保 elbos 是列表
    if not isinstance(elbos, (list, np.ndarray)):
        elbos = [float(elbos)] if elbos is not None else [np.nan] * num_iters
    else:
        elbos = list(elbos)

    print(f"  ✓ ELBO={elbos[-1] if len(elbos) > 0 else 'N/A':.1f}")
    print(f"  ✓ z:{z_map.shape}, x:{x_map.shape}")

    return rslds, z_map.astype(int), x_map, elbos


def _parse_fit_results(results, num_iters):
    """解析fit返回的结果"""
    if isinstance(results, tuple):
        if len(results) == 2:
            return results[0], np.asarray(results[1])
        else:
            return results[0], np.full(num_iters, np.nan)
    else:
        q = results
        if hasattr(q, 'elbos'):
            elbos = np.asarray(q.elbos)
        else:
            elbos = np.full(num_iters, np.nan)
        return q, elbos


def _fit_hmm_fallback(Y, downsample=15, K=2):
    """HMM兜底方案"""
    Y_ds = Y[::downsample]
    T, N = Y_ds.shape
    print(f"  HMM: T={T}, N={N}, K={K}")

    hmm = ssm.HMM(K=K, D=N, observations="gaussian")
    hmm.fit(Y_ds, method="em", num_em_iters=50)

    z_map = hmm.most_likely_states(Y_ds)

    pca = PCA(n_components=2)
    x_map = pca.fit_transform(Y_ds)

    lls = [np.nan] * 25

    print(f"  ✓ HMM fallback complete")
    return hmm, z_map, x_map, lls, K, 2


# ======================= 诊断和可视化（保持不变）=======================
def diagnose_attractor(rslds, x_map, z_map, K, D):
    """线吸引子诊断"""
    print("\n" + "=" * 60)
    print("ATTRACTOR DIAGNOSIS")
    print("=" * 60)

    if not hasattr(rslds, 'dynamics') or not hasattr(rslds.dynamics, 'As'):
        print("⚠️  模型无动力学参数（可能是HMM）")
        return None

    attractor_results = {}

    for k in range(K):
        print(f"\nState {k + 1}:")

        A_k = rslds.dynamics.As[k]
        eigenvalues, eigenvectors = np.linalg.eig(A_k)

        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        lambdas = np.abs(eigenvalues)
        print(f"  特征值: {lambdas}")

        lambda_1 = lambdas[0]
        lambda_2 = lambdas[1] if len(lambdas) > 1 else 0

        if 0.9 <= lambda_1 <= 1.0 and lambda_2 < 0.7:
            attractor_type = "线吸引子"
            icon = "✅"
        elif all(l < 0.5 for l in lambdas):
            attractor_type = "点吸引子（收缩）"
            icon = "❌"
        else:
            attractor_type = "不确定"
            icon = "⚠️"

        print(f"  {icon} 类型: {attractor_type}")

        attractor_results[k] = {
            'type': attractor_type,
            'eigenvalues': lambdas,
            'main_eigenvector': eigenvectors[:, 0].real
        }

    pca = PCA()
    pca.fit(x_map)
    var_ratios = pca.explained_variance_ratio_

    print(f"\n轨迹PCA分析:")
    for i, var in enumerate(var_ratios[:min(5, len(var_ratios))]):
        print(f"  成分{i + 1}: {var:.3f}")

    if var_ratios[0] > 0.95:
        print("  ⚠️  第1成分解释>95%方差 → 可能是降维假象")

    return attractor_results


def plot_temporal_dynamics(x_map, z_map, K, D, t_ds, reunion_abs, out_dir):
    print("\n生成时间动力学图...")

    t_rel = t_ds - reunion_abs
    colors = plt.cm.Set2(np.linspace(0, 1, K))
    n_dims = min(D, x_map.shape[1])

    fig = plt.figure(figsize=(18, 10))

    for dim in range(n_dims):
        ax = fig.add_subplot(3, 3, dim + 1)
        for k in range(K):
            mask = z_map == k
            if mask.sum() > 0:
                ax.scatter(t_rel[mask], x_map[mask, dim],
                           c=[colors[k]], s=3, alpha=0.6, label=f'S{k + 1}' if dim == 0 else '')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Latent {dim + 1}')
        ax.set_title(f'Latent {dim + 1}', fontweight='bold')
        ax.grid(alpha=0.3)
        if dim == 0:
            ax.legend(fontsize=8)

    ax = fig.add_subplot(3, 3, 4)
    velocity = np.linalg.norm(np.diff(x_map, axis=0), axis=1)
    t_vel = t_rel[:-1]
    for k in range(K):
        mask = z_map[:-1] == k
        if mask.sum() > 0:
            ax.scatter(t_vel[mask], velocity[mask],
                       c=[colors[k]], s=5, alpha=0.6, label=f'S{k + 1}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed')
    ax.set_title('Speed', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(3, 3, 5)
    cumsum = np.concatenate([[0], np.cumsum(velocity)])
    ax.plot(t_rel, cumsum, linewidth=2)
    ax.fill_between(t_rel, 0, cumsum, alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cumulative Distance')
    ax.set_title('Path Length', fontweight='bold')
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(3, 3, 6)
    for k in range(K):
        mask = z_map == k
        if mask.sum() > 0:
            ax.scatter(t_rel[mask], np.full(mask.sum(), k + 1),
                       c=[colors[k]], s=10, alpha=0.7, label=f'S{k + 1}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('State')
    ax.set_title('State Sequence', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    if n_dims >= 2:
        pairs = [(0, 1)] + ([(0, 2), (1, 2)] if n_dims >= 3 else [])
        for idx, (d1, d2) in enumerate(pairs[:3]):
            ax = fig.add_subplot(3, 3, 7 + idx)
            for k in range(K):
                mask = z_map == k
                if mask.sum() > 0:
                    ax.scatter(x_map[mask, d1], x_map[mask, d2],
                               c=[colors[k]], s=8, alpha=0.5, label=f'S{k + 1}')
            ax.set_xlabel(f'L{d1 + 1}')
            ax.set_ylabel(f'L{d2 + 1}')
            ax.set_title(f'L{d1 + 1} vs L{d2 + 1}', fontweight='bold')
            ax.grid(alpha=0.3)
            ax.set_aspect('equal')
            if idx == 0:
                ax.legend(fontsize=8)

    plt.suptitle('Temporal Dynamics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'temporal_dynamics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: temporal_dynamics.png")


def plot_latent_space(x_map, z_map, K, D, out_dir):
    n_dims = min(D, x_map.shape[1])
    colors = plt.cm.Set2(np.linspace(0, 1, K))

    if n_dims >= 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for k in range(K):
            mask = z_map == k
            if mask.sum() > 0:
                ax.scatter(x_map[mask, 0], x_map[mask, 1], x_map[mask, 2],
                           c=[colors[k]], s=5, alpha=0.5, label=f'S{k + 1}')
        ax.set_xlabel('L1')
        ax.set_ylabel('L2')
        ax.set_zlabel('L3')
        ax.legend()
        ax.set_title('3D Latent Space', fontweight='bold')
    else:
        fig, ax = plt.subplots(figsize=(8, 8))
        for k in range(K):
            mask = z_map == k
            if mask.sum() > 0:
                ax.scatter(x_map[mask, 0], x_map[mask, 1],
                           c=[colors[k]], s=10, alpha=0.6, label=f'S{k + 1}')
        ax.set_xlabel('L1')
        ax.set_ylabel('L2')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_aspect('equal')
        ax.set_title('2D Latent Space', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'latent_space.png'), dpi=300)
    plt.close()
    print("✓ Saved: latent_space.png")


# ======================= 主流程 =======================
def main():
    print("\n" + "=" * 70)
    print("rSLDS ANALYSIS: REUNION+CONTACT EPOCH")
    print("=" * 70)

    Y, t, dt = load_neural_data(data_path)
    social_intervals, _ = load_behavior(beh_path, reunion_abs)
    Y_epoch, t_epoch, mask_epoch = extract_reunion_contact_epoch(
        Y, t, reunion_abs, social_intervals
    )

    out_dir = os.path.join(base_out_dir, "rslds_reunion_contact")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n✓ Output: {out_dir}")

    rslds, z_map, x_map, lls, K_final, D_final = fit_rslds_with_fallback(
        Y_epoch, downsample=downsample, K_try=K_TRY, D_try=D_TRY, num_iters=num_iters
    )

    print(f"\n✅ Final Model: K={K_final}, D={D_final}")

    t_ds = t_epoch[::downsample][:len(z_map)]

    attractor_info = diagnose_attractor(rslds, x_map, z_map, K_final, D_final)

    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    plot_latent_space(x_map, z_map, K_final, D_final, out_dir)
    plot_temporal_dynamics(x_map, z_map, K_final, D_final, t_ds, reunion_abs, out_dir)

    np.savez(
        os.path.join(out_dir, 'results.npz'),
        z_map=z_map,
        x_map=x_map,
        t_ds=t_ds,
        K_final=K_final,
        D_final=D_final,
        attractor_info=attractor_info
    )

    print(f"\n✅ Analysis Complete!")
    print(f"✓ Results saved in: {out_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
