# mouse_switching_ramping_analysis.py
# -*- coding: utf-8 -*-
"""
基于：
Maternal motivation overcomes innate fear via prefrontal switching dynamics

流程：
1) 载入 miniscope 神经数据 & 行为时间戳
2) (可选) 钙信号 → spike inference（OASIS）
3) 用“社交接触 vs 非社交”构造高/低动机标签，做 GLM2 得到动机维度 β_motor
4) 把神经活动投影到动机维度，得到 1D 动机时间序列 x_t
5) 在 x_t 上拟合 2-state AR-HMM (HMM-AR)，得到状态 1/2 的 (a, b)，以及转移矩阵
6) 把 (a, b) 变成时间常数 τ 和渐近值 M，计算每个状态的停留概率 p_stay
7) 画图 & 保存结果
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture

from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore

import ssm  # 需要 pip install ssm

# 如果你之后想用 OASIS 去卷积，把这个改成 True，并安装 oasis
USE_OASIS = False
if USE_OASIS:
    try:
        from oasis.functions import deconvolve
    except ImportError:
        print("⚠️ 未找到 OASIS，自动关闭 USE_OASIS")
        USE_OASIS = False

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ======================= 参数设置 =======================
base_out_dir = r"F:\工作文件\RA\python\吸引子\figs"
os.makedirs(base_out_dir, exist_ok=True)

data_path = r"F:\工作文件\RA\python\吸引子\Mouse1_for_ssm.npz"
beh_path = r"F:\工作文件\RA\数据集\时间戳\FVB_Panneuronal_Reunion_Isolation_Day3_Mouse#1.xlsx"

# Reunion 在绝对时间轴上的时间（秒），和你现有脚本保持一致
reunion_abs = 903.0

# 时间窗（GLM2 的样本）长度（秒）
BIN_SIZE = 1.0

# HMM-AR 相关超参数
ARHMM_K = 3           # 两个状态：non-ramping / ramping
ARHMM_ITERS = 200
DOWNSAMPLE_FOR_HMM = 1  # 在 x_t 上做 HMM 的下采样步长（>1 表示稀疏取点）


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

    print(f"✓ T={len(t)}, N={Y.shape[1]}, dt={dt:.3f}s")
    return Y_z, t, dt


def load_behavior(beh_path, reunion_abs):
    """
    从 Excel 里读出 “重聚期开始”、“社交开始/结束” 时刻。
    返回：
      - social_intervals: 列表 [(start_abs, end_abs), ...]（绝对时间）
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
      1 = 处于社交接触区间（高动机）
      0 = 不在社交区间（低动机）
    """
    social_flag = np.zeros_like(t, dtype=int)
    for (start, end) in social_intervals:
        social_flag[(t >= start) & (t <= end)] = 1
    return social_flag


def extract_reunion_epoch(Y, t, reunion_abs, social_intervals, extra_after=60.0):
    """
    按论文思路：取重聚后的整段（直到最后一次社交结束 + 一点 buffer），
    保留 low/high motivation 都在内。
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


# ======================= (可选) 钙 → spike inference =======================
def calcium_to_spikes(Y):
    """
    用 OASIS 做钙去卷积，返回 z-scored spike-like 活动。
    如果 USE_OASIS=False，就直接把 Y 当作已经是活动。
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
        c, s, *_ = deconvolve(F_trace, penalty=1)  # 这里只是示意，实际可以调参数
        spikes[:, i] = s
    spikes_z = zscore(spikes, axis=0)
    print("✓ OASIS complete")
    return spikes_z


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
    这里只放神经，不额外加条件变量，方便先跑。
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


def project_to_motivation_dim(spikes_epoch, beta_motor):
    """
    把每一帧的神经活动投影到动机维度上，得到 1D 时间序列 x_t
    """
    print("\n" + "=" * 60)
    print("PROJECTING TO MOTIVATION DIMENSION")
    print("=" * 60)

    x_t = spikes_epoch @ beta_motor  # (T_epoch,)
    x_t = (x_t - x_t.mean()) / (x_t.std() + 1e-6)
    print(f"✓ x_t shape: {x_t.shape}, mean≈0, std≈1")
    return x_t

# ======================= K 选择：对不同 K 的 AR-HMM 做模型比较 =======================
def fit_arhmm_1d(x, K, num_em_iters=100, seed=0):
    """
    在一维序列 x 上拟合 K 状态 AR-HMM（不下采样）。
    用于模型选择（交叉验证 & BIC）。
    """
    np.random.seed(seed)
    Y = x[:, None]  # (T, 1)
    model = ssm.HMM(
        K=K,
        D=1,
        observations="ar",
        observation_kwargs={"lags": 1}
    )
    model.fit([Y], method="em", num_em_iters=num_em_iters)
    return model


def blocked_folds(T, n_folds=4):
    """
    把时间序列 [0..T-1] 划分成 n_folds 个连续块，返回 (train_idx, test_idx) 列表。
    注意：这里 train_idx 是把所有非测试块拼在一起（有间断），
    但我们简单地当成一条长序列来拟合，作为近似。
    """
    idx = np.arange(T)
    fold_sizes = np.full(n_folds, T // n_folds, dtype=int)
    fold_sizes[:T % n_folds] += 1

    folds = []
    start = 0
    for fs in fold_sizes:
        test = idx[start:start + fs]
        train = np.setdiff1d(idx, test)
        folds.append((train, test))
        start += fs
    return folds


def approx_num_params_arhmm_1d(K):
    """
    粗略估计 1D AR-HMM 的参数个数：
      - 初始分布: K-1
      - 转移矩阵: K*(K-1)
      - 每个状态的 AR(1) + 方差: (a, b, log_sigma2) = 3K
      总计: K^2 + 2K - 1
    """
    return (K - 1) + K * (K - 1) + 3 * K  # = K^2 + 2K - 1


def select_K_by_cv_bic(x, K_list=(1, 2, 3, 4), n_folds=4,
                       restarts=1, num_em_iters=50):
    """
    对给定的 K_list 中每个 K：
      - 做 blocked cross-validation，计算平均 CV-NLL
      - 在全数据上再拟合一次，计算 BIC
    返回 [(K, cv_nll, bic), ...]
    """
    T = len(x)
    folds = blocked_folds(T, n_folds=n_folds)
    results = []

    for K in K_list:
        cv_nlls = []

        for r in range(restarts):
            for train_idx, test_idx in folds:
                x_train = x[train_idx]
                x_test = x[test_idx]

                model = fit_arhmm_1d(
                    x_train, K,
                    num_em_iters=num_em_iters,
                    seed=1000 * K + 10 * r
                )

                # 用训练好的模型评估测试段对数似然（按长度归一化）
                ll_test = model.log_probability([x_test[:, None]])
                cv_nlls.append(-ll_test / len(x_test))

        cv_nll = float(np.mean(cv_nlls))

        # 在全数据上再拟合一次，用于计算 BIC
        full_model = fit_arhmm_1d(
            x, K,
            num_em_iters=num_em_iters,
            seed=9999 + K
        )
        full_ll = full_model.log_probability([x[:, None]])
        p = approx_num_params_arhmm_1d(K)
        bic = -2 * full_ll + p * np.log(T)

        results.append((K, cv_nll, float(bic)))

    return results


def run_K_sweep_and_report(x_t, out_dir,
                           K_list=(1, 2, 3, 4),
                           n_folds=4, restarts=1, num_em_iters=50):
    """
    对若干 K 做模型比较，并打印 / 保存结果。
    返回建议使用的 K*（按 CV-NLL 最小优先，其次 BIC）。
    """
    print("\n" + "=" * 60)
    print("MODEL SELECTION: AR-HMM NUMBER OF STATES (K)")
    print("=" * 60)

    results = select_K_by_cv_bic(
        x_t,
        K_list=K_list,
        n_folds=n_folds,
        restarts=restarts,
        num_em_iters=num_em_iters
    )

    print("\nK\tCV-NLL\t\tBIC")
    for K, cv_nll, bic in results:
        print(f"{K}\t{cv_nll:.3f}\t{bic:.1f}")

    # 按 CV-NLL 最小优先，其次 BIC 最小
    best = min(results, key=lambda r: (r[1], r[2]))
    best_K = best[0]

    # 保存到 npz，方便之后画图或复查
    Ks = np.array([r[0] for r in results])
    cv_nlls = np.array([r[1] for r in results])
    bics = np.array([r[2] for r in results])

    np.savez(
        os.path.join(out_dir, "K_sweep_results.npz"),
        K=Ks,
        cv_nll=cv_nlls,
        bic=bics,
        best_K=best_K
    )

    print(f"\n→ Suggested number of states K* = {best_K}")
    return best_K

# ======================= 在 x_t 上拟合 HMM-AR =======================
def fit_arhmm_on_motivation(x_t, K=2, num_em_iters=200, downsample=1):
    """
    在 1D 动机时间序列 x_t 上拟合 K 状态 AR-HMM（HMM-AR）。
    使用 ssm.HMM(observations='ar')。
    返回：
      arhmm, z_map, a (K,), b (K,), trans_mat (K,K), idx_ds（下采样后对应的原始索引）
    """
    print("\n" + "=" * 60)
    print("FITTING HMM-AR ON MOTIVATION SIGNAL")
    print("=" * 60)

    # 下采样
    x_ds = x_t[::downsample]
    T = len(x_ds)
    Y = x_ds[:, None]  # (T, 1)

    print(f"Input for AR-HMM: T={T}, downsample={downsample}")

    D = 1
    # 指定 lags=1 保证是 AR(1)
    arhmm = ssm.HMM(
        K=K,
        D=D,
        observations="ar",
        observation_kwargs={"lags": 1}
    )

    # 拟合
    arhmm.fit([Y], method="em", num_em_iters=num_em_iters)
    z_map = arhmm.most_likely_states(Y)

    # ===== 取出 AR 参数 a, b =====
    obs = arhmm.observations
    As = obs.As  # 形状依版本可能是 (K, D, D*lags) / (K, D, D) / (K, D, D, lags)
    bs = obs.bs  # 通常 (K, D)

    if As.ndim == 3:
        # (K, D, D*lags) 或 (K, D, D)
        a = As[:, 0, 0]
    elif As.ndim == 4:
        # (K, D, D, lags)
        a = As[:, 0, 0, 0]
    else:
        raise ValueError(f"Unexpected As shape: {As.shape}")

    b = bs[:, 0]

    # ===== 取出转移矩阵 P (K,K) =====
    trans_obj = arhmm.transitions

    log_P = None

    # 1) 优先尝试 log_Ps 属性（很多 py-ssm 版本都有）
    if hasattr(trans_obj, "log_Ps"):
        log_P_raw = np.asarray(trans_obj.log_Ps)
        if log_P_raw.ndim == 2:
            # (K, K)
            log_P = log_P_raw
        elif log_P_raw.ndim == 3:
            # (1, K, K) 或 (T, K, K) → 时间/批次平均
            log_P = log_P_raw.mean(axis=0)
        else:
            raise ValueError(f"Unexpected log_Ps shape: {log_P_raw.shape}")

    # 2) 其次尝试 transition_matrix 属性（有些版本有）
    elif hasattr(trans_obj, "transition_matrix"):
        P = np.asarray(trans_obj.transition_matrix)
        if P.ndim == 2:
            log_P = np.log(P + 1e-16)
        elif P.ndim == 3:
            log_P = np.log(P.mean(axis=0) + 1e-16)
        else:
            raise ValueError(f"Unexpected transition_matrix shape: {P.shape}")

    # 3) 最后才兜底调用 log_transition_matrices(…, input, mask, tag)
    elif hasattr(trans_obj, "log_transition_matrices"):
        try:
            # 你的报错显示这个函数要求三个额外参数 input/mask/tag
            # 对于 StationaryTransitions 它通常不会用到这些值，所以用 None 占位即可
            log_P_raw = trans_obj.log_transition_matrices(
                None,  # input
                None,  # mask
                None   # tag
            )
            log_P_raw = np.asarray(log_P_raw)

            if log_P_raw.ndim == 2:
                log_P = log_P_raw
            elif log_P_raw.ndim == 3:
                log_P = log_P_raw.mean(axis=0)
            elif log_P_raw.ndim == 4:
                log_P = log_P_raw.mean(axis=(0, 1))
            else:
                raise ValueError(f"Unexpected log_transition_matrices shape: {log_P_raw.shape}")
        except TypeError as e:
            raise TypeError(
                "log_transition_matrices exists but signature不兼容，"
                "请检查 ssm 版本或改用 log_Ps / transition_matrix。"
            ) from e
    else:
        raise AttributeError(
            "Cannot find transition matrix: no log_Ps, "
            "transition_matrix or log_transition_matrices on arhmm.transitions."
        )

    trans_mat = np.exp(log_P)

    print(f"✓ Fitted AR-HMM: K={K}")
    print(f"  a: {a}")
    print(f"  b: {b}")
    print(f"  trans_mat:\n{trans_mat}")

    # idx_ds: 下采样后的每个点对应原始 x_t 的索引
    idx_ds = np.arange(len(x_t))[::downsample]

    return arhmm, z_map, a, b, trans_mat, idx_ds



def compute_ramping_params(a, b):
    """
    根据论文，把 AR(1) 的 a,b 转成：
      τ = 1 / (1 - a)
      M = b / (1 - a)
    """
    tau = 1.0 / (1.0 - a)
    M = b / (1.0 - a)
    return tau, M


# ======================= 可视化 =======================
def plot_motivation_and_states(x_t, t_epoch, z_map, idx_ds, reunion_abs, out_dir):
    """
    画：
      - x_t 随时间的轨迹，并用不同颜色标出 HMM-AR 的状态
      - 状态序列随时间
    """
    print("\n生成动机维度时间序列 + 状态图...")

    os.makedirs(out_dir, exist_ok=True)
    t_rel = t_epoch - reunion_abs

    # 下采样后的时间轴
    t_ds = t_epoch[idx_ds]
    t_ds_rel = t_ds - reunion_abs

    K = len(np.unique(z_map))
    colors = plt.cm.Set2(np.linspace(0, 1, K))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax = axes[0]
    ax.plot(t_rel, x_t, color="gray", alpha=0.6, label="Motivation (x_t)")

    for k in range(K):
        mask = z_map == k
        if mask.sum() == 0:
            continue
        ax.scatter(t_ds_rel[mask], x_t[idx_ds][mask],
                   c=[colors[k]], s=10, alpha=0.8, label=f"State {k+1}")
    ax.set_ylabel("Motivation (z-score)")
    ax.set_title("Motivation dimension with HMM-AR states")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for k in range(K):
        mask = z_map == k
        if mask.sum() == 0:
            continue
        ax.scatter(t_ds_rel[mask], np.full(mask.sum(), k+1),
                   c=[colors[k]], s=10, alpha=0.8, label=f"State {k+1}")
    ax.set_xlabel("Time relative to reunion (s)")
    ax.set_ylabel("State")
    ax.set_title("Discrete state sequence (HMM-AR)")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_file = os.path.join(out_dir, "motivation_hmmar_timeseries.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"✓ Saved: {out_file}")


def plot_state_histograms(z_map, a, b, tau, M, trans_mat, out_dir):
    """
    一些简单的直方图/打印，看看状态性质：
      - 状态持续时间分布
      - 每个状态的 τ, M, p_stay
    """
    print("生成状态统计图...")
    os.makedirs(out_dir, exist_ok=True)

    K = len(np.unique(z_map))
    colors = plt.cm.Set2(np.linspace(0, 1, K))

    # 状态持续时间分布
    durations = {k: [] for k in range(K)}
    current_state = z_map[0]
    run_len = 1
    for s in z_map[1:]:
        if s == current_state:
            run_len += 1
        else:
            durations[current_state].append(run_len)
            current_state = s
            run_len = 1
    durations[current_state].append(run_len)

    fig, axes = plt.subplots(1, K, figsize=(5 * K, 4))
    if K == 1:
        axes = [axes]

    for k in range(K):
        ax = axes[k]
        durs = np.array(durations[k])
        ax.hist(durs, bins=20, color=colors[k], alpha=0.7)
        ax.set_xlabel("Duration (steps)")
        ax.set_ylabel("Count")
        ax.set_title(f"State {k+1} duration")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out_file = os.path.join(out_dir, "state_durations.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"✓ Saved: {out_file}")

    # 打印参数
    p_stay = np.diag(trans_mat)
    for k in range(K):
        print(f"\nState {k+1}:")
        print(f"  a = {a[k]:.3f}, b = {b[k]:.3f}")
        print(f"  tau = {tau[k]:.3f}, M = {M[k]:.3f}")
        print(f"  p_stay = {p_stay[k]:.3f}")

    # 保存成 npz
    np.savez(
        os.path.join(out_dir, "hmmar_params.npz"),
        a=a,
        b=b,
        tau=tau,
        M=M,
        trans_mat=trans_mat,
        p_stay=p_stay,
        z_map=z_map
    )
    print("✓ Saved: hmmar_params.npz")


# ======================= 主流程 =======================
def main():
    print("\n" + "=" * 70)
    print("SWITCHING RAMPING ANALYSIS (HMM-AR on MOTIVATION DIMENSION)")
    print("=" * 70)

    # 1. 载入神经数据 & 行为时间戳
    Y, t, dt = load_neural_data(data_path)
    social_intervals, _ = load_behavior(beh_path, reunion_abs)

    # 2. (可选) 钙信号 → spike inference
    spikes = calcium_to_spikes(Y)   # (T, N)

    # 3. 构造社交标签 + 抽取重聚后 epoch
    social_flag_full = build_social_flag(t, social_intervals)
    Y_epoch, t_epoch, mask_epoch = extract_reunion_epoch(Y, t, reunion_abs, social_intervals)
    spikes_epoch = spikes[mask_epoch]
    social_flag_epoch = social_flag_full[mask_epoch]

    out_dir = os.path.join(base_out_dir, "switching_ramping")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n✓ Output dir: {out_dir}")

    # 4. 时间窗 binning + GLM2 -> 动机维度 β_motor
    X_bins, y_bins, t_bins = bin_data(spikes_epoch, t_epoch, social_flag_epoch, bin_size=BIN_SIZE)
    beta_motor, glm_intercept = fit_glm2_and_get_motivation_axis(X_bins, y_bins)

    # 5. 把每一帧的 spike 活动投到动机维度 -> x_t
    x_t = project_to_motivation_dim(spikes_epoch, beta_motor)

    # 5.5 对不同 K 做模型比较（在 1D x_t 上），选一个最合适的 K
    # 例如在 K = 1,2,3,4 中选
    best_K = run_K_sweep_and_report(
        x_t,
        out_dir,
        K_list=(1, 2, 3, 4),
        n_folds=4,
        restarts=1,
        num_em_iters=50  # 为了节省时间，CV 时迭代可以少一些
    )

    # 你可以选择用 best_K，或者强制用 ARHMM_K=2。
    # 下面我用 best_K 作为最终 HMM-AR 的状态数：
    K_final = best_K

    # 6. 在 x_t 上拟合 K_final-state HMM-AR（用于后续分析和画图）
    arhmm, z_map, a, b, trans_mat, idx_ds = fit_arhmm_on_motivation(
        x_t,
        K=K_final,
        num_em_iters=ARHMM_ITERS,
        downsample=DOWNSAMPLE_FOR_HMM
    )

    # 7. 把 a, b 换成 τ, M，并简单可视化
    tau, M = compute_ramping_params(a, b)

    print("\n" + "=" * 70)
    print("GENERATING PLOTS & SAVING RESULTS")
    print("=" * 70)

    plot_motivation_and_states(x_t, t_epoch, z_map, idx_ds, reunion_abs, out_dir)
    plot_state_histograms(z_map, a, b, tau, M, trans_mat, out_dir)

    # 额外保存一些关键结果
    np.savez(
        os.path.join(out_dir, "motivation_results.npz"),
        x_t=x_t,
        t_epoch=t_epoch,
        beta_motor=beta_motor,
        glm_intercept=glm_intercept,
        z_map=z_map,
        a=a,
        b=b,
        tau=tau,
        M=M,
        trans_mat=trans_mat,
        idx_ds=idx_ds
    )

    print("\n✅ Analysis Complete!")
    print(f"✓ Results saved in: {out_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
