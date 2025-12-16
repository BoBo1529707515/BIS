# 社交需求_一维HMM_AR动态.py
# -*- coding: utf-8 -*-
"""
基于 Need_z(t) 的一维动力学建模：
- 高斯(iid)模型
- AR(1) 模型
- 二元 GMM
- 二元 Gaussian HMM
- HMM 状态内的 AR(1)（近似 HMM-AR 思路）

依赖:
    pip install numpy matplotlib scikit-learn hmmlearn
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from hmmlearn.hmm import GaussianHMM

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ======================= 路径配置 =======================
base_out_dir = r"F:\工作文件\RA\python\吸引子\figs"

# STEP1 生成的社交需求时间序列 npz 文件
need_npz_dir = os.path.join(base_out_dir, "social_need_from_three_state")
need_npz_file = os.path.join(
    need_npz_dir,
    "social_need_timeseries_from_three_state_glm.npz"  # 确保和 STEP1 脚本里一致
)

# 本脚本输出目录：专门放“一维 HMM/AR 动力学”结果
dyn_out_dir = os.path.join(base_out_dir, "social_need_dynamics_HMM_AR")
os.makedirs(dyn_out_dir, exist_ok=True)

# 最大自相关滞后时间（秒）
ACF_MAX_LAG_SEC = 10.0


# ======================= 工具函数 =======================
def load_need_npz(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 Need(t) 文件，请先运行 STEP1 脚本生成:\n  {path}")

    dat = np.load(path, allow_pickle=True)
    need_z = np.asarray(dat["need_z"])           # (T,)
    t_epoch = np.asarray(dat["t_epoch"])         # (T,)
    labels_epoch = np.asarray(dat["labels_epoch"]).astype(int)  # (T,)
    dt = float(dat["dt"].item())
    reunion_abs = float(dat["reunion_abs"].item())
    epoch_start = float(dat["epoch_start"].item())
    epoch_end = float(dat["epoch_end"].item())
    return need_z, t_epoch, labels_epoch, dt, reunion_abs, epoch_start, epoch_end


def compute_gaussian_iid_loglik(x):
    """ 简单高斯独立模型：估计整体 mean/std 后，计算 log-likelihood """
    mu = x.mean()
    var = x.var()
    if var <= 1e-12:
        var = 1e-12
    n = len(x)
    loglik = -0.5 * n * (np.log(2 * np.pi * var)) - 0.5 * ((x - mu) ** 2 / var).sum()
    k = 2  # 参数数：mu, var
    return loglik, k, mu, var


def compute_ar1_loglik(x):
    """
    估计一阶 AR 模型：x_t = a * x_{t-1} + eps
    用最小二乘估 a，估残差方差，再算 log-likelihood
    """
    x_prev = x[:-1]
    x_next = x[1:]
    # 最小二乘估计 a
    denom = (x_prev ** 2).sum()
    if denom <= 1e-12:
        a = 0.0
    else:
        a = (x_prev * x_next).sum() / denom

    # 残差
    eps = x_next - a * x_prev
    var_eps = eps.var()
    if var_eps <= 1e-12:
        var_eps = 1e-12

    n = len(x_next)
    # 条件高斯 likelihood：p(x_1) 用边缘高斯（这里忽略第一项，近似）
    loglik = -0.5 * n * np.log(2 * np.pi * var_eps) - 0.5 * (eps ** 2 / var_eps).sum()
    k = 2  # 参数数：a, var_eps
    return loglik, k, a, var_eps


def compute_bic_aic(loglik, k, n):
    """ AIC / BIC """
    aic = 2 * k - 2 * loglik
    bic = k * np.log(n) - 2 * loglik
    return aic, bic


def fit_gmm_2states(x):
    """
    在 Need_z 上拟合 2-state GMM。
    x: (T,)
    """
    X = x.reshape(-1, 1)
    gmm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        random_state=0,
        n_init=10
    )
    gmm.fit(X)
    loglik = gmm.score(X) * len(x)  # score 是平均 log likelihood
    # 参数数估算：每个成分 1 个均值 + 1 个方差 + 1 个混合权（但总和=1，少一个自由度）
    k = 2 * 2 + (2 - 1)  # 2*(mu,var) + 1 mixing weight
    return gmm, loglik, k


def fit_hmm_gaussian_2states(x):
    X = x.reshape(-1, 1)
    hmm = GaussianHMM(
        n_components=2,
        covariance_type="diag",
        n_iter=500,
        random_state=0
    )
    hmm.fit(X)
    loglik = hmm.score(X)        # ★ 不要再乘 len(x) 了

    # 参数数估计：
    # 初始分布: 2-1 自由度 = 1
    # 转移矩阵: 2*2 每行少 1，自由度=2
    # 每个状态: 均值1 + 方差1 = 2，合计4
    k = 1 + 2 + 4  # =7
    z = hmm.predict(X)
    return hmm, z, loglik, k



def compute_statewise_ar1(x, z, dt):
    """
    在 HMM 状态上估计每个状态各自的 AR(1) 参数：
        x_t = a_s * x_{t-1} + eps
    仅使用连续处于同一状态的 (t-1, t) 对。
    返回:
        dict[state_idx] = { 'a':..., 'var_eps':..., 'n_pairs':..., 'tau':... }
    """
    res = {}
    states = np.unique(z)
    for s in states:
        # 找所有 t, 满足 z_t == s 且 z_{t-1} == s
        idx = np.where((z[1:] == s) & (z[:-1] == s))[0] + 1  # t 的索引
        if len(idx) < 2:
            res[int(s)] = {
                "a": np.nan,
                "var_eps": np.nan,
                "n_pairs": len(idx),
                "tau": np.nan
            }
            continue

        x_prev = x[idx - 1]
        x_next = x[idx]
        denom = (x_prev ** 2).sum()
        if denom <= 1e-12:
            a_s = 0.0
        else:
            a_s = (x_prev * x_next).sum() / denom

        eps = x_next - a_s * x_prev
        var_eps = eps.var()
        if var_eps <= 1e-12:
            var_eps = 1e-12

        # 如果 0 < a_s < 1，可以定义一个类似 time constant:
        # tau = -dt / ln(a_s)
        if 0 < a_s < 1:
            tau = -dt / np.log(a_s)
        else:
            tau = np.nan

        res[int(s)] = {
            "a": a_s,
            "var_eps": var_eps,
            "n_pairs": len(idx),
            "tau": tau
        }
    return res


def autocorr(x, max_lag):
    """ 计算从 lag=0..max_lag 的自相关（正规化为 1 at lag 0） """
    x = x - x.mean()
    n = len(x)
    var = (x ** 2).sum()
    if var <= 1e-12:
        return np.zeros(max_lag + 1)
    ac = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag == 0:
            ac[lag] = 1.0
        else:
            ac[lag] = (x[:-lag] * x[lag:]).sum() / var
    return ac


# ======================= 画图函数 =======================
def plot_need_with_hmm_states(need_z, t_epoch, reunion_abs, z, out_dir):
    t_rel = t_epoch - reunion_abs

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    # 用颜色区分 HMM 状态
    # 为了简单，用两条颜色不同的线
    for s, color, label in zip([0, 1], ["tab:blue", "tab:orange"], ["HMM state 0", "HMM state 1"]):
        mask = (z == s)
        ax.plot(t_rel[mask], need_z[mask], ".", markersize=1.5, alpha=0.7, label=label)

    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.set_xlabel("相对重聚时间 (s)")
    ax.set_ylabel("Need_z(t)")
    ax.set_title("Need_z(t) 及其 HMM 高斯隐状态分段")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(out_dir, "HMM_need_timecourse_states.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"✓ Saved: {out_path}")
    return out_path


def plot_hmm_state_hist(need_z, z, out_dir):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    bins = np.linspace(need_z.min(), need_z.max(), 40)
    for s, color, label in zip([0, 1], ["tab:blue", "tab:orange"], ["state 0", "state 1"]):
        ax.hist(need_z[z == s], bins=bins, alpha=0.5, density=True, label=label)

    ax.set_xlabel("Need_z")
    ax.set_ylabel("概率密度")
    ax.set_title("HMM 状态下的 Need_z 分布")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(out_dir, "HMM_state_need_hist.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"✓ Saved: {out_path}")
    return out_path


def plot_acf(need_z, dt, z, out_dir, max_lag_sec=ACF_MAX_LAG_SEC):
    max_lag = int(round(max_lag_sec / dt))
    lags = np.arange(max_lag + 1) * dt

    # 整体 ACF
    ac_all = autocorr(need_z, max_lag)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(lags, ac_all, label="整体 Need_z", linewidth=2)

    # 各 HMM 状态内 ACF（只做简单版本：不分段，仅对属于该状态的样本做全局 ACF）
    for s, color, label in zip([0, 1], ["tab:blue", "tab:orange"],
                               ["state 0 内", "state 1 内"]):
        x_s = need_z[z == s]
        if len(x_s) > max_lag + 2:
            ac_s = autocorr(x_s, max_lag)
            ax.plot(lags, ac_s, linestyle="--", label=label)
        else:
            print(f"⚠️ 状态 {s} 内样本太少，跳过 ACF 绘制。")

    ax.set_xlabel("滞后时间 (s)")
    ax.set_ylabel("自相关系数")
    ax.set_title("Need_z 的自相关 (整体 + HMM 状态内)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_path = os.path.join(out_dir, "Need_ACF_overall_and_states.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"✓ Saved: {out_path}")
    return out_path


def plot_ar1_coeffs(ar1_state_dict, out_dir):
    states = sorted(ar1_state_dict.keys())
    a_vals = [ar1_state_dict[s]["a"] for s in states]
    tau_vals = [ar1_state_dict[s]["tau"] for s in states]
    n_pairs = [ar1_state_dict[s]["n_pairs"] for s in states]

    x_pos = np.arange(len(states))

    fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))
    ax1.bar(x_pos, a_vals, tick_label=[f"state {s}" for s in states])
    ax1.set_ylabel("AR(1) 系数 a_s")
    ax1.set_title("HMM 每个隐状态内的 AR(1) 系数")
    for i, (a, tau, n) in enumerate(zip(a_vals, tau_vals, n_pairs)):
        text = f"a={a:.2f}\nn={n}"
        if not np.isnan(tau):
            text += f"\nτ≈{tau:.2f}s"
        ax1.text(i, a_vals[i] + 0.02, text, ha="center", va="bottom", fontsize=8)

    ax1.set_ylim(min(a_vals) - 0.1, max(a_vals) + 0.3)
    ax1.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(out_dir, "HMM_state_AR1_coeffs.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"✓ Saved: {out_path}")
    return out_path


# ======================= 写 README =======================
def write_readme(dyn_out_dir, metrics, time_fig, hist_fig, acf_fig, ar_fig):
    """
    metrics: dict 包含各模型 loglik/AIC/BIC/HMM 参数等
    """
    lines = []
    lines.append("# 基于 Need_z(t) 的一维 HMM / AR 动力学分析")
    lines.append("")
    lines.append("本文件夹由脚本 `社交需求_一维HMM_AR动态.py` 自动生成，用于在已经构造好的社交需求指数轨迹 Need_z(t) 上进行一维动力学建模。")
    lines.append("")
    lines.append("## 一、数据来源")
    lines.append("")
    lines.append("- Need_z(t) 来源于上一脚本 `社交需求指数_三态GLM.py`：")
    lines.append("  - 先用三状态 GLM 得到 β_state0, β_state1, β_state2。")
    lines.append("  - 定义“触摸 vs 无触摸”社交轴：w = β_state2 − β_state1。")
    lines.append("  - 对每一帧群体活动 spikes(t) 做投影并 z-score：Need_z(t) = zscore(w^T · spikes(t))。")
    lines.append("")
    lines.append("## 二、模型简介")
    lines.append("")
    lines.append("1. **高斯(iid)模型**")
    lines.append("   - 假设 Need_z(t) 独立同分布，来自一个 N(μ, σ²)。")
    lines.append("   - 用整体均值和方差拟合，并计算对数似然 / AIC / BIC。")
    lines.append("")
    lines.append("2. **AR(1) 模型**")
    lines.append("   - 假设 Need_z(t) 连续时间有记忆：x_t = a · x_{t-1} + eps。")
    lines.append("   - 用最小二乘估计 a 和残差方差，计算对数似然 / AIC / BIC。")
    lines.append("")
    lines.append("3. **二维 GMM 模型**")
    lines.append("   - 假设 Need_z(t) 的静态分布是两个高斯的混合（不考虑时间结构）。")
    lines.append("   - 拟合两个高斯成分及混合权重，计算整体 log-likelihood / AIC / BIC。")
    lines.append("")
    lines.append("4. **二维 Gaussian HMM 模型**")
    lines.append("   - 假设存在两个隐状态 z_t ∈ {0,1}，每个状态下 Need_z(t) 服从不同的高斯分布。")
    lines.append("   - 同时估计状态转移矩阵和各状态的均值/方差，得到最可能的状态序列 z_t。")
    lines.append("   - 这一步体现了“状态切换”的时间结构。")
    lines.append("")
    lines.append("5. **HMM 状态内 AR(1)（近似 HMM-AR）**")
    lines.append("   - 在 HMM 解码出的 z_t 上，分别在每个隐状态内估计 AR(1) 系数 a_s。")
    lines.append("   - 如果某个状态的 a_s 接近 0，表示该状态下 Need_z 变化更像白噪声；")
    lines.append("   - 如果 a_s 较大（接近 1），说明该状态下 Need_z 有较强的持续性 / ramping。")
    lines.append("")
    lines.append("## 三、模型比较结果（logL / AIC / BIC）")
    lines.append("")
    lines.append("以下为在完整 Need_z(t) 上拟合得到的指标（数值仅供该数据集内部比较）：")
    lines.append("")
    for name, info in metrics.items():
        lines.append(f"- **{name}**:")
        lines.append(f"  - log-likelihood = {info['loglik']:.2f}")
        lines.append(f"  - AIC = {info['AIC']:.2f}")
        lines.append(f"  - BIC = {info['BIC']:.2f}")
        if "extra" in info and info["extra"]:
            for key, val in info["extra"].items():
                lines.append(f"  - {key}: {val}")
        lines.append("")
    lines.append("")
    lines.append("一般来说，AIC/BIC 更小的模型说明对数据的解释更好（同时惩罚参数数量）。")
    lines.append("若 HMM 或 HMM+AR 状态内的 ACF 明显高于高斯/简单 AR 模型，则支持“存在隐状态切换 + ramping”的动力学结构。")
    lines.append("")
    lines.append("## 四、图像说明")
    lines.append("")
    lines.append(f"### 1）{os.path.basename(time_fig)}")
    lines.append("Need_z(t) 随时间变化，并用颜色区分 HMM-Gaussian 推断出的两个隐状态：")
    lines.append("- 蓝色点：隐状态 0；")
    lines.append("- 橙色点：隐状态 1；")
    lines.append("可以直观看出哪些时间段处于高/低需求状态，以及状态切换大致发生在何处。")
    lines.append("")
    lines.append(f"### 2）{os.path.basename(hist_fig)}")
    lines.append("HMM 每个隐状态下 Need_z 的静态分布：")
    lines.append("- 若两个直方图中心相差较大，说明两个状态在平均需求水平上明显不同（例如：低 vs 高 Need）；")
    lines.append("- 若一个状态的分布更宽，可能代表该状态内部的需求波动更大。")
    lines.append("")
    lines.append(f"### 3）{os.path.basename(acf_fig)}")
    lines.append("Need_z 的自相关函数（ACF）：")
    lines.append("- 粗线：整体 Need_z 的自相关；")
    lines.append("- 虚线：在各个 HMM 状态内分别计算的自相关（若样本足够）。")
    lines.append("- 若某个状态的 ACF 在较长滞后时间内仍然偏高，说明该状态具有较强的“记忆”或 ramping 特性。")
    lines.append("")
    lines.append(f"### 4）{os.path.basename(ar_fig)}")
    lines.append("HMM 每个隐状态内估计得到的 AR(1) 系数 a_s：")
    lines.append("- a_s 越接近 1，表示该状态下需求信号变化更为缓慢、具有更强的持续性；")
    lines.append("- a_s 接近 0，表示该状态下信号更接近白噪声。")
    lines.append("- 若一个状态的 a_s 高、另一个低，可对照行为标签，将其解释为“ramping 高需求状态” vs “非 ramping 低需求状态”。")
    lines.append("")
    lines.append("## 五、如何解释这些结果？")
    lines.append("")
    lines.append("如果在该数据中观察到：")
    lines.append("- HMM 模型相对于单一高斯或简单 AR 模型，具有更低的 AIC/BIC；")
    lines.append("- HMM 中一个状态的均值明显更高（或更低），且该状态更常出现在某些行为阶段（如接近触摸、触摸期）；")
    lines.append("- HMM 状态内的 AR(1) 系数显示：高需求状态 a_s 较大（更持续），低需求状态 a_s 较小；")
    lines.append("那么可以自然地将这两个 HMM 状态解释为：")
    lines.append("- **状态 0：低需求/非 ramping 状态**（例如，长时间无社交接触、或稳定的基线期）；")
    lines.append("- **状态 1：高需求/ramping 状态**（例如，接近社交触摸开始前后的阶段）。")
    lines.append("")
    lines.append("这与原论文中“动机维度上存在隐状态切换 + ramping 动力学”的结论是一致的，只是这里换成了基于三态 GLM 定义的社交需求指数 Need_z(t)。")
    lines.append("")

    readme_path = os.path.join(dyn_out_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"✓ Saved: {readme_path}")
    return readme_path


# ======================= 主函数 =======================
def main():
    print("\n" + "=" * 70)
    print("基于 Need_z(t) 的一维 HMM / AR 动力学分析")
    print("=" * 70)

    # 1) 载入 Need_z(t)
    need_z, t_epoch, labels_epoch, dt, reunion_abs, epoch_start, epoch_end = \
        load_need_npz(need_npz_file)
    T = len(need_z)
    print(f"✓ Loaded Need_z time series: T={T}, dt≈{dt:.3f}s")

    metrics = {}

    # 2) 高斯(iid) 模型
    loglik_g, k_g, mu_g, var_g = compute_gaussian_iid_loglik(need_z)
    aic_g, bic_g = compute_bic_aic(loglik_g, k_g, T)
    metrics["Gaussian(iid)"] = {
        "loglik": loglik_g,
        "AIC": aic_g,
        "BIC": bic_g,
        "extra": {
            "mu": f"{mu_g:.3f}",
            "var": f"{var_g:.3f}",
        }
    }
    print(f"[Gaussian] loglik={loglik_g:.2f}, AIC={aic_g:.2f}, BIC={bic_g:.2f}")

    # 3) AR(1) 模型
    loglik_ar, k_ar, a_ar, var_eps_ar = compute_ar1_loglik(need_z)
    aic_ar, bic_ar = compute_bic_aic(loglik_ar, k_ar, T)
    metrics["AR(1)"] = {
        "loglik": loglik_ar,
        "AIC": aic_ar,
        "BIC": bic_ar,
        "extra": {
            "a": f"{a_ar:.3f}",
            "var_eps": f"{var_eps_ar:.3f}",
        }
    }
    print(f"[AR(1)] loglik={loglik_ar:.2f}, AIC={aic_ar:.2f}, BIC={bic_ar:.2f}, a={a_ar:.3f}")

    # 4) 2-state GMM
    gmm, loglik_gmm, k_gmm = fit_gmm_2states(need_z)
    aic_gmm, bic_gmm = compute_bic_aic(loglik_gmm, k_gmm, T)
    metrics["GMM(2 comp)"] = {
        "loglik": loglik_gmm,
        "AIC": aic_gmm,
        "BIC": bic_gmm,
        "extra": {
            "means": ", ".join([f"{m[0]:.3f}" for m in gmm.means_]),
        }
    }
    print(f"[GMM-2] loglik={loglik_gmm:.2f}, AIC={aic_gmm:.2f}, BIC={bic_gmm:.2f}, "
          f"means={gmm.means_.ravel()}")

    # 5) 2-state Gaussian HMM
    hmm, z, loglik_hmm, k_hmm = fit_hmm_gaussian_2states(need_z)
    aic_hmm, bic_hmm = compute_bic_aic(loglik_hmm, k_hmm, T)
    metrics["HMM-Gaussian(2)"] = {
        "loglik": loglik_hmm,
        "AIC": aic_hmm,
        "BIC": bic_hmm,
        "extra": {
            "means": ", ".join([f"{m[0]:.3f}" for m in hmm.means_]),
            "transmat": np.array2string(hmm.transmat_, precision=3)
        }
    }
    print(f"[HMM-2] loglik={loglik_hmm:.2f}, AIC={aic_hmm:.2f}, BIC={bic_hmm:.2f}")
    print("        means:", hmm.means_.ravel())
    print("        transmat:\n", hmm.transmat_)

    # 6) 在 HMM 状态内估计 AR(1)（近似 HMM-AR）
    ar1_state_dict = compute_statewise_ar1(need_z, z, dt)
    for s, info in ar1_state_dict.items():
        print(f"[HMM state {s}] AR(1): a={info['a']:.3f}, var_eps={info['var_eps']:.3f}, "
              f"n_pairs={info['n_pairs']}, tau≈{info['tau']:.3f}s")

    # ☆ 新增：保存 HMM 结果，方便后续和行为做对齐分析
    hmm_npz_path = os.path.join(dyn_out_dir, "social_need_HMM_results.npz")
    np.savez(
        hmm_npz_path,
        need_z=need_z,
        t_epoch=t_epoch,
        labels_epoch=labels_epoch,
        dt=dt,
        reunion_abs=reunion_abs,
        z=z,
        hmm_means=hmm.means_,
        hmm_transmat=hmm.transmat_,
        ar1_state_info=ar1_state_dict
    )
    print(f"✓ Saved HMM results: {hmm_npz_path}")

    # 7) 画图
    time_fig = plot_need_with_hmm_states(need_z, t_epoch, reunion_abs, z, dyn_out_dir)
    hist_fig = plot_hmm_state_hist(need_z, z, dyn_out_dir)
    acf_fig = plot_acf(need_z, dt, z, dyn_out_dir)
    ar_fig = plot_ar1_coeffs(ar1_state_dict, dyn_out_dir)

    # 8) 写 README
    readme_path = write_readme(dyn_out_dir, metrics, time_fig, hist_fig, acf_fig, ar_fig)

    print("\n✅ Need_z 一维 HMM / AR 动力学分析完成。")
    print(f"  输出目录: {dyn_out_dir}")
    print(f"  已生成 README: {readme_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
