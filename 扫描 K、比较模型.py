# 社交需求_一维HMM_状态数扫描_模型比较.py
# -*- coding: utf-8 -*-
"""
在一维社交需求指数 Need_z(t) 上，对不同模型 / 不同状态数 K 做系统比较：

模型：
  1) 单一高斯 (Gaussian iid)
  2) 单状态 AR(1)
  3) GMM-K (K=1..K_MAX)
  4) HMM-Gaussian-K (K=1..K_MAX)

统一计算：
  - log-likelihood
  - AIC
  - BIC

用于回答：
  - Need_z 是否需要多个隐状态来解释？
  - 若是，K=2 是否合理？是否优于 K=1、K=3 等？
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

# 从 STEP1 生成的 Need_z npz 里读数据
need_npz_file = os.path.join(
    base_out_dir,
    "social_need_from_three_state",
    "social_need_timeseries_from_three_state_glm.npz",
)

# 输出目录：专门放模型比较结果
scan_out_dir = os.path.join(base_out_dir, "social_need_HMM_model_scan")
os.makedirs(scan_out_dir, exist_ok=True)

K_MAX = 4  # 扫描的最大状态数（GMM/HMM）


# ======================= 基础工具函数 =======================
def load_need_npz(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 Need(t) 文件，请先运行 STEP1 脚本生成:\n  {path}")
    dat = np.load(path, allow_pickle=True)
    need_z = np.asarray(dat["need_z"])   # (T,)
    t_epoch = np.asarray(dat["t_epoch"]) # (T,)
    dt = float(dat["dt"].item())
    return need_z, t_epoch, dt


def compute_gaussian_iid_loglik(x):
    """ 单一高斯 iid 模型 """
    mu = x.mean()
    var = x.var()
    if var <= 1e-12:
        var = 1e-12
    n = len(x)
    loglik = -0.5 * n * (np.log(2 * np.pi * var)) - 0.5 * ((x - mu) ** 2 / var).sum()
    k = 2  # mu, var
    return loglik, k, mu, var


def compute_ar1_loglik(x):
    """ 单状态 AR(1): x_t = a x_{t-1} + eps """
    x_prev = x[:-1]
    x_next = x[1:]
    denom = (x_prev ** 2).sum()
    if denom <= 1e-12:
        a = 0.0
    else:
        a = (x_prev * x_next).sum() / denom

    eps = x_next - a * x_prev
    var_eps = eps.var()
    if var_eps <= 1e-12:
        var_eps = 1e-12

    n = len(x_next)
    loglik = -0.5 * n * np.log(2 * np.pi * var_eps) - 0.5 * (eps ** 2 / var_eps).sum()
    k = 2  # a, var_eps
    return loglik, k, a, var_eps


def compute_aic_bic(loglik, k, n):
    aic = 2 * k - 2 * loglik
    bic = k * np.log(n) - 2 * loglik
    return aic, bic


# ======================= GMM & HMM 拟合 =======================
def fit_gmm_k(x, k_comp):
    """
    GMM-K 静态模型：只看分布，不考虑时间结构。
    注意：sklearn 的 score(X) 返回的是“平均 loglik per sample”，
         所以总 loglik = score * n_samples。
    """
    X = x.reshape(-1, 1)
    n = len(x)
    gmm = GaussianMixture(
        n_components=k_comp,
        covariance_type="full",
        random_state=0,
        n_init=10
    )
    gmm.fit(X)
    avg_loglik = gmm.score(X)  # 平均 loglik
    loglik = avg_loglik * n
    # 参数数估计：
    # 每个成分：均值1 + 方差1 => 2 参数
    # 混合权：K 个权重但和=1 => K-1 自由度
    k_param = 2 * k_comp + (k_comp - 1)
    return gmm, loglik, k_param


def fit_hmm_gaussian_k(x, k_state):
    """
    HMM-Gaussian-K：带时间结构。
    注意：hmmlearn 的 score(X) 返回的是整条序列的 loglik，
         不需要再乘 n_samples。
    """
    X = x.reshape(-1, 1)
    hmm = GaussianHMM(
        n_components=k_state,
        covariance_type="diag",
        n_iter=500,
        random_state=0
    )
    hmm.fit(X)
    loglik = hmm.score(X)

    # 参数数估算：
    # 初始分布：K-1 自由度
    # 转移矩阵：K 行，每行 K 个元素但和=1 => 每行 K-1 自由度 => K*(K-1)
    # 观测高斯：每个状态 1 均值 + 1 方差 => 2K
    k_param = (k_state - 1) + k_state * (k_state - 1) + 2 * k_state
    return hmm, loglik, k_param


# ======================= 画模型比较图（可选） =======================
def plot_model_compare(metrics_list, out_dir):
    """
    metrics_list: 每个元素是 dict，包含：
        'name', 'model_type', 'K', 'loglik', 'AIC', 'BIC'
    画两张简单图：
        - AIC 对比
        - BIC 对比
    """
    # 转成表格形式
    names = [m["name"] for m in metrics_list]
    AICs  = [m["AIC"] for m in metrics_list]
    BICs  = [m["BIC"] for m in metrics_list]

    x = np.arange(len(names))

    # AIC 图
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.bar(x, AICs)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("AIC")
    ax.set_title("不同模型的 AIC 对比（越低越好）")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_aic = os.path.join(out_dir, "model_compare_AIC.png")
    fig.savefig(out_aic, dpi=300)
    plt.close(fig)
    print(f"✓ Saved: {out_aic}")

    # BIC 图
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.bar(x, BICs)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("BIC")
    ax.set_title("不同模型的 BIC 对比（越低越好）")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_bic = os.path.join(out_dir, "model_compare_BIC.png")
    fig.savefig(out_bic, dpi=300)
    plt.close(fig)
    print(f"✓ Saved: {out_bic}")

    return out_aic, out_bic


# ======================= README =======================
def write_readme(scan_out_dir, metrics_list, aic_fig, bic_fig, K_MAX):
    lines = []
    lines.append("# Need_z(t) 一维动力学的模型比较与状态数扫描")
    lines.append("")
    lines.append("本文件夹由脚本 `社交需求_一维HMM_状态数扫描_模型比较.py` 自动生成，")
    lines.append("用于在一维社交需求指数 Need_z(t) 上，系统比较不同统计模型及不同隐状态数 K 的拟合效果。")
    lines.append("")
    lines.append("## 一、模型概述")
    lines.append("")
    lines.append("本分析中考虑的模型包括：")
    lines.append("")
    lines.append("1. **Gaussian(iid)**：")
    lines.append("   - 假设所有时间点独立同分布，来自同一个高斯分布 N(μ, σ²)。")
    lines.append("")
    lines.append("2. **AR(1)**：")
    lines.append("   - 单一连续动力学状态：x_t = a · x_{t-1} + ε_t。")
    lines.append("   - 允许时间上的一阶记忆，但不考虑多隐状态切换。")
    lines.append("")
    lines.append(f"3. **GMM-K (K = 1..{K_MAX})**：")
    lines.append("   - 仅考虑 Need_z 的静态分布，由 K 个高斯分布的混合组成；")
    lines.append("   - 不考虑时间顺序和状态切换，仅反映“分布是否多峰”。")
    lines.append("")
    lines.append(f"4. **HMM-Gaussian-K (K = 1..{K_MAX})**：")
    lines.append("   - 带时间结构的模型：存在 K 个隐状态 z_t ∈ {1..K}；")
    lines.append("   - 每个状态下观测值服从不同的高斯分布，并通过转移矩阵在状态之间切换。")
    lines.append("")
    lines.append("后续若需要，还可以在此基础上进一步实现 HMM-AR 模型（每个隐状态对应不同的 AR(1) 动力学），")
    lines.append("本脚本目前先聚焦在 Gaussian / AR(1) / GMM / HMM-Gaussian 这一组模型上。")
    lines.append("")
    lines.append("## 二、模型比较结果（loglik / AIC / BIC）")
    lines.append("")
    lines.append("各模型的 log-likelihood、AIC 和 BIC 统计如下（具体数值请以实际输出为准）：")
    lines.append("")
    for m in metrics_list:
        lines.append(f"- **{m['name']}**")
        lines.append(f"  - 类型: {m['model_type']}")
        if m["K"] is not None:
            lines.append(f"  - 状态数 / 成分数 K = {m['K']}")
        lines.append(f"  - loglik = {m['loglik']:.2f}")
        lines.append(f"  - AIC = {m['AIC']:.2f}")
        lines.append(f"  - BIC = {m['BIC']:.2f}")
        lines.append("")
    lines.append("")
    lines.append("一般来说：")
    lines.append("- **AIC / BIC 数值越小，模型在考虑参数复杂度后对数据的解释越好；**")
    lines.append("- 若 HMM-Gaussian-K 相比 GMM-K 显著降低 BIC，说明时间结构（状态切换）对数据是必要的；")
    lines.append("- 若 GMM-K 或 HMM-K 在 K=2 或 3 时 BIC 最优，则支持“存在 2–3 个隐状态/峰值”的解释；")
    lines.append("- 若 AR(1) 的 BIC 比所有 HMM/GMM 模型都低，则说明一维连续动力学已经足够解释 Need_z。")
    lines.append("")
    lines.append("## 三、AIC / BIC 图像")
    lines.append("")
    lines.append(f"- {os.path.basename(aic_fig)}：按模型名称排列的 AIC 柱状图；")
    lines.append(f"- {os.path.basename(bic_fig)}：按模型名称排列的 BIC 柱状图。")
    lines.append("")
    lines.append("可以通过观察哪些模型在 AIC/BIC 上最小，来判断：")
    lines.append("- 是否需要多隐状态（K>1）；")
    lines.append("- 若需要，多隐状态模型中 K=2 是否已经足够，或是否存在 K=3/4 更优的可能；")
    lines.append("- HMM 是否比静态的 GMM 更适合描述 Need_z(t) 的结构。")
    lines.append("")
    lines.append("## 四、如何理解“到底有几个状态”？")
    lines.append("")
    lines.append("如果比较结果显示：")
    lines.append("- GMM-K 在 K=2 时 BIC 显著低于 K=1，但 K>2 又没有明显改善；")
    lines.append("- HMM-Gaussian-2 的 BIC 明显低于 HMM-1（即单状态高斯）、同时不输给 K=3/4；")
    lines.append("则可以认为 Need_z(t) 在统计上更像是由两个隐状态/模式生成的，而不是单一均匀状态。")
    lines.append("")
    lines.append("此时，再结合前面 HMM 分解得到的：")
    lines.append("- 不同状态下 Need_z 的均值差异（低 vs 高需求）；")
    lines.append("- 各自的时间常数 τ（快恢复 vs 慢恢复）；")
    lines.append("- 与行为标签（独处/无触摸/触摸）的对齐关系；")
    lines.append("就可以比较有信心地解释这两个隐状态为：")
    lines.append("- 一个是长期存在的低需求 / 基线状态；")
    lines.append("- 一个是由持续社交接触渐渐积累、并维持较长时间的高需求 / 激活状态。")
    lines.append("")
    readme_path = os.path.join(scan_out_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"✓ Saved: {readme_path}")
    return readme_path


# ======================= 主函数 =======================
def main():
    print("\n" + "=" * 70)
    print("Need_z 一维动力学：模型比较 + 隐状态数 K 扫描")
    print("=" * 70)

    need_z, t_epoch, dt = load_need_npz(need_npz_file)
    T = len(need_z)
    print(f"✓ Loaded Need_z time series: T={T}, dt≈{dt:.3f}s")

    metrics_list = []

    # 1) Gaussian(iid)
    loglik_g, k_g, mu_g, var_g = compute_gaussian_iid_loglik(need_z)
    aic_g, bic_g = compute_aic_bic(loglik_g, k_g, T)
    metrics_list.append({
        "name": "Gaussian_iid",
        "model_type": "Gaussian(iid)",
        "K": None,
        "loglik": loglik_g,
        "AIC": aic_g,
        "BIC": bic_g,
    })
    print(f"[Gaussian] loglik={loglik_g:.2f}, AIC={aic_g:.2f}, BIC={bic_g:.2f}")

    # 2) AR(1)
    loglik_ar, k_ar, a_ar, var_eps_ar = compute_ar1_loglik(need_z)
    aic_ar, bic_ar = compute_aic_bic(loglik_ar, k_ar, T)
    metrics_list.append({
        "name": "AR1_single_state",
        "model_type": "AR(1)",
        "K": None,
        "loglik": loglik_ar,
        "AIC": aic_ar,
        "BIC": bic_ar,
    })
    print(f"[AR(1)] loglik={loglik_ar:.2f}, AIC={aic_ar:.2f}, BIC={bic_ar:.2f}, a={a_ar:.3f}")

    # 3) GMM-K
    for K in range(1, K_MAX + 1):
        print(f"[GMM] K={K} ...")
        gmm, loglik_gmm, k_param_gmm = fit_gmm_k(need_z, K)
        aic_gmm, bic_gmm = compute_aic_bic(loglik_gmm, k_param_gmm, T)
        name = f"GMM_K{K}"
        metrics_list.append({
            "name": name,
            "model_type": "GMM",
            "K": K,
            "loglik": loglik_gmm,
            "AIC": aic_gmm,
            "BIC": bic_gmm,
        })
        print(f"  loglik={loglik_gmm:.2f}, AIC={aic_gmm:.2f}, BIC={bic_gmm:.2f}")

    # 4) HMM-Gaussian-K
    for K in range(1, K_MAX + 1):
        print(f"[HMM-Gaussian] K={K} ...")
        hmm, loglik_hmm, k_param_hmm = fit_hmm_gaussian_k(need_z, K)
        aic_hmm, bic_hmm = compute_aic_bic(loglik_hmm, k_param_hmm, T)
        name = f"HMM_Gaussian_K{K}"
        metrics_list.append({
            "name": name,
            "model_type": "HMM-Gaussian",
            "K": K,
            "loglik": loglik_hmm,
            "AIC": aic_hmm,
            "BIC": bic_hmm,
        })
        print(f"  loglik={loglik_hmm:.2f}, AIC={aic_hmm:.2f}, BIC={bic_hmm:.2f}")

    # 5) 保存结果 & 画图
    #   也可以顺手存一个 npz 方便后续查
    metrics_npz = os.path.join(scan_out_dir, "model_scan_metrics.npz")
    np.savez(metrics_npz, metrics=metrics_list)
    print(f"✓ Saved metrics to {metrics_npz}")

    aic_fig, bic_fig = plot_model_compare(metrics_list, scan_out_dir)
    readme_path = write_readme(scan_out_dir, metrics_list, aic_fig, bic_fig, K_MAX)

    print("\n✅ Need_z 模型比较 + K 扫描完成")
    print(f"  输出目录: {scan_out_dir}")
    print(f"  README: {readme_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
