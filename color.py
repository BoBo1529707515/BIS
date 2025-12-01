# -*- coding: utf-8 -*-
"""
基于 Mouse1 数据的 line-attractor-like 分析

步骤：
1. 拟合 3 维 LDS (pykalman)，得到平滑后的 latent 轨迹 x_smooth_fit
2. 从 Excel 读重聚时间 & 社交 bout（开始/结束）
3. 用平滑 latent 的时间结构自动提取：
   - x1: 慢维度（integration 候选）
   - x2: 快维度（orthogonal 快维度）
4. 在 (x1,x2) 空间里画：
   - 行为上色散点图（重聚前 / 重聚后非社交 / 社交）
   - x1, x2 时间轨迹
   - baseline & social 的 velocity landscape
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import pandas as pd

# ================== 0. 路径设置 ==================
out_dir = r"F:\工作文件\RA\python\吸引子\figs"
os.makedirs(out_dir, exist_ok=True)

data_path = r"F:\工作文件\RA\python\吸引子\Mouse1_for_ssm.npz"
beh_path  = r"F:\工作文件\RA\数据集\时间戳\FVB_Panneuronal_Reunion_Isolation_Day3_Mouse#1.xlsx"

# ================== 1. 读入神经数据 ==================
dat = np.load(data_path, allow_pickle=True)
Y  = dat["Y"]          # (T, N)
t  = dat["t"]          # (T,)
dt = float(dat["dt"])  # 0.1 s
T, N = Y.shape

print("Y shape:", Y.shape, "dt:", dt)

# ================== 2. 下采样，加速拟合 ==================
downsample = 5
Y_fit = Y[::downsample]
t_fit = t[::downsample]
dt_fit = dt * downsample
T_fit = Y_fit.shape[0]

print(f"Use downsample = {downsample}, Y_fit shape: {Y_fit.shape}, dt_fit = {dt_fit}")

# ================== 3. 拟合 3 维 LDS ==================
D = 3  # latent 维度

kf = KalmanFilter(
    n_dim_obs=N,
    n_dim_state=D
)

total_iters = 10
print(f"Start fitting LDS (pykalman), total EM iterations = {total_iters} ...")
for i in range(total_iters):
    kf = kf.em(Y_fit, n_iter=1)
    print(f"EM progress: {i+1}/{total_iters} ({(i+1)/total_iters*100:.1f}%)")
print("Done fitting.")

# ================== 4. 平滑，得到 latent 轨迹 ==================
print("Start smoothing...")
x_smooth_fit, P_smooth_fit = kf.smooth(Y_fit)   # (T_fit, D)
print("Smoothing done.")
print("x_smooth (downsampled) shape:", x_smooth_fit.shape)

# ================== 5. 从 Excel 读重聚 & 社交时间 ==================
beh_df = pd.read_excel(beh_path, header=None)
beh_df = beh_df.dropna(how="all")
beh_df[0] = pd.to_numeric(beh_df[0], errors="coerce")
beh_df[1] = beh_df[1].astype(str)

# 5.1 找到“重聚期开始”在 Excel 中的时间（相对）
reunion_rows = beh_df[beh_df[1].str.contains("重聚期开始")]
if len(reunion_rows) == 0:
    print("警告：Excel 里没有找到 '重聚期开始'，先假设 0 s 为重聚起点。")
    reunion_rel = 0.0
else:
    reunion_rel = float(reunion_rows.iloc[0, 0])
print("Relative reunion time in Excel (s):", reunion_rel)

# 5.2 在神经数据里，重聚发生在 903 s（你之前脚本里的设定）
reunion_abs = 903.0

# 5.3 提取所有社交 bout（相对 Excel 时间）
start_mask = beh_df[1].str.contains("社交开始")
end_mask   = beh_df[1].str.contains("社交结束")

start_times_rel = sorted(list(beh_df.loc[start_mask, 0].values))
end_times_rel   = sorted(list(beh_df.loc[end_mask, 0].values))

if len(start_times_rel) != len(end_times_rel):
    print("警告：'社交开始' 和 '社交结束' 数量不一致，请检查 Excel。")
    min_len = min(len(start_times_rel), len(end_times_rel))
    start_times_rel = start_times_rel[:min_len]
    end_times_rel   = end_times_rel[:min_len]

social_intervals_rel = list(zip(start_times_rel, end_times_rel))
print("Social intervals (relative, s):")
for s_rel, e_rel in social_intervals_rel:
    print(f"{s_rel:.2f}  -->  {e_rel:.2f}")

# 5.4 转到神经数据时间轴上的绝对时间
social_intervals = [
    (reunion_abs + (s_rel - reunion_rel),
     reunion_abs + (e_rel - reunion_rel))
    for (s_rel, e_rel) in social_intervals_rel
]

print("Social intervals (absolute, s):")
for s_abs, e_abs in social_intervals:
    print(f"{s_abs:.2f}  -->  {e_abs:.2f}")

# ================== 6. 构造时间轴 & 行为 mask ==================
t_arr = t_fit
t_rel = t_arr - reunion_abs          # 以重聚为 0

mask_before = t_rel < 0
mask_after  = t_rel >= 0

is_social = np.zeros_like(t_arr, dtype=bool)
for (s_abs, e_abs) in social_intervals:
    is_social |= (t_arr >= s_abs) & (t_arr <= e_abs)

mask_after_nonsocial = mask_after & (~is_social)

print("T_fit:", T_fit)
print("before reunion frames:", mask_before.sum())
print("after reunion non-social frames:", mask_after_nonsocial.sum())
print("social frames:", is_social.sum())

# ================== 7. 构造慢/快维度 x1/x2（稳健版） ==================

def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def gram_schmidt(v1, v2):
    """把 v2 正交化到 v1，再都归一化"""
    v1 = normalize(v1)
    v2 = v2 - np.dot(v2, v1) * v1
    v2 = normalize(v2)
    return v1, v2

def tau_from_lambda(lam, dt_):
    """把离散特征值 lambda 换成“等效时间常数”（只用于排序）"""
    mag = np.abs(lam)
    mag = np.clip(mag, 1e-6, 1 - 1e-6)  # 避免 log(1)=0
    return -dt_ / np.log(mag)

x = x_smooth_fit.copy()
x = x - x.mean(axis=0, keepdims=True)   # 去掉整体均值，稳定一点

D = x.shape[1]

# ---------- (1) 用回归估计 A_hat ----------
X0 = x[:-1].T    # D x (T-1)
X1 = x[1:].T     # D x (T-1)

ridge = 1e-6
XX = X0 @ X0.T + ridge * np.eye(D)
A_hat = (X1 @ X0.T) @ np.linalg.pinv(XX)

print("A_hat (regressed from smoothed states):\n", A_hat)

eigvals, eigvecs = np.linalg.eig(A_hat)
taus = np.array([tau_from_lambda(l, dt_fit) for l in eigvals], dtype=float)

# 如果所有 tau 几乎一样，就认为 A_hat 不提供有效慢/快信息
use_A = True
if np.any(~np.isfinite(taus)) or (np.nanmax(taus) - np.nanmin(taus) < 1e-3):
    use_A = False

if use_A:
    # 慢：tau 最大；快：tau 最小
    idx_slow = int(np.nanargmax(taus))
    idx_fast = int(np.nanargmin(taus))
    if idx_fast == idx_slow:
        idx_fast = int((idx_slow + 1) % D)

    v1 = np.real(eigvecs[:, idx_slow])
    v2 = np.real(eigvecs[:, idx_fast])
    v1, v2 = gram_schmidt(v1, v2)
    method_used = "eig(A_hat)"
else:
    # ---------- (2) lag-1 自相关最大化：C1 w = λ C0 w ----------
    x0 = x[:-1]
    x1_ = x[1:]
    C0 = (x0.T @ x0) / (x0.shape[0])
    C1 = (x0.T @ x1_) / (x0.shape[0])

    M = np.linalg.pinv(C0) @ C1
    evals2, evecs2 = np.linalg.eig(M)
    evals2 = np.real(evals2)
    evecs2 = np.real(evecs2)

    # 慢：λ 最大（最接近 1）；快：λ 最小
    idx_slow = int(np.argmax(evals2))
    idx_fast = int(np.argmin(evals2))
    if idx_fast == idx_slow:
        idx_fast = int((idx_slow + 1) % D)

    v1 = evecs2[:, idx_slow]
    v2 = evecs2[:, idx_fast]

    if np.linalg.norm(v1) < 1e-8 or np.linalg.norm(v2) < 1e-8:
        # ---------- (3) fallback: PCA(x) + PCA(dx) ----------
        covx = np.cov(x.T)
        w, V = np.linalg.eigh(covx)
        v1 = V[:, np.argmax(w)]

        dx_tmp = np.diff(x, axis=0)
        covdx = np.cov(dx_tmp.T)
        w2, V2 = np.linalg.eigh(covdx)
        v2 = V2[:, np.argmax(w2)]

        v1, v2 = gram_schmidt(v1, v2)
        method_used = "PCA(x) + PCA(dx)"
    else:
        v1, v2 = gram_schmidt(v1, v2)
        method_used = "gen-eig(C0^{-1}C1)"

W = np.stack([v1, v2], axis=1)  # D x 2
print("Slow/Fast projection method used:", method_used)
print("Projection W (columns = x1,x2):\n", W)

# ================== 8. 投影到 (x1,x2) 空间 ==================
x_proj = x_smooth_fit @ W     # (T_fit, 2)
x1 = x_proj[:, 0]
x2 = x_proj[:, 1]

# ================== 9. (x1,x2) 行为上色散点图 ==================
plt.figure(figsize=(6, 6))

plt.scatter(x1[mask_before],          x2[mask_before],
            s=8, alpha=0.4, color="0.7", label="before reunion")
plt.scatter(x1[mask_after_nonsocial], x2[mask_after_nonsocial],
            s=8, alpha=0.7, color="tab:blue", label="after reunion non-social")
plt.scatter(x1[is_social],            x2[is_social],
            s=10, alpha=0.9, color="tab:red", label="social interaction")

plt.xlabel("x1 (slow mode)")
plt.ylabel("x2 (fast mode)")
plt.title("Mouse1 latent space (projected onto slow/fast modes)")
plt.legend(loc="best", fontsize=8)
plt.tight_layout()

scatter_path = os.path.join(out_dir, "mouse1_x1x2_scatter_slow_fast.png")
plt.savefig(scatter_path, dpi=300)
print("Saved:", scatter_path)

# ================== 10. x1/x2 时间轨迹（重聚对齐 + 社交上色） ==================
fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

for d, (traj, label) in enumerate(zip([x1, x2], ["x1 (slow)", "x2 (fast)"])):
    ax = axes[d]

    # 整条轨迹灰色
    ax.plot(t_rel, traj, color="0.7", linewidth=0.8, label="whole trajectory")

    # 重聚后非社交蓝
    ax.plot(t_rel[mask_after_nonsocial], traj[mask_after_nonsocial],
            color="tab:blue", linewidth=1.5, label="after reunion non-social")

    # 社交红
    for (s_abs, e_abs) in social_intervals:
        mask = (t_arr >= s_abs) & (t_arr <= e_abs)
        if not np.any(mask):
            continue
        ax.plot(t_rel[mask], traj[mask],
                color="tab:red", linewidth=2.0, label="social interaction")

    # 重聚时间线
    ax.axvline(0.0, linestyle="--", linewidth=1, color="k")
    if d == 0:
        y_top = ax.get_ylim()[1]
        ax.text(0.0, y_top, " reunion", ha="left", va="top", fontsize=8)

    ax.set_ylabel(label)

axes[-1].set_xlabel("time from reunion (s)")

# legend 去重
handles, labels_ = axes[0].get_legend_handles_labels()
uniq = dict(zip(labels_, handles))
axes[0].legend(uniq.values(), uniq.keys(), loc="upper right", fontsize=8)

plt.tight_layout()
timeproj_path = os.path.join(out_dir, "mouse1_x1x2_time_slow_fast.png")
plt.savefig(timeproj_path, dpi=300)
print("Saved:", timeproj_path)

# ================== 11. velocity landscape 辅助函数 ==================
def plot_velocity_landscape(x1, x2, v, mask, title, fname,
                            nbins=60, vmax_percentile=95):
    """
    在 (x1,x2) 平面上，用指定 mask 的时间点构建经验 velocity landscape。
    x1, x2: (T,)
    v: 速度大小 (T,)
    mask: bool (T,)
    """
    x1_sel = x1[mask]
    x2_sel = x2[mask]
    v_sel  = v[mask]

    if x1_sel.size < 10:
        print(f"[WARN] Not enough points for {title}, skip.")
        return

    x1_min, x1_max = x1_sel.min(), x1_sel.max()
    x2_min, x2_max = x2_sel.min(), x2_sel.max()

    xbins = np.linspace(x1_min, x1_max, nbins + 1)
    ybins = np.linspace(x2_min, x2_max, nbins + 1)

    v_sum, _, _ = np.histogram2d(x1_sel, x2_sel, bins=[xbins, ybins],
                                 weights=v_sel)
    count, _, _ = np.histogram2d(x1_sel, x2_sel, bins=[xbins, ybins])

    with np.errstate(invalid="ignore", divide="ignore"):
        v_mean = v_sum / count

    mask_valid = np.isfinite(v_mean)
    if np.any(mask_valid):
        v_min = np.nanmin(v_mean[mask_valid])
        v_mean[~mask_valid] = v_min
    else:
        v_mean[:] = 0.0
        v_min = 0.0

    v_max = np.nanpercentile(v_mean, vmax_percentile)
    if v_max > v_min:
        v_mean_clipped = np.clip(v_mean, v_min, v_max)
    else:
        v_mean_clipped = v_mean

    plt.figure(figsize=(5.5, 5))
    ax = plt.gca()

    im = ax.imshow(
        v_mean_clipped.T,
        origin="lower",
        extent=[x1_min, x1_max, x2_min, x2_max],
        cmap="bwr",
        aspect="auto"
    )

    # 在同一子集上叠加黑色轨迹
    ax.plot(x1_sel, x2_sel, color="k", linewidth=1.0, alpha=0.9)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("velocity magnitude in (x1,x2)")

    ax.set_xlabel("x1 (slow mode)")
    ax.set_ylabel("x2 (fast mode)")
    ax.set_title(title)

    plt.tight_layout()
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=300)
    print("Saved:", out_path)

# ================== 12. 计算速度并画 baseline/social velocity landscape ==================
x12 = x_proj           # (T_fit, 2)
dx = np.diff(x12, axis=0)
v = np.linalg.norm(dx, axis=1) / dt_fit   # (T_fit-1,)
v = np.concatenate([v, v[-1:]])           # 补到长度 T_fit

# 12.1 重聚前 baseline
plot_velocity_landscape(
    x1, x2, v,
    mask_before,
    title="Velocity landscape (baseline, pre-reunion)",
    fname="mouse1_velocity_landscape_baseline_x1x2.png"
)

# 12.2 重聚后社交时期
plot_velocity_landscape(
    x1, x2, v,
    is_social,
    title="Velocity landscape (social bouts after reunion)",
    fname="mouse1_velocity_landscape_social_x1x2.png"
)

print("All done.")
