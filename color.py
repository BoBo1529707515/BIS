# mouse1_with_behavior.py
import os
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import pandas as pd

# ================== 0. 输出文件夹 ==================
out_dir = r"F:\工作文件\RA\python\吸引子\figs"  # 你可以改成自己想要的目录
os.makedirs(out_dir, exist_ok=True)

# ---------------- 1. 读入神经数据 ----------------
data_path = r"F:\工作文件\RA\python\吸引子\Mouse1_for_ssm.npz"
dat = np.load(data_path, allow_pickle=True)

Y = dat["Y"]          # (T, N)
t = dat["t"]          # (T,)
dt = float(dat["dt"]) # 0.1 s
T, N = Y.shape

print("Y shape:", Y.shape, "dt:", dt)

# ---------------- 2. 下采样，加速拟合 ----------------
downsample = 5   # 先用 5，之后想更细再改 2 或 1
Y_fit = Y[::downsample]
t_fit = t[::downsample]
dt_fit = dt * downsample
T_fit = Y_fit.shape[0]

print(f"Use downsample = {downsample}, Y_fit shape: {Y_fit.shape}, dt_fit = {dt_fit}")

# ---------------- 3. 建 LDS 模型并 EM 拟合 ----------------
D = 3  # latent 维度

kf = KalmanFilter(
    n_dim_obs=N,
    n_dim_state=D
)

total_iters = 10   # 先 10 次，稳定后可以往上调
print(f"Start fitting LDS (pykalman), total EM iterations = {total_iters} ...")

for i in range(total_iters):
    kf = kf.em(Y_fit, n_iter=1)
    current = i + 1
    pct = current / total_iters * 100
    print(f"EM progress: {current}/{total_iters} ({pct:.1f}%)")

print("Done fitting.")

# ---------------- 4. 平滑，得到 latent 轨迹 ----------------
print("Start smoothing...")
x_smooth_fit, P_smooth_fit = kf.smooth(Y_fit)   # (T_fit, D)
print("Smoothing done.")
print("x_smooth (downsampled) shape:", x_smooth_fit.shape)

# ---------------- 5. 从 Excel 读重聚 & 社交时间轴 ----------------
beh_path = r"F:\工作文件\RA\数据集\时间戳\FVB_Panneuronal_Reunion_Isolation_Day3_Mouse#1.xlsx"

# 不假设表头，直接按两列读：
# 第 0 列：时间（相对重聚开始，单位 s）
# 第 1 列：事件名称（字符串，包含“重聚期开始 / 社交开始 / 社交结束”）
beh_df = pd.read_excel(beh_path, header=None)
beh_df = beh_df.dropna(how="all")  # 去掉全空行

# 确保第 0 列是数字
beh_df[0] = pd.to_numeric(beh_df[0], errors="coerce")

# 事件名转成字符串，方便匹配
beh_df[1] = beh_df[1].astype(str)

# 找到重聚期开始那一行（相对时间 = 0）
reunion_rows = beh_df[beh_df[1].str.contains("重聚期开始")]
if len(reunion_rows) == 0:
    print("警告：Excel 里没有找到 '重聚期开始'，先假设 0 s 为重聚起点。")
    reunion_rel = 0.0
else:
    reunion_rel = float(reunion_rows.iloc[0, 0])  # 理论上应该是 0
print("Relative reunion time in Excel (s):", reunion_rel)

# 在神经数据中，重聚发生在 903 s（绝对时间）
reunion_abs = 903.0

# ---------------- 6. 从 Excel 自动提取所有社交 bout ----------------
start_mask = beh_df[1].str.contains("社交开始")
end_mask   = beh_df[1].str.contains("社交结束")

start_times_rel = list(beh_df.loc[start_mask, 0].values)
end_times_rel   = list(beh_df.loc[end_mask, 0].values)

# 按时间排序，以免顺序乱掉
start_times_rel = sorted(start_times_rel)
end_times_rel   = sorted(end_times_rel)

if len(start_times_rel) != len(end_times_rel):
    print("警告：'社交开始' 和 '社交结束' 数量不一致，请检查 Excel。")
    # 这里简单地截取到最短长度
    min_len = min(len(start_times_rel), len(end_times_rel))
    start_times_rel = start_times_rel[:min_len]
    end_times_rel   = end_times_rel[:min_len]

social_intervals_rel = list(zip(start_times_rel, end_times_rel))

print("Social intervals (relative, s):")
for s_rel, e_rel in social_intervals_rel:
    print(f"{s_rel:.2f}  -->  {e_rel:.2f}")

# 转为绝对时间：对齐到神经数据里的重聚时刻（903 s）
social_intervals = [
    (reunion_abs + (s_rel - reunion_rel),
     reunion_abs + (e_rel - reunion_rel))
    for (s_rel, e_rel) in social_intervals_rel
]

print("Social intervals (absolute, s):")
for s_abs, e_abs in social_intervals:
    print(f"{s_abs:.2f}  -->  {e_abs:.2f}")

# ---------------- 7. 构造 mask ----------------
t_arr = t_fit                # (T_fit,)
t_rel = t_arr - reunion_abs  # 以重聚为 0 点的相对时间

# 社交时间 mask：落在任一社交 bout 内即为 social
is_social = np.zeros_like(t_arr, dtype=bool)
for (s_abs, e_abs) in social_intervals:
    is_social |= (t_arr >= s_abs) & (t_arr <= e_abs)

# 重聚后但非社交
mask_after_reunion_nonsocial = (t_arr >= reunion_abs) & (~is_social)

# ---------------- 8. 时间轨迹图：一条线 + 上色社交/非社交 ----------------
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
labels = ["latent 1", "latent 2", "latent 3"]

for d in range(3):
    ax = axes[d]
    x_d = x_smooth_fit[:, d]

    # ① 整条轨迹：灰色细线（从 recording 开始到结束）
    ax.plot(
        t_rel,
        x_d,
        linewidth=0.8,
        alpha=0.7,
        color="0.6",    # 灰色
        label="whole trajectory" if d == 0 else None
    )

    # ② 重聚后非社交段：蓝色加粗覆盖
    mask_blue = mask_after_reunion_nonsocial
    if np.any(mask_blue):
        ax.plot(
            t_rel[mask_blue],
            x_d[mask_blue],
            linewidth=1.5,
            alpha=0.9,
            color="tab:blue",
            label="after reunion non-social" if d == 0 else None
        )

    # ③ 社交时间段：红色再覆盖（优先级最高）
    for (s_abs, e_abs) in social_intervals:
        mask = (t_arr >= s_abs) & (t_arr <= e_abs)
        if not np.any(mask):
            continue
        ax.plot(
            t_rel[mask],
            x_d[mask],
            linewidth=2.0,
            alpha=0.95,
            color="tab:red",
            label="social interaction" if d == 0 else None
        )

    # ④ 标记重聚时刻：相对时间 0
    ax.axvline(0.0, linestyle="--", linewidth=1, color="k")
    if d == 0:
        y_top = ax.get_ylim()[1]
        ax.text(0.0, y_top, " reunion", va="top", ha="left")

    ax.set_ylabel(labels[d])

axes[-1].set_xlabel("time from reunion (s)")

# 图例只放在最后一张
handles, legend_labels = axes[0].get_legend_handles_labels()
axes[-1].legend(handles, legend_labels, loc="upper right", fontsize=8)

plt.tight_layout()
time_fig_path = os.path.join(out_dir, "mouse1_lds_latents_time_ds_color.png")
plt.savefig(time_fig_path, dpi=300)
print("Saved time-series figure to:", time_fig_path)

# ---------------- 9. 相图：latent1 vs latent2，用颜色分状态 ----------------
plt.figure(figsize=(5, 5))

plt.scatter(
    x_smooth_fit[t_rel < 0, 0],      # 重聚前：t_rel < 0
    x_smooth_fit[t_rel < 0, 1],
    s=4, alpha=0.5, color="0.6", label="before reunion"
)

plt.scatter(
    x_smooth_fit[mask_after_reunion_nonsocial, 0],
    x_smooth_fit[mask_after_reunion_nonsocial, 1],
    s=4, alpha=0.5, color="tab:blue", label="after reunion non-social"
)

plt.scatter(
    x_smooth_fit[is_social, 0],
    x_smooth_fit[is_social, 1],
    s=6, alpha=0.9, color="tab:red", label="social interaction"
)

plt.xlabel("latent 1")
plt.ylabel("latent 2")
plt.title("Mouse1 latent space with social bouts (from Excel)")
plt.legend(loc="best", fontsize=8)
plt.tight_layout()
phase_fig_path = os.path.join(out_dir, "mouse1_lds_latent1_vs_2_ds_color.png")
plt.savefig(phase_fig_path, dpi=300)
print("Saved phase-space figure to:", phase_fig_path)
# ================== 11. Dynamic velocity landscape ==================
# 目标：仿照 Fig.1k，画出 x1–x2 平面上“平均速度”的地形图 + 叠加真实轨迹

# 取前两维 latent 作为 x1, x2
x1 = x_smooth_fit[:, 0]
x2 = x_smooth_fit[:, 1]

# ---- 11.1 计算每个时间点的速度大小 ----
# 用相邻两个时间点的差分近似速度：v_t = |dx| / dt_fit
x12 = x_smooth_fit[:, :2]           # (T_fit, 2)
dx = np.diff(x12, axis=0)           # (T_fit-1, 2)
v = np.linalg.norm(dx, axis=1) / dt_fit  # (T_fit-1,)

# 为了和 x1,x2 对齐（长度 T_fit），简单在最后补一个值
v = np.concatenate([v, v[-1:]])     # (T_fit,)

# ---- 11.2 在 x1–x2 平面上做网格，并计算每个格子的平均速度 ----
nbins = 60  # 网格数，越大越细

x1_min, x1_max = x1.min(), x1.max()
x2_min, x2_max = x2.min(), x2.max()

xbins = np.linspace(x1_min, x1_max, nbins + 1)
ybins = np.linspace(x2_min, x2_max, nbins + 1)

# 速度求和
v_sum, _, _ = np.histogram2d(x1, x2, bins=[xbins, ybins], weights=v)
# 每个格子的采样次数
count, _, _ = np.histogram2d(x1, x2, bins=[xbins, ybins])

with np.errstate(invalid="ignore", divide="ignore"):
    v_mean = v_sum / count  # 平均速度

# 对没有数据的格子，填上全局最小速度，避免 NaN
mask_valid = np.isfinite(v_mean)
if np.any(mask_valid):
    v_min = np.nanmin(v_mean[mask_valid])
    v_mean[~mask_valid] = v_min
else:
    # 极端情况：如果全是 NaN，就直接用 0
    v_mean[:] = 0.0

# 为了避免少数特别大的 outlier 把颜色拉爆，可以截到 95 百分位
v_max = np.nanpercentile(v_mean, 95)
v_min = np.nanmin(v_mean)
if v_max > v_min:
    v_mean_clipped = np.clip(v_mean, v_min, v_max)
else:
    v_mean_clipped = v_mean

# ---- 11.3 画 dynamic velocity landscape ----
plt.figure(figsize=(5.5, 5))

ax = plt.gca()

# 背景：x1–x2 平面上的速度地形图（蓝=低速，红=高速）
im = ax.imshow(
    v_mean_clipped.T,               # 注意转置，跟坐标对应
    origin="lower",
    extent=[x1_min, x1_max, x2_min, x2_max],
    cmap="bwr",                     # 蓝-白-红，和原文 high/low 一致
    aspect="auto"
)

# 叠加黑色轨迹线（神经状态随时间的移动）
ax.plot(
    x1,
    x2,
    color="k",
    linewidth=1.5,
    alpha=0.9
)

# （可选）标几个时间点，比如 t0 / t50 / t100（相对重聚的时间）
for t_mark, label in [(0.0, "t0"), (50.0, "t50"), (100.0, "t100")]:
    # 只有在数据时间范围内才标
    if (t_rel.min() <= t_mark <= t_rel.max()):
        idx = int(np.argmin(np.abs(t_rel - t_mark)))
        ax.scatter(
            x1[idx],
            x2[idx],
            s=30,
            facecolor="white",
            edgecolor="k",
            zorder=5
        )
        ax.text(
            x1[idx],
            x2[idx],
            label,
            fontsize=8,
            ha="left",
            va="bottom"
        )

# 也可以标一下重聚时刻在 latent 空间的位置
idx_reunion = int(np.argmin(np.abs(t_rel - 0.0)))
ax.scatter(
    x1[idx_reunion],
    x2[idx_reunion],
    s=40,
    facecolor="yellow",
    edgecolor="k",
    zorder=6,
    label="reunion"
)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("velocity magnitude in latent space")

ax.set_xlabel("latent 1 (x1)")
ax.set_ylabel("latent 2 (x2)")
ax.set_title("Dynamic velocity landscape (Mouse 1)")

plt.tight_layout()
vel_fig_path = os.path.join(out_dir, "mouse1_dynamic_velocity_landscape.png")
plt.savefig(vel_fig_path, dpi=300)
print("Saved dynamic velocity landscape figure to:", vel_fig_path)
