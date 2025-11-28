# mouse1.py  或 fit_mouse1_lds.py 都可以
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter   # 使用 pykalman 替代 ssm.LDS

# 1. 读入 npz
data_path = r"F:\工作文件\RA\python\吸引子\Mouse1_for_ssm.npz"
dat = np.load(data_path, allow_pickle=True)

Y = dat["Y"]          # (T, N) z-score 后的神经矩阵
t = dat["t"]          # (T,) 时间
dt = float(dat["dt"]) # 0.1 s
T, N = Y.shape

print("Y shape:", Y.shape, "dt:", dt)

# ===== 先做一个简单的下采样，加速拟合 =====
# 每 5 个时间点取 1 个，可以自己改成 2、3、10 等
downsample = 5

Y_fit = Y[::downsample]
t_fit = t[::downsample]
dt_fit = dt * downsample
T_fit = Y_fit.shape[0]

print(f"Use downsample = {downsample}, Y_fit shape: {Y_fit.shape}, dt_fit = {dt_fit}")

# 2. 建 LDS 模型
D = 3  # latent 维度

kf = KalmanFilter(
    n_dim_obs=N,   # 观测维度（125）
    n_dim_state=D  # latent 维度（3）
)

# 3. 拟合模型（EM），带进度百分比
total_iters = 10   # 先用 10 次试试，确认没问题再往上调

print(f"Start fitting LDS (pykalman), total EM iterations = {total_iters} ...")

for i in range(total_iters):
    # 每次只跑 1 步 EM
    kf = kf.em(Y_fit, n_iter=1)

    current = i + 1
    pct = current / total_iters * 100

    # 这里可以改成每几步打印一次
    print(f"EM progress: {current}/{total_iters} ({pct:.1f}%)")

print("Done fitting.")

# 4. 平滑（在下采样后的时间轴上）
print("Start smoothing...")
x_smooth_fit, P_smooth_fit = kf.smooth(Y_fit)   # (T_fit, D)
print("Smoothing done.")
print("x_smooth (downsampled) shape:", x_smooth_fit.shape)

# 5. 画 latent 维度随时间（用下采样后的 t_fit）
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(t_fit, x_smooth_fit[:, 0])
axes[0].set_ylabel("latent 1")

axes[1].plot(t_fit, x_smooth_fit[:, 1])
axes[1].set_ylabel("latent 2")

axes[2].plot(t_fit, x_smooth_fit[:, 2])
axes[2].set_ylabel("latent 3")
axes[2].set_xlabel("time (s)")

plt.tight_layout()
plt.savefig("mouse1_lds_latents_time_ds.png", dpi=150)

# 6. latent1 vs latent2 相图（下采样版）
plt.figure(figsize=(5, 5))
plt.plot(x_smooth_fit[:, 0], x_smooth_fit[:, 1], linewidth=0.5)
plt.xlabel("latent 1")
plt.ylabel("latent 2")
plt.title("Mouse1 LDS latent trajectory (pykalman, downsampled)")
plt.tight_layout()
plt.savefig("mouse1_lds_latent1_vs_2_ds.png", dpi=150)
