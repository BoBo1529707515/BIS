# fit_mouse1_lds.py
import numpy as np
import matplotlib.pyplot as plt
import ssm

# 1. 读入我们刚才生成的 npz
data_path = r"F:\工作文件\RA\python\吸引子\Mouse1_for_ssm.npz"
dat = np.load(data_path, allow_pickle=True)

Y = dat["Y"]          # (T, N) z-score 后的神经矩阵
t = dat["t"]          # (T,) 时间
dt = float(dat["dt"]) # 0.1 s
T, N = Y.shape

print("Y shape:", Y.shape, "dt:", dt)

# 2. 建一个最简单的 LDS 模型
#    N  = 观测维度（神经元个数 = 125）
#    D  = latent 维度（比如 3 维，后面可以改成 5、8 等）
D = 3
lds = ssm.LDS(
    N,          # 观测维度
    D,          # latent 维度
    emissions="gaussian",   # 观测是连续值
    dynamics="gaussian"     # 标准线性高斯动力学
)

# 3. 拟合模型（EM）
#    可以先跑 50 轮试试，看到效果再调
print("Start fitting LDS...")
lds.fit(Y, method="laplace_em", num_iters=50)

print("Done.")

# 4. 平滑（smoothing），得到每个时间点的 latent 均值 x(t)
#    x_smooth: (T, D)
x_smooth = lds.smooth(data=Y)

print("x_smooth shape:", x_smooth.shape)

# 5. 简单画一画 latent 维度随时间的轨迹
#   （只画前 2 个 latent 维度）
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(t, x_smooth[:, 0])
axes[0].set_ylabel("latent 1")

axes[1].plot(t, x_smooth[:, 1])
axes[1].set_ylabel("latent 2")

axes[2].plot(t, x_smooth[:, 2])
axes[2].set_ylabel("latent 3")
axes[2].set_xlabel("time (s)")

plt.tight_layout()
plt.savefig("mouse1_lds_latents_time.png", dpi=150)

# 6. 再画一个 latent1 vs latent2 的相图，看一下轨迹形状
plt.figure(figsize=(5, 5))
plt.plot(x_smooth[:, 0], x_smooth[:, 1], linewidth=0.5)
plt.xlabel("latent 1")
plt.ylabel("latent 2")
plt.title("Mouse1 LDS latent trajectory")
plt.tight_layout()
plt.savefig("mouse1_lds_latent1_vs_2.png", dpi=150)
