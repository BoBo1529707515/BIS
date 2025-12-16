import os
import numpy as np

# ======= 1. 路径：你这里已经给了二态的 =======
two_state_npz = r"F:\工作文件\RA\python\吸引子\figs\touch_glm2\touch_glm2_outputs.npz"
three_state_npz = r"F:\工作文件\RA\python\吸引子\figs\three_state_glm\three_state_glm_outputs.npz"

print("二态 GLM npz:", two_state_npz)
print("三态 GLM npz:", three_state_npz)

# ======= 2. 读二态 GLM 结果 =======
two = np.load(two_state_npz, allow_pickle=True)
print("\n二态 GLM 文件包含字段:", two.files)

# 这里假设你在二态脚本里把权重存成 'beta_motor'
# 如果你用的是别的名字（比如 'betas' 或 'coef'），就在这里改一下键名
if "beta_motor" in two.files:
    beta_2state = np.asarray(two["beta_motor"])   # (N,)
else:
    raise KeyError("二态 npz 里没找到 'beta_motor'，请打印 two.files 看看字段名改一下。")

print("二态 β_motor 形状:", beta_2state.shape)

# ======= 3. 读三态 GLM 结果 =======
three = np.load(three_state_npz, allow_pickle=True)
print("\n三态 GLM 文件包含字段:", three.files)

if "betas" in three.files:
    betas_3state = np.asarray(three["betas"])     # (3, N)
else:
    raise KeyError("三态 npz 里没找到 'betas'，请打印 three.files 看看字段名改一下。")

print("三态 betas 形状:", betas_3state.shape)

# ======= 4. 构造三态里的“触摸轴”：β_state2 - β_state1 =======
beta_3state_motor = betas_3state[2] - betas_3state[1]   # (N,)

# ======= 5. 核实神经元数是否一致 =======
if beta_2state.shape != beta_3state_motor.shape:
    raise ValueError(f"两个模型神经元数不一样！二态 {beta_2state.shape}, 三态 {beta_3state_motor.shape}")

N = beta_2state.shape[0]
print(f"\n✅ 神经元数一致: N = {N}")

# ======= 6. 看两套 β 的相关性 =======
corr = np.corrcoef(beta_2state, beta_3state_motor)[0, 1]
print(f"\n二态 β_motor 与 三态 (β_state2 - β_state1) 的相关系数: {corr:.3f}")

# ======= 7. 看 top neuron 是否重合 =======
top_k = 20

idx_top_2state = np.argsort(-np.abs(beta_2state))[:top_k]
idx_top_3state = np.argsort(-np.abs(beta_3state_motor))[:top_k]

print(f"\n二态 GLM |β| Top{top_k} 神经元索引:\n", idx_top_2state)
print(f"\n三态构造 β_motor |β| Top{top_k} 神经元索引:\n", idx_top_3state)

overlap = set(idx_top_2state).intersection(set(idx_top_3state))
print(f"\nTop{top_k} 重合神经元个数: {len(overlap)}")
print("重合的神经元索引:", sorted(overlap))
