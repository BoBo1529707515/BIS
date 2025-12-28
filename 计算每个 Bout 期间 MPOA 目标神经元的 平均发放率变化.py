# -*- coding: utf-8 -*-
"""
Satiety Index（满足指数）/“Need Encoding”分析
------------------------------------------------
你要的流程（对应你描述的 4 步）：
1) 从行为文件提取所有 Social Bout（社交接触）时间段
2) 对每个 Bout，计算每个 MPOA 目标神经元的平均响应幅度：Δ(ΔF/F)
  - 这里默认定义：Bout 内平均 ΔF/F  -  Bout 前 baseline 窗口平均 ΔF/F
3) 画 Δ(ΔF/F) vs Bout Sequence（第几次接触）
4) 找两类神经元：
  Type A (Pure Sensory): 每次接触都反应、幅度不衰减（slope ~ 0）
  Type B (Need Encoding): 第一次反应大，随接触次数衰减（slope < 0 且显著）
  并基于 Type B 构造 Satiety Index（Need 随 Bout 序号衰减的群体量）

注意：
- 最好用“原始 ΔF/F”做这个分析（不要 z-score 后再算 Δ(ΔF/F)，会改变幅度含义）
- 下面代码默认 npz 里 Y 就是 ΔF/F（T×N），t 是秒；如果你之前保存的是 z-score，请改用原始 ΔF/F 文件。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ===================== 用户参数 =====================
NPZ_PATH = r"F:\工作文件\RA\python\吸引子\Mouse1_for_ssm.npz"  # 需要包含 Y (T,N), t (T,), dt
BEH_XLSX = r"F:\工作文件\RA\数据集\时间戳\FVB_Panneuronal_Reunion_Isolation_Day3_Mouse#1.xlsx"

OUT_DIR = r"F:\工作文件\RA\python\吸引子\figs\satiety_index"
os.makedirs(OUT_DIR, exist_ok=True)

# 你之前的对齐方式：reunion_abs 是录制绝对时间轴上的“重聚时刻”
REUNION_ABS = 903.0

# baseline 窗口：Bout 开始前多少秒作为 baseline（避免包含刚进入触摸的过渡）
BASELINE_SEC = 5.0

# bout 内响应统计方式：mean（也可以 median / trimmed mean，自行改）
AGG_FUN = "mean"  # "mean" or "median"

# Type A / Type B 判别阈值（可按数据调）
MIN_BOUTS = 6                 # 至少有这么多 bout 才做分类
RESP_THRESHOLD = 0.02         # “有反应”的幅度阈值（Δ(ΔF/F)），按你数据量级改
EARLY_BOUTS = 3               # “第一次很大”可放宽为前 1~3 次平均
SLOPE_ALPHA = 0.05            # slope 显著性阈值
NEG_SLOPE_EPS = 1e-6          # 防止数值恰好为 0
PERSIST_FRAC_A = 0.6          # Type A：至少多少比例的 bouts 都“有反应”

# Satiety Index 输出：用 Type B 群体响应得到一个随 bout 序号变化的 1D 量
# 定义为 Need(t) = mean_i[resp_i(bout)]，然后线性归一化到 [0,1]
# 也可以做指数拟合/累积等（见下方注释）


# ===================== 工具函数 =====================
def load_npz(npz_path: str):
   dat = np.load(npz_path, allow_pickle=True)
   Y = np.asarray(dat["Y"])   # (T,N) 期望是 ΔF/F
   t = np.asarray(dat["t"])   # (T,) 秒
   dt = float(dat["dt"].item()) if "dt" in dat else np.median(np.diff(t))
   return Y, t, dt


def load_social_bouts_from_excel(beh_path: str, reunion_abs: float):
   """
   复用你之前的逻辑：Excel 里记录“重聚期开始”“社交开始/结束”为相对时间，
   通过 reunion_abs 映射到录制绝对时间轴。
   """
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
   # 过滤掉异常 bout（start>=end）
   social_intervals = [(s, e) for (s, e) in social_intervals if np.isfinite(s) and np.isfinite(e) and e > s]
   return social_intervals


def _agg(x: np.ndarray, how: str):
   if x.size == 0:
       return np.nan
   if how == "median":
       return np.nanmedian(x)
   return np.nanmean(x)


def compute_bout_responses(Y: np.ndarray, t: np.ndarray, bouts, baseline_sec: float, agg_fun: str):
   """
   计算每个 bout、每个神经元的响应矩阵 R：
     R[j,i] = mean(Y in bout_j) - mean(Y in baseline window before bout_j)

   返回：
     R: (K,N)  bout×neuron
     valid_bouts: 实际用于计算的 bout（可能会丢弃不完整/越界的）
   """
   T, N = Y.shape
   dt = np.median(np.diff(t))
   baseline_frames = max(1, int(round(baseline_sec / dt)))

   R_list = []
   valid_bouts = []

   for (s, e) in bouts:
       # bout mask
       bout_mask = (t >= s) & (t <= e)
       if not np.any(bout_mask):
           continue

       # baseline: immediately before s
       # 找到 s 对应的索引位置
       s_idx = np.searchsorted(t, s, side="left")
       b0 = max(0, s_idx - baseline_frames)
       b1 = s_idx  # [b0, b1)
       if b1 - b0 < max(1, baseline_frames // 2):
           # baseline 不足，跳过这个 bout
           continue

       Y_bout = Y[bout_mask, :]      # (Tb,N)
       Y_base = Y[b0:b1, :]          # (Tb0,N)

       bout_mean = np.array([_agg(Y_bout[:, i], agg_fun) for i in range(N)])
       base_mean = np.array([_agg(Y_base[:, i], agg_fun) for i in range(N)])

       R = bout_mean - base_mean     # Δ(ΔF/F)
       R_list.append(R)
       valid_bouts.append((s, e))

   if len(R_list) == 0:
       return None, []

   R_mat = np.vstack(R_list)  # (K,N)
   return R_mat, valid_bouts


def classify_neurons(R: np.ndarray):
   """
   输入：R (K,N)  bout×neuron 的响应幅度（Δ(ΔF/F)）
   输出：
     typeA_idx, typeB_idx
     per-neuron stats（slope, p, r, early_resp, frac_responsive）
   判别逻辑（可按你数据调）：
     - Type A：多数 bouts 有反应（|R|>RESP_THRESHOLD 或 R>阈值），且 slope 不显著为负
     - Type B：早期反应大（前 EARLY_BOUTS 次平均 > RESP_THRESHOLD），且 slope 显著为负
   """
   K, N = R.shape
   x = np.arange(1, K + 1)  # bout sequence: 1..K

   slopes = np.full(N, np.nan)
   pvals = np.full(N, np.nan)
   rs = np.full(N, np.nan)
   early_resp = np.full(N, np.nan)
   frac_resp = np.full(N, np.nan)

   typeA = []
   typeB = []

   for i in range(N):
       y = R[:, i].astype(float)
       valid = np.isfinite(y)
       if valid.sum() < max(MIN_BOUTS, 3):
           continue

       xv = x[valid]
       yv = y[valid]

       lr = linregress(xv, yv)
       slopes[i] = lr.slope
       pvals[i] = lr.pvalue
       rs[i] = lr.rvalue

       # “早期响应”
       k0 = min(EARLY_BOUTS, yv.size)
       early_resp[i] = np.nanmean(yv[:k0])

       # “持续有反应”的比例（这里用正向响应阈值；如你要双向就改成 np.abs(yv)>RESP_THRESHOLD）
       frac_resp[i] = np.mean(yv > RESP_THRESHOLD)

       # Type B: Need Encoding（早期大 + 随 bout 衰减）
       is_B = (early_resp[i] > RESP_THRESHOLD) and (slopes[i] < -NEG_SLOPE_EPS) and (pvals[i] < SLOPE_ALPHA)

       # Type A: Pure Sensory（一直反应 + 不衰减）
       is_A = (frac_resp[i] >= PERSIST_FRAC_A) and (not ((slopes[i] < -NEG_SLOPE_EPS) and (pvals[i] < SLOPE_ALPHA)))

       if is_B:
           typeB.append(i)
       if is_A:
           typeA.append(i)

   stats = {
       "slope": slopes,
       "p_slope": pvals,
       "r": rs,
       "early_resp": early_resp,
       "frac_resp": frac_resp,
   }
   return np.array(typeA, dtype=int), np.array(typeB, dtype=int), stats


def plot_resp_vs_bout(R: np.ndarray, out_dir: str, typeA_idx=None, typeB_idx=None):
   K, N = R.shape
   x = np.arange(1, K + 1)

   # 画所有神经元的响应（浅色）+ TypeA/TypeB 强调
   plt.figure(figsize=(9, 5))
   for i in range(N):
       plt.plot(x, R[:, i], alpha=0.15, linewidth=1)

   if typeA_idx is not None and typeA_idx.size > 0:
       for i in typeA_idx[: min(20, typeA_idx.size)]:
           plt.plot(x, R[:, i], alpha=0.9, linewidth=2, label="Type A (example)" if i == typeA_idx[0] else None)

   if typeB_idx is not None and typeB_idx.size > 0:
       for i in typeB_idx[: min(20, typeB_idx.size)]:
           plt.plot(x, R[:, i], alpha=0.9, linewidth=2, label="Type B (example)" if i == typeB_idx[0] else None)

   plt.axhline(0, linestyle="--")
   plt.xlabel("Bout sequence (1,2,3,...)")
   plt.ylabel("Δ(ΔF/F) per bout (bout mean - pre-bout baseline)")
   plt.title("Neuron responses across social bouts")
   plt.grid(alpha=0.3)
   plt.legend(fontsize=8, loc="best")
   path = os.path.join(out_dir, "dff_response_vs_bout_sequence_all_neurons.png")
   plt.tight_layout()
   plt.savefig(path, dpi=300)
   plt.close()
   print(f"✓ Saved: {path}")


def compute_and_plot_satiety_index(R: np.ndarray, typeB_idx: np.ndarray, out_dir: str):
   """
   Satiety Index（你描述里的“Need Encoding”量）：
     1) 先取 Type B 神经元的 bout 响应均值：need[bout] = mean_i R[bout,i]
     2) 再把 need 归一化到 [0,1]，默认让“早期需求高=1，后期低=0”
   """
   K, _ = R.shape
   x = np.arange(1, K + 1)

   if typeB_idx.size == 0:
       print("⚠️ No Type B neurons found; cannot compute Satiety Index.")
       return None

   need = np.nanmean(R[:, typeB_idx], axis=1)  # (K,)

   # 归一化：让前期高、后期低更直观
   # 如果 need 是负向（比如抑制型 Need Encoding），你可以改成 -need 再归一化
   need_norm = (need - np.nanmin(need)) / (np.nanmax(need) - np.nanmin(need) + 1e-8)

   # 这里给一个“满足指数 Satiety”版本：satiety = 1 - need_norm
   satiety = 1.0 - need_norm

   plt.figure(figsize=(7, 4))
   plt.plot(x, need, linewidth=2, label="Need (Type B mean Δ(ΔF/F))")
   plt.axhline(0, linestyle="--")
   plt.xlabel("Bout sequence")
   plt.ylabel("Need signal (Δ(ΔF/F))")
   plt.title("Need signal across bouts (Type B population)")
   plt.grid(alpha=0.3)
   plt.legend(fontsize=8)
   p1 = os.path.join(out_dir, "need_signal_typeB_mean.png")
   plt.tight_layout()
   plt.savefig(p1, dpi=300)
   plt.close()
   print(f"✓ Saved: {p1}")

   plt.figure(figsize=(7, 4))
   plt.plot(x, satiety, linewidth=2, label="Satiety Index = 1 - normalized Need")
   plt.ylim(-0.05, 1.05)
   plt.xlabel("Bout sequence")
   plt.ylabel("Satiety Index (0..1)")
   plt.title("Satiety Index across bouts")
   plt.grid(alpha=0.3)
   plt.legend(fontsize=8)
   p2 = os.path.join(out_dir, "satiety_index.png")
   plt.tight_layout()
   plt.savefig(p2, dpi=300)
   plt.close()
   print(f"✓ Saved: {p2}")

   np.savez(
       os.path.join(out_dir, "satiety_index_results.npz"),
       need=need,
       need_norm=need_norm,
       satiety=satiety,
       typeB_idx=typeB_idx,
   )
   print(f"✓ Saved: {os.path.join(out_dir, 'satiety_index_results.npz')}")

   return satiety


# ===================== 主流程 =====================
def main():
   # 1) 读数据
   Y, t, dt = load_npz(NPZ_PATH)
   print(f"Loaded Y: {Y.shape}, t: {t.shape}, dt={dt:.3f}s")

   # 2) 读 bouts
   bouts = load_social_bouts_from_excel(BEH_XLSX, REUNION_ABS)
   print(f"Found social bouts: {len(bouts)}")

   # 3) 计算每个 bout 的神经元响应 R[bout, neuron]
   R, valid_bouts = compute_bout_responses(Y, t, bouts, BASELINE_SEC, AGG_FUN)
   if R is None:
       raise RuntimeError("No valid bouts for response computation.")
   print(f"Computed R: {R.shape} (K bouts × N neurons), valid_bouts={len(valid_bouts)}")

   # 4) 分类 Type A / Type B
   typeA_idx, typeB_idx, stats = classify_neurons(R)
   print(f"Type A (Pure Sensory): {typeA_idx.size} neurons")
   print(f"Type B (Need Encoding): {typeB_idx.size} neurons")
   np.savez(
       os.path.join(OUT_DIR, "neuron_type_classification.npz"),
       R=R,
       valid_bouts=np.array(valid_bouts, dtype=float),
       typeA_idx=typeA_idx,
       typeB_idx=typeB_idx,
       **stats
   )
   print(f"✓ Saved: {os.path.join(OUT_DIR, 'neuron_type_classification.npz')}")

   # 5) 画 Δ(ΔF/F) vs bout sequence
   plot_resp_vs_bout(R, OUT_DIR, typeA_idx=typeA_idx, typeB_idx=typeB_idx)

   # 6) 计算并画 Satiety Index（基于 Type B）
   _ = compute_and_plot_satiety_index(R, typeB_idx, OUT_DIR)

   # 7) 额外：把 Type B 的 slope 分布画出来（可选）
   if typeB_idx.size > 0:
       plt.figure(figsize=(6, 4))
       plt.hist(stats["slope"][typeB_idx], bins=20, alpha=0.9)
       plt.axvline(0, linestyle="--")
       plt.xlabel("Slope of response vs bout sequence")
       plt.ylabel("Neuron count")
       plt.title("Type B slopes (should be < 0)")
       plt.grid(alpha=0.3)
       p = os.path.join(OUT_DIR, "typeB_slope_hist.png")
       plt.tight_layout()
       plt.savefig(p, dpi=300)
       plt.close()
       print(f"✓ Saved: {p}")

   print("✅ Done.")


if __name__ == "__main__":
   main()

"""
你可能会想做的两个小改动（按数据特性选）：
1) 如果你的 Need Encoding 是“抑制型”（第一次强烈下降，后面趋近 0），
  那么 R 可能整体为负，compute_and_plot_satiety_index 里可以用 need = -mean(R[:, typeB])
2) 如果 bout 内长度差异很大，bout_mean 可以改为对每个 bout 做积分/面积：
  bout_auc = sum(Y_bout - base_mean) * dt
"""
