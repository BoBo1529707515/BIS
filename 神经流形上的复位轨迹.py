# -*- coding: utf-8 -*-
"""
神经流形上的“复位轨迹 / 吸引子”分析 + 你现有 raw(z-score) 数据与行为时间戳整合版
----------------------------------------------------------------------
你给的脚本里已经包含了所有“数据位置/时间定义/状态定义”的关键信息。
本脚本在保留你原有的：时间斜率GLM + non-touch ramp-down axis 的基础上，
新增并整合“状态空间/吸引子/复位轨迹（bout-aligned）/简单分类器（isolation vs satiety）”模块。

输出内容（会保存到 base_out_dir/state_space_reset 下）：
1) PCA 低维状态空间散点：isolation vs satiety 分布
2) 状态轴投影 z(t) 与 “satiety吸引子距离” d_M(t) 的时间轨迹（含三段：iso / reunion / re-iso）
3) bout 对齐的复位曲线：平均 d_M(τ) 与指数复位拟合参数 τ_r
4) 基于状态轴的简单分类器：P(satiety|t) 随时间
5) 与你原 non-touch ramp-down population axis 的对照（可选）：相关与叠图

依赖：
numpy, pandas, matplotlib, scipy, statsmodels, scikit-learn
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import zscore
from scipy.signal import medfilt
from scipy.optimize import curve_fit

import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from sklearn.decomposition import PCA


# ======================= 参数设置 =======================
base_out_dir = r"F:\工作文件\RA\python\吸引子\figs"
os.makedirs(base_out_dir, exist_ok=True)

data_path = r"F:\工作文件\RA\python\吸引子\Mouse1_for_ssm.npz"
beh_path  = r"F:\工作文件\RA\数据集\时间戳\FVB_Panneuronal_Reunion_Isolation_Day3_Mouse#1.xlsx"

# Reunion 绝对时间（秒）
reunion_abs = 903.0

# 是否用 OASIS 去卷积（默认不用）
USE_OASIS = False
if USE_OASIS:
   try:
       from oasis.functions import deconvolve
   except ImportError:
       print("⚠️ 未找到 OASIS，自动关闭 USE_OASIS")
       USE_OASIS = False

# 可选：手动指定 re-isolation 时间段（绝对时间，秒）
# 如果都为 None，则默认从“最后一次 social 结束”到实验结束
REISO_START = None  # 例如 1600.0
REISO_END   = None  # 例如 1900.0

# 状态定义相关
PRE_REUNION_WINDOW = 200.0   # state0: 重聚前 200s isolation
REUNION_EXTRA_AFTER = 60.0   # 提取 reunion epoch：最后一次social结束后再多取 60s

# Satiety 集合定义（用于“吸引子”）
# 选项：
#   "late_reunion" : reunion epoch 最后 SATIETY_WINDOW 秒（包含 state1+2）
#   "touch_only"   : state2（触摸段）
SATIETY_DEF = "late_reunion"
SATIETY_WINDOW = 200.0  # 仅在 SATIETY_DEF="late_reunion" 时生效

# PCA/状态空间参数
N_PCS = 10

# bout 对齐的“复位”分析参数
ALIGN_TO = "end"      # "start" 或 "end"
BOUT_PRE  = 5.0       # bout 前取 5s
BOUT_POST = 20.0      # bout 后取 20s
MIN_POST_FIT = 1.0    # 拟合指数时，从 bout 后 1s 开始（避开瞬时跃迁）

# 你原来时间斜率 GLM 的阈值
FDR_ALPHA = 0.05
EFFECT_THR = 0.10

# 画图字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ======================= 数据加载 & 状态构造 =======================
def load_neural_data(npz_path):
   """
   读取 npz: 需要包含 Y (T, N), t (T,), dt
   只做 z-score，不做滤波。
   """
   print("=" * 60)
   print("LOADING NEURAL DATA (raw z-scored, no smoothing)")
   print("=" * 60)
   dat = np.load(npz_path, allow_pickle=True)
   Y  = np.asarray(dat["Y"])     # (T, N)
   t  = np.asarray(dat["t"])     # (T,)
   dt = float(dat["dt"].item())

   Y_z = zscore(Y, axis=0)       # raw -> z-score
   print(f"✓ T={len(t)}, N={Y.shape[1]}, dt={dt:.3f}s")
   return Y_z, t, dt


def load_behavior(beh_path, reunion_abs):
   """
   从 Excel 读出 “重聚期开始”、“社交开始/结束” 时刻。
   返回 social_intervals（绝对时间）和 reunion_rel（Excel 内的重聚相对时间）。
   """
   print("\n" + "=" * 60)
   print("LOADING BEHAVIOR DATA")
   print("=" * 60)
   beh_df = pd.read_excel(beh_path, header=None).dropna(how="all")
   beh_df[0] = pd.to_numeric(beh_df[0], errors="coerce")
   beh_df[1] = beh_df[1].astype(str).str.strip().str.lower()

   reunion_rows = beh_df[beh_df[1].str.contains("重聚期开始", na=False)]
   reunion_rel = float(reunion_rows.iloc[0, 0]) if len(reunion_rows) > 0 else 0.0

   starts = beh_df[beh_df[1].str.contains("社交开始", na=False)][0].values
   ends   = beh_df[beh_df[1].str.contains("社交结束", na=False)][0].values

   social_intervals = [
       (reunion_abs + (s - reunion_rel), reunion_abs + (e - reunion_rel))
       for s, e in zip(starts, ends)
   ]

   print(f"✓ Found {len(social_intervals)} social bouts")
   return social_intervals, reunion_rel


def build_social_flag(t, social_intervals):
   flag = np.zeros_like(t, dtype=int)
   for (start, end) in social_intervals:
       flag[(t >= start) & (t <= end)] = 1
   return flag


def build_three_state_labels(t, social_intervals, reunion_abs, pre_reunion_window=200.0):
   """
   state0: 重聚前 pre_reunion_window 秒内独处
   state1: 重聚后，非触摸（social_flag=0）
   state2: 重聚后，触摸（social_flag=1）
   其他时间点设为 -1。
   """
   T = len(t)
   labels = np.full(T, -1, dtype=int)
   social_flag = build_social_flag(t, social_intervals)

   labels[social_flag == 1] = 2

   mask_state0 = (t >= (reunion_abs - pre_reunion_window)) & (t < reunion_abs)
   labels[mask_state0] = 0

   mask_after_reunion = (t >= reunion_abs)
   mask_state1 = mask_after_reunion & (social_flag == 0)
   labels[mask_state1] = 1
   return labels


def extract_reunion_epoch(Y, t, reunion_abs, social_intervals, extra_after=60.0):
   """
   取重聚后的整段（直到最后一次社交结束 + buffer）。
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


def calcium_to_spikes(Y):
   """
   如 USE_OASIS=False，则直接用 Y（z-scored）当作活动。
   """
   if not USE_OASIS:
       print("\n" + "=" * 60)
       print("SKIP OASIS: using Y (z-scored) as 'activity'")
       print("=" * 60)
       return Y

   print("\n" + "=" * 60)
   print("RUNNING OASIS DECONVOLUTION")
   print("=" * 60)
   T, N = Y.shape
   spikes = np.zeros_like(Y)
   for i in range(N):
       F_trace = Y[:, i]
       c, s, *_ = deconvolve(F_trace, penalty=1)
       spikes[:, i] = s
   spikes_z = zscore(spikes, axis=0)
   print("✓ OASIS complete")
   return spikes_z


# ======================= 你原来的：时间斜率 GLM + FDR + effect size =======================
def run_time_slope_glm(spikes,
                      t,
                      labels_3state,
                      reunion_abs,
                      out_dir,
                      target_states,
                      condition_name,
                      time_ref="reunion",
                      fdr_alpha=0.05,
                      effect_thr=0.10):
   """
   通用时间斜率 GLM：
     activity ~ β0 + β_time * time_z

   target_states: 元组，如 (1,) / (2,) / (1,2)
   """
   os.makedirs(out_dir, exist_ok=True)

   mask_valid = np.isin(labels_3state, list(target_states))
   t_cond = t[mask_valid]
   spikes_cond = spikes[mask_valid]

   if t_cond.size == 0:
       print(f"⚠️ 条件 {condition_name} 下没有时间点，跳过")
       return None

   if time_ref == "reunion":
       t_rel = t_cond - reunion_abs
   elif time_ref == "segment_min":
       t_rel = t_cond - t_cond.min()
   else:
       raise ValueError(f"Unknown time_ref: {time_ref}")

   t_rel_z = (t_rel - t_rel.mean()) / (t_rel.std() + 1e-6)

   X = np.column_stack([np.ones_like(t_rel_z), t_rel_z])

   T_cond, N = spikes_cond.shape
   betas = np.zeros((N, 2))
   pvals = np.zeros((N, 2))

   print("\n" + "=" * 60)
   print(f"RUNNING TIME-SLOPE GLM [{condition_name}]")
   print("=" * 60)
   print(f"  target_states = {target_states}, T = {T_cond}, N = {N}")
   print(f"  FDR alpha = {fdr_alpha}, effect_thr = {effect_thr}")

   for i in range(N):
       y = spikes_cond[:, i]
       if np.allclose(y, y[0]):
           betas[i, :] = np.nan
           pvals[i, :] = np.nan
           continue

       model = sm.GLM(y, X, family=sm.families.Gaussian())
       try:
           res = model.fit()
           betas[i, :] = res.params
           pvals[i, :] = res.pvalues
       except Exception as e:
           print(f"  Neuron {i} GLM failed: {e}")
           betas[i, :] = np.nan
           pvals[i, :] = np.nan

   beta_time = betas[:, 1]
   p_time = pvals[:, 1]

   # FDR 校正
   p_fdr = np.full_like(p_time, np.nan)
   valid = ~np.isnan(p_time)
   if valid.sum() > 0:
       _, p_fdr_valid, _, _ = multipletests(
           p_time[valid], alpha=fdr_alpha, method="fdr_bh"
       )
       p_fdr[valid] = p_fdr_valid

   # 宽松 & 严格筛选
   dec_mask_loose = (beta_time < 0) & (p_time < 0.05)
   dec_idx_loose = np.where(dec_mask_loose)[0]

   dec_mask_strict = (beta_time < -effect_thr) & (p_fdr < fdr_alpha)
   dec_idx_strict = np.where(dec_mask_strict)[0]

   print(f"\n[{condition_name}] LOOSE  : β_time < 0 & p_time < 0.05 → {dec_idx_loose.size} neurons")
   print(f"[{condition_name}] STRICT : β_time < -{effect_thr} & FDR < {fdr_alpha} → {dec_idx_strict.size} neurons")

   # 直方图：所有 neuron vs 严格筛选
   finite_beta = beta_time[~np.isnan(beta_time)]
   if finite_beta.size > 0:
       bins = np.linspace(finite_beta.min(), finite_beta.max(), 25)
       plt.figure(figsize=(7, 4))
       plt.hist(finite_beta, bins=bins, alpha=0.4, label="All neurons")
       if dec_idx_strict.size > 0:
           plt.hist(beta_time[dec_mask_strict], bins=bins, alpha=0.8,
                    label="Strict: β_time < -thr & FDR<α")
       plt.axvline(0, color="red", linestyle="--", label="β_time = 0")
       plt.xlabel("β_time (Gaussian GLM)")
       plt.ylabel("Neuron count")
       plt.title(f"β_time in {condition_name} period\n(all vs strictly decreasing)")
       plt.grid(alpha=0.3)
       plt.legend(fontsize=8)
       hist_path = os.path.join(out_dir, f"beta_time_hist_{condition_name}_FDR_effect.png")
       plt.tight_layout()
       plt.savefig(hist_path, dpi=300)
       plt.close()
       print(f"✓ Saved: {hist_path}")

   out_npz = os.path.join(out_dir, f"time_slope_{condition_name}_results.npz")
   np.savez(
       out_npz,
       betas=betas, pvals=pvals, beta_time=beta_time, p_time=p_time, p_fdr=p_fdr,
       dec_mask_loose=dec_mask_loose, dec_idx_loose=dec_idx_loose,
       dec_mask_strict=dec_mask_strict, dec_idx_strict=dec_idx_strict,
       target_states=np.array(target_states),
       time_ref=time_ref, fdr_alpha=fdr_alpha, effect_thr=effect_thr
   )
   print(f"✓ Saved: {out_npz}")

   return {
       "betas": betas,
       "pvals": pvals,
       "beta_time": beta_time,
       "p_time": p_time,
       "p_fdr": p_fdr,
       "dec_mask": dec_mask_loose,
       "dec_idx": dec_idx_loose,
       "dec_mask_strict": dec_mask_strict,
       "dec_idx_strict": dec_idx_strict,
       "t_cond": t_cond,
       "spikes_cond": spikes_cond,
       "t_rel": t_rel,
       "t_rel_z": t_rel_z,
       "condition_name": condition_name
   }


def build_non_touch_ramp_axis(glm_non, N):
   """
   non-touch ramp-down axis：
     w_i = -β_time_i（只对严格筛选下降 neuron，其他为 0）
   """
   beta_time = glm_non["beta_time"]
   dec_mask_strict = glm_non["dec_mask_strict"]
   if dec_mask_strict.sum() == 0:
       print("⚠️ non-touch 严格筛选为空，改用宽松 dec_mask")
       dec_mask_strict = glm_non["dec_mask"]

   w = np.zeros(N)
   w[dec_mask_strict] = -beta_time[dec_mask_strict]
   if np.allclose(w, 0):
       return None
   return w


# ======================= 状态空间 / 吸引子 / 分类器 / 复位轨迹 =======================
def get_reiso_mask(t_full, social_intervals):
   if REISO_START is not None and REISO_END is not None:
       return (t_full >= REISO_START) & (t_full <= REISO_END)
   if len(social_intervals) == 0:
       return np.zeros_like(t_full, dtype=bool)
   last_end = max(e for (_, e) in social_intervals)
   return t_full >= last_end


def define_satiety_mask(t_full, labels_3state_full, t_epoch, mask_epoch):
   """
   在 full 时间轴上定义 satiety_mask（用于吸引子估计）。
   """
   if SATIETY_DEF == "touch_only":
       return labels_3state_full == 2

   if SATIETY_DEF == "late_reunion":
       # reunion epoch 的最后 SATIETY_WINDOW 秒
       if len(t_epoch) == 0:
           return np.zeros_like(t_full, dtype=bool)
       t_end = t_epoch[-1]
       t_start = max(reunion_abs, t_end - SATIETY_WINDOW)
       return (t_full >= t_start) & (t_full <= t_end) & np.isin(labels_3state_full, [1, 2])

   raise ValueError(f"Unknown SATIETY_DEF: {SATIETY_DEF}")


def fit_pca_on_relevant_time(spikes_full, labels_3state_full):
   """
   PCA 最稳的做法：用与“状态对比”相关的时间点拟合（state0/1/2）。
   """
   mask_pca = np.isin(labels_3state_full, [0, 1, 2])
   X = spikes_full[mask_pca]
   if X.shape[0] < 10:
       raise RuntimeError("用于 PCA 的有效时间点太少（<10），检查 labels_3state_full 是否正确。")

   pca = PCA(n_components=min(N_PCS, X.shape[1]))
   pca.fit(X)
   return pca


def build_state_axis_and_attractor(Y_pc_full, iso_mask, satiety_mask):
   """
   在 PCA 空间中：
   - 状态轴 w = μ_sat - μ_iso
   - satiety attractor: μ_sat, Σ_sat（用于 Mahalanobis 距离）
   """
   Y_iso = Y_pc_full[iso_mask]
   Y_sat = Y_pc_full[satiety_mask]

   if Y_iso.shape[0] < 5:
       raise RuntimeError("Isolation 点太少，无法估计 μ_iso。")
   if Y_sat.shape[0] < 5:
       raise RuntimeError("Satiety 点太少，无法估计 μ_sat/Σ_sat。")

   mu_iso = np.nanmean(Y_iso, axis=0)
   mu_sat = np.nanmean(Y_sat, axis=0)

   w = mu_sat - mu_iso
   w_norm = np.linalg.norm(w) + 1e-12
   w = w / w_norm

   # satiety 协方差（加一点 shrinkage 防止不可逆）
   X = Y_sat - mu_sat
   Sigma = np.cov(X.T)
   # ridge
   Sigma = Sigma + 1e-6 * np.eye(Sigma.shape[0])
   Sigma_inv = np.linalg.inv(Sigma)

   return w, mu_iso, mu_sat, Sigma, Sigma_inv


def project_and_distance(Y_pc_full, w_state, mu_sat, Sigma_inv):
   """
   z(t) = w^T y(t)
   d_M(t) = mahalanobis distance to satiety attractor
   """
   z_t = Y_pc_full @ w_state
   dM = np.sqrt(np.einsum("ti,ij,tj->t", (Y_pc_full - mu_sat), Sigma_inv, (Y_pc_full - mu_sat)))
   return z_t, dM


def sigmoid(x):
   return 1.0 / (1.0 + np.exp(-x))


def build_simple_classifier(z_t, iso_mask, satiety_mask):
   """
   用 1D 状态轴投影 z(t) 做一个非常稳的“软分类器”：
   - 阈值 z0 = (mean_iso + mean_sat)/2
   - 标度 s = pooled std（防止过陡）
   p_sat = sigmoid((z - z0) / s)
   """
   z_iso = z_t[iso_mask]
   z_sat = z_t[satiety_mask]
   z0 = 0.5 * (np.nanmean(z_iso) + np.nanmean(z_sat))

   s = np.nanstd(np.concatenate([z_iso, z_sat]))
   s = max(s, 1e-6)

   p_sat = sigmoid((z_t - z0) / s)
   return p_sat, z0, s


def aligned_matrix(signal, t_full, event_times, pre, post, dt):
   """
   将 signal(t) 按 event_times 对齐，返回 [n_events, n_bins]，缺失用 NaN。
   这里假设 t_full 是等间隔 dt（你的数据就是这样）。
   """
   n_bins = int(np.round((pre + post) / dt)) + 1
   tau = np.linspace(-pre, post, n_bins)

   # 用索引对齐（避免插值的漂移）
   t0 = t_full[0]
   idx_events = np.round((np.array(event_times) - t0) / dt).astype(int)

   out = np.full((len(event_times), n_bins), np.nan, dtype=float)
   for k, idx0 in enumerate(idx_events):
       idx_start = idx0 - int(np.round(pre / dt))
       idx_end   = idx0 + int(np.round(post / dt))
       if idx_end < 0 or idx_start >= len(signal):
           continue
       s0 = max(idx_start, 0)
       s1 = min(idx_end, len(signal) - 1)
       seg = signal[s0:s1+1]

       # 放到对应位置
       left_pad = s0 - idx_start
       out[k, left_pad:left_pad+len(seg)] = seg

   return tau, out


def exp_recovery(tau, d_inf, A, tau_r):
   """
   d(τ) = d_inf + A * exp(-τ/tau_r)
   """
   return d_inf + A * np.exp(-tau / (tau_r + 1e-12))


def fit_exponential_reset(tau, mean_curve, min_post_fit=1.0):
   """
   在 τ >= min_post_fit 的区间拟合指数回归。
   """
   mask = (tau >= min_post_fit) & np.isfinite(mean_curve)
   x = tau[mask]
   y = mean_curve[mask]
   if len(x) < 10:
       return None

   d_inf0 = np.nanmedian(y[-max(3, len(y)//10):])
   A0 = max(y[0] - d_inf0, 1e-3)
   tau0 = max((x[-1] - x[0]) / 3.0, 0.5)

   try:
       popt, pcov = curve_fit(
           exp_recovery, x, y,
           p0=[d_inf0, A0, tau0],
           bounds=([0.0, 0.0, 1e-3], [np.inf, np.inf, np.inf]),
           maxfev=20000
       )
       return popt, pcov
   except Exception as e:
       print(f"⚠️ exp fit failed: {e}")
       return None


def plot_state_space_scatter(out_dir, Y_pc_full, labels_3state_full, iso_mask, satiety_mask):
   os.makedirs(out_dir, exist_ok=True)
   plt.figure(figsize=(7, 6))
   # 默认灰
   plt.scatter(Y_pc_full[:, 0], Y_pc_full[:, 1], s=3, alpha=0.10, label="All (state0/1/2)")

   plt.scatter(Y_pc_full[iso_mask, 0], Y_pc_full[iso_mask, 1], s=6, alpha=0.35, label="Isolation (state0)")
   plt.scatter(Y_pc_full[satiety_mask, 0], Y_pc_full[satiety_mask, 1], s=6, alpha=0.35, label="Satiety set")

   plt.xlabel("PC1")
   plt.ylabel("PC2")
   plt.title("Neural state space (PCA)\nIsolation vs Satiety set")
   plt.grid(alpha=0.25)
   plt.legend(fontsize=8)
   fp = os.path.join(out_dir, "state_space_scatter_iso_vs_satiety.png")
   plt.tight_layout()
   plt.savefig(fp, dpi=300)
   plt.close()
   print(f"✓ Saved: {fp}")


def plot_time_courses(out_dir, t_full, reunion_abs, z_t, dM, p_sat,
                     labels_3state_full, social_intervals):
   os.makedirs(out_dir, exist_ok=True)
   t_rel = t_full - reunion_abs

   iso_mask = labels_3state_full == 0
   non_touch_mask = labels_3state_full == 1
   touch_mask = labels_3state_full == 2
   reiso_mask = get_reiso_mask(t_full, social_intervals)

   # 1) z(t)
   plt.figure(figsize=(11, 6))
   ax1 = plt.subplot(3, 1, 1)
   ax1.plot(t_rel[iso_mask], z_t[iso_mask], linewidth=1.0)
   ax1.set_title("State axis projection z(t) (Isolation)")
   ax1.set_ylabel("z(t)")
   ax1.grid(alpha=0.25)

   ax2 = plt.subplot(3, 1, 2, sharey=ax1)
   ax2.plot(t_rel[non_touch_mask], z_t[non_touch_mask], linewidth=1.0, color="tab:orange", label="non-touch")
   ax2.plot(t_rel[touch_mask], z_t[touch_mask], linewidth=1.0, color="tab:blue", alpha=0.8, label="touch")
   ax2.set_title("z(t) during Reunion (state1 & state2)")
   ax2.set_ylabel("z(t)")
   ax2.grid(alpha=0.25)
   ax2.legend(fontsize=8)

   ax3 = plt.subplot(3, 1, 3, sharey=ax1)
   if np.any(reiso_mask):
       ax3.plot(t_rel[reiso_mask], z_t[reiso_mask], linewidth=1.0, color="tab:green")
   ax3.set_title("z(t) (Re-isolation)")
   ax3.set_xlabel("Time relative to reunion (s)")
   ax3.set_ylabel("z(t)")
   ax3.grid(alpha=0.25)

   fp = os.path.join(out_dir, "timecourse_state_axis_z.png")
   plt.tight_layout()
   plt.savefig(fp, dpi=300)
   plt.close()
   print(f"✓ Saved: {fp}")

   # 2) dM(t)
   plt.figure(figsize=(11, 6))
   bx1 = plt.subplot(3, 1, 1)
   bx1.plot(t_rel[iso_mask], dM[iso_mask], linewidth=1.0)
   bx1.set_title("Mahalanobis distance to Satiety attractor dM(t) (Isolation)")
   bx1.set_ylabel("dM(t)")
   bx1.grid(alpha=0.25)

   bx2 = plt.subplot(3, 1, 2, sharey=bx1)
   bx2.plot(t_rel[non_touch_mask], dM[non_touch_mask], linewidth=1.0, color="tab:orange", label="non-touch")
   bx2.plot(t_rel[touch_mask], dM[touch_mask], linewidth=1.0, color="tab:blue", alpha=0.8, label="touch")
   bx2.set_title("dM(t) during Reunion (state1 & state2)")
   bx2.set_ylabel("dM(t)")
   bx2.grid(alpha=0.25)
   bx2.legend(fontsize=8)

   bx3 = plt.subplot(3, 1, 3, sharey=bx1)
   if np.any(reiso_mask):
       bx3.plot(t_rel[reiso_mask], dM[reiso_mask], linewidth=1.0, color="tab:green")
   bx3.set_title("dM(t) (Re-isolation)")
   bx3.set_xlabel("Time relative to reunion (s)")
   bx3.set_ylabel("dM(t)")
   bx3.grid(alpha=0.25)

   fp = os.path.join(out_dir, "timecourse_distance_dM.png")
   plt.tight_layout()
   plt.savefig(fp, dpi=300)
   plt.close()
   print(f"✓ Saved: {fp}")

   # 3) classifier p(satiety)
   plt.figure(figsize=(11, 3.5))
   plt.plot(t_rel[np.isin(labels_3state_full, [0, 1, 2])],
            p_sat[np.isin(labels_3state_full, [0, 1, 2])],
            linewidth=1.0)
   plt.ylim([-0.05, 1.05])
   plt.xlabel("Time relative to reunion (s)")
   plt.ylabel("P(satiety)")
   plt.title("Simple state classifier from z(t)")
   plt.grid(alpha=0.25)

   fp = os.path.join(out_dir, "timecourse_classifier_p_satiety.png")
   plt.tight_layout()
   plt.savefig(fp, dpi=300)
   plt.close()
   print(f"✓ Saved: {fp}")


def plot_bout_aligned_reset(out_dir, dt, t_full, dM, social_intervals, align_to="end"):
   os.makedirs(out_dir, exist_ok=True)

   if len(social_intervals) == 0:
       print("⚠️ 没有 social bouts，跳过 bout-aligned reset 分析")
       return

   if align_to == "start":
       events = [s for (s, _) in social_intervals]
       tag = "bout_start"
   elif align_to == "end":
       events = [e for (_, e) in social_intervals]
       tag = "bout_end"
   else:
       raise ValueError("align_to must be 'start' or 'end'")

   tau, M = aligned_matrix(dM, t_full, events, BOUT_PRE, BOUT_POST, dt)
   mean_curve = np.nanmean(M, axis=0)
   sem_curve = np.nanstd(M, axis=0) / np.sqrt(np.sum(np.isfinite(M), axis=0).clip(min=1))

   # 指数拟合
   fit = fit_exponential_reset(tau, mean_curve, min_post_fit=MIN_POST_FIT)
   fit_txt = "fit: failed"
   fit_params = None
   if fit is not None:
       (d_inf, A, tau_r), pcov = fit
       fit_params = dict(d_inf=float(d_inf), A=float(A), tau_r=float(tau_r))
       fit_txt = f"fit: d_inf={d_inf:.3g}, A={A:.3g}, tau_r={tau_r:.3g}s"

   # 画图
   plt.figure(figsize=(8, 4.8))
   plt.plot(tau, mean_curve, linewidth=2.0, label="mean dM(τ)")
   plt.fill_between(tau, mean_curve - sem_curve, mean_curve + sem_curve, alpha=0.2, label="SEM")
   plt.axvline(0, color="k", linestyle="--", alpha=0.7, label=f"{tag}=0")

   if fit is not None:
       xfit = tau[tau >= MIN_POST_FIT]
       plt.plot(xfit, exp_recovery(xfit, d_inf, A, tau_r), linestyle="--", linewidth=2.0, label="exp recovery fit")

   plt.xlabel("Time from bout event τ (s)")
   plt.ylabel("dM(τ) to Satiety attractor")
   plt.title(f"Bout-aligned reset (align_to={align_to})\n{fit_txt}")
   plt.grid(alpha=0.25)
   plt.legend(fontsize=8)

   fp = os.path.join(out_dir, f"bout_aligned_reset_{tag}_dM.png")
   plt.tight_layout()
   plt.savefig(fp, dpi=300)
   plt.close()
   print(f"✓ Saved: {fp}")

   # 保存数据
   np.savez(
       os.path.join(out_dir, f"bout_aligned_reset_{tag}_dM.npz"),
       tau=tau, M=M, mean_curve=mean_curve, sem_curve=sem_curve,
       events=np.array(events),
       fit_params=fit_params
   )
   if fit_params is not None:
       print(f"✓ Exp reset parameters: {fit_params}")


def compare_with_ramp_axis(out_dir, spikes_full, t_full, reunion_abs, labels_3state_full,
                          w_ramp, z_t, social_intervals):
   """
   你原 non-touch ramp-down axis 与 state-axis z(t) 的对照：
   - 计算 ramp 信号 S_ramp(t)
   - 叠图（只在 state1 画 ramp，其他空）
   - 相关（reunion non-touch 段）
   """
   if w_ramp is None or np.allclose(w_ramp, 0):
       print("⚠️ w_ramp 为空，跳过 ramp-axis 对照")
       return

   os.makedirs(out_dir, exist_ok=True)
   t_rel = t_full - reunion_abs

   S = spikes_full @ w_ramp
   S = (S - np.nanmean(S)) / (np.nanstd(S) + 1e-6)

   mask_state1 = labels_3state_full == 1
   S_plot = np.where(mask_state1, S, np.nan)
   z_plot = np.where(mask_state1, z_t, np.nan)

   # 相关（state1 段）
   if np.sum(mask_state1) > 10:
       r = np.corrcoef(S[mask_state1], z_t[mask_state1])[0, 1]
   else:
       r = np.nan

   plt.figure(figsize=(10, 4))
   plt.plot(t_rel, S_plot, linewidth=1.2, label="Non-touch ramp axis (state1 only)")
   plt.plot(t_rel, z_plot, linewidth=1.2, label="State axis projection z(t) (state1 only)")
   plt.xlabel("Time relative to reunion (s)")
   plt.ylabel("z-scored signal")
   plt.title(f"Ramp axis vs State axis (state1 only), corr={r:.3g}")
   plt.grid(alpha=0.25)
   plt.legend(fontsize=8)

   fp = os.path.join(out_dir, "compare_ramp_axis_vs_state_axis.png")
   plt.tight_layout()
   plt.savefig(fp, dpi=300)
   plt.close()
   print(f"✓ Saved: {fp}")

   np.savez(
       os.path.join(out_dir, "compare_ramp_axis_vs_state_axis.npz"),
       t_rel=t_rel, S=S, z_t=z_t, mask_state1=mask_state1, corr=r
   )


# ======================= 主流程 =======================
def main():
   print("\n" + "=" * 70)
   print("STATE-SPACE RESET (attractor) + TIME-SLOPE GLM (raw z-score)")
   print("=" * 70)

   # 1) 全程神经 & 行为
   Y, t_full, dt = load_neural_data(data_path)
   social_intervals, reunion_rel = load_behavior(beh_path, reunion_abs)
   spikes_full = calcium_to_spikes(Y)

   # 2) 三状态标签（full）
   labels_3state_full = build_three_state_labels(
       t_full, social_intervals, reunion_abs, pre_reunion_window=PRE_REUNION_WINDOW
   )

   # 3) 提取 reunion epoch（用于跑你原来的 GLM）
   Y_epoch, t_epoch, mask_epoch = extract_reunion_epoch(
       Y, t_full, reunion_abs, social_intervals, extra_after=REUNION_EXTRA_AFTER
   )
   spikes_epoch = spikes_full[mask_epoch]
   labels_3state_epoch = labels_3state_full[mask_epoch]

   # 4) 你原来的三种条件时间斜率 GLM（只在 reunion epoch 内做）
   glm_results = {}
   conds = [
       ("non_touch",  (1,)),
       ("social_only", (2,)),
       ("all_reunion", (1, 2))
   ]
   for cond_name, states in conds:
       out_dir_cond = os.path.join(base_out_dir, f"time_slope_{cond_name}")
       glm_results[cond_name] = run_time_slope_glm(
           spikes_epoch, t_epoch, labels_3state_epoch,
           reunion_abs=reunion_abs,
           out_dir=out_dir_cond,
           target_states=states,
           condition_name=cond_name,
           time_ref="reunion",
           fdr_alpha=FDR_ALPHA,
           effect_thr=EFFECT_THR
       )

   # 5) 构建 non-touch ramp axis（用于对照）
   glm_non = glm_results.get("non_touch", None)
   w_ramp = None
   if glm_non is not None:
       w_ramp = build_non_touch_ramp_axis(glm_non, N=spikes_full.shape[1])

   # ======================= 复位轨迹/吸引子：PCA + 状态轴 + 距离 + 分类器 =======================
   out_dir_reset = os.path.join(base_out_dir, "state_space_reset")
   os.makedirs(out_dir_reset, exist_ok=True)

   print("\n" + "=" * 60)
   print("STATE SPACE (PCA) + STATE AXIS + ATTRACTOR")
   print("=" * 60)

   # PCA：用 state0/1/2 时间点拟合
   pca = fit_pca_on_relevant_time(spikes_full, labels_3state_full)

   # 把 full 的 state0/1/2 都投影到 PC 空间（只保留这些点，其他点为 NaN 方便对齐）
   mask_valid = np.isin(labels_3state_full, [0, 1, 2])
   Y_pc_full = np.full((len(t_full), pca.n_components_), np.nan, dtype=float)
   Y_pc_full[mask_valid] = pca.transform(spikes_full[mask_valid])

   # 定义 isolation & satiety 集合（都在 full 时间轴上）
   iso_mask = labels_3state_full == 0
   satiety_mask = define_satiety_mask(t_full, labels_3state_full, t_epoch, mask_epoch)

   # 只在 valid 范围内取（防止 NaN）
   iso_mask = iso_mask & mask_valid
   satiety_mask = satiety_mask & mask_valid

   # 计算状态轴与吸引子
   w_state, mu_iso, mu_sat, Sigma_sat, Sigma_inv = build_state_axis_and_attractor(
       Y_pc_full[mask_valid],  # 注意：这里传入的是“有效点的矩阵”
       iso_mask[mask_valid],
       satiety_mask[mask_valid]
   )

   # 回到 full 长度的 z(t), dM(t)（无效点为 NaN）
   z_t = np.full(len(t_full), np.nan, dtype=float)
   dM  = np.full(len(t_full), np.nan, dtype=float)
   z_valid, dM_valid = project_and_distance(Y_pc_full[mask_valid], w_state, mu_sat, Sigma_inv)
   z_t[mask_valid] = z_valid
   dM[mask_valid]  = dM_valid

   # 简单分类器 p(satiety)
   p_sat = np.full(len(t_full), np.nan, dtype=float)
   p_sat_valid, z0, s = build_simple_classifier(z_valid, iso_mask[mask_valid], satiety_mask[mask_valid])
   p_sat[mask_valid] = p_sat_valid

   # 保存关键参数
   np.savez(
       os.path.join(out_dir_reset, "state_axis_attractor_params.npz"),
       w_state=w_state, mu_iso=mu_iso, mu_sat=mu_sat,
       Sigma_sat=Sigma_sat,
       z0=z0, s=s,
       pca_components=pca.components_,
       pca_explained_variance_ratio=pca.explained_variance_ratio_
   )
   print("✓ Saved: state_axis_attractor_params.npz")

   # 画 state space scatter（只画 valid 点）
   plot_state_space_scatter(
       out_dir_reset,
       Y_pc_full[mask_valid],
       labels_3state_full[mask_valid],
       iso_mask[mask_valid],
       satiety_mask[mask_valid]
   )

   # 画时间轨迹：z(t), dM(t), p(satiety)
   plot_time_courses(
       out_dir_reset,
       t_full, reunion_abs,
       z_t, dM, p_sat,
       labels_3state_full,
       social_intervals
   )

   # bout-aligned reset：用 dM(τ)
   plot_bout_aligned_reset(
       out_dir_reset, dt,
       t_full, dM,
       social_intervals,
       align_to=ALIGN_TO
   )

   # 与 non-touch ramp axis 对照（可选）
   compare_with_ramp_axis(
       out_dir_reset,
       spikes_full, t_full, reunion_abs, labels_3state_full,
       w_ramp, z_t,
       social_intervals
   )

   print("\n✅ All analysis complete!")
   print("=" * 70 + "\n")


if __name__ == "__main__":
   main()
