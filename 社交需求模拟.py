# 社交需求指数_三态GLM.py
# -*- coding: utf-8 -*-

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ======================= 路径配置 =======================
base_out_dir = r"F:\工作文件\RA\python\吸引子\figs"
three_state_npz = os.path.join(base_out_dir, "three_state_glm", "three_state_glm_outputs.npz")

# 输出目录：专门放“社交需求”分析结果
need_out_dir = os.path.join(base_out_dir, "social_need_from_three_state")
os.makedirs(need_out_dir, exist_ok=True)

# 预触摸窗口（秒）
PRE_TOUCH_WINDOW = 5.0

# ======================================================


def load_three_state_npz(path):
    dat = np.load(path, allow_pickle=True)
    betas = dat["betas"]                     # (3, N)
    spikes_epoch = dat["spikes_epoch"]       # (T, N)
    t_epoch = dat["t_epoch"]                 # (T,)
    labels_epoch = dat["labels_epoch"].astype(int)  # (T,)
    dt = float(dat["dt"].item())
    reunion_abs = float(dat["reunion_abs"].item())
    epoch_start = float(dat["epoch_start"].item())
    epoch_end = float(dat["epoch_end"].item())
    return betas, spikes_epoch, t_epoch, labels_epoch, dt, reunion_abs, epoch_start, epoch_end


# =============== 1. 构造社交需求指数 Need_z(t) = (β2 - β1)^T spike(t) ===============
def compute_social_need(betas, spikes_epoch):
    """
    betas: (3, N)  -> 取 state2 - state1 作为“社交轴”
    spikes_epoch: (T, N)
    返回：
        need_raw: (T,)
        need_z  : (T,)  z-score 后
        w       : (N,)  轴向量
    """
    beta2 = betas[2]
    beta1 = betas[1]
    w = beta2 - beta1  # “touch vs no-touch” 轴

    need_raw = spikes_epoch @ w  # 逐帧投影
    need_z = zscore(need_raw)    # 在时间维度上 z-score

    return need_raw, need_z, w


# =============== 2. 找每次触摸开始 index，用于对齐 PSTH ===============
def find_touch_onsets(labels_epoch):
    """
    labels_epoch: (T,), 0/1/2
    返回所有 state2 段的起点 index 列表
    """
    onsets = []
    for i in range(len(labels_epoch)):
        if labels_epoch[i] == 2:
            if i == 0 or labels_epoch[i-1] != 2:
                onsets.append(i)
    return onsets


def extract_need_trials(need_z, labels_epoch, t_epoch, touch_onsets,
                        dt, pre_s=5.0, post_s=10.0):
    """
    对齐每次触摸开始（touch_onsets），取 need_z 的窗口：
      [touch_start - pre_s, touch_start + post_s]
    只要窗口在 epoch 内就保留。
    """
    T = len(need_z)
    frames_pre = int(round(pre_s / dt))
    frames_post = int(round(post_s / dt))
    win_len = frames_pre + frames_post + 1

    trials = []
    for i0 in touch_onsets:
        s = i0 - frames_pre
        e = i0 + frames_post
        if s < 0 or e >= T:
            continue
        seg = need_z[s:e+1]
        if len(seg) == win_len:
            trials.append(seg)

    if len(trials) == 0:
        return None, None

    trials = np.stack(trials, axis=0)  # (n_trials, win_len)
    t_rel = (np.arange(win_len) - frames_pre) * dt
    return trials, t_rel


# =============== 3. 画图 ===============

def plot_need_timecourse(need_z, t_epoch, labels_epoch, reunion_abs, out_dir):
    """
    时间进程图：
      - Need_z(t)  vs  time-from-reunion
      - 背景用颜色条表示三状态
    """
    t_rel = t_epoch - reunion_abs

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 4))

    # Need(t)
    ax1.plot(t_rel, need_z, linewidth=0.8, color="tab:purple", label="Social need index (z)")
    ax1.set_xlabel("相对重聚时间 (s)")
    ax1.set_ylabel("社交需求指数 Need_z(t)")
    ax1.grid(alpha=0.3)

    # 用第二个坐标轴画状态条（0/1/2）
    ax2 = ax1.twinx()
    ax2.plot(t_rel, labels_epoch, drawstyle="steps-mid", linewidth=0.6, color="gray", alpha=0.5)
    ax2.set_ylabel("状态标签 (0/1/2)", fontsize=8)
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(["alone", "no-touch", "touch"])

    ax1.set_title("社交需求指数随时间变化（背景为三状态）")
    fig.tight_layout()

    out_path = os.path.join(out_dir, "social_need_timecourse.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"✓ Saved: {out_path}")
    return out_path


def plot_need_touch_psth(need_z, labels_epoch, t_epoch, dt,
                         reunion_abs, out_dir,
                         pre_s=5.0, post_s=10.0):
    """
    对齐触摸开始的 Need_z PSTH：
      X: 相对触摸开始时间
      Y: 平均 Need_z
    """
    touch_onsets = find_touch_onsets(labels_epoch)
    trials, t_rel = extract_need_trials(
        need_z, labels_epoch, t_epoch, touch_onsets,
        dt, pre_s=pre_s, post_s=post_s
    )
    if trials is None:
        print("⚠️ 没有足够完整的触摸 trial 用于 PSTH。")
        return None

    mean = trials.mean(axis=0)
    sem = trials.std(axis=0) / np.sqrt(trials.shape[0])

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(t_rel, mean, color="tab:red", label=f"平均 Need_z (n={trials.shape[0]} bouts)")
    ax.fill_between(t_rel, mean - sem, mean + sem, color="tab:red", alpha=0.2)

    ax.axvline(0, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel("相对触摸开始时间 (s)")
    ax.set_ylabel("社交需求指数 Need_z(t)")
    ax.set_title(f"对齐每次触摸开始的社交需求 PSTH\n(左侧 {pre_s:.0f}s = 预触摸窗口)")

    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(out_dir, "social_need_touch_onset_PSTH.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"✓ Saved: {out_path}")
    return out_path


def compute_statewise_need(need_z, labels_epoch, t_epoch, dt,
                           pre_touch_window=5.0):
    """
    计算不同条件下 Need_z 的分布：
      - state0 全部
      - state1 中的“普通 no-touch”
      - state1 中的“预触摸窗口”（触摸前 pre_touch_window 秒）
      - state2 全部
    返回 dict：[name -> array]
    """
    res = {}

    mask0 = labels_epoch == 0
    mask1 = labels_epoch == 1
    mask2 = labels_epoch == 2

    res["state0_alone"] = need_z[mask0]
    res["state2_touch"] = need_z[mask2]

    # 找所有触摸 onset，构建预触摸窗口 mask
    T = len(need_z)
    frames_pre = int(round(pre_touch_window / dt))
    pre_mask = np.zeros_like(labels_epoch, dtype=bool)

    touch_onsets = find_touch_onsets(labels_epoch)
    for i0 in touch_onsets:
        s = i0 - frames_pre
        e = i0    # 不含 i0
        if s < 0:
            continue
        pre_mask[s:e] = True

    # 预触摸窗口：必须在 state1 内
    pretouch_mask = pre_mask & (labels_epoch == 1)
    normal_notouch_mask = (labels_epoch == 1) & (~pretouch_mask)

    res["state1_notouch_normal"] = need_z[normal_notouch_mask]
    res["state1_notouch_pretouch"] = need_z[pretouch_mask]

    return res


def plot_need_state_boxplot(need_dict, out_dir):
    """
    做一个简单的 boxplot / violin（这里用 boxplot），比较四个条件下 Need_z 分布。
    """
    labels = [
        "state0_alone",
        "state1_notouch_normal",
        "state1_notouch_pretouch",
        "state2_touch"
    ]
    data = [need_dict[k] for k in labels]

    plt.figure(figsize=(10, 4))
    bp = plt.boxplot(data, labels=[
        "0:独处", "1:无触摸(非预触摸)", "1:预触摸窗口", "2:触摸期"
    ], showfliers=False)

    plt.ylabel("社交需求指数 Need_z")
    plt.title("不同状态/时间窗口下的社交需求指数分布")
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "social_need_state_boxplot.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✓ Saved: {out_path}")
    return out_path


# =============== README（中文说明） ===============
def write_readme(need_out_dir, time_fig, psth_fig, box_fig,
                 pre_touch_window):
    lines = []
    lines.append("# 基于三状态 GLM 的社交需求指数分析")
    lines.append("")
    lines.append("本文件夹由脚本 `社交需求指数_三态GLM.py` 自动生成，用于从 mPFC 群体活动中构造并可视化一个“社交需求指数”。")
    lines.append("")
    lines.append("## 一、指数定义")
    lines.append("")
    lines.append("我们基于三状态 GLM 的权重向量：")
    lines.append("- β_state0：重聚前 200s 独处期（state0）")
    lines.append("- β_state1：重聚后但无触摸（state1）")
    lines.append("- β_state2：社交触摸期（state2）")
    lines.append("")
    lines.append("构造一条“触摸 vs 无触摸”轴：")
    lines.append("  w = β_state2 − β_state1")
    lines.append("")
    lines.append("然后对每一帧的群体活动 spikes(t) 做投影：")
    lines.append("  Need_raw(t) = w^T · spikes(t)")
    lines.append("再在时间维度上做 z-score，得到：")
    lines.append("  Need_z(t) = zscore(Need_raw(t))")
    lines.append("")
    lines.append("直观解释：")
    lines.append("- Need_z 高 → 当下神经活动更像“触摸状态”（社交获得/接触已发生）。")
    lines.append("- Need_z 低 → 更像“有对方但暂时没有触摸/保持距离”的状态。")
    lines.append("")
    lines.append("## 二、图像说明")
    lines.append("")
    lines.append(f"### 1）{os.path.basename(time_fig)}")
    lines.append("整段扩展 epoch 上的社交需求指数随时间变化：")
    lines.append("- 纵轴：Need_z(t)。")
    lines.append("- 横轴：相对重聚时间（秒）。")
    lines.append("- 紫色曲线：社交需求指数。")
    lines.append("- 灰色阶梯线：三状态标签（0=独处，1=重聚无触摸，2=触摸）。")
    lines.append("")
    lines.append(f"### 2）{os.path.basename(psth_fig)}")
    lines.append("对齐每次“社交触摸开始”时间点的 Need_z PSTH：")
    lines.append("- X 轴：相对触摸开始的时间（秒，0 点为“社交开始”）。")
    lines.append("- Y 轴：Need_z 的平均 ± SEM。")
    lines.append(f"- 触摸开始前 {pre_touch_window:.0f} 秒被视为“预触摸窗口”，可用于观察是否存在需求的 ramping。")
    lines.append("")
    lines.append(f"### 3）{os.path.basename(box_fig)}")
    lines.append("比较不同状态/时间窗口下社交需求指数的分布：")
    lines.append("- 0:独处期（state0_alone）。")
    lines.append("- 1:无触摸(非预触摸)（state1_notouch_normal）。")
    lines.append(f"- 1:预触摸窗口（每次触摸开始前 {pre_touch_window:.0f}s 内，state1_notouch_pretouch）。")
    lines.append("- 2:触摸期（state2_touch）。")
    lines.append("")
    lines.append("如果在数据中观察到：")
    lines.append("- Need_z 在独处期较低；")
    lines.append("- 重聚后无触摸期中等；")
    lines.append("- 预触摸窗口中 Need_z 逐渐上升；")
    lines.append("- 真正触摸期达到峰值或平台；")
    lines.append("那么这个 Need_z(t) 可以自然地被解释为一个“社交需求/社交驱动力指数”。")
    lines.append("")
    lines.append("## 三、数据文件")
    lines.append("")
    lines.append("- `social_need_timecourse.png`：时间进程图。")
    lines.append("- `social_need_touch_onset_PSTH.png`：对齐触摸起点的 Need_z PSTH。")
    lines.append("- `social_need_state_boxplot.png`：不同状态/时间窗口下的 Need_z 分布比较。")
    lines.append("- `社交需求指数_三态GLM.py`：生成以上所有结果的脚本（如有保存）。")
    lines.append("")
    readme_path = os.path.join(need_out_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"✓ Saved: {readme_path}")
    return readme_path


# =============== 主函数 ===============
def main():
    print("\n" + "=" * 70)
    print("根据三状态 GLM 构建社交需求指数")
    print("=" * 70)

    betas, spikes_epoch, t_epoch, labels_epoch, dt, reunion_abs, epoch_start, epoch_end = \
        load_three_state_npz(three_state_npz)
    print(f"✓ Loaded three_state_glm_outputs: T={len(t_epoch)}, N={spikes_epoch.shape[1]}, dt≈{dt:.3f}s")

    # 1) 计算社交需求指数
    need_raw, need_z, w = compute_social_need(betas, spikes_epoch)
    print("✓ Computed social need index (raw + z-score)")

    # 2) 画时间进程图
    time_fig = plot_need_timecourse(need_z, t_epoch, labels_epoch, reunion_abs, need_out_dir)

    # 3) 对齐触摸开始的 PSTH
    psth_fig = plot_need_touch_psth(
        need_z, labels_epoch, t_epoch, dt, reunion_abs,
        need_out_dir, pre_s=PRE_TOUCH_WINDOW, post_s=10.0
    )

    # 4) 不同状态/窗口下的 Need_z 分布
    need_dict = compute_statewise_need(
        need_z, labels_epoch, t_epoch, dt,
        pre_touch_window=PRE_TOUCH_WINDOW
    )
    box_fig = plot_need_state_boxplot(need_dict, need_out_dir)

    # 5) 写 README
    write_readme(need_out_dir, time_fig, psth_fig, box_fig, PRE_TOUCH_WINDOW)

    print("\n✅ 社交需求指数分析完成。")
    print(f"  输出目录: {need_out_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
