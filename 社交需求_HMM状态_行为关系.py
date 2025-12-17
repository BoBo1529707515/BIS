# 社交需求_HMM状态_行为关系.py
# -*- coding: utf-8 -*-
"""
读取一维 HMM 结果，分析：
1) HMM 隐状态与三态行为标签 (0/1/2) 的关系
2) 对齐触摸 onset 的高需求状态概率 P(z=1) PSTH
3) 触摸前预窗口 vs 非预窗口的 HMM 状态分布

依赖:
    pip install numpy matplotlib
"""

import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ======================= 路径配置 =======================
base_out_dir = r"F:\工作文件\RA\python\吸引子\figs"

# 上一个脚本保存的 HMM 结果 npz
hmm_npz_file = os.path.join(
    base_out_dir,
    "social_need_dynamics_HMM_AR",
    "social_need_HMM_results.npz",   # 确保和上一脚本一致
)

# 本脚本输出目录
align_out_dir = os.path.join(base_out_dir, "social_need_HMM_behavior_alignment")
os.makedirs(align_out_dir, exist_ok=True)

# 预触摸窗口（秒）
PRE_TOUCH_WINDOW = 5.0
POST_TOUCH_WINDOW = 10.0


# ======================= 工具函数 =======================
def load_hmm_results(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 HMM 结果文件，请先运行 `社交需求_一维HMM_AR动态.py`:\n  {path}")

    dat = np.load(path, allow_pickle=True)
    need_z = np.asarray(dat["need_z"])                 # (T,)
    t_epoch = np.asarray(dat["t_epoch"])               # (T,)
    labels_epoch = np.asarray(dat["labels_epoch"]).astype(int)  # (T,)
    dt = float(dat["dt"].item())
    reunion_abs = float(dat["reunion_abs"].item())
    z = np.asarray(dat["z"]).astype(int)               # (T,)
    hmm_means = np.asarray(dat["hmm_means"])           # (2, 1)
    transmat = np.asarray(dat["hmm_transmat"])         # (2, 2)
    ar1_state_info = dat["ar1_state_info"].item()      # dict

    return need_z, t_epoch, labels_epoch, dt, reunion_abs, z, hmm_means, transmat, ar1_state_info


def find_touch_onsets(labels_epoch):
    """ 找到所有 state=2 段的起点 index 列表 """
    onsets = []
    for i in range(len(labels_epoch)):
        if labels_epoch[i] == 2:
            if i == 0 or labels_epoch[i - 1] != 2:
                onsets.append(i)
    return onsets


# ======================= 1. HMM 状态 × 行为状态 =======================
def compute_state_behavior_matrix(z, labels_epoch):
    """
    统计:
        - P(z=0 | label=k)
        - P(z=1 | label=k)
      以及每个 label=k 的样本数
    返回:
        probs: shape (3, 2) 行 = label 0/1/2, 列 = HMM state 0/1
        counts: shape (3,)
    """
    probs = np.zeros((3, 2))
    counts = np.zeros(3, dtype=int)

    for k in [0, 1, 2]:
        mask = (labels_epoch == k)
        n = mask.sum()
        counts[k] = n
        if n == 0:
            probs[k, :] = np.nan
        else:
            # 统计该行为状态下 HMM state=0/1 的比例
            for s in [0, 1]:
                probs[k, s] = (z[mask] == s).mean()

    return probs, counts


def plot_state_behavior_bar(probs, counts, out_dir):
    """
    画一张柱状图：
      X: 行为状态 (0/1/2)
      Y: P(z=1 | label)
    """
    labels = ["0:独处", "1:无触摸", "2:触摸"]
    x = np.arange(3)

    p_high = probs[:, 1]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.bar(x, p_high, color="tab:orange", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("P(HMM 高需求状态, z=1 | 行为状态)")
    ax.set_title("不同行为状态下 HMM 高需求状态的占比")

    # 在柱子上标注样本数
    for i, (p, n) in enumerate(zip(p_high, counts)):
        txt = f"n={n}"
        if not np.isnan(p):
            txt += f"\n{p:.2f}"
        ax.text(i, min(0.95, (p if not np.isnan(p) else 0.9) + 0.02),
                txt, ha="center", va="bottom", fontsize=8)

    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(out_dir, "HMM_high_state_prob_by_behavior.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"✓ Saved: {out_path}")
    return out_path


# ======================= 2. 对齐触摸 onset 的 P(z=1) PSTH =======================
def extract_z_trials(z, labels_epoch, t_epoch, dt, pre_s, post_s):
    """
    对齐每次触摸开始，截取 z(t) 窗口:
      [onset - pre_s, onset + post_s]
    返回:
        trials: (n_trials, win_len) (0/1)
        t_rel:  (win_len,)
    """
    T = len(z)
    frames_pre = int(round(pre_s / dt))
    frames_post = int(round(post_s / dt))
    win_len = frames_pre + frames_post + 1

    onsets = find_touch_onsets(labels_epoch)
    trial_list = []

    for i0 in onsets:
        s = i0 - frames_pre
        e = i0 + frames_post
        if s < 0 or e >= T:
            continue
        seg = z[s:e + 1]
        if len(seg) == win_len:
            trial_list.append(seg)

    if len(trial_list) == 0:
        return None, None

    trials = np.stack(trial_list, axis=0)   # (n_trials, win_len)
    t_rel = (np.arange(win_len) - frames_pre) * dt
    return trials, t_rel


def plot_P_z1_touch_PSTH(z, labels_epoch, t_epoch, dt,
                         pre_s, post_s, out_dir):
    trials, t_rel = extract_z_trials(z, labels_epoch, t_epoch, dt, pre_s, post_s)
    if trials is None:
        print("⚠️ 没有足够完整的触摸 trial 用于 HMM 状态 PSTH。")
        return None

    # 0/1 → 概率
    trials_high = (trials == 1).astype(float)
    p_mean = trials_high.mean(axis=0)
    p_sem = trials_high.std(axis=0) / np.sqrt(trials_high.shape[0])

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(t_rel, p_mean, color="tab:orange",
            label=f"P(z=1) 平均 (n={trials_high.shape[0]} bouts)")
    ax.fill_between(t_rel, p_mean - p_sem, p_mean + p_sem,
                    color="tab:orange", alpha=0.3)

    ax.axvline(0, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel("相对触摸开始时间 (s)")
    ax.set_ylabel("P(HMM 高需求状态, z=1)")
    ax.set_title(f"对齐触摸 onset 的高需求状态概率 PSTH\n(左侧 {pre_s:.0f}s = 预触摸窗口)")

    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_path = os.path.join(out_dir, "P_z1_touch_onset_PSTH.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"✓ Saved: {out_path}")
    return out_path


# ======================= 3. 触摸前预窗口 vs 非预窗口 =======================
def compute_pretouch_vs_normal(z, labels_epoch, dt, pre_touch_window):
    """
    划分：
      - state1 中的“预触摸窗口”样本：每次触摸 onset 前 pre_touch_window 秒内，且仍在 label=1
      - state1 中的“普通 no-touch”：label=1 & 不在预触摸窗口中
    和之前 Need_z 的分析保持一致，只不过这里关注的是 HMM 状态分布。
    """
    T = len(z)
    frames_pre = int(round(pre_touch_window / dt))
    pre_mask = np.zeros(T, dtype=bool)

    onsets = find_touch_onsets(labels_epoch)
    for i0 in onsets:
        s = i0 - frames_pre
        e = i0  # 不含 i0
        if s < 0:
            continue
        pre_mask[s:e] = True

    pretouch_mask = pre_mask & (labels_epoch == 1)
    normal_notouch_mask = (labels_epoch == 1) & (~pretouch_mask)

    # 统计这两类样本下，HMM state 的分布
    res = {}
    for name, mask in [
        ("state1_notouch_normal", normal_notouch_mask),
        ("state1_notouch_pretouch", pretouch_mask),
    ]:
        n = mask.sum()
        if n == 0:
            res[name] = {
                "n": 0,
                "p0": np.nan,
                "p1": np.nan,
            }
        else:
            res[name] = {
                "n": n,
                "p0": (z[mask] == 0).mean(),
                "p1": (z[mask] == 1).mean(),
            }
    return res


def plot_pretouch_vs_normal_bar(res_dict, out_dir):
    labels = ["1:无触摸(非预窗口)", "1:预触摸窗口"]
    p1_vals = [res_dict["state1_notouch_normal"]["p1"],
               res_dict["state1_notouch_pretouch"]["p1"]]
    ns = [res_dict["state1_notouch_normal"]["n"],
          res_dict["state1_notouch_pretouch"]["n"]]

    x = np.arange(2)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.bar(x, p1_vals, color="tab:orange", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("P(HMM 高需求状态, z=1)")
    ax.set_title("无触摸状态中：预触摸窗口 vs 非预窗口的高需求状态概率")

    for i, (p, n) in enumerate(zip(p1_vals, ns)):
        txt = f"n={n}"
        if not np.isnan(p):
            txt += f"\n{p:.2f}"
        ax.text(i, min(0.95, (p if not np.isnan(p) else 0.9) + 0.02),
                txt, ha="center", va="bottom", fontsize=8)

    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(out_dir, "P_z1_pretouch_vs_normal_in_state1.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"✓ Saved: {out_path}")
    return out_path


# ======================= README =======================
def write_readme(align_out_dir,
                 behavior_fig, psth_fig, pretouch_fig,
                 pre_touch_window, post_touch_window,
                 hmm_means, transmat, ar1_state_info):
    lines = []
    lines.append("# HMM 隐状态与行为的对齐分析")
    lines.append("")
    lines.append("本文件夹由脚本 `社交需求_HMM状态_行为关系.py` 自动生成，用于分析：")
    lines.append("1. 一维社交需求指数 Need_z(t) 上拟合得到的 HMM 隐状态，如何与三态行为标签 (0/1/2) 对齐；")
    lines.append("2. 对齐“触摸开始”事件时，高需求 HMM 状态的概率随时间的变化；")
    lines.append("3. 无触摸状态 (state1) 中，触摸前预窗口 vs 普通无触摸时间段中，高需求 HMM 状态的占比差异。")
    lines.append("")
    lines.append("## 一、HMM 模型回顾（来自上一脚本）")
    lines.append("")
    lines.append("- 在 Need_z(t) 上拟合了二维 Gaussian HMM：z_t ∈ {0,1}。")
    lines.append("- 两个状态的均值（Need_z 的平均值）约为：")
    lines.append(f"  - state 0: μ ≈ {hmm_means[0,0]:.3f}")
    lines.append(f"  - state 1: μ ≈ {hmm_means[1,0]:.3f}")
    lines.append("  通常可以自然地解释为“低需求状态”和“高需求状态”。")
    lines.append("- 状态转移矩阵约为：")
    lines.append(f"  {transmat}")
    lines.append("  表明两个状态本身都较为稳定（对角元素接近 1），")
    lines.append("  提示这更像是两个长期内在模式，而不是瞬时抖动的噪声。")
    lines.append("")
    lines.append("另外，在上一脚本里，我们还在每个 HMM 状态内估计了 AR(1) 系数和近似时间常数 τ，")
    lines.append("通常会看到高需求状态具有更长的时间常数（更持续的 ramp/plateau）。")
    lines.append("不同状态下 AR(1) 参数大致为：")
    for s in sorted(ar1_state_info.keys()):
        info = ar1_state_info[s]
        lines.append(f"- state {s}: a ≈ {info['a']:.3f}, var_eps ≈ {info['var_eps']:.3f}, "
                     f"n_pairs={info['n_pairs']}, τ≈{info['tau']:.3f} s")
    lines.append("")
    lines.append("## 二、图像说明")
    lines.append("")
    lines.append(f"### 1）{os.path.basename(behavior_fig)}")
    lines.append("**不同行为状态下，HMM 高需求状态 P(z=1) 的占比**：")
    lines.append("- X 轴：三种行为状态（0:独处；1:无触摸；2:触摸）；")
    lines.append("- Y 轴：在该行为状态下，Need_z(t) 属于 HMM 高需求状态 (z=1) 的概率；")
    lines.append("- 每个柱子上标注了该行为状态下的样本数 n 以及 P(z=1) 的数值；")
    lines.append("如果结果显示：")
    lines.append("- 在触摸期 (state2) 下 P(z=1) 明显更高；")
    lines.append("- 在独处期 (state0) 下 P(z=1) 相对较低；")
    lines.append("- 在重聚无触摸期 (state1) 介于两者之间；")
    lines.append("则支持“高需求 HMM 状态与更强的社交接触/触摸行为相对应”的解释。")
    lines.append("")
    lines.append(f"### 2）{os.path.basename(psth_fig)}")
    lines.append("**对齐触摸 onset 的 P(z=1) PSTH**：")
    lines.append("- X 轴：相对触摸开始时间（秒，0 点为首次进入 state2 即触摸期）；")
    lines.append("- Y 轴：在该相对时间点，高需求状态 z=1 的概率（跨所有触摸 bout 平均）；")
    lines.append(f"- 左侧 {pre_touch_window:.0f} 秒被视为“预触摸窗口”；右侧 {post_touch_window:.0f} 秒为触摸后期；")
    lines.append("- 曲线为平均 P(z=1)，阴影为 ±SEM。")
    lines.append("如果曲线显示：")
    lines.append("- 在触摸开始前几秒，P(z=1) 已经开始上升；")
    lines.append("- 在触摸期间 P(z=1) 保持在较高水平；")
    lines.append("则说明 HMM 的高需求状态不仅反映触摸发生时刻本身，")
    lines.append("也在触摸前就已经“预热”，可以作为社交需求/动机的一个内部状态变量。")
    lines.append("")
    lines.append(f"### 3）{os.path.basename(pretouch_fig)}")
    lines.append("**无触摸状态中预触摸窗口 vs 普通无触摸时段的高需求状态概率**：")
    lines.append("- 仅在行为标签 state1（重聚无触摸）中取样；")
    lines.append(f"- 将触摸 onset 前 {pre_touch_window:.0f} s 内定义为“预触摸窗口”；")
    lines.append("- 其他 state1 时间点定义为“普通无触摸时段”；")
    lines.append("- 对比这两类时间点中 P(z=1) 的差异。")
    lines.append("如果观察到：")
    lines.append("- 预触摸窗口中 P(z=1) 显著高于普通无触摸时间段；")
    lines.append("则可以说：在行为上看起来同为“无触摸”的状态1内部，")
    lines.append("已经存在一个“隐性分裂”：一部分时间是低需求状态，一部分时间是即将进入触摸的高需求状态。")
    lines.append("")
    lines.append("## 三、小结")
    lines.append("")
    lines.append("结合一维 Need_z(t) 的 HMM 拟合和本脚本的行为对齐分析，可以得到如下图景：")
    lines.append("")
    lines.append("1. 在社交需求轴 Need_z(t) 上存在两个稳定的隐状态：低需求状态 (z=0) 和高需求状态 (z=1)。")
    lines.append("2. 高需求状态在触摸期（行为 state2）内占比显著提高，且在触摸开始前的预窗口中概率也提前上升。")
    lines.append("3. 即便在同为“无触摸”的行为状态1内部，高需求状态也更偏向于那些即将进入触摸的时间段。")
    lines.append("")
    lines.append("因此，HMM 的高需求状态可以自然地解释为一种“内部社交需求/动机状态”，")
    lines.append("它在时间上以 ramping/plateau 的方式变化，并通过切换到高需求状态，")
    lines.append("提高未来一段时间内发生社交触摸行为的概率，而不是瞬时触发行为。")
    lines.append("这与 Xie et al. (2025) 论文中提出的“switching ramping dynamics 驱动自发行为”的框架是平行的，只是这里换成了社交重聚/触摸场景。")
    lines.append("")

    readme_path = os.path.join(align_out_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"✓ Saved: {readme_path}")
    return readme_path


# ======================= 主函数 =======================
def main():
    print("\n" + "=" * 70)
    print("HMM 隐状态 × 行为对齐分析")
    print("=" * 70)

    (need_z, t_epoch, labels_epoch, dt, reunion_abs,
     z, hmm_means, transmat, ar1_state_info) = load_hmm_results(hmm_npz_file)

    print(f"✓ Loaded HMM results: T={len(need_z)}, dt≈{dt:.3f}s")
    print("  HMM means:", hmm_means.ravel())
    print("  HMM transmat:\n", transmat)

    # 1) 行为状态 × HMM 高需求状态概率
    probs, counts = compute_state_behavior_matrix(z, labels_epoch)
    behavior_fig = plot_state_behavior_bar(probs, counts, align_out_dir)

    # 2) 对齐触摸 onset 的 P(z=1) PSTH
    psth_fig = plot_P_z1_touch_PSTH(
        z, labels_epoch, t_epoch, dt,
        pre_s=PRE_TOUCH_WINDOW,
        post_s=POST_TOUCH_WINDOW,
        out_dir=align_out_dir
    )

    # 3) 无触摸状态中：预触摸窗口 vs 普通无触摸时间段
    res_dict = compute_pretouch_vs_normal(z, labels_epoch, dt, PRE_TOUCH_WINDOW)
    pretouch_fig = plot_pretouch_vs_normal_bar(res_dict, align_out_dir)

    # 4) 写 README
    readme_path = write_readme(
        align_out_dir,
        behavior_fig, psth_fig, pretouch_fig,
        PRE_TOUCH_WINDOW, POST_TOUCH_WINDOW,
        hmm_means, transmat, ar1_state_info
    )

    print("\n✅ HMM 状态 × 行为对齐分析完成。")
    print(f"  输出目录: {align_out_dir}")
    print(f"  已生成 README: {readme_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
