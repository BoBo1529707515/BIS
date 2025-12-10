# three_state_top3_and_pretouch_plots.py
# -*- coding: utf-8 -*-

"""
基于你现有的 Excel 读取格式：
  第0列：相对时间
  第1列：事件文本（含“重聚期开始”、“社交开始”、“社交结束”）

目标：
  1) 将时间扩展到重聚前 200s
  2) 构造三状态：
     state0 = 重聚前 alone
     state1 = 重聚后无触摸
     state2 = 社交触摸
  3) 1s bin 多分类 GLM (multinomial)
  4) 找每个状态 Top3 代表神经元
  5) 绘图：
     - state0_top3_activity.png
     - state1_top3_activity.png
     - state2_top3_activity.png
     - reunion_pretouch_timeline.png
  6) 新建输出文件夹并写 README.md
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore

# ----------------- Matplotlib 中文支持（可选） -----------------
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ======================= 路径 & 参数（按你当前工程写死） =======================
base_out_dir = r"F:\工作文件\RA\python\吸引子\figs"
data_path = r"F:\工作文件\RA\python\吸引子\Mouse1_for_ssm.npz"
beh_path = r"F:\工作文件\RA\数据集\时间戳\FVB_Panneuronal_Reunion_Isolation_Day3_Mouse#1.xlsx"

reunion_abs = 903.0     # 与你 HMM 脚本一致
PRE_BEFORE = 200.0      # 重聚前扩展 200s
EXTRA_AFTER = 60.0      # 最后一次社交结束后 buffer
BIN_SIZE = 1.0          # 1s bin

# “预触摸窗口”定义：每次社交开始前多少秒
PRE_TOUCH_WINDOW = 5.0

# 每个状态要画的代表神经元数
TOP_N_PER_STATE = 3

# 输出文件夹名（会自动新建）
REPORT_FOLDER_NAME = "three_state_pre200_top3_report"
# ============================================================================


# ======================= 数据加载 & 预处理 =======================
def load_neural_data(npz_path):
    dat = np.load(npz_path, allow_pickle=True)
    Y = np.asarray(dat["Y"])     # (T, N)
    t = np.asarray(dat["t"])     # (T,)
    dt = float(dat["dt"].item())

    # 轻度预处理：z-score + clip + 平滑
    Y_z = (Y - Y.mean(axis=0)) / (Y.std(axis=0) + 1e-6)
    Y_z = np.clip(Y_z, -3, 3)
    Y_z = gaussian_filter1d(Y_z, sigma=1, axis=0)

    print(f"✓ Loaded neural: T={len(t)}, N={Y.shape[1]}, dt≈{dt:.3f}s")
    return Y_z, t, dt


def calcium_to_spikes(Y):
    # 与你现有三状态脚本一致：直接 zscore 当 spike-like
    spikes = zscore(Y, axis=0)
    return spikes


# ======================= 行为读取（模仿你 HMM 脚本） =======================
def load_behavior_like_hmm_script(beh_path, reunion_abs):
    beh_df = pd.read_excel(beh_path, header=None).dropna(how="all")
    beh_df[0] = pd.to_numeric(beh_df[0], errors="coerce")
    beh_df[1] = beh_df[1].astype(str).str.strip().str.lower()

    reunion_rows = beh_df[beh_df[1].str.contains("重聚期开始", na=False)]
    if len(reunion_rows) == 0:
        print("⚠️ Excel 中未找到 '重聚期开始' 行，默认 reunion_rel=0")
        reunion_rel = 0.0
    else:
        reunion_rel = float(reunion_rows.iloc[0, 0])

    starts = beh_df[beh_df[1].str.contains("社交开始", na=False)][0].values
    ends = beh_df[beh_df[1].str.contains("社交结束", na=False)][0].values

    social_intervals = [
        (reunion_abs + (s - reunion_rel), reunion_abs + (e - reunion_rel))
        for s, e in zip(starts, ends)
    ]

    print(f"✓ Found {len(social_intervals)} social bouts")
    return social_intervals, reunion_rel


# ======================= 三状态标签构造 =======================
def build_three_state_labels(t, social_intervals, reunion_abs):
    labels = np.zeros_like(t, dtype=int)  # 0=pre-alone 默认

    labels[t >= reunion_abs] = 1          # 1=reunion no-touch（先粗标）

    for (start, end) in social_intervals:
        labels[(t >= start) & (t <= end)] = 2  # 2=touch

    return labels


def extract_extended_epoch(spikes, t, labels, social_intervals,
                           reunion_abs, pre_before=200.0, extra_after=60.0):
    t_min = float(t.min())
    t_max = float(t.max())

    epoch_start = max(t_min, reunion_abs - pre_before)

    if len(social_intervals) > 0:
        last_end = max(e for (_, e) in social_intervals)
        epoch_end = min(t_max, last_end + extra_after)
    else:
        epoch_end = min(t_max, reunion_abs + extra_after)

    mask = (t >= epoch_start) & (t <= epoch_end)

    spikes_epoch = spikes[mask]
    t_epoch = t[mask]
    labels_epoch = labels[mask]

    print(f"✓ Epoch: [{epoch_start:.1f}, {epoch_end:.1f}] s "
          f"(duration≈{t_epoch[-1]-t_epoch[0]:.1f}s, frames={len(t_epoch)})")
    return spikes_epoch, t_epoch, labels_epoch, epoch_start, epoch_end


# ======================= 1s bin 多分类 GLM =======================
def bin_multiclass(spikes_epoch, t_epoch, labels_epoch, bin_size=1.0):
    dt = np.median(np.diff(t_epoch))
    frames_per_bin = max(1, int(round(bin_size / dt)))

    T = len(t_epoch)
    n_bins = T // frames_per_bin

    X_bins, y_bins, t_bins = [], [], []

    for b in range(n_bins):
        s = b * frames_per_bin
        e = s + frames_per_bin
        w_spk = spikes_epoch[s:e]
        w_lab = labels_epoch[s:e]

        X_bins.append(w_spk.mean(axis=0))
        counts = np.bincount(w_lab, minlength=3)
        y_bins.append(int(np.argmax(counts)))
        t_bins.append(t_epoch[s:e].mean())

    X_bins = np.vstack(X_bins)
    y_bins = np.array(y_bins)
    t_bins = np.array(t_bins)

    print("\nBinning for 3-state GLM:")
    print(f"  dt≈{dt:.3f}s, frames/bin={frames_per_bin}, n_bins={n_bins}")
    for k in [0, 1, 2]:
        print(f"  state{k} fraction={float((y_bins==k).mean()):.3f}")

    return X_bins, y_bins, t_bins


def fit_three_state_glm(X_bins, y_bins):
    clf = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=5000
        # multi_class 参数已在新版 sklearn 被弃用，保持默认即可
    )
    clf.fit(X_bins, y_bins)

    y_hat = clf.predict(X_bins)
    acc = accuracy_score(y_bins, y_hat)
    cm = confusion_matrix(y_bins, y_hat, labels=[0, 1, 2])

    print(f"\n✓ Training accuracy (3-state): {acc:.3f}")
    print("✓ Confusion matrix (rows=true, cols=pred, order=[0,1,2]):")
    print(cm)

    return clf.coef_, clf.intercept_, clf


# ======================= 选 Top 神经元 =======================
def top_neurons_for_state(betas, state_k, top_n=3):
    w = betas[state_k]
    idx = np.argsort(np.abs(w))[::-1][:top_n]
    return idx, w[idx]


# ======================= 绘图辅助 =======================
def get_touch_spans(t_epoch, labels_epoch):
    """返回触摸(state2)段的 (start, end) 列表，用于背景高亮。"""
    spans = []
    in_seg = False
    seg_start = None

    for i in range(len(t_epoch)):
        is_touch = (labels_epoch[i] == 2)
        if is_touch and not in_seg:
            in_seg = True
            seg_start = t_epoch[i]
        elif (not is_touch) and in_seg:
            in_seg = False
            seg_end = t_epoch[i]
            spans.append((seg_start, seg_end))

    if in_seg:
        spans.append((seg_start, t_epoch[-1]))

    return spans


def shade_touch_background(ax, touch_spans, reunion_abs):
    """在活动图上把触摸区间用浅色背景标出来（x 轴用相对重聚时间）。"""
    for s, e in touch_spans:
        ax.axvspan(s - reunion_abs, e - reunion_abs, alpha=0.12)


def plot_state_top3_activity(spikes_epoch, t_epoch, labels_epoch,
                             betas, state_k, out_dir, reunion_abs):
    """
    画某个状态的 Top3 代表神经元活动图。
    """
    top_idx, top_w = top_neurons_for_state(betas, state_k, TOP_N_PER_STATE)
    touch_spans = get_touch_spans(t_epoch, labels_epoch)

    title_map = {
        0: "State 0 (pre-alone, -200s to reunion)",
        1: "State 1 (reunion, no-touch)",
        2: "State 2 (social touch)"
    }

    fig, axes = plt.subplots(TOP_N_PER_STATE, 1, figsize=(12, 2.6 * TOP_N_PER_STATE), sharex=True)

    if TOP_N_PER_STATE == 1:
        axes = [axes]

    t_rel = t_epoch - reunion_abs

    for ax, ni, wi in zip(axes, top_idx, top_w):
        trace = spikes_epoch[:, ni]
        ax.plot(t_rel, trace, linewidth=0.8)
        shade_touch_background(ax, touch_spans, reunion_abs)

        ax.set_ylabel(f"Cell {int(ni)}\nβ={float(wi):+.3f}", fontsize=9)
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("Time from reunion (s)")
    fig.suptitle(f"{title_map.get(state_k, f'State {state_k}')} - Top {TOP_N_PER_STATE} representative neurons",
                 y=1.02)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"state{state_k}_top{TOP_N_PER_STATE}_activity.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Saved: {out_path}")
    return top_idx, top_w, out_path


def plot_reunion_pretouch_timeline(t_epoch, labels_epoch, social_intervals,
                                   out_dir, reunion_abs, pretouch_window=5.0):
    """
    “重聚-预触摸发生时间图”
    - 上：三状态标签随时间（相对重聚）
    - 竖线：每次社交开始
    - 阴影：每次社交开始前 pretouch_window 秒（预触摸窗口）
    """
    t_rel = t_epoch - reunion_abs

    fig, ax = plt.subplots(1, 1, figsize=(12, 3.2))

    # 三状态 step 曲线
    ax.plot(t_rel, labels_epoch, drawstyle="steps-mid", linewidth=1.0)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["pre-alone", "reunion no-touch", "touch"])

    # 标记每次社交开始 + 预触摸窗
    for (start, end) in social_intervals:
        s_rel = start - reunion_abs
        ax.axvline(s_rel, linestyle="--", linewidth=0.8, alpha=0.6)
        # 预触摸窗口阴影
        ax.axvspan(s_rel - pretouch_window, s_rel, alpha=0.12)

    ax.set_xlabel("Time from reunion (s)")
    ax.set_ylabel("3-state label")
    ax.set_title(f"Reunion timeline with pre-touch windows ({pretouch_window:.0f}s before each touch onset)")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "reunion_pretouch_timeline.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"✓ Saved: {out_path}")
    return out_path


def write_readme(out_dir, top_info):
    """
    写 README.md，详细描述输出内容。
    top_info: dict，包含每个 state 的 top_idx/top_w/fig_path
    """
    lines = []
    lines.append("# Three-state pre200 report (Top neurons & pre-touch)")
    lines.append("")
    lines.append("This folder was generated by `three_state_top3_and_pretouch_plots.py`.")
    lines.append("")
    lines.append("## States definition")
    lines.append("- **State 0**: 200s before reunion (mouse alone, no other mouse, no touch).")
    lines.append("- **State 1**: After reunion but **outside** any social-touch bouts (other mouse present, no touch).")
    lines.append("- **State 2**: Social-touch bouts (as labeled by Excel “社交开始/社交结束”).")
    lines.append("")
    lines.append("Neural features were z-scored and lightly smoothed, then averaged in 1s bins for a multinomial logistic regression.")
    lines.append("The model provides a weight vector for each state; neurons with larger |β| contribute more strongly to classifying that state.")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for k in [0, 1, 2]:
        idxs = top_info[k]["idx"]
        ws = top_info[k]["w"]
        figp = os.path.basename(top_info[k]["fig"])
        lines.append(f"### {figp}")
        lines.append(f"Top {len(idxs)} representative neurons for **State {k}** (ranked by |β_state{k}|).")
        lines.append("Shown as activity traces over the extended epoch (time relative to reunion).")
        lines.append("Background light shading indicates **touch periods (State 2)** for reference.")
        lines.append("")
        lines.append("Selected neurons:")
        for i, w in zip(idxs, ws):
            lines.append(f"- neuron {int(i)}  β_state{k} = {float(w):+.4f}")
        lines.append("")

    lines.append("### reunion_pretouch_timeline.png")
    lines.append("A compact timeline of the extended epoch with three-state labels.")
    lines.append("- Dashed vertical lines: **touch onset** (each “社交开始”).")
    lines.append("- Light shaded blocks: **pre-touch windows** defined as the *5 seconds before each touch onset*.")
    lines.append("This helps visualize where state1 (no-touch) transitions into state2 (touch) around each bout.")
    lines.append("")
    lines.append("## Files")
    lines.append("- `three_state_glm_outputs.npz`: saved model weights, epoch data, and labels for reproducibility.")
    lines.append("- `README.md`: this description.")

    readme_path = os.path.join(out_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✓ Saved: {readme_path}")


# ======================= 主流程 =======================
def main():
    print("\n" + "=" * 70)
    print("3-STATE TOP3 NEURON ACTIVITY + PRE-TOUCH TIMELINE")
    print("=" * 70)

    out_dir = os.path.join(base_out_dir, REPORT_FOLDER_NAME)
    os.makedirs(out_dir, exist_ok=True)
    print(f"✓ New report folder: {out_dir}")

    # 1) load neural
    Y, t, dt = load_neural_data(data_path)
    spikes = calcium_to_spikes(Y)

    # 2) load behavior (same style as your HMM script)
    social_intervals, reunion_rel = load_behavior_like_hmm_script(beh_path, reunion_abs)

    # 3) build 3-state labels
    labels_full = build_three_state_labels(t, social_intervals, reunion_abs)

    # 4) extract extended epoch
    spikes_epoch, t_epoch, labels_epoch, epoch_start, epoch_end = extract_extended_epoch(
        spikes, t, labels_full, social_intervals,
        reunion_abs, pre_before=PRE_BEFORE, extra_after=EXTRA_AFTER
    )

    # 5) bin & fit multinomial GLM
    X_bins, y_bins, t_bins = bin_multiclass(spikes_epoch, t_epoch, labels_epoch, bin_size=BIN_SIZE)
    betas, intercepts, clf = fit_three_state_glm(X_bins, y_bins)

    # 6) plot top3 per state
    top_info = {}
    for k in [0, 1, 2]:
        idx, w, figp = plot_state_top3_activity(
            spikes_epoch, t_epoch, labels_epoch,
            betas, k, out_dir, reunion_abs
        )
        top_info[k] = {"idx": idx, "w": w, "fig": figp}

    # 7) plot pre-touch timeline
    pretouch_fig = plot_reunion_pretouch_timeline(
        t_epoch, labels_epoch, social_intervals,
        out_dir, reunion_abs, pretouch_window=PRE_TOUCH_WINDOW
    )

    # 8) save outputs package
    np.savez(
        os.path.join(out_dir, "three_state_glm_outputs.npz"),
        betas=betas,
        intercepts=intercepts,
        spikes_epoch=spikes_epoch,
        t_epoch=t_epoch,
        labels_epoch=labels_epoch,
        t_bins=t_bins,
        y_bins=y_bins,
        dt=dt,
        reunion_abs=reunion_abs,
        epoch_start=epoch_start,
        epoch_end=epoch_end,
        social_intervals=np.array(social_intervals, dtype=float),
        pretouch_window=PRE_TOUCH_WINDOW
    )
    print(f"✓ Saved: {os.path.join(out_dir, 'three_state_glm_outputs.npz')}")

    # 9) write README
    write_readme(out_dir, top_info)

    print("\n✅ Done. Figures and README are in:")
    print(out_dir)
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
