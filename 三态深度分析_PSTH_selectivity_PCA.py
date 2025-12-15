# 三态深度分析_PSTH_selectivity_PCA.py
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 路径配置 =====================
base_out_dir = r"F:\工作文件\RA\python\吸引子\figs"
three_state_npz = os.path.join(base_out_dir, "three_state_glm", "three_state_glm_outputs.npz")

# 新的分析输出文件夹
deep_out_dir = os.path.join(base_out_dir, "three_state_glm_deepdive")
os.makedirs(deep_out_dir, exist_ok=True)

# 要画 PSTH 的典型神经元
PSTH_NEURONS = [82, 112, 11, 84, 30, 2]

# PSTH 窗宽（相对“状态起点”时间）：前/后多少秒
PSTH_PRE_SEC = 10.0
PSTH_POST_SEC = 20.0

# selectivity 分类参数
SELECT_MIN_ABS_BETA = 0.3   # 最大 |β| 小于这个就算 weak
SELECT_MARGIN = 0.1         # 最大 |β| 比第二大至少高这么多才算“选择性”

# ===================================================


def load_three_state_npz(path):
    dat = np.load(path, allow_pickle=True)
    betas = dat["betas"]             # (3, N)
    spikes_epoch = dat["spikes_epoch"]   # (T, N)
    t_epoch = dat["t_epoch"]         # (T,)
    labels_epoch = dat["labels_epoch"].astype(int)  # (T,)
    dt = float(dat["dt"].item())
    reunion_abs = float(dat["reunion_abs"].item())
    return betas, spikes_epoch, t_epoch, labels_epoch, dt, reunion_abs


# ===================== 1. PSTH 相关 =====================
def find_state_onsets(labels_epoch, state_k):
    """
    找到每个 state_k 连续段的起点 index 列表
    """
    idxs = []
    for i in range(len(labels_epoch)):
        if labels_epoch[i] == state_k:
            if i == 0 or labels_epoch[i - 1] != state_k:
                idxs.append(i)
    return idxs


def extract_trials_for_neuron(spikes_epoch, labels_epoch, neuron_idx,
                              state_k, dt, pre_s, post_s):
    """
    对某个 neuron & state，取一批 trial：
      每个 trial 以 state_k 的片段开始为 0 时刻，
      向前 pre_s 秒、向后 post_s 秒。
    返回：
      trials: (n_trials, T_win)
      t_rel: (T_win,)
    """
    T = spikes_epoch.shape[0]
    frames_pre = int(round(pre_s / dt))
    frames_post = int(round(post_s / dt))
    win_len = frames_pre + frames_post + 1

    onsets = find_state_onsets(labels_epoch, state_k)
    trials = []

    for i0 in onsets:
        i_start = i0 - frames_pre
        i_end = i0 + frames_post

        if i_start < 0 or i_end >= T:
            # 窗口超出范围就丢掉
            continue

        trace = spikes_epoch[i_start:i_end + 1, neuron_idx]
        if trace.shape[0] == win_len:
            trials.append(trace)

    if len(trials) == 0:
        return None, None

    trials = np.stack(trials, axis=0)  # (n_trials, win_len)
    t_rel = (np.arange(win_len) - frames_pre) * dt

    return trials, t_rel


def plot_psth_for_neuron(neuron_idx, betas, spikes_epoch, labels_epoch,
                         dt, reunion_abs, out_dir,
                         pre_s=PSTH_PRE_SEC, post_s=PSTH_POST_SEC):
    """
    给一个 neuron 画 PSTH：
      三条曲线：对齐 state0 / state1 / state2 的开始
    """
    colors = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}
    state_names = {
        0: "state0 (pre-alone)",
        1: "state1 (reunion no-touch)",
        2: "state2 (social touch)"
    }

    plt.figure(figsize=(8, 4))

    legend_items = []

    for k in [0, 1, 2]:
        trials, t_rel = extract_trials_for_neuron(
            spikes_epoch, labels_epoch, neuron_idx,
            k, dt, pre_s, post_s
        )
        if trials is None:
            continue

        mean = trials.mean(axis=0)
        sem = trials.std(axis=0) / np.sqrt(trials.shape[0])

        plt.plot(t_rel, mean, color=colors[k], label=f"{state_names[k]} (n={trials.shape[0]})")
        plt.fill_between(t_rel, mean - sem, mean + sem, color=colors[k], alpha=0.2)

        legend_items.append(k)

    plt.axvline(0, color="k", linestyle="--", linewidth=0.8)
    plt.xlabel("Time from state onset (s)")
    plt.ylabel("Activity (z / spike-like)")
    plt.title(f"Neuron {neuron_idx} - PSTH aligned to 3-state onsets")

    if legend_items:
        plt.legend(fontsize=8)
    plt.grid(alpha=0.2)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"neuron{neuron_idx}_PSTH_3states.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"✓ PSTH saved: {out_path}")
    return out_path


# ===================== 2. selectivity 分类 =====================
def classify_selectivity(betas, min_abs=0.3, margin=0.1):
    """
    betas: (3, N)
    返回 DataFrame：
      neuron_index, beta0, beta1, beta2, max_state, max_abs_beta, group
    group:
      - 'weak'          : 最大 |β| < min_abs
      - 'state0_pos'    : state0 选择性，β0>0
      - 'state0_neg'    : state0 选择性，β0<0
      - 同理 state1_pos / state1_neg / state2_pos / state2_neg
      - 'mixed'         : 有两个或以上 state 的 |β| 很接近（<=margin）
    """
    n_states, N = betas.shape
    assert n_states == 3

    records = []

    for j in range(N):
        b0, b1, b2 = betas[:, j]
        abs_b = np.abs([b0, b1, b2])
        max_idx = int(np.argmax(abs_b))
        max_val = abs_b[max_idx]

        # 默认 group
        if max_val < min_abs:
            group = "weak"
        else:
            # 第二大的 |β|
            sorted_abs = np.sort(abs_b)[::-1]
            second_val = sorted_abs[1]
            if max_val - second_val <= margin:
                group = "mixed"
            else:
                sign = "pos" if betas[max_idx, j] > 0 else "neg"
                group = f"state{max_idx}_{sign}"

        records.append({
            "neuron_index": j,
            "beta_state0": b0,
            "beta_state1": b1,
            "beta_state2": b2,
            "max_state": max_idx,
            "max_abs_beta": max_val,
            "group": group
        })

    df = pd.DataFrame.from_records(records)
    return df


def summarize_groups(df, out_dir):
    counts = df["group"].value_counts().sort_index()
    print("\n=== Selectivity groups summary ===")
    for g, c in counts.items():
        print(f"{g:12s} : {c:3d}")

    csv_path = os.path.join(out_dir, "three_state_selectivity_groups.csv")
    df.to_csv(csv_path, index=False)
    print(f"✓ Selectivity CSV saved: {csv_path}")
    return csv_path


# ===================== 3. PCA 降维可视化 =====================
def plot_pca_spikes(spikes_epoch, labels_epoch, out_dir):
    """
    对 spikes_epoch 做 PCA(2D)，按三状态上色。
    """
    # 再 zscore 一次以防万一（按 neuron）
    X = spikes_epoch - spikes_epoch.mean(axis=0, keepdims=True)
    std = spikes_epoch.std(axis=0, keepdims=True) + 1e-6
    X = X / std

    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)

    colors = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}
    labels_text = {0: "state0 (pre-alone)", 1: "state1 (no-touch)", 2: "state2 (touch)"}

    plt.figure(figsize=(6, 5))
    for k in [0, 1, 2]:
        mask = labels_epoch == k
        if mask.sum() == 0:
            continue
        plt.scatter(X2[mask, 0], X2[mask, 1],
                    s=5, alpha=0.5, color=colors[k], label=labels_text[k])

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of spikes_epoch colored by 3-state label")
    plt.legend(markerscale=3, fontsize=8)
    plt.grid(alpha=0.2)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "pca_spikes_epoch_3state.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"✓ PCA plot saved: {out_path}")
    return out_path


# ===================== README =====================
def write_readme(deep_out_dir, psth_paths, select_csv, pca_path):
    lines = []
    lines.append("# Three-state deep-dive: PSTH, selectivity, PCA")
    lines.append("")
    lines.append("This folder contains additional analyses based on `three_state_glm_outputs.npz`:")
    lines.append("")
    lines.append("## 1. PSTH of representative neurons")
    lines.append("")
    lines.append("For each of the following neurons: 82, 112, 11, 84, 30, 2,")
    lines.append("we computed peristimulus time histograms (PSTHs) aligned to **state onsets**:")
    lines.append("- State 0 onset: beginning of a pre-alone (state0) segment.")
    lines.append("- State 1 onset: beginning of a reunion-no-touch (state1) segment.")
    lines.append("- State 2 onset: beginning of a social-touch (state2) bout.")
    lines.append("")
    lines.append("Each figure `neuronXX_PSTH_3states.png` shows:")
    lines.append("- X-axis: time relative to state onset (s).")
    lines.append("- Y-axis: average activity (z / spike-like).")
    lines.append("- Three colored traces: mean ± SEM across trials for state0, state1, state2.")
    lines.append("")
    lines.append("Generated PSTH figures:")
    for p in psth_paths:
        lines.append(f"- {os.path.basename(p)}")
    lines.append("")
    lines.append("## 2. Three-state selectivity classification")
    lines.append("")
    lines.append("Using the 3-state GLM weights (β_state0, β_state1, β_state2) for each neuron,")
    lines.append("we defined a simple selectivity index:")
    lines.append("- If max |β| < 0.3  → group: `weak`.")
    lines.append("- Else if the largest |β| exceeds the second largest by > 0.1:")
    lines.append("  - group: `stateK_pos` if β_stateK > 0")
    lines.append("  - group: `stateK_neg` if β_stateK < 0")
    lines.append("- Otherwise → group: `mixed` (similar contributions to multiple states).")
    lines.append("")
    lines.append(f"The full table is saved as `{os.path.basename(select_csv)}` with columns:")
    lines.append("- neuron_index")
    lines.append("- beta_state0, beta_state1, beta_state2")
    lines.append("- max_state, max_abs_beta")
    lines.append("- group (e.g. `state2_pos`, `state1_neg`, `weak`, `mixed`).")
    lines.append("")
    lines.append("## 3. Low-dimensional projection (PCA)")
    lines.append("")
    lines.append(f"Figure `{os.path.basename(pca_path)}` shows a 2D PCA projection of `spikes_epoch`")
    lines.append("with each time point colored by its 3-state label:")
    lines.append("- blue: state0 (pre-alone)")
    lines.append("- orange: state1 (reunion no-touch)")
    lines.append("- green: state2 (social touch)")
    lines.append("")
    lines.append("This allows visual check of whether the three states occupy separable regions")
    lines.append("in population activity space.")
    lines.append("")
    lines.append("## Source")
    lines.append("")
    lines.append("All analyses are derived from:")
    lines.append("- `three_state_glm/three_state_glm_outputs.npz`")
    lines.append("")
    readme_path = os.path.join(deep_out_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"✓ README saved: {readme_path}")
    return readme_path


# ===================== 主函数 =====================
def main():
    print("\n" + "=" * 70)
    print("Three-state deep-dive: PSTH, selectivity, PCA")
    print("=" * 70)

    betas, spikes_epoch, t_epoch, labels_epoch, dt, reunion_abs = load_three_state_npz(three_state_npz)
    T, N = spikes_epoch.shape
    print(f"✓ Loaded three_state_glm_outputs: T={T}, N={N}, dt≈{dt:.3f}s")

    # 1) PSTH for selected neurons
    psth_paths = []
    for ni in PSTH_NEURONS:
        if ni < 0 or ni >= N:
            print(f"⚠️ neuron index {ni} out of range, skipped.")
            continue
        p = plot_psth_for_neuron(ni, betas, spikes_epoch, labels_epoch, dt, reunion_abs, deep_out_dir)
        psth_paths.append(p)

    # 2) selectivity classification
    sel_df = classify_selectivity(betas, min_abs=SELECT_MIN_ABS_BETA, margin=SELECT_MARGIN)
    select_csv = summarize_groups(sel_df, deep_out_dir)

    # 3) PCA projection
    pca_path = plot_pca_spikes(spikes_epoch, labels_epoch, deep_out_dir)

    # 4) README
    write_readme(deep_out_dir, psth_paths, select_csv, pca_path)

    print("\n✅ Deep-dive analysis complete.")
    print(f"  Output folder: {deep_out_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
