# -*- coding: utf-8 -*-
"""
Advanced Visualization & Classification Analysis
1. 自动纠正数据维度 (Neurons x Time)
2. 生成状态分类指纹图 (Heatmap)
3. 生成重叠相似度矩阵 (Jaccard Matrix)
4. 生成特定群组的平均轨迹图 (Average Traces with Shading)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import mannwhitneyu, zscore

# 尝试导入 seaborn 用于美化
try:
    import seaborn as sns

    SNS_AVAIL = True
    sns.set(style="whitegrid")
except ImportError:
    SNS_AVAIL = False
    print("[INFO] Seaborn not installed, using standard matplotlib.")

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']  # 适配中文
plt.rcParams['axes.unicode_minus'] = False

# ===================== 配置区域 =====================

# 1. 路径设置 (请确认)
DATA_NPZ = r"F:\工作文件\RA\python\吸引子\Mouse1_for_ssm.npz"
BEH_XLSX = r"F:\工作文件\RA\数据集\时间戳\FVB_Panneuronal_Reunion_Isolation_Day3_Mouse#1.xlsx"
OUTDIR = r"F:\工作文件\RA\python\吸引子\figs\Advanced_Visualization"

# 2. 参数设置
REUNION_ABS = 903.0
BASELINE_WIN = (-200.0, 0.0)
ALPHA_FDR = 0.05
BIN_SIZE_S = 1.0  # 绘图时的平滑窗口

# 3. 你想看轨迹的群组 (定义感兴趣的交集)
# 格式: ("图标题", ["条件1", "条件2"...]) -> 取交集并画轨迹
TRACE_GROUPS = [
    ("Persistent_Up_Neurons", ["Touch_Up", "NonTouch_Up", "ReIso_Up"]),  # 全程兴奋
    ("Persistent_Down_Neurons", ["Touch_Down", "NonTouch_Down", "ReIso_Down"]),  # 全程抑制
    ("Touch_Specific_Up", ["Touch_Up", "NonTouch_NS"]),  # 只在 Touch 兴奋
    ("ReIso_Emergent_Up", ["Touch_NS", "ReIso_Up"]),  # 再次隔离后才兴奋
]


# ===================== 核心函数 =====================

def load_data_corrected(npz_path):
    """加载并自动纠正维度，返回 (Neurons, Time)"""
    dat = np.load(npz_path, allow_pickle=True)
    Y = np.asarray(dat["Y"])
    t = np.asarray(dat["t"]).flatten()
    dt = float(dat["dt"].item())

    # 维度修正
    if Y.shape[1] == len(t):
        Yz = zscore(Y, axis=1)  # (N, T)
    elif Y.shape[0] == len(t):
        print(f"   [Info] Transposing Y from {Y.shape}...")
        Yz = zscore(Y.T, axis=1)  # 转置后 Z-score
    else:
        raise ValueError(f"Shape mismatch: t={len(t)}, Y={Y.shape}")
    return Yz, t, dt


def load_behavior_masks(xlsx_path, t_abs, reunion_abs):
    """读取 Excel 生成行为 Masks"""
    df = pd.read_excel(xlsx_path, header=None).dropna(how="all")
    df[0] = pd.to_numeric(df[0], errors="coerce")
    df[1] = df[1].astype(str).str.strip().str.lower()

    # 获取 Excel 中的 Reunion 时间 (用于对齐)
    reunion_rows = df[df[1].str.contains("重聚期开始", na=False)]
    excel_reu_t = float(reunion_rows.iloc[0, 0]) if len(reunion_rows) > 0 else 0.0

    starts = df[df[1].str.contains("社交开始", na=False)][0].values
    ends = df[df[1].str.contains("社交结束", na=False)][0].values

    # 转为相对时间
    intervals = []
    for s, e in zip(starts, ends):
        intervals.append((s - excel_reu_t, e - excel_reu_t))

    last_social_end = max([e for s, e in intervals]) if intervals else 0
    t_rel = t_abs - reunion_abs

    # 生成 Mask
    mask_touch = np.zeros(len(t_rel), dtype=bool)
    for s, e in intervals:
        mask_touch[(t_rel >= s) & (t_rel <= e)] = True

    mask_nontouch = (t_rel > 0) & (~mask_touch)  # 广义 NonTouch
    mask_reiso = (t_rel > last_social_end)  # 再次隔离

    masks = {
        "Touch": mask_touch,
        "NonTouch": mask_nontouch,
        "ReIso": mask_reiso,
        "Reunion_All": (t_rel > 0)
    }
    return masks, t_rel, intervals


def classify_neurons(Yz, t_rel, masks, baseline_win):
    """MWU 检验分类"""
    n_neurons = Yz.shape[0]
    base_mask = (t_rel >= baseline_win[0]) & (t_rel < baseline_win[1])

    # 预计算 Baseline 均值 (加速)
    # 为了避免循环过慢，这里不做全矩阵操作，还是逐个神经元稳健处理
    results = pd.DataFrame({"Neuron_Index": np.arange(n_neurons)})

    for state, mask in masks.items():
        if state == "Reunion_All": continue  # 可选跳过
        print(f"   Classifying {state}...")

        pvals, deltas, labels = [], [], []

        # 降采样步长 (加速计算 + 去自相关)
        step = 10

        for i in range(n_neurons):
            row = Yz[i, :]
            v_base = row[base_mask][::step]
            v_state = row[mask][::step]

            if len(v_base) < 3 or len(v_state) < 3:
                pvals.append(1.0);
                deltas.append(0);
                labels.append("NS")
                continue

            try:
                _, p = mannwhitneyu(v_state, v_base, alternative='two-sided')
                d = np.mean(v_state) - np.mean(v_base)
                pvals.append(p)
                deltas.append(d)
            except:
                pvals.append(1.0);
                deltas.append(0)

        # FDR
        p_clean = np.array(pvals)
        # 简单 BH 校正
        sort_idx = np.argsort(p_clean)
        p_sorted = p_clean[sort_idx]
        q_vals = p_sorted * len(p_clean) / np.arange(1, len(p_clean) + 1)
        q_vals = np.minimum.accumulate(q_vals[::-1])[::-1]

        # 恢复顺序
        q_final = np.zeros_like(q_vals)
        q_final[sort_idx] = q_vals

        # 打标签
        final_labels = []
        for i in range(n_neurons):
            if q_final[i] < ALPHA_FDR:
                final_labels.append("Up" if deltas[i] > 0 else "Down")
            else:
                final_labels.append("NS")

        results[f"{state}_Dir"] = final_labels

    return results


# ===================== 绘图函数 =====================

def plot_classification_heatmap(df, outpath):
    """
    1. 全局分类热图 (Neurons x States)
    """
    # 提取所有 _Dir 列
    cols = [c for c in df.columns if c.endswith("_Dir")]
    if not cols: return

    # 映射数值: Up=1, NS=0, Down=-1
    map_dict = {"Up": 1, "NS": 0, "Down": -1}
    data = df[cols].replace(map_dict)

    # 排序：让相似模式的神经元聚在一起
    # 简单按第一列排序，或者按所有列排序
    data_sorted = data.sort_values(by=cols, ascending=False)

    plt.figure(figsize=(6, 8))
    # 自定义颜色: 蓝(-1), 白(0), 红(1)
    if SNS_AVAIL:
        cmap = sns.diverging_palette(240, 10, as_cmap=True, center="light")
        sns.heatmap(data_sorted, cmap=cmap, cbar=False, yticklabels=False)
    else:
        plt.imshow(data_sorted, aspect='auto', cmap='coolwarm', interpolation='nearest')

    # 手动添加图例
    colors = ["blue", "white", "red"]
    labels = ["Down", "NS", "Up"]
    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title("Neuron State Classification Fingerprint")
    plt.xlabel("States")
    plt.ylabel(f"Neurons (Sorted, N={len(df)})")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_counts_bar(df, outpath):
    """
    2. 计数柱状图
    """
    cols = [c for c in df.columns if c.endswith("_Dir")]
    summary = []
    for c in cols:
        vc = df[c].value_counts()
        state_name = c.replace("_Dir", "")
        summary.append({
            "State": state_name,
            "Up": vc.get("Up", 0),
            "Down": vc.get("Down", 0),
            "NS": vc.get("NS", 0)
        })

    df_sum = pd.DataFrame(summary)

    # 绘图
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df_sum))
    width = 0.35

    # 堆叠柱状图? 或者分组
    # 这里画分组: 上方画 Up (红), 下方画 Down (蓝)
    ax.bar(x, df_sum["Up"], width, label='Up', color='#d62728')
    ax.bar(x, -df_sum["Down"], width, label='Down', color='#1f77b4')

    # 标注数值
    for i, v in enumerate(df_sum["Up"]):
        ax.text(i, v + 5, str(v), ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(df_sum["Down"]):
        ax.text(i, -v - 20, str(v), ha='center', va='top', fontsize=9)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df_sum["State"])
    ax.set_ylabel("Neuron Count (+Up / -Down)")
    ax.set_title("Number of Modulated Neurons per State")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_jaccard_heatmap(group_sets, outpath):
    """
    3. Jaccard 相似度热图
    """
    keys = sorted(group_sets.keys())
    # 过滤掉 NS，只看 Up/Down
    keys = [k for k in keys if "NS" not in k]
    n = len(keys)
    mat = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            s1 = group_sets[keys[i]]
            s2 = group_sets[keys[j]]
            intersect = len(s1 & s2)
            union = len(s1 | s2)
            mat[i, j] = intersect / union if union > 0 else 0

    plt.figure(figsize=(10, 8))
    if SNS_AVAIL:
        sns.heatmap(mat, xticklabels=keys, yticklabels=keys, annot=True, fmt=".2f", cmap="Greens")
    else:
        plt.imshow(mat, cmap="Greens")
        plt.xticks(np.arange(n), keys, rotation=45, ha="right")
        plt.yticks(np.arange(n), keys)
        plt.colorbar(label="Jaccard Index")

    plt.title("Overlap Similarity (Jaccard Index)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_group_traces(Yz, t_rel, intervals, df_res, group_defs, outdir):
    """
    4. 群组平均轨迹图 (带阴影)
    group_defs: [("Title", ["Touch_Up", "ReIso_Up"]), ...]
    """
    for title, conditions in group_defs:
        # 1. 找到符合所有条件的神经元索引
        # 这是一个 AND 逻辑
        valid_indices = df_res["Neuron_Index"].values

        for cond in conditions:
            # cond 比如 "Touch_Up"
            # 解析: State = Touch, Dir = Up
            state_name, direction = cond.rsplit("_", 1)
            col_name = f"{state_name}_Dir"

            if col_name not in df_res.columns:
                print(f"[Warn] Column {col_name} not found. Skipping condition.")
                continue

            # 筛选
            current_subset = df_res[df_res[col_name] == direction]["Neuron_Index"].values
            # 取交集
            valid_indices = np.intersect1d(valid_indices, current_subset)

        n_count = len(valid_indices)
        if n_count == 0:
            print(f"   [Plot Trace] Group '{title}' is empty. Skipping.")
            continue

        # 2. 提取这些神经元的轨迹并平均
        traces = Yz[valid_indices, :]  # (N_subset, T)
        mean_trace = np.mean(traces, axis=0)
        sem_trace = np.std(traces, axis=0) / np.sqrt(n_count)

        # 3. 绘图
        plt.figure(figsize=(12, 5))
        ax = plt.gca()

        # 画 Touch 阴影背景
        for s, e in intervals:
            ax.axvspan(s, e, color='orange', alpha=0.15, lw=0)

        # 画重聚线
        ax.axvline(0, color='black', linestyle='--', label='Reunion')

        # 画曲线
        ax.plot(t_rel, mean_trace, color='#333333', lw=1.5, label='Mean Z-score')
        ax.fill_between(t_rel, mean_trace - sem_trace, mean_trace + sem_trace, color='#333333', alpha=0.3)

        # 装饰
        ax.set_title(f"{title} (n={n_count} neurons)\nConditions: {' + '.join(conditions)}")
        ax.set_xlabel("Time from Reunion (s)")
        ax.set_ylabel("Z-score Activity")
        ax.set_xlim(t_rel[0], t_rel[-1])

        # 添加图例
        # 创建一个 dummy patch for Touch
        touch_patch = mpatches.Patch(color='orange', alpha=0.15, label='Social Touch')
        handles, labels = ax.get_legend_handles_labels()
        handles.append(touch_patch)
        ax.legend(handles=handles, loc='upper right')

        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"Trace_{title}.png"), dpi=300)
        plt.close()
        print(f"   [Plot Trace] Saved trace for '{title}' (n={n_count})")


# ===================== 主程序 =====================

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"Output Directory: {OUTDIR}")

    # 1. 加载数据
    print("1. Loading Data...")
    Yz, t, dt = load_data_corrected(DATA_NPZ)
    print(f"   Shape: {Yz.shape} (Neurons x Time)")

    # 2. 生成 Masks
    print("2. Parsing Behavior...")
    masks, t_rel, intervals = load_behavior_masks(BEH_XLSX, t, REUNION_ABS)

    # 3. 分类
    print("3. Classifying Neurons...")
    df_res = classify_neurons(Yz, t_rel, masks, BASELINE_WIN)
    df_res.to_csv(os.path.join(OUTDIR, "Classification_Table.csv"), index=False)

    # 4. 构建 Group Sets 用于 Jaccard
    group_sets = {}
    cols = [c for c in df_res.columns if c.endswith("_Dir")]
    for c in cols:
        state = c.replace("_Dir", "")
        for d in ["Up", "Down"]:
            idx = set(df_res[df_res[c] == d]["Neuron_Index"].values)
            group_sets[f"{state}_{d}"] = idx

    # ================= 绘图环节 =================
    print("4. Generating Plots...")

    # A. 计数图
    plot_counts_bar(df_res, os.path.join(OUTDIR, "Plot_Counts_Bar.png"))

    # B. 全局热图
    plot_classification_heatmap(df_res, os.path.join(OUTDIR, "Plot_Classification_Heatmap.png"))

    # C. Jaccard 矩阵
    plot_jaccard_heatmap(group_sets, os.path.join(OUTDIR, "Plot_Overlap_Matrix.png"))

    # D. 群组平均轨迹 (Kill Feature)
    plot_group_traces(Yz, t_rel, intervals, df_res, TRACE_GROUPS, OUTDIR)

    print("\n✅ All Advanced Visualizations Completed!")


if __name__ == "__main__":
    main()
