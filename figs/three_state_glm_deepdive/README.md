# 三状态深度分析：PSTH、选择性分群与 PCA

本文件夹包含基于 `three_state_glm/three_state_glm_outputs.npz` 的进一步分析结果，
用来更细致地理解三种行为状态下 mPFC 神经元的活动模式：

- **state0**：重聚前 200 秒的独处期（pre-alone）  
- **state1**：重聚后但当前没有社交触摸（reunion no-touch）  
- **state2**：社交触摸期（social touch）

---

## 1. 代表性神经元的 PSTH（对齐平均活动）

文件示例：
- `neuron82_PSTH_3states.png`
- `neuron112_PSTH_3states.png`
- `neuron11_PSTH_3states.png`
- `neuron84_PSTH_3states.png`
- `neuron30_PSTH_3states.png`
- `neuron2_PSTH_3states.png`

这些图都是 **单个神经元** 的 PSTH（Peri-Stimulus Time Histogram），
这里的 “stimulus” 换成了 “状态开始”。

### 1.1 对齐方式

以某条曲线为例（绿色 = state2）：

1. 在整段扩展时间内，找到所有 **state2 开始** 的时刻（每一次“触摸 bout 开始”的时间点）。
2. 以每一个开始时刻为 0 秒，提取一个时间窗口：  
   `[-10 s, +20 s]`。
3. 把所有这些窗口的时间轴都平移，使“状态开始”对齐到 0 秒。
4. 对所有窗口在每个时间点上取平均（并计算 SEM），
   得到这条神经元在“状态开始前后”的**平均活动轨迹**。

同理，橙色曲线对应 **state1 开始**，也是对齐后多次 trial 的平均。

图例中的 `n=37 / n=36` 表示：
- 这条曲线是由多少次 **状态开始事件** 参与平均得到的  
  （例如 `n=37` 表示平均了 37 段 state1 起始片段）。

### 1.2 如何阅读这些 PSTH 图

以 `Neuron 112` 为例（pro-touch 细胞）：

- 0 秒左侧（状态开始前 ~10 秒），橙色/绿色曲线差异不大  
  → 在进入 state1 / state2 之前，这颗细胞的活动类似。
- 在 0 秒（虚线）附近，绿色曲线（state2）突然显著上升，
  橙色（state1）只略有抬升  
  → **进入触摸状态本身是强刺激，Neuron 112 在触摸开始时被明显激活**。
- 0 秒之后一段时间，绿色整体高于橙色  
  → 触摸期内活动持续偏高，符合三状态 GLM 中 `β_state2` 显著为正的结果。

而 `Neuron 82` 的 PSTH 则常表现为：
- 相比触摸开始（state2），在 **state1 开始** 时更活跃，  
  在触摸期活动压低  
  → 对应于 `state2_neg / pro-no-touch` 类型细胞，
    在模型中起的是“远离触摸状态”的作用。

通过这些 PSTH，我们可以在**时间维度**上看到：
- 不同类型神经元在状态切换前后是 ramping、瞬间跳变还是被抑制，
- 从而更直观地理解三态 GLM 中 β 的生理含义。

---

## 2. 三状态选择性分类（selectivity）

文件：
- `three_state_selectivity_groups.csv`

该表使用三状态 GLM 得到的每个神经元权重：
- `beta_state0`：对独处期（state0）的权重
- `beta_state1`：对重聚无触摸（state1）的权重
- `beta_state2`：对社交触摸（state2）的权重

按照如下规则，对每个神经元进行粗分类：

### 2.1 表格列说明

- `neuron_index`：神经元编号（从 0 开始）
- `beta_state0, beta_state1, beta_state2`：三状态的 GLM 权重
- `max_state`：三个状态中 |β| 最大的是哪一个（0/1/2）
- `max_abs_beta`：三个状态中最大的 |β| 值
- `group`：根据以下逻辑得到的分组标签  

### 2.2 分组规则

- 若 `max_abs_beta < 0.3`  
  → 记为 **`weak`**（在这三状态判别中贡献较弱）

- 否则，若最大 |β| 比第二大至少高 0.1：
  - 若该状态的 β > 0  
    → 记为 **`stateK_pos`**（对 stateK **正向选择性编码**）
  - 若该状态的 β < 0  
    → 记为 **`stateK_neg`**（对 stateK **负向/抑制性编码**）

- 否则  
  → 记为 **`mixed`**（对多个状态贡献相近）

### 2.3 一些典型例子（帮助理解）

- **触摸正向细胞（`state2_pos`）**  
  例如 neuron 2 / 12 / 17：
  - `beta_state2` 大正值  
  - 活动高会把模型推向“触摸状态”  
  - 是典型的 **pro-touch / 社交获得** 细胞

- **触摸负向细胞（`state2_neg`）**  
  例如 neuron 3 / 5：
  - 对 state2 的权重最大但为负  
  - 活动高时模型会远离“触摸”，更像 state0 或 state1  
  - 是强烈 **反对触摸 / pro-no-touch** 的关键细胞

- **重聚无触摸细胞（`state1_pos / state1_neg`）**  
  例如 neuron 24（`state1_pos`）：
  - 在“有对方但未触摸”阶段活动高，编码“保持距离”的场景；  
  neuron 10（`state1_neg`）：  
  - 对 state1 为强负 → 活动高时更像独处或触摸，是一种“状态切换”型细胞。

- **混合型（`mixed`）**  
  例如 neuron 11：  
  - 对 state0、state1 都有较大 |β|，差距不够大  
  - 更像“情境变化/阶段性”细胞，而不是特定针对触摸本身。

整体来说，这个分群让我们可以从三态 GLM 的 β 中抽出：
- **触摸获得群（pro-touch, state2_pos）**
- **触摸抑制/保持距离群（anti-touch / pro-no-touch, state2_neg / state1_pos）**
- **独处/情境切换相关群（多为 state0 相关）**
- 以及一批混合 / 弱贡献神经元。

---

## 3. 群体低维投影（PCA）

文件：
- `pca_spikes_epoch_3state.png`

该图展示了扩展 epoch 内 `spikes_epoch` 的 **主成分分析（PCA）结果**：

- 首先对每个神经元在时间上做 z-score；
- 再将所有时间点的群体活动降到 2 维（PC1, PC2）；
- 每个时间点画成一个点，并按三种状态上色：
  - 蓝色：state0（pre-alone，重聚前独处）
  - 橙色：state1（reunion no-touch，重聚无触摸）
  - 绿色：state2（social touch，社交触摸）

通过这个图可以直观看到：
- 三类行为状态在群体神经活动空间中是否占据不同的区域；
- 是否存在从 **独处 → 重聚无触摸 → 触摸** 的“轨迹”或分离的簇。

这与三态 GLM 的高分类准确率（约 0.97）是互相印证的：
- 独处 vs 重聚在状态空间中完全分开；
- 重聚后又沿着一条类似 “no-touch ↔ touch” 的轴区分开来。

---

## 4. 数据来源

所有本文件夹中的分析都基于：
- `three_state_glm/three_state_glm_outputs.npz`

该文件包含：
- `betas`：三态 GLM 的权重矩阵（3 × 神经元数）
- `spikes_epoch`：扩展时间段内的群体活动
- `t_epoch`：对应的时间轴
- `labels_epoch`：逐帧的三态标签（0/1/2）
- 以及用于重聚时间/epoch 起止等对齐的信息。
