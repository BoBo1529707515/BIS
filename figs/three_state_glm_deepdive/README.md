# Three-state deep-dive: PSTH, selectivity, PCA

This folder contains additional analyses based on `three_state_glm_outputs.npz`:

## 1. PSTH of representative neurons

For each of the following neurons: 82, 112, 11, 84, 30, 2,
we computed peristimulus time histograms (PSTHs) aligned to **state onsets**:
- State 0 onset: beginning of a pre-alone (state0) segment.
- State 1 onset: beginning of a reunion-no-touch (state1) segment.
- State 2 onset: beginning of a social-touch (state2) bout.

Each figure `neuronXX_PSTH_3states.png` shows:
- X-axis: time relative to state onset (s).
- Y-axis: average activity (z / spike-like).
- Three colored traces: mean ± SEM across trials for state0, state1, state2.

Generated PSTH figures:
- neuron82_PSTH_3states.png
- neuron112_PSTH_3states.png
- neuron11_PSTH_3states.png
- neuron84_PSTH_3states.png
- neuron30_PSTH_3states.png
- neuron2_PSTH_3states.png

## 2. Three-state selectivity classification

Using the 3-state GLM weights (β_state0, β_state1, β_state2) for each neuron,
we defined a simple selectivity index:
- If max |β| < 0.3  → group: `weak`.
- Else if the largest |β| exceeds the second largest by > 0.1:
  - group: `stateK_pos` if β_stateK > 0
  - group: `stateK_neg` if β_stateK < 0
- Otherwise → group: `mixed` (similar contributions to multiple states).

The full table is saved as `three_state_selectivity_groups.csv` with columns:
- neuron_index
- beta_state0, beta_state1, beta_state2
- max_state, max_abs_beta
- group (e.g. `state2_pos`, `state1_neg`, `weak`, `mixed`).

## 3. Low-dimensional projection (PCA)

Figure `pca_spikes_epoch_3state.png` shows a 2D PCA projection of `spikes_epoch`
with each time point colored by its 3-state label:
- blue: state0 (pre-alone)
- orange: state1 (reunion no-touch)
- green: state2 (social touch)

This allows visual check of whether the three states occupy separable regions
in population activity space.

## Source

All analyses are derived from:
- `three_state_glm/three_state_glm_outputs.npz`
一、这张表每一列在说什么？
你这张表来自我们之前的规则：

beta_state0 / 1 / 2：
三状态多分类 GLM 的权重（每个状态一条 β 向量）。
对某个神经元 j：

β_state2,j > 0：
它活动高会把模型推向“触摸 state2”
β_state2,j < 0：
它活动高会把模型推离“触摸”，更像 state0/1
同理适用于 state0、state1。

max_state：
这个神经元在三个状态里，哪一个状态的 |β| 最大（0/1/2）。

max_abs_beta：
三个状态中最大的 |β| 值。

group：
我们按你脚本设定的规则给的粗分类：

weak：max |β| < 0.3（你设的阈值）
mixed：最大 |β| 和第二大 |β| 差 ≤ 0.1
stateK_pos：
对 stateK 选择性偏正向（β_stateK > 0 且明显高于其他状态）
stateK_neg：
对 stateK 选择性偏负向（β_stateK < 0 且明显高于其他状态）
注意：
state2_neg 的含义不是“它属于 state2”，而是
“它在 state2 的权重最强，但方向是负的”
→ 也就是“强烈反对触摸状态”的关键细胞。
这类细胞在你数据里其实特别重要（比如 82 那一类）。

二、结合你这段数据逐行“读懂”
我挑你这段里最有信息量的几类举例：

1）典型的触摸正向细胞（state2_pos）
比如：

neuron 2
β0 = -0.1915
β1 = -0.6321
β2 = +0.8237
group = state2_pos
解释：
这颗细胞活动升高时，模型稳定地推向触摸状态。
这就是你“社交触摸/获得”候选的核心细胞之一。

neuron 12 / 17 也属于这一类
β2 分别 ~ +0.63 / +0.46
2）典型的触摸负向细胞（state2_neg）
比如：

neuron 3

β0 = +0.2443
β1 = +0.5141
β2 = -0.7584
group = state2_neg
neuron 5

β1 = +0.6438
β2 = -0.7520
group = state2_neg
解释：
这两颗在触摸状态上是“强负向编码/强抑制型”：
它们活动高的时候，模型会非常明确地说：

“现在不像触摸，更像 no-touch 或 pre-alone。”

这类细胞常常就是你之前看到的
**state1（有对方但没接触）**的“对抗端”。

3）“重聚无触摸”选择性细胞（state1_pos / state1_neg）
neuron 24
β0 = -0.3882
β1 = +0.6104
β2 = -0.2222
group = state1_pos
解释：
这颗更像你“有对方在但没有触摸/保持距离期”的编码者。

neuron 10
β0 = +0.3878
β1 = -0.8353
β2 = +0.4475
group = state1_neg
解释：
它对 state1 是强负向——
活动升高时更像 state0 或 state2。
这种细胞常见于“从 no-touch 向 touch 过渡”的轴上。

4）mixed（混合型）
比如：

neuron 11
β0 = -0.4410
β1 = +0.5281
β2 ≈ -0.0871
group = mixed
解释：
它对 state0、state1 都挺用力，差距没有拉开到“选择性阈值”。
这种细胞更像“情境/阶段性”细胞，
能分“重聚前 vs 重聚后”，但不一定专门指向触摸本身。

5）weak（弱贡献）
你这段里多数是 weak（0、1、4、8、13、14、16、18…）

解释：
这不等于“没用”，只是在这三个状态的线性判别里
它们不是决定性的那一撮。
但是它们可能在更细的行为维度（速度/探索/朝向）里有意义。

三、把你现在的“核心故事”再压缩成一句
从你三态大结果看：

state0 主要由一批“环境/情境改变”细胞提示
“我从独处进入重聚场景了”。

state1 vs state2 的区分最清晰、权重也最大，
形成了一个很强的 no-touch ↔ touch 对抗轴：

state2_pos（如 2、12、17…）
state2_neg / state1_pos（如 3、5、24 这一类）
这也解释了为什么你三态模型准确率能到 0.967。