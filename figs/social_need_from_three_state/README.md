# 基于三状态 GLM 的社交需求指数分析

本文件夹由脚本 `社交需求指数_三态GLM.py` 自动生成，用于从 mPFC 群体活动中构造并可视化一个“社交需求指数”。

## 一、指数定义

我们基于三状态 GLM 的权重向量：
- β_state0：重聚前 200s 独处期（state0）
- β_state1：重聚后但无触摸（state1）
- β_state2：社交触摸期（state2）

构造一条“触摸 vs 无触摸”轴：
  w = β_state2 − β_state1

然后对每一帧的群体活动 spikes(t) 做投影：
  Need_raw(t) = w^T · spikes(t)
再在时间维度上做 z-score，得到：
  Need_z(t) = zscore(Need_raw(t))

直观解释：
- Need_z 高 → 当下神经活动更像“触摸状态”（社交获得/接触已发生）。
- Need_z 低 → 更像“有对方但暂时没有触摸/保持距离”的状态。

## 二、图像说明

### 1）social_need_timecourse.png
整段扩展 epoch 上的社交需求指数随时间变化：
- 纵轴：Need_z(t)。
- 横轴：相对重聚时间（秒）。
- 紫色曲线：社交需求指数。
- 灰色阶梯线：三状态标签（0=独处，1=重聚无触摸，2=触摸）。

### 2）social_need_touch_onset_PSTH.png
对齐每次“社交触摸开始”时间点的 Need_z PSTH：
- X 轴：相对触摸开始的时间（秒，0 点为“社交开始”）。
- Y 轴：Need_z 的平均 ± SEM。
- 触摸开始前 5 秒被视为“预触摸窗口”，可用于观察是否存在需求的 ramping。

### 3）social_need_state_boxplot.png
比较不同状态/时间窗口下社交需求指数的分布：
- 0:独处期（state0_alone）。
- 1:无触摸(非预触摸)（state1_notouch_normal）。
- 1:预触摸窗口（每次触摸开始前 5s 内，state1_notouch_pretouch）。
- 2:触摸期（state2_touch）。

如果在数据中观察到：
- Need_z 在独处期较低；
- 重聚后无触摸期中等；
- 预触摸窗口中 Need_z 逐渐上升；
- 真正触摸期达到峰值或平台；
那么这个 Need_z(t) 可以自然地被解释为一个“社交需求/社交驱动力指数”。

## 三、数据文件

- `social_need_timecourse.png`：时间进程图。
- `social_need_touch_onset_PSTH.png`：对齐触摸起点的 Need_z PSTH。
- `social_need_state_boxplot.png`：不同状态/时间窗口下的 Need_z 分布比较。
- `社交需求指数_三态GLM.py`：生成以上所有结果的脚本（如有保存）。
