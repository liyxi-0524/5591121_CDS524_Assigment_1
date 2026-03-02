# Agent 参数手册

本文档汇总本项目中训练与演示相关的 Agent 与训练参数，覆盖：统一 Agent（UnifiedAgent）、Q-Learning 超参数、训练控制与收敛判定、日志与可视化、以及环境奖励（Shaping）。

> 代码入口：
> - UnifiedAgent（统一/混合策略）：[unified.py](file:///Users/liyongxi/Desktop/贪吃蛇/snake/agents/unified.py)
> - Q-Learning 训练与收敛逻辑：[qlearning.py](file:///Users/liyongxi/Desktop/贪吃蛇/snake/agents/qlearning.py)
> - 命令行与可视化：[main.py](file:///Users/liyongxi/Desktop/贪吃蛇/main.py)
> - 环境与奖励配置：[game.py](file:///Users/liyongxi/Desktop/贪吃蛇/snake/game.py)

## UnifiedAgent（统一/混合）

- 模式（mode）
  - `q`：表格 Q-Learning（训练/演示）。
  - `greedy`：启发式贪心（不学习，仅演示/对比）。
  - `mixed`：混合策略，行为次序为：
    1) 若当前状态 Q 行“近乎等值”（`np.allclose(row, row[0])`）且提供了环境，则使用“安全贪心”补全动作；
    2) 否则（非等值），若 `training` 且触发 ε 探索则随机；
    3) 否则按 Q 表 `argmax`。
- 接口
  - `act(state, training=True|False)`：返回相对动作（0 左/1 直/2 右）。
  - `train(...)`：Q-Learning 训练，返回完整训练指标（见下文“返回值”）。
  - `save(path)` / `load(path)`：保存/加载 Q 表（`.npy`）。
  - `set_env(env)`：为 mixed/greedy 行为提供环境引用（用于安全碰撞判断）。

> 菜单/CLI：
> - 菜单新增“Agent (Unified/mixed)”可直接以 mixed 模式演示。
> - CLI 新增 `play-unified` 子命令；`train` 子命令新增 `--mode {q,greedy,mixed}`（默认 `q`）。

## Q-Learning 超参数

- `alpha`（学习率，默认 0.1）：Q 更新步幅。
- `gamma`（折扣因子，默认 0.9）：长期回报权重。
- `epsilon`（初始探索率，默认 0.1）：训练中以概率 ε 随机探索（mixed 模式下“贪心兜底”优先于随机探索）。

> 参考实现与更新公式见 [qlearning.py](file:///Users/liyongxi/Desktop/贪吃蛇/snake/agents/qlearning.py#L38-L46)。

## 训练控制（命令行）

在 `train` 或 `auto` 子命令中常用参数：

- 基础
  - `--episodes`：训练上限轮次（`auto` 可早停，`train` 可通过收敛参数控制）。
  - `--mode`：`q`/`greedy`/`mixed`，训练时默认 `q`；若设为 `mixed` 将启用“Q 主导 + 贪心兜底”。
  - `--model`：Q 表路径（默认 `models/q_table.npy`）。
  - `--log`：训练日志 CSV 路径（含 Q 更新统计列）。
  - `--print-every`：每隔 N 局输出一次关键统计。
- 衰减
  - `--eps-decay`、`--min-eps`：ε 的指数衰减与下限。
  - `--alpha-decay`、`--min-alpha`：α 的指数衰减与下限。
- 收敛判定
  - `--window`：滑动窗口大小（用于均值统计）。
  - `--conv-threshold`：奖励窗口均值收敛阈值（绝对差）。
  - `--conv-patience`：连续满足阈值的次数；`0` 表示不早停。
  - `--conv-qdelta-threshold`：Q 表变化窗口均值阈值（与奖励判据同时满足时早停）。
- 评估
  - `--eval-episodes`：训练结束后在“无探索（greedy）”下评估的局数。

## 收敛判定（内置）

- 奖励收敛：最近 `window` 局的 `reward_ma` 与上一窗口均值之差的绝对值 ≤ `conv_threshold`，连续 `conv_patience` 次，且样本 ≥ `2*window`。
- Q 表变化收敛（可选）：最近 `window` 局的 `q_delta_mean` 窗口均值 ≤ `conv_qdelta_threshold`，连续 `conv_patience` 次。
- 早停策略：
  - 仅设置奖励阈值：满足“奖励收敛”即早停。
  - 同时设置 Q 表阈值：仅当“奖励 + Q 表变化”都收敛时早停。
  - `conv_patience=0`：不早停。

**返回值字段**：`rewards`、`scores`、`epsilons`、`q_update_mean`、`q_delta_mean`、`converged_at`、`converged_reward_at`、`converged_qdelta_at`。

## 日志与可视化

- CSV 日志列
  - `episode`、`reward`、`score`、`epsilon`
  - `q_update_mean`（每集内单步 |ΔQ| 的平均值）
  - `q_delta_mean`（与上集 Q 表差值 |Q − Q_prev_ep| 的均值）
  - `reward_ma`、`score_ma`（按 `--window` 计算的移动平均）
- PNG（四合一）
  1) 奖励与滑动平均（左上，叠加 ε 曲线）；
  2) ε 衰减（右上）；
  3) 单步 |ΔQ| 平均（左下）；
  4) 每集 Q 表变化均值（右下）。

生成示例：
```bash
python main.py plot-png \
  --log models/train_log.csv \
  --out models/train_plot.png \
  --fig-width 12 --fig-height 8 --dpi 150
```

## 环境奖励（Shaping）

- 基础奖励：
  - 吃到食物：`reward_food`（默认 18.0）
  - 步长惩罚：`reward_step`（默认 −0.005）
  - 撞墙/撞身终局：`reward_death`（默认 −10.0）
  - 长时间未吃到食物终止：`reward_timeout`（默认 −5.0）
- 目标 Shaping：
  - 接近食物（曼哈顿距离减小）：`reward_closer`（默认 +0.2）
  - 远离食物（曼哈顿距离增大）：`reward_farther`（默认 −0.2）
- 安全 Shaping：
  - 与身体最小距离变大：`reward_body_farther`（默认 +0.1）
  - 与身体最小距离变小：`reward_body_closer`（默认 −0.1）

> 奖励实现与参数定义见：[game.py](file:///Users/liyongxi/Desktop/贪吃蛇/snake/game.py)。必要时可将这些系数暴露到 CLI 以便实验对比。

## 常用脚本示例

1) 混合模式从零训练 100000 局（不早停、关键节点输出）：
```bash
python main.py train \
  --episodes 100000 --mode mixed \
  --alpha 0.1 --alpha-decay 0.99995 --min-alpha 0.015 \
  --gamma 0.95 \
  --epsilon 0.2 --eps-decay 0.9997 --min-eps 0.015 \
  --conv-patience 0 \
  --model models/q_table.npy \
  --eval-episodes 100 \
  --log models/train_log_100k_mixed.csv \
  --window 100 \
  --print-every 5000
python main.py plot-png \
  --log models/train_log_100k_mixed.csv \
  --out models/train_plot_100k_mixed.png \
  --fig-width 12 --fig-height 8 --dpi 150
```

2) 收敛判定（奖励 + Q 表变化）并早停：
```bash
python main.py train \
  --episodes 20000 \
  --window 100 \
  --conv-threshold 0.05 \
  --conv-qdelta-threshold 0.02 \
  --conv-patience 3 \
  --model models/q_table.npy \
  --eval-episodes 100 \
  --log models/train_log.csv
```

3) 菜单与演示
```bash
python main.py         # 菜单内可选 “Agent (Unified/mixed)”
python main.py play-unified
python main.py play-q --model models/q_table.npy
```

## 参数调优建议

- alpha：0.05–0.2；建议配合 `alpha-decay`（0.9999–0.99997），`min-alpha` 0.01–0.03。
- gamma：0.9–0.99；地图增大时倾向 ≥0.95。
- epsilon：起始 0.2–0.3；`eps-decay` 0.9996–0.9998；`min-eps` 0.01–0.05。
- Shaping：
  - `reward_closer` 0.2–0.3；`reward_farther` −0.15–−0.25
  - `reward_body_farther` 0.05–0.15；`reward_body_closer` −0.08–−0.12
  - `reward_step` 0 至 −0.005（出现原地循环时再引入轻惩罚）

> 经验：更长探索期 + α 衰减有助于摆脱早期局部最优；若单局负值频发，可弱化“远离食物/靠近身体”的惩罚，或适度提高吃食奖励以提升正例权重。
