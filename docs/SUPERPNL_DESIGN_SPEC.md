# SuperPnL 设计说明书

## 1. 背景

旧方案 FactorBarNet 的目标是直接预测未来固定长度的 K 线路径，并从路径中评估若干固定 horizon。

这个目标适合验证因子是否改善价格预测，但如果最终目标是交易收益，它还缺少三个关键环节：

- 预测正确不等于扣费后可交易。
- 方向正确不等于仓位应该足够大。
- K 线误差下降不等于净 PnL、Sharpe 或回撤改善。

SuperPnL 重新定义目标：**直接面向净 PnL 学习可执行 edge 和目标仓位**。

项目长期可以容纳所有 PnL-first 预测模型。第一版只做现货，不做永续、杠杆、借币做空和实盘下单。

---

## 2. 设计目标

### 2.1 主目标

输入决策时刻 `t` 及以前的信息：

```text
过去 L 根 bar:
open, high, low, close, volume, amount

过去 L 根特征:
技术特征、市场环境特征、时间特征
```

输出：

```text
edge_{h}        未来可交易收益估计
target_pos_{h} 现货目标仓位，范围 [0, 1]
```

其中：

```text
0   = 全部现金
0.5 = 半仓现货
1.0 = 满仓现货
```

其中 `h` 来自策略配置，而不是模型硬编码。数据层统一使用 1min K 线：

```text
bar_size = "1m"
strategy_horizons = ["5m", "15m"] 或 ["5m", "15m", "30m"]
```

`5m` 对应未来 5 根 1min bar，`15m` 对应未来 15 根 1min bar。策略可以只交易 `5m/15m`，也可以增加 `30m/60m`。不建议把 1min horizon 作为默认主交易周期，因为手续费和滑点对极短周期信号更敏感。

### 2.2 约束目标

- 只允许使用 `t` 及以前的数据生成信号。
- 信号在 `t` 收盘后产生，最早只能在 `t+1` 执行。
- 当前阶段不把盘口、成本、流动性作为训练特征。
- 回测使用固定成本配置；如果手续费或滑点设为 0，必须在报告中明确标注。
- 评测必须包含 no-trade、buy-and-hold、OHLCV-only 和有因子模型。
- 所有收益必须同时报告 gross PnL 和 net PnL。
- 任何新增特征必须说明是否可能产生未来信息泄漏。

### 2.3 非目标

第一版不做以下事情：

- 不做永续合约。
- 不做杠杆。
- 不做借币做空。
- 不做强化学习。
- 不做实盘自动下单。
- 不做复杂盘口撮合模拟。
- 不做盘口、成本、流动性特征建模。
- 不直接以单次回测收益作为模型有效的证据。

---

## 3. 总体架构

推荐第一版架构：

```text
                     ┌────────────────────────┐
OHLCV history ──────▶│ Bar Encoder             │
                     │ Causal Dilated TCN      │
                     └───────────┬────────────┘
                                 │
                     ┌───────────▼────────────┐
Feature history ────▶│ Feature Encoder         │
                     │ Causal TCN / MLP        │
                     └───────────┬────────────┘
                                 │
                     ┌───────────▼────────────┐
                     │ Gated / FiLM Fusion     │
                     └───────────┬────────────┘
                                 │
             ┌───────────────────┼───────────────────┐
             ▼                   ▼                   ▼
     ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
     │ Path Head     │    │ Alpha Head    │    │ Position Head │
     │ auxiliary     │    │ edge/rank     │    │ [0, 1]        │
     └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
            │                   │                   │
            └───────────────────▼───────────────────┘
                            Backtest Engine
                                 │
                                 ▼
                   net PnL / Sharpe / drawdown
```

### 3.1 为什么保留 Path Head

Path Head 不是最终目标，但仍有价值：

- 帮助模型理解未来价格路径。
- 给 high/low/volatility 预测提供辅助监督。
- 防止 Position Head 只追逐噪声 PnL。

因此第一版采用多任务学习：

```text
价格路径预测是辅助任务
alpha/rank 是中间任务
净 PnL 是最终评测目标
```

---

## 4. 输入设计

### 4.1 Bar 粒度

SuperPnL 数据层只获取 1min K 线。更长时间窗口都从 1min bar 计算：

```text
bar_size = "1m"
feature_windows = ["5m", "15m", "30m"]
strategy_horizons = ["5m", "15m"]
```

换算规则：

```text
window_bars = window_minutes
horizon_bars = horizon_minutes
```

例如 `ret_30m` 使用过去 30 根 1min bar，`strategy_horizons=["15m"]` 表示预测未来 15 根 1min bar。

### 4.2 主 K 线输入

每个 bar 输入：

```text
open
high
low
close
volume
amount
```

建议 lookback：

```text
L = 256
```

价格不输入绝对值，使用局部归一化：

```text
price_norm = log(price / close_t)
volume_norm = log1p(volume) 的窗口 z-score
amount_norm = log1p(amount) 的窗口 z-score
```

### 4.3 特征输入

第一版特征分三类：

```text
technical features
market context features
time features
```

所有特征必须只来自 `<= t` 的历史信息。

### 4.4 固定成本配置

当前阶段不把交易成本或流动性作为模型输入。回测只保留固定配置：

```text
fixed_fee_bps
fixed_slippage_bps
edge_buffer_bps
```

这里的 bps 是单边成本：

```text
1 bps = 0.01%
10 bps = 0.10%
```

一次买入扣一边成本，一次卖出再扣一边成本。默认建议：

| 配置 | fixed_fee_bps | fixed_slippage_bps | 适用场景 |
| --- | ---: | ---: | --- |
| `zero_cost` | 0 | 0 | 只看信号毛收益上限 |
| `small_maker` | 8 | 0 | 小资金、挂单成交假设 |
| `small_taker` | 10 | 1-3 | 小资金、吃单成交假设 |

如果交易量很小，可以把 `fixed_slippage_bps` 设得很低，甚至在研究阶段设为 0。但报告必须明确这是固定成本或零成本假设，不能和真实大容量策略混淆。

---

## 5. 输出设计

### 5.1 Alpha Head

Alpha Head 输出策略定义的多个 horizon 的预测收益：

```text
pred_ret_{h} for h in strategy_horizons
```

对应真值：

```text
ret_{h} = log(exec_exit_price_{t+horizon_bars} / exec_entry_price_{t+1})
```

注意：收益真值必须使用可执行价格假设，而不是直接用同一时刻 close。

### 5.2 Edge Head

Edge 是可交易收益估计。当前阶段成本不作为模型输入，只在回测和开仓阈值中使用固定配置：

```text
edge_h = pred_ret_h
       - fixed_fee_bps
       - fixed_slippage_bps
       - edge_buffer_bps
```

第一版可以直接让模型输出：

```text
pred_edge_{h} for h in strategy_horizons
```

也可以直接使用 `pred_ret_h` 作为 score，并把固定成本只放在回测引擎里扣除。两种口径必须在实验报告中分开标注。

### 5.3 Position Head

现货不做空，所以仓位范围是：

```text
target_pos_{h} ∈ [0, 1]
```

推荐参数化：

```text
target_pos_{h} = sigmoid(raw_pos_{h})
```

更保守的执行规则：

```text
if pred_edge_{h} <= edge_threshold:
    target_pos_{h} = 0
else:
    target_pos_{h} = sigmoid(raw_pos_{h})
```

这样模型必须先证明收益足以超过策略设定的阈值，才允许开仓。

### 5.4 Path Head

辅助预测未来 K 线路径，长度覆盖策略需要的最大 horizon：

```text
open_gap[i]
close_ret[i]
upper_range[i]
lower_range[i]
volume_ret[i]
amount_ret[i]
```

其中：

```text
path_len_bars = max(horizon_bars(h) for h in strategy_horizons)
i = 1..path_len_bars
```

Path Head 的输出不直接决定交易，但用于辅助训练和诊断。

---

## 6. 交易与回测定义

### 6.1 决策时间

严格采用：

```text
t 这根 bar 收盘
        ↓
使用 <= t 的数据生成信号
        ↓
t+1 open 或 t+1 VWAP 假设成交
```

禁止：

```text
用 t close 生成信号，又用 t close 成交
```

这会造成不可执行的乐观回测。

### 6.2 现货持仓收益

现货仓位：

```text
pos_t ∈ [0, 1]
```

单步净收益：

```text
gross_pnl_t = pos_{t-1} * ret_t
cost_t = abs(pos_t - pos_{t-1}) * (fixed_fee_bps + fixed_slippage_bps) / 10000
net_pnl_t = gross_pnl_t - cost_t
```

其中：

```text
ret_t = log(exec_price_t / exec_price_{t-1})
```

### 6.3 调仓约束

为了避免模型靠高频抖动制造虚假收益，第一版建议加入：

```text
max_position = 1.0
min_trade_size = 0.05
max_turnover_per_step = 0.25
cooldown_minutes = 5
edge_threshold_bps = fixed_fee_bps + fixed_slippage_bps + edge_buffer_bps
```

这些约束必须在训练、验证、测试中一致。

---

## 7. 模型模块

### 7.1 Bar Encoder

推荐使用 Causal Dilated TCN：

```text
输入: [B, L, 6]
输出: [B, L, D]
```

建议参数：

```text
D = 128
kernel_size = 3
dilations = [1, 2, 4, 8, 16, 32, 64]
dropout = 0.05
```

### 7.2 Feature Encoder

特征也是时间序列：

```text
输入: [B, L, F]
输出: [B, L, D]
```

第一版建议用轻量 TCN，便于学习动量、波动和市场状态的历史变化。

### 7.3 Gated / FiLM Fusion

让外生特征调制 K 线 hidden：

```text
gamma, beta, gate = MLP(feature_hidden)
bar_mod = bar_hidden * (1 + tanh(gamma)) + beta
fused = bar_hidden + sigmoid(gate) * bar_mod
```

含义：

- `gamma` 控制价格表示的缩放。
- `beta` 控制价格表示的偏移。
- `gate` 控制外部特征影响强度。

---

## 8. 训练目标

### 8.1 多任务损失

第一版不建议只用 PnL loss。PnL 噪声很大，容易过拟合。

推荐：

```text
loss = w_path * path_loss
     + w_alpha * alpha_loss
     + w_rank * rank_loss
     + w_pos * position_loss
     + w_turnover * turnover_penalty
```

建议起点：

```text
w_path = 0.20
w_alpha = 0.30
w_rank = 0.25
w_pos = 0.15
w_turnover = 0.10
```

### 8.2 Alpha Loss

预测未来收益：

```text
alpha_loss = Huber(pred_ret_h, true_ret_h)
```

重点 horizon 来自策略配置：

```text
strategy_horizons
```

### 8.3 Rank Loss

PnL 依赖排序质量。每个时间 bucket 内，对多个币种做截面排序：

```text
rank_loss = pairwise_rank_loss(score_h, true_ret_h)
```

或先用 Rank-IC 作为评测指标，训练阶段用 Huber 简化。

### 8.4 Position Loss

第一版可以使用启发式最优仓位作为弱标签：

```text
target_pos_label = 1 if true_edge_h > threshold else 0
```

训练：

```text
position_loss = BCE(pred_pos_h, target_pos_label)
```

等回测链路稳定后，再考虑可微 PnL loss。

### 8.5 Turnover Penalty

抑制频繁调仓：

```text
turnover_penalty = mean(abs(pos_t - pos_{t-1}))
```

这个项建议保留，用来抑制仓位抖动；如果当前实验完全忽略成本，可以降低权重但仍应报告换手率。

---

## 9. 评测指标

### 9.1 PnL 指标

必须报告：

```text
gross_pnl
net_pnl
annualized_return
sharpe
sortino
max_drawdown
calmar
win_rate
profit_factor
```

### 9.2 执行指标

必须报告：

```text
turnover
average_position
average_holding_minutes
trade_count
fixed_fee_bps
fixed_slippage_bps
```

### 9.3 稳定性指标

必须分桶报告：

```text
by_symbol_pnl
by_month_pnl
by_market_regime_pnl
long_exposure_distribution
```

如果收益只来自单个币种或单个月份，不能认为模型稳定有效。

### 9.4 预测辅助指标

仍然保留：

```text
close_return_mae_{h}
Rank-IC_{h}
direction_hit_rate_{h}
```

但这些不是最终成功标准。

---

## 10. Baseline 设计

必须做以下 baseline：

### 10.1 No-trade baseline

始终空仓：

```text
pos = 0
```

净 PnL 应接近 0。这是检查回测费用和现金逻辑的基准。

### 10.2 Buy-and-hold baseline

对每个币或等权币篮子长期持有：

```text
pos = 1
```

用于判断模型是否只是吃到了市场 beta。

### 10.3 OHLCV-only trading model

只使用 K 线，不使用外生特征：

```text
OHLCV -> TCN -> Alpha/Position Head
```

这是判断特征是否有边际价值的核心 baseline。

### 10.4 Technical-feature trading model

加入技术特征：

```text
OHLCV + technical features -> Alpha/Position Head
```

需要注意：技术特征是 OHLCV 的派生信息，提升不一定代表新信息，只能说明归纳偏置有帮助。

### 10.5 Full-feature trading model

加入全部现货可用特征：

```text
OHLCV + technical + market context + time features
```

这是第一版主模型。

---

## 11. 数据切分

### 11.1 推荐第一轮数据

现货：

```text
OKX spot
Top20
1min
最近 6 到 12 个月
```

币池按当前成交额排序后，还必须过滤：

```text
1min 历史长度 >= 180 天 或 >= 365 天
排除稳定币和明显非交易目标
```

### 11.2 Split

必须使用时间切分：

```text
train: 前 70%
val: 中间 15%
test: 最后 15%
```

test 必须是最后一段时间，不能随机切。

### 11.3 Walk-forward

为了验证 PnL 稳定性，正式报告应增加 walk-forward：

```text
train 90 days -> test 30 days
roll forward 30 days
```

所有窗口都必须分别报告净 PnL、Sharpe、回撤和换手。

---

## 12. 防泄漏要求

任何 rolling 特征只能使用当前及过去数据：

```text
rolling_mean(x, window).at[t] 只能使用 <= t
```

禁止：

```text
center=True
未来窗口 z-score
全样本标准化后再切分
用未来成交量估计当前滑点
用 t close 生成信号并在 t close 成交
用测试集结果调 edge_threshold
```

当前阶段不使用盘口、成本、流动性特征。未来如果重新加入这些特征，必须重新检查时间戳，确保输入只来自 `<= t`。

---

## 13. 第一版实现范围

建议第一版只实现：

```text
1. 现货数据加载
2. 特征生成
3. OHLCV-only trading baseline
4. Full-feature trading model
5. Alpha/Position Head
6. 固定成本配置，可在研究阶段设为 0
7. 事件驱动回测
8. PnL、Sharpe、回撤、换手、分桶稳定性评测
```

暂不实现：

```text
永续合约
杠杆和做空
强化学习
tick 级撮合模拟
实盘下单
Transformer
LoRA
```

---

## 14. 当前建议结论

第一版采用：

```text
SuperPnL
spot only
lookback = 256
strategy_horizons = 策略自定义，例如 ["5m", "15m"]
TCN encoder
Gated / FiLM fusion
Alpha Head + Position Head + auxiliary Path Head
fixed-cost or zero-cost backtest
net PnL as primary evaluation
```

这个方案的核心不是证明模型能预测价格，而是证明模型在明确的固定成本或零成本假设下，仍然能产生稳定、可重复、非单一币种贡献的现货 PnL。
