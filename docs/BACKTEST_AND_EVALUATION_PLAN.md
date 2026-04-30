# 回测与评测计划

## 1. 实验目标

第一阶段只回答一个问题：

```text
在明确的固定成本或零成本假设下，SuperPnL 是否能比 no-trade、buy-and-hold 和 OHLCV-only 模型产生更稳定的现货 PnL？
```

不要一开始追求复杂模型。先用严格的时间切分验证信号是否有交易价值。当前阶段不引入盘口、成本、流动性训练特征。

---

## 2. 数据范围

### 2.1 现货第一轮

建议：

```text
OKX spot
Top20
1min
最近 6 到 12 个月
```

理由：

- BTC/ETH 太少，截面排序统计意义弱。
- Top100 长尾币缺失和噪声更多。
- Top20 是训练速度、流动性和截面数量的折中。
- 币池必须过滤 1min 历史长度不足的刚上市币，建议最低 `>= 180 days`，正式训练优先 `>= 365 days`。

### 2.2 暂不做永续

第一版不纳入：

```text
funding
open interest
basis
leverage
short selling
```

避免永续合约数据质量和资金费率口径干扰现货 PnL 框架验证。

---

## 3. 回测假设

### 3.1 决策与成交

严格采用：

```text
t 收盘后生成信号
t+1 open 或 t+1 VWAP 假设成交
```

禁止：

```text
t close 生成信号
t close 成交
```

### 3.2 固定成本配置

当前交易量很小，不把成本作为训练特征。回测只保留固定配置：

```text
fixed_fee_bps
fixed_slippage_bps
edge_buffer_bps
```

这些都是单边 bps。买入扣一次，卖出再扣一次。

建议先跑三档：

| 配置 | fixed_fee_bps | fixed_slippage_bps | 用途 |
| --- | ---: | ---: | --- |
| `zero_cost` | 0 | 0 | 看信号毛收益上限 |
| `small_maker` | 8 | 0 | 小资金、挂单成交假设 |
| `small_taker` | 10 | 1-3 | 小资金、吃单成交假设 |

研究阶段可以把 `fixed_slippage_bps` 设为 0。如果连手续费也设为 0，报告中必须标注为零成本回测。

### 3.3 仓位限制

现货 long-only：

```text
target_pos ∈ [0, 1]
```

建议第一版约束：

```text
max_position = 1.0
max_turnover_per_step = 0.25
min_trade_size = 0.05
cooldown_minutes = 5
```

所有实验必须使用同一套交易约束，否则 PnL 不可比。

---

## 4. 实验组

### 4.1 必跑实验

| 实验 | 输入 | 目的 |
| --- | --- | --- |
| `no_trade` | 始终空仓 | 检查回测现金和费用逻辑 |
| `buy_and_hold_equal_weight` | 等权长期持有 Top20 | 判断是否只吃市场 beta |
| `naive_momentum_rule` | 简单 `ret_30m` 阈值策略 | 传统规则 baseline |
| `ohlcv_tcn_trader` | 只用 OHLCV | 判断纯价格序列交易价值 |
| `technical_tcn_trader` | OHLCV + 技术特征 | 判断 OHLCV 派生特征是否改善交易 |
| `full_feature_tcn_trader` | OHLCV + 技术 + 市场 + 时间特征 | 判断完整 schema 增益 |

### 4.2 固定成本敏感性实验

如果当前实验不考虑成本，可以先跳过成本敏感性。正式报告建议至少补一组小固定成本：

```text
zero_cost
small_maker
small_taker
```

如果 `zero_cost` 盈利但 `small_maker` 或 `small_taker` 不盈利，需要明确说明该信号目前只证明了毛收益潜力，还没有证明真实可交易净收益。

---

## 5. 评测指标

### 5.1 PnL 指标

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

核心比较对象：

```text
full_feature_tcn_trader - ohlcv_tcn_trader
full_feature_tcn_trader - buy_and_hold_equal_weight
full_feature_tcn_trader - naive_momentum_rule
```

不要只看 full feature 的绝对 PnL。

### 5.2 交易行为

必须报告：

```text
turnover
trade_count
average_position
average_holding_minutes
fixed_fee_bps
fixed_slippage_bps
```

如果使用零成本回测，必须在结果表中明确显示 `fixed_fee_bps = 0` 和 `fixed_slippage_bps = 0`。

### 5.3 稳定性分解

必须分解：

```text
by_symbol_pnl
by_month_pnl
by_horizon_pnl
by_market_regime_pnl
```

认为模型有效，不能只靠单个币种、单个月份或单个行情段。

### 5.4 辅助预测指标

保留但不作为最终目标：

```text
close_return_mae_{h}
Rank-IC_{h}
direction_hit_rate_{h}
```

如果 IC 提升但 PnL 下降，优先相信 PnL 回测，并检查换手。

---

## 6. 训练验证策略

### 6.1 时间切分

第一版建议：

```text
train: 前 70%
val: 中间 15%
test: 最后 15%
```

test 必须是最后一段时间，不能随机切。

### 6.2 验证集选择指标

不要只看训练 loss。验证集主指标建议：

```text
val_net_pnl
val_sharpe
val_max_drawdown
val_turnover
val_rank_ic_{h}
```

第一版模型选择可以用：

```text
score = val_net_pnl
      + 0.2 * val_sharpe
      - 0.2 * abs(val_max_drawdown)
      - 0.1 * val_turnover
```

如果这个过早复杂化，可以先用 `val_net_pnl`，但报告中必须补充回撤和换手。

### 6.3 Walk-forward

正式结果必须补充 walk-forward：

```text
train 90 days
validate 15 days
test 30 days
roll forward 30 days
```

每个窗口单独报告结果，再汇总均值、中位数和最差窗口。

---

## 7. 决策标准

认为模型有交易价值，至少需要满足：

```text
1. full_feature_tcn_trader 的 test net PnL 高于 no_trade、buy-and-hold 和 ohlcv_tcn_trader
2. full_feature_tcn_trader 的 Sharpe 高于 buy-and-hold
3. max_drawdown 不显著劣于 buy-and-hold
4. 若做固定成本敏感性，small_maker 或 small_taker 下结果不应完全崩溃
5. 收益不只来自单个币种
6. 收益不只来自单个月份
7. 换手率和平均持仓时间与策略 horizon 匹配
```

如果只满足 IC 提升，但净 PnL 没有提升，则不能认为策略有效。

---

## 8. 建议执行顺序

### Step 1: 回测引擎 smoke test

```text
no_trade
buy_and_hold_equal_weight
naive_momentum_rule
```

先验证现金、仓位、费用、滑点和调仓逻辑。

### Step 2: OHLCV-only trader

```text
OHLCV -> TCN -> Alpha/Position Head
```

验证模型能否比规则策略更好。

### Step 3: 加技术特征

```text
OHLCV + technical features
```

观察净 PnL、换手和回撤是否改善。

### Step 4: 加市场和时间特征

```text
OHLCV + technical + market + time
```

这是第一版主实验。

### Step 5: Walk-forward 和固定成本敏感性

只有单次 test 通过后，才做 walk-forward。固定成本敏感性可以从 zero_cost、small_maker、small_taker 三档开始。

---

## 9. 预期耗时

粗略估计：

| 任务 | 数据规模 | 预计耗时 |
| --- | --- | --- |
| 回测引擎 smoke | Top5, 30 天 | 数分钟 |
| OHLCV-only trader | Top20, 6 个月 | 0.5 到 2 小时 |
| full feature trader | Top20, 6 个月 | 1 到 3 小时 |
| walk-forward | Top20, 12 个月 | 视窗口数量线性增加 |

实际耗时取决于 batch size、`strategy_horizons`、模型宽度、数据 IO 和回测事件粒度。
