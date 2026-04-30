# 实验与评测计划

## 1. 实验目标

第一阶段只回答一个问题：

```text
加入 5/15/30 技术因子和市场/合约因子后，未来 K 线预测是否稳定变准？
```

不要一开始追求复杂模型。先用快模型验证因子价值。

---

## 2. 数据范围

### 2.1 现货第一轮

建议：

```text
OKX spot
Top30
1min
最近 6 到 12 个月
```

理由：

- BTC/ETH 太少，截面 IC 统计意义弱。
- Top100 太杂，长尾币噪声和缺失更多。
- Top30 是训练速度、流动性和截面数量的折中。

### 2.2 永续第一轮

建议：

```text
OKX USDT perpetual
Top10 或 Top20
1min
最近 90 天到 6 个月
```

理由：

- funding 和 OI 历史覆盖有限。
- 永续 sidecar 数据质量比现货 K 线更容易出问题。
- 先用较少主流合约验证链路更稳。

---

## 3. 实验组

### 3.1 必跑实验

| 实验 | 输入 | 目的 |
| --- | --- | --- |
| `naive` | 当前价格延续 | 最低基准 |
| `ohlcv_tcn` | 只用 OHLCV | 判断纯价格序列可预测性 |
| `technical_tcn` | OHLCV + 技术因子 | 判断技术因子增益 |
| `market_tcn` | OHLCV + 技术因子 + BTC 市场因子 | 判断市场环境增益 |
| `full_factor_tcn` | 全部可用因子 | 判断完整 schema 增益 |

### 3.2 永续额外实验

| 实验 | 输入 | 目的 |
| --- | --- | --- |
| `perp_without_contract` | 不含 funding/OI/basis | 永续基础 baseline |
| `perp_with_funding` | 加 funding | 判断资金费率价值 |
| `perp_with_oi` | 加 OI | 判断持仓变化价值 |
| `perp_with_basis` | 加 basis | 判断基差价值 |
| `perp_full` | funding + OI + basis | 完整永续因子 |

---

## 4. 评测指标

### 4.1 K 线预测误差

对每个 horizon 分别评估：

```text
h1
h5
h15
h30
```

指标：

```text
close_mae
close_rmse
close_return_mae
open_mae
high_mae
low_mae
volume_log_mae
amount_log_mae
```

### 4.2 方向命中率

```text
hit_rate_h = mean(sign(pred_return_h) == sign(true_return_h))
```

需要注意：

- 命中率不是唯一标准。
- 如果收益绝对值很小，方向命中可能没有交易意义。
- 必须和 IC / Rank-IC 一起看。

### 4.3 IC / Rank-IC / ICIR

每个时间 bucket 上做截面相关：

```text
score_h = log(pred_close_h / current_close)
ret_h = log(true_close_h / current_close)
```

计算：

```text
IC = Pearson(score_h, ret_h)
Rank-IC = Spearman(score_h, ret_h)
ICIR = mean(IC) / std(IC)
```

重点看：

```text
full_factor_tcn - ohlcv_tcn
technical_tcn - ohlcv_tcn
perp_full - perp_without_contract
```

不要只看绝对 IC。

### 4.4 K 线合法率

必须为 100%：

```text
high >= max(open, close)
low <= min(open, close)
volume > 0
amount > 0
```

如果不是 100%，说明输出还原逻辑有问题。

---

## 5. 训练验证策略

### 5.1 Split

第一版建议：

```text
train: 70%
val: 15%
test: 15%
```

test 必须是最后一段时间，不能随机切。

### 5.2 Early stopping

不要只看训练 loss。验证集主指标建议：

```text
val_close_return_mae_h30
val_rank_ic_h30
val_close_return_mae_h15
```

第一版可以用组合分数：

```text
score = -val_close_return_mae_h30 + 0.2 * val_rank_ic_h30
```

如果这个过早复杂化，先用 `val_close_return_mae_h30`，但报告中必须补充 IC。

---

## 6. 决策标准

认为因子有效，至少需要满足：

```text
1. full_factor_tcn 的 close_return_mae 低于 ohlcv_tcn
2. full_factor_tcn 的 h15/h30 Rank-IC 高于 ohlcv_tcn
3. 提升不只出现在单个币种
4. 提升不只出现在单个时间段
5. naive baseline 被稳定打败
```

如果只在某个 horizon 上提升：

- h1 提升：说明短期噪声处理有用，但要看手续费和滑点。
- h5/h15 提升：优先考虑交易价值。
- h30 提升：说明趋势/状态因子有效，稳定性通常更好。

---

## 7. 建议执行顺序

### Step 1: 现货 Top30

```text
naive
ohlcv_tcn
technical_tcn
market_tcn
```

先不跑永续，避免 funding/OI/basis 数据质量干扰。

### Step 2: 现货 full factor

如果 technical / market 有增益，再跑完整现货 schema。

### Step 3: 永续 Top10

先只做：

```text
perp_without_contract
perp_full
```

如果有提升，再拆 funding/OI/basis 消融。

### Step 4: 模型升级

只有在 TCN 证明因子有效后，才考虑：

```text
quantile decoder
Transformer hybrid
pretrained backbone + LoRA
```

---

## 8. 预期耗时

粗略估计：

| 任务 | 数据规模 | 预计耗时 |
| --- | --- | --- |
| OHLCV-only TCN smoke | Top5, 30 天 | 数分钟 |
| 现货 Top30 6 个月 | 1 张高端 GPU | 0.5 到 2 小时 |
| 现货 Top30 12 个月 | 1 张高端 GPU | 1 到 4 小时 |
| 永续 Top10 90 天 | 1 张高端 GPU | 0.5 到 1 小时 |

实际耗时取决于 batch size、future_len、模型宽度和数据 IO。
