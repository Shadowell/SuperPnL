# FactorBarNet 设计说明书

## 1. 背景

现有 Kronos 路线的核心是：

```text
K 线连续值 -> tokenizer -> 离散 token -> predictor -> 未来 token -> decode -> 未来 K 线
```

这条路线的优点是能复用预训练模型，但它也带来几个问题：

- 外生因子很难自然接入 token generation 流程。
- 如果目标是提高未来 K 线精度，训练目标容易绕远。
- tokenizer 的重建误差会限制最终 K 线精度。
- 评测容易变成收益头 IC，而不是完整 K 线路径质量。

FactorBarNet 重新定义目标：**直接预测未来 K 线路径**。模型不预测 token，也不以单一收益值作为主输出，而是直接输出未来 `1..30` 根 K 线的相对变化。

---

## 2. 设计目标

### 2.1 主目标

输入历史窗口：

```text
过去 L 根 1min K 线:
open, high, low, close, volume, amount

过去 L 根外生因子:
技术因子、市场因子、永续合约因子
```

输出未来路径：

```text
未来 30 根 1min K 线:
open, high, low, close, volume, amount
```

从完整路径中自然得到：

```text
h1  = t+1
h5  = t+5
h15 = t+15
h30 = t+30
```

### 2.2 约束目标

- 预测出的 K 线必须合法：`high >= max(open, close)`，`low <= min(open, close)`。
- 不能使用未来外生因子，所有输入只允许来自 `t` 时刻及以前。
- 第一版必须训练快，便于做因子消融。
- 评测必须能解释因子是否真的带来增益。

### 2.3 非目标

第一版不做以下事情：

- 不做 tokenizer。
- 不做自回归 token 采样。
- 不做大规模预训练。
- 不做多交易所统一大模型。
- 不直接上复杂 Transformer 作为第一版 baseline。

---

## 3. 总体架构

推荐第一版架构：

```text
                 ┌──────────────────────┐
OHLCV history ──▶│ Bar Encoder           │
                 │ Causal Dilated TCN    │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
Factors history ─▶ Factor Encoder        │
                 │ MLP / Causal TCN      │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Gated / FiLM Fusion   │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ 30-step K-line Decoder│
                 └──────────┬───────────┘
                            │
                            ▼
        future open/high/low/close/volume/amount
```

第一版优先用 **TCN**，不是 Transformer。原因是 1min crypto 的短周期预测更依赖局部趋势、波动和动量；TCN 的训练速度、显存占用和调参复杂度都更适合先验证因子是否有效。

---

## 4. 输入设计

### 4.1 主 K 线输入

每个时间点输入 6 维：

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

含义是模型查看过去 256 根 1min K 线，约 4 小时 16 分钟。

### 4.2 价格输入归一化

不建议直接输入绝对价格。推荐使用窗口内最后一个 close 或窗口均值做局部归一化：

```text
price_norm = log(price / close_t)
volume_norm = log1p(volume) 的窗口 z-score
amount_norm = log1p(amount) 的窗口 z-score
```

这样模型更容易跨币种学习，不会被 BTC 和小币种的绝对价格尺度影响。

### 4.3 外生因子输入

外生因子只使用历史窗口内的数据：

```text
exog[0:L]
```

不能把未来 `t+1..t+30` 的真实外生因子喂给模型。否则评测会包含未来信息泄漏。

### 4.4 现货和永续

现货模型输入：

```text
OHLCV + 技术因子 + 市场因子 + 时间因子
```

永续模型输入：

```text
OHLCV + 技术因子 + 市场因子 + 时间因子 + funding/OI/basis
```

第一版建议现货和永续分开训练，不建议直接混在一个模型里。原因是永续合约有 funding、open interest、basis，这些变量在现货中不存在。

---

## 5. 输出设计

### 5.1 不直接预测绝对 OHLC

直接预测未来绝对价格有两个问题：

- 不同币种价格尺度差异大。
- 很难保证 K 线合法。

所以模型输出相对参数：

```text
open_gap[i]
close_ret[i]
upper_range[i]
lower_range[i]
volume_ret[i]
amount_ret[i]
```

其中 `i = 1..30`。

### 5.2 K 线还原公式

给定上一根 close：

```text
open_i  = prev_close * exp(open_gap_i)
close_i = open_i * exp(close_ret_i)
high_i  = max(open_i, close_i) * exp(softplus(upper_range_i))
low_i   = min(open_i, close_i) * exp(-softplus(lower_range_i))
volume_i = prev_volume * exp(volume_ret_i)
amount_i = prev_amount * exp(amount_ret_i)
```

也可以把 `amount_i` 简化为：

```text
amount_i = close_i * volume_i
```

第一版建议先单独预测 `amount_ret`，评测后再决定是否改成派生值。

### 5.3 为什么这样设计

这个参数化天然保证：

```text
high_i >= max(open_i, close_i)
low_i <= min(open_i, close_i)
volume_i > 0
amount_i > 0
```

它也让模型学习更稳定，因为模型预测的是相对变化，而不是绝对价格。

---

## 6. 模型模块

### 6.1 Bar Encoder

推荐使用 Causal Dilated TCN：

```text
输入: [B, L, 6]
输出: [B, L, D]
```

建议参数：

```text
D = 128 或 256
kernel_size = 3
dilations = [1, 2, 4, 8, 16, 32, 64]
dropout = 0.05
```

TCN 的感受野约等于：

```text
1 + (kernel_size - 1) * sum(dilations)
```

对于上面的配置，感受野约 255，刚好覆盖 lookback 256。

### 6.2 Factor Encoder

外生因子可以用轻量 TCN 或 MLP：

```text
输入: [B, L, F]
输出: [B, L, D]
```

第一版建议用轻量 TCN，因为因子也是时间序列，不只是当前截面值。

建议参数：

```text
D = 128
kernel_size = 3
dilations = [1, 2, 4, 8, 16, 32]
dropout = 0.05
factor_dropout = 0.05
```

`factor_dropout` 的作用是随机屏蔽部分因子，防止模型过度依赖某几个不稳定指标。

### 6.3 Gated / FiLM Fusion

不要简单 concat。推荐让因子调制价格 hidden：

```text
gamma, beta, gate = MLP(factor_hidden)
bar_mod = bar_hidden * (1 + tanh(gamma)) + beta
fused = bar_hidden + sigmoid(gate) * bar_mod
```

含义：

- `gamma` 控制价格表示的缩放。
- `beta` 控制价格表示的偏移。
- `gate` 控制因子影响强度。

这样比简单拼接更适合“因子增强价格预测”的目标。

### 6.4 Multi-Horizon Decoder

Decoder 一次性输出未来 30 根 K 线，不做自回归逐根生成：

```text
输入: fused[:, -1, :]
输出: [B, 30, 6]
```

推荐加入 horizon embedding：

```text
horizon_emb[i] 表示第 i 根未来 K 线
decoder_input[i] = concat(context, horizon_emb[i])
```

这样模型知道自己预测的是 t+1、t+5 还是 t+30。

### 6.5 是否需要 Transformer

第一版不建议直接用 Transformer。

原因：

- 训练更慢。
- 显存更高。
- 超参数更多。
- 当前阶段最重要的是验证因子是否有效。

如果 TCN baseline 已经证明因子有效，第二版可以升级为：

```text
TCN local encoder + Transformer global encoder + gated factor fusion
```

---

## 7. LoRA 判断

如果完全新建 FactorBarNet，**第一版不需要 LoRA**。

LoRA 的适用场景是：

```text
已有大模型 / 预训练模型
想低成本适配新数据
```

FactorBarNet 第一版是从头训练的小模型，参数量预计远小于 Kronos，不需要 LoRA。

如果后续引入预训练 Transformer backbone，才考虑 LoRA：

```text
冻结 backbone
在 attention / MLP linear 层加 LoRA
正常训练 factor encoder + fusion + decoder
```

但第一版建议避免这个复杂度。

---

## 8. 训练目标

### 8.1 主损失

主损失直接对齐未来 K 线：

```text
L_close = Huber(pred_close_ret, true_close_ret)
L_open = Huber(pred_open_gap, true_open_gap)
L_range = Huber(pred_upper/lower_range, true_upper/lower_range)
L_volume = Huber(pred_volume_ret, true_volume_ret)
L_amount = Huber(pred_amount_ret, true_amount_ret)
```

总损失：

```text
L = w_close * L_close
  + w_open * L_open
  + w_range * L_range
  + w_volume * L_volume
  + w_amount * L_amount
```

建议第一版权重：

```text
w_close = 2.0
w_open = 1.0
w_range = 0.5
w_volume = 0.3
w_amount = 0.3
```

理由是 close 对收益、方向、IC 最关键。

### 8.2 多 horizon 加权

未来 30 根不能完全等权。短周期和关键周期应更重：

```text
loss = 0.30 * path_loss_all
     + 0.25 * loss_h1
     + 0.20 * loss_h5
     + 0.15 * loss_h15
     + 0.10 * loss_h30
```

也可以反过来给 h30 更高权重。如果你的主交易周期仍偏 h30，可以使用：

```text
loss = 0.30 * path_loss_all
     + 0.15 * loss_h1
     + 0.15 * loss_h5
     + 0.20 * loss_h15
     + 0.20 * loss_h30
```

第一版建议用第二组，因为你之前 h30 更稳定。

### 8.3 分位数输出

为了表达不确定性，可以输出 `p10/p50/p90` 三个分位数：

```text
输出 shape = [B, 30, 6, 3]
```

主预测使用 `p50`，训练用 pinball loss。

第一版可以先做 deterministic 输出：

```text
输出 shape = [B, 30, 6]
```

等 deterministic baseline 跑通后，再加 quantile。

---

## 9. 评测指标

### 9.1 K 线误差

必须评估：

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

### 9.2 方向指标

对 h1 / h5 / h15 / h30 分别计算：

```text
direction_hit_rate = sign(pred_return_h) == sign(true_return_h)
```

注意：命中率不是充分指标。它容易被弱波动样本影响，所以必须和 IC、误差一起看。

### 9.3 IC 指标

对每个 horizon 计算：

```text
IC
Rank-IC
ICIR
```

`score` 来自预测路径的未来收益：

```text
score_h = log(pred_close[t+h] / close[t])
```

真值：

```text
ret_h = log(true_close[t+h] / close[t])
```

### 9.4 K 线合法率

虽然参数化理论上保证合法，仍建议记录：

```text
valid_bar_rate
high_violation_count
low_violation_count
negative_volume_count
```

如果这些指标不为 0，说明实现有 bug。

### 9.5 交易模拟指标

第一版不做复杂策略，但可以做简单 top/bottom 分层：

```text
每个时间点按 pred_return_h 排序
做多 top 20%
做空 bottom 20%
计算 next-h return spread
```

这可以帮助判断 IC 是否有实际交易意义。

---

## 10. Baseline 设计

必须做以下 baseline：

### 10.1 Naive baseline

假设未来价格不变：

```text
future_close = current_close
future_volume = current_volume
```

这是所有模型必须打败的最低基准。

### 10.2 OHLCV-only TCN

只使用 K 线，不使用任何外生因子：

```text
OHLCV -> TCN -> Decoder
```

这是判断因子是否有用的核心 baseline。

### 10.3 Technical-factor TCN

加入技术因子：

```text
OHLCV + ret/rsi/boll/macd -> prediction
```

### 10.4 Full-factor TCN

加入全部可用因子：

```text
OHLCV + 技术因子 + 市场因子 + 合约因子
```

现货没有合约因子，使用现货 full schema。

### 10.5 LightGBM / XGBoost baseline

对 h1 / h5 / h15 / h30 分别训练 tabular baseline。

它不预测完整路径，但能判断因子本身是否有线性/非线性 alpha。

---

## 11. 数据切分

### 11.1 推荐第一轮数据

现货：

```text
Top30 crypto spot
1min
最近 6 到 12 个月
```

永续：

```text
Top10 或 Top20 USDT perpetual
1min
最近 90 天到 6 个月
```

永续不要第一轮直接上 Top100，因为 funding / OI / basis 覆盖和质量更容易出问题。

### 11.2 Split

建议：

```text
train: 前 70%
val: 中间 15%
test: 最后 15%
```

也可以沿用 block interleave validation，但 test 必须严格在时间后段。

### 11.3 防泄漏要求

任何 rolling 因子必须只使用当前及过去数据：

```text
rolling_mean(x, window).at[t] 只能使用 <= t
```

禁止：

```text
center=True
未来窗口 z-score
全样本标准化后再切分
使用未来 funding / OI / basis
```

---

## 12. 训练配置建议

第一版建议：

```text
lookback = 256
future_len = 30
batch_size = 256 或按显存调整
epochs = 30
optimizer = AdamW
lr = 1e-3 for new small model
weight_decay = 1e-4
scheduler = cosine decay with warmup
dropout = 0.05
early_stop_patience = 5
mixed_precision = true
```

如果训练不稳定：

```text
lr 降到 3e-4
gradient_clip = 1.0
increase dropout to 0.1
reduce D from 256 to 128
```

---

## 13. 第一版实现范围

建议第一版只实现：

```text
1. 数据加载
2. 5/15/30 因子生成
3. OHLCV-only TCN baseline
4. FactorBarNet TCN + factor fusion
5. deterministic 30-step decoder
6. K 线误差 + h1/h5/h15/h30 IC 评测
```

暂不实现：

```text
quantile output
Transformer encoder
LoRA
online serving
multi-exchange normalization
```

这样可以最快回答一个核心问题：

```text
新因子是否真的提高未来 K 线预测准确性？
```

---

## 14. 后续升级路线

### Phase 1: 快速验证

```text
TCN + deterministic decoder
Top30 spot
h1/h5/h15/h30 评测
```

### Phase 2: 因子消融

```text
OHLCV only
OHLCV + technical
OHLCV + technical + market
OHLCV + technical + market + perp extras
```

### Phase 3: 不确定性预测

```text
p10/p50/p90 quantile decoder
pinball loss
calibration evaluation
```

### Phase 4: 永续合约

```text
Top10/Top20 swap
funding/OI/basis coverage report
spot vs swap separate models
```

### Phase 5: Transformer / Hybrid

如果 TCN 已经证明因子有效，再测试：

```text
TCN local encoder + Transformer global encoder
```

---

## 15. 当前建议结论

我建议第一版采用：

```text
FactorBarNet-TCN
lookback 256
future_len 30
direct K-line path prediction
5/15/30 因子窗口
Gated / FiLM factor fusion
deterministic output
Top30 spot first
```

这个方案比直接上 Transformer 或 LoRA 更快、更清晰，也更适合先验证因子价值。
