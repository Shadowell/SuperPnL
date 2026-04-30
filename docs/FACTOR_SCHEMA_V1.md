# Factor Schema v1

## 1. 设计原则

第一版因子只服务一个目标：提高未来 K 线预测准确性。

设计原则：

- 统一使用 `5/15/30` 窗口。
- 不使用 K 线形态因子，例如上影线、下影线、实体比例。
- 不使用未来信息。
- 现货和永续共享技术因子，永续额外加入合约结构因子。
- 第一版控制因子数量，不追求一次性堆满。

---

## 2. 现货因子

### 2.1 收益 / 动量

```text
ret_5
ret_15
ret_30
```

定义：

```text
ret_N = log(close_t / close_{t-N})
```

解释：

- `ret_5` 表示过去 5 分钟动量。
- `ret_15` 表示过去 15 分钟趋势。
- `ret_30` 表示接近预测主周期的历史趋势。

### 2.2 RSI

```text
rsi_5
rsi_15
rsi_30
```

定义：

```text
RSI = avg_gain / (avg_gain + avg_loss)
rsi_scaled = RSI - 0.5
```

解释：

- 大于 0 表示近期上涨强于下跌。
- 小于 0 表示近期下跌强于上涨。
- 使用 `5/15/30` 是为了避免 `rsi_14` 这种孤立窗口。

### 2.3 波动率

```text
vol_std_5
vol_std_15
vol_std_30
```

定义：

```text
vol_std_N = rolling_std(log(close_t / close_{t-1}), N)
```

解释：

- 衡量过去 N 分钟价格波动强度。
- 波动率不是方向信号，但对预测 high/low range 很重要。

### 2.4 均线偏离

```text
ma5_dev
ma15_dev
ma30_dev
```

定义：

```text
maN_dev = close_t / rolling_mean(close, N) - 1
```

解释：

- 表示当前价格相对均线偏高还是偏低。
- 对短期回归和趋势延续都有帮助。

### 2.5 布林位置

```text
boll_z_5
boll_z_15
boll_z_30
```

定义：

```text
boll_z_N = (close_t - MA_N) / (2 * STD_N)
```

解释：

- 接近 0：价格在布林中轨附近。
- 大于 0：价格偏上。
- 小于 0：价格偏下。
- 绝对值越大，代表偏离越明显。

### 2.6 MACD

```text
macd_5_15
macd_15_30
```

定义：

```text
macd_5_15 = (EMA(close, 5) - EMA(close, 15)) / close_t
macd_15_30 = (EMA(close, 15) - EMA(close, 30)) / close_t
```

解释：

- `macd_5_15` 表示极短期趋势相对中短期趋势的强弱。
- `macd_15_30` 表示中短期趋势相对 30 分钟趋势的强弱。
- 不使用传统 `12/26/9`，因为窗口不符合本项目统一尺度。

### 2.7 市场环境

```text
market_ret_30
market_vol_30
```

定义：

```text
market_ret_30 = log(BTC_close_t / BTC_close_{t-30})
market_vol_30 = rolling_std(BTC_1min_return, 30)
```

解释：

- 用 BTC 作为市场基准。
- 对大多数 crypto，BTC 环境会影响个币短周期走势。

### 2.8 时间周期

```text
hour_sin
hour_cos
```

定义：

```text
hour_sin = sin(2*pi*hour/24)
hour_cos = cos(2*pi*hour/24)
```

解释：

- 表示一天内的交易时段。
- crypto 24/7 交易，但不同时段流动性不同。

---

## 3. 永续合约额外因子

### 3.1 资金费率

```text
funding_rate_z
```

定义：

```text
funding_rate_z = rolling_zscore(funding_rate, window)
```

解释：

- funding 高，说明多头拥挤或市场愿意为做多支付成本。
- funding 低或负，说明空头压力较强。
- funding 本身频率低，不强行做 5/15/30。

### 3.2 持仓变化

```text
oi_change_5
oi_change_15
oi_change_30
```

定义：

```text
oi_change_N = log(open_interest_t / open_interest_{t-N})
```

解释：

- OI 上升表示资金进入合约市场。
- OI 下降表示仓位退出。
- 和价格方向结合后可以区分趋势增仓、反弹减仓等状态。

### 3.3 基差

```text
basis_z_5
basis_z_15
basis_z_30
```

定义：

```text
basis = swap_close / spot_close - 1
basis_z_N = (basis_t - rolling_mean(basis, N)) / rolling_std(basis, N)
```

解释：

- basis 高，表示合约相对现货偏贵。
- basis 低，表示合约相对现货偏便宜。
- 对永续合约短期预测有代表性。

---

## 4. 因子数量

现货：

```text
21 个因子
```

永续：

```text
28 个因子
```

第一版不强行凑 32 维。新项目没有 Kronos 的 `n_exog=32` 历史包袱，因子维度可以按真实 schema 设定。

---

## 5. 暂不使用的因子

| 因子 | 原因 |
| --- | --- |
| `upper_shadow` | K 线形态因子，暂不使用 |
| `lower_shadow` | K 线形态因子，暂不使用 |
| `body_ratio` | K 线形态因子，暂不使用 |
| `amplitude` | K 线形态因子，暂不使用 |
| `rsi_14` | 窗口不统一 |
| `atr_14` | 窗口不统一 |
| `macd_12_26_9` | 窗口不统一 |
| `roc_*` | 与 `ret_*` 信息重复 |
| `vwap_dev` | amount/volume 口径可能不一致，放到 v2 |
| `obv_z` | 解释成本高，放到 v2 |
