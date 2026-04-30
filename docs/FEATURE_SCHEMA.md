# Feature Schema

## 1. 设计原则

第一版特征只服务一个目标：提高现货策略 PnL。

设计原则：

- 只做现货，不使用 funding、OI、basis 等永续合约特征。
- 数据层只获取 1min K 线；特征窗口使用真实时间长度命名，例如 `5m/15m/30m`。
- `feature_windows` 是配置项，默认可以用 `["5m", "15m", "30m"]`，但策略可以改成 `["15m", "30m", "60m"]`。
- 同时覆盖收益、波动、市场环境和时间周期。
- 所有特征只能使用决策时刻 `t` 及以前的数据。
- 任何新增特征都必须说明是否可能产生未来信息泄漏。

---

## 2. OHLCV 派生技术特征

### 2.1 收益 / 动量

```text
ret_5m
ret_15m
ret_30m
```

定义：

```text
ret_{W} = log(close_t / close_{t-window_bars(W)})
```

其中 `W` 是真实时间窗口。因为数据层统一使用 1min K 线，`window_bars(W)` 就等于分钟数，例如 `ret_15m` 使用过去 15 根 1min bar。

泄漏判断：无未来信息泄漏，只使用历史 close。

### 2.2 RSI

```text
rsi_5m
rsi_15m
rsi_30m
```

定义：

```text
RSI = avg_gain / (avg_gain + avg_loss)
rsi_scaled = RSI - 0.5
```

泄漏判断：rolling 计算必须只使用 `<= t` 的收益，不能使用 centered window。

### 2.3 波动率

```text
vol_std_5m
vol_std_15m
vol_std_30m
```

定义：

```text
vol_std_{W} = rolling_std(log(close_t / close_{t-1}), window_bars(W))
```

泄漏判断：无未来信息泄漏，但标准化参数必须只在 train 上拟合。

### 2.4 均线偏离

```text
ma_dev_5m
ma_dev_15m
ma_dev_30m
```

定义：

```text
ma_dev_{W} = close_t / rolling_mean(close, window_bars(W)) - 1
```

泄漏判断：rolling mean 只能使用 `<= t`。

### 2.5 布林位置

```text
boll_z_5m
boll_z_15m
boll_z_30m
```

定义：

```text
boll_z_{W} = (close_t - MA_{W}) / (2 * STD_{W})
```

泄漏判断：MA 和 STD 都只能使用历史窗口。

### 2.6 MACD

```text
macd_5m_15m
macd_15m_30m
```

定义：

```text
macd_5m_15m = (EMA(close, window_bars(5m)) - EMA(close, window_bars(15m))) / close_t
macd_15m_30m = (EMA(close, window_bars(15m)) - EMA(close, window_bars(30m))) / close_t
```

泄漏判断：EMA 必须按时间递推计算，不允许全样本双向平滑。

---

## 3. 市场环境特征

### 3.1 BTC 市场收益

```text
market_ret_5m
market_ret_15m
market_ret_30m
```

定义：

```text
market_ret_{W} = log(BTC_close_t / BTC_close_{t-window_bars(W)})
```

泄漏判断：只使用历史 BTC close，无未来信息泄漏。

注意：如果目标币是 BTC，应禁用该特征或使用不含 BTC 的市场指数，否则 BTC 样本和其他币样本的含义不同。

### 3.2 BTC 市场波动

```text
market_vol_5m
market_vol_15m
market_vol_30m
```

定义：

```text
market_vol_{W} = rolling_std(BTC_bar_return, window_bars(W))
```

泄漏判断：rolling std 只能使用 `<= t`。

### 3.3 截面市场强度

```text
cross_section_ret_rank_5m
cross_section_ret_rank_15m
cross_section_ret_rank_30m

cross_section_vol_rank_5m
cross_section_vol_rank_15m
cross_section_vol_rank_30m
```

定义：

```text
cross_section_ret_rank_{W} = 当前币 ret_{W} 在同一时刻币池中的分位数
cross_section_vol_rank_{W} = 当前币 vol_std_{W} 在同一时刻币池中的分位数
```

泄漏判断：只能使用同一决策时刻已经可见的币池数据。不能使用未来成分股、未来成交额排序或事后筛选出的 survivor universe。

---

## 4. 时间周期特征

```text
hour_sin
hour_cos
dayofweek_sin
dayofweek_cos
```

定义：

```text
hour_sin = sin(2*pi*hour/24)
hour_cos = cos(2*pi*hour/24)
dayofweek_sin = sin(2*pi*dayofweek/7)
dayofweek_cos = cos(2*pi*dayofweek/7)
```

泄漏判断：无未来信息泄漏，时间本身在决策时刻已知。

---

## 5. 特征数量

默认窗口下：

```text
33 个左右
```

第一版不强行凑固定维度。特征维度按 `feature_windows` 和真实可用数据确定。

---

## 6. 暂不使用的特征

| 特征 | 原因 |
| --- | --- |
| funding_rate | 永续特征，当前现货阶段不使用 |
| open_interest | 永续特征，当前现货阶段不使用 |
| basis | 永续特征，当前现货阶段不使用 |
| fee_bps | 当前阶段只作为回测配置，不作为训练特征 |
| slippage_bps_est | 当前阶段只作为回测配置，不作为训练特征 |
| spread_bps | 暂不使用盘口数据 |
| depth_10bps | 暂不使用盘口数据 |
| depth_50bps | 暂不使用盘口数据 |
| order_imbalance_10bps | 暂不使用盘口数据 |
| capacity_ratio | 当前交易量很小，暂不建模容量 |
| future_realized_slippage | 未来信息，不能作为输入 |
| future_volume | 未来信息泄漏 |
| centered_rolling_* | 未来信息泄漏 |
| full_sample_zscore | 会把验证/测试分布泄漏给训练 |
| survivor_universe_rank | 会产生幸存者偏差 |
