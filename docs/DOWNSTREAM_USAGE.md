# Downstream Usage

本文档说明 SuperPnL 第一版模型如何从数据、配置、训练结果进入下游策略使用。

## 1. 当前默认配置

| 项 | 默认值 |
| --- | --- |
| market | OKX spot |
| universe | 成交额 Top 20 的非稳定币 `*-USDT` 现货 |
| bar_size | `1m` |
| history | 365 天 |
| lookback | 256 根 1min bar |
| strategy_horizons | `5m,15m` |
| feature_windows | `5m,15m,30m` |
| cost model | 固定费率和固定滑点，默认实验可设为 0 |
| train/val/test | 按时间切分 70% / 15% / 15% |

当前阶段不把盘口、成本、流动性作为模型输入特征。成本只在回测阶段作为配置进入 PnL 计算；如果 `fixed_fee_bps=0` 且 `fixed_slippage_bps=0`，报告必须明确标注是零成本回测。

## 2. 数据准备

远端服务器可以直接访问 OKX，因此数据下载在服务器执行：

```bash
python3 scripts/download_okx_spot_1m.py \
  --out /opt/bitpro/data/superpnl/okx_spot_1m_top20_365d \
  --top 20 \
  --days 365 \
  --workers 4 \
  --rate 8
```

下载完成后同步到本地：

```bash
rsync -az root@47.79.36.92:/opt/bitpro/data/superpnl/okx_spot_1m_top20_365d/ \
  data/okx_spot_1m_top20_365d/
```

数据目录结构：

```text
data/okx_spot_1m_top20_365d/
├── metadata.json
└── csv/
    ├── BTC-USDT.csv.gz
    ├── ETH-USDT.csv.gz
    └── ...
```

`metadata.json` 固化当次下载的币池、时间范围和 OKX candle 字段映射。下游复现实验时必须保留该文件，否则 Top20 币池可能因为 24h 成交额变化而变化。

## 3. 特征与泄漏约束

默认特征全部来自决策时刻 `t` 及以前的数据：

- OHLCV bar input：`open_rel/high_rel/low_rel/close_rel/volume_z_30m/amount_z_30m`
- 技术特征：`ret/rsi/vol_std/ma_dev/boll_z/macd`
- 市场特征：`market_ret/market_vol`
- 截面特征：`cross_section_ret_rank/cross_section_vol_rank`
- 时间特征：`hour_sin/hour_cos/dayofweek_sin/dayofweek_cos`

新增特征时必须检查两类风险：

- 未来信息泄漏：例如 future volume、centered rolling、用测试集拟合标准化参数。
- 幸存者偏差：例如用未来成交额排序得到历史币池，或训练期内使用测试期才上市/才进入 Top20 的标的。

当前实现的标准化参数只在 train split 上拟合，再应用到 val/test。

## 4. 训练与回测

本地训练入口：

```bash
PYTHONPATH=src python3 scripts/run_superpnl_experiment.py \
  --raw-dir data/okx_spot_1m_top20_365d \
  --cache-dir data/cache/okx_spot_1m_top20_365d_l256_h5_15 \
  --out-dir outputs/superpnl_top20_365d \
  --lookback 256 \
  --horizons 5,15 \
  --feature-windows 5,15,30 \
  --epochs 5 \
  --samples-per-epoch 200000 \
  --batch-size 256 \
  --hidden-dim 128 \
  --fixed-fee-bps 0 \
  --fixed-slippage-bps 0
```

训练脚本会同时跑：

- `ohlcv_tcn`：无外生因子的 OHLCV-only baseline。
- `full_feature_tcn`：OHLCV + 外生因子模型。
- `no_trade`：空仓 baseline。
- `buy_and_hold_equal_weight`：等权持有 baseline。
- `naive_momentum`：规则动量 baseline。

输出目录包含：

```text
outputs/superpnl_top20_365d/
├── REPORT.md
├── metrics.json
├── run_config.json
├── ohlcv_tcn.pt
├── ohlcv_tcn_history.json
├── ohlcv_tcn_test_predictions.npz
├── full_feature_tcn.pt
├── full_feature_tcn_history.json
└── full_feature_tcn_test_predictions.npz
```

## 5. 下游在线使用方式

每分钟在最新 1min K 线确认后执行：

1. 读取当前交易池每个 symbol 最近 `lookback=256` 根 1min K 线。
2. 用训练时完全相同的 `feature_windows` 生成特征。
3. 使用训练缓存中的 train 均值和标准差做标准化；不能用线上全量历史重新拟合。
4. 加载 `full_feature_tcn.pt`，得到每个 horizon 的 `pred_ret` 和 `pos_logit`。
5. 策略层选择 horizon，例如只用 `5m` 或在 `5m/15m` 间择优。
6. 将 `pred_ret > threshold` 或 `sigmoid(pos_logit) > threshold` 转成目标仓位。
7. 执行层按现货 long-only 规则把目标仓位限制在 `0..1`。

第一版推荐下游只消费两个字段：

```text
pred_ret_5m
pred_ret_15m
```

仓位规则先保持简单：

```text
target_pos = 1.0 if pred_ret_h > threshold else 0.0
```

后续如果要更平滑，可以改成：

```text
target_pos = clip(pred_ret_h / target_return_scale, 0.0, 1.0)
```

## 6. 稳定性检查

正式结果至少检查：

- 总 PnL 是否优于 no-trade、buy-and-hold、naive momentum、OHLCV-only。
- `full_feature_tcn` 是否在 `5m` 和 `15m` horizon 都有可解释表现。
- 测试期按月收益是否只集中在单月。
- 测试期按 symbol 收益是否只依赖单个币。
- 换手是否过高；如果加入成本后收益消失，说明当前信号不可交易。

如果这些条件不满足，下游不应直接上线，只能作为研究结果继续迭代。
