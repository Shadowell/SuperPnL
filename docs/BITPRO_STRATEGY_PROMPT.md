# BitPro Strategy Prompt

本文档是一份可直接复制给 BitPro 项目编码代理的策略实现提示词。

目标：在 BitPro 当前 `BaseStrategy` 框架内，接入 SuperPnL 打包后的模型包，做 **实时 1min K 线推理**，输出 `pred_ret_15m`，再由策略层执行现货 long-only、低换手、含成本可评估的交易逻辑。

注意：

- 这份提示词面向 **BitPro 项目**，不是 SuperPnL 训练仓库本身。
- 主方案必须是实时模型推理，不允许把历史 test prediction 文件当作模拟盘或实盘信号。
- 历史预测文件最多只能用于离线一致性测试或回归测试。

## SuperPnL 模型包

SuperPnL 训练仓库已经提供打包脚本：

```bash
PYTHONPATH=src python3 scripts/package_superpnl_model.py --force
```

本次模型包输出：

```text
artifacts/superpnl_full_feature_tcn_15m_top20_20260430/
artifacts/superpnl_full_feature_tcn_15m_top20_20260430.tar.gz
```

模型包已经上传到 Hugging Face：

```text
https://huggingface.co/Shadowell/SuperPnL
```

BitPro 下游服务推荐直接下载 Hugging Face 模型仓库：

```bash
hf download Shadowell/SuperPnL \
  --local-dir /opt/bitpro/artifacts/superpnl \
  --exclude "*.tar.gz"
```

模型包内容：

```text
model.pt
model_config.json
feature_schema.json
normalization_stats.npz
universe.json
data_contract.json
metrics_summary.json
manifest.json
README.md
```

BitPro 实现时应从模型包加载：

- `model.pt`：PyTorch 权重。
- `model_config.json`：`lookback=256`、`bar_dim=6`、`feature_dim=33`、`hidden_dim=64`、`horizons=[5,15]`、`recommended_horizon_index=1`。
- `feature_schema.json`：bar feature 和 factor feature 的精确顺序。
- `normalization_stats.npz`：训练集拟合的 `bar_mean/bar_std/feature_mean/feature_std`。
- `universe.json`：SuperPnL `BTC-USDT` 和 BitPro `BTC/USDT` 的 symbol 映射。
- `data_contract.json`：实时推理契约和防未来泄漏约束。

模型包不应提交到普通 git 仓库。BitPro 可以通过本地路径、部署 artifact 目录或环境变量加载，例如：

```text
SUPERPNL_MODEL_PACKAGE_DIR=/opt/bitpro/artifacts/superpnl
```

## 使用方式

把下面整段 prompt 粘贴到 BitPro 项目的新任务中执行。

````text
你现在在 BitPro 项目中实现一个 SuperPnL 实时推理低换手现货策略。请先阅读并遵守：

1. AGENTS.md
2. README.md
3. docs/spec.md
4. docs/progress.md
5. docs/contracts/module_map_v2.md
6. docs/strategy_development_guide.md
7. backend/app/core/execution/base_strategy.py
8. backend/app/strategies/kairos_30m_horizon_dca_strategy.py
9. backend/app/services/strategy_registry.py
10. data/seed/strategies.json

目标：
基于 SuperPnL 已训练好的模型包，在 BitPro 的 `BaseStrategy` 框架里实现一个实时 1min K 线推理策略。策略每分钟使用最新确认 K 线生成和训练一致的特征，加载 `full_feature_tcn` 模型实时输出 `pred_ret_15m`，再按阈值、Top-K、最短持仓、冷却时间和再平衡间隔生成现货 long-only 目标仓位。

不要重新训练模型。不要读取历史 test prediction 文件作为模拟盘或实盘信号。不要写旧函数式 `strategy(ctx)` / `setup(ctx)`。不要直接使用 Backtrader、CCXT、数据库裸调用或交易所 API。策略必须是一套 `BaseStrategy` 代码，能被 BitPro 回测、模拟盘、实盘同构运行。

严格禁止修改 BitPro 策略引擎核心框架，包括：

- `BaseStrategy`
- broker / execution engine
- backtest engine
- 数据库 schema
- 现有策略行为
- 交易所 API 封装

除非发现当前框架完全无法支持该策略，否则只能通过新增 strategy、service、registry entry 和 seed config 接入。如果确实需要框架级修改，必须先停止并说明原因，不得自行改动。

背景：

- SuperPnL 结果显示：
  - `full_feature_tcn_15m` 零成本 test total_return = +62.46%
  - `5m` 不可用
  - `30m` 当前 PnL 为负
  - 主要问题是逐分钟 `pred_ret > 0` 翻仓导致换手极高，加入手续费后收益被成本打穿
- 当前只做 OKX spot，不做永续、不做杠杆、不做借币做空
- 目标不是复现零成本最高收益，而是在实时模型推理基础上降低换手，让策略在含成本下仍可能可交易

SuperPnL 模型包：

```text
SUPERPNL_MODEL_PACKAGE_DIR=/opt/bitpro/artifacts/superpnl
```

模型包包含：

```text
model.pt
model_config.json
feature_schema.json
normalization_stats.npz
universe.json
data_contract.json
metrics_summary.json
manifest.json
README.md
```

如果本地没有该模型包，请不要 mock 预测，不要生成随机信号，不要 fallback 到 momentum。应在最终说明中明确“缺少 SuperPnL 模型包，策略无法产生实时信号”。

需要实现的新代码：

- `backend/app/services/superpnl_model_inference_service.py`
- `backend/app/services/superpnl_feature_builder.py`
- `backend/app/strategies/superpnl_15m_low_turnover_strategy.py`
- 更新 `backend/app/services/strategy_registry.py`
- 更新 `data/seed/strategies.json`
- 更新 `docs/progress.md`
- 必要时更新 `docs/spec.md`

策略类要求：

- 类名：`SuperPnL15mLowTurnoverStrategy`
- 必须继承 `BaseStrategy`
- 必须实现 `async def on_init(self) -> None`
- 必须实现 `async def on_bar(self, bar: BarData) -> None`
- 可选实现 `async def on_warmup_bar(self, bar: BarData) -> None`
- 只允许新增一个 `BaseStrategy` 子类
- 不允许旧函数式 API
- 不允许在策略类里直接读模型文件、预测文件、网络、交易所或数据库
- 预测不可用时必须显式跳过交易并输出诊断日志

策略 key：

```text
superpnl_15m_low_turnover
```

注册方式：

```python
from app.strategies.superpnl_15m_low_turnover_strategy import SuperPnL15mLowTurnoverStrategy

_BASE_STRATEGY_REGISTRY["superpnl_15m_low_turnover"] = SuperPnL15mLowTurnoverStrategy
```

种子配置写入 `data/seed/strategies.json`，示例：

```json
{
  "name": "SuperPnL 15m 实时推理低换手现货策略",
  "description": "加载 SuperPnL full_feature_tcn 模型包，基于实时 1min K 线生成 pred_ret_15m，并按阈值、Top-K、最短持仓、冷却时间和再平衡间隔降低换手。现货 long-only，不做杠杆或做空。",
  "strategy_key": "superpnl_15m_low_turnover",
  "config": {
    "strategy_key": "superpnl_15m_low_turnover",
    "timeframe": "1m",
    "horizon": "15m",
    "warmup_bars": 300,

    "model_package_dir": "${SUPERPNL_MODEL_PACKAGE_DIR}",
    "signal_provider": "superpnl_model_inference",
    "signal_horizon": "15m",

    "threshold_bps": 10,
    "top_k": 3,
    "rebalance_interval_bars": 15,
    "min_holding_bars": 30,
    "cooldown_bars": 30,

    "max_position_per_symbol": 0.2,
    "max_total_position": 0.6,
    "allow_cash": true,

    "fee_bps": 8,
    "slippage_bps": 0,

    "strategy_diagnostic_ws": true,
    "strategy_diagnostic_every_n_bars": 1
  },
  "exchange": "okx",
  "symbols": [
    "BTC/USDT",
    "ETH/USDT",
    "DOGE/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "PEPE/USDT",
    "TRX/USDT",
    "XAUT/USDT",
    "BIO/USDT",
    "PENGU/USDT",
    "PI/USDT",
    "ZKJ/USDT",
    "TRUMP/USDT",
    "SUI/USDT",
    "FIL/USDT",
    "ADA/USDT",
    "APE/USDT",
    "CHZ/USDT",
    "LINK/USDT",
    "LTC/USDT"
  ],
  "script_file": null
}
```

实时模型推理服务：

实现 `backend/app/services/superpnl_model_inference_service.py`，提供统一接口：

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class SuperPnLSignal:
    symbol: str
    timestamp_ms: int
    horizon: str
    pred_ret: float
    score_bps: float
    pos_score: float
    source: str


class SuperPnLModelInferenceService:
    async def initialize(self, model_package_dir: str) -> None:
        ...

    async def update_bar(self, bar: BarData) -> None:
        ...

    async def predict_timestamp(self, timestamp_ms: int, horizon: str = "15m") -> dict[str, SuperPnLSignal]:
        ...

    async def get_signal(
        self,
        symbol: str,
        timestamp_ms: int,
        horizon: str = "15m",
    ) -> SuperPnLSignal | None:
        ...
```

服务要求：

- 从 `model_package_dir` 加载模型包，不从历史 prediction `.npz` 文件加载交易信号。
- 使用 `model_config.json` 实例化模型结构。
- 使用 `model.pt` 加载权重。
- 使用 `feature_schema.json` 的特征顺序构造输入。
- 使用 `normalization_stats.npz` 做标准化。
- 使用 `universe.json` 做 `BTC/USDT` 与 `BTC-USDT` 映射。
- `symbol` 必须兼容 BitPro 的 `BTC/USDT` 和 SuperPnL 的 `BTC-USDT`。
- 如果模型包不存在、权重加载失败、特征不足、timestamp 对不上或 symbol 不在 universe 中，返回 `None` 并输出诊断日志。
- 不允许生成假信号。
- 不允许 fallback 到 momentum/template/random/synthetic。

实时特征构造：

实现 `backend/app/services/superpnl_feature_builder.py`，严格对齐模型包里的 `feature_schema.json`。

每个已确认 1min bar 到达后：

1. 维护 universe 内所有 symbol 的 rolling bar buffer。
2. 至少需要 `lookback=256` 根 1min K 线。
3. 为了计算 `30m` rolling 特征，建议 warmup 至少 `lookback + 30 = 286`，seed 中设 `warmup_bars=300`。
4. 只使用 `<= timestamp_ms` 的数据。
5. 不能使用当前未确认 K 线。
6. 不能使用未来 volume、未来 slippage、centered rolling、全样本 z-score。
7. 标准化只能使用 `normalization_stats.npz`，不能在线重新拟合。

必须构造的 bar inputs：

```text
open_rel
high_rel
low_rel
close_rel
volume_z_30m
amount_z_30m
```

必须构造的 factor inputs：

```text
ret_5m, ret_15m, ret_30m
rsi_5m, rsi_15m, rsi_30m
vol_std_5m, vol_std_15m, vol_std_30m
ma_dev_5m, ma_dev_15m, ma_dev_30m
boll_z_5m, boll_z_15m, boll_z_30m
macd_5m_15m, macd_15m_30m
cross_section_ret_rank_5m, cross_section_vol_rank_5m
cross_section_ret_rank_15m, cross_section_vol_rank_15m
cross_section_ret_rank_30m, cross_section_vol_rank_30m
market_ret_5m, market_vol_5m
market_ret_15m, market_vol_15m
market_ret_30m, market_vol_30m
hour_sin, hour_cos
dayofweek_sin, dayofweek_cos
```

注意：

- `market_ret_*` 和 `market_vol_*` 使用 BTC 作为市场基准。
- `cross_section_*_rank_*` 必须使用同一 timestamp 上 universe 内已可见的 symbol 计算。
- 如果某个 timestamp 的 Top20 bar 不齐，优先跳过该 timestamp 的组合推理，不要填未来数据。
- 模型输入 shape：

```text
bar:      [n_symbols, 256, 6]
features: [n_symbols, 256, 33]
```

推理输出：

```python
pred_ret, pos_logit = model(bar_tensor, feature_tensor)
pred_ret_15m = pred_ret[:, recommended_horizon_index]
pos_score_15m = sigmoid(pos_logit[:, recommended_horizon_index])
score_bps = pred_ret_15m * 10000
```

策略逻辑：

1. 只处理 `timeframe="1m"` 的已收盘 K 线。
2. 多 symbol 运行时，`on_bar` 会按 symbol 被调用。策略维护每个 symbol 的最新 bar、最新信号、持仓状态、持仓开始 bar、冷却结束 bar。
3. 每次 `on_bar` 先把 bar 交给 `SuperPnLModelInferenceService.update_bar(bar)`。
4. 当同一 timestamp 的 universe bar 足够且到达再平衡点时，调用 `predict_timestamp(timestamp_ms, horizon="15m")` 批量生成信号。
5. 每 `rebalance_interval_bars` 才允许做一次组合级再平衡。
6. 再平衡时：
   - 对所有已有最新信号的 symbol 排序
   - 只保留 `pred_ret > threshold_bps / 10000`
   - 选择 `pred_ret` 最高的 `top_k`
   - long-only，不做空
   - 目标仓位范围 `[0, 1]`
   - 单币目标仓位不超过 `max_position_per_symbol`
   - 组合总仓位不超过 `max_total_position`
   - 若候选少于 top_k，剩余资金保持 cash
7. 最短持仓：
   - 已持仓但未达到 `min_holding_bars` 时，不允许平仓或降低仓位
8. 冷却：
   - 某 symbol 平仓后进入 `cooldown_bars`
   - 冷却期内不能重新买入该 symbol
9. 下单：
   - 使用 `await self.buy(symbol, qty)`
   - 使用 `await self.sell(symbol, qty)` 或 `await self.close_position(symbol)`
   - 不直接调用 broker 以外接口
   - 买入数量由目标 USDT 名义 / 当前 close 计算
   - 估算账户权益时优先使用 `self.state.positions["_capital"]` + 当前持仓市值
10. 成本：
   - BitPro broker 会处理真实/模拟成交成本时，策略层不用重复扣 PnL
   - 策略诊断日志中必须记录本次调仓估算成本：

```text
estimated_cost_bps = fee_bps + slippage_bps
estimated_turnover = abs(target_position - current_position)
```

诊断日志：

参考 `Kairos30mHorizonDcaStrategy._maybe_emit_bar_diagnostic`，通过：

```python
await self.broadcast_strategy_channel(payload)
```

输出结构化诊断。

至少包含：

```json
{
  "type": "bar_diag",
  "decision": "buy_filled / sell_filled / skip_no_signal / skip_model_not_ready / skip_below_threshold / skip_rebalance_interval / skip_min_holding / skip_cooldown / rebalance",
  "decision_label": "中文说明",
  "summary": "人能直接看懂的一句话",
  "symbol": "BTC/USDT",
  "bar_ts_ms": 1234567890,
  "close": 123.45,
  "pred_ret": 0.0012,
  "pred_ret_bps": 12.0,
  "pos_score": 0.56,
  "threshold_bps": 10,
  "rank": 1,
  "target_position": 0.2,
  "current_position": 0.0,
  "max_total_position": 0.6,
  "top_k": 3,
  "rebalance_interval_bars": 15,
  "min_holding_bars": 30,
  "cooldown_bars": 30,
  "model_package_dir": "/opt/bitpro/artifacts/superpnl_full_feature_tcn_15m_top20_20260430"
}
```

必须实现的决策枚举：

```python
DECISION_LABELS = {
    "warm_up_history": "历史K线不足，继续预热",
    "skip_model_package_missing": "未交易：SuperPnL 模型包不存在",
    "skip_model_not_ready": "未交易：SuperPnL 模型尚未就绪",
    "skip_no_signal": "未交易：SuperPnL 实时信号不可用",
    "skip_missing_universe_bar": "未交易：同一时间点币池K线不完整",
    "skip_below_threshold": "未交易：预测收益低于阈值",
    "skip_rebalance_interval": "未交易：未到再平衡时间",
    "skip_min_holding": "未卖出：未达到最短持仓时间",
    "skip_cooldown": "未买入：仍在冷却期",
    "skip_qty_zero": "未交易：下单数量为0",
    "rebalance": "组合再平衡",
    "buy_filled": "买入成交",
    "sell_filled": "卖出成交",
    "broker_error": "下单失败",
}
```

禁止事项：

- 不允许读取 `full_feature_tcn_test_predictions.npz` 作为模拟盘或实盘信号。
- 不允许在策略类中直接加载 `model.pt`，模型加载必须封装在 service。
- 不允许策略类直接计算复杂特征，特征构造必须封装在 feature builder。
- 不允许为了让策略有交易而生成 mock / random / synthetic / momentum fallback 信号。
- 不允许用 test split 调参。
- 不允许修改策略引擎核心框架。

历史预测文件的唯一允许用途：

- 可以在独立测试脚本或单元测试中，用历史 prediction `.npz` 对比实时 feature builder + model inference 在同一历史窗口上的输出是否一致。
- 该测试不能接入策略模拟盘/实盘信号源。
- 如果实现这个测试，请命名为 `tests/test_superpnl_inference_parity.py` 或独立脚本，不要放进策略类。

参数搜索：

请不要把参数搜索写进策略类。策略类只执行给定 config。

如果要做搜索，请新增独立脚本：

```text
scripts/sweep_superpnl_strategy.py
```

搜索要求：

- 只用 val split 调参
- test split 只评估一次
- 搜索使用实时推理服务或离线重放服务，但不能直接用 test 调参

搜索参数：

```text
threshold_bps: [0, 2, 5, 10, 20, 30, 50]
top_k: [1, 2, 3, 5, 10]
rebalance_interval_bars: [5, 15, 30, 60]
min_holding_bars: [15, 30, 60, 120]
cooldown_bars: [0, 15, 30, 60]
max_position_per_symbol: [0.1, 0.2, 0.3, 0.5]
max_total_position: [0.3, 0.5, 0.8, 1.0]
fee_bps: [8]
slippage_bps: [0, 1, 2]
```

评估指标：

必须输出：

- total_return
- annualized_return
- sharpe
- sortino
- max_drawdown
- calmar
- turnover
- trade_count
- average_position
- average_holding_minutes
- win_rate
- profit_factor
- monthly_returns
- by_symbol_pnl
- by_symbol_trade_count
- by_symbol_avg_position

对照组：

必须同时输出：

- no_trade
- buy_and_hold_equal_weight
- 原始 naive：每分钟 `pred_ret_15m > 0` 翻仓
- 新 BitPro 低换手策略层

成功标准：

- 在 `fee_bps=8`, `slippage_bps=0` 下 test total_return > 0
- trade_count 相比原始 naive 下降至少 80%
- max_drawdown 不高于 buy-and-hold
- by_symbol_pnl 不能只靠 1-2 个 symbol
- 参数必须来自 val split，test 只评估一次

需要改的地方：

1. 新增 `backend/app/services/superpnl_model_inference_service.py`
2. 新增 `backend/app/services/superpnl_feature_builder.py`
3. 新增 `backend/app/strategies/superpnl_15m_low_turnover_strategy.py`
4. 更新 `backend/app/services/strategy_registry.py`
5. 更新 `data/seed/strategies.json`
6. 更新 `docs/progress.md`
7. 如新增配置或架构说明，更新 `docs/spec.md`
8. 如该策略需要上线展示，说明需要执行远端 seed sync：

```bash
export BITPRO_SEED_SSH=user@production-host
./scripts/seed_strategies_remote.sh
```

如果当前没有 SSH / 生产权限，请在 `docs/progress.md` 标记“生产 strategies 表待同步”，不要声称线上已生效。

验证：

必须至少运行：

```bash
python3 -m compileall -q backend/app
./scripts/check.sh
```

如果 `./scripts/check.sh` 因既有前端 lint 或环境问题失败，需要在最终说明中明确失败原因，并说明后端编译是否通过。

交付要求：

- 小步提交
- 修改完成后提交并 push 到 GitHub
- 最终说明包含：
  - 新增文件
  - 策略 key
  - 模型包路径和加载方式
  - seed 配置
  - 如何运行回测/模拟
  - 如何确认正在使用实时模型推理，而不是历史 prediction 文件
  - 验证命令结果
  - 若生产 DB 未同步，明确待执行命令
````

## 设计约束摘要

给执行方的关键提醒：

- 不要重新训练 SuperPnL。
- 不要用 `test` 调参。
- 不要让策略类直接读模型文件或预测文件，模型加载必须封装在 inference service。
- 不要把历史 prediction `.npz` 当作模拟盘或实盘信号源。
- 不要在 BitPro 策略里直接调用 OKX / CCXT / DB。
- 不要生成 mock 预测。
- 不要修改 BitPro 策略引擎核心框架。
- 先证明 `15m` 实时推理信号在 `maker 8bps` 成本下还能不能活。
- 先解决换手，再谈实盘。

## 关联 SuperPnL 结果

主结果来自：

```text
outputs/superpnl_top20_365d_l256_h5_15_hd64_e3/REPORT.md
docs/TOP20_12M_EXPERIMENT_REPORT.md
docs/REPRODUCIBILITY_AND_TOP20_UNIVERSE.md
artifacts/superpnl_full_feature_tcn_15m_top20_20260430/metrics_summary.json
```

核心结果：

| model | horizon | zero-cost test return | conclusion |
| --- | ---: | ---: | --- |
| full_feature_tcn | 5m | -14.49% | 不可用 |
| full_feature_tcn | 15m | +62.46% | 当前唯一主线 |
| full_feature_tcn | 30m | -5.80% | 暂不作为主线 |

当前问题不是模型完全没有信号，而是默认逐分钟翻仓规则换手过高。BitPro 策略层的第一目标是通过阈值、Top-K、再平衡间隔、最短持仓、冷却时间和仓位上限把换手压下来。
