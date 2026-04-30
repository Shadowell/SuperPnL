---
license: other
library_name: pytorch
tags:
  - time-series
  - finance
  - crypto
  - pnl
  - okx
  - pytorch
---

# SuperPnL

SuperPnL 是一个面向可交易 PnL 的加密货币现货预测模型。当前上传的是第一版 OKX spot Top20 / 1min K 线模型包，推荐只使用 `15m` horizon 的实时推理结果。

Hugging Face repo:

```text
Shadowell/SuperPnL
```

## 文件

```text
model.pt
model_config.json
feature_schema.json
normalization_stats.npz
universe.json
data_contract.json
metrics_summary.json
manifest.json
superpnl_full_feature_tcn_15m_top20_20260430.tar.gz
```

下游服务应读取根目录文件。`.tar.gz` 是同一模型包的压缩备份。

## 模型配置

```text
model = full_feature_tcn
bar_size = 1m
lookback = 256
horizons = [5m, 15m]
recommended_horizon = 15m
recommended_horizon_index = 1
bar_dim = 6
feature_dim = 33
hidden_dim = 64
```

输入 shape：

```text
bar:      [batch, 256, 6]
features: [batch, 256, 33]
```

输出：

```text
pred_ret[:, 1]      -> pred_ret_15m
sigmoid(pos_logit[:, 1]) -> pos_score_15m
score_bps = pred_ret_15m * 10000
```

## 下载

CLI：

```bash
hf download Shadowell/SuperPnL \
  --local-dir /opt/bitpro/artifacts/superpnl \
  --exclude "*.tar.gz"
```

Python：

```python
from huggingface_hub import snapshot_download

model_dir = snapshot_download(
    repo_id="Shadowell/SuperPnL",
    local_dir="/opt/bitpro/artifacts/superpnl",
    ignore_patterns=["*.tar.gz"],
)
```

## 加载模型

下游需要有 `SuperPnLModel` 结构定义，和训练仓库 `src/superpnl/model.py` 保持一致。源码仓库：

```text
https://github.com/Shadowell/SuperPnL
```

```python
import json
from pathlib import Path

import numpy as np
import torch

from superpnl.model import SuperPnLModel

model_dir = Path("/opt/bitpro/artifacts/superpnl")

config = json.loads((model_dir / "model_config.json").read_text())
stats = np.load(model_dir / "normalization_stats.npz")
checkpoint = torch.load(model_dir / "model.pt", map_location="cpu")

model = SuperPnLModel(
    bar_dim=config["bar_dim"],
    feature_dim=config["feature_dim"],
    num_horizons=config["num_horizons"],
    hidden_dim=config["hidden_dim"],
    dropout=config["dropout"],
    use_features=True,
)
model.load_state_dict(checkpoint["model"])
model.eval()
```

## 实时推理流程

每根 1min K 线确认后执行：

1. 维护 `universe.json` 中 Top20 symbol 的 rolling 1min K 线窗口。
2. 至少保留 `lookback=256` 根；建议 `warmup_bars=300`，覆盖 30m rolling 特征。
3. 只使用 `timestamp <= t` 的已确认 K 线生成特征。
4. 按 `feature_schema.json` 的顺序生成 `bar` 和 `features`。
5. 使用 `normalization_stats.npz` 的训练集 mean/std 标准化。
6. 批量推理 `[n_symbols, 256, dim]`。
7. 读取 `recommended_horizon_index=1` 的 `pred_ret_15m`。
8. 策略层再做 threshold、top-k、最短持仓、冷却时间、再平衡间隔和成本约束。

不要直接用：

```text
target_pos = 1 if pred_ret_15m > 0 else 0
```

这个逐分钟翻仓规则在零成本下表现好，但在真实手续费下会被高换手打穿。

## 特征泄漏约束

- rolling / EMA / rank 都只能使用 `<= t` 的历史数据。
- 标准化参数只能使用模型包里的训练集 mean/std，不能在线重新拟合。
- 截面 rank 必须使用同一 timestamp 上已经可见的 universe 数据。
- 不能使用 future volume、future slippage、centered rolling、全样本 z-score。
- 不能用未来成交额或未来上市状态重新选择历史币池。

## 当前实验结果

主实验是零成本回测：

```text
fixed_fee_bps = 0
fixed_slippage_bps = 0
threshold_bps = 0
```

| model | horizon | zero-cost total_return | sharpe | max_drawdown | turnover | conclusion |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| full_feature_tcn | 5m | -14.49% | -3.249 | -19.33% | 0.3167 | 不可用 |
| full_feature_tcn | 15m | +62.46% | 9.099 | -5.79% | 0.2472 | 当前唯一主线 |
| full_feature_tcn | 30m | -5.80% | -1.532 | -7.95% | 0.1594 | 暂不推荐 |

重要限制：

- 当前收益是零成本 test PnL，不能直接当作实盘净收益。
- 加入 maker/taker 成本后，逐分钟翻仓策略会被成本打穿。
- 下游必须先做低换手策略层，再用真实成本重新回测。
- 本模型不是投资建议，也不是自动实盘交易系统。

## 推荐下游架构

```text
BitPro 实时 1min K线
        ↓
SuperPnL feature builder
        ↓
SuperPnL model inference
        ↓
pred_ret_15m / pos_score_15m
        ↓
低换手策略层
        ↓
BitPro broker / execution
```

历史 prediction `.npz` 文件没有上传到 Hugging Face，也不应作为模拟盘或实盘信号源。
