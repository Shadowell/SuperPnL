# AGENTS.md

这个项目是 FactorBarNet 的独立设计与实现仓库。

## 协作规则

1. 回答用户时使用中文。
2. 当前阶段优先完善设计文档，不急于实现训练代码。
3. 不要把原始数据、checkpoint、`artifacts/`、`.venv/` 提交进仓库。
4. 任何新增因子都必须说明是否可能产生未来信息泄漏。
5. 评测必须同时包含无因子 baseline 和有因子模型，不能只报告绝对指标。

## 项目定位

FactorBarNet 不依赖 Kronos 的 tokenizer / predictor 架构。目标是直接使用历史 OHLCV 与外生因子预测未来多根 K 线，并从预测路径中评估 h1 / h5 / h15 / h30。
