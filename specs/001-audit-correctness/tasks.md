# Tasks: 审查问题修复

**Input**: spec.md 与 plan.md。每项行为缺陷包含先失败后通过的回归验证。

## Phase 1: Setup
- [x] T001 初始化 .specify/、建立 spec.md/plan.md/tasks.md，创建 GitHub issues #1–#11。

## Phase 2: US1 / US2 回测和数据
- [x] T002 [P] [US2] 修复 #1 src/superpnl/data.py 下一开盘收益，tests/test_data.py 验证跳空。
- [x] T003 [US2] 修复 #3 src/superpnl/data.py 分区 purge，测试未来价格扰动。
- [x] T004 [US2] 修复 #4 src/superpnl/data.py 分钟网格校验，测试缺口与缺失币种。
- [x] T005 [P] [US1] 修复 #6 src/superpnl/metrics.py 初始净值回撤，tests/test_metrics.py。
- [x] T006 [US1] 修复 #2 src/superpnl/training.py 组合账本，tests/test_backtest.py；同步分币归因。
- [x] T007 [US1] 修复 #7 src/superpnl/training.py 动量反标准化及阈值测试。
- [x] T008 [US2] 修复 #8 data.py 与 run_superpnl_experiment.py 的缓存版本、配置及源指纹校验，tests/test_cache.py。

## Phase 3: US3 模型交付
- [x] T009 [P] [US3] 修复 #9 scripts/package_superpnl_model.py 从实际 checkpoint 取架构，tests/test_package.py。
- [x] T010 [US3] 修复 #10 scripts/package_superpnl_model.py 失败安全替换，测试旧包保留。
- [ ] T011 [US3] 修复 #5 data.py/training.py/package_superpnl_model.py 数据契约及不匹配拒绝测试。

## Phase 4: Completion
- [x] T012 修复 #11 pyproject.toml matplotlib 依赖，验证绘图 CLI。
- [ ] T013 同步 README.md 和 docs/ 口径，标记历史结果需重算；添加并运行 CPU CLI 集成测试。
- [ ] T014 全量测试、独立审查、converge 与 git diff --check；确保每 issue 独立 commit。

## Dependencies & Execution Order
T002→T003→T004→T008；T005→T006→T007；T009→T010。
T011 依赖 T008、T007、T010；T012 可独立；T013/T014 最后。
仅不同文件的任务并行；根 agent 串行提交，其他 agent 不操作 Git index。
