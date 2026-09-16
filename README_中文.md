# CKD 纵向动态预测代码

## 一键运行

Python 3.10，建议使用独立环境：

```bash
python -m pip install -r requirements.txt
python scripts/run_all.py
```

默认流程：**模拟数据 → 患者级7:3划分 → 开发集内五折比较与校准 → 冻结选定LSTM → 独立内部测试 → 独立外部验证 → 测试集风险分层与解释**。

## 数据与结果归属

| 数据集 | 用途 | 输出目录 |
|---|---|---|
| development set，70% | 模型构建、五折交叉验证、调优与选择、拟合预处理和校准、确定分层阈值 | `development_cv/`，仅开发阶段诊断 |
| internal test set，30% | 对选定且冻结的LSTM进行最终内部评估，不参与训练、调优、早停、模型选择或校准参数拟合 | `internal_validation/` |
| 独立外部队列 | 应用与内部测试相同的冻结预测器，不重新调参 | `external_validation/` |

默认600名模拟源队列对象划分为420名开发对象、180名内部测试对象；另生成240名外部对象。`synthetic_development.csv` 是划分前的600人，并非划分后的development set。所有记录均由随机程序生成，未使用真实患者数据。

最终内部性能仅来自预留测试集，不把开发集交叉验证指标作为最终内部结果。四类模型的开发比较另行保留，用于开发过程追溯。

示例沿用论文开发阶段已经选定的LSTM，不声称随机数据重新证实LSTM最优；RNN/LSTM采用保留的选定参数，RSF执行缩减调优。完整调优源码在 `study_sources/`。冻结预测器是五个开发折LSTM及其预处理器的集成，加上开发集拟合的校准参数；不是在全部开发患者上另行重拟合的单一模型。

`model_lock.json` 在内部评估前记录模型来源、开发患者ID及参数文件哈希。内部测试和外部验证共用同一个预测函数。完整性测试检查患者独立性、文件冻结及改变测试结局不会改变预测。

## 输出

每次运行创建独立 `results/synthetic_*` 目录，`RUN_SUMMARY.json` 记录配置、耗时与状态，`logs/` 保留日志。`internal_validation/` 输出最终内部指标、校准评价、DCA及患者级bootstrap；`figures/` 的主要分层图来自内部测试集，阈值由开发集确定并用于内部及外部对象；`interpretation/` 解释测试样本的冻结校准集成预测，参考值只来自开发数据。

## 可选分析

```bash
python scripts/run_all.py --with-sensitivity
python scripts/run_all.py --with-recalibration
python scripts/run_all.py --three-seeds
```

ART模拟重编码及三种子分析仍属于开发诊断，不能当作最终内部测试结果。外部再校准单独标记为二次适配，区别于主外部验证。D:A:D/VHA及中心轮转的历史正式源码保留，不进入默认流程。

见 [数据集—结果对应表](docs/DATASET_RESULT_MAP.md) 和 [审稿意见对应表](docs/REVIEWER_CODE_MAP.md)。旧源码命名和历史结果映射保留用于追溯，不代表当前独立测试结果。**原论文数值未重新计算，不能把旧开发评价数值直接改名为内部测试结果。** 新口径的真实论文结果需要实际预留患者的独立评价。

模拟运行使用缩减训练预算，只验证实现；不复现论文数值。不公开原始数据或真实模型权重。死亡未作为竞争事件单独建模；模型归因不代表因果作用。
