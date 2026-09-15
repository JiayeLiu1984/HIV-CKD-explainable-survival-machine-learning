# CKD 纵向动态预测代码

默认主流程：**随机数据生成 → 预处理 → 四类模型训练与比较 → LSTM 外部验证 → 风险分层与解释**。

## 一键运行

使用 Python 3.10，在独立环境安装后运行：

```bash
python -m pip install -r requirements.txt
python scripts/run_all.py
```

默认生成600名模拟源队列对象和240名独立外部对象。源队列按患者7:3划分，70%开发集内进行五折比较，预留内部测试集不进入主OOF评价。super-landmark Cox/RSF使用历史摘要，RNN/LSTM使用截至预测时点的有序纵向历史。外部验证固定使用论文开发阶段选定的LSTM及开发集拟合的预处理和校准参数。

本机默认主流程约49秒，6项完整性检查通过；不同硬件耗时不同。默认2轮神经网络训练、20次bootstrap为快速运行设置，不是论文训练预算。

每次输出到独立的 `results/synthetic_*`：`evaluation/` 为四模型比较，`external_validation/` 为外部验证，`figures/` 为风险分层与示意图，`interpretation/` 为归因及验证。`RUN_SUMMARY.json` 记录参数、每步耗时与状态，`logs/` 保留完整日志。

## 可选分析

```bash
python scripts/run_all.py --with-sensitivity
python scripts/run_all.py --with-recalibration
python scripts/run_all.py --three-seeds
```

敏感性分析与外部再校准不进入默认流程。正式ART分析、中心轮转、三种子分析及landmark 0的D:A:D/VHA比较仍保留于 `study_sources/`。这些完整论文源码需要配套本地输入，不能把所有历史替代版本直接串联执行。早期基线RSF/SHAP代码保存在Git历史 `4538bff`，已移出当前目录。

## 论文可追溯性与数据说明

见 [结果—代码映射](docs/RESULT_CODE_MAP.md)、[审稿意见—代码映射](docs/REVIEWER_CODE_MAP.md) 和 [完整分析说明](docs/REPRODUCTION.md)。

仅公开独立随机生成的虚构数据，不含原始或脱敏患者记录、真实模型权重或患者级预测。模拟事件分布经过人为设置，便于小样本测试；模拟结果不能复现论文数值。完整论文预算、校准和集成源码保留在 `study_sources/`，快速示例的评价和解释范围较小。

模型为单一CKD事件的右删失预测，未将死亡单独作为竞争事件。IG和遮蔽分析解释模型预测，不代表因果作用。原始临床数据提取、eGFR输入与上游插补仍需原始记录核验；代码示例不替代这些核查。
