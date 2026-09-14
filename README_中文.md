# CKD 动态预测代码包

本包按旧 GitHub 项目的编号模块形式整理。新版主线为 Landmark Cox、Landmark RSF、RNN、LSTM-v2；旧版 CKD-ML.ipynb 的全部代码单元及原五模块另存于 `legacy_baseline/`，不与新版结果混用。

## 数据

`data/` 仅含程序随机生成的虚构记录：600名开发对象、240名外部对象，以及供旧版代码测试的数值型基线表。生成器不读取原始数据，也不使用真实患者训练出的生成模型。为使小样本五折测试可计算各时间窗指标，模拟事件被有意分布到六个预测窗口，不能解释为真实发病率。

不提供原始/脱敏患者表、真实患者预测、真实模型权重、拟合预处理器或原 notebook 的输出。重新运行会在本地生成仅来自合成数据的中间张量和测试模型。

## 运行

```bash
python -m pip install -r requirements.txt
python scripts/run_all.py
python scripts/run_all.py --three-seeds
```

默认训练轮数和 bootstrap 次数较少，用于检查代码能否执行。不能把这些测试数值填入文章。正式训练、调参、校准、配对bootstrap、IG、外部验证和三折三种子中心轮转源码在 `study_sources/`；输入要求和结果对应关系见 `docs/REPRODUCTION.md`、`docs/RESULT_CODE_MAP.md`。

`run_all.py` 每次创建独立输出目录。单独按00—07顺序运行模块时，共用 `results/synthetic_run`；如需换配置，请使用新的 `CKD_WORKDIR`，避免重复使用旧输出。

ART 敏感性分析按六个年度 landmark 等权汇总，原模型和校准冻结的评价与二次校准结果分别报告。Landmark 0 的改编 D:A:D/VHA 比较代码位于 `study_sources/clinical_scores/`；该比较存在缺失评分项、代理变量和基线源日期待核实等限制。上游临床数据提取、插补与时间对齐仍需原始记录核实，合成数据测试不替代这些研究核查。
