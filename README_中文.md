# CKD 纵向动态预测：主分析模拟示例

**synthetic data仅用于演示论文分析流程和数据隔离原则，生成的性能值不代表真实研究结果，也不要求与论文数值一致。**

## 一键运行

Python 3.10测试环境，建议先建立独立环境：

```bash
python -m pip install -r requirements.txt
python run_all.py
```

依次执行01—08脚本，再自动执行基本隔离检查。本机从头运行及六项检查约218秒，版本和测试记录见`TEST_REPORT.json`。默认CPU运行，生成600名Shenzhen–Nanning源队列对象和240名独立Chongqing外部对象。源队列固定划分为420名development set和180名reserved internal test set。

## 八步流程

1. `01_generate_synthetic_data.py`：随机生成纵向变量、ART暴露、CKD时间和删失时间；6个月网格，测量位于前3个月窗口，0—5年六个年度landmarks。
2. `02_split_and_prepare_data.py`：按患者及“中心×事件”分层固定7:3划分，保存split manifest；仅开发集分五折，此时不拟合或转换测试数据。
3. `03_development_cv_models.py`：每折仅以训练患者拟合预处理；训练super-landmark Cox、super-landmark RSF、RNN和LSTM，生成开发集OOF预测。
4. `04_model_comparison_and_lock.py`：输出四模型比较和校准；按论文设计指定LSTM，不要求模拟排名中LSTM最优。仅开发数据拟合最终校准、确定风险三分位阈值，保存`model_lock.json`。
5. `05_internal_test_evaluation.py`：首次对预留测试对象应用冻结预测器，输出最终内部性能，不训练、不调优、不重新校准。
6. `06_external_validation.py`：将同一个冻结预测函数应用于独立外部对象，不使用外部结局进行参数适配。
7. `07_risk_stratification.py`：开发集阈值原样用于内部与外部对象，输出lower/intermediate/higher组人数、事件数、观察5年风险和KM风险曲线。
8. `08_model_interpretation.py`：提供冻结校准LSTM集成的简洁IG示例，包括主要预测变量、当前与历史信息、单名对象随landmark更新的风险；参考值仅来自开发折的训练患者。

## 文件位置

- `data/`：运行生成的两个人工队列表及字段字典。
- `artifacts/split_manifest.csv`、`split_manifest.json`：固定患者划分和开发折。
- `artifacts/model_lock.json`：模型配置、开发拟合患者、各项冻结文件哈希。
- `outputs/development_cv/`：开发OOF比较、校准和分层阈值，仅用于开发阶段。
- `outputs/internal_test/`：最终内部评价，均标记`dataset = internal_test`。
- `outputs/external_validation/`：独立外部验证，均标记`dataset = external_validation`。
- `outputs/risk_stratification/`：两队列风险分层表与图。
- `outputs/model_interpretation/`：IG、当前/历史信息贡献、单个对象的更新风险。

内部与外部均输出landmark-specific C-index/iAUC/IBS、六landmark均值、calibration和decision curve。查看`outputs/run_summary.json`及`outputs/logs/`可追溯执行状态。重复运行会把旧生成目录移入`.run_history/`后重新开始，不混用旧结果。

## 冻结与隔离

开发阶段保留五折OOF用于模型比较；最终内部性能只来自30%预留测试集。最终预测器是五个开发折LSTM的生存曲线均值，加开发集拟合的landmark校准参数。内部与外部共用同一预测器，不另行重拟合。

超参数在`config.json`中预先设定。简化Cox/RSF使用当前值、历史均值和变化趋势；RNN/LSTM使用截至landmark的有序历史。简洁示例保留核心设计，但不复制原研究完整神经网络结构和搜索预算，也不声称开发OOF比较是完全嵌套验证。

运行末尾的六项检查核实：患者和折无交叉；训练/预处理/校准仅使用开发对象；内部和外部评价前后冻结哈希一致；阈值确由开发预测计算；更改测试/外部结局不会改变固定对象时点的预测；概率有效、IG及未来信息检查通过。

IG仅是**预测归因，不是因果效应**；示例的“global”图来自少量明确标注的内部测试样本。所有生成数据、模型和结果均留在本地并被Git忽略。不含真实患者数据或真实模型。仓库只保留主分析，不包含补充或审稿人专项分析；旧版材料仅保留在Git历史。
