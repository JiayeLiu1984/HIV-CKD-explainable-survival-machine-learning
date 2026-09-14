# Figure legends and interpretation boundaries

All figures use Times New Roman. Titles and captions are intentionally outside the graphics. S1–S4 are provisional package-local numbers.

## Figure S1 Existing two-seed training variability

EN: C-index (A), iAUC (B), and IBS (C) for the two saved training seeds within each fixed validation fold of the final Step10E LSTM_v2. Each seed-specific value is the equal-weight mean over six landmarks. Black markers and error bars show mean ± sample SD (n=2); colored points show the individual seed-position results. Actual integer seeds differ across folds and are listed in Table S1. These are uncalibrated training-validation diagnostics, not five independent pipeline replications, not bootstrap intervals, and not fully nested validation.

CN：展示每个固定验证折内两个已有训练种子的C-index、iAUC及IBS。每个点先对六个landmark等权平均；黑点和误差线为两个种子的均值±样本标准差，彩色点为各独立训练结果。实际种子整数见数值表。该图不能证明全流程稳定，也不是置信区间图。

## Figure S2 Full history versus current information only

EN: Landmark-specific C-index (A), iAUC (B), and IBS (C) from aligned saved uncalibrated predictions of the full-history Step10E and separately trained Step13A current-information-only model. Error bars are 95% percentile intervals from 1,000 paired patient-level bootstrap replicates. The within-landmark common-cohort censoring reference and fitted predictions are fixed. Metrics use the existing 6–59.999-month grid, whose last saved prediction represents 60 months. The comparison includes removal of cumulative ART and laboratory observation/recency/change channels, not sequence history alone. Patient risk sets and absolute prediction windows differ by landmark. Existing input-timing and non-nested model-selection limitations apply.

CN：比较完整历史与另行训练的仅当前信息模型。横轴为预测时点，每个时点预测随后5年，不是同一固定终点。误差线来自1,000次患者级配对bootstrap。两组使用未校准预测，不能直接替换既往校准后的主结果。删去的还包括累计暴露和实验室辅助通道，因此应解释为更丰富时间相关输入表示的比较。

## Figure S3 Center subgroups of mixed-center OOF predictions

EN: C-index (A), iAUC (B), and IBS (C) in Shenzhen and Nanning subgroups of the existing full-history uncalibrated OOF predictions. Shaded bands indicate pointwise 95% patient-bootstrap intervals; censoring distributions are estimated within each center–landmark subgroup and held fixed during resampling. Training for each OOF fold included both centers. This is NOT leave-one-center-out validation. Differences in case mix and event incidence affect interpretation, particularly of IBS; a lower IBS does not by itself establish better transportability. Center-specific counts and events are provided in Table S3.

CN：展示现有混合中心OOF预测在深圳与南宁患者中的分层表现，阴影为逐点95%置信区间。各折训练包含两个中心，该图不是跨中心留一验证，不能据此证明向未见中心迁移的能力。IBS还受事件发生水平及病例构成影响，不能仅凭其高低对两个中心作模型优劣排名。

## Figure S4 Frozen-model clinical-domain reference occlusion

EN: Existing paired-bootstrap iAUC losses following reference occlusion of ten predefined clinical domains at landmarks 0, 1, 3 and 5 years (A–D). Positive values indicate lower discrimination after occlusion. Selected dynamic channels are replaced at active history steps by heldout-fold-excluded training-risk-set references; corresponding static/age channels are replaced when included in the domain. Laboratory value, observation, recency and change channels are replaced together. Row masks, landmark context, trained model and original Step11 calibrator remain unchanged. Intervals are the existing 1,000 paired patient-bootstrap percentile intervals. This is conditional frozen-model dependence, NOT retrained feature removal, NOT a test of adding new biomarkers, and NOT independent or causal feature effects. Axes have panel-specific scales; changes across landmarks may reflect different risk sets/history lengths.

CN：整理已有四个landmark下十个变量组遮挡后的iAUC变化及95%置信区间。正值表示遮挡后区分度下降。替换值来自排除对应验证折的训练风险集，模型和校准器冻结；结果不能解释为重新训练后的变量选择敏感性、变量独立效应或因果效应。四个面板横轴范围不同，应按数值比较而非按线段长度比较。完整C-index、IBS及风险变化指标均保留于CSV。
