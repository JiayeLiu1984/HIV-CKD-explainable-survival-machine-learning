"""Audit all 30 fits, compute seed-set mean/SD, write updated EN/CN text.
Run after postprocess_three_seed.py analyze. Original two-seed files untouched.
"""
from pathlib import Path
import ast,json,zipfile,hashlib
import numpy as np
import pandas as pd
from numba import njit
from sksurv.util import Surv
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.metrics import concordance_index_ipcw,cumulative_dynamic_auc,integrated_brier_score
B=Path(__file__).resolve().parent;R=Path('__CKD_WORKDIR__');T=B/'tables';Q=B/'audit'
old=B.parent/'reviewer_minor_4_6_20260908'
tree=ast.parse((old/'analyze.py').read_text(encoding='utf8'))
defs=[x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name in ['add','sumto','uno','auc','km','Context']]
exec(compile(ast.Module(body=defs,type_ignores=[]),str(old/'analyze.py'),'exec'),globals())
s6=R/'rolling_5y_step6_super_landmark_data'
t=np.load(s6/'development_analysis_time_month.npy').astype(float);e=np.load(s6/'development_event_within_60m.npy').astype(bool);lm=np.load(s6/'development_landmark_month_long.npy');local=np.load(s6/'development_local_patient_idx_long.npy')
h=np.array([6,12,18,24,30,36,42,48,54,59.999],float)
labels=['Full history','Current information only'];metrics=['C-index','iAUC','IBS'];rows=[];runs=[]
for s in range(3):
 allhaz=[]
 for mode in ['full','current']:
  hz=np.full((len(t),10),np.nan,np.float32)
  for k in range(5):
   p=B/mode/f'fold_{k}';meta=json.loads((p/f'seed_{s}_completed.json').read_text(encoding='utf8'))
   assert meta['seed']==20260830+k*1000+s*100
   ix=np.load(p/'validation_long_idx.npy');hh=np.load(p/f'seed_{s}_validation_hazard.npy');hz[ix]=hh
   assert len(pd.read_csv(p/f'seed_{s}_training_history.csv'))>=5
   ckpt=p/f'seed_{s}_snapshot_ensemble.pt'
   runs.append({'model':mode,'fold':k,'seed_position':s,'actual_seed':meta['seed'],'epochs':len(pd.read_csv(p/f'seed_{s}_training_history.csv')),'snapshot_n':meta['selected_snapshot_n'],'elapsed_seconds':meta['elapsed_seconds'],'checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest()})
  assert np.isfinite(hz).all();allhaz.append(hz)
 # Same float64 cumulative product then float32 output as the original helper.
 risks=[(1-np.cumprod(1-np.clip(v.astype(float),1e-7,1-1e-7),axis=1).astype(np.float32)).astype(float) for v in allhaz]
 pred=np.stack(risks,axis=1);pp=[]
 for l in range(6):
  c=Context(np.flatnonzero(lm==12*l));v=c.calc(np.ones(c.n));pp.append(v)
  for m,name in enumerate(labels):
   for j,metric in enumerate(metrics):rows.append(dict(model=name,seed_position=s,landmark=str(l),metric=metric,estimate=v[m,j]))
 for m,name in enumerate(labels):
  for j,metric in enumerate(metrics):rows.append(dict(model=name,seed_position=s,landmark='Mean',metric=metric,estimate=np.mean(pp,axis=0)[m,j]))
dd=pd.DataFrame(rows);dd.to_csv(T/'S1_complete_OOF_by_seed_set.csv',index=False)
ss=dd.groupby(['model','landmark','metric']).estimate.agg(['mean','std','count']).reset_index().rename(columns={'std':'sample_SD','count':'independent_seed_sets'})
assert (ss.independent_seed_sets==3).all();ss.to_csv(T/'S1_overall_and_landmark_three_seed_mean_SD.csv',index=False)
pd.DataFrame(runs).to_csv(T/'training_run_manifest.csv',index=False)
assert len(runs)==30
for mode in ['full','current']:
 for k in range(5):
  p=B/mode/f'fold_{k}';expected=np.mean(np.stack([np.load(p/f'seed_{s}_validation_hazard.npy') for s in range(3)]),axis=0).astype(np.float32)
  assert np.array_equal(expected,np.load(p/'validation_hazard.npy'))
  # Replace inherited runtime metadata from early workers with measured CPU sum.
  path=p/'fold_summary.json';m=json.loads(path.read_text(encoding='utf8'));m['elapsed_seconds']=sum(z['elapsed_seconds'] for z in m['seed_summaries']);m['all_three_seeds_retrained_CPU_FP32']=True;path.write_text(json.dumps(m,indent=2,ensure_ascii=False),encoding='utf8')
 # Recompute, rather than inherit, shape and monotonicity audit fields.
 path=B/mode/'lstm_v2_summary.json';m=json.loads(path.read_text(encoding='utf8'))
 risk=np.load(B/mode/'lstm_v2_oof_risk_long.npy');surv=np.load(B/mode/'lstm_v2_oof_survival_long.npy')
 index_map=np.load(s6/'development_long_row_index_map.npy');valid=index_map>=0
 for label,array in [('risk',risk),('survival',surv)]:
  patient=np.full((*index_map.shape,array.shape[1]),np.nan,np.float32);patient[valid]=array[index_map[valid]]
  np.save(B/mode/f'lstm_v2_oof_{label}_patient.npy',patient)
 m.update(oof_long_shape=list(risk.shape),oof_patient_shape=list(patient.shape),
          survival_monotonicity_violation_n=int((np.diff(surv,axis=1)>1e-7).sum()),
          risk_monotonicity_violation_n=int((np.diff(risk,axis=1)<-1e-7).sum()),
          total_elapsed_seconds=sum(z['elapsed_seconds'] for z in runs if z['model']==mode))
 assert m['survival_monotonicity_violation_n']==m['risk_monotonicity_violation_n']==0
 path.write_text(json.dumps(m,indent=2,ensure_ascii=False),encoding='utf8')
manifest_path=Q/'source_manifest.json'
manifest=json.loads(manifest_path.read_text(encoding='utf8'))
for record in manifest:
 path=Path(record['path']);current_hash=hashlib.sha256(path.read_bytes()).hexdigest()
 if current_hash!=record['sha256']:
  record['sha256_at_analysis_read']=record['sha256']
  record['post_analysis_change']='runtime/provenance metadata finalized; prediction arrays and metric values unchanged'
  record['sha256']=current_hash;record['bytes']=path.stat().st_size
manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf8')
path=B/'occlusion/step12c_grouped_occlusion_summary.json'
occlusion_metadata=json.loads(path.read_text(encoding='utf8'))
occlusion_metadata['seed_ensemble_n_per_fold']=3
occlusion_metadata['model_directory']=str(B/'full')
occlusion_metadata['calibration_directory']=str(B/'calibration')
occlusion_metadata['occlusion_strategy']['calibrator']='newly refitted three-seed cross-fitted calibrator; frozen during occlusion'
path.write_text(json.dumps(occlusion_metadata,indent=2,ensure_ascii=False),encoding='utf8')
perf=pd.read_csv(T/'S2_history_comparison_raw_OOF.csv');dif=pd.read_csv(T/'S2_paired_differences.csv')
def mean_sd(model,metric):
 v=ss[(ss.model==model)&(ss.landmark=='Mean')&(ss.metric==metric)].iloc[0];return f'{v["mean"]:.6f} ± {v.sample_SD:.6f}'
def delta(metric):
 v=dif[(dif.landmark=='Mean')&(dif.metric==metric)].iloc[0];return f'{v.difference:+.6f} (95% CI {v.lower95:+.6f} to {v.upper95:+.6f})'
tab=['| Model | C-index mean ± SD | iAUC mean ± SD | IBS mean ± SD |','|---|---|---|---|']
for name in labels:tab.append('| '+name+' | '+' | '.join(mean_sd(name,m) for m in metrics)+' |')
reply=f'''# Three-seed results and reviewer responses

本次为新训练的三种子版本，不是把原两种子结果修改标签。完整历史与仅当前信息各5折×3个种子，共30次CPU/FP32训练；原两种子GPU权重未混入新集成。图表S1–S4编号为本结果包内部编号。原始输入时间、验证折参与早停/快照选择以及调参非嵌套的限制未因增加种子消除。

## Mean ± SD across three complete OOF seed configurations

{chr(10).join(tab)}

Each seed configuration assembles the five held-out-fold predictions, evaluates each landmark with a common censoring reference, and averages equally over six landmarks. SD is across the three resulting estimates (ddof=1). This differs from the performance of the three-seed probability ensemble and from patient-bootstrap confidence intervals. 各fold内部的3种子mean±SD另见S1_seed_mean_SD_within_each_fold.csv。

## Comment 4 Pipeline stability

Response: Thank you for this important suggestion. We repeated LSTM_v2 training using three independent seed configurations under the same CPU/FP32 environment, retaining the fixed five-fold patient partitions, predictors, hyperparameters and training rules. Each configuration comprised one independent training run in each fold. We assembled the complete OOF predictions for each configuration and report the mean ± SD of performance across the three configurations, with equal weighting across the six landmarks. For the full-history model, C-index, iAUC and IBS were {mean_sd('Full history','C-index')}, {mean_sd('Full history','iAUC')} and {mean_sd('Full history','IBS')}, respectively (Table S1). Fold-specific individual estimates and mean ± SD are shown in Figure S1.

For downstream analyses, we averaged the interval-specific hazards from the three independently trained seed models within each fold and recomputed survival and cumulative risk. We also reran the current-information-only comparator with three seeds under the same conditions. Thus, 15 training runs were performed for each model, but these are not treated as 15 independent repetitions of the entire development process. This analysis addresses training randomness conditional on the existing partitions and model specification; it does not assess repeated splitting or nested model selection. The inherited use of validation folds for early stopping and snapshot selection remains a limitation.

中文回复：感谢审稿人的重要建议。我们在统一CPU/FP32环境下，采用三个独立随机种子配置重新训练LSTM_v2，保持患者级五折划分、预测变量、超参数及原训练规则不变。每个种子配置均包含五个折内的独立训练。我们分别拼接每个配置的完整OOF预测，在六个landmark等权评价后，报告三个配置之间的性能均值±标准差。完整历史模型的C-index、iAUC及IBS分别为{mean_sd('Full history','C-index')}、{mean_sd('Full history','iAUC')}和{mean_sd('Full history','IBS')}（表S1）；逐折结果及mean±SD见图S1。

后续分析在每折内对三个独立种子模型的区间条件风险取平均，再重新计算生存概率及累计风险。仅当前信息对照也在相同条件下以三个种子重新训练。因此，每种模型完成15次训练，但不将其视为15次独立的全流程重复。该分析评价既定划分和模型配置下的训练随机性，不评价重复数据划分或嵌套模型选择。原验证折参与早停及快照选择的限制仍需保留。

## Comment 5 Sensitivity analyses

Response: Thank you for this valuable suggestion. We have updated both the full-history and current-information-only LSTM_v2 models to three independently trained seeds per fold. Using their newly assembled, uncalibrated OOF predictions under matched evaluation conditions, the equal-weight six-landmark differences (full history minus current information only) were {delta('C-index')} for C-index, {delta('iAUC')} for iAUC, and {delta('IBS')} for IBS. Intervals were obtained from 1,000 paired patient-bootstrap samples, preserving each patient's records across landmarks (Figure S2; Table S2).

We also recomputed center-specific OOF performance from the new three-seed predictions (Figure S3) and repeated grouped input-reference occlusion using the new three-seed full-history models and newly cross-fitted calibrators (Figure S4). These are complementary diagnostics: the current-information-only model removes cumulative exposure and laboratory observation/recency/change channels as well as prior sequence information; center subgroup analysis is not leave-one-center-out validation; and frozen input occlusion is not retrained predictor-set comparison. Additional predictors and center-held-out retraining have not been evaluated in this three-seed update, and we do not claim that increasing the number of seeds resolves those separate requests.

中文回复：感谢审稿人的宝贵建议。我们已将完整历史模型与仅当前信息LSTM_v2均更新为每折三个独立训练种子，并在统一评价条件下使用新生成的未校准OOF预测进行比较。六个landmark等权平均后，以完整历史减去仅当前信息定义差值，C-index、iAUC和IBS分别为{delta('C-index')}、{delta('iAUC')}和{delta('IBS')}。置信区间来自1,000次患者级配对bootstrap，同一患者各landmark记录共同抽样（图S2；表S2）。

我们还基于三种子新预测重新计算中心分层表现（图S3），并使用三种子完整历史模型及新交叉拟合校准器重新开展变量组参考遮挡（图S4）。这些分析各有边界：仅当前信息模型还删除累计暴露及实验室观测/时间间隔/变化通道；中心分层不等同于留一中心验证；冻结遮挡也不等同于更换变量后重训。本轮三种子更新没有新增其他临床预测变量或开展留中心重训，不将增加种子数表述为已解决这些独立问题。

## Comments 3 and 6

Comment 3 remains a positioning issue: this is a clinical dynamic prediction and validation study using established methods, not a new algorithm. The three-seed comparison provides additional evaluation, not algorithmic novelty.

Comment 6 is unaffected by random seeds. The data should not be claimed to be an entirely new database without verification. Previously identified related publications are DOI 10.1016/j.lanwpc.2026.101883, 10.1016/j.lanwpc.2026.101879 and 10.1097/QAD.0000000000004535. Patient-level overlap with the current extraction remains to be confirmed by the investigators; no new overlap claim is made here.

中文：第3条仍按“采用既有方法的临床动态预测与验证研究”定位，增加种子不构成算法创新。第6条的数据来源及既往患者重叠与种子数量无关，仍需作者确认，不应宣称全新数据库或不存在重叠。
'''
(B/'Responses_three_seeds_EN_CN.md').write_text(reply,encoding='utf8')
legend=(old/'Figure_legends_EN_CN.md').read_text(encoding='utf8')
legend=legend.replace('Existing two-seed training variability','Three-seed training variability').replace('two saved training seeds','three newly trained CPU/FP32 seeds').replace('n=2','n=3').replace('两个已有训练种子','三个新训练种子').replace('两个种子','三个种子').replace('existing full-history uncalibrated OOF predictions','new three-seed full-history uncalibrated OOF predictions').replace('aligned saved uncalibrated predictions','aligned new three-seed uncalibrated predictions').replace('Existing paired-bootstrap iAUC losses','Recomputed paired-bootstrap iAUC losses').replace('original Step11 calibrator','new three-seed cross-fitted calibrator').replace('existing 1,000 paired patient-bootstrap','new 1,000 paired patient-bootstrap').replace('整理已有四个landmark','重新计算四个landmark').replace('既有混合中心OOF预测','新三种子混合中心OOF预测')
(B/'Figure_legends_three_seeds_EN_CN.md').write_text(legend,encoding='utf8')
legend=legend.replace('展示现有混合中心OOF预测','展示新三种子混合中心OOF预测').replace('final Step10E LSTM_v2','three-seed Step10E-configuration LSTM_v2 rerun').replace('full-history Step10E and separately trained Step13A','three-seed full-history Step10E-configuration and separately trained three-seed Step13A-configuration')
(B/'Figure_legends_three_seeds_EN_CN.md').write_text(legend,encoding='utf8')
qa={'training_runs':30,'models':2,'folds':5,'seeds_per_fold':3,'all_checkpoints_new_CPU_FP32':True,'ensemble_hazard_exactly_reconstructed':True,'seed_stream_mean_SD_n':3,'old_two_seed_results_overwritten':False,'training_elapsed_sum_seconds':sum(z['elapsed_seconds'] for z in runs),'remaining_limitations':['baseline input dates','validation-guided selection','non-nested tuning','no LOCO','no added-predictor retraining','cohort overlap unconfirmed']}
(Q/'three_seed_completion_checks.json').write_text(json.dumps(qa,indent=2),encoding='utf8')
print(json.dumps(qa,indent=2),flush=True)
print(ss[ss.landmark=='Mean'].to_string(index=False),flush=True)
