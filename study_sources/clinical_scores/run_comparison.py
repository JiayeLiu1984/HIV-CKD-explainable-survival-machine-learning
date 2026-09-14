"""Exploratory replication of the user's legacy score definitions.
NOT validation of complete original D:A:D/VHA. No LSTM training.
Dependencies: numpy, pandas, scipy, scikit-survival, numba and matplotlib.
Run: python run_comparison.py
"""
from pathlib import Path
import ast,json,hashlib,logging,time,os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numba import njit
from sksurv.util import Surv
from sksurv.nonparametric import CensoringDistributionEstimator,kaplan_meier_estimator
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_ipcw,cumulative_dynamic_auc,brier_score

B=Path(__file__).resolve().parent;R=Path(os.environ.get('CKD_WORKDIR','private_study_inputs'));A=B.parent/'baseline_clinical_scores_audit_20260908'
logging.basicConfig(filename=B/'analysis.log',level=logging.INFO,format='%(asctime)s %(message)s',encoding='utf8')
def log(x): print(x,flush=True);logging.info(x)
def savejson(name,x): (B/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf8')
names=['LSTM_v2','DAD_baseline_CD4_proxy','VHA_5_component']
labels=['LSTM_v2','D:A:D (CD4 proxy)','VHA (5 components)']
colors=['#d62728','#1f77b4','#2ca02c'];horizons=np.arange(6,61,6,dtype=float)
plt.rcParams.update({'font.family':'Times New Roman','font.size':11,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white'})

from legacy_score_functions import compute_dad_full, scherzer_ckd_points_no_nan
nbpath=B/'legacy_score_functions.py'

d=pd.read_csv(A/'development_baseline_predictions_NOT_YET_CLEARED.csv')
s=pd.read_csv(R/'深圳南宁随访数据表2.csv',encoding='gb18030');s=s[s.time_bin==0].copy()
assert not d.ID.duplicated().any() and not s.ID.duplicated().any()
sourcecols=['ID','Sex','Age','Course','CD4','eGFR','HCV_status','hypertension_status','diabetes_status','CVD_status','GLU','TG']
d=d.merge(s[sourcecols],on='ID',how='left',validate='one_to_one',indicator=True)
assert (d['_merge']=='both').all()
assert d.Course.isin(['Drugs','Heterosexual','Male to male','Others']).all()
df=d.rename(columns={'HCV_status':'HCV','hypertension_status':'Hypertension','diabetes_status':'Diabetes','CVD_status':'CVD'}).copy()
df['Drugs']=(df.Course=='Drugs').astype(int) # explicit legacy one-hot mapping; not an independently verified IDU history
df['female']=(df.Sex=='Female').astype(int)
for new,old in [('idu','Drugs'),('hcv','HCV'),('htn','Hypertension'),('dm','Diabetes'),('cvd','CVD')]:df[new]=df[old].astype(int)
df['cd4_nadir_proxy']=df.CD4
dad=compute_dad_full(df);vha=scherzer_ckd_points_no_nan(df)
N=len(d);e=d.original_event.to_numpy().astype(bool);t=d.original_observed_time_month.to_numpy(float);fold=d.fold_id.to_numpy(int)
y=Surv.from_arrays(e,t);y5=Surv.from_arrays(d.event_within_60m.to_numpy().astype(bool),d.analysis_time_month.to_numpy(float))
raw=np.column_stack([d.lstm_risk_5y.to_numpy(),dad,vha]);assert np.isfinite(raw).all()
pred=np.empty((N,3,10))
pred[:,0,:]=np.load(R/'rolling_5y_step11_v2_four_model_oof_comparison_lstm_v2/lstm_v2_crossfit_calibrated_oof_risk_long.npy')[d.long_row_index.to_numpy(int)]
assert np.allclose(pred[:,0,-1],raw[:,0])
mapping=[]
for k in range(5):
 train=fold!=k;test=fold==k
 assert not set(d.loc[train,'ID'])&set(d.loc[test,'ID'])
 for m in [1,2]:
  cox=CoxPHSurvivalAnalysis().fit(raw[train,m,None],y5[train])
  pred[test,m,:]=1-np.vstack([fn(horizons) for fn in cox.predict_survival_function(raw[test,m,None])])
  mapping.append({'heldout_fold':k,'model':names[m],'coefficient':float(cox.coef_[0]),'train_n':int(train.sum()),'test_n':int(test.sum()),'patient_overlap':0})
assert np.isfinite(pred).all() and (pred>=0).all() and (pred<=1).all()
pd.DataFrame(mapping).to_csv(B/'crossfit_Cox_mapping.csv',index=False)
np.save(B/'three_model_crossfit_risk_6_to_60m.npy',pred)
pd.DataFrame({'ID':d.ID,'fold_id':fold,'time_month':t,'event':e,'LSTM_risk5':pred[:,0,-1],'DAD_proxy_points':dad,'VHA5_points':vha,'DAD_proxy_Cox_risk5':pred[:,1,-1],'VHA5_Cox_risk5':pred[:,2,-1]}).to_csv(B/'patient_predictions_PRIVATE.csv',index=False,encoding='utf-8-sig')
audit={'analysis':'exploratory legacy-score replication; not validation of original scores','cohort_n':N,'events_within_60m':int(d.event_within_60m.sum()),'unmatched_source_n':0,'finite_score_n':{names[m]:int(np.isfinite(raw[:,m]).sum()) for m in [1,2]},'source_missing_n':{c:int(d[c].isna().sum()) for c in sourcecols if c!='ID'},'Course_Others_n':int((d.Course=='Others').sum()),'source_notebook':str(nbpath),'notebook_sha256':hashlib.sha256(nbpath.read_bytes()).hexdigest(),'limitations':['DAD uses baseline CD4 as nadir proxy; not original DAD','VHA omits SBP and proteinuria; all other missing defaults follow legacy code','Course Drugs mapped to old Drugs dummy; Others maps to zero','Existing baseline inputs have unresolved/post-ART source-date findings; not strict ART-day validation','Cox score-risk mappings fitted in other four folds only; local adaptation, not published original risk','Bootstrap conditions on frozen predictions and full-cohort censoring reference; no refitting LSTM or Cox during bootstrap','Internal OOF is not a locked independent test and tuning was not fully nested'],'horizons_month':horizons.tolist()}
savejson('audit.json',audit);log(f'Prepared {N} matched patients; legacy scores and five-fold Cox mappings complete.')

# IPCW reference estimated once on the common cohort; same reference for all models.
cens=CensoringDistributionEstimator().fit(y)
G=cens.predict_proba(np.minimum(t,60));Gt=cens.predict_proba(horizons)
assert (G[t<=60]>0).all() and (Gt>0).all()
Wcase=np.where(e&(t<=60),1/G,0.);Wuno=np.where(e&(t<60),1/G**2,0.)
order=np.argsort(t,kind='stable');starts=np.r_[0,np.flatnonzero(np.diff(t[order]))+1];ends=np.r_[starts[1:],N]

@njit
def bitadd(bit,i,v):
 while i<len(bit):bit[i]+=v;i+=i&-i
@njit
def bitsum(bit,i):
 ans=0.
 while i>0:ans+=bit[i];i-=i&-i
 return ans
@njit
def rankstats(count,e,t,order,starts,ends,pos,lo,hi,wuno,wcase):
 bit=np.zeros(len(count)+1);total=0.;num=0.;den=0.
 for z in range(len(starts)-1,-1,-1):
  a=starts[z];b=ends[z]
  for j in range(a,b):
   i=order[j]
   if not e[i]:bitadd(bit,pos[i],count[i]);total+=count[i]
  for j in range(a,b):
   i=order[j]
   if e[i] and wuno[i]>0 and count[i]>0:
    lower=bitsum(bit,lo[i]);upper=bitsum(bit,hi[i]);w=count[i]*wuno[i]
    num+=w*(lower+.5*(upper-lower));den+=w*total
  for j in range(a,b):
   i=order[j]
   if e[i]:bitadd(bit,pos[i],count[i]);total+=count[i]
 c=num/den if den>0 else np.nan
 bit=np.zeros(len(count)+1);controls=0.;num=0.;den=0.
 for i in range(len(count)):
  if t[i]>60:bitadd(bit,pos[i],count[i]);controls+=count[i]
 for i in range(len(count)):
  if wcase[i]>0 and count[i]>0:
   low=bitsum(bit,lo[i]);up=bitsum(bit,hi[i]);w=count[i]*wcase[i]
   num+=w*(low+.5*(up-low));den+=w*controls
 return c,num/den if den>0 else np.nan

ranks=[]
for m in range(3):
 vals=np.sort(raw[:,m]);ranks.append((np.searchsorted(vals,raw[:,m],side='right'),np.searchsorted(vals,raw[:,m]-1e-8,side='left'),np.searchsorted(vals,raw[:,m]+1e-8,side='right')))
def rankmetric(count,m):return rankstats(count,e,t,order,starts,ends,*ranks[m],Wuno,Wcase)

# Probability errors per patient/time; censored-before-horizon contribute zero.
loss=np.zeros_like(pred)
for j,h in enumerate(horizons):
 case=e&(t<=h);control=t>h
 loss[case,:,j]=(1-pred[case,:,j])**2/cens.predict_proba(t[case])[:,None]
 loss[control,:,j]=pred[control,:,j]**2/Gt[j]
trapw=np.array([.5]+[1.]*8+[.5])/9
ibs_patient=(loss*trapw).sum(axis=2)
ones=np.ones(N);point=np.zeros((3,4));curves=[];verification=[]
for m in range(3):
 point[m,:2]=rankmetric(ones,m);point[m,2]=loss[:,m,-1].mean();point[m,3]=ibs_patient[:,m].mean()
 skc=concordance_index_ipcw(y,y,raw[:,m],tau=60)[0]
 aucs,iauc=cumulative_dynamic_auc(y,y,raw[:,m] if m else pred[:,0,:],horizons)
 _,bs=brier_score(y,y,1-pred[:,m,:],horizons)
 assert np.isclose(point[m,0],skc,atol=1e-10)
 assert np.isclose(point[m,1],aucs[-1],atol=1e-8)
 assert np.allclose(loss[:,m,:].mean(axis=0),bs,atol=1e-10)
 verification.append({'model':names[m],'C_absolute_error_vs_sksurv':float(abs(point[m,0]-skc)),'AUC5_absolute_error_vs_sksurv':float(abs(point[m,1]-aucs[-1])),'Brier_max_absolute_error_vs_sksurv':float(abs(loss[:,m,:].mean(axis=0)-bs).max())})
 for j,h in enumerate(horizons):curves.append({'model':names[m],'horizon_month':h,'auc':aucs[j],'brier':bs[j],'integrated_auc_reference':iauc})
savejson('metric_verification.json',verification);pd.DataFrame(curves).to_csv(B/'time_dependent_metrics.csv',index=False)

nb=1000;rng=np.random.default_rng(20260908);boot=np.empty((nb,3,4))
for z in range(nb):
 ix=rng.integers(0,N,size=N);count=np.bincount(ix,minlength=N).astype(float)
 for m in range(3):boot[z,m,:2]=rankmetric(count,m)
 boot[z,:,2]=count@loss[:,:,-1]/N;boot[z,:,3]=count@ibs_patient/N
 if (z+1)%100==0:log(f'Patient-paired bootstrap {z+1}/{nb}')
assert np.isfinite(boot).all();np.save(B/'paired_patient_bootstrap.npy',boot)
metrics=['Uno_C_5y','AUC_5y','Brier_5y_Cox_mapped_scores','IBS_6_to_60m_Cox_mapped_scores']
rows=[];deltas=[]
for m in range(3):
 for j,metric in enumerate(metrics):
  lo,hi=np.quantile(boot[:,m,j],[.025,.975]);rows.append({'model':names[m],'metric':metric,'estimate':point[m,j],'lower95':lo,'upper95':hi,'bootstrap_n':nb,'n':N})
for m in [1,2]:
 for j,metric in enumerate(metrics):
  delta=boot[:,0,j]-boot[:,m,j];lo,hi=np.quantile(delta,[.025,.975]);deltas.append({'contrast':names[0]+' minus '+names[m],'metric':metric,'difference':point[0,j]-point[m,j],'lower95':lo,'upper95':hi})
pd.DataFrame(rows).to_csv(B/'performance.csv',index=False);pd.DataFrame(deltas).to_csv(B/'paired_differences.csv',index=False)

def km5(mask):
 if mask.sum()==0:return (np.nan,)*3
 kt,ks,ci=kaplan_meier_estimator(e[mask],t[mask],conf_type='log-log');j=np.searchsorted(kt,60,side='right')-1
 if j<0:return 0.,0.,0.
 return float(1-ks[j]),float(1-ci[1,j]),float(1-ci[0,j])
cal=[];dca=[]
thresholds=np.arange(.01,.201,.01);case=e&(t<=60);control=t>60
for m in range(3):
 bins=pd.qcut(pd.Series(pred[:,m,-1]),q=10,labels=False,duplicates='drop').to_numpy()
 for g in np.unique(bins):
  use=bins==g;obs,lo,hi=km5(use);cal.append({'model':names[m],'bin':int(g),'n':int(use.sum()),'mean_predicted':float(pred[use,m,-1].mean()),'KM_observed':obs,'lower95':lo,'upper95':hi})
 for th in thresholds:
  positive=pred[:,m,-1]>=th
  nbv=(Wcase[positive].sum()-(positive&control).sum()/Gt[-1]*th/(1-th))/N
  allv=(Wcase.sum()-control.sum()/Gt[-1]*th/(1-th))/N
  dca.append({'model':names[m],'threshold':th,'net_benefit':nbv,'treat_all':allv,'treat_none':0})
pd.DataFrame(cal).to_csv(B/'calibration_deciles.csv',index=False);pd.DataFrame(dca).to_csv(B/'DCA_IPCW.csv',index=False)

# Descriptive score tertiles fixed by the development distribution, not outcomes.
# These are not published clinical cutoffs and may collapse when points are tied.
group_rows=[];group_ids=[];cuts=[]
for m in range(3):
 c=np.quantile(raw[:,m],[1/3,2/3]);g=np.searchsorted(c,raw[:,m],side='left');group_ids.append(g)
 cuts.append({'model':names[m],'cut1':c[0],'cut2':c[1],'rule':'<=cut1; >cut1 and <=cut2; >cut2; descriptive development-distribution tertiles, not validated clinical categories'})
 for q in range(3):
  use=g==q;obs,lo,hi=km5(use);group_rows.append({'model':names[m],'group':['Lower','Middle','Upper'][q],'n':int(use.sum()),'events_5y':int((use&case).sum()),'KM_risk5':obs,'lower95':lo,'upper95':hi})
pd.DataFrame(group_rows).to_csv(B/'descriptive_risk_groups.csv',index=False);pd.DataFrame(cuts).to_csv(B/'descriptive_group_cutoffs.csv',index=False)
for m in [1,2]:pd.crosstab(pd.Series(group_ids[0],name='LSTM_group'),pd.Series(group_ids[m],name=names[m])).to_csv(B/f'group_correspondence_{names[m]}.csv')

figs=[]
def finish(fig,name):
 fig.tight_layout();fig.savefig(B/f'{name}.pdf',bbox_inches='tight');fig.savefig(B/f'{name}.png',dpi=180,bbox_inches='tight');figs.append(fig)
fig,ax=plt.subplots(figsize=(8,5));ci=np.quantile(boot[:,:,0],[.025,.975],axis=0)
ax.bar(np.arange(3),point[:,0],color=colors,width=.55);ax.errorbar(np.arange(3),point[:,0],yerr=[point[:,0]-ci[0],ci[1]-point[:,0]],fmt='none',ecolor='black',capsize=4)
ax.set_xticks(np.arange(3),labels);ax.set_ylabel('5-year Uno C-index');ax.set_ylim(0,1)
for m in range(3):ax.text(m,ci[1,m]+.02,f'{point[m,0]:.3f}',ha='center')
finish(fig,'Figure1_Uno_C')
tab=pd.DataFrame(curves)
fig,ax=plt.subplots(figsize=(8,5))
for m in range(3):v=tab[tab.model==names[m]];ax.plot(v.horizon_month/12,v.auc,'o-',color=colors[m],label=labels[m],lw=2,ms=3)
ax.axhline(.5,color='gray',ls='--');ax.set(xlabel='Years after baseline',ylabel='Time-dependent AUC',xlim=(0,5),ylim=(.4,1));ax.legend(frameon=True);finish(fig,'Figure2_AUC')
fig,ax=plt.subplots(figsize=(8,5))
for m in range(3):v=tab[tab.model==names[m]];ax.plot(v.horizon_month/12,v.brier,color=colors[m],label=labels[m],lw=2)
ax.set(xlabel='Years after baseline',ylabel='Brier score',xlim=(0,5),ylim=(0,None));ax.legend();finish(fig,'Figure3_Brier')
fig,axes=plt.subplots(1,3,figsize=(13,4))
for m,ax in enumerate(axes):
 v=pd.DataFrame(cal);v=v[v.model==names[m]];ax.plot([0,1],[0,1],'--',color='gray');ax.errorbar(v.mean_predicted,v.KM_observed,yerr=[v.KM_observed-v.lower95,v.upper95-v.KM_observed],fmt='o',color=colors[m],capsize=2,label=labels[m]);lim=min(1,max(v.mean_predicted.max(),v.upper95.max())*1.1);ax.set(xlim=(0,lim),ylim=(0,lim),xlabel='Mean predicted 5-year risk',ylabel='Observed 5-year risk (KM)');ax.legend(fontsize=9)
finish(fig,'Figure4_Calibration')
fig,ax=plt.subplots(figsize=(8,5));dt=pd.DataFrame(dca)
for m in range(3):v=dt[dt.model==names[m]];ax.plot(v.threshold*100,v.net_benefit,color=colors[m],lw=2,label=labels[m])
ax.plot(thresholds*100,v.treat_all,'--',color='gray',label='Treat all');ax.axhline(0,color='black',ls=':',label='Treat none');ax.set(xlabel='5-year risk threshold (%)',ylabel='Net benefit',ylim=(-.015,None));ax.legend();finish(fig,'Figure5_DCA')
fig,ax=plt.subplots(figsize=(8,5));box=ax.boxplot([boot[:,m,3] for m in range(3)],tick_labels=labels,patch_artist=True,showfliers=False)
for patch,c in zip(box['boxes'],colors):patch.set_facecolor(c);patch.set_alpha(.85)
for median in box['medians']:median.set_color('black');median.set_linewidth(1.5)
ax.set_ylabel('Integrated Brier score (6-60 months)');finish(fig,'Figure6_Actual_bootstrap_IBS')
with PdfPages(B/'All_figures.pdf') as pdf:
 for fig in figs:pdf.savefig(fig,bbox_inches='tight')
for fig in figs:plt.close(fig)

lines=['# Exploratory comparison using legacy score definitions','', '**Not a validation of the complete original D:A:D or VHA scores. Baseline input timing limitations remain.**','',f'Internal common cohort: {N:,}; events within five years: {int(case.sum()):,}. No LSTM retraining.','', '| Model | Uno C5 (95% CI) | AUC5 (95% CI) | Brier5 (95% CI) | IBS 6-60m (95% CI) |','|---|---|---|---|---|']
for m in range(3):
 parts=[]
 for j in range(4):lo,hi=np.quantile(boot[:,m,j],[.025,.975]);parts.append(f'{point[m,j]:.4f} ({lo:.4f}-{hi:.4f})')
 lines.append('| '+labels[m]+' | '+' | '.join(parts)+' |')
lines+=['','## Methods and interpretation','Legacy D:A:D uses baseline CD4 as a nadir proxy. Legacy VHA omits SBP and proteinuria. The remaining legacy missing-value and age-band behavior is retained explicitly. These implementations are adaptations, not the complete validated scores. Course=Drugs is mapped to the legacy binary Drugs field; other categories map to zero.','Discrimination uses raw score points and the LSTM five-year risk. Brier, calibration, and DCA use five-fold out-of-fold single-score Cox risk mappings for the two adapted scores; they are not published absolute-risk equations. Cox mappings are fitted using five-year administratively censored outcomes from the other four folds. LSTM uses its existing cross-fitted hazard calibration.','IPCW uses the common-cohort censoring KM, frozen across 1,000 paired patient bootstrap replicates. Intervals are conditional on saved predictions and the reference censoring estimate; neither model fitting nor Cox mapping is repeated within bootstrap. Faster metrics were checked against scikit-survival.','The saved LSTM and score inputs use the existing baseline row. Source-date problems previously identified are not repaired by extracting landmark 0. Thus this is an exploratory replication of legacy definitions, not a fair strict-ART-day validation. Existing full-cohort LSTM figures may differ slightly because the censoring reference/method is now explicitly shared across models.','Descriptive groups use common-cohort development-distribution tertiles without outcomes; ties may produce unequal or empty groups. These are not original score clinical risk categories.','', '## Figure legends','Figures use white backgrounds, Times New Roman, red LSTM, blue D:A:D proxy, and green five-component VHA. No overall titles or bottom captions appear inside figures.','1. Five-year Uno C-index with paired patient-bootstrap percentile intervals.','2. Cumulative/dynamic AUC over 6-60 months; LSTM uses horizon-specific risks and adapted scores use fixed raw points.','3. IPCW Brier curves; clinical-score curves use locally cross-fitted Cox risk mappings.','4. Five-year calibration by prediction quantile bins, with KM log-log confidence intervals; tied predictions can reduce the number of bins.','5. IPCW decision curves for the saved LSTM and locally mapped adapted scores. These do not establish the utility of the original scores.','6. Actual patient-bootstrap IBS distributions over 6-60 months, not simulated distributions inferred from a confidence interval.','', '## 中文结果与限制',f'沿用旧计分口径，在 {N:,} 名共同内部患者中进行了探索性比较（五年内 {int(case.sum()):,} 例 CKD）。最终 Step10E LSTM_v2 未重新训练，提取 landmark=0 的既有 OOF 预测。D:A:D 使用基线 CD4 代理历史最低 CD4，VHA 省略收缩压和蛋白尿；两者均不是完整原始评分。上表为本轮真实重新计算值，患者级配对差值见 paired_differences.csv。','评分概率指标来自其他四折拟合的单变量 Cox 映射，不是原论文的五年风险公式。输入的基线时间来源问题尚未消除，因此本结果只适合作为旧实现复现与探索性分析，不能声称已完成严格 ART 时点的公平验证或证明优于完整原评分。','', '## Reviewer-response draft / 审稿回复草稿','We performed an exploratory baseline comparison of the final LSTM_v2 using the clinical-score implementations available in our previous analysis. These were adaptations: baseline CD4 substituted for nadir CD4 in D:A:D, and systolic blood pressure and proteinuria were unavailable for the VHA/Scherzer score. We report these limitations explicitly and do not present this analysis as an external validation of the complete original scores. Probability-based comparisons used out-of-fold local Cox mappings of the adapted scores. Baseline source-date discrepancies remain to be resolved before this analysis can support a strict ART-initiation comparison.','我们基于既往可实施的评分版本进行了最终 LSTM_v2 的基线探索性比较。D:A:D 使用基线 CD4 替代历史最低 CD4，VHA 缺少收缩压和蛋白尿；因此未将本分析表述为完整原评分的验证。概率指标使用评分的折外本地 Cox 映射。严格 ART 启动时点的比较仍需解决输入来源日期问题。','', '## Reproducibility','Run `D:/Python310/python.exe run_comparison.py`. Original data and models are read-only. PRIVATE files contain local identifiers and must not be shared publicly. The original RSF outputs are not copied into any metric.']
(B/'Results_Methods_Legends_EN_CN.md').write_text('\n'.join(lines),encoding='utf8')
log('Completed exploratory legacy-score comparison. See audit.json for non-original-score and baseline-time limitations.')
print(pd.DataFrame(rows).to_string(index=False))
