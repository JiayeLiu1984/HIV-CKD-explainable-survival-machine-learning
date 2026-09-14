"""Read-only reanalysis of saved final Step10E outputs. No fabricated reruns.
Run with python -X utf8 analyze.py
Outputs are conditional exploratory diagnostics, not fully nested validation.
"""
from pathlib import Path
import json, hashlib, time, logging
import numpy as np
import pandas as pd
from numba import njit
from sksurv.util import Surv
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.metrics import concordance_index_ipcw,cumulative_dynamic_auc,integrated_brier_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

B=Path(__file__).resolve().parent; R=Path('__CKD_WORKDIR__')
T=B/'tables'; F=B/'figures'; Q=B/'audit'
for p in [T,F,Q]:p.mkdir(parents=True,exist_ok=True)
logging.basicConfig(filename=B/'analysis.log',level=logging.INFO,encoding='utf8',format='%(asctime)s %(message)s')
def log(x):print(x,flush=True);logging.info(x)
def js(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf8')
manifest=[]
def tracked(p):
 manifest.append({'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()});return p
def csv(p):return pd.read_csv(tracked(p))
def arr(p):return np.load(tracked(p))
full=R/'rolling_5y_step10e_lstm_v2_final_oof_selected_existing_trials'
current=R/'rolling_5y_step13a_lstm_v2_current_only_oof_fixed_hyperparameters'
S6=R/'rolling_5y_step6_super_landmark_data'; S1=R/'rolling_5y_step1_new_split'
metrics=['C-index','iAUC','IBS']; fields=['uno_c_index_5y','iauc','ibs']
seedrows=[];check=[]
for model,p in [('Full history',full),('Current information only',current)]:
 for fold in range(5):
  summary=json.loads(tracked(p/f'fold_{fold}/fold_summary.json').read_text(encoding='utf8'))
  for seedpos in range(2):
   d=csv(p/f'fold_{fold}/seed_{seedpos}_landmark_metrics.csv')
   sm=summary['seed_summaries'][seedpos]
   for name,col,summarycol in zip(metrics,fields,['mean_uno_c','mean_iauc','mean_ibs']):
    assert abs(d[col].mean()-sm[summarycol])<1e-10
    seedrows.append(dict(model=model,fold=fold,seed_position=seedpos,actual_seed=sm['seed'],metric=name,estimate=d[col].mean(),landmark='equal_weight_mean'))
    for _,row in d.iterrows():seedrows.append(dict(model=model,fold=fold,seed_position=seedpos,actual_seed=sm['seed'],metric=name,estimate=row[col],landmark=str(int(row.landmark_month/12))))
seed=pd.DataFrame(seedrows);seed.to_csv(T/'S1_individual_seed_metrics.csv',index=False)
ss=seed.groupby(['model','fold','metric','landmark']).estimate.agg(['mean','std','count']).reset_index().rename(columns={'std':'sample_SD','count':'seed_n'})
ss.to_csv(T/'S1_seed_mean_SD_within_each_fold.csv',index=False)

# Exact aligned saved predictions, without adding a new calibration fit.
lm=arr(S6/'development_landmark_month_long.npy').astype(int)
t=arr(S6/'development_analysis_time_month.npy').astype(float)
e=arr(S6/'development_event_within_60m.npy').astype(bool)
map_=arr(S6/'development_long_row_index_map.npy').astype(int)
dev=arr(S1/'development_idx.npy').astype(int);npats=len(dev)
local=np.full(len(t),-1,dtype=int)
for l in range(6):
 use=map_[:,l]>=0;local[map_[use,l]]=np.flatnonzero(use)
assert (local>=0).all() and len(t)==100122
pred=np.stack([arr(p/'lstm_v2_oof_risk_long.npy').astype(float) for p in [full,current]],axis=1)
assert pred.shape==(len(t),2,10) and np.isfinite(pred).all()
for p in [full,current]:
 for k in range(5):
  ix=arr(p/f'fold_{k}/validation_long_idx.npy').astype(int)
  saved=arr(p/f'fold_{k}/validation_risk.npy')
  assert np.allclose(saved,pred[ix,0 if p==full else 1],atol=1e-7)
  peer=current if p==full else full
  assert np.array_equal(ix,np.load(peer/f'fold_{k}/validation_long_idx.npy'))
info=csv(S1/'patient_info_new_split.csv').set_index('patient_index')
centers=info.loc[dev,'center'].to_numpy()[local]
h=np.array([6,12,18,24,30,36,42,48,54,59.999],float)

@njit
def add(bit,i,v):
 while i<len(bit):bit[i]+=v;i+=i&-i
@njit
def sumto(bit,i):
 s=0.
 while i>0:s+=bit[i];i-=i&-i
 return s
@njit
def uno(count,e,order,starts,ends,pos,lo,hi,w):
 bit=np.zeros(len(count)+1);total=0.;num=0.;den=0.
 for z in range(len(starts)-1,-1,-1):
  for j in range(starts[z],ends[z]):
   i=order[j]
   if not e[i]:add(bit,pos[i],count[i]);total+=count[i]
  for j in range(starts[z],ends[z]):
   i=order[j]
   if e[i] and w[i]>0 and count[i]>0:
    low=sumto(bit,lo[i]);up=sumto(bit,hi[i]);ww=count[i]*w[i]
    num+=ww*(low+.5*(up-low));den+=ww*total
  for j in range(starts[z],ends[z]):
   i=order[j]
   if e[i]:add(bit,pos[i],count[i]);total+=count[i]
 return num/den if den>0 else np.nan
@njit
def auc(count,t,e,w,order,starts,ends,horizon):
 lower=0.;num=0.;wc=0.
 for z in range(len(starts)):
  cc=0.;ww=0.
  for k in range(starts[z],ends[z]):
   i=order[k]
   if t[i]>horizon:cc+=count[i]
   elif e[i]:ww+=count[i]*w[i]
  num+=ww*(lower+.5*cc);wc+=ww;lower+=cc
 return num/(wc*lower) if wc*lower>0 else np.nan
@njit
def km(count,t,e,order,starts,ends,h):
 risk=count.sum();s=1.;out=np.ones(len(h));j=0
 for z in range(len(starts)):
  tm=t[order[starts[z]]]
  while j<len(h) and h[j]<tm:out[j]=s;j+=1
  if j==len(h):break
  de=0.;removed=0.
  for k in range(starts[z],ends[z]):
   i=order[k];removed+=count[i]
   if e[i]:de+=count[i]
  if risk>0:s*=1-de/risk
  risk-=removed
 while j<len(h):out[j]=s;j+=1
 return out

class Context:
 def __init__(self,ix):
  self.ix=ix;self.t=t[ix];self.e=e[ix];self.p=pred[ix];self.local=local[ix];n=len(ix);self.n=n
  yy=Surv.from_arrays(self.e,self.t);self.yy=yy
  c=CensoringDistributionEstimator().fit(yy);self.w=np.zeros(n)
  cases=self.e&(self.t<=h[-1]);self.w[cases]=1/c.predict_proba(self.t[cases])
  self.wuno=np.where(self.e&(self.t<h[-1]),self.w**2,0.)
  self.order=np.argsort(self.t,kind='stable');self.starts=np.r_[0,np.flatnonzero(np.diff(self.t[self.order]))+1];self.ends=np.r_[self.starts[1:],n]
  self.ranks=[];self.aucs=[]
  for m in range(2):
   v=self.p[:,m,-1];sv=np.sort(v)
   self.ranks.append((np.searchsorted(sv,v,side='right'),np.searchsorted(sv,v-1e-8,side='left'),np.searchsorted(sv,v+1e-8,side='right')))
   aa=[]
   for j in range(10):
    order=np.argsort(self.p[:,m,j],kind='stable');v=self.p[order,m,j];st=np.r_[0,np.flatnonzero(np.diff(v)>1e-8)+1];aa.append((order,st,np.r_[st[1:],n]))
   self.aucs.append(aa)
  loss=np.zeros_like(self.p)
  for j,hh in enumerate(h):
   ca=self.e&(self.t<=hh);co=self.t>hh
   loss[ca,:,j]=(1-self.p[ca,:,j])**2*self.w[ca,None]
   loss[co,:,j]=self.p[co,:,j]**2/c.predict_proba(np.array([hh]))[0]
  self.ibs=np.trapezoid(loss,h,axis=2)/(h[-1]-h[0])
 def calc(self,count):
  s=km(count,self.t,self.e,self.order,self.starts,self.ends,h);weights=-np.diff(np.r_[1.,s]);res=np.zeros((2,3))
  for m in range(2):
   res[m,0]=uno(count,self.e,self.order,self.starts,self.ends,*self.ranks[m],self.wuno)
   av=np.array([auc(count,self.t,self.e,self.w,*self.aucs[m][j],h[j]) for j in range(10)])
   res[m,1]=av@weights/(1-s[-1]);res[m,2]=count@self.ibs[:,m]/count.sum()
  return res

contexts=[Context(np.flatnonzero(lm==year*12)) for year in range(6)]
point=np.stack([c.calc(np.ones(c.n)) for c in contexts])
for l,c in enumerate(contexts):
 for m in range(2):
  ref=[concordance_index_ipcw(c.yy,c.yy,c.p[:,m,-1],tau=h[-1])[0],cumulative_dynamic_auc(c.yy,c.yy,c.p[:,m,:],h)[1],integrated_brier_score(c.yy,c.yy,1-c.p[:,m,:],h)]
  err=float(np.max(np.abs(point[l,m]-ref)));assert err<1e-8
  check.append({'analysis':'pooled_history_comparison','landmark':l,'model':m,'max_metric_error_vs_sksurv':err})
log('Aligned predictions and all 36 metric values verified against scikit-survival.')
rng=np.random.default_rng(20260908);nboot=1000;boot=np.empty((nboot,6,2,3));start=time.time()
for b in range(nboot):
 count=np.bincount(rng.integers(0,npats,size=npats),minlength=npats).astype(float)
 for l,c in enumerate(contexts):boot[b,l]=c.calc(count[c.local])
 if (b+1)%100==0:log(f'Paired patient bootstrap {b+1}/{nboot}; elapsed {time.time()-start:.1f}s')
assert np.isfinite(boot).all();np.save(Q/'paired_patient_bootstrap_history.npy',boot)
rows=[];diffs=[]
for l in list(range(6))+['Mean']:
 pp=point.mean(axis=0) if l=='Mean' else point[l];bb=boot.mean(axis=1) if l=='Mean' else boot[:,l]
 for m,name in enumerate(['Full history','Current information only']):
  for j,metric in enumerate(metrics):
   lo,hi=np.quantile(bb[:,m,j],[.025,.975]);rows.append(dict(landmark=l,model=name,metric=metric,estimate=pp[m,j],lower95=lo,upper95=hi,n=npats if l=='Mean' else contexts[l].n))
 for j,metric in enumerate(metrics):
  lo,hi=np.quantile(bb[:,0,j]-bb[:,1,j],[.025,.975]);diffs.append(dict(landmark=l,metric=metric,contrast='Full history minus current information only',difference=pp[0,j]-pp[1,j],lower95=lo,upper95=hi))
pd.DataFrame(rows).to_csv(T/'S2_history_comparison_raw_OOF.csv',index=False);pd.DataFrame(diffs).to_csv(T/'S2_paired_differences.csv',index=False)

# Descriptive center subgroups only. These are NOT center-held-out models.
cr=[];center_contexts=[]
for center,label in [('深圳','Shenzhen'),('南宁','Nanning')]:
 for l in range(6):
  c=Context(np.flatnonzero((lm==l*12)&(centers==center)));center_contexts.append((label,l,c))
center_boot=np.empty((nboot,12,3));rng=np.random.default_rng(20260908)
for b in range(nboot):
 count=np.bincount(rng.integers(0,npats,size=npats),minlength=npats).astype(float)
 for z,(_,_,c) in enumerate(center_contexts):center_boot[b,z]=c.calc(count[c.local])[0]
 if (b+1)%250==0:log(f'Center subgroup bootstrap {b+1}/{nboot}')
assert np.isfinite(center_boot).all()
for z,(label,l,c) in enumerate(center_contexts):
 pp=c.calc(np.ones(c.n))[0]
 for j,metric in enumerate(metrics):
  lo,hi=np.quantile(center_boot[:,z,j],[.025,.975])
  cr.append(dict(center=label,landmark=l,metric=metric,estimate=pp[j],lower95=lo,upper95=hi,n=c.n,events=int(c.e.sum()),validation='mixed-center OOF subgroup; NOT leave-one-center-out'))
np.save(Q/'center_subgroup_bootstrap.npy',center_boot)
pd.DataFrame(cr).to_csv(T/'S3_center_subgroups_NOT_LOCO.csv',index=False)
oc=csv(R/'rolling_5y_step12c2_grouped_occlusion_paired_bootstrap_v1/grouped_occlusion_paired_bootstrap_CI_all.csv')
oc.to_csv(T/'S4_existing_frozen_group_occlusion.csv',index=False)

plt.rcParams.update({'font.family':'Times New Roman','font.size':11,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white'})
figures=[]
def save(fig,name):
 fig.tight_layout();fig.savefig(F/f'{name}.pdf',bbox_inches='tight');fig.savefig(F/f'{name}.png',dpi=180,bbox_inches='tight');figures.append(fig)
colors=['#d62728','#1f77b4'];letters='ABCDEF'
fig,axs=plt.subplots(1,3,figsize=(12,4))
for j,ax in enumerate(axs):
 v=ss[(ss.model=='Full history')&(ss.metric==metrics[j])&(ss.landmark=='equal_weight_mean')].sort_values('fold')
 ax.errorbar(v.fold+1,v['mean'],yerr=v.sample_SD,fmt='o',color='black',capsize=5,label='Mean +/- SD (2 seeds)')
 for s in range(2):
  w=seed[(seed.model=='Full history')&(seed.metric==metrics[j])&(seed.landmark=='equal_weight_mean')&(seed.seed_position==s)].sort_values('fold')
  ax.scatter(w.fold+1+(-.07 if s==0 else .07),w.estimate,s=25,color=colors[s],label=f'Seed position {s+1}',zorder=3)
 ax.set(xlabel='Validation fold',ylabel=metrics[j],xticks=np.arange(1,6));ax.text(-.12,1.02,letters[j],transform=ax.transAxes,fontweight='bold')
axs[0].legend(fontsize=8,loc='best');save(fig,'FigureS1_Existing_two_seed_variability')
fig,axs=plt.subplots(1,3,figsize=(12,4))
rt=pd.DataFrame(rows)
for j,ax in enumerate(axs):
 for m,name in enumerate(['Full history','Current information only']):
  v=rt[(rt.model==name)&(rt.metric==metrics[j])&(rt.landmark!='Mean')]
  ax.errorbar(v.landmark.astype(float)+(-.045 if m==0 else .045),v.estimate,yerr=[v.estimate-v.lower95,v.upper95-v.estimate],fmt='o-',capsize=3,color=colors[m],ms=4,label=name)
 ax.set(xlabel='Prediction landmark (years)',ylabel=metrics[j],xticks=np.arange(6));ax.text(-.12,1.02,letters[j],transform=ax.transAxes,fontweight='bold')
axs[0].legend(fontsize=9);save(fig,'FigureS2_Full_history_vs_current_raw_OOF')
fig,axs=plt.subplots(1,3,figsize=(12,4));ct=pd.DataFrame(cr)
for j,ax in enumerate(axs):
 for m,name in enumerate(['Shenzhen','Nanning']):
  v=ct[(ct.center==name)&(ct.metric==metrics[j])];ax.plot(v.landmark,v.estimate,'o-',color=colors[m],label=name+' OOF subgroup',ms=4);ax.fill_between(v.landmark,v.lower95,v.upper95,color=colors[m],alpha=.12)
 ax.set(xlabel='Prediction landmark (years)',ylabel=metrics[j],xticks=np.arange(6));ax.text(-.12,1.02,letters[j],transform=ax.transAxes,fontweight='bold')
axs[0].legend(fontsize=9);save(fig,'FigureS3_Center_subgroups_NOT_LOCO')
fig,axs=plt.subplots(2,2,figsize=(12,9))
names={'renal_function':'Renal function','hiv_disease':'HIV disease','hematologic':'Hematologic','metabolic_biochemistry':'Metabolic biochemistry','liver_viral_coinfection':'Liver / viral coinfection','cardiometabolic_comorbidities':'Cardiometabolic comorbidities','metabolic_medications':'Metabolic medications','current_art_regimen':'Current ART regimen','cumulative_art_exposure':'Cumulative ART exposure','demographic_anthropometric':'Demographic / anthropometric'}
for a,(ax,l) in enumerate(zip(axs.flat,[0,1,3,5])):
 v=oc[oc.landmark_year==l].sort_values('group_position');yy=np.arange(len(v))
 ax.errorbar(v.iAUC_loss,yy,xerr=[v.iAUC_loss-v.iAUC_loss_ci_lower,v.iAUC_loss_ci_upper-v.iAUC_loss],fmt='o',capsize=3,color='#1f77b4',ms=4,label=f'Landmark {l} years')
 ax.axvline(0,color='gray',ls='--',lw=1);ax.set_yticks(yy,[names[x] for x in v.group_key]);ax.invert_yaxis();ax.set_xlabel('iAUC loss after reference occlusion');ax.legend(fontsize=9);ax.text(-.12,1.02,letters[a],transform=ax.transAxes,fontweight='bold')
save(fig,'FigureS4_Frozen_group_occlusion_NOT_retrained')
with PdfPages(F/'Supplementary_figures_Comments4_5.pdf') as pdf:
 for fig in figures:pdf.savefig(fig,bbox_inches='tight')
for fig in figures:plt.close(fig)
js(Q/'metric_verification.json',check);js(Q/'source_manifest.json',manifest)
js(Q/'analysis_scope.json',{'bootstrap_n':1000,'bootstrap_seed':20260908,'bootstrap_unit':'development patient, same multiplicity across landmarks and both models','prediction_scale':'raw uncalibrated saved Step10E/Step13A risks','evaluation_grid_months':h.tolist(),'G_reference':'within-landmark common OOF cohort; fixed across bootstrap','main_aggregate':'equal average over six landmarks; not average over folds','new_training_performed':False,'limitations':['Two saved training seeds per fold, not five or ten pipeline repetitions','Validation folds selected early stopping and snapshots; tuning was not nested','Source-date audit identifies post-ART values in baseline inputs','Center subgroup results are not center-held-out cross-validation','Grouped occlusion freezes model and does not evaluate adding unavailable predictors','Current-only model also removes cumulative ART and observation/recency/delta channels, not just past rows','Mean/SD of seed records is computed within fold, not ten independent pipeline repetitions']})
log('COMPLETE: four diagnostic figures, combined PDF, numerical tables, and provenance hashes.')
log(pd.DataFrame(diffs).query("landmark == 'Mean'").to_string(index=False))
