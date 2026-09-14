"""Add iAUC and grouped five-year ICI with paired patient-bootstrap CIs.
Uses frozen existing predictions, fixed censoring reference and fixed calibration bins.
"""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from numba import njit
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import Surv
B=Path(__file__).resolve().parent;O=B/'summary_table';O.mkdir(exist_ok=True)
d=pd.read_csv(B/'patient_predictions_PRIVATE.csv');pred=np.load(B/'three_model_crossfit_risk_6_to_60m.npy');N=len(d)
t=d.time_month.to_numpy(float);e=d.event.to_numpy(bool);h=np.arange(6,61,6,dtype=float)
c=CensoringDistributionEstimator().fit(Surv.from_arrays(e,t));g=c.predict_proba(np.minimum(t,60));ipcw=np.where(e,1/g,0.)
timeorder=np.argsort(t,kind='stable');starts=np.r_[0,np.flatnonzero(np.diff(t[timeorder]))+1];ends=np.r_[starts[1:],N]
names=['LSTM_v2','DAD_baseline_CD4_proxy','VHA_5_component'];scores=[pred[:,0,:],np.repeat(d.DAD_proxy_points.to_numpy()[:,None],10,axis=1),np.repeat(d.VHA5_points.to_numpy()[:,None],10,axis=1)]
sorts=[];ts=[];te=[]
for m in range(3):
 for j in range(10):
  order=np.argsort(scores[m][:,j],kind='stable');vals=scores[m][order,j];st=np.r_[0,np.flatnonzero(np.diff(vals)>1e-8)+1]
  sorts.append(order);ts.append(st);te.append(np.r_[st[1:],N])
bins=np.column_stack([pd.qcut(pd.Series(pred[:,m,-1]),10,labels=False,duplicates='drop').to_numpy(int) for m in range(3)])

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
 return num/(wc*lower)

@njit
def km_and_ici(count,t,e,order,starts,ends,h,bins,p5):
 risk=count.sum();S=1.;surv=np.ones(10);j=0
 grisk=np.zeros((3,10));ng=np.zeros((3,10));psum=np.zeros((3,10));gs=np.ones((3,10))
 for i in range(len(t)):
  for m in range(3):
   q=bins[i,m];grisk[m,q]+=count[i];ng[m,q]+=count[i];psum[m,q]+=count[i]*p5[i,m]
 for z in range(len(starts)):
  tm=t[order[starts[z]]]
  while j<10 and h[j]<tm:surv[j]=S;j+=1
  if tm>60:break
  de=0.;removed=0.;gd=np.zeros((3,10));gr=np.zeros((3,10))
  for k in range(starts[z],ends[z]):
   i=order[k];removed+=count[i]
   if e[i]:de+=count[i]
   for m in range(3):
    q=bins[i,m];gr[m,q]+=count[i]
    if e[i]:gd[m,q]+=count[i]
  if risk>0:S*=1-de/risk
  risk-=removed
  for m in range(3):
   for q in range(10):
    if grisk[m,q]>0:gs[m,q]*=1-gd[m,q]/grisk[m,q]
    grisk[m,q]-=gr[m,q]
 while j<10:surv[j]=S;j+=1
 ici=np.zeros(3)
 for m in range(3):
  for q in range(10):
   if ng[m,q]>0:ici[m]+=ng[m,q]/count.sum()*abs(psum[m,q]/ng[m,q]-(1-gs[m,q]))
 return surv,ici

def compute(count):
 surv,ici=km_and_ici(count,t,e,timeorder,starts,ends,h,bins,pred[:,:,-1]);w=-np.diff(np.r_[1.,surv]);iauc=[]
 for m in range(3):
  av=np.array([auc(count,t,e,ipcw,sorts[m*10+j],ts[m*10+j],te[m*10+j],h[j]) for j in range(10)])
  iauc.append(float(av@w/(1-surv[-1])))
 return np.column_stack([iauc,ici])

point=compute(np.ones(N));curve=pd.read_csv(B/'time_dependent_metrics.csv');cal=pd.read_csv(B/'calibration_deciles.csv')
errors=[]
for m,name in enumerate(names):
 v=cal[cal.model==name];refici=np.average(abs(v.mean_predicted-v.KM_observed),weights=v.n);refauc=curve[curve.model==name].integrated_auc_reference.iloc[0]
 assert np.isclose(point[m,0],refauc,atol=1e-10)
 assert np.isclose(point[m,1],refici,atol=1e-10)
 errors.append({'model':name,'iAUC_error':abs(point[m,0]-refauc),'grouped_ICI_error':abs(point[m,1]-refici)})
rng=np.random.default_rng(20260908);boot=np.empty((1000,3,2));start=time.time()
for b in range(1000):
 count=np.bincount(rng.integers(0,N,size=N),minlength=N).astype(float);boot[b]=compute(count)
 if (b+1)%100==0:print(f'{b+1}/1000 bootstrap, {time.time()-start:.1f}s',flush=True)
assert np.isfinite(boot).all();np.save(O/'iAUC_grouped_ICI_bootstrap.npy',boot)
rows=[]
for m,name in enumerate(names):
 for j,metric in enumerate(['iAUC_6_to_60m','grouped_ICI_5y']):
  lo,hi=np.quantile(boot[:,m,j],[.025,.975]);rows.append({'model':name,'metric':metric,'estimate':point[m,j],'lower95':lo,'upper95':hi,'bootstrap_n':1000,'n':N})
old=pd.read_csv(B/'performance.csv');allrows=pd.concat([old,pd.DataFrame(rows)],ignore_index=True);allrows.to_csv(O/'all_metrics_with_95CI.csv',index=False,encoding='utf-8-sig')
(O/'supplemental_metric_QA.json').write_text(json.dumps({'checks':errors,'replicates':1000,'seed':20260908,'fixed_calibration_bins':True,'fixed_censoring_reference':True,'same_bootstrap_draws_as_existing_metrics':True,'scope':'landmark0_only_not_six_landmark_average'},indent=2),encoding='utf8')
print(pd.DataFrame(rows).to_string(index=False))
