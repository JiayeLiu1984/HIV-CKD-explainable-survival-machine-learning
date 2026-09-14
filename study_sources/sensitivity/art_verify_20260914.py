from pathlib import Path
import sys,os,json,time
W=Path(__file__).resolve().parent
# Install requirements in your own environment; no bundled runtime is required.
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,pandas as pd
from sksurv.metrics import concordance_index_ipcw,cumulative_dynamic_auc,integrated_brier_score
from sksurv.util import Surv
from sksurv.nonparametric import kaplan_meier_estimator
from concurrent.futures import ProcessPoolExecutor
P=Path(os.environ.get('CKD_WORKDIR', 'private_study_inputs'));S=P/'comment1_ART_censoring_sensitivity_20260905';D=P/'rolling_5y_step6_super_landmark_data';M=P/'rolling_5y_step11_v2_four_model_oof_comparison_lstm_v2'
times=np.r_[np.arange(6.,55.,6.),59.999]
def init():
 global pid,lm,t,e,risk,refs,idxs,npat,bins
 meta=pd.read_csv(D/'super_landmark_development_metadata.csv',usecols=['local_patient_index','landmark_index'])
 pid=meta.local_patient_index.to_numpy(int);lm=meta.landmark_index.to_numpy(int);npat=22337
 t=np.load(S/'sensitivity_analysis_time_month.npy').astype(float);e=np.load(S/'sensitivity_event_within_60m.npy').astype(bool)
 risk=np.load(M/'lstm_v2_crossfit_calibrated_oof_risk_long.npy',mmap_mode='r')
 assert np.all(np.load(S/'sensitivity_eligible_long.npy')) and len(t)==100122
 idxs=[np.flatnonzero(lm==k) for k in range(6)];refs=[Surv.from_arrays(e[ix],t[ix]) for ix in idxs]
 bins=[]
 for ix in idxs:
  order=np.argsort(risk[ix,-1],kind='stable');b=np.empty(len(ix),int);b[order]=np.minimum(9,np.arange(len(ix))*10//len(ix));bins.append(b)
def metric(ix,k,b):
 y=Surv.from_arrays(e[ix],t[ix]);r=np.asarray(risk[ix],float)
 c=concordance_index_ipcw(refs[k],y,r[:,-1],tau=59.999)[0]
 a=cumulative_dynamic_auc(refs[k],y,r,times)[1]
 ib=integrated_brier_score(refs[k],y,1-r,times)
 ici=0.
 for z in range(10):
  sel=b==z
  if not sel.any():continue
  kt,ks=kaplan_meier_estimator(e[ix][sel],t[ix][sel]);where=np.flatnonzero(kt<=60);obs=1-ks[where[-1]] if len(where) else 0
  ici+=sel.mean()*abs(r[sel,-1].mean()-obs)
 return [c,a,ib,ici]
def replicate(seed):
 counts=np.bincount(np.random.default_rng(seed).integers(0,npat,size=npat),minlength=npat)
 result=[]
 for k,base in enumerate(idxs):
  mult=counts[pid[base]];ix=np.repeat(base,mult);b=np.repeat(bins[k],mult)
  result.append(metric(ix,k,b))
 return result
if __name__=='__main__':
 init();start=time.time();point=np.array([metric(ix,k,bins[k]) for k,ix in enumerate(idxs)])
 archived=pd.read_csv(S/'landmark_performance_comparison.csv');f=archived[archived.stage=='ART_sensitivity_frozen_primary_calibration'].sort_values('landmark_year')
 assert np.allclose(point[:,:3],f[['uno_c_index_5y','integrated_dynamic_auc','integrated_brier']].to_numpy(),atol=1e-10)
 np.save(W/'ART_point_verified.npy',point)
 print('POINT VERIFIED',point.mean(axis=0).tolist(),flush=True)
 n=int(sys.argv[1]) if len(sys.argv)>1 else 1000
 boot=[]
 with ProcessPoolExecutor(max_workers=4,initializer=init) as pool:
  for i,x in enumerate(pool.map(replicate,range(2026091400,2026091400+n),chunksize=5)):
   boot.append(x)
   if (i+1)%50==0:print('Completed',i+1,'elapsed',round(time.time()-start),flush=True)
 boot=np.array(boot);np.save(W/'ART_patient_bootstrap_PRIVATE.npy',boot)
 rows=[]
 for k in range(7):
  p=point[k] if k<6 else point.mean(axis=0);bs=boot[:,k,:] if k<6 else boot.mean(axis=1)
  row={'landmark':str(k) if k<6 else 'Six-landmark mean','risk_set_n':len(idxs[k]) if k<6 else '', 'events':int(e[idxs[k]].sum()) if k<6 else ''}
  for j,label in enumerate(['C_index','iAUC','IBS','grouped_ICI']):row[label]=float(p[j]);row[label+'_lower']=float(np.quantile(bs[:,j],.025));row[label+'_upper']=float(np.quantile(bs[:,j],.975))
  rows.append(row)
 pd.DataFrame(rows).to_csv(W/'ART_verified_six_landmarks.csv',index=False)
 (W/'ART_verification.json').write_text(json.dumps({'replicates':n,'seed_start':2026091400,'bootstrap_unit':'development participant; same multiplicities across all six landmarks','calibration':'frozen primary cross-fitted calibration; no model or calibration refitting','censoring_reference':'fixed original sensitivity risk set at each landmark','ICI':'size-weighted absolute difference between mean predicted risk and Kaplan-Meier risk within fixed full-sample prediction deciles','point_matches_archived':True,'elapsed_seconds':time.time()-start},indent=2))
 print('DONE',time.time()-start,flush=True)
