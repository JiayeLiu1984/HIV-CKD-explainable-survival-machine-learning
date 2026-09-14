"""Demonstration cross-fit calibration, censoring-aware metrics and paired bootstrap.

For the full study implementation, see 11a_four_model_calibration_comparison.py.
All bootstrap samples use patient clusters shared by models and landmarks.
"""
import argparse, itertools
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc, integrated_brier_score
from sksurv.nonparametric import kaplan_meier_estimator, CensoringDistributionEstimator
from _runtime import WORK, dump_json

TIMES=np.array([6,12,18,24,30,36,42,48,54,59.999])

def to_hazard(s):
    return np.clip(1-s/np.maximum(np.c_[np.ones(len(s)),s[:,:-1]],1e-7),1e-6,1-1e-6)

def fit_offsets(h,y,mask):
    offsets=[]
    for j in range(10):
        use=mask[:,j].astype(bool);z=logit(h[use,j]);event=y[use,j]
        if not use.any(): offsets.append(0.);continue
        result=minimize_scalar(lambda a:np.mean(np.logaddexp(0,z+a)-event*(z+a)),bounds=(-10,10),method='bounded')
        offsets.append(float(result.x))
    return np.asarray(offsets)

def apply_offsets(h,offsets):return np.cumprod(1-expit(logit(h)+offsets),axis=1)

def calibrate(h,y,mask,meta):
    s=np.zeros_like(h);all_offsets=[]
    for lm in range(6):
        rows=meta.landmark_index.to_numpy()==lm
        for fold in range(5):
            fit=rows&(meta.fold_id.to_numpy()!=fold);held=rows&(meta.fold_id.to_numpy()==fold)
            off=fit_offsets(h[fit],y[fit],mask[fit]);s[held]=apply_offsets(h[held],off)
        all_offsets.append(fit_offsets(h[rows],y[rows],mask[rows]))
    return s,np.asarray(all_offsets)

def metrics(meta,s,reference=None):
    reference=meta if reference is None else reference
    out=[]
    for lm in range(6):
        rows=meta.landmark_index.to_numpy()==lm;r=reference[reference.landmark_index==lm];m=meta[rows]
        y=Surv.from_arrays(m.event_within_60m.astype(bool),m.analysis_time_month)
        yr=Surv.from_arrays(r.event_within_60m.astype(bool),r.analysis_time_month)
        values={'landmark_index':lm,'n':len(m),'events':int(m.event_within_60m.sum())}
        supported=(TIMES>=min(y['time']))&(TIMES<max(y['time']))
        grid=TIMES[supported];estimate=np.clip(s[rows][:,supported],1e-7,1.)
        for key in ['Uno_C_index','iAUC','IBS']:values[key]=np.nan
        try:values['Uno_C_index']=float(concordance_index_ipcw(yr,y,1-s[rows,-1],tau=59.999)[0])
        except ValueError:pass
        if len(grid)>1:
            try:values['IBS']=float(integrated_brier_score(yr,y,estimate,grid))
            except ValueError:pass
            try:values['iAUC']=float(cumulative_dynamic_auc(yr,y,1-estimate,grid)[1])
            except ValueError:pass
        values['evaluation_start_month']=float(grid[0]) if len(grid) else np.nan
        values['evaluation_end_month']=float(grid[-1]) if len(grid) else np.nan
        out.append(values)
    return pd.DataFrame(out)

def km_risk(m,horizon=59.999):
    if not len(m):return np.nan
    t,s=kaplan_meier_estimator(m.event_within_60m.astype(bool).to_numpy(),m.analysis_time_month.to_numpy())
    i=np.searchsorted(t,horizon,side='right')-1
    return 0. if i<0 else 1-float(s[i])

def dca(meta,risk):
    y=Surv.from_arrays(meta.event_within_60m.astype(bool),meta.analysis_time_month)
    g=CensoringDistributionEstimator().fit(y);time=y['time'];event=y['event']
    cases=event&(time<=59.999);controls=time>59.999
    weights=np.zeros(len(y));weights[cases]=1/np.maximum(g.predict_proba(time[cases]),1e-6)
    weights[controls]=1/max(float(g.predict_proba([59.999])[0]),1e-6)
    result=[]
    for threshold in np.linspace(.01,.5,50):
        positive=risk>=threshold;odds=threshold/(1-threshold)
        nb=(np.sum(weights[cases&positive])-odds*np.sum(weights[controls&positive]))/len(y)
        all_nb=(np.sum(weights[cases])-odds*np.sum(weights[controls]))/len(y)
        result.append({'threshold':threshold,'net_benefit':nb,'treat_all':all_nb,'treat_none':0.})
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--bootstrap',type=int,default=20);a=p.parse_args()
    base=WORK/'rolling_5y_step6_super_landmark_data';meta=pd.read_csv(base/'super_landmark_development_metadata.csv')
    y=np.load(base/'development_future_event_long.npy');mask=np.load(base/'development_future_at_risk_long.npy')
    out=WORK/'evaluation';out.mkdir(exist_ok=True);survivals={};tables=[];calibration=[];decisions=[]
    for model in ['cox','rsf','rnn','lstm']:
        s0=np.load(WORK/'demo_predictions'/f'{model}_survival.npy');s,offsets=calibrate(to_hazard(s0),y,mask,meta)
        survivals[model]=s;np.save(out/f'{model}_calibrated_survival.npy',s);np.save(out/f'{model}_development_offsets.npy',offsets)
        table=metrics(meta,s);table['model']=model;tables.append(table)
        for lm in range(6):
            use=meta.landmark_index.to_numpy()==lm;m=meta[use];r=1-s[use,-1]
            for b,ids in enumerate(np.array_split(np.argsort(r),5)):
                calibration.append({'model':model,'landmark':lm,'bin':b,'n':len(ids),'predicted':float(r[ids].mean()),'observed_1_KM':km_risk(m.iloc[ids])})
            decisions.extend({'model':model,'landmark':lm,**v} for v in dca(m,r))
    table=pd.concat(tables,ignore_index=True);table.to_csv(out/'landmark_performance.csv',index=False)
    overall=table.groupby('model')[['Uno_C_index','iAUC','IBS']].mean();overall.to_csv(out/'six_landmark_mean.csv')
    pd.DataFrame(calibration).to_csv(out/'calibration.csv',index=False);pd.DataFrame(decisions).to_csv(out/'decision_curves.csv',index=False)
    rng=np.random.default_rng(20260912);ids=meta.ID.unique();index={i:np.flatnonzero(meta.ID.to_numpy()==i) for i in ids};boot=[]
    for b in range(a.bootstrap):
        idx=np.concatenate([index[i] for i in rng.choice(ids,len(ids),replace=True)])
        for model,s in survivals.items():
            result=metrics(meta.iloc[idx].reset_index(drop=True),s[idx],reference=meta)
            # Strict six-landmark estimand: do not average an incomplete bootstrap.
            vals=result[['Uno_C_index','iAUC','IBS']]
            boot.append({'replicate':b,'model':model,**{k:float(vals[k].mean()) if vals[k].notna().all() else np.nan for k in vals}})
    boot=pd.DataFrame(boot);boot.to_csv(out/'patient_bootstrap.csv',index=False);pairs=[]
    for first,second in itertools.combinations(survivals,2):
        aa=boot[boot.model==first].set_index('replicate');bb=boot[boot.model==second].set_index('replicate')
        for metric in ['Uno_C_index','iAUC','IBS']:
            diff=(aa[metric]-bb[metric]).dropna()
            pairs.append({'first_model':first,'second_model':second,'metric':metric,'difference':overall.loc[first,metric]-overall.loc[second,metric],'lower_95':diff.quantile(.025),'upper_95':diff.quantile(.975),'valid_bootstraps':len(diff)})
    pd.DataFrame(pairs).to_csv(out/'paired_model_comparisons.csv',index=False)
    dump_json(out/'evaluation_notes.json',{'synthetic_only':True,'bootstrap_n':a.bootstrap,'bootstrap_unit':'patient, shared across models and landmarks','calibration':'Cross-fitted interval intercepts by landmark; final development offsets fitted on OOF predictions for synthetic external application.','censoring_reference':'Fixed synthetic development reference; conditional bootstrap uncertainty.','limitation':'Synthetic enriched event distribution and very short training; no manuscript estimates reproduced. Use original study_sources for full study calibration and CI implementation.'})
    print(overall)

if __name__=='__main__':main()
