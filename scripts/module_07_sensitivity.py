"""Synthetic ART outcome recoding and optional three genuine training seeds."""
import argparse
import numpy as np
import pandas as pd
import torch
from _runtime import WORK, DATA, dump_json
from module_03_model_evaluation import metrics, calibrate, to_hazard
from module_02_model_training import train_lstm

def main():
    p=argparse.ArgumentParser();p.add_argument('--three-seeds',action='store_true');p.add_argument('--epochs',type=int,default=2);args=p.parse_args();torch.set_num_threads(2)
    out=WORK/'sensitivity';out.mkdir(exist_ok=True);base=WORK/'rolling_5y_step6_super_landmark_data';meta=pd.read_csv(base/'super_landmark_development_metadata.csv')
    panel=pd.read_csv(DATA/'synthetic_development.csv').drop_duplicates('ID').set_index('ID')
    affected=meta.ID.map(panel.synthetic_art_affected).astype(bool).to_numpy();alt=meta.copy()
    alt.loc[affected,'event_within_60m']=0
    s=np.load(WORK/'evaluation/lstm_calibrated_survival.npy')
    primary=metrics(meta,s);sensitivity=metrics(alt,s)
    joined=primary.merge(sensitivity,on='landmark_index',suffixes=('_primary','_sensitivity'))
    for k in ['Uno_C_index','iAUC','IBS']:joined[k+'_difference']=joined[k+'_sensitivity']-joined[k+'_primary']
    joined.to_csv(out/'art_event_to_censoring.csv',index=False)
    info={'synthetic_only':True,'art_sensitivity':'Recode simulated affected events to censoring at the same time; frozen predictions. No relabelling of any clinical data.','three_seeds_executed':args.three_seeds}
    if args.three_seeds:
        y=np.load(base/'development_future_event_long.npy');mask=np.load(base/'development_future_at_risk_long.npy');tables=[]
        for seed in [20260912,20261012,20261112]:
            raw=train_lstm(args.epochs,seed_base=seed,save_checkpoints=False);cal,_=calibrate(to_hazard(raw),y,mask,meta)
            t=metrics(meta,cal);t['seed']=seed;tables.append(t)
        t=pd.concat(tables,ignore_index=True);t.to_csv(out/'three_seed_landmark_metrics.csv',index=False)
        means=t.groupby('seed')[['Uno_C_index','iAUC','IBS']].mean();means.to_csv(out/'three_seed_overall_metrics.csv')
        means.agg(['mean','std']).to_csv(out/'three_seed_mean_SD.csv')
        info['seed_design']='Five fixed patient folds; three separate CPU training runs per fold, same synthetic settings, not repartitioning or retuning.'
    dump_json(out/'scope.json',info)

if __name__=='__main__':main()
