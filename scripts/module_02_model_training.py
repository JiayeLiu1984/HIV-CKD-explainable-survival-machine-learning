"""Synthetic five-fold integration run: native Cox, RSF, RNN and LSTM-v2.

Short training budgets exercise code; these outputs are NOT paper results.
Original full-budget tuning/training source is retained under study_sources.
"""
import argparse
import numpy as np
import pandas as pd
import torch
from _runtime import ROOT, WORK, dump_json, load_source
import json

def train_rnn(epochs):
    m=load_source('study_sources/dynamic/09b_rnn_oof.py',overrides={'MAX_EPOCHS':epochs,'PATIENCE':epochs,'REQUIRE_CUDA':False,'USE_AMP':False})
    c=m.load_common_data();n=len(c.y_long);surv=np.full((n,10),np.nan)
    for k in range(5):
        f=m.prepare_fold_data(c,k);r=m.fit_fold(f,torch.device('cpu'),False,20260912+k)
        surv[f.validation_long_idx]=r.survival
        r.history.to_csv(WORK/f'rnn_fold_{k}_training_history.csv',index=False)
    return surv

def train_lstm(epochs,seed_base=20260912,save_checkpoints=True):
    m=load_source('study_sources/dynamic/lstm_core.py',overrides={'MAX_EPOCHS':epochs,'MIN_EPOCHS':1,'EARLY_STOPPING_PATIENCE':epochs,'TOP_SNAPSHOT_N':1,'REQUIRE_CUDA':False,'USE_AMP':False})
    c=m.load_common_data();p=json.loads((ROOT/'configs/lstm_selected_parameters.json').read_text())
    surv=np.full((len(c.y_long_all),10),np.nan)
    for k in range(5):
        f=m.prepare_fold_data(c,k)
        r=m.fit_fold_model(c,f,p,seed_base+k,torch.device('cpu'),False,progress_callback=lambda x:print(x,flush=True))
        surv[f['validation_long_idx']]=r['survival']
        r['history'].to_csv(WORK/f'lstm_seed_{seed_base}_fold_{k}_training_history.csv',index=False)
        # Save only newly fitted SYNTHETIC weights for demonstration interpretation/external prediction.
        if save_checkpoints:torch.save({'state':r['snapshot_state_dicts'][0],'parameters':p,'synthetic_only':True},WORK/f'synthetic_lstm_fold_{k}.pt')
    return surv

def main():
    a=argparse.ArgumentParser();a.add_argument('--epochs',type=int,default=2);a.add_argument('--models',nargs='+',default=['cox','rsf','rnn','lstm']);args=a.parse_args()
    torch.set_num_threads(2)
    outputs=WORK/'demo_predictions';outputs.mkdir(exist_ok=True)
    if 'cox' in args.models:
        load_source('study_sources/dynamic/07_landmark_cox.py',execute_main=True)
        d=WORK/'rolling_5y_step7_unpenalized_cox_fixed95_v5'
        np.save(outputs/'cox_survival.npy',np.load(d/'cox_oof_survival_long.npy'))
    if 'rsf' in args.models:
        common={'N_JOBS':2,'HEARTBEAT_SECONDS':10,'TUNING_TREE_N':20,'FINAL_TREE_N':32,'TUNING_TOTAL_COMPLETE_TRIAL_TARGET':1,'TUNING_MAX_TOTAL_ATTEMPT_N':3,'OPTUNA_STARTUP_TRIAL_N':1}
        load_source('study_sources/dynamic/08_landmark_rsf_tuning_oof.py',overrides={**common,'RUN_STAGE':'tune'},execute_main=True)
        load_source('study_sources/dynamic/08_landmark_rsf_tuning_oof.py',overrides={**common,'RUN_STAGE':'final'},execute_main=True)
        d=WORK/'rolling_5y_step8_pooled_landmark_rsf_resume_v2/final_oof'
        np.save(outputs/'rsf_survival.npy',np.load(d/'rsf_oof_survival_long.npy'))
    if 'rnn' in args.models:np.save(outputs/'rnn_survival.npy',train_rnn(args.epochs))
    if 'lstm' in args.models:np.save(outputs/'lstm_survival.npy',train_lstm(args.epochs))
    for f in outputs.glob('*_survival.npy'):
        s=np.load(f);assert np.isfinite(s).all() and ((s>=0)&(s<=1)).all() and (np.diff(s,axis=1)<=1e-6).all(),f
    dump_json(WORK/'training_completed.json',{'synthetic_only':True,'models':args.models,'neural_epochs':args.epochs,'rsf_trials':1,'rsf_trees':32,'folds':5,'neural_seeds_per_fold':1,'snapshots':1,'note':'Uses native classes/functions. Short budgets and one seed for testing, not the manuscript ensemble or a new validation result.'})

if __name__=='__main__':main()
