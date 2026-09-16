"""Final evaluation of the development-selected frozen LSTM on the locked 30% test set."""
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from _runtime import WORK, DATA, dump_json
from module_04_external_validation import frozen_predict
from module_03_model_evaluation import metrics, km_risk, dca


def verify_lock():
    lock=json.loads((WORK/'model_lock.json').read_text(encoding='utf-8'))
    assert lock['selected_model']=='lstm'
    for name,digest in lock['artifact_sha256'].items():
        assert hashlib.sha256((WORK/name).read_bytes()).hexdigest()==digest,name
    return lock


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bootstrap',type=int,default=20);args=parser.parse_args()
    lock=verify_lock()
    base=WORK/'rolling_5y_step6_super_landmark_data'
    test=pd.read_csv(base/'super_landmark_test_metadata.csv')
    test_ids=set(test.ID)
    assert not test_ids.intersection(lock['development_ids'])
    panel=pd.read_csv(DATA/'synthetic_development.csv',dtype={'ID':str,'WHOstage':str})
    panel=panel[panel.ID.isin(test_ids)].copy()
    meta,s,_,_,_=frozen_predict(panel)
    assert set(meta.ID)==test_ids
    out=WORK/'internal_validation';out.mkdir(exist_ok=True)
    meta.to_csv(out/'internal_test_metadata.csv',index=False)
    np.save(out/'lstm_frozen_survival.npy',s)
    table=metrics(meta,s);table['model']='lstm';table['dataset']='internal_test'
    table.to_csv(out/'landmark_performance.csv',index=False)
    keys=['Uno_C_index','iAUC','IBS']
    calibration=[];decisions=[]
    for lm in range(6):
        use=meta.landmark_index.to_numpy()==lm;m=meta[use];risk=1-s[use,-1]
        for bin_id,idx in enumerate(np.array_split(np.argsort(risk),5)):
            calibration.append({'dataset':'internal_test','landmark':lm,'bin':bin_id,'n':len(idx),'predicted':float(risk[idx].mean()),'observed_1_KM':km_risk(m.iloc[idx])})
        decisions.extend({'dataset':'internal_test','landmark':lm,**r} for r in dca(m,risk))
    pd.DataFrame(calibration).to_csv(out/'calibration.csv',index=False)
    pd.DataFrame(decisions).to_csv(out/'decision_curves.csv',index=False)
    rng=np.random.default_rng(20260912);ids=meta.ID.unique();lookup={i:np.flatnonzero(meta.ID.to_numpy()==i) for i in ids};boot=[]
    for replicate in range(args.bootstrap):
        idx=np.concatenate([lookup[i] for i in rng.choice(ids,len(ids),replace=True)])
        values=metrics(meta.iloc[idx].reset_index(drop=True),s[idx],reference=meta)[keys]
        boot.append({'replicate':replicate,**{k:float(values[k].mean()) if values[k].notna().all() else np.nan for k in keys}})
    boot=pd.DataFrame(boot);boot.to_csv(out/'patient_bootstrap.csv',index=False)
    pd.DataFrame([{'dataset':'internal_test','model':'lstm','metric':k,'estimate':table[k].mean() if table[k].notna().all() else np.nan,'lower_95':boot[k].quantile(.025),'upper_95':boot[k].quantile(.975),'valid_bootstraps':int(boot[k].notna().sum())} for k in keys]).to_csv(out/'six_landmark_mean.csv',index=False)
    verify_lock()
    dump_json(out/'scope.json',{'synthetic_only':True,'dataset':'internal_test','n_patients':len(test_ids),'selected_model':'lstm','used_for_training_tuning_selection_or_calibration':False,'artifacts_unchanged':True,'bootstrap_unit':'patient','censoring_distribution':'Estimated for evaluation only; never changes predictor or calibration','model_lock':'../model_lock.json'})
    print('Final internal evaluation completed on',len(test_ids),'reserved patients.')


if __name__=='__main__':main()
