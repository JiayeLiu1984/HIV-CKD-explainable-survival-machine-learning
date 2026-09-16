"""Frozen synthetic LSTM ensemble, then explicitly separate local recalibration."""
import argparse
import numpy as np
import pandas as pd
import torch, joblib
from sklearn.model_selection import StratifiedKFold
from _runtime import ROOT, WORK, DATA, load_source, dump_json
from module_00_generate_synthetic_data import LABS, ART, STATUS, MEDS
from module_03_model_evaluation import to_hazard, apply_offsets, calibrate, metrics, dca

CONT=['Age','BMI']+LABS+[a+'_cum_month' for a in ART]
BIN=['Oppinfection']+STATUS+MEDS+['current_'+a for a in ART]
CAT=['Sex','Marriage','Course','WHOstage']

def panel_arrays(d):
    ids=sorted(d.ID.unique());n=len(ids)
    cont=np.zeros((n,11,len(CONT)),np.float32);binary=np.zeros((n,11,len(BIN)),np.float32);cat=np.full((n,11,4),'__NO_TIME_ROW__',dtype=object);mask=np.zeros((n,11),bool)
    y=np.zeros((n,6,10),np.float32);risk=np.zeros_like(y);pairs=[];rows=[];age=np.zeros(n,np.float32)
    lookup={s:i for i,s in enumerate(ids)}
    for ident,g in d.groupby('ID'):
        i=lookup[ident];b=g.time_bin.to_numpy(int);cont[i,b]=g[CONT];binary[i,b]=g[BIN];cat[i,b]=g[CAT].astype(str);mask[i,b]=True
        first=g.iloc[0];age[i]=first.Age;t=float(first.interval);event=int(first.CKDstatus)
        for lm,s in enumerate(range(0,61,12)):
            rem=t-s
            if rem<=0 or (not event and rem<6):continue
            for j in range(10):
                start=6*j;end=6*(j+1)
                if event:
                    risk[i,lm,j]=float(rem>start);y[i,lm,j]=float(start<rem<=end)
                else:risk[i,lm,j]=float(rem>=end)
            pairs.append((i,lm));rows.append({'ID':ident,'local_patient_index':i,'landmark_index':lm,'landmark_month':s,'analysis_time_month':min(rem,60),'event_within_60m':int(event and rem<=60)})
    return cont,binary,cat,mask,age,y,risk,np.asarray(pairs),pd.DataFrame(rows)

def frozen_members(d):
    """Build datasets using development-fitted transforms and frozen models."""
    torch.set_num_threads(2)
    c,b,cat,mask,age,y,risk,pairs,meta=panel_arrays(d);n=len(c);members=[]
    m=load_source('study_sources/dynamic/lstm_core.py');m.EXPECTED_DEVELOPMENT_N=n
    for fold in range(5):
        prep=joblib.load(WORK/f'rolling_5y_step4_preprocessed/fold_{fold}/preprocessor.joblib')
        names=pd.read_csv(WORK/f'rolling_5y_step4_preprocessed/fold_{fold}/feature_names.csv').feature_name.tolist()
        x=np.zeros((n,11,57),np.float32)
        x[mask]=np.c_[prep['scaler'].transform(c[mask]),b[mask],prep['encoder'].transform(cat[mask])]
        static_names=['BMI','Oppinfection']+[k for k in names if any(k.startswith(v+'_') for v in CAT)]
        dynamic=x[:,:,[names.index(k) for k in m.DYNAMIC_FEATURES]]
        enhanced,_=m.build_enhanced_dynamic_array(dynamic,c[:,:,[CONT.index(k) for k in LABS]],mask)
        ds=m.EnhancedPatientLandmarkDataset(pairs,enhanced,mask,x[:,0,[names.index(k) for k in static_names]],age,y,risk,prep['scaler'].mean_[0],prep['scaler'].scale_[0])
        checkpoint=torch.load(WORK/f'synthetic_lstm_fold_{fold}.pt',map_location='cpu',weights_only=False)
        assert checkpoint['synthetic_only'];model=m.build_model(checkpoint['parameters'],torch.device('cpu'));model.load_state_dict(checkpoint['state']);model.eval()
        members.append((m,model,ds))
    return members,meta,y,risk,pairs


def frozen_predict(d):
    """Identical frozen predictor for internal test and external validation."""
    import json,hashlib
    lock=json.loads((WORK/'model_lock.json').read_text(encoding='utf-8'))
    for name,digest in lock['artifact_sha256'].items():
        assert hashlib.sha256((WORK/name).read_bytes()).hexdigest()==digest,name
    members,meta,y,risk,pairs=frozen_members(d);all_s=[]
    for m,model,ds in members:
        hazards=[]
        with torch.no_grad():
            for raw in torch.utils.data.DataLoader(ds,batch_size=256):
                batch=m.move_batch_to_device(raw,torch.device('cpu'))
                logits=model(**{k:batch[k] for k in ['dynamic_sequence','row_mask','static_baseline','age_at_landmark','landmark_normalized','landmark_bin']})
                hazards.append(torch.sigmoid(logits).numpy())
        all_s.append(np.cumprod(1-np.concatenate(hazards),axis=1))
    raw_s=np.mean(all_s,axis=0);h=to_hazard(raw_s)
    # Development-fitted calibration is frozen before any external outcome use.
    offsets=np.load(WORK/'development_cv/lstm_development_offsets.npy');primary=np.empty_like(raw_s)
    for lm in range(6):
        select=meta.landmark_index.to_numpy()==lm;primary[select]=apply_offsets(h[select],offsets[lm])
    return meta,primary,y,risk,pairs


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--with-recalibration",action="store_true");args=parser.parse_args()
    d=pd.read_csv(DATA/'synthetic_external.csv',dtype={'ID':str,'WHOstage':str})
    meta,primary,y,risk,pairs=frozen_predict(d);n=d.ID.nunique()
    out=WORK/'external_validation';out.mkdir(exist_ok=True);np.save(out/'primary_frozen_survival.npy',primary)
    meta.to_csv(out/'external_metadata.csv',index=False);metrics(meta,primary).to_csv(out/'primary_performance.csv',index=False)
    analyses=[("primary_frozen",primary)]
    if args.with_recalibration:
        # Secondary adaptation uses a patient-level five-fold assignment inside the synthetic external set.
        outcomes=pd.read_csv(DATA/'synthetic_external.csv').drop_duplicates('ID').set_index('ID').CKDstatus
        id_order=meta.ID.drop_duplicates().tolist();fold_map={}
        for fold,(_,held) in enumerate(StratifiedKFold(5,shuffle=True,random_state=20260912).split(id_order,outcomes.loc[id_order])):
            for i in held:fold_map[id_order[i]]=fold
        meta['fold_id']=meta.ID.map(fold_map);yl=y[pairs[:,0],pairs[:,1]];ml=risk[pairs[:,0],pairs[:,1]]
        secondary,_=calibrate(to_hazard(primary),yl,ml,meta)
        np.save(out/'secondary_recalibrated_survival.npy',secondary);metrics(meta,secondary).to_csv(out/'secondary_performance.csv',index=False)
        analyses.append(("secondary_local_recalibration",secondary))
    curves=[]
    for label,s in analyses:
        for lm in range(6):
            idx=meta.landmark_index.to_numpy()==lm
            curves.extend({'analysis':label,'landmark':lm,**row} for row in dca(meta[idx],1-s[idx,-1]))
    pd.DataFrame(curves).to_csv(out/'decision_curves.csv',index=False)
    dump_json(out/'scope.json',{'synthetic_only':True,'primary':'Development fold transforms, synthetic fold models and development calibration frozen. No external fitting.','secondary':'Separate cross-fitted intercept-only adaptation using synthetic external outcomes.','secondary_executed':args.with_recalibration,'external_n':int(n),'dataset':'external_validation','model_artifact':'development-selected frozen five-fold LSTM ensemble','all_ids_disjoint':True})
    print('Frozen external prediction completed; secondary recalibration:', args.with_recalibration)

if __name__=='__main__':main()
