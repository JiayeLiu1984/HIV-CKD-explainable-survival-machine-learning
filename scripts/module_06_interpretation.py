"""Internal-test attribution of the frozen calibrated LSTM ensemble; no fitting."""
import numpy as np
import pandas as pd
import torch
from _runtime import WORK, DATA, dump_json, load_source
from module_04_external_validation import frozen_members

KEYS=['dynamic_sequence','row_mask','static_baseline','age_at_landmark','landmark_normalized','landmark_bin']


def main():
    torch.set_num_threads(2)
    meta=pd.read_csv(WORK/'internal_validation/internal_test_metadata.csv')
    panel=pd.read_csv(DATA/'synthetic_development.csv',dtype={'ID':str,'WHOstage':str})
    members,meta,_,_,_=frozen_members(panel[panel.ID.isin(meta.ID)].copy())
    chosen=[]
    for lm in range(6):chosen.extend(np.flatnonzero(meta.landmark_index.to_numpy()==lm)[:2].tolist())
    inputs=[];references=[];names=None
    for fold,(m,model,ds) in enumerate(members):
        raw=next(iter(torch.utils.data.DataLoader(torch.utils.data.Subset(ds,chosen),batch_size=len(chosen))))
        batch=m.move_batch_to_device(raw,torch.device('cpu'));inputs.append(batch)
        # References use only development training patients for this member.
        m.EXPECTED_DEVELOPMENT_N=len(pd.read_csv(WORK/'rolling_5y_step6_super_landmark_data/super_landmark_development_metadata.csv').ID.unique())
        common=m.load_common_data();f=m.prepare_fold_data(common,fold);train=f['train_dataset'];names=f['enhanced_dynamic_features']
        ids=np.unique(train.sample_pairs[:,0]);valid=train.sequence_mask[ids]
        ref={k:v.clone() for k,v in batch.items()}
        ref['dynamic_sequence']=torch.tensor(train.enhanced_dynamic[ids][valid].mean(axis=0))[None,None,:].expand_as(batch['dynamic_sequence']).clone()*batch['row_mask'][:,:,None]
        ref['static_baseline']=torch.tensor(train.static_baseline[ids].mean(axis=0))[None,:].expand_as(batch['static_baseline']).clone()
        ref['age_at_landmark']=torch.zeros_like(batch['age_at_landmark']);references.append(ref)
    offsets=torch.tensor(np.load(WORK/'development_cv/lstm_development_offsets.npy')[meta.iloc[chosen].landmark_index.to_numpy()],dtype=torch.float32)
    def prediction(batches):
        curves=[]
        for (_,model,_),batch in zip(members,batches):
            curves.append(torch.cumprod(1-torch.sigmoid(model(**{k:batch[k] for k in KEYS})),dim=1))
        s=torch.stack(curves).mean(0)
        prior=torch.cat([torch.ones((len(s),1)),s[:,:-1]],dim=1)
        hazard=(1-s/prior.clamp_min(1e-7)).clamp(1e-6,1-1e-6)
        return 1-torch.prod(1-torch.sigmoid(torch.logit(hazard)+offsets),dim=1)
    attrs=['dynamic_sequence','static_baseline','age_at_landmark'];steps=256
    grads=[{k:torch.zeros_like(b[k]) for k in attrs} for b in inputs]
    for j,alpha in enumerate(torch.linspace(0,1,steps+1)):
        batches=[];variables=[]
        for b,r in zip(inputs,references):
            z=dict(b)
            for k in attrs:z[k]=(r[k]+alpha*(b[k]-r[k])).detach().requires_grad_(True);variables.append(z[k])
            batches.append(z)
        values=torch.autograd.grad(prediction(batches).sum(),variables)
        for fold,g in enumerate(grads):
            for k,value in zip(attrs,values[fold*3:(fold+1)*3]):g[k]+=value*(.5 if j in [0,steps] else 1.)/steps
    ig=[{k:(b[k]-r[k])*g[k] for k in attrs} for b,r,g in zip(inputs,references,grads)]
    with torch.no_grad():full=prediction(inputs);reference=prediction(references)
    total=sum(v.reshape(len(chosen),-1).sum(1) for values in ig for v in values.values());residual=full-reference-total
    out=WORK/'interpretation';out.mkdir(exist_ok=True)
    ids=meta.iloc[chosen][['ID','landmark_index']].reset_index(drop=True)
    ids.assign(risk=full.numpy(),reference_risk=reference.numpy(),sum_IG=total.detach().numpy(),completeness_residual=residual.detach().numpy()).to_csv(out/'IG_completeness.csv',index=False)
    combined=sum(v['dynamic_sequence'] for v in ig)
    pd.DataFrame({'feature':names,'mean_absolute_IG':combined.abs().sum(1).mean(0).detach().numpy()}).to_csv(out/'global_dynamic_IG.csv',index=False)
    occluded=[{k:v.clone() for k,v in b.items()} for b in inputs]
    perturbed=[{k:v.clone() for k,v in b.items()} for b in inputs]
    for fold,b in enumerate(inputs):
        for i,lm in enumerate(b['landmark_bin']):
            occluded[fold]['dynamic_sequence'][i,:int(lm)]=references[fold]['dynamic_sequence'][i,:int(lm)]
            perturbed[fold]['dynamic_sequence'][i,int(lm)+1:]+=100
    with torch.no_grad():masked=prediction(occluded);future=prediction(perturbed)
    ids.assign(full_risk=full.numpy(),history_occluded_risk=masked.numpy(),difference=(masked-full).numpy()).to_csv(out/'temporal_occlusion.csv',index=False)
    expected=np.load(WORK/'internal_validation/lstm_frozen_survival.npy')[chosen,-1]
    prediction_error=float(np.max(np.abs(full.numpy()-(1-expected))))
    delta=float((future-full).abs().max())
    dump_json(out/'verification.json',{'synthetic_only':True,'dataset':'internal_test','tested_origins':len(chosen),'IG_steps':steps,'max_completeness_residual':float(residual.abs().max()),'future_perturbation_max_risk_change':delta,'frozen_prediction_max_error':prediction_error,'reference':'Development fold-training means; fixed masks and landmarks.','scope':'Frozen calibrated five-member LSTM ensemble on internal test samples; no fitting.'})
    assert delta<1e-6 and prediction_error<1e-5


if __name__=='__main__':main()
