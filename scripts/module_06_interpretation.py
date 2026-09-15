"""Synthetic native-LSTM IG completeness and temporal-occlusion integration test."""
import numpy as np
import pandas as pd
import torch
from _runtime import WORK, load_source, dump_json

KEYS=['dynamic_sequence','row_mask','static_baseline','age_at_landmark','landmark_normalized','landmark_bin']

def prediction(model,batch):return 1-torch.prod(1-torch.sigmoid(model(**{k:batch[k] for k in KEYS})),dim=1)

def main():
    torch.set_num_threads(2);m=load_source('study_sources/dynamic/lstm_core.py');common=m.load_common_data();f=m.prepare_fold_data(common,0)
    ck=torch.load(WORK/'synthetic_lstm_fold_0.pt',map_location='cpu',weights_only=False);assert ck['synthetic_only']
    model=m.build_model(ck['parameters'],torch.device('cpu'));model.load_state_dict(ck['state']);model.eval()
    # Use fold-training means only; keep missing-row masks and landmark context fixed.
    train=f['train_dataset'];dynamic=train.enhanced_dynamic
    train_pat=np.unique(train.sample_pairs[:,0]);valid=train.sequence_mask[train_pat]
    ref_dynamic=dynamic[train_pat][valid].mean(axis=0)
    ref_static=train.static_baseline[train_pat].mean(axis=0)
    val=f['validation_dataset'];chosen=[]
    for lm in range(6):chosen.extend(np.flatnonzero(val.sample_pairs[:,1]==lm)[:2].tolist())
    batch=next(iter(torch.utils.data.DataLoader(torch.utils.data.Subset(val,chosen),batch_size=len(chosen))))
    b=m.move_batch_to_device(batch,torch.device('cpu'));base={k:v.clone() for k,v in b.items()}
    base['dynamic_sequence']=torch.tensor(ref_dynamic)[None,None,:].expand_as(b['dynamic_sequence']).clone()*b['row_mask'][:,:,None]
    base['static_baseline']=torch.tensor(ref_static)[None,:].expand_as(b['static_baseline']).clone();base['age_at_landmark']=torch.zeros_like(b['age_at_landmark'])
    attributes=['dynamic_sequence','static_baseline','age_at_landmark'];nsteps=128;grads={k:torch.zeros_like(b[k]) for k in attributes}
    for j,alpha in enumerate(torch.linspace(0,1,nsteps+1)):
        interpolated={k:v for k,v in b.items()}
        for k in attributes:interpolated[k]=(base[k]+alpha*(b[k]-base[k])).detach().requires_grad_(True)
        gs=torch.autograd.grad(prediction(model,interpolated).sum(),[interpolated[k] for k in attributes])
        for k,g in zip(attributes,gs):grads[k]+=g*(.5 if j in [0,nsteps] else 1.)/nsteps
    ig={k:(b[k]-base[k])*grads[k] for k in attributes}
    with torch.no_grad():full=prediction(model,b);reference=prediction(model,base)
    total=sum(v.reshape(len(chosen),-1).sum(1) for v in ig.values());residual=full-reference-total
    out=WORK/'interpretation';out.mkdir(exist_ok=True)
    pd.DataFrame({'synthetic_origin':chosen,'landmark':b['landmark_index'].numpy(),'risk':full.numpy(),'reference_risk':reference.numpy(),'sum_IG':total.detach().numpy(),'completeness_residual':residual.detach().numpy()}).to_csv(out/'IG_completeness.csv',index=False)
    pd.DataFrame({'feature':f['enhanced_dynamic_features'],'mean_absolute_IG':ig['dynamic_sequence'].abs().sum(1).mean(0).detach().numpy()}).to_csv(out/'global_dynamic_IG.csv',index=False)
    occluded={k:v.clone() for k,v in b.items()}
    for i,lm in enumerate(b['landmark_bin']):occluded['dynamic_sequence'][i,:int(lm)]=base['dynamic_sequence'][i,:int(lm)]
    with torch.no_grad():masked=prediction(model,occluded)
    pd.DataFrame({'synthetic_origin':chosen,'landmark':b['landmark_index'].numpy(),'full_risk':full.numpy(),'history_occluded_risk':masked.numpy(),'difference':(masked-full).numpy()}).to_csv(out/'temporal_occlusion.csv',index=False)
    # Critical regression: features after the landmark must not alter this prediction.
    perturbed={k:v.clone() for k,v in b.items()}
    for i,lm in enumerate(b['landmark_bin']):perturbed['dynamic_sequence'][i,int(lm)+1:]+=100
    with torch.no_grad():future=prediction(model,perturbed)
    delta=float((future-full).abs().max());assert delta<1e-6,delta
    dump_json(out/'verification.json',{'synthetic_only':True,'tested_origins':len(chosen),'IG_steps':nsteps,'max_completeness_residual':float(residual.abs().max()),'future_perturbation_max_risk_change':delta,'reference':'Means from training patients excluding evaluated fold; landmark and masks fixed.','scope':'Native synthetic LSTM risk before calibration; full formal calibrated study IG is retained separately.'})
    print('IG/occlusion and future-information check completed.')

if __name__=='__main__':main()
