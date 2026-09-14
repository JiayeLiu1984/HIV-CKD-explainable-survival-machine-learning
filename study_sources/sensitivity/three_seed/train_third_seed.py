"""Rerun three genuine seeds in the same CPU/FP32 environment.
Original inputs, hyperparameters, losses, snapshot selection and fold membership
are preserved. Existing GPU/AMP runs are NOT mixed into new CPU/FP32 results.
This is not a new nested-validation design. Original outputs are read-only.
Run: python -X utf8 train_third_seed.py
"""
from pathlib import Path
import os,sys,json,ast,types,time,hashlib,shutil,gc,argparse
import numpy as np
import pandas as pd
import torch

B=Path(__file__).resolve().parent;R=Path('__CKD_WORKDIR__')
os.environ['CKD_LSTM_PROJECT_DIR']=str(R);os.environ['CKD_HISTORY_MODE']='current_only'
ORIGINAL={'full':R/'rolling_5y_step10e_lstm_v2_final_oof_selected_existing_trials','current':R/'rolling_5y_step13a_lstm_v2_current_only_oof_fixed_hyperparameters'}
SOURCES={'full':((B.parent/'source_reference/CKD_dynamic_prediction.ipynb'),23),'current':((B.parent/'source_reference/CKD_dynamic_prediction.before_final_interpretation_cleanup_20260817_154537.ipynb'),41)}
def savejson(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf8')
def log(x):
 s=f'{time.strftime("%Y-%m-%d %H:%M:%S")} {x}';print(s,flush=True)
 with (B/'training.log').open('a',encoding='utf8') as f:f.write(s+'\n')
def core(mode):
 p,i=SOURCES[mode];src=''.join(json.loads(p.read_text(encoding='utf8'))['cells'][i]['source'])
 node=next(x for x in ast.parse(src).body if isinstance(x,ast.FunctionDef) and x.name=='default_device')
 code='\n'.join(src.splitlines()[:node.end_lineno])+'\n'
 dest=B/'reference_code';dest.mkdir(exist_ok=True)
 (dest/f'{mode}_original_core.py').write_text(code,encoding='utf8')
 m=types.ModuleType('reference_'+mode);m.__file__=str(dest/f'{mode}_original_core.py');sys.modules[m.__name__]=m
 exec(compile(code,m.__file__,'exec'),m.__dict__)
 m.REQUIRE_CUDA=False;m.USE_AMP=False
 return m,{'notebook':str(p),'cell_index':i,'notebook_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'core_sha256':hashlib.sha256(code.encode()).hexdigest()}

def run(mode,only_fold=None,benchmark=False):
 m,source=core(mode);old=ORIGINAL[mode];out=B/mode;out.mkdir(exist_ok=True)
 summary=json.loads((old/'lstm_v2_summary.json').read_text(encoding='utf8'));params=summary['selected_parameters']
 torch.set_num_threads(int(os.getenv('CKD_RUN_THREADS','4')));torch.set_num_interop_threads(1) if not getattr(run,'threads_set',False) else None;run.threads_set=True
 common=m.load_common_data();device=torch.device('cpu')
 savejson(out/'run_configuration.json',{'source':source,'old_model':str(old),'seed_positions':[0,1,2],'seed_formula':'20260830 + fold_id*1000 + seed_position*100','parameters':params,'device':'cpu','torch':torch.__version__,'AMP':False,'all_three_seeds_retrained_in_same_environment':True,'old_GPU_runs_not_averaged_into_new_model':True,'seed_ensemble_n':3,'limitations':['Legacy baseline input timing unchanged','Legacy validation-guided early stopping and snapshot selection unchanged','Original tuning and folds unchanged','This is a new CPU/FP32 three-seed rerun, not the original saved Step10E weights']})
 for k in range(5):
  if only_fold is not None and k!=only_fold:continue
  fold=out/f'fold_{k}';fold.mkdir(exist_ok=True)
  if (fold/'completed.json').exists():log(f'{mode} fold {k}: completed; preserving result');continue
  data=m.prepare_fold_data(common,k)
  assert np.array_equal(data['validation_long_idx'],np.load(old/f'fold_{k}/validation_long_idx.npy'))
  actualseed=20260830+k*1000+200
  if benchmark:
   m.set_random_seed(actualseed);model=m.build_model(params,device);loss=m.LandmarkBalancedSurvivalLoss(positive_weight=1.,focal_gamma=0.,auxiliary_5y_weight=0.,ranking_weight=.1,smoothness_weight=.0005)
   loader,_=m.create_data_loaders(data,256,actualseed,device);raw=next(iter(loader));batch=m.move_batch_to_device(raw,device);opt=torch.optim.AdamW(model.parameters(),lr=.0003)
   times=[]
   for j in range(4):
    tic=time.time();opt.zero_grad();y=model(**{n:batch[n] for n in ['dynamic_sequence','row_mask','static_baseline','age_at_landmark','landmark_normalized','landmark_bin']});v,_=loss(y,batch['event_target'],batch['at_risk_mask'],batch['landmark_index']);v.backward();opt.step();times.append(time.time()-tic)
   savejson(B/'CPU_benchmark.json',{'batch_times_seconds':times,'training_batches_per_epoch':len(loader),'estimated_train_only_epoch_seconds':np.median(times[1:])*len(loader)})
   log(f'Benchmark {times}, {len(loader)} batches/epoch');return
  ss=json.loads((old/f'fold_{k}/fold_summary.json').read_text(encoding='utf8'));ss['seed_summaries']=[];hazards=[]
  for seed_position in range(3):
   actualseed=20260830+k*1000+seed_position*100
   prefix=f'seed_{seed_position}'
   if (fold/f'{prefix}_completed.json').exists():
    ss['seed_summaries'].append(json.loads((fold/f'{prefix}_completed.json').read_text(encoding='utf8')))
    hazards.append(np.load(fold/f'{prefix}_validation_hazard.npy'));log(f'RESUME {mode} fold={k} seed={actualseed}');continue
   log(f'START {mode} fold {k}/4 seed={actualseed}; authentic CPU training')
   def epoch(msg):
    log(f'{mode} fold={k} seed={actualseed} {msg}')
    savejson(fold/'progress.json',{'mode':mode,'fold':k,'seed':actualseed,'last_epoch_message':msg,'status':'training','updated':time.strftime('%Y-%m-%d %H:%M:%S')})
   result=m.fit_fold_model(common=common,fold_data=data,parameters=params,seed=actualseed,device=device,use_amp=False,progress_callback=epoch)
   result['history'].to_csv(fold/f'{prefix}_training_history.csv',index=False)
   result['landmark_metrics'].to_csv(fold/f'{prefix}_landmark_metrics.csv',index=False)
   np.save(fold/f'{prefix}_validation_hazard.npy',result['hazard']);hazards.append(result['hazard'])
   torch.save({'stage':'three_seed_rerun','mode':mode,'fold_id':k,'seed':actualseed,'selected_trial_number':66,'parameters':params,'snapshot_epochs':result['snapshot_epochs'],'snapshot_state_dicts':result['snapshot_state_dicts'],'feature_names':{'static':data['static_features'],'enhanced_dynamic':data['enhanced_dynamic_features']},'device':'CPU/FP32'},fold/f'{prefix}_snapshot_ensemble.pt')
   seedsummary={'fold_id':k,'seed_position':seed_position,'seed':actualseed,'selected_snapshot_n':result['selected_snapshot_n'],'snapshot_epochs':result['snapshot_epochs'],**result['metric_summary'],'parameter_n':result['parameter_n'],'elapsed_seconds':result['elapsed_seconds'],'backend':'CPU/FP32'}
   ss['seed_summaries'].append(seedsummary);savejson(fold/f'{prefix}_completed.json',seedsummary)
   del result;gc.collect()
  assert len({x['seed'] for x in ss['seed_summaries']})==3
  haz=np.mean(np.stack(hazards,axis=0),axis=0).astype(np.float32)
  survival=m.hazards_to_survival(haz);risk=(1-survival).astype(np.float32)
  met,tab=m.compute_equal_weight_landmark_metrics(common,data['train_long_idx'],data['validation_long_idx'],haz)
  for name,v in [('hazard',haz),('survival',survival),('risk',risk)]:np.save(fold/f'validation_{name}.npy',v)
  for name in ['validation_long_idx','validation_patient_local','validation_landmark_index']:
   shutil.copy2(old/f'fold_{k}/{name}.npy',fold/f'{name}.npy')
  tab.insert(0,'fold_id',k);tab.to_csv(fold/'landmark_metrics.csv',index=False)
  ss.update(seed_ensemble_n=3,final_mean_ibs=met['mean_ibs'],final_mean_iauc=met['mean_iauc'],final_mean_uno_c=met['mean_uno_c'],elapsed_seconds=sum(x['elapsed_seconds'] for x in ss['seed_summaries']))
  savejson(fold/'fold_summary.json',ss);savejson(fold/'completed.json',{'completed':True,'third_seed':actualseed,'new_ensemble_seed_n':3,'completed_at':time.strftime('%Y-%m-%d %H:%M:%S')})
  log(f'COMPLETE {mode} fold={k}; {met}')
  del data,hazards;gc.collect()
 if not all((out/f'fold_{k}/completed.json').exists() for k in range(5)):return
 allh=np.full((len(common.analysis_time_month),10),np.nan,dtype=np.float32);tabs=[]
 for k in range(5):
  ix=np.load(out/f'fold_{k}/validation_long_idx.npy');allh[ix]=np.load(out/f'fold_{k}/validation_hazard.npy');tabs.append(pd.read_csv(out/f'fold_{k}/landmark_metrics.csv'))
 assert np.isfinite(allh).all()
 alls=m.hazards_to_survival(allh);allr=1-alls
 for name,v in [('hazard',allh),('survival',alls),('risk',allr)]:np.save(out/f'lstm_v2_oof_{name}_long.npy',v)
 table=pd.concat(tabs);table.to_csv(out/'lstm_v2_fold_landmark_metrics.csv',index=False)
 summary.pop('total_elapsed_seconds',None)
 summary.update(seed_ensemble_n_per_fold=3,stage='Step10E_three_seed_rerun' if mode=='full' else 'Step13A_three_seed_rerun',five_fold_mean_ibs=float(table.ibs.mean()),five_fold_mean_iauc=float(table.iauc.mean()),five_fold_mean_uno_c=float(table.uno_c_index_5y.mean()),output_dir=str(out),cpu_threads=4,old_GPU_weights_reused=False)
 savejson(out/'lstm_v2_summary.json',summary);log(f'ALL FIVE FOLDS COMPLETE for {mode}')

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--mode',choices=['full','current','both'],default='both');a.add_argument('--fold',type=int);a.add_argument('--benchmark',action='store_true');args=a.parse_args()
 for mode in (['full','current'] if args.mode=='both' else [args.mode]):run(mode,args.fold,args.benchmark)
