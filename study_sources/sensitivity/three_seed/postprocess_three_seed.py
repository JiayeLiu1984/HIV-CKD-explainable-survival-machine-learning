"""Recompute calibration, domain occlusion, and all four supplementary figures.
No old two-seed predictions are used as new three-seed outputs.
Run stages in order: calibrate, occlusion, bootstrap, analyze.
"""
from pathlib import Path
import json,ast,types,sys,os,argparse
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import minimize
from typing import Any
B=Path(__file__).resolve().parent;R=Path('__CKD_WORKDIR__')
NB=(B.parent/'source_reference/CKD_dynamic_prediction.before_final_interpretation_cleanup_20260817_154537.ipynb')
OLD=B.parent/'reviewer_minor_4_6_20260908'
os.environ['CKD_LSTM_PROJECT_DIR']=str(R);os.environ['MPLBACKEND']='Agg'

def calibrated(modes=('full','current')):
 src=''.join(json.loads(NB.read_text(encoding='utf8'))['cells'][24]['source']);tree=ast.parse(src)
 keep=['survival_to_hazard','hazard_to_risk','probability_logit','fit_interval_hazard_calibrator','apply_interval_hazard_calibrator']
 ns=dict(np=np,pd=pd,Any=Any,expit=expit,minimize=minimize,EPS=1e-7,EXPECTED_INTERVAL_N=10,CALIBRATION_RIDGE=1e-6)
 exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in keep],type_ignores=[]),str(NB),'exec'),ns)
 s6=R/'rolling_5y_step6_super_landmark_data';s1=R/'rolling_5y_step1_new_split'
 fold=np.load(s6/'development_fold_id_long.npy');lm=np.load(s6/'development_landmark_index_long.npy');local=np.load(s6/'development_local_patient_idx_long.npy');dev=np.load(s1/'development_idx.npy')
 target=np.load(s1/'future_event_matrix.npy')[dev[local],lm];mask=np.load(s1/'future_at_risk_mask.npy')[dev[local],lm]
 assert np.array_equal(target,np.load(s6/'development_future_event_long.npy'))
 assert np.array_equal(mask,np.load(s6/'development_future_at_risk_long.npy'))
 for mode in modes:
  raw=ns['survival_to_hazard'](np.load(B/mode/'lstm_v2_oof_survival_long.npy'))
  out=B/('calibration' if mode=='full' else 'calibration_current');out.mkdir(exist_ok=True)
  prediction=np.full_like(raw,np.nan);rows=[]
  for l in range(6):
   for k in range(5):
    tr=(lm==l)&(fold!=k);te=(lm==l)&(fold==k)
    fit=ns['fit_interval_hazard_calibrator'](raw[tr],target[tr],mask[tr]);prediction[te]=ns['apply_interval_hazard_calibrator'](raw[te],fit)
    for j in range(10):rows.append(dict(model='LSTM-v2',fit_scope='crossfit',heldout_fold=k,landmark_month=l*12,landmark_year=l,interval_position=j,interval_end_month=(j+1)*6,alpha_interval=fit['alpha'][j],beta_common_slope=fit['beta'],train_interval_n=fit['valid_interval_n'],train_event_n=fit['event_interval_n']))
    print(f'Calibration {mode} landmark={l} heldout_fold={k}',flush=True)
  assert np.isfinite(prediction).all();risk=ns['hazard_to_risk'](prediction)
  np.save(out/'lstm_v2_crossfit_calibrated_oof_hazard_long.npy',prediction);np.save(out/'lstm_v2_crossfit_calibrated_oof_risk_long.npy',risk)
  pd.DataFrame(rows).to_csv(out/'four_model_hazard_calibration_parameters.csv',index=False)

def legacy_stage(cell):
 src=''.join(json.loads(NB.read_text(encoding='utf8'))['cells'][cell]['source'])
 replacements={'rolling_5y_step10e_lstm_v2_final_oof_selected_existing_trials':str(B/'full').replace('\\','/'),'rolling_5y_step11_v2_four_model_oof_comparison_lstm_v2':str(B/'calibration').replace('\\','/'),'rolling_5y_step12c_lstm_v2_grouped_reference_occlusion_v1':str(B/'occlusion').replace('\\','/'),'rolling_5y_step12c2_grouped_occlusion_paired_bootstrap_v1':str(B/'occlusion_bootstrap').replace('\\','/')}
 for a,b in replacements.items():src=src.replace(a,b)
 # Change only the inference device helper; preserve FP32 model arithmetic.
 if cell==35:
  node=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='default_device')
  original=ast.get_source_segment(src,node)
  src=src.replace(original,"def default_device() -> torch.device:\n    torch.set_num_threads(4)\n    return torch.device('cpu')")
 if cell==36:
  node=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='run_bootstrap')
  original=ast.get_source_segment(src,node)
  fast=(B/'fast_occlusion_bootstrap.py').read_text(encoding='utf8')
  fastnode=next(n for n in ast.parse(fast).body if isinstance(n,ast.FunctionDef) and n.name=='run_bootstrap')
  src=src.replace(original,ast.get_source_segment(fast,fastnode))
 dest=B/'reference_code'/f'adapted_cell_{cell}.py';dest.write_text(src,encoding='utf8')
 exec(compile(src,str(dest),'exec'),{'__name__':'__main__','__file__':str(dest)})

def analysis():
 src=(OLD/'analyze.py').read_text(encoding='utf8')
 src=src.replace('Read-only reanalysis of saved final Step10E outputs. No fabricated reruns.','Reanalysis of newly trained three-seed Step10E/Step13A outputs.')
 # Match scikit-survival tie groups after each bootstrap sample, including
 # removal of unsampled intermediate risk values that could bridge a tie.
 node=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='auc')
 exact_auc="""def auc(count,t,e,w,order,score,horizon):
 lower=0.;num=0.;wc=0.;cc=0.;ww=0.;previous=0.;started=False
 for z in range(len(order)):
  i=order[z]
  if count[i]<=0:continue
  if started and score[i]-previous>1e-8:
   num+=ww*(lower+.5*cc);wc+=ww;lower+=cc;cc=0.;ww=0.
  started=True;previous=score[i]
  if t[i]>horizon:cc+=count[i]
  elif e[i]:ww+=count[i]*w[i]
 num+=ww*(lower+.5*cc);wc+=ww;lower+=cc
 return num/(wc*lower) if wc*lower>0 else np.nan"""
 src=src.replace(ast.get_source_segment(src,node),exact_auc).replace('aa.append((order,st,np.r_[st[1:],n]))','aa.append((order,self.p[:,m,j]))')
 src=src.replace('@njit\n','@njit(nogil=True)\n')
 src=src.replace('rng=np.random.default_rng(20260908);nboot=1000;', 'from concurrent.futures import ThreadPoolExecutor\npool=ThreadPoolExecutor(max_workers=4)\nrng=np.random.default_rng(20260908);nboot=1000;')
 src=src.replace('for l,c in enumerate(contexts):boot[b,l]=c.calc(count[c.local])','for l,value in enumerate(pool.map(lambda c: c.calc(count[c.local]),contexts)):boot[b,l]=value')
 src=src.replace('for z,(_,_,c) in enumerate(center_contexts):center_boot[b,z]=c.calc(count[c.local])[0]','for z,value in enumerate(pool.map(lambda item: item[2].calc(count[item[2].local])[0],center_contexts)):center_boot[b,z]=value')
 src=src.replace('assert np.isfinite(center_boot).all()', 'pool.shutdown()\nassert np.isfinite(center_boot).all()')
 src=src.replace("full=R/'rolling_5y_step10e_lstm_v2_final_oof_selected_existing_trials'","full=B/'full'").replace("current=R/'rolling_5y_step13a_lstm_v2_current_only_oof_fixed_hyperparameters'","current=B/'current'")
 src=src.replace('for seedpos in range(2):','for seedpos in range(3):').replace('for s in range(2):','for s in range(3):')
 src=src.replace("Mean +/- SD (2 seeds)","Mean +/- SD (3 seeds)").replace("color=colors[s]","color=['#d62728','#1f77b4','#2ca02c'][s]").replace("(-.07 if s==0 else .07)","(.08*(s-1))")
 src=src.replace('FigureS1_Existing_two_seed_variability','FigureS1_Three_seed_variability')
 src=src.replace("R/'rolling_5y_step12c2_grouped_occlusion_paired_bootstrap_v1/grouped_occlusion_paired_bootstrap_CI_all.csv'","B/'occlusion_bootstrap/grouped_occlusion_paired_bootstrap_CI_all.csv'")
 src=src.replace("'new_training_performed':False","'new_training_performed':True").replace('Two saved training seeds per fold, not five or ten pipeline repetitions','Three newly trained seeds per fold under fixed partitions; not repeated partitioning').replace('Mean/SD of seed records is computed within fold, not ten independent pipeline repetitions','Mean/SD is computed within fold across three CPU runs, not 15 independent pipeline repetitions').replace('raw uncalibrated saved Step10E/Step13A risks','raw uncalibrated newly trained three-seed Step10E/Step13A risks')
 src=src.replace("S4_existing_frozen_group_occlusion.csv","S4_three_seed_frozen_group_occlusion.csv")
 dest=B/'reference_code'/'analyze_three_seed_expanded.py';dest.write_text(src,encoding='utf8')
 exec(compile(src,str(dest),'exec'),{'__name__':'__main__','__file__':str(B/'analysis_entry.py')})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['calibrate','occlusion','bootstrap','analyze']);p.add_argument('--model',choices=['full','current','both'],default='both');a=p.parse_args()
 if a.stage=='calibrate':calibrated(['full','current'] if a.model=='both' else [a.model])
 elif a.stage=='occlusion':legacy_stage(35)
 elif a.stage=='bootstrap':legacy_stage(36)
 else:analysis()
