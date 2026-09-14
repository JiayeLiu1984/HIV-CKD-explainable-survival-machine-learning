from pathlib import Path
import sys,json,os
W=Path(__file__).parent;# Install requirements in your own environment; no bundled runtime is required.
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import Surv
P=Path(os.environ.get('CKD_WORKDIR', 'private_study_inputs'));S=P/'comment1_ART_censoring_sensitivity_20260905';stage='ART_sensitivity_frozen_primary_calibration'
plt.rcParams.update({'font.family':'Times New Roman','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
perf=pd.read_csv(W/'ART_verified_six_landmarks.csv').iloc[:6]
colors=plt.get_cmap('tab10')(np.arange(6))
for key,title,num in [('C_index','Uno C-index',1),('iAUC','Integrated AUC',2)]:
 fig,ax=plt.subplots(figsize=(8,5))
 y=perf[key].to_numpy();ax.errorbar(np.arange(6),y,yerr=np.array([y-perf[key+'_lower'],perf[key+'_upper']-y]),fmt='o-',color='#215e85',capsize=5)
 ax.set(xticks=np.arange(6),xlabel='Prediction landmark after ART initiation (years)',ylabel=title,ylim=(.76,.91))
 ax.set_title('ART sensitivity with frozen primary calibration',pad=14);ax.grid(axis='y',alpha=.2)
 fig.tight_layout();fig.savefig(W/f'ART_FigS9_{num}.png',dpi=220);plt.close(fig)
h=pd.read_csv(S/'sensitivity_horizon_metrics.csv');h=h[h.stage==stage]
fig,ax=plt.subplots(figsize=(8,5))
for k,g in h.groupby('landmark_position'):ax.plot(g.horizon_month/12,g.brier,label=f'Landmark {k}',color=colors[k])
ax.set(xlabel='Years after prediction landmark',ylabel='IPCW Brier score',xlim=(.5,5));ax.set_title('ART sensitivity with frozen primary calibration');ax.legend(ncol=3,frameon=False);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(W/'ART_FigS9_3.png',dpi=220);plt.close(fig)
c=pd.read_csv(S/'sensitivity_calibration_deciles.csv');c=c[(c.stage==stage)&(c.horizon_month==60)]
fig,axs=plt.subplots(2,3,figsize=(10,7),sharex=True,sharey=True)
for k,ax in enumerate(axs.flat):
 g=c[c.landmark_position==k];ax.plot([0,.5],[0,.5],'--',color='.55');ax.errorbar(g.mean_predicted_risk,g.km_observed_risk,yerr=[g.km_observed_risk-g.km_lower_95,g.km_upper_95-g.km_observed_risk],fmt='o-',capsize=2,color=colors[k]);ax.set(title=f'Landmark {k}',xlim=(0,.5),ylim=(0,.5));ax.grid(alpha=.15)
fig.supxlabel('Mean predicted five-year CKD risk');fig.supylabel('Observed five-year CKD risk');fig.suptitle('ART sensitivity with frozen primary calibration');fig.tight_layout();fig.savefig(W/'ART_FigS9_4.png',dpi=220);plt.close(fig)
meta=pd.read_csv(P/'rolling_5y_step6_super_landmark_data/super_landmark_development_metadata.csv',usecols=['landmark_index']);t=np.load(S/'sensitivity_analysis_time_month.npy').astype(float);e=np.load(S/'sensitivity_event_within_60m.npy').astype(bool);r=np.load(P/'rolling_5y_step11_v2_four_model_oof_comparison_lstm_v2/lstm_v2_crossfit_calibrated_oof_risk_long.npy',mmap_mode='r')
thresholds=np.linspace(.01,.30,100);rows=[];fig,axs=plt.subplots(2,3,figsize=(10,7),sharex=True,sharey=True)
for k,ax in enumerate(axs.flat):
 ix=np.flatnonzero(meta.landmark_index.to_numpy()==k);tt=t[ix];ee=e[ix];rr=np.asarray(r[ix,-1]);horizon=59.999
 ce=CensoringDistributionEstimator().fit(Surv.from_arrays(ee,tt));case=ee&(tt<=horizon);control=tt>horizon
 wc=np.zeros(len(ix));wc[case]=1/ce.predict_proba(tt[case]);wn=control.astype(float)/ce.predict_proba(np.array([horizon]))[0]
 nb=[];allnb=[]
 for z in thresholds:
  pos=rr>=z;odds=z/(1-z);v=(wc[pos].sum()-odds*wn[pos].sum())/len(ix);a=(wc.sum()-odds*wn.sum())/len(ix);nb.append(v);allnb.append(a);rows.append({'landmark':k,'threshold':z,'LSTM_net_benefit':v,'treat_all':a,'treat_none':0})
 ax.plot(thresholds,nb,color=colors[k],label='LSTM');ax.plot(thresholds,allnb,'--',color='.5',label='Treat all');ax.axhline(0,color='black',lw=.8,label='Treat none');ax.set(title=f'Landmark {k}',ylim=(-.02,.065));ax.grid(alpha=.15)
axs[0,0].legend(frameon=False,fontsize=9);fig.supxlabel('Threshold probability');fig.supylabel('IPCW net benefit');fig.suptitle('ART sensitivity with frozen primary calibration');fig.tight_layout();fig.savefig(W/'ART_FigS9_5.png',dpi=220);plt.close(fig)
pd.DataFrame(rows).to_csv(W/'ART_FigS9_DCA_aggregate.csv',index=False)
print('Five ART components rebuilt from verified frozen-primary sources')
