"""Synthetic risk tertiles, transferred thresholds, characteristics and figures."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sksurv.nonparametric import kaplan_meier_estimator
from _runtime import WORK, DATA, dump_json
from module_03_model_evaluation import km_risk

def main():
    out=WORK/'figures';out.mkdir(exist_ok=True)
    dev=pd.read_csv(WORK/'rolling_5y_step6_super_landmark_data/super_landmark_development_metadata.csv')
    ext=pd.read_csv(WORK/'external_validation/external_metadata.csv')
    dev['risk']=1-np.load(WORK/'evaluation/lstm_calibrated_survival.npy')[:,-1]
    ext['risk']=1-np.load(WORK/'external_validation/primary_frozen_survival.npy')[:,-1]
    rows=[];characteristics=[];thresholds=[];fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for lm in range(6):
        d=dev[dev.landmark_index==lm].copy();e=ext[ext.landmark_index==lm].copy();cuts=np.quantile(d.risk,[1/3,2/3])
        thresholds.append({'landmark':lm,'lower':cuts[0],'upper':cuts[1]})
        for label,m in [('development',d),('external_transported',e)]:
            groups=np.searchsorted(cuts,m.risk,side='right')
            panel=pd.read_csv(DATA/f'synthetic_{"development" if label=="development" else "external"}.csv')
            panel=panel[panel.month<=lm*12].sort_values('month').groupby('ID').tail(1).set_index('ID')
            for group in range(3):
                part=m.iloc[np.flatnonzero(groups==group)]
                rows.append({'cohort':label,'landmark':lm,'group':group,'n':len(part),'events':int(part.event_within_60m.sum()),'observed_5y_risk':km_risk(part)})
                if lm in [0,5]:
                    for name in ['Age','eGFR','CD4','GLU','BMI','diabetes_status','hypertension_status']:
                        v=panel.loc[part.ID,name]
                        characteristics.append({'cohort':label,'landmark':lm,'group':group,'feature':name,'n':len(v),'median':v.median(),'Q1':v.quantile(.25),'Q3':v.quantile(.75)})
                if label=='development' and len(part):
                    t,s=kaplan_meier_estimator(part.event_within_60m.to_numpy(bool),part.analysis_time_month.to_numpy())
                    axes.flat[lm].step(np.r_[0,t]/12,np.r_[0,1-s],where='post',label=['Lower','Intermediate','Higher'][group])
        axes.flat[lm].set(title=f'Landmark {lm} years',xlabel='Years after landmark',ylabel='1 − KM survival',xlim=(0,5),ylim=(0,1));axes.flat[lm].legend(fontsize=8)
    fig.suptitle('SYNTHETIC DATA — descriptive risk groups');fig.savefig(out/'synthetic_risk_groups.png',dpi=160);plt.close(fig)
    pd.DataFrame(rows).to_csv(out/'risk_groups.csv',index=False);pd.DataFrame(characteristics).to_csv(out/'risk_group_characteristics.csv',index=False);pd.DataFrame(thresholds).to_csv(out/'development_tertile_cutpoints.csv',index=False)
    table=pd.read_csv(WORK/'evaluation/landmark_performance.csv');fig,axes=plt.subplots(1,3,figsize=(12,3.7),constrained_layout=True)
    for ax,metric in zip(axes,['Uno_C_index','iAUC','IBS']):
        for model,g in table.groupby('model'):ax.plot(g.landmark_index,g[metric],marker='o',label=model.upper())
        ax.set(title=metric,xlabel='Landmark (years)');ax.legend(fontsize=8)
    fig.suptitle('SYNTHETIC DATA — short-budget model test');fig.savefig(out/'synthetic_model_comparison.png',dpi=160);plt.close(fig)
    dump_json(out/'scope.json',{'synthetic_only':True,'thresholds':'Development tertiles transferred unchanged to synthetic external predictions; descriptive only.','clinical_benefit_demonstrated':False})
    print('Synthetic risk-group summaries and figures completed.')

if __name__=='__main__':main()
