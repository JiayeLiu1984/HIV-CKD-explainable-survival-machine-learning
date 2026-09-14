"""python. Reorganize observed OOF results using prediction tertiles.
No training, calibration fitting, outcome-optimized cutoffs or primary-data edits.
"""
from pathlib import Path
import json,hashlib,ast,shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
BASE=Path(__file__).resolve().parent;OLD=BASE.parent/'risk_stratification_20260908';ROOT=Path('__CKD_WORKDIR__')
for folder in ['manuscript','tables','audit','archive_fixed_thresholds']:(BASE/folder).mkdir(exist_ok=True)
# Reuse only the previously audited numerical KM function, not its top-level actions.
source=OLD/'build_risk_stratification.py';tree=ast.parse(source.read_text(encoding='utf8'));node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='km')
exec(compile(ast.Module(body=[node],type_ignores=[]),'<audited_km>','exec'))
shutil.copy2(source,BASE/'audit/original_KM_implementation.py')
meta_path=ROOT/'rolling_5y_step6_super_landmark_data/super_landmark_development_metadata.csv'
pred_path=ROOT/'rolling_5y_step11_v2_four_model_oof_comparison_lstm_v2/lstm_v2_crossfit_calibrated_oof_risk_long.npy'
m=pd.read_csv(meta_path);p=np.load(pred_path,mmap_mode='r')[:,-1].astype(float)
assert np.array_equal(m.long_row_index,np.arange(len(m))) and len(p)==len(m)
assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
assert not m.duplicated(['local_patient_index','landmark_month']).any()
prior=pd.read_csv(OLD/'tables/patient_landmark_assignments.csv');prior=prior[prior.model=='LSTM-v2']
aligned=m.merge(prior,on=['local_patient_index'],suffixes=('','_prior'))
# Explicit two-key merge avoids inferring that old export row order matches metadata.
key=m.assign(landmark_year=m.landmark_month//12).merge(prior,on=['local_patient_index','landmark_year'],validate='one_to_one')
assert np.allclose(p[key.long_row_index.to_numpy(int)],key.risk,rtol=0,atol=1e-15)
assert np.allclose(key.analysis_time_month,key.time_month,rtol=0,atol=1e-12)
assert np.array_equal(key.event_within_60m,key.event)
records=[];curves=[];assignments=[];cutrows=[];checks=[]
names=['Lower','Intermediate','Higher']
for lm in range(6):
    ix=np.flatnonzero(m.landmark_month.to_numpy()==lm*12);q=p[ix];d=m.iloc[ix];t=d.analysis_time_month.to_numpy();e=d.event_within_60m.to_numpy()
    c1,c2=np.quantile(q,[1/3,2/3],method='linear');assert c1<c2
    g=np.searchsorted([c1,c2],q,side='right')
    cutrows.append(dict(landmark_year=lm,n=len(ix),lower_cut_probability=c1,upper_cut_probability=c2,lower_cut_percent=100*c1,upper_cut_percent=100*c2,ties_at_lower=int((q==c1).sum()),ties_at_upper=int((q==c2).sum())))
    assignments.append(pd.DataFrame({'model':'LSTM-v2','local_patient_index':d.local_patient_index,'landmark_year':lm,'risk':q,'group':g,'time_month':t,'event':e}))
    for j in range(3):
        sel=g==j;tt=t[sel];ee=e[sel];u,f,lo,hi,at,ev=km(tt,ee)
        # Independent scalar event-time calculation.
        ss=1.
        for et in np.unique(tt[ee==1]):ss*=1-((tt==et)&(ee==1)).sum()/(tt>=et).sum()
        assert abs(1-ss-f[-1])<1e-12
        curves.append(pd.DataFrame(dict(landmark_year=lm,group=j,time_month=u,km_ckd=f,lower95=lo,upper95=hi,at_risk=at,events=ev)))
        records.append(dict(model='LSTM-v2',landmark_year=lm,group=j,label=names[j],n=int(sel.sum()),events=int(ee.sum()),at_risk_5y=int((tt>=60).sum()),mean_predicted_risk=float(q[sel].mean()),km_5y=float(f[-1]),lower95=float(lo[-1]),upper95=float(hi[-1]),cut1=float(c1),cut2=float(c2)))
        checks.append(abs(1-ss-f[-1]))
s=pd.DataFrame(records);cur=pd.concat(curves);a=pd.concat(assignments);cuts=pd.DataFrame(cutrows)
old=pd.read_csv(OLD/'tables/risk_group_summary.csv').query("model=='LSTM-v2' and scheme=='tertiles'")
v=s.merge(old,on=['model','landmark_year','group'],suffixes=('','_old'),validate='one_to_one')
for col in ['n','events','km_5y','lower95','upper95','cut1','cut2']:assert np.allclose(v[col],v[col+'_old'],rtol=0,atol=1e-12)
for df,name in [(s,'tertile_summary.csv'),(cur,'tertile_KM_curves.csv'),(a,'patient_landmark_assignments.csv'),(cuts,'tertile_cutpoints.csv')]:df.to_csv(BASE/'tables'/name,index=False,encoding='utf-8-sig')
for name in ['times.ttf','timesbd.ttf']:font_manager.fontManager.addfont('C:/Windows/Fonts/'+name)
plt.rcParams.update({'font.family':'Times New Roman','font.size':11,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
colors=['#0072B2','#E6AB02','#D55E00'];fig=plt.figure(figsize=(15,10.8));gs=fig.add_gridspec(2,3,hspace=.38,wspace=.27)
ymax=float(np.ceil(cur.upper95.max()*100/5)*5)
for lm in range(6):
    grid=gs[lm//3,lm%3].subgridspec(2,1,height_ratios=[4,1.1],hspace=.38);ax=fig.add_subplot(grid[0]);ar=fig.add_subplot(grid[1]);ar.axis('off')
    for j,col in enumerate(colors):
        d=cur[(cur.landmark_year==lm)&(cur.group==j)];x=np.r_[0,d.time_month/12]
        ax.step(x,np.r_[0,d.km_ckd*100],where='post',color=col,lw=1.8,label=names[j]+' predicted risk')
        ax.fill_between(x,np.r_[0,d.lower95*100],np.r_[0,d.upper95*100],step='post',color=col,alpha=.12)
        tt=a.loc[(a.landmark_year==lm)&(a.group==j),'time_month'].to_numpy()/12
        ar.text(0,.56-j*.28,names[j],ha='left',va='center',fontsize=9,color=col,transform=ar.transAxes)
        for year in range(6):ar.text(.29+.71*year/5,.56-j*.28,str(int((tt>=year).sum())),ha='center',va='center',fontsize=9,transform=ar.transAxes)
    ar.text(0,.9,'At risk / year',fontsize=9,transform=ar.transAxes)
    for year in range(6):ar.text(.29+.71*year/5,.9,str(year),ha='center',fontsize=9,transform=ar.transAxes)
    ax.set(xlim=(0,5),ylim=(0,ymax),xticks=range(6),xlabel=f'Years after year-{lm} landmark',ylabel='CKD probability, 1 - KM (%)')
    ax.text(-.16,1.04,chr(97+lm),transform=ax.transAxes,fontweight='bold',fontsize=15);ax.legend(frameon=False,fontsize=8,loc='upper left');ax.grid(alpha=.13)
fig.savefig(BASE/'manuscript/Figure1_LSTM_tertiles.pdf',bbox_inches='tight');fig.savefig(BASE/'manuscript/Figure1_LSTM_tertiles.png',bbox_inches='tight',dpi=180);plt.close(fig)
for path in [OLD/'figures/Figure1_LSTM_risk_groups.pdf',OLD/'figures/Tables_risk_groups.pdf',OLD/'tables/risk_group_summary.csv']:
    shutil.copy2(path,BASE/'archive_fixed_thresholds'/path.name)
protocol=dict(primary='landmark-specific relative predicted-risk tertiles',quantile_method='numpy linear 1/3 and 2/3',boundary_rule='lower: p<c1; intermediate: c1<=p<c2; higher: p>=c2. Equal predictions are not split by ID.',origin='previously calculated sensitivity analysis promoted to main display in response to user/reviewer; not preregistered',population='development-cohort calibrated OOF; all eligible patients at each landmark',new_training=False,model='Step10E chain A LSTM-v2',future_window='five years after each landmark',clinical_threshold=False,external_validation=False,threshold_uses_outcomes=False,group_order_checked_in_existing_sensitivity=True,KM_scalar_error_max=max(checks),prior_tertile_results_reproduced=True,plot_CI_upper_bound=ymax,clinical_scores_comparison='not completed; Cox is not DAD/VHA')
(BASE/'audit/protocol_and_verification.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
manifest=[dict(file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in [source,meta_path,pred_path,OLD/'tables/risk_group_summary.csv']]
(BASE/'audit/source_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf8')
print(s.to_string(index=False));print(cuts.to_string(index=False))
