from pathlib import Path
import json, hashlib, ast
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
B=Path(__file__).resolve().parent; R=Path('__CKD_WORKDIR__')
P=R/'重庆外部验证_step7R_FINAL_fold_specific'; S6=R/'重庆外部验证_step6_LSTM_v2'; S5=R/'重庆外部验证_step5_GLU更新'
source=B.parent/'risk_stratification_20260908/build_risk_stratification.py'
node=next(x for x in ast.parse(source.read_text(encoding='utf8')).body if isinstance(x,ast.FunctionDef) and x.name=='km')
exec(compile(ast.Module(body=[node],type_ignores=[]),'<audited KM>','exec'))
m=pd.read_csv(S6/'super_landmark_external_metadata.csv'); export=pd.read_csv(P/'重庆_LSTM_v2_FINAL_patient_landmark_predictions.csv')
p=np.load(P/'external_calibrated_risk_long.npy')[:,-1].astype(float)
assert len(m)==len(p)==15714 and m.ID.nunique()==2673
assert np.array_equal(m.long_row_index,np.arange(len(m))) and not m.duplicated(['ID','landmark_month']).any()
for col in m.columns:
    if col in export: assert m[col].equals(export[col]),col
assert np.allclose(p,export.predicted_risk_5y,rtol=1e-6,atol=1e-9)
for col,file in [('analysis_time_month','external_analysis_time_month.npy'),('event_within_60m','external_event_within_60m.npy'),('local_patient_index','external_local_patient_idx_long.npy')]:assert np.allclose(m[col],np.load(S6/file))
assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
assert ((m.analysis_time_month>0)&(m.analysis_time_month<=60)).all()
info=pd.read_csv(S5/'patient_info.csv'); assert np.array_equal(info.patient_index,np.arange(len(info)))
assert np.array_equal(info.ID.to_numpy()[m.local_patient_index],m.ID)
fg=json.loads((R/'rolling_5y_step3_raw_features/feature_groups.json').read_text())
raw=pd.read_csv(S5/'重庆_source47_imputed_GLU_updated.csv');assert not raw.duplicated(['ID','time_bin']).any()
raw=raw.set_index(['ID','time_bin']); mask=np.load(S5/'sequence_row_mask.npy').astype(bool)
assert np.array_equal(mask,np.load(S6/'sequence_row_mask.npy').astype(bool))
ii,tt=np.where(mask); rr=raw.reindex(pd.MultiIndex.from_arrays([info.ID.to_numpy()[ii],tt])); matches={}
for group,file in [('continuous_vars','continuous_raw_0_60.npy'),('binary_vars','binary_raw_0_60.npy'),('categorical_vars','categorical_raw_0_60.npy')]:
    ar=np.load(S5/file,allow_pickle=True)
    assert ar.shape==(2677,11,len(fg[group])),(group,ar.shape)
    for j,v in enumerate(fg[group]):
        a=ar[ii,tt,j];b=rr[v].to_numpy()
        equal=np.isclose(a.astype(float),b.astype(float),atol=1e-4,rtol=1e-5,equal_nan=True) if group!='categorical_vars' else a.astype(str)==b.astype(str)
        assert equal.all(),(v,int((~equal).sum()))
        matches[v]=int(equal.sum())
dev=pd.read_csv(B.parent/'risk_tertiles_main_20260908/tables/tertile_cutpoints.csv')
records=[];curves=[];assign=[];cuts=[]
for scheme in ['development_cutpoints','external_tertiles']:
  for lm in range(6):
    ix=np.flatnonzero(m.landmark_month.to_numpy()==lm*12);d=m.iloc[ix];q=p[ix]
    cp=dev.loc[dev.landmark_year==lm,['lower_cut_probability','upper_cut_probability']].iloc[0].to_numpy() if scheme=='development_cutpoints' else np.quantile(q,[1/3,2/3],method='linear')
    g=np.searchsorted(cp,q,side='right');t=d.analysis_time_month.to_numpy();e=d.event_within_60m.to_numpy()
    cuts.append(dict(scheme=scheme,landmark_year=lm,lower_cut_percent=cp[0]*100,upper_cut_percent=cp[1]*100))
    assign.append(pd.DataFrame(dict(model='LSTM-v2',scheme=scheme,local_patient_index=d.local_patient_index,landmark_year=lm,risk=q,group=g,time_month=t,event=e)))
    for j in range(3):
      sel=g==j;assert sel.any();tm=t[sel];ev=e[sel];u,f,lo,hi,at,de=km(tm,ev)
      independent=1.
      for et in np.unique(tm[ev==1]):independent*=1-((tm==et)&(ev==1)).sum()/(tm>=et).sum()
      assert abs(1-independent-f[-1])<1e-12
      supported=bool(tm.max()>=60)
      # Do not extend a KM estimate beyond the observed support.
      records.append(dict(scheme=scheme,landmark_year=lm,group=j,n=int(sel.sum()),events=int(ev.sum()),at_risk_5y=int((tm>=60).sum()),mean_predicted_risk=q[sel].mean(),km_5y=f[-1] if supported else np.nan,lower95=lo[-1] if supported and ev.sum()>0 else np.nan,upper95=hi[-1] if supported and ev.sum()>0 else np.nan,cut1=cp[0],cut2=cp[1],five_year_supported=supported,ci_estimable=bool(ev.sum()>0)))
      if ev.sum()==0:lo[:]=np.nan;hi[:]=np.nan
      curves.append(pd.DataFrame(dict(scheme=scheme,landmark_year=lm,group=j,time_month=u,km_ckd=f,lower95=lo,upper95=hi)))
s=pd.DataFrame(records);c=pd.concat(curves);a=pd.concat(assign)
for df,name in [(s,'risk_group_summary'),(c,'km_curves'),(a,'assignments_all'),(pd.DataFrame(cuts),'cutpoints_all')]:df.to_csv(B/'tables'/f'{name}.csv',index=False,encoding='utf-8-sig')
a[a.scheme=='development_cutpoints'].to_csv(B/'tables/patient_landmark_assignments.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(cuts).query("scheme=='development_cutpoints'").to_csv(B/'tables/tertile_cutpoints.csv',index=False,encoding='utf-8-sig')
for f in ['times.ttf','timesbd.ttf']:font_manager.fontManager.addfont('C:/Windows/Fonts/'+f)
plt.rcParams.update({'font.family':'Times New Roman','font.size':11,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
names=['Lower','Intermediate','Higher'];colors=['#0072B2','#E6AB02','#D55E00']
ymax=max(5,float(np.ceil(c.upper95.max()*100/5)*5))
for scheme in ['development_cutpoints','external_tertiles']:
    fig=plt.figure(figsize=(15,10.8));gs=fig.add_gridspec(2,3,hspace=.38,wspace=.27)
    for lm in range(6):
        sg=gs[lm//3,lm%3].subgridspec(2,1,height_ratios=[4,1.1],hspace=.38);ax=fig.add_subplot(sg[0]);ar=fig.add_subplot(sg[1]);ar.axis('off')
        for j,col in enumerate(colors):
            d=c[(c.scheme==scheme)&(c.landmark_year==lm)&(c.group==j)];x=np.r_[0,d.time_month/12]
            ax.step(x,np.r_[0,d.km_ckd*100],where='post',color=col,lw=1.8,label=names[j]+' predicted risk')
            ax.fill_between(x,np.r_[0,d.lower95*100],np.r_[0,d.upper95*100],step='post',color=col,alpha=.12)
            tm=a.loc[(a.scheme==scheme)&(a.landmark_year==lm)&(a.group==j),'time_month'].to_numpy()/12
            ar.text(0,.56-j*.28,names[j],ha='left',va='center',fontsize=9,color=col,transform=ar.transAxes)
            for yr in range(6):ar.text(.29+.71*yr/5,.56-j*.28,str(int((tm>=yr).sum())),ha='center',va='center',fontsize=9,transform=ar.transAxes)
        ar.text(0,.9,'At risk / year',fontsize=9,transform=ar.transAxes)
        for yr in range(6):ar.text(.29+.71*yr/5,.9,str(yr),ha='center',fontsize=9,transform=ar.transAxes)
        ax.set(xlim=(0,5),ylim=(0,ymax),xticks=range(6),xlabel=f'Years after year-{lm} landmark',ylabel='CKD probability, 1 - KM (%)')
        ax.text(-.16,1.04,chr(65+lm),transform=ax.transAxes,fontweight='bold',fontsize=15);ax.legend(frameon=False,fontsize=8,loc='upper left');ax.grid(alpha=.13)
    fig.savefig(B/'figures'/f'Chongqing_{scheme}.pdf',bbox_inches='tight');fig.savefig(B/'figures'/f'Chongqing_{scheme}.png',bbox_inches='tight',dpi=180);plt.close(fig)
sources=[P/'step7R_FINAL_manifest.json',P/'external_calibrated_risk_long.npy',S6/'super_landmark_external_metadata.csv',S5/'重庆_source47_imputed_GLU_updated.csv',B.parent/'risk_tertiles_main_20260908/tables/tertile_cutpoints.csv']
(B/'audit/source_verification.json').write_text(json.dumps(dict(model='Step7R frozen external LSTM-v2; Step11 development-only calibration',external_recalibration=False,model_retrained=False,primary='development cutpoints transferred unchanged per landmark',sensitivity='external cohort landmark-specific tertiles; descriptive',feature_array_source_agreement=matches,sources=[dict(path=str(x),sha256=hashlib.sha256(x.read_bytes()).hexdigest()) for x in sources]),ensure_ascii=False,indent=2),encoding='utf8')
print(s.to_string(index=False))
