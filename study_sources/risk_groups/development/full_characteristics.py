from pathlib import Path
import json,zipfile,hashlib
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from scipy.stats import kruskal,chi2_contingency,random_table

OUT=Path(__file__).resolve().parent;ROOT=Path('__CKD_WORKDIR__');S3=ROOT/'rolling_5y_step3_raw_features'
REF=Path('__CKD_LOCAL_INPUT__/多中心艾滋ML/修稿数据/Table1_baseline_开发队列.docx')
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
with zipfile.ZipFile(REF) as z:root=ET.fromstring(z.read('word/document.xml'))
tr=root.find('.//w:tbl',ns).findall('w:tr',ns)
labels=[''.join(r.find('w:tc',ns).itertext()) for r in []]
labels=[''.join(t.text or '' for t in r.find('w:tc',ns).findall('.//w:t',ns)) for r in tr]
fg=json.loads((S3/'feature_groups.json').read_text());cont=np.load(S3/'continuous_raw_0_60.npy',mmap_mode='r');binary=np.load(S3/'binary_raw_0_60.npy',mmap_mode='r');cat=np.load(S3/'categorical_raw_0_60.npy',allow_pickle=True);mask=np.load(S3/'sequence_row_mask.npy')
m=pd.read_csv(ROOT/'rolling_5y_step6_super_landmark_data/super_landmark_development_metadata.csv');a=pd.read_csv(OUT.parent/'tables/patient_landmark_assignments.csv');a=a[a.model=='LSTM-v2'];m['landmark_year']=m.landmark_month//12
m=m.merge(a[['local_patient_index','landmark_year','risk','group']],left_on=['local_patient_index','landmark_year'],right_on=['local_patient_index','landmark_year'],validate='one_to_one')
assert len(m)==100122
# Serum creatinine was not a tensor feature: use only keyed compatible records from the available source.
source=ROOT/'深圳南宁随访数据最终.csv'
raw=pd.read_csv(source,encoding='gb18030',usecols=['ID','time_bin','SCr']+fg['continuous_vars'][:17],low_memory=False)
assert not raw.duplicated(['ID','time_bin']).any()
raw=raw.set_index(['ID','time_bin'])
continuous={1:'Age',23:'BMI',24:'HIVRNA_log10',25:'CD4',26:'CD8',27:'ratio',28:'eGFR',29:'SCr',30:'Urea',31:'WBC',32:'PLT',33:'HB',34:'TC',35:'TG',36:'HDL',37:'LDL',38:'GLU',39:'ALT',40:'AST'}
categorical={2:('Sex',['Female','Male']),5:('Marriage',['Never married','Married or cohabiting','Divorced-separated-or-widowed','Others']),10:('Course',['Heterosexual','Male to male','Drugs','Others']),15:('Oppinfection',[0,1]),18:('WHOstage',[1,2,3,4]),41:('CVD_status',[0,1]),44:('hypertension_status',[0,1]),47:('diabetes_status',[0,1]),50:('hypercholesterolemia_status',[0,1]),53:('antihypertensive_med',[0,1]),56:('antidiabetic_med',[0,1]),59:('antilipid_med',[0,1]),62:('HBV_status',[0,1]),65:('HCV_status',[0,1]),68:('ART',list(range(9)))}
all_results=[];qa=[]
def fmtp(p):return 'n.a.' if p is None else ('<0.001' if p<.001 else f'{p:.3f}')
for lm in [0,5]:
    d=m[m.landmark_year==lm].copy();gid=d.global_patient_index.to_numpy(int);step=int(lm*2);ok=mask[gid,step].astype(bool);groups=d.group.to_numpy(int)
    x={v:cont[gid,step,i].astype(float) for i,v in enumerate(fg['continuous_vars'])}
    x.update({v:binary[gid,step,i].astype(float) for i,v in enumerate(fg['binary_vars'])})
    x.update({v:cat[gid,step,i].astype(object) for i,v in enumerate(fg['categorical_vars'])})
    x['WHOstage']=pd.to_numeric(pd.Series(x['WHOstage']),errors='coerce').to_numpy()
    x['ratio']=np.divide(x['CD4'],x['CD8'],out=np.full(len(d),np.nan),where=x['CD8']>0)
    match=raw.reindex(pd.MultiIndex.from_arrays([d.ID,np.full(len(d),step)]));compatible=ok.copy()
    source_agreement={v:int(np.isclose(x[v].astype(float),match[v].to_numpy(float),atol=1e-4,rtol=1e-5,equal_nan=False).sum()) for v in fg['continuous_vars'][:17]}
    # BMI and Urea differ across exports. Never substitute those tensor variables.
    # The supplementary SCr column is keyed uniquely and checked against age,
    # eGFR and 13 other shared laboratory fields; its distinct provenance is retained.
    for v in [v for v in fg['continuous_vars'][:17] if v not in ['BMI','Urea']]:compatible&=np.isclose(x[v].astype(float),match[v].to_numpy(float),atol=1e-4,rtol=1e-5,equal_nan=False)
    x['SCr']=np.where(compatible,match.SCr.to_numpy(float),np.nan)
    artnames=[v for v in fg['binary_vars'] if v.startswith('current_')];art=np.column_stack([x[v] for v in artnames]);assert len(artnames)==8
    nart=(art==1).sum(1);x['ART']=np.where(nart==0,8,np.argmax(art,axis=1)).astype(float);x['ART'][nart>1]=np.nan
    table=[[labels[i],'','','',''] for i in range(78)]
    sizes=[int((groups==j).sum()) for j in range(3)]
    table[0]=['Characteristic']+[f'{name}\n(n = {n:,})' for name,n in zip(['Lower risk','Intermediate risk','Higher risk'],sizes)]+['P value']
    details=[]
    for row,v in continuous.items():
        vals=x[v].astype(float).copy();vals[~ok]=np.nan;sets=[vals[(groups==j)&np.isfinite(vals)] for j in range(3)]
        p=float(kruskal(*sets).pvalue) if all(len(s)>0 for s in sets) and len(np.unique(np.concatenate(sets)))>1 else None
        for j,s in enumerate(sets):
            q=np.quantile(s,[.25,.5,.75]) if len(s) else np.full(3,np.nan)
            table[row][j+1]=f'{q[1]:,.2f} [{q[0]:,.2f}, {q[2]:,.2f}]' if len(s) else 'n.a.'
            details.append(dict(row=row,variable=v,group=j,n=sizes[j],nonmissing_n=len(s),missing_n=sizes[j]-len(s),q25=float(q[0]),median=float(q[1]),q75=float(q[2]),p_value=p,test='Kruskal-Wallis'))
        table[row][4]=fmtp(p)
    for heading,(v,levels) in categorical.items():
        vals=x[v];valid=ok&pd.Series(vals).isin(levels).to_numpy();counts=np.array([[int(((groups==j)&valid&(vals==lev)).sum()) for j in range(3)] for lev in levels]);assert np.array_equal(counts.sum(0),[int(((groups==j)&valid).sum()) for j in range(3)])
        counts_test=counts[counts.sum(1)>0];method='Pearson chi-square';p=None;minexp=None
        if len(counts_test)>1 and (counts_test.sum(0)>0).all():
            stat,p,_,ex=chi2_contingency(counts_test,correction=False);minexp=float(ex.min())
            if ex.min()<1 or (ex<5).mean()>.2:
                rng=np.random.default_rng(20260908+lm*100+heading);sim=random_table.rvs(counts_test.sum(1),counts_test.sum(0),size=9999,random_state=rng)
                ss=(((sim-ex)**2)/ex).sum((1,2));p=float((1+(ss>=stat-1e-12).sum())/10000);method='Conditional Monte Carlo Pearson statistic (9999 tables)'
        table[heading+1][4]=fmtp(p)
        for li,lev in enumerate(levels):
            for j in range(3):
                nn=int(counts[:,j].sum());count=int(counts[li,j]);table[heading+1+li][j+1]=f'{count:,} ({100*count/nn:.1f})' if nn else 'n.a.'
                details.append(dict(row=heading+1+li,variable=v,category=str(lev),group=j,n=sizes[j],nonmissing_n=nn,missing_n=sizes[j]-nn,count=count,p_value=float(p) if p is not None else None,test=method,minimum_expected_count=minexp))
    assert [r[0] for r in table]==labels
    for i in range(1,78):
        if i not in categorical:assert all(table[i][j] for j in [1,2,3]),i
    qa.append(dict(landmark_year=lm,n=len(d),sizes=sizes,SCr_compatible_n=int(compatible.sum()),SCr_incompatible_or_missing_n=int((~compatible).sum()),current_row_missing=[int(((groups==j)&~ok).sum()) for j in range(3)],multiple_current_ART_n=int(((nart>1)&ok).sum()),source_variable_agreement=source_agreement))
    all_results.append(dict(landmark_year=lm,table=table,details=details))
(OUT/'full_table_values.json').write_text(json.dumps(all_results,ensure_ascii=False,indent=2,allow_nan=True),encoding='utf8')
(OUT/'data_audit.json').write_text(json.dumps({'reference_sha256':hashlib.sha256(REF.read_bytes()).hexdigest(),'source_SCr':str(source),'qa':qa,'grouping':'new landmark-specific prediction tertiles; frozen predictions unchanged'},indent=2),encoding='utf8')
print(json.dumps(qa,indent=2))
