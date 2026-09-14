"""Independently simulate fictitious subjects; NEVER reads any clinical file."""
import argparse, hashlib
import numpy as np
import pandas as pd
from _runtime import ROOT, DATA, dump_json

LABS=['HIVRNA_log10','CD4','CD8','Urea','WBC','PLT','HB','TC','TG','HDL','LDL','GLU','ALT','AST','eGFR']
ART=['TDF_NNRTI_3TC_FTC','TDF_PI_3TC_FTC','nonTDF_PI','BIC_FTC_TAF','EVGc_FTC_TAF','TDF_INSTI_3TC_FTC','nonTDF_DTG','nonTDF_traditional_NNRTI']
STATUS=['CVD_status','diabetes_status','hypertension_status','hypercholesterolemia_status','HBV_status','HCV_status']
MEDS=['antidiabetic_med','antihypertensive_med','antilipid_med']

def generate(n, seed, external=False):
    rng=np.random.default_rng(seed);rows=[]
    for i in range(n):
        ident=f'SYN_{"EXT" if external else "DEV"}_{i:06d}'
        center='Chongqing' if external else ['Shenzhen','Nanning'][int(rng.integers(0,2))]
        age=float(rng.uniform(20,70));bmi=float(rng.uniform(18,32))
        base=np.array([4.,400.,650.,5.,6.,200.,140.,5.,2.,1.3,3.,6.,30.,30.,100.])
        scale=np.array([.8,110.,160.,1.,1.,40.,15.,.8,.5,.25,.7,1.3,12.,12.,15.])
        labs=np.maximum(.1,base+rng.normal(size=15)*scale)
        sex=['Female','Male'][int(rng.integers(0,2))]
        marriage=['Married or cohabiting','Never married','Divorced-separated-or-widowed','Others'][int(rng.integers(0,4))]
        course=['Heterosexual','Male to male','Drugs','Others'][int(rng.integers(0,4))]
        who=str(rng.integers(1,5))
        opp=int(rng.random()<.25);status=(rng.random(6)<.15).astype(int)
        med=(rng.random(3)<.18).astype(int)
        regimen=int(rng.integers(0,9));cum=np.zeros(8);patient=[]
        censor=float(rng.uniform(122,180));art_flag=0
        # Enrich artificial events across all six windows so a small five-fold
        # integration dataset has evaluable horizons. This is NOT an incidence model.
        event_probability=1/(1+np.exp(-(.5+.025*(age-40))))
        event_time=(float(rng.choice(np.arange(0,61,12)))+float(rng.uniform(.2,5.9))) if rng.random()<event_probability else np.inf
        for t in range(0,181,6):
            if t>=min(event_time,censor):break
            if t and rng.random()<.25:regimen=int(rng.integers(0,9))
            if t:
                status=np.maximum(status,(rng.random(6)<.035).astype(int))
                med=np.maximum(med,(rng.random(3)<(.03+.05*status[1:4])).astype(int))
                labs=np.maximum(.1,labs+rng.normal(size=15)*scale*.20)
                labs[-1]=max(15.,labs[-1]-rng.uniform(0,.9))
            rec={'ID':ident,'data':center,'time_bin':t//6,'month':t,'Age':age,'BMI':bmi,'Sex':sex,'Marriage':marriage,'Course':course,'WHOstage':who,'Oppinfection':opp}
            rec.update(zip(LABS,labs));rec.update(zip(STATUS,status));rec.update(zip(MEDS,med))
            rec.update({'current_'+a:int(regimen==j) for j,a in enumerate(ART)})
            rec.update({a+'_cum_month':float(cum[j]) for j,a in enumerate(ART)})
            if t==0 or rng.random()>.10: patient.append(rec)
            if t < event_time <= t+6: art_flag=int(regimen in [1,2,3,4,5,6])
            if regimen<8:cum[regimen]+=6
        observed=min(event_time,censor);event=int(event_time<censor)
        for rec in patient:
            rec.update(interval=observed,CKDstatus=event,synthetic_art_affected=int(event and art_flag and i%2==0),synthetic=True)
            if rec['month']<observed and rec['month']<=60:rows.append(rec)
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=600);p.add_argument('--external-n',type=int,default=240);p.add_argument('--seed',type=int,default=20260912);a=p.parse_args()
    if a.n<300 or a.external_n<100: p.error('Use at least 300 development and 100 external synthetic subjects for five-fold tests.')
    DATA.mkdir(exist_ok=True)
    dev=generate(a.n,a.seed);ext=generate(a.external_n,a.seed+1,True)
    for name,df in [('development',dev),('external',ext)]:df.to_csv(DATA/f'synthetic_{name}.csv',index=False,encoding='utf-8')
    baseline=dev.loc[dev.time_bin==0,['ID','interval','CKDstatus','Age','BMI','Oppinfection']+LABS].copy()
    baseline.to_csv(DATA/'example_dataset.csv',index=False,encoding='utf-8')
    meta={'synthetic_only':True,'generator':'module_00_generate_synthetic_data.py','seed':a.seed,'development_n':a.n,'external_n':a.external_n,'origin':'Independent NumPy draws with manually specified distributions; no patient rows, cohort-fitted generator, or trained model used.','warning':'Tests software execution only; cannot reproduce manuscript estimates.'}
    meta['files']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in DATA.glob('*.csv')}
    dump_json(DATA/'SYNTHETIC_DATA_PROVENANCE.json',meta)
    records=[]
    for c in dev.columns:
        role='outcome' if c in ['interval','CKDstatus','synthetic_art_affected'] else 'identifier/time' if c in ['ID','data','month','time_bin','synthetic'] else 'predictor'
        records.append({'column':c,'dtype':str(dev[c].dtype),'role':role,'description':{'interval':'Event/censor time in months from ART initiation; repeated per patient.','time_bin':'Six-month bin, month = 6 * time_bin.','synthetic_art_affected':'Simulated outcome sensitivity flag; NEVER a predictor.','CKDstatus':'Single-event CKD outcome: 1=event, 0=right censoring. No competing-risk model.','data':'Synthetic centre label; disjoint IDs across centres.'}.get(c,'Artificial value; see generator. Cumulative ART columns use months.')})
    pd.DataFrame(records).to_csv(DATA/'data_dictionary.csv',index=False)
    print(f'Generated {a.n} development and {a.external_n} external fictional subjects.')

if __name__=='__main__':main()
