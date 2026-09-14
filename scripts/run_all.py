"""Run all portable modules in independent processes. Run from any working directory."""
import argparse, os, subprocess, sys
from datetime import datetime
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=600);p.add_argument('--external-n',type=int,default=240);p.add_argument('--epochs',type=int,default=2);p.add_argument('--bootstrap',type=int,default=20);p.add_argument('--three-seeds',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parent;env=os.environ.copy();env['MPLBACKEND']='Agg';env['PYTHONIOENCODING']='utf-8'
    env.setdefault('CKD_WORKDIR',str(root.parent/'results'/('synthetic_'+datetime.now().strftime('%Y%m%d_%H%M%S'))))
    commands=[['module_00_generate_synthetic_data.py','--n',str(a.n),'--external-n',str(a.external_n)],['module_01_data_preprocessing.py'],['module_02_model_training.py','--epochs',str(a.epochs)],['module_03_model_evaluation.py','--bootstrap',str(a.bootstrap)],['module_04_external_validation.py'],['module_05_interpretation.py'],['module_06_risk_groups_and_figures.py'],['module_07_sensitivity.py','--epochs',str(a.epochs)]+(['--three-seeds'] if a.three_seeds else [])]
    for command in commands:
        subprocess.run([sys.executable,'-X','utf8',str(root/command[0]),*command[1:]],check=True,env=env,cwd=root.parent)
    subprocess.run([sys.executable,'-m','unittest','discover','-s',str(root.parent/'tests'),'-v'],check=True,env=env,cwd=root.parent)
    print('All requested synthetic modules and integrity tests passed.')

if __name__=='__main__':main()
