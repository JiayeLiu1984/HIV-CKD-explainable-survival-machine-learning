"""Run the six native data modules on the independently simulated long table."""
from _runtime import load_source, WORK, dump_json

def main():
    modules=['01_patient_split_labels','02_patient_folds','03_raw_tensors','04_fold_preprocessing','05_landmark_summaries','06_landmark_long_data']
    for name in modules:
        print('\nRunning',name,flush=True)
        load_source('study_sources/dynamic/'+name+'.py',execute_main=True)
    dump_json(WORK/'preprocessing_completed.json',{'modules':modules,'data':'synthetic only','patient_level_splits':True,'original_70_30_design_retained':True})

if __name__=='__main__':main()
