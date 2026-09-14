"""Portable entry point for exported analytical source, with explicit configuration."""
from pathlib import Path
import ast, json, os, sys, types
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get('CKD_WORKDIR', ROOT/'results'/'synthetic_run')).resolve()
DATA = ROOT/'data'

def dump_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def dimensions():
    df=pd.read_csv(DATA/'synthetic_development.csv',dtype={'ID':str})
    p=df.drop_duplicates('ID');n=len(p);nt=int(np.ceil(.3*n))
    d={'EXPECTED_TOTAL_N':n,'EXPECTED_EVENT_N':int(p.CKDstatus.sum()),
       'EXPECTED_DEVELOPMENT_N':n-nt,'EXPECTED_TEST_N':nt}
    file=WORK/'rolling_5y_step6_super_landmark_data'/'super_landmark_development_metadata.csv'
    if file.exists():
        k=len(pd.read_csv(file))
        for key in ['EXPECTED_LONG_N','EXPECTED_LONG_RECORD_N','EXPECTED_DEVELOPMENT_VALID_ORIGIN_N']:
            d[key]=k
    return d

class Overrides(ast.NodeTransformer):
    def __init__(self,values):self.values=values
    def visit_Assign(self,node):
        self.generic_visit(node)
        if len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            key=node.targets[0].id
            if key in self.values:
                node.value=ast.parse(repr(self.values[key]),mode='eval').body
        return node

def load_source(relative, *, overrides=None, execute_main=False, synthetic=True):
    """Load source into a fresh module; alter only paths and named configuration.

    Synthetic sizes are derived from generated data. For private reproduction pass
    synthetic=False and explicit overrides. Never point the synthetic run at clinical data.
    """
    WORK.mkdir(parents=True,exist_ok=True)
    path=ROOT/relative
    text=path.read_text(encoding='utf-8')
    # These filenames are source contracts, not files copied from a cohort.
    text=text.replace('__CKD_WORKDIR__/深圳南宁随访数据表2.csv',(DATA/'synthetic_development.csv').as_posix())
    text=text.replace('__CKD_WORKDIR__',WORK.as_posix())
    # The supplied RSF cell refers to v3, while the retained Cox and comparison use v5.
    text=text.replace('rolling_5y_step7_unpenalized_cox_fixed95_v3','rolling_5y_step7_unpenalized_cox_fixed95_v5')
    text=text.replace('__CKD_LEGACY_WORKDIR__',(ROOT/'results'/'legacy_work').as_posix())
    if '__CKD_SUPPORT_ROOT__' in text or '__CKD_SOURCE_REFERENCE__' in text:
        raise ValueError('Supplementary source needs explicit input-path adaptation; see docs/REPRODUCTION.md.')
    values=dimensions() if synthetic else {}
    values.update(overrides or {})
    tree=Overrides(values).visit(ast.parse(text));ast.fix_missing_locations(tree)
    name='ckd_export_'+path.stem.replace('-','_')
    mod=types.ModuleType(name);mod.__file__=str(path);sys.modules[name]=mod
    if execute_main:mod.__name__='__main__'
    mod.display=lambda obj: print(obj.to_string() if hasattr(obj,'to_string') else obj)
    os.environ['CKD_LSTM_PROJECT_DIR']=str(WORK)
    os.environ.setdefault('MPLBACKEND','Agg')
    exec(compile(tree,str(path),'exec'),mod.__dict__)
    return mod
