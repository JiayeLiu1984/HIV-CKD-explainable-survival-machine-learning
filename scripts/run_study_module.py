"""Execute a retained study source with explicit local input/output configuration.

The configuration is local to the operator. No clinical dataset accompanies this release.
"""
import argparse, ast, json, os, sys, types
from pathlib import Path
from _runtime import ROOT, Overrides

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,help='Path relative to the release root.');p.add_argument('--config',required=True);a=p.parse_args()
    source=(ROOT/a.source).resolve()
    if ROOT not in source.parents or source.suffix!='.py':p.error('Source must be a Python module within this release.')
    cfg=json.loads(Path(a.config).read_text(encoding='utf-8-sig'));work=Path(cfg['workdir']).resolve();work.mkdir(parents=True,exist_ok=True)
    text=source.read_text(encoding='utf-8-sig')
    aliases={'__CKD_WORKDIR__':work.as_posix(),**cfg.get('path_aliases',{})}
    for old,new in sorted(aliases.items(),key=lambda x:-len(x[0])):text=text.replace(old,str(new).replace('\\','/'))
    if '__CKD_' in text:raise ValueError('Provide every remaining path placeholder in path_aliases. No implicit search for clinical files is performed.')
    overrides=cfg.get('overrides',{});tree=Overrides(overrides).visit(ast.parse(text));ast.fix_missing_locations(tree)
    os.environ['CKD_LSTM_PROJECT_DIR']=str(work)
    os.environ.update({str(k):str(v) for k,v in cfg.get('environment',{}).items()})
    module=types.ModuleType('__main__');module.__file__=str(source);module.display=print
    sys.modules['__main__']=module
    exec(compile(tree,str(source),'exec'),module.__dict__)

if __name__=='__main__':main()
