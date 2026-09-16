"""Run the eight main stages, then check isolation. No supplementary analyses."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parent
STAGES = [
    '01_generate_synthetic_data.py',
    '02_split_and_prepare_data.py',
    '03_development_cv_models.py',
    '04_model_comparison_and_lock.py',
    '05_internal_test_evaluation.py',
    '06_external_validation.py',
    '07_risk_stratification.py',
    '08_model_interpretation.py',
]


def main():
    # Preserve prior generated runs instead of mixing old and new artifacts.
    previous = [ROOT/name for name in ['data', 'artifacts', 'outputs'] if (ROOT/name).exists()]
    if previous:
        archive = ROOT/'.run_history'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        archive.mkdir(parents=True)
        for folder in previous:
            if folder.is_symlink() or folder.resolve().parent != ROOT.resolve():
                raise ValueError(f'Refusing to move a generated directory outside this repository: {folder}')
            folder.rename(archive/folder.name)
        print(f'Previous generated files preserved at {archive}', flush=True)
    logs = ROOT/'outputs/logs'; logs.mkdir(parents=True)
    env = os.environ.copy(); env.update(PYTHONIOENCODING='utf-8', MPLBACKEND='Agg')
    report = {'synthetic_only': True, 'configuration': json.loads((ROOT/'config.json').read_text()), 'stages': [], 'status': 'running'}
    jobs = [(name, [sys.executable, '-X', 'utf8', str(ROOT/name)]) for name in STAGES]
    jobs += [('isolation_checks', [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'])]
    start = time.monotonic()
    for index, (name, command) in enumerate(jobs, 1):
        print(f'[{index}/{len(jobs)}] {name}', flush=True)
        tick = time.monotonic(); log = logs/f'{name}.log'
        with log.open('w', encoding='utf-8') as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        report['stages'].append({'name': name, 'seconds': round(time.monotonic()-tick, 2), 'returncode': result.returncode})
        report['elapsed_seconds'] = round(time.monotonic()-start, 2)
        report['status'] = 'failed' if result.returncode else 'running'
        (ROOT/'outputs/run_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        if result.returncode:
            print(log.read_text(encoding='utf-8')[-6000:])
            raise SystemExit(f'Failed: {name}; full log: {log}')
    report['status'] = 'passed'
    (ROOT/'outputs/run_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'All stages and isolation checks passed in {report["elapsed_seconds"]} seconds. See outputs/.')


if __name__ == '__main__':
    main()
