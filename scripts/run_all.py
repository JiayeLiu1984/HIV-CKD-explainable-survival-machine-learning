"""One-command synthetic demonstration; optional analyses are opt-in."""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n', type=int, default=600)
    p.add_argument('--external-n', type=int, default=240)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--bootstrap', type=int, default=20)
    p.add_argument('--with-sensitivity', action='store_true')
    p.add_argument('--three-seeds', action='store_true', help='Also run sensitivity and three LSTM training seeds')
    p.add_argument('--with-recalibration', action='store_true')
    a = p.parse_args()
    root = Path(__file__).resolve().parent
    env = os.environ.copy()
    env.update(MPLBACKEND='Agg', PYTHONIOENCODING='utf-8')
    env.setdefault('CKD_WORKDIR', str(root.parent / 'results' / ('synthetic_' + datetime.now().strftime('%Y%m%d_%H%M%S'))))
    work = Path(env['CKD_WORKDIR']).resolve()
    env['CKD_WORKDIR'] = str(work)
    logs = work / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    commands = [
        ['module_00_generate_synthetic_data.py', '--n', str(a.n), '--external-n', str(a.external_n)],
        ['module_01_data_preprocessing.py'],
        ['module_02_model_training.py', '--epochs', str(a.epochs)],
        ['module_03_model_evaluation.py', '--bootstrap', str(a.bootstrap)],
        ['module_03b_internal_validation.py', '--bootstrap', str(a.bootstrap)],
        ['module_04_external_validation.py'] + (['--with-recalibration'] if a.with_recalibration else []),
        ['module_05_risk_groups_and_figures.py'],
        ['module_06_interpretation.py'],
    ]
    if a.with_sensitivity or a.three_seeds:
        commands.append(['module_07_sensitivity.py', '--epochs', str(a.epochs)] + (['--three-seeds'] if a.three_seeds else []))
    jobs = [(c[0], [sys.executable, '-X', 'utf8', str(root / c[0]), *c[1:]]) for c in commands]
    jobs.append(('integrity_tests', [sys.executable, '-m', 'unittest', 'discover', '-s', str(root.parent / 'tests'), '-v']))
    report = {'synthetic_only': True, 'settings': vars(a), 'stages': [], 'status': 'running'}
    started = time.monotonic()
    print(f'Synthetic demonstration: {work}', flush=True)
    for i, (name, command) in enumerate(jobs, 1):
        print(f'[{i}/{len(jobs)}] {name}', flush=True)
        tick = time.monotonic()
        log = logs / (name + '.log')
        with log.open('w', encoding='utf-8') as stream:
            result = subprocess.run(command, env=env, cwd=root.parent, stdout=stream, stderr=subprocess.STDOUT)
        report['stages'].append({'name': name, 'seconds': round(time.monotonic()-tick, 2), 'returncode': result.returncode, 'log': str(log.relative_to(work))})
        report['status'] = 'failed' if result.returncode else 'running'
        report['elapsed_seconds'] = round(time.monotonic()-started, 2)
        (work / 'RUN_SUMMARY.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        if result.returncode:
            print(f'Failed; see {log}', flush=True)
            raise SystemExit(result.returncode)
    report['status'] = 'passed'
    (work / 'RUN_SUMMARY.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Passed in {report["elapsed_seconds"]} seconds. Outputs and logs: {work}', flush=True)


if __name__ == '__main__':
    main()
