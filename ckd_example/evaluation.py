"""Development-only calibration and frozen evaluation on two independent cohorts."""
import json
import joblib
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc, integrated_brier_score
from sksurv.nonparametric import kaplan_meier_estimator, CensoringDistributionEstimator
from sksurv.util import Surv

from .data import ROOT, ARTIFACTS, OUTPUTS, TIMES, config, dump, digest, samples, read_panel, manifest
from .models import MODELS, SequenceModel, neural_survival

METRIC_TIMES = np.minimum(TIMES, 59.999)


def hazards(survival):
    return np.clip(1 - survival / np.maximum(np.column_stack([np.ones(len(survival)), survival[:, :-1]]), 1e-7), 1e-6, 1-1e-6)


def fit_calibration(survival, meta):
    """One intercept per landmark, shared across the ten conditional hazard intervals."""
    h = hazards(survival); offsets = []
    for lm in range(6):
        use = meta.landmark.to_numpy() == lm
        time = meta.loc[use, 'time'].to_numpy()[:, None]
        event = meta.loc[use, 'event'].to_numpy(bool)[:, None]
        y = event & (time > TIMES - 6) & (time <= TIMES)
        mask = np.where(event, time > TIMES - 6, time >= TIMES)
        z = logit(h[use])[mask]; target = y[mask].astype(float)
        if len(z) == 0:
            raise ValueError(f'No development calibration observations for landmark {lm}.')
        result = minimize_scalar(lambda a: np.mean(np.logaddexp(0, z+a) - target*(z+a)),
                                 bounds=(-8, 8), method='bounded')
        offsets.append(float(result.x))
    return np.asarray(offsets)


def apply_calibration(survival, meta, offsets):
    h = expit(logit(hazards(survival)) + offsets[meta.landmark.to_numpy()][:, None])
    return np.cumprod(1-h, axis=1)


def observed_risk(meta):
    if not len(meta):
        return float('nan')
    time, survival = kaplan_meier_estimator(meta.event.to_numpy(bool), meta.time.to_numpy())
    index = np.searchsorted(time, 59.999, side='right') - 1
    return 0. if index < 0 else float(1 - survival[index])


def performance(meta, survival, dataset):
    rows = []
    for lm in range(6):
        use = meta.landmark.to_numpy() == lm; m = meta[use]
        y = Surv.from_arrays(m.event.astype(bool), m.time)
        row = {'dataset': dataset, 'landmark': lm, 'n': len(m), 'events': int(m.event.sum())}
        supported = (METRIC_TIMES >= y['time'].min()) & (METRIC_TIMES < y['time'].max())
        grid = METRIC_TIMES[supported]; s = survival[use][:, supported]
        row.update(Uno_C_index=np.nan, iAUC=np.nan, IBS=np.nan, metric_status='ok')
        try:
            row['Uno_C_index'] = float(concordance_index_ipcw(y, y, 1-survival[use, -1], tau=59.999)[0])
            if len(grid) < 2:
                raise ValueError('Insufficient supported evaluation times.')
            row['iAUC'] = float(cumulative_dynamic_auc(y, y, 1-s, grid)[1])
            row['IBS'] = float(integrated_brier_score(y, y, s, grid))
        except ValueError as exc:
            row['metric_status'] = str(exc)
        rows.append(row)
    return pd.DataFrame(rows)


def calibration_table(meta, survival, dataset):
    rows = []
    for lm in range(6):
        use = meta.landmark.to_numpy() == lm; m = meta[use]; risk = 1-survival[use, -1]
        for bin_id, idx in enumerate(np.array_split(np.argsort(risk), 5)):
            rows.append({'dataset': dataset, 'landmark': lm, 'bin': bin_id+1, 'n': len(idx),
                         'predicted_5y_risk': float(risk[idx].mean()), 'observed_5y_risk': observed_risk(m.iloc[idx])})
    return pd.DataFrame(rows)


def decision_table(meta, survival, dataset):
    rows = []
    for lm in range(6):
        use = meta.landmark.to_numpy() == lm; m = meta[use]; risk = 1-survival[use, -1]
        y = Surv.from_arrays(m.event.astype(bool), m.time)
        censor = CensoringDistributionEstimator().fit(y)
        cases = y['event'] & (y['time'] <= 59.999); controls = y['time'] > 59.999
        weights = np.zeros(len(m))
        weights[cases] = 1 / np.maximum(censor.predict_proba(y['time'][cases]), 1e-6)
        weights[controls] = 1 / max(float(censor.predict_proba([59.999])[0]), 1e-6)
        for threshold in np.linspace(.01, .5, 50):
            positive = risk >= threshold; odds = threshold/(1-threshold)
            rows.append({'dataset': dataset, 'landmark': lm, 'threshold': threshold,
                         'net_benefit': (weights[cases & positive].sum()-odds*weights[controls & positive].sum())/len(m),
                         'treat_all': (weights[cases].sum()-odds*weights[controls].sum())/len(m), 'treat_none': 0.})
    return pd.DataFrame(rows)


def compare_and_lock():
    out = OUTPUTS / 'development_cv'
    meta = pd.read_csv(out / 'prediction_metadata.csv')
    raw = np.load(ARTIFACTS / 'development_predictions.npz')
    comparisons = []; tables = []; calibration = []
    for model in MODELS:
        crossfit = np.empty_like(raw[model])
        for fold in range(5):
            train = meta.fold.to_numpy() != fold; held = ~train
            offsets = fit_calibration(raw[model][train], meta[train])
            crossfit[held] = apply_calibration(raw[model][held], meta[held], offsets)
        table = performance(meta, crossfit, 'development'); table['model'] = model; tables.append(table)
        cal = calibration_table(meta, crossfit, 'development'); cal['model'] = model; calibration.append(cal)
        comparisons.append({'dataset': 'development', 'model': model, 'evaluation': 'five_fold_OOF_crossfit_calibration',
                            **{k: table[k].mean() if table[k].notna().all() else np.nan for k in ['Uno_C_index', 'iAUC', 'IBS']}})
    pd.DataFrame(comparisons).to_csv(out / 'four_model_comparison.csv', index=False)
    pd.concat(tables).to_csv(out / 'landmark_performance.csv', index=False)
    pd.concat(calibration).to_csv(out / 'calibration.csv', index=False)
    offsets = fit_calibration(raw['LSTM'], meta)
    np.save(ARTIFACTS / 'calibration_offsets.npy', offsets)
    final_dev = apply_calibration(raw['LSTM'], meta, offsets)
    np.save(ARTIFACTS / 'development_calibrated_predictions.npy', final_dev)
    cuts = []
    for lm in range(6):
        risk = 1-final_dev[meta.landmark.to_numpy() == lm, -1]
        low, high = np.quantile(risk, [1/3, 2/3])
        cuts.append({'source_dataset': 'development', 'landmark': lm, 'n': len(risk), 'lower': low, 'upper': high})
    pd.DataFrame(cuts).to_csv(out / 'risk_tertile_cutpoints.csv', index=False)
    files = [ROOT / 'config.json', ARTIFACTS / 'calibration_offsets.npy', ARTIFACTS / 'split_manifest.csv',
             ARTIFACTS / 'training_audit.json', out / 'risk_tertile_cutpoints.csv']
    for fold in range(5):
        files += [ARTIFACTS / f'fold_{fold}' / name for name in ['lstm.pt', 'preprocessing.joblib']]
    dev_ids = sorted(meta.ID.unique().tolist())
    dump(ARTIFACTS / 'model_lock.json', {
        'synthetic_only': True, 'selected_model': 'LSTM',
        'internal_test_used_for_selection': False, 'external_used_for_selection': False,
        'internal_test_used_for_calibration': False, 'external_used_for_calibration': False,
        'selection_rule': 'LSTM selected by study design; synthetic rankings need not favor LSTM.',
        'development_ids': dev_ids, 'calibration_fit_ids': dev_ids, 'risk_cutpoint_fit_ids': dev_ids,
        'predictor': 'Mean survival of five development-fold LSTMs, then frozen landmark hazard-intercept calibration.',
        'configuration': config(), 'artifact_hashes': {str(p.relative_to(ROOT)): digest(p) for p in files}})


def verify_lock():
    lock = json.loads((ARTIFACTS / 'model_lock.json').read_text(encoding='utf-8'))
    for name, expected in lock['artifact_hashes'].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Frozen artifact changed: {name}')
    return lock


def load_members(panel):
    torch.set_num_threads(2)
    members = []
    for fold in range(5):
        folder = ARTIFACTS / f'fold_{fold}'
        prep = joblib.load(folder / 'preprocessing.joblib')
        checkpoint = torch.load(folder / 'lstm.pt', map_location='cpu', weights_only=True)
        assert checkpoint['synthetic_only']
        model = SequenceModel(checkpoint['features'], checkpoint['hidden'], 'LSTM')
        model.load_state_dict(checkpoint['state']); model.eval()
        members.append((prep, model, samples(panel, prep)))
    return members


def frozen_predict(panel):
    verify_lock()
    members = load_members(panel)
    meta = members[0][2]['meta']
    raw = np.mean([neural_survival(model, batch) for _, model, batch in members], axis=0)
    calibrated = apply_calibration(raw, meta, np.load(ARTIFACTS / 'calibration_offsets.npy'))
    return meta, calibrated


def evaluation_panel(dataset):
    if dataset == 'external_validation':
        return read_panel('external_cohort')
    if dataset != 'internal_test':
        raise ValueError(dataset)
    ids = manifest().query("dataset == 'internal_test'").ID
    panel = read_panel('source_cohort')
    return panel[panel.ID.isin(ids)].copy()


def evaluate(dataset):
    before = verify_lock()
    panel = evaluation_panel(dataset)
    assert not set(panel.ID) & set(before['development_ids'])
    meta, survival = frozen_predict(panel)
    out = OUTPUTS / dataset; out.mkdir(parents=True, exist_ok=True)
    table = performance(meta, survival, dataset)
    table.to_csv(out / 'landmark_performance.csv', index=False)
    pd.DataFrame([{'dataset': dataset, 'model': 'LSTM', **{k: table[k].mean() if table[k].notna().all() else np.nan for k in ['Uno_C_index', 'iAUC', 'IBS']}}]).to_csv(out / 'six_landmark_mean.csv', index=False)
    calibration_table(meta, survival, dataset).to_csv(out / 'calibration.csv', index=False)
    decision_table(meta, survival, dataset).to_csv(out / 'decision_curve.csv', index=False)
    meta = meta.assign(dataset=dataset)
    for year, index in [(1, 1), (3, 5), (5, 9)]:
        meta[f'predicted_{year}y_risk'] = 1-survival[:, index]
    meta.to_csv(out / 'predictions.csv', index=False)
    np.save(ARTIFACTS / f'{dataset}_survival.npy', survival)
    after = verify_lock()
    assert before['artifact_hashes'] == after['artifact_hashes']
    dump(out / 'evaluation_audit.json', {'dataset': dataset, 'n_patients': int(meta.ID.nunique()),
         'selected_model': 'LSTM', 'training': False, 'tuning': False, 'model_selection': False,
         'calibration_fitting': False, 'frozen_hashes_unchanged': True,
         'artifact_hashes_before': before['artifact_hashes'], 'artifact_hashes_after': after['artifact_hashes'],
         'censoring_note': 'Outcome-dependent censoring estimates are evaluation statistics, not predictor fitting.'})
