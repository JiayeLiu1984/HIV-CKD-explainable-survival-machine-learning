"""Development-defined risk groups and predictive (not causal) attribution."""
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator

from .data import ARTIFACTS, OUTPUTS, config, dump
from .evaluation import verify_lock, observed_risk, evaluation_panel, load_members


def risk_stratification():
    verify_lock()
    cuts = pd.read_csv(OUTPUTS / 'development_cv/risk_tertile_cutpoints.csv')
    assert set(cuts.source_dataset) == {'development'}
    out = OUTPUTS / 'risk_stratification'; out.mkdir(exist_ok=True)
    rows = []; labels = ['lower', 'intermediate', 'higher']
    for dataset in ['internal_test', 'external_validation']:
        predictions = pd.read_csv(OUTPUTS / dataset / 'predictions.csv')
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
        for lm in range(6):
            m = predictions[predictions.landmark == lm].copy()
            cut = cuts[cuts.landmark == lm].iloc[0]
            group = np.searchsorted([cut.lower, cut.upper], m.predicted_5y_risk, side='right')
            for k, label in enumerate(labels):
                part = m.iloc[np.flatnonzero(group == k)]
                rows.append({'dataset': dataset, 'landmark': lm, 'group': label, 'n': len(part),
                             'CKD_events': int(part.event.sum()), 'observed_5y_CKD_risk': observed_risk(part),
                             'cutpoint_source': 'development', 'lower_cutpoint': cut.lower, 'upper_cutpoint': cut.upper})
                if len(part):
                    t, s = kaplan_meier_estimator(part.event.to_numpy(bool), part.time.to_numpy())
                    axes.flat[lm].step(np.r_[0, t]/12, np.r_[0, 1-s], where='post', label=f'{label} (n={len(part)})')
            axes.flat[lm].set(title=f'Landmark {lm} years', xlabel='Years after landmark', ylabel='Observed CKD risk (1 − KM)', xlim=(0, 5), ylim=(0, 1))
            axes.flat[lm].legend(fontsize=7)
        fig.suptitle(f'Synthetic {dataset}; unchanged development cutpoints')
        fig.savefig(out / f'{dataset}_risk_groups.png', dpi=150); plt.close(fig)
    pd.DataFrame(rows).to_csv(out / 'risk_group_summary.csv', index=False)
    cuts.to_csv(out / 'applied_cutpoints.csv', index=False)
    verify_lock()
    dump(out / 'audit.json', {'cutpoints_fitted_on': 'development', 'test_or_external_outcomes_used_for_cutpoints': False})


def interpretation():
    verify_lock()
    members = load_members(evaluation_panel('internal_test'))
    meta = members[0][2]['meta']
    chosen = []
    for lm in range(6):
        chosen.extend(np.flatnonzero(meta.landmark.to_numpy() == lm)[:2].tolist())
    # Include all eligible landmarks for one illustrative patient.
    patient = meta.groupby('ID').size().idxmax()
    chosen = sorted(set(chosen) | set(np.flatnonzero(meta.ID.to_numpy() == patient)))
    inputs = [torch.tensor(batch['x'][chosen]) for _, _, batch in members]
    lengths = torch.tensor(members[0][2]['lengths'][chosen])
    lm = meta.iloc[chosen].landmark.to_numpy()
    references = []
    for (prep, _, _), x in zip(members, inputs):
        r = torch.tensor(prep.reference)[None, None, :].expand_as(x).clone()
        for i, length in enumerate(lengths):
            r[i, int(length):] = 0
        references.append(r)
    offsets = torch.tensor(np.load(ARTIFACTS / 'calibration_offsets.npy')[lm], dtype=torch.float32)[:, None]

    def prediction(values):
        curves = [torch.cumprod(1-torch.sigmoid(model(x, lengths)), dim=1)
                  for (_, model, _), x in zip(members, values)]
        s = torch.stack(curves).mean(0)
        prev = torch.cat([torch.ones((len(s), 1)), s[:, :-1]], dim=1)
        h = (1-s/prev.clamp_min(1e-7)).clamp(1e-6, 1-1e-6)
        return 1-torch.prod(1-torch.sigmoid(torch.logit(h)+offsets), dim=1)

    steps = config()['ig_steps']; grads = [torch.zeros_like(x) for x in inputs]
    for i, alpha in enumerate(torch.linspace(0, 1, steps+1)):
        values = [(r+alpha*(x-r)).detach().requires_grad_(True) for x, r in zip(inputs, references)]
        gradient = torch.autograd.grad(prediction(values).sum(), values)
        for total, g in zip(grads, gradient):
            total += g * (.5 if i in [0, steps] else 1.) / steps
    attributions = [(x-r)*g for x, r, g in zip(inputs, references, grads)]
    with torch.no_grad():
        full = prediction(inputs); reference_risk = prediction(references)
        future = [x.clone() for x in inputs]
        for x in future:
            for i, length in enumerate(lengths):
                x[i, int(length):] += 100
        future_change = float((prediction(future)-full).abs().max())
    residual = full-reference_risk-sum(a.sum((1, 2)) for a in attributions)
    stored = np.load(ARTIFACTS / 'internal_test_survival.npy')
    prediction_error = float(np.max(np.abs(full.numpy() - (1-stored[chosen, -1]))))
    # Fold-specific one-hot representations are mapped back to clinical variables.
    groups = {}
    current = np.zeros(len(chosen)); earlier = np.zeros(len(chosen))
    for (prep, _, _), attribution in zip(members, attributions):
        a = attribution.detach().numpy()
        for j, name in enumerate(prep.feature_groups):
            groups.setdefault(name, np.zeros((len(chosen), 11)))
            groups[name] += a[:, :, j]
    for values in groups.values():
        for i, length in enumerate(lengths):
            current[i] += abs(values[i, int(length)-1])
            earlier[i] += np.abs(values[i, :int(length)-1]).sum()
    out = OUTPUTS / 'model_interpretation'; out.mkdir(exist_ok=True)
    top = pd.DataFrame([{'dataset': 'internal_test', 'predictor': name,
                         'mean_absolute_IG': np.abs(value.sum(axis=1)).mean()} for name, value in groups.items()]).sort_values('mean_absolute_IG', ascending=False)
    top.to_csv(out / 'global_predictors.csv', index=False)
    components = pd.DataFrame({'dataset': 'internal_test', 'landmark': lm, 'current': current, 'earlier': earlier}).groupby(['dataset', 'landmark'], as_index=False)[['current', 'earlier']].mean()
    total = (components.current+components.earlier).replace(0, np.nan)
    components['current_share'] = components.current/total
    components['earlier_share'] = components.earlier/total
    components.to_csv(out / 'current_vs_earlier.csv', index=False)
    pd.DataFrame({'dataset': 'internal_test', 'ID': meta.iloc[chosen].ID.to_numpy(), 'landmark': lm,
                  'predicted_risk': full.numpy(), 'reference_risk': reference_risk.numpy(),
                  'completeness_residual': residual.detach().numpy()}).to_csv(out / 'IG_verification.csv', index=False)
    predictions = pd.read_csv(OUTPUTS / 'internal_test/predictions.csv')
    trajectory = predictions[predictions.ID == patient][['dataset', 'ID', 'landmark', 'predicted_5y_risk']]
    trajectory.to_csv(out / 'patient_updated_risk.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    display = top.head(10).iloc[::-1]
    axes[0].barh(display.predictor, display.mean_absolute_IG); axes[0].set_title('Top predictive attributions')
    axes[1].bar(components.landmark, components.current_share, label='Current')
    axes[1].bar(components.landmark, components.earlier_share, bottom=components.current_share, label='Earlier')
    axes[1].set(xlabel='Landmark (years)', ylabel='Absolute attribution share'); axes[1].legend()
    axes[2].plot(trajectory.landmark, trajectory.predicted_5y_risk, marker='o')
    axes[2].set(xlabel='Landmark (years)', ylabel='Subsequent 5-year predicted risk', title='One synthetic patient')
    fig.suptitle('Synthetic internal-test example: predictive attribution, not causal effects')
    fig.savefig(out / 'interpretation.png', dpi=150); plt.close(fig)
    verify_lock()
    audit = {'dataset': 'internal_test', 'background_source': 'development training patients only',
             'scope': 'Small illustrative sample; not a whole-cohort attribution estimate.',
             'interpretation': 'Predictive attribution, not causal effects.', 'n_examples': len(chosen),
             'max_completeness_residual': float(residual.abs().max()),
             'future_input_max_risk_change': future_change, 'frozen_prediction_max_error': prediction_error}
    dump(out / 'audit.json', audit)
    assert audit['max_completeness_residual'] < 1e-3 and future_change < 1e-6 and prediction_error < 1e-5
