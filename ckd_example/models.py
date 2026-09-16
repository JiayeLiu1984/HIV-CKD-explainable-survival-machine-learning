"""Compact illustrative model families, not the manuscript's exact neural architecture."""
import json
import random
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from lifelines import CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

from .data import (ARTIFACTS, OUTPUTS, TIMES, Preprocessor, config,
                   development_panel, manifest, samples, summaries, dump)

MODELS = ['Cox', 'RSF', 'RNN', 'LSTM']


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(2)


class SequenceModel(nn.Module):
    def __init__(self, features, hidden, family):
        super().__init__()
        self.recurrent = (nn.LSTM if family == 'LSTM' else nn.RNN)(features, hidden, batch_first=True)
        self.head = nn.Linear(hidden + 1, 10)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, state = self.recurrent(packed)
        hidden = state[0] if isinstance(state, tuple) else state
        landmark = ((lengths.float() - 1) / 10)[:, None]
        return self.head(torch.cat([hidden[-1], landmark], dim=1))


def fit_neural(batch, family, c, seed):
    seed_all(seed)
    model = SequenceModel(batch['x'].shape[-1], c['hidden_size'], family)
    optimizer = torch.optim.Adam(model.parameters(), lr=c['learning_rate'])
    dataset = torch.utils.data.TensorDataset(torch.tensor(batch['x']), torch.tensor(batch['lengths']),
                                             torch.tensor(batch['y']), torch.tensor(batch['mask']))
    loader = torch.utils.data.DataLoader(dataset, batch_size=c['batch_size'], shuffle=True)
    for _ in range(c['epochs']):
        model.train()
        for x, lengths, y, mask in loader:
            optimizer.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(model(x, lengths), y, reduction='none')
            # Equal weighting of represented prediction landmarks within a minibatch.
            terms = [((loss[lengths == length] * mask[lengths == length]).sum() /
                      mask[lengths == length].sum().clamp_min(1)) for length in lengths.unique()]
            torch.stack(terms).mean().backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
    return model.eval()


def neural_survival(model, batch):
    with torch.no_grad():
        chunks = []
        for start in range(0, len(batch['x']), 256):
            logits = model(torch.tensor(batch['x'][start:start+256]), torch.tensor(batch['lengths'][start:start+256]))
            chunks.append(torch.cumprod(1 - torch.sigmoid(logits), dim=1).numpy())
    return np.concatenate(chunks)


def baseline_survival(train, valid, family, c):
    x, z = summaries(train), summaries(valid)
    y = Surv.from_arrays(train['meta'].event.astype(bool), train['meta'].time)
    if family == 'RSF':
        model = RandomSurvivalForest(n_estimators=c['rsf_trees'], min_samples_leaf=c['rsf_min_leaf'],
                                     max_features='sqrt', n_jobs=2, random_state=c['seed']).fit(x, y)
        return np.stack([fn(TIMES) for fn in model.predict_survival_function(z)])
    # Landmark-stratified Cox with fixed regularization; constant columns excluded on training only.
    keep = np.std(x[:, :-1], axis=0) > 1e-8
    columns = [f'x{i}' for i in np.flatnonzero(keep)]
    frame = pd.DataFrame(x[:, :-1][:, keep], columns=columns)
    frame['landmark'] = train['meta'].landmark.to_numpy()
    frame['time'] = train['meta'].time.to_numpy()
    frame['event'] = train['meta'].event.to_numpy()
    model = CoxPHFitter(penalizer=c['cox_penalizer']).fit(frame, duration_col='time', event_col='event', strata=['landmark'])
    other = pd.DataFrame(z[:, :-1][:, keep], columns=columns)
    other['landmark'] = valid['meta'].landmark.to_numpy()
    prediction = model.predict_survival_function(other, times=TIMES)
    return prediction.reindex(columns=other.index).to_numpy().T


def development_cv():
    c = config(); panel = development_panel(); allocation = manifest()
    dev_ids = set(panel.ID); test_ids = set(allocation.query("dataset == 'internal_test'").ID)
    assert not dev_ids & test_ids
    pieces = {name: [] for name in MODELS}; meta_parts = []; audit = []
    for fold in range(5):
        train_ids = set(allocation.query("dataset == 'development' and fold != @fold").ID)
        valid_ids = set(allocation.query("dataset == 'development' and fold == @fold").ID)
        assert not train_ids & valid_ids and not (train_ids | valid_ids) & test_ids
        train_panel = panel[panel.ID.isin(train_ids)]
        prep = Preprocessor().fit(train_panel)
        train = samples(train_panel, prep); valid = samples(panel[panel.ID.isin(valid_ids)], prep)
        meta = valid['meta'].copy(); meta['fold'] = fold; meta_parts.append(meta)
        folder = ARTIFACTS / f'fold_{fold}'; folder.mkdir(exist_ok=True)
        joblib.dump(prep, folder / 'preprocessing.joblib')
        for name in MODELS:
            print(f'Development fold {fold+1}/5: {name}', flush=True)
            if name in ['Cox', 'RSF']:
                survival = baseline_survival(train, valid, name, c)
            else:
                model = fit_neural(train, name, c, c['seed'] + fold)
                survival = neural_survival(model, valid)
                if name == 'LSTM':
                    torch.save({'synthetic_only': True, 'features': train['x'].shape[-1],
                                'hidden': c['hidden_size'], 'state': model.state_dict()}, folder / 'lstm.pt')
            assert survival.shape == (len(meta), 10) and np.isfinite(survival).all()
            pieces[name].append(survival)
        audit.append({'fold': fold, 'training_ids': sorted(train_ids), 'validation_ids': sorted(valid_ids),
                      'preprocessing_fit_ids': prep.fit_ids, 'hyperparameters': c,
                      'test_or_external_used': False})
    out = OUTPUTS / 'development_cv'; out.mkdir(parents=True, exist_ok=True)
    pd.concat(meta_parts, ignore_index=True).assign(dataset='development').to_csv(out / 'prediction_metadata.csv', index=False)
    np.savez(ARTIFACTS / 'development_predictions.npz', **{k: np.concatenate(v) for k, v in pieces.items()})
    dump(ARTIFACTS / 'training_audit.json', audit)
