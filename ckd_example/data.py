"""Patient partitions, development-fitted transforms and causal history construction."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .synthetic import LABS, ART, STATUS, MEDS, generate

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
ARTIFACTS = ROOT / 'artifacts'
OUTPUTS = ROOT / 'outputs'
CONT = ['Age', 'BMI'] + LABS + [a + '_cum_month' for a in ART]
BIN = ['Oppinfection'] + STATUS + MEDS + ['current_' + a for a in ART]
CAT = ['Sex', 'Marriage', 'Course', 'WHOstage']
FEATURES = CONT + BIN + CAT
TIMES = np.arange(6., 61., 6.)


def config():
    return json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_panel(name):
    return pd.read_csv(DATA / f'{name}.csv', dtype={'ID': str, 'WHOstage': str})


def generate_data():
    c = config()
    if c['source_n'] < 300 or c['external_n'] < 100:
        raise ValueError('Use >=300 source and >=100 external patients for this five-fold example.')
    DATA.mkdir(exist_ok=True)
    for name, n, offset, external in [('source_cohort', c['source_n'], 0, False),
                                      ('external_cohort', c['external_n'], 1, True)]:
        generate(n, c['seed'] + offset, external).to_csv(DATA / f'{name}.csv', index=False)
    pd.DataFrame([{'column': k, 'role': 'predictor', 'kind': 'continuous' if k in CONT else 'binary' if k in BIN else 'categorical'} for k in FEATURES]
                 + [{'column': k, 'role': 'identifier/time/outcome; excluded from predictors', 'kind': 'metadata'} for k in ['ID', 'data', 'month', 'time_bin', 'measurement_month', 'interval', 'CKDstatus', 'synthetic']]).to_csv(DATA / 'data_dictionary.csv', index=False)
    dump(DATA / 'provenance.json', {'synthetic_only': True, 'seed': c['seed'],
         'origin': 'Independent random draws; no clinical files, learned generator or patient data.',
         'warning': 'Artificial event enrichment supports small examples; performance is not a clinical result.',
         'files': {f.name: digest(f) for f in DATA.glob('*.csv')}})


def split_data():
    c = config()
    patients = read_panel('source_cohort').drop_duplicates('ID').sort_values('ID')
    strata = patients.data.astype(str) + '_' + patients.CKDstatus.astype(str)
    dev, test = train_test_split(patients.ID.to_numpy(), test_size=.30,
                                 random_state=c['seed'], stratify=strata)
    manifest = patients[['ID', 'data']].copy()
    manifest['dataset'] = np.where(manifest.ID.isin(dev), 'development', 'internal_test')
    manifest['fold'] = -1
    development = patients[patients.ID.isin(dev)]
    strata = development.data.astype(str) + '_' + development.CKDstatus.astype(str)
    for fold, (_, held) in enumerate(StratifiedKFold(5, shuffle=True, random_state=c['seed']).split(development, strata)):
        manifest.loc[manifest.ID.isin(development.iloc[held].ID), 'fold'] = fold
    assert not set(dev) & set(test)
    ARTIFACTS.mkdir(exist_ok=True)
    manifest.to_csv(ARTIFACTS / 'split_manifest.csv', index=False)
    dump(ARTIFACTS / 'split_manifest.json', {'unit': 'patient', 'seed': c['seed'],
         'development_ids': sorted(dev.tolist()), 'internal_test_ids': sorted(test.tolist()),
         'n_development': len(dev), 'n_internal_test': len(test),
         'internal_test_transformed_during_development': False})


def manifest():
    return pd.read_csv(ARTIFACTS / 'split_manifest.csv')


def development_panel():
    ids = manifest().query("dataset == 'development'").ID
    d = read_panel('source_cohort')
    return d[d.ID.isin(ids)].copy()


class Preprocessor:
    """Every estimated value comes from this fold's development training patients."""
    def fit(self, panel):
        self.fit_ids = sorted(panel.ID.unique().tolist())
        self.medians = panel[CONT].median().fillna(0.)
        self.scaler = StandardScaler().fit(panel[CONT].fillna(self.medians))
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(panel[CAT].astype(str))
        self.feature_names = CONT + BIN + self.encoder.get_feature_names_out(CAT).tolist()
        self.feature_groups = CONT + BIN + [name for name, cats in zip(CAT, self.encoder.categories_) for _ in cats]
        self.reference = self.transform(panel).mean(axis=0).astype(np.float32)
        return self

    def transform(self, panel):
        return np.column_stack([self.scaler.transform(panel[CONT].fillna(self.medians)),
                                panel[BIN].fillna(0.).to_numpy(),
                                self.encoder.transform(panel[CAT].fillna('__MISSING__').astype(str))]).astype(np.float32)


def samples(panel, prep):
    """At each landmark, truncate inputs first; carry history forward, never backward."""
    histories, rows, event_targets, masks, lengths = [], [], [], [], []
    for ident, group in panel.groupby('ID', sort=True):
        g = group.sort_values('month')
        first = g.iloc[0]
        duration, event = float(first.interval), int(first.CKDstatus)
        for lm in range(6):
            remaining = duration - 12 * lm
            if remaining <= 0 or (not event and remaining < 6):
                continue
            length = 2 * lm + 1
            # A generated observation is already within the preceding three-month window.
            observed = g[g.month <= 12 * lm].set_index('time_bin')
            aligned = observed.reindex(range(length))[FEATURES].ffill()
            assert not aligned[CAT].isna().any().any(), 'Synthetic baseline must be observed.'
            x = np.zeros((11, len(prep.feature_names)), np.float32)
            x[:length] = prep.transform(aligned)
            target = ((TIMES - 6 < remaining) & (remaining <= TIMES) & bool(event)).astype(np.float32)
            mask = ((remaining > TIMES - 6) if event else (remaining >= TIMES)).astype(np.float32)
            histories.append(x); event_targets.append(target); masks.append(mask); lengths.append(length)
            rows.append({'ID': ident, 'landmark': lm, 'time': min(remaining, 60.), 'event': int(event and remaining <= 60.)})
    return {'x': np.stack(histories), 'lengths': np.asarray(lengths), 'meta': pd.DataFrame(rows),
            'y': np.stack(event_targets), 'mask': np.stack(masks)}


def summaries(batch):
    x, lengths = batch['x'], batch['lengths']
    last = x[np.arange(len(x)), lengths - 1]
    mean = x.sum(axis=1) / lengths[:, None]
    slope = (last - x[:, 0]) / np.maximum((lengths - 1) / 2, 1)[:, None]
    return np.column_stack([last, mean, slope, batch['meta'].landmark.to_numpy()]).astype(np.float64)
