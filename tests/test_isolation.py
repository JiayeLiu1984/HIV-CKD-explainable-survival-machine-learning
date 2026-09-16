"""Tests use the completed synthetic run and never train models."""
import json
import unittest
import joblib
import numpy as np
import pandas as pd
from ckd_example.data import ARTIFACTS, OUTPUTS, read_panel, manifest, digest, ROOT
from ckd_example.evaluation import verify_lock, evaluation_panel, frozen_predict


class IsolationTests(unittest.TestCase):
    def test_patient_split_and_folds(self):
        m = manifest(); self.assertFalse(m.ID.duplicated().any())
        dev = set(m.query("dataset == 'development'").ID)
        test = set(m.query("dataset == 'internal_test'").ID)
        external = set(read_panel('external_cohort').ID)
        self.assertFalse(dev & test or dev & external or test & external)
        self.assertEqual(len(test), int(np.ceil(len(m)*.3)))
        self.assertEqual(set(m.query("dataset == 'development'").fold), set(range(5)))
        self.assertEqual(set(m.query("dataset == 'internal_test'").fold), {-1})

    def test_training_and_calibration_use_development_only(self):
        lock = verify_lock(); m = manifest()
        dev = set(m.query("dataset == 'development'").ID)
        audit = json.loads((ARTIFACTS/'training_audit.json').read_text())
        for row in audit:
            train, valid = set(row['training_ids']), set(row['validation_ids'])
            self.assertFalse(train & valid); self.assertEqual(train | valid, dev)
            prep = joblib.load(ARTIFACTS/f"fold_{row['fold']}/preprocessing.joblib")
            self.assertEqual(set(prep.fit_ids), train)
            self.assertEqual(set(row['preprocessing_fit_ids']), train)
        self.assertEqual(set(lock['calibration_fit_ids']), dev)
        self.assertFalse(lock['internal_test_used_for_selection'] or lock['external_used_for_selection'])
        self.assertFalse(lock['internal_test_used_for_calibration'] or lock['external_used_for_calibration'])
        self.assertEqual(lock['selected_model'], 'LSTM')

    def test_frozen_hashes_and_evaluation_labels(self):
        lock = verify_lock()
        for dataset in ['internal_test', 'external_validation']:
            audit = json.loads((OUTPUTS/dataset/'evaluation_audit.json').read_text())
            self.assertEqual(audit['artifact_hashes_before'], lock['artifact_hashes'])
            self.assertEqual(audit['artifact_hashes_after'], lock['artifact_hashes'])
            for flag in ['training', 'tuning', 'model_selection', 'calibration_fitting']:
                self.assertFalse(audit[flag])
            for name in ['predictions', 'landmark_performance', 'six_landmark_mean', 'calibration', 'decision_curve']:
                self.assertEqual(set(pd.read_csv(OUTPUTS/dataset/f'{name}.csv').dataset), {dataset})

    def test_cutpoints_are_development_derived(self):
        lock = verify_lock()
        self.assertEqual(set(lock['risk_cutpoint_fit_ids']), set(lock['development_ids']))
        cuts = pd.read_csv(OUTPUTS/'development_cv/risk_tertile_cutpoints.csv')
        meta = pd.read_csv(OUTPUTS/'development_cv/prediction_metadata.csv')
        s = np.load(ARTIFACTS/'development_calibrated_predictions.npy')
        for lm in range(6):
            expected = np.quantile(1-s[meta.landmark.to_numpy() == lm, -1], [1/3, 2/3])
            row = cuts[cuts.landmark == lm].iloc[0]
            np.testing.assert_allclose([row.lower, row.upper], expected)
        applied = pd.read_csv(OUTPUTS/'risk_stratification/applied_cutpoints.csv')
        pd.testing.assert_frame_equal(cuts, applied)
        groups = pd.read_csv(OUTPUTS/'risk_stratification/risk_group_summary.csv')
        for _, row in groups.iterrows():
            cut = cuts[cuts.landmark == row.landmark].iloc[0]
            np.testing.assert_allclose([row.lower_cutpoint, row.upper_cutpoint], [cut.lower, cut.upper])

    def test_outcomes_cannot_change_frozen_predictions(self):
        for dataset in ['internal_test', 'external_validation']:
            panel = evaluation_panel(dataset); panel['CKDstatus'] = 1-panel.CKDstatus
            meta, changed = frozen_predict(panel)
            original_meta = pd.read_csv(OUTPUTS/dataset/'predictions.csv')
            original = np.load(ARTIFACTS/f'{dataset}_survival.npy')
            pairs = original_meta[['ID', 'landmark']].assign(a=np.arange(len(original_meta))).merge(
                meta[['ID', 'landmark']].assign(b=np.arange(len(meta))), on=['ID', 'landmark'])
            self.assertGreater(len(pairs), 0)
            np.testing.assert_allclose(original[pairs.a], changed[pairs.b], atol=1e-6)
        verify_lock()

    def test_probabilities_and_interpretation(self):
        for dataset in ['internal_test', 'external_validation']:
            s = np.load(ARTIFACTS/f'{dataset}_survival.npy')
            self.assertTrue(np.isfinite(s).all() and ((s >= 0) & (s <= 1)).all())
            self.assertTrue((np.diff(s, axis=1) <= 1e-6).all())
            table = pd.read_csv(OUTPUTS/dataset/'landmark_performance.csv')
            self.assertEqual(len(table), 6)
            self.assertTrue(np.isfinite(table[['Uno_C_index', 'iAUC', 'IBS']]).all().all())
        audit = json.loads((OUTPUTS/'model_interpretation/audit.json').read_text())
        self.assertLess(audit['max_completeness_residual'], 1e-3)
        self.assertLess(audit['future_input_max_risk_change'], 1e-6)
        self.assertLess(audit['frozen_prediction_max_error'], 1e-5)


if __name__ == '__main__':
    unittest.main()
