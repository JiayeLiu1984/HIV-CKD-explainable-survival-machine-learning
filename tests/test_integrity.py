from pathlib import Path
import hashlib, importlib.util, json, sys, unittest
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from _runtime import WORK
from module_00_generate_synthetic_data import generate

class IntegrityTests(unittest.TestCase):
    def test_generated_data_provenance_and_determinism(self):
        provenance=json.loads((ROOT/'data/SYNTHETIC_DATA_PROVENANCE.json').read_text())
        self.assertTrue(provenance['synthetic_only'])
        for name,digest in provenance['files'].items():self.assertEqual(hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest(),digest,name)
        pd.testing.assert_frame_equal(generate(12,88),generate(12,88))
    def test_disjoint_patients_and_no_post_event_inputs(self):
        dev=pd.read_csv(ROOT/'data/synthetic_development.csv');ext=pd.read_csv(ROOT/'data/synthetic_external.csv')
        self.assertFalse(set(dev.ID)&set(ext.ID))
        for d in [dev,ext]:
            self.assertTrue(d.ID.str.startswith('SYN_').all());self.assertFalse(d.duplicated(['ID','time_bin']).any())
            self.assertTrue((d.month<d.interval).all());self.assertTrue((d.month==6*d.time_bin).all())
            self.assertTrue((d.groupby('ID').CKDstatus.nunique()==1).all())
    def test_patient_fold_integrity_and_excluded_outcomes(self):
        m=pd.read_csv(WORK/'rolling_5y_step6_super_landmark_data/super_landmark_development_metadata.csv')
        self.assertTrue((m.groupby('ID').fold_id.nunique()==1).all())
        test=pd.read_csv(WORK/'rolling_5y_step6_super_landmark_data/super_landmark_test_metadata.csv')
        self.assertFalse(set(m.ID)&set(test.ID))
        names=pd.read_csv(WORK/'rolling_5y_step4_preprocessed/fold_0/feature_names.csv').feature_name
        self.assertFalse(set(names)&{'CKDstatus','interval','synthetic_art_affected','ID','data','synthetic'})
    def test_four_model_probability_shapes(self):
        n=len(pd.read_csv(WORK/'rolling_5y_step6_super_landmark_data/super_landmark_development_metadata.csv'))
        for model in ['cox','rsf','rnn','lstm']:
            s=np.load(WORK/'demo_predictions'/f'{model}_survival.npy');self.assertEqual(s.shape,(n,10));self.assertTrue(np.isfinite(s).all());self.assertTrue(((s>=0)&(s<=1)).all());self.assertTrue((np.diff(s,axis=1)<=1e-6).all())
    def test_future_information_and_IG(self):
        a=json.loads((WORK/'interpretation/verification.json').read_text());self.assertLess(a['future_perturbation_max_risk_change'],1e-6);self.assertLess(a['max_completeness_residual'],1e-3)
    def test_external_scope(self):
        a=json.loads((WORK/'external_validation/scope.json').read_text());self.assertTrue(a['synthetic_only'])
        for name in ['primary_frozen_survival.npy','secondary_recalibrated_survival.npy']:
            s=np.load(WORK/'external_validation'/name);self.assertTrue(np.isfinite(s).all());self.assertTrue((np.diff(s,axis=1)<=1e-6).all())

if __name__=='__main__':unittest.main()
