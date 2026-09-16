# Active dataset and result contract

| Artifact | Population | Role |
|---|---|---|
| data/synthetic_development.csv | Unsplit synthetic source cohort | Input only; not synonymous with development set |
| development_cv/landmark_performance.csv, six_landmark_mean.csv, paired_model_comparisons.csv | 70% development patients under five-fold cross-validation | Construction/comparison diagnostics, never final internal estimates |
| model_lock.json | Development IDs and development-fitted artifacts | Frozen before either evaluation |
| internal_validation/landmark_performance.csv, six_landmark_mean.csv | Reserved 30% patients; selected LSTM only | Final internal performance |
| internal_validation/calibration.csv, decision_curves.csv, patient_bootstrap.csv | Same reserved patients and unchanged predictor | Internal assessment; no parameter fitting |
| external_validation/primary_performance.csv | Independent external patients | Frozen primary external performance |
| external_validation/secondary_performance.csv | External patients, optional cross-fitted local recalibration | Secondary adaptation only |
| figures/risk_groups.csv | internal_test or external_transported, identified in each row | Development cutpoints applied unchanged |
| interpretation/*.csv | Internal-test examples; development reference values | Frozen calibrated ensemble attribution/occlusion |
| sensitivity/* | Development data | Optional diagnostics, not final internal performance |

The selected model remains the study-selected LSTM. The artificial comparison does not claim to reproduce the study's model-family selection. The default predictor uses the five development-fold models as a frozen ensemble; no internal or external patient influences any member's fit or calibration parameters. Outcome information used to assess censoring, observed risk and metrics does not update the predictor.

Historical clinical numbers and figure/table mappings must not be relabeled as test results. This change supplies an independent synthetic test evaluation; it does not generate clinical internal-test findings.
