# Reviewer concerns and current implementation

| Concern | Current implementation |
|---|---|
| Independent internal evaluation | Patient-level 70:30 split; development-only construction and comparison; scripts/module_03b_internal_validation.py evaluates the frozen selected LSTM only on reserved test patients. |
| Four model families and model selection | Development-only five-fold Cox/RSF/RNN/LSTM comparison; development_cv/ is not a final internal-performance report. The synthetic demo retains the LSTM chosen during the manuscript's development stage. |
| Frozen external validation | scripts/module_04_external_validation.py uses the identical frozen predictor as internal evaluation; calibration fitted in development only. |
| Longitudinal information | Annual prediction landmarks 0–5; summary models versus ordered-history RNN/LSTM; no future inputs at each prediction. |
| Clinical impact and risk stratification | Internal-test and external assessment with unchanged development cutpoints; evaluation includes calibration and DCA. |
| Interpretation | Frozen calibrated ensemble on internal-test examples, references fitted from development training data; predictive attribution, not causation. |
| Sensitivity and stability | Optional development-only diagnostics; full historical ART/centre/seed sources remain separately traceable. |
| Existing scores | Historical adapted landmark-0 D:A:D/VHA source retained, not relabeled as independent test results. |
| Code availability | One synthetic command, patient-disjointness checks, artifact locks and outcome-perturbation tests. |

Reference reviewer documents remain the supplied Manuscript_ckd_dl.docx and Response to reviewers.docx. This revision changes the runnable validation design; it does not retroactively change clinical results or assert that earlier responses already describe the new test evaluation. See DATASET_RESULT_MAP.md for exact result populations.
