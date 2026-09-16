# Synthetic longitudinal CKD prediction example

**Synthetic data demonstrate the paper's main analysis workflow and data-isolation principles only. Generated performance is not a real study result and is not expected to match the manuscript.** No clinical data, clinical model weights or patient predictions are included.

## Quick start

Python 3.10 was used for testing. In a separate environment:

```bash
python -m pip install -r requirements.txt
python run_all.py
```

The runner executes the eight numbered scripts in order, followed by isolation checks. It runs on CPU. Default settings are in `config.json`: 600 source patients, 240 external patients, five development folds and two neural training epochs. These are short demonstration settings. A complete tested CPU run, including six isolation checks, took about 218 seconds on the recorded environment (see `TEST_REPORT.json`); hardware changes the runtime. Previous generated runs are preserved under `.run_history/` when the runner is repeated.

## Main workflow

| Script | Operation |
|---|---|
| `01_generate_synthetic_data.py` | Independently simulate the Shenzhen–Nanning source cohort and Chongqing external cohort, 47 clinical/ART predictors, event/censoring times and six-month observations. |
| `02_split_and_prepare_data.py` | Fixed patient-level 70:30 split, stratified by source centre and event; save split manifest and five patient folds within development. No test preprocessing is fitted or applied here. |
| `03_development_cv_models.py` | Fit preprocessing separately within each training fold; train super-landmark Cox, super-landmark RSF, RNN and LSTM; collect development five-fold OOF predictions. |
| `04_model_comparison_and_lock.py` | Development OOF comparison, calibration assessment and development-only calibration fitting; select LSTM according to the study design; lock artifacts and development risk tertiles. |
| `05_internal_test_evaluation.py` | First predictor application to the reserved 30% patients; frozen LSTM internal evaluation without training, tuning or recalibration. |
| `06_external_validation.py` | Apply exactly the same frozen pipeline to independent Chongqing patients, with no external adaptation. |
| `07_risk_stratification.py` | Transfer development tertile cutpoints unchanged to both evaluation cohorts; output group counts, CKD events, observed five-year risk and figures. |
| `08_model_interpretation.py` | Small IG example for the frozen calibrated LSTM ensemble: predictors, current/earlier information and one patient's updated risk. References come only from development training patients. |

## Data and model design

Annual prediction landmarks are 0, 1, 2, 3, 4 and 5 years after ART initiation. Patients must remain under follow-up and CKD-free at a landmark. Histories use a six-month grid; simulated measurements fall within the preceding three-month window. Missing grid rows use previous observations only, never backward filling or future observations. Risk targets cover the next five years in ten six-month intervals. Evaluation includes 1-, 3- and 5-year predicted risks, with five-year Uno C-index, iAUC and IBS at each landmark.

Cox and RSF use current values, historical means and linear change summaries; Cox is stratified by landmark and RSF includes landmark time. RNN/LSTM consume ordered histories truncated at each landmark. These compact example architectures are not a copy of the original neural architecture or full search budget. Simple hyperparameters are fixed in `config.json` before evaluation; no test/external-guided tuning is performed.

The development comparison uses five-fold OOF predictions and cross-fitted landmark-specific hazard-intercept calibration. This is development-stage comparison, not a fully nested performance claim. **LSTM is selected by the study design even when synthetic rankings favor another family.** No attempt is made to force synthetic results to favor LSTM.

The final predictor averages survival curves from the five fitted development-fold LSTMs, then applies calibration fitted on all development LSTM OOF predictions. It is a frozen fold ensemble, not a separate full-development refit. Internal and external validation call the same prediction function and use identical artifacts.

## Files and outputs

```text
01_generate_synthetic_data.py … 08_model_interpretation.py
run_all.py
config.json
ckd_example/                 Shared implementation; no historical/supplementary modules
tests/                      Isolation and frozen-prediction checks
data/                       Generated source/external tables and field dictionary
artifacts/                  Generated split manifest, preprocessing, models and model_lock.json
outputs/
   development_cv/           Four-model comparison, calibration and development cutpoints
   internal_test/            Final frozen LSTM internal results
   external_validation/      Independent frozen external results
   risk_stratification/      Internal/external groups and Kaplan–Meier risk figures
   model_interpretation/     IG example and updated risk trajectory
   logs/                     Full execution logs
   run_summary.json          Stage status, settings and timing
```

Generated data, artifacts, results and previous runs are excluded from Git. All are generated from scratch by the main command. Stage scripts may also be executed separately in numbered order; `run_all.py` starts a fresh run.

Every internal evaluation table has `dataset = internal_test`; every external evaluation table has `dataset = external_validation`. Each includes landmark-specific and six-landmark mean performance, calibration assessment and decision curves. Calibration **assessment** uses observed evaluation outcomes to measure agreement; it never refits the calibration mapping. Censoring-distribution estimates and Kaplan–Meier curves are evaluation statistics, not model-training parameters.

## Isolation checks

`artifacts/model_lock.json` records `selected_model = LSTM`, `internal_test_used_for_selection = false`, `external_used_for_selection = false`, development fitting IDs and artifact SHA-256 hashes. Configuration, models, preprocessing, calibration and cutpoints are locked. Evaluation audits record hashes before and after both evaluations.

The six checks verify patient/fold separation; development-only training and calibration; frozen hashes and dataset labels; recomputed development-only tertiles; prediction invariance to altered test/external outcomes; and valid probabilities plus IG/future-information checks. Run them after the pipeline with:

```bash
python -m unittest discover -s tests -v
```

IG is predictive attribution relative to a development reference, **not a causal effect**. The global example uses a small explicitly identified internal-test sample. The single-event right-censored endpoint does not model death as a competing risk. Artificial event enrichment exists solely to support a small runnable example, not to reproduce real disease incidence.

Only the main example is included. This version replaces the earlier larger repository; older sources remain in Git history and are not part of this run.
