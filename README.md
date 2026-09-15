# Longitudinal CKD prediction in people with HIV

This repository provides the organized code for **Development and external validation of a longitudinal prediction model for dynamic chronic kidney disease risk in people with HIV**. The current workflow covers super-landmark Cox, super-landmark RSF, RNN and longitudinal LSTM, model comparison, external validation and Integrated Gradients. It preserves the numbered-module organization of the earlier baseline workflow.

**All included example observations are independently simulated. No original or de-identified patient data, fitted clinical models, patient-level predictions, notebook outputs, or clinical data-derived generative model is distributed.** The artificial event distribution is deliberately enriched so a small test dataset supports survival metrics. Synthetic results test software execution and cannot reproduce the manuscript estimates or establish clinical validity.

## Structure

```text
README.md / README_中文.md
requirements.txt / requirements-tested.txt
data/                 Independently generated development, external and baseline CSVs
scripts/              Numbered portable test modules and full-source runner
configs/              Public model hyperparameters and configuration example
study_sources/        Retained study implementations, organized by analysis
docs/                 Result-to-code mapping, provenance, scope and reproduction notes
tests/                Data, splitting, probability and future-information checks
results/              Created locally at runtime; not shipped
```

## Install

Python 3.10 was used for the integration test. Use a separate environment:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

For the exact versions used in the CPU smoke test, see `requirements-tested.txt`. PyTorch CPU/GPU distributions depend on the platform; the demo uses CPU. Formal neural tuning can require substantial compute.

## Run the synthetic integration workflow

From the release directory:

```bash
python scripts/run_all.py
# Optional analyses, excluded from the default run:
python scripts/run_all.py --with-sensitivity
python scripts/run_all.py --with-recalibration
python scripts/run_all.py --three-seeds
```

Each run writes to a fresh timestamped `results/synthetic_*` directory. No original paths are searched. The default dataset contains 600 fictitious development subjects and 240 independent fictitious external subjects; the 70/30 patient split in the supplied native source is preserved. Neural models run for 2 epochs with one seed/snapshot per fold; the RSF test uses one trial and 32 final trees. These are explicitly reduced test settings, not manuscript settings. Use `--n`, `--external-n`, `--epochs`, and `--bootstrap` to adjust the test.

The default order is **random data → preprocessing → four-model training and comparison → frozen LSTM external validation → risk stratification → interpretation**. LSTM is the model selected in the study; the demo does not reselect the external model from random-data rankings. A `RUN_SUMMARY.json` records settings, stage durations and status; `logs/` contains full execution logs. Sensitivity analyses are opt-in. See [reviewer traceability](docs/REVIEWER_CODE_MAP.md). The default CPU integration run took about 49 seconds on the tested workstation; time varies by hardware.

For stepwise execution, use the following order. By default, individual modules share `results/synthetic_run`; optionally set `CKD_WORKDIR` to another empty result directory before starting.

| Module | Input | Output |
|---|---|---|
| `module_00_generate_synthetic_data.py` | Fixed random seed only | Synthetic CSVs, dictionary, provenance |
| `module_01_data_preprocessing.py` | Synthetic longitudinal table | Native patient split, five folds, labels, tensors, 146 history summaries |
| `module_02_model_training.py` | Fold data | Five-fold Cox/RSF/RNN/LSTM survival predictions |
| `module_03_model_evaluation.py` | Predictions and simulated outcomes | Cross-fit calibration, Uno C-index, iAUC, IBS, DCA, paired patient bootstrap |
| `module_04_external_validation.py` | Synthetic external table, frozen synthetic models | Primary frozen external predictions; local recalibration only with `--with-recalibration` |
| `module_05_risk_groups_and_figures.py` | Calibrated synthetic predictions | Risk groups, transferred cutpoints, characteristics, two example figures |
| `module_06_interpretation.py` | Synthetic fold model and fold inputs | IG completeness, global attribution, temporal occlusion, future-input check |
| `module_07_sensitivity.py` | Synthetic outcomes/predictions | Event-to-censoring analysis; optional three training seeds |

## Retained study code

`study_sources/` contains the original analytical implementations exported from the local notebooks and supplemental scripts. Comments and mathematical code are retained. Machine-specific roots are replaced by explicit placeholders. Notebook outputs and execution metadata are excluded. `docs/source_manifest.json` lists source names, cell indices, hashes and intended roles.

The portable test wrappers run the native data-preparation code and native Cox, RSF, RNN and LSTM model implementations. Small demo wrappers separately exercise evaluation, external prediction and IG; the full study calibration, ensemble and bootstrap code is preserved alongside them. The demo is not a replacement implementation for reproducing the paper numerically.

For a full study module with approved local inputs:

```bash
python scripts/run_study_module.py --source study_sources/dynamic/01_patient_split_labels.py --config my_local_config.json
```

Copy `configs/study_config.example.json`, provide your own local paths and validate the cohort-specific counts and feature schema. See `docs/REPRODUCTION.md` for module order and the remaining input/version requirements. The source begins with an already aligned/imputed six-month panel; reconstruction of hospital-specific raw records is outside the portable demo.

## Scientific scope

Models predict a single right-censored CKD outcome. Death is not separately modeled as a competing risk. Predictions therefore are not competing-risk cumulative incidence. Landmark windows change over follow-up. IG and occlusion describe the fitted predictor, not causal treatment effects. Risk tertiles are descriptive, not validated intervention thresholds.

Primary external validation keeps the development model, preprocessing and calibration frozen. Local external recalibration is a separate secondary analysis. The centre-rotation source uses three inner folds and three seeds, with original locked internal-test subjects excluded and previously processed inputs retained; it is not a fully nested reconstruction of preprocessing and model selection.

The earlier baseline RSF/SHAP workflow is preserved in Git history at commit `4538bff`; the current checkout focuses on the revised longitudinal analysis.
