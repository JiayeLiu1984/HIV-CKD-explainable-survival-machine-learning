# Longitudinal CKD prediction in people with HIV

A runnable synthetic example of the revised development / internal test / external validation design. No patient records or clinical model weights are distributed.

## Run

Python 3.10 in a separate environment:

```bash
python -m pip install -r requirements.txt
python scripts/run_all.py
```

Default flow: **simulate data → patient-level 70:30 split → development-only five-fold model comparison and calibration → lock the selected LSTM → final internal test evaluation → frozen external validation → test-set risk groups and interpretation**.

The default simulates 600 source-cohort patients (420 development, 180 internal test) and 240 independent external patients. Neural training uses 2 epochs; the RSF search and bootstrap budgets are intentionally small. These numbers demonstrate software execution, not the manuscript estimates.

## Dataset roles

| Dataset | Permitted role | Output |
|---|---|---|
| Development set (70%) | Model construction, five-fold cross-validation, tuning/comparison, model selection, fitting preprocessing and calibration, defining risk cutpoints | `development_cv/`: development diagnostics only |
| Reserved internal test set (30%) | Final evaluation of the selected frozen LSTM; no fitting, selection, early stopping or recalibration | `internal_validation/`: final internal metrics, calibration assessment, DCA and patient bootstrap |
| Independent external cohort | Apply the identical frozen predictor without retuning | `external_validation/`: primary external metrics |

The synthetic example fixes LSTM from the manuscript's development-stage selection; it does not claim that LSTM must win a random-data comparison. RNN/LSTM use the retained selected configurations, while the shortened RSF search exercises tuning. Full-budget tuning sources remain in `study_sources/`.

The locked predictor is the ensemble of five development-fold LSTMs with their corresponding fitted preprocessing and one set of development-derived calibration offsets. It is **not** a new single model refitted on all development patients. Exactly this predictor is used for both internal and external evaluation. Neither evaluation dataset changes its parameters.

`model_lock.json` records development IDs, model-selection provenance and hashes of every model, preprocessor and calibration file before the internal test is evaluated. Internal testing verifies these hashes. Tests also perturb internal outcomes and verify that predictions remain unchanged.

## Outputs

Each run creates a fresh `results/synthetic_*` directory, with `RUN_SUMMARY.json` and full `logs/`.

- `development_cv/`: four-family development comparison and fitted calibration; **not final internal performance**.
- `internal_validation/`: selected LSTM performance on the reserved 30% test patients only.
- `external_validation/`: independent frozen LSTM validation.
- `figures/`: internal-test risk groups and externally transported groups, using development-derived cutpoints; the four-model figure is explicitly a development comparison.
- `interpretation/`: calibrated frozen-ensemble IG and temporal occlusion on internal-test examples, using development-only reference values.

## Optional analyses

```bash
python scripts/run_all.py --with-sensitivity
python scripts/run_all.py --with-recalibration
python scripts/run_all.py --three-seeds
```

Synthetic ART recoding and seed stability remain **development diagnostics**, separate from final internal evaluation. Optional external recalibration uses external outcomes and is labeled secondary adaptation; it is not the frozen primary result. Adapted D:A:D/VHA comparison and centre rotation are retained as full-source optional analyses with their input requirements.

## Traceability

See [dataset/result mapping](docs/DATASET_RESULT_MAP.md), [reviewer mapping](docs/REVIEWER_CODE_MAP.md), [historical manuscript-result mapping](docs/RESULT_CODE_MAP.md), and [execution scope](docs/REPRODUCTION.md).

Historical source names/comments may retain the previous cross-validation terminology and filenames because they are source-level provenance. Those files do not establish independent internal-test performance. Existing clinical manuscript numbers have not been regenerated, relabeled or replaced by this synthetic demonstration. The revised internal-test results require evaluation on the actual reserved clinical cohort before manuscript claims can be changed.

All data in `data/` are independent random draws. The filename `synthetic_development.csv` denotes the **600-person source cohort before splitting**, not the 420-person development set. Generated models and patient-level outputs stay locally under ignored `results/`. The endpoint is single-event right-censored CKD; death is not modeled as a competing risk. IG describes predictions, not causal effects.
