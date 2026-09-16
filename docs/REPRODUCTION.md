# Current execution contract

Run scripts/run_all.py for development-only five-fold comparison, model locking, reserved internal-test evaluation, frozen external validation and test-set interpretation. See DATASET_RESULT_MAP.md. The selected predictor is a five-fold LSTM ensemble with frozen development calibration, not an all-development single-model refit.

The remainder documents historical full-study source order and limitations. Historical development comparison outputs are not independent internal-test results. Do not use their clinical estimates as such.

# Reproduction guide and analytical boundaries

## Two execution modes

The portable synthetic integration workflow is complete from generated panel through model training, comparison, external evaluation and interpretation. It has a deliberately short training budget. Its numbers are not estimates from the study.

The retained full study sources use original training rules, ensemble logic, analysis contracts and output filenames. They require locally held input panels/tensors and matching outputs of preceding stages. They have been syntax checked and mapped to source files; the entire full-budget study, every historical plotting cell and every supplementary raw-record adapter have not been rerun. Availability of a source file does not establish that a previous exported result used that exact version. The original cohort is not distributed.

## Primary study module order

1. `dynamic/01_patient_split_labels.py` through `06_landmark_long_data.py`: construct labels, fixed patient folds, fold-specific scaling/encoding and history summaries.
2. `dynamic/07_landmark_cox.py`: unpenalized, landmark-stratified Cox using 95 fixed summaries.
3. `dynamic/08_landmark_rsf_tuning_oof.py`: set `RUN_STAGE=tune`, then `RUN_STAGE=final`; pooled RSF using the 95 features plus landmark year.
4. `dynamic/09a_rnn_tuning.py`, then `09b_rnn_oof.py`: RNN tuning and OOF fitting. The retained final cell contains its locked selected parameters.
5. `dynamic/10a_lstm_v2_tuning.py`, `10b_lstm_v2_select_trial.py`, then `10c_lstm_v2_oof.py`: LSTM-v2 tuning, explicit trial selection and final seed/snapshot ensemble. Verify the selected-trial record; do not silently substitute a different local study.
6. `dynamic/11a_four_model_calibration_comparison.py`: common OOF calibration, metrics and paired patient bootstrap. The 11b–11d modules generate the corresponding plots.
7. `external/01_external_preprocessing.py` through `04_primary_evaluation.py`: external transforms, tensors, frozen prediction and primary metrics. `05_secondary_crossfit_recalibration.py` is a separate local adaptation. The 06/07 plot/table sources require checking the frozen versus local-recalibrated input names.
8. `interpretation/01*` through `08*`: formal OOF IG, all-feature summary, temporal occlusion, grouped occlusion and patient illustrations. `dynamic/12a_integrated_gradients.py` and `12b_individual_explanation.py` retain the alternative later notebook implementations; their availability is not confirmation of a particular Fig. 4/5-year IG scope.
9. `sensitivity/centre_rotation_3fold_3seed.py`: the complete source reconstructed from `CKD_crosscenter_TRAIN.ipynb` code cells 1, 3, 5 and 7. This is the actual three-fold/three-seed fixed-input centre-rotation version; earlier five-seed reconstruction code is excluded from the release.
10. `sensitivity/three_seed/`, `sensitivity/art_outcome_evaluation.py`, and `risk_groups/`: retained analysis-specific scripts. Read their input contracts. Their original local-support paths are placeholders, not hidden data downloads.

All paths above are relative to `study_sources/`. The sources are not executed by import unless explicitly requested. Use `scripts/run_study_module.py` for explicit source/config execution. Do not run all historical alternatives in one shared output directory.

## Configuration

`configs/study_config.example.json` illustrates path replacement and named assignment overrides. Supply a new local work directory, input aliases, and the actual counts required by the specific module. The original sources contain assertions for their cohort sizes (31,911 total, 22,337 development and 9,574 locked test); these are preserved in full-source mode. They are replaced with generated dimensions only in the synthetic wrappers. Feature dimensions, label masking, loss definitions and patient partitions are not weakened by the test wrapper.

The source starts from an aligned, previously processed six-month panel. It cannot itself verify the provenance of earlier imputation or reconstruct original observation masks from already imputed values. The clinical raw-record extraction and final case-definition adjudication are not invented in this package. `data/data_dictionary.csv` specifies the synthetic panel contract.

## Recorded portability changes

- Replaced machine roots with `__CKD_WORKDIR__`, `__CKD_LEGACY_WORKDIR__`, `__CKD_SUPPORT_ROOT__` or explicit source-reference placeholders.
- Removed notebook outputs/metadata by exporting source text only. Fixed the accidental notebook-only `SSS#` prefix where encountered.
- The synthetic runner resolves the RSF cell's old Cox `v3` directory to the retained Cox `v5` directory. In full-source mode, set equivalent explicit aliases if needed.
- Synthetic-size assertions use generated patient counts and actual landmark row counts. All shape/partition checks remain in the native data/model code.
- For the demo only, neural epochs=2, seeds=1, snapshots=1; RSF trial count=1, final trees=32; patient bootstrap=20. Original source defaults remain available. Demo calibration/external/IG wrappers have narrower scope than the full study source, stated in their outputs.
- No original weights or fitted clinical preprocessing objects are included. Synthetic weights created during a run are marked `synthetic_only` and are excluded by `.gitignore`.

## Interpretation of validation

Passing the synthetic run establishes that the packaged interfaces and model computations can execute with fictional inputs. It does not certify the full study's statistical design or reproduce its reported performance. In particular, fold-based early stopping/tuning and cross-fit calibration remain conditional on the supplied design; the release does not retroactively turn them into fully nested validation.

The primary four-model sources preserve an original 70/30 patient split, while prose referring to the full source cohort may use a different denominator. The locked internal-test results' connection to the final LSTM-v2 version remains a separate manuscript check. The source-cohort follow-up table is not the model-eligible risk-set table.

For ART sensitivity, distinguish development-set event accounting (1,744/325/1,419) from source-cohort accounting (2,491/469/2,022); see ART_SOURCE_SCOPE.md and the retained aggregate verification. Modified D:A:D/VHA comparisons and unresolved diagnostic-code tables are not presented as confirmed reproduced results.

## Historical baseline code

The earlier baseline scripts remain available in Git history at commit `4538bff`. They are not required for the revised main workflow.

## Sharing

The release archive contains only selected source files, documentation, public hyperparameters and independently simulated CSVs. It contains no local clinical results directory, model checkpoint, tensor, cache, notebook output or clinical manuscript document. Keep original data outside the repository. Running scripts creates synthetic results locally; replacing their input with clinical data requires your own governance and must not be committed or uploaded.
