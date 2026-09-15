# Manuscript and reviewer traceability

Reference versions: Manuscript_ckd_dl.docx and Response to reviewers.docx supplied from the 99_原始文件备份_20260912 folder. The documents and patient records are not redistributed. This is a topic-to-code crosswalk, not a claim that all reported clinical results have been independently reproduced.

| Manuscript / reviewer concern | Code and execution scope |
|---|---|
| Longitudinal prediction; R1 major 4, R3 comment 2 | Default preprocessing and four models; study_sources/dynamic/01–10c. Annual landmarks 0–5, six-month grid; Cox/RSF use summaries, RNN/LSTM ordered histories available by each landmark. |
| Model comparison; R1 major 5, R3 comment 7 | Default four-model OOF evaluation; dynamic/11a–11d. The revised design has four families. The reserved 30% test set is excluded from development OOF metrics. |
| External validation; R1 major 2/6, R2 generalizability, R3 comment 5 | Default frozen LSTM; external/01–04. Development preprocessing, model and calibration are frozen. |
| Calibration and clinical impact; R2 clinical impact | Default metrics, DCA and risk groups; full external/ and risk_groups/. External recalibration is explicitly secondary and opt-in. |
| Interpretation; R1 major 4/10/11 | Default IG and temporal occlusion; full interpretation/ and dynamic/12a–12b. Attributions are not causal effects. |
| ART sensitivity; R1 major 1/12, R2 sensitivity, R3 outcome/confounding | Optional --with-sensitivity; full sensitivity/ and ART_SOURCE_SCOPE.md. Synthetic flags demonstrate recoding mechanics, not clinical adjudication. |
| Training stability; R2 pipeline stability | Optional --three-seeds; full sensitivity/three_seed/. |
| Centre rotation; R2 generalizability, R3 external validation | Optional formal sensitivity/centre_rotation_3fold_3seed.py; fixed-input three-fold/three-seed analysis, not fully nested preprocessing. |
| Existing scores; R1 major 5 and minor 15 | Optional clinical_scores/: exploratory adapted D:A:D/VHA at landmark 0. Missing components and input-version limits remain documented. |
| Follow-up, missingness, data sources; R1 major 7/9, R2 data source | Optional descriptive/; the synthetic dictionary defines a model-ready panel, not hospital extraction or upstream imputation. |
| Censoring; R3 comment 1/6 | Single right-censored CKD endpoint; death is not modeled as a competing event. |
| Code availability; R3 comment 6/7 | One command runs the synthetic main pipeline and integrity checks; full sources and input contracts are separately retained. |

Code-family paths are relative to study_sources/. Residual confounding and unavailable clinical variables remain limitations. Synthetic execution cannot verify original raw-record definitions, eGFR inputs or upstream imputation. See RESULT_CODE_MAP.md for figure/table mappings and REPRODUCTION.md for full-source order.
