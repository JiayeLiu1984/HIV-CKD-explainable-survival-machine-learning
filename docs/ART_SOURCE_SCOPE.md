# ART sensitivity verification on 14 September 2026

The corrected Supplementary Table S14 and Figure S9 use the existing LSTM-v2 primary cross-fitted risk predictions and unchanged primary calibration, evaluated under the ART-censoring sensitivity labels. All six annual landmarks are given equal weight. Original clinical model weights were not retrained.

All six landmark point estimates were recomputed from the saved sensitivity labels and risk arrays and matched the archived frozen-primary landmark metrics within 1e-10. One thousand participant-level bootstrap resamples used common patient multiplicities across landmarks, fixed original sensitivity-outcome censoring references and fixed full-sample deciles for grouped ICI. Six-landmark means: C-index 0.8212333144, iAUC 0.8572361051, IBS 0.0221108653. The included aggregate table gives intervals and all individual landmarks.

The historical four-landmark bootstrap and outcome-specific recalibration scripts are retained for provenance. They are not the corrected primary six-landmark source. Secondary recalibration means are separately identified in the supplement. The earlier mixed-version table and plots are preserved in archived original documents, not mixed into the corrected results.

The new verification and plotting scripts require private approved model-ready arrays and cannot run on the supplied synthetic data without adapting the input contract. No such arrays or clinical weights are included. The portable synthetic modules remain software demonstrations, not clinical numerical replication.
