# Model card

Research model for updated five-year CKD risk among adults receiving ART in participating Chinese HIV-care cohorts. Prediction landmarks occur from ART initiation through year five. This is not a validated treatment recommendation system.

The runnable model is the study-selected longitudinal LSTM, with Cox, RSF and RNN development comparators. Development patients are used for construction, five-fold comparison and calibration. Final internal performance now uses the reserved 30% test patients only. The predictor is a frozen ensemble of the five development-fold LSTMs, corresponding preprocessing and development-derived calibration. Independent external validation applies the same artifacts unchanged. Historical manuscript performance numbers have not been recalculated and must not be relabeled as internal-test estimates.

Death is right-censored. Output is not a competing-risk cumulative incidence. External absolute risks were overestimated. Cohorts are predominantly male; transportability outside participating centres is unestablished. IG explains predictions, not causal effects. Tertile and fixed risk bands are descriptive.

The archive distributes code and independent synthetic data, with no original patient data or fitted clinical weights. Synthetic data were deliberately chosen to exercise survival metrics on a small sample and do not represent real prevalence. The demo uses a short training budget and is not a numerical replication of the clinical study.

Review status: ART frozen-primary six-landmark metrics and bootstrap intervals were verified on 14 September 2026. Upstream clinical extraction/imputation, author declarations and ethics coverage still require source/author verification.
