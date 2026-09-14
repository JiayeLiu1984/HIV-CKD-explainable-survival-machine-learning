# Model card

Research model for updated five-year CKD risk among adults receiving ART in participating Chinese HIV-care cohorts. Prediction landmarks occur from ART initiation through year five. This is not a validated treatment recommendation system.

The revised primary model is a longitudinal LSTM-v2; comparator strategies are landmark Cox, RSF and RNN. Main internal performance uses patient-level five-fold development-set OOF predictions. A reserved 30% internal test split exists but is not the source of reported OOF estimates. Primary Chongqing predictions use frozen development preprocessing, model weights and calibration. Secondary local calibration and post hoc centre rotation have different training scopes.

Death is right-censored. Output is not a competing-risk cumulative incidence. External absolute risks were overestimated. Cohorts are predominantly male; transportability outside participating centres is unestablished. IG explains predictions, not causal effects. Tertile and fixed risk bands are descriptive.

The archive distributes code and independent synthetic data, with no original patient data or fitted clinical weights. Synthetic data were deliberately chosen to exercise survival metrics on a small sample and do not represent real prevalence. The demo uses a short training budget and is not a numerical replication of the clinical study.

Review status: ART frozen-primary six-landmark metrics and bootstrap intervals were verified on 14 September 2026. Upstream clinical extraction/imputation, author declarations and ethics coverage still require source/author verification.
