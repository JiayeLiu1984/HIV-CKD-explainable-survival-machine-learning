# Source selection

The current dynamic notebook contains many duplicate tuning, plotting and patient-case experiments. The release selects the six-step preprocessing, Cox v5, pooled RSF, final RNN, LSTM-v2 tuning/final fit and four-model comparison. The full old CKD-ML source cells remain in legacy_baseline. Patient-specific exploratory notebooks, prior model checkpoints and outputs are not included.

The exact three-fold/three-seed centre-rotation code was recovered from CKD_crosscenter_TRAIN.ipynb. Earlier five-seed alternative reconstruction scripts are not part of this package.

Sparse source-reference notebooks contain only cells required by the archived three-seed scripts, with empty output arrays and no metadata; they are code dependencies, not a second executable main pipeline. Full original paper plotting requires corresponding private intermediate outputs.

See source_manifest.json for every selected file, source identity, original code hash and released hash. Later path sanitization and code-dependency relocation are reflected in released hashes.
