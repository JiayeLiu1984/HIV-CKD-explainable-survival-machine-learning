"""Apply the same frozen LSTM to the independent external cohort."""
from ckd_example.evaluation import evaluate

if __name__ == "__main__":
    evaluate('external_validation')
