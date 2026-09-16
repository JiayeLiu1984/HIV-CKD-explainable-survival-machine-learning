"""Evaluate the frozen LSTM on reserved internal test patients."""
from ckd_example.evaluation import evaluate

if __name__ == "__main__":
    evaluate('internal_test')
