"""Train four model families using development patients only."""
from ckd_example.models import development_cv

if __name__ == "__main__":
    development_cv()
