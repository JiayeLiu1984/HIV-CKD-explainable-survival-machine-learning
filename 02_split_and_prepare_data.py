"""Fix patient-level 70:30 partition and development folds; no test preprocessing."""
from ckd_example.data import split_data

if __name__ == "__main__":
    split_data()
