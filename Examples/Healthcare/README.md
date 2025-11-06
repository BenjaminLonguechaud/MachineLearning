# Healthcare — Overview

This README explains the purpose of the updated example code in:
- MachineLearning/Examples/Healthcare.py (healthcare data preprocessing, training, evaluation and a small membership-inference demo)
- MachineLearning/Examples/file_handling.py (small I/O helpers for reading/writing CSVs and saving/loading models)

## Purpose of the updated code

Healthcare.py
- Provides an end-to-end, pedagogical workflow for tabular healthcare classification (HeartDisease example).
- Key utilities and updates:
  - interleave(...) — balance and interleave positive/negative class rows (0/1) for downstream splitting.
  - split(...) — normalize, balance and split raw CSV rows into training/testing/validation DataFrames.
  - encode(...) — convert categorical feature columns to numeric values suitable for scikit-learn.
  - fit_model_and_predict(...) — train a scikit-learn estimator, evaluate on the test set, save the trained model and return it.
  - membership_inference_demo(...) — illustrative demo comparing model predictions/confidences for a training member vs an artificial non-member to show membership-inference behavior.
- The code emphasizes clarity and repeatability (returns DataFrames, prints concise metrics, saves the model to disk).

file_handling.py
- Small, reusable helpers used by Healthcare.py:
  - write_csv(path, rows) — write rows to a CSV file.
  - load_trained_model(...) — load a joblib model from the cache folder.
  - save_model(...) — persist a picklable object (joblib) ensuring target directory exists.
  - load_kaggle_config(...) — optional helper to load kagglehub / Kaggle configuration from MachineLearning/resources and set environment variables (KAGGLE_CONFIG_DIR, KAGGLEHUB_CACHE).

## Requirements
- Python 3.8+
- Recommended packages:
  - pandas
  - numpy
  - scikit-learn
  - joblib
  - (optional) kagglehub — if you use Kaggle helpers or dataset download utilities

Install with:
```bash
pip install pandas numpy scikit-learn joblib
# optional
pip install kagglehub
```

