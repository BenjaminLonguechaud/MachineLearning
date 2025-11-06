# Healthcare — Overview

Generalized Linear Model (GLM) and fundamental supervised learning for classification problems.
This example demonstrates a compact, end-to-end supervised learning workflow for tabular healthcare data. They show how to load CSV data, preprocess and balance classes, encode categorical features, train a simple Generalized Linear Model (Logistic Regression), evaluate model performance, save/load trained models, and illustrate privacy-related concepts (e.g. a basic membership-inference demo). The code is pedagogical and intended to be adapted to your local environment and datasets.

This README explains the purpose of the example code in:
- Healthcare.py (healthcare data preprocessing, training, evaluation and a membership-inference demo)
- file_handling.py (small I/O helpers for reading/writing CSVs and saving/loading models)

## Purpose

Healthcare.py
- Provides an end-to-end, pedagogical workflow for tabular healthcare classification (HeartDisease example).
- Key utilities and updates:
  - interleave(...) - balance and interleave positive/negative class rows (0/1) for downstream splitting.
  - split(...) - normalize, balance and split raw CSV rows into training/testing/validation DataFrames.
  - encode(...) - convert categorical feature columns to numeric values suitable for scikit-learn.
  - fit_model_and_predict(...) - train a scikit-learn estimator, evaluate on the test set, save the trained model and return it.
  - membership_inference(...) - illustrative demo comparing model predictions/confidences for a training member vs an artificial non-member to show membership-inference behavior.

kaggle.json
  - Contains your Kaggle API credentials (username and key). Complete the `kaggle.json` file with your own Kaggle credentials for any dataset-download helpers to work.
  - Contains the path to the directory where Kaggle data will be downloaded.

file_handling.py
- Small, reusable helpers used by Healthcare.py:
  - write_csv(path, rows) - write rows to a CSV file.
  - load_trained_model(...) - load a joblib model from the cache folder.
  - save_model(...) - persist a picklable object (joblib) ensuring target directory exists.
  - load_kaggle_config(...) - optional helper to load kagglehub / Kaggle configuration from MachineLearning/resources and set environment variables (KAGGLEHUB_CACHE).


## Requirements
- Python 3.8+
- Recommended packages:
  - pandas
  - numpy
  - scikit-learn
  - joblib
  - (optional) kagglehub - if you use Kaggle helpers or dataset download utilities

Install with:
```bash
pip install pandas numpy scikit-learn joblib
# optional
pip install kagglehub
```

