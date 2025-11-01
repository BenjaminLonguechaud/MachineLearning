# Healthcare.py — Overview

Healthcare.py is an end-to-end example that demonstrates a practical workflow for tabular healthcare data classification (HeartDisease). It shows how to load CSV data, preprocess and balance the dataset, encode categorical features, train a simple classifier, and evaluate / print predictions.

## Purpose
- Provide a runnable example illustrating common preprocessing and modelling steps for healthcare tabular data.
- Serve as a reference for reading/writing CSVs, handling headers, balancing classes, encoding categorical values, and running a scikit-learn model.

## Key capabilities
- Read datasets from CSV files in the `MachineLearning/resources/` folder.
- Balance classes so the training/testing sets contain equal numbers of samples for class labels (HeartDisease = 0 and 1).
- Encode categorical features to numeric values suitable for scikit-learn.
- Train a LogisticRegression classifier and evaluate it on a testing split.
- Print evaluation metrics (accuracy, confusion matrix, classification report) and per-sample predictions for the test set.
- Optionally integrate with kagglehub by loading configuration from the `resources` folder.

## Main functions (high level)
- `load_kaggle_key()` — reads an optional JSON config in `MachineLearning/resources/` to set Kaggle / cache environment variables.
- Preprocessing utilities — reading CSV, splitting into train/test, filling/mapping values, and class balancing.
- `encode_data()` — converts categorical columns to numeric encodings.
- Model training & evaluation — fits a LogisticRegression model, prints metrics, and prints individual test predictions.

## Requirements
- Python 3.8+
- Packages:
  - pandas
  - scikit-learn
  - numpy
  - (optional) kagglehub — if using Kaggle helpers
Install via:
```bash
pip install pandas scikit-learn numpy kagglehub
```

## Configuration
- Optional config file: `MachineLearning/resources/kagglehub_config.json`
  Example keys:
  - `KAGGLE_CONFIG_DIR` — path to kaggle.json directory
  - `KAGGLEHUB_CACHE` — cache directory for kagglehub / model downloads
- Alternatively place `kaggle.json` directly in `MachineLearning/resources/`.

## How to run
From repository root:
```bash
python MachineLearning/Tutorials/Healthcare.py
```

See top-of-file docstring and comments for additional runtime options and expected input file names.

## Output
- Console: training progress, accuracy, confusion matrix, classification report, and per-sample test predictions.
- Optionally writes or reads CSV files in `MachineLearning/resources/` for input/output datasets.

Notes:
- The example is intentionally simple and pedagogical. For production use, persist encoders fit on training data, add pipeline steps, and validate assumptions about missing values and categorical mapping.

