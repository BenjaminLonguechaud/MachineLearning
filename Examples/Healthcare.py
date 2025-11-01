# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]

from sklearn.calibration import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import kagglehub
import os
import csv
import math
import json
import joblib
# import datasets
from pathlib import Path
import pandas as pd

resources_folder = Path(__file__).parent.parent / "resources"

def load_kaggle_config():
  """
  Load Kaggle configuration and set environment variables.

  Behavior:
  - Looks for a JSON config file at MachineLearning/resources/kagglehub_config.json.
    Expected keys (optional):
      - "KAGGLE_CONFIG_DIR": path to directory containing kaggle.json
      - "KAGGLEHUB_CACHE": path for kagglehub cache
  - If the JSON file is present it sets the corresponding environment variables.
  - If the JSON file is absent but resources/kaggle.json exists, it sets KAGGLE_CONFIG_DIR
    to the resources folder.
  - If values are missing, falls back to a sensible default for KAGGLEHUB_CACHE.

  Example config (MachineLearning/resources/kagglehub_config.json):
  {
    "KAGGLE_CONFIG_DIR": "C:/Users/***REMOVED***Coding/GitHub/MachineLearning/resources",
    "KAGGLEHUB_CACHE": "C:/Users/***REMOVED***Downloads/MLData"
  }
  """
  config_path = resources_folder / "kagglehub_config.json"
  kaggle_json = resources_folder / "kaggle.json"
  os.environ['KAGGLE_CONFIG_DIR'] = str(kaggle_json)

  if config_path.exists():
    try:
      with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
      if "KAGGLEHUB_CACHE" in cfg and cfg["KAGGLEHUB_CACHE"]:
        os.environ["KAGGLEHUB_CACHE"] = str(cfg["KAGGLEHUB_CACHE"])
        print("kagglehub_config.json found, KAGGLEHUB_CACHE set.")
    except Exception as e:
      print("Warning: failed to read kagglehub_config.json:", e)
  else:
    print("Warning: No kagglehub_config.json found, using defaults.")

def to_continuous_distribution(rows, header, index):
  """
  Given a list of rows (including header), filter so the result has an equal number of rows with
  index == 0 and index == 1. The elements in the returned list are interleaved, i.e., 0, 1, 0, 1, ...

  Args:
    rows (list of list): The CSV data, including the header as the first row.
    header (list): The header row (column names).

  Returns:
    list of list: The filtered list, including the header as the first row.
  """
  if not rows or not header:
    return rows
  idx = header.index(index)
  zero_rows = []
  one_rows = []
  for row in rows:
    if row[idx] == '0':
      zero_rows.append(row)
    elif row[idx] == '1':
      one_rows.append(row)

  if len(zero_rows) == 0 or len(one_rows) == 0:
    return [header]
  filtered = []
  for row_zero, row_one in zip(zero_rows, one_rows):
    filtered.append(row_zero)
    filtered.append(row_one)

  return [header] + filtered

def write_csv(file_path, rows):
  """
  Write the given rows to a CSV file at the specified path.

  Args:
    file_path (str): The path to the output CSV file.
    rows (list of list): The data to write, including the header as the first row.
  """
  with open(file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(rows)

def read_split(data_path):
  """
  Split the dataset into three CSV files: training, testing, and validation.

  This function reads the original dataset, divides the rows into three equal parts, and writes each part to a new CSV file.
  It also normalizes each split for balanced classes.
  """
  with open(data_path, 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)
    csvfile.seek(0)

    header = next(reader)
    rows = list(reader)  # Read all data rows into a list

    preprocessed_data = to_continuous_distribution(rows, header, 'HeartDisease')
    total_rows = len(preprocessed_data)

    # Calculate the number of rows for each file
    training_rows = math.ceil(total_rows * 70 / 100)
    test_rows = math.ceil(total_rows * 20 / 100) + training_rows

    # Divide the rows into three chunks
    training = preprocessed_data[1:training_rows-1]
    testing = preprocessed_data[training_rows:test_rows-1]
    validation = preprocessed_data[test_rows:]

    # print("Training data created with", len(training), "rows")
    # print("Testing data created with", len(testing), "rows")
    # print("Validation data created with", len(validation), "rows")
    # Write each split to a new CSV file
    # write_csv('training.csv', training)
    # write_csv('testing.csv', testing)
    # write_csv('validation.csv', validation)

    training_df = pd.DataFrame({col: [row[i] for row in training] for i, col in enumerate(header)})
    testing_df = pd.DataFrame({col: [row[i] for row in testing] for i, col in enumerate(header)})
    validation_df = pd.DataFrame({col: [row[i] for row in validation] for i, col in enumerate(header)})

    return training_df, testing_df, validation_df


def encode(training_df, testing_df):
    """
    Encode dataset columns so they can be used by scikit-learn models.

    This function:
    - Splits the provided DataFrames into feature matrices and label vector.
    - Encodes categorical (object dtype) feature columns into numeric values using
      a LabelEncoder for each column.
    - Returns training_features (DataFrame), training_labels (Series), testing_features (DataFrame).

    Notes:
    - Encoding is applied independently for training and testing features in the
      current implementation (LabelEncoder is fit separately per DataFrame column).
      For production use, fit encoders on training data only and apply the same
      transforms to testing data to avoid label mismatch.
    - Non-categorical columns are left unchanged.
    """
    training_features = training_df.drop('HeartDisease', axis=1)
    testing_features = testing_df.drop('HeartDisease', axis=1)
    training_labels = training_df['HeartDisease']

    # print("Original Training DataFrame:")
    # print(training_features)

    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Convert categorical features to numeric using LabelEncoder.
    # For each object-typed column, fit-transform training data and transform testing data.
    for column in training_features.columns:
        if training_features[column].dtype == object:
            training_features[column] = label_encoder.fit_transform(training_features[column])
    for column in testing_features.columns:
        if testing_features[column].dtype == object:
            testing_features[column] = label_encoder.fit_transform(testing_features[column])

    # print("\nDataFrame after converting to numeric and filling NaNs with 0:")
    # print(training_features)
    return training_features, training_labels, testing_features


def fit_model_and_predict(training_features, training_labels, testing_features, testing_df):
    """
    Train a Logistic Regression classifier and evaluate it on the testing set.

    Steps performed:
    - Instantiate a LogisticRegression model and fit it on training_features and training_labels.
    - Use the trained model to predict labels for testing_features.
    - Compute and print evaluation metrics: accuracy, confusion matrix, and classification report.

    Args:
      training_features (pd.DataFrame): Feature matrix for training (numeric).
      training_labels (pd.Series): Labels for training (HeartDisease values).
      testing_features (pd.DataFrame): Feature matrix for testing (numeric).
      testing_df (pd.DataFrame): Original testing DataFrame (used to obtain true labels).

    Side effects:
      - Prints accuracy, confusion matrix and classification report to stdout.

    Return:
      None
    """
    model = LogisticRegression(random_state=42, max_iter=400)

    # Fit the model on the training data
    model.fit(training_features, training_labels)

    coefficients = model.coef_
    intercept = model.intercept_
    print(f"Coefficients: {coefficients}")
    print(f"Intercept: {intercept}")

    testing_labels = testing_df['HeartDisease']

    # Predict labels for all samples in the testing_features DataFrame
    testing_results = model.predict(testing_features)

    # Print prediction for each test sample
    # for i, pred in enumerate(testing_results):
    #   print(f"Test sample {i}: predicted HeartDisease = {pred}")

    print("Accuracy:", accuracy_score(testing_labels, testing_results))
    print("Confusion Matrix:\n", confusion_matrix(testing_labels, testing_results))
    print("Classification Report:\n", classification_report(testing_labels, testing_results))



load_kaggle_config()
data_path = kagglehub.dataset_download("fedesoriano/heart-failure-prediction")
data_path += "\\heart.csv"

training_df, testing_df, validation_df = read_split(data_path)
training_features, training_labels, testing_features = encode(training_df, testing_df)
fit_model_and_predict(training_features, training_labels, testing_features, testing_df)


