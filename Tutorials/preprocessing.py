
# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
import os
import csv
import math
import datasets
from pathlib import Path
import pandas as pd

resources_folder = Path(__file__).parent.parent / "resources"

def load_kaggle_key():
  """
  Set the KAGGLE_CONFIG_DIR environment variable to the location of kaggle.json for authentication.

  This function determines the path to the kaggle.json file in the resources directory and sets the environment variable
  so that Kaggle API can authenticate.
  """
  config_file = resources_folder / "kaggle.json"
  print("Path to Kaggle config files:", config_file)
  os.environ['KAGGLE_CONFIG_DIR'] = str(config_file)

def preprocessing(rows, header, index):
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

def split_dataset(data_path):
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

    preprocessed_data = preprocessing(rows, header, 'HeartDisease')
    total_rows = len(preprocessed_data)

    # Calculate the number of rows for each file
    training_rows = math.ceil(total_rows * 70 / 100)
    test_rows = math.ceil(total_rows * 20 / 100) + training_rows

    # Divide the rows into three chunks
    training = preprocessed_data[1:training_rows-1]
    testing = preprocessed_data[training_rows:test_rows-1]
    validation = preprocessed_data[test_rows:]

    print("Training data created with", len(training), "rows")
    print("Testing data created with", len(testing), "rows")
    print("Validation data created with", len(validation), "rows")

    # Write each split to a new CSV file
    # write_csv('training.csv', training)
    # write_csv('testing.csv', testing)
    # write_csv('validation.csv', validation)

    training_df = pd.DataFrame({col: [row[i] for row in training] for i, col in enumerate(header)})
    testing_df = pd.DataFrame({col: [row[i] for row in testing] for i, col in enumerate(header)})

    training_df = datasets.Dataset.from_dict(training_df)
    testing_df = datasets.Dataset.from_dict(testing_df)

    dataset_dict = datasets.DatasetDict({
      "train": training_df,
      "test": testing_df
    })
    print(dataset_dict)

load_kaggle_key()
data_path = kagglehub.dataset_download("fedesoriano/heart-failure-prediction")
data_path += "\\heart.csv"
print("Path to dataset files:", data_path)

split_dataset(data_path)

