import csv
import json
import os
from pathlib import Path

import joblib

resources_folder = Path(__file__).parent

def load_kaggle_config():
  """
  Load Kaggle configuration and set environment variables.

  Behavior:
  - Looks for a JSON kaggle_config.json file.
    Expected keys (optional):
      - "KAGGLE_CONFIG_DIR": path to directory containing kaggle.json
      - "username": Kaggle username
      - "key": Kaggle API key
  - If the JSON file is present it sets the corresponding environment variables.
  - If values are missing, falls back to a sensible default for KAGGLEHUB_CACHE.
  """
  kaggle_json = resources_folder / "kaggle.json"
  os.environ['KAGGLE_CONFIG_DIR'] = str(kaggle_json)

  if kaggle_json.exists():
    try:
      with open(kaggle_json, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
      if "KAGGLEHUB_CACHE" in cfg and cfg["KAGGLEHUB_CACHE"]:
        os.environ["KAGGLEHUB_CACHE"] = str(cfg["KAGGLEHUB_CACHE"])
    except Exception as e:
      print("Warning: failed to read kaggle.json:", e)
  else:
    print("Warning: No kaggle.json found, using defaults.")


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

def save_model(model, filename):
    """
    Save a picklable Python object (e.g. a scikit-learn model) to disk using joblib.

    Args:
      model: The object to serialize and save (must be picklable).
      filename (str or Path): Destination file path where the model will be written.

    Behavior:
      - Ensures the target directory exists before writing.
      - Uses joblib.dump for efficient serialization of large numpy arrays.
      - Prints a confirmation message on success.

    Raises:
      OSError / IOError: If the directory cannot be created or the file cannot be written.
    """
    # Ensure destination directory exists
    dest_dir = os.path.dirname(str(filename))
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)

    # Write the model to disk
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_trained_model(cache_env_var='KAGGLEHUB_CACHE', model_filename='logistic_regression_model.joblib'):
    """
    Load a trained model saved with joblib from the cache directory.

    Args:
      cache_env_var (str): Environment variable name that points to the cache directory.
      model_filename (str): Filename of the joblib model to load.

    Returns:
      The deserialized model object loaded by joblib.

    Raises:
      EnvironmentError: if the environment variable is not defined.
      FileNotFoundError: if the model file does not exist at the constructed path.
      Exception: propagated from joblib.load on failure to deserialize.
    """
    cache_dir = os.environ.get(cache_env_var)
    if not cache_dir:
        raise EnvironmentError(f"Environment variable '{cache_env_var}' is not set.")
    model_path = os.path.join(cache_dir, model_filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)
    return model
