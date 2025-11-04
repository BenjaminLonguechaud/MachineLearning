# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]

from sklearn.calibration import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import kagglehub
import os
import math
from file_handling import load_kaggle_config, read_csv, save_model
import pandas as pd

"""
Generalized Linear Model (GLM) and fundamental supervised learning for classification problems.
"""

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

def split(header, rows):
  """
  Split raw CSV data into training, testing and validation pandas DataFrames.

  This function:
  - Normalizes the input rows using to_continuous_distribution(...) so that the
    positive (HeartDisease == '1') and negative (HeartDisease == '0') examples are
    balanced and interleaved (0,1,0,1,...).
  - Divides the normalized rows into three splits using fixed proportions:
      * training: ~70% of the normalized rows
      * testing:  ~20% of the normalized rows
      * validation: remaining rows (~10%)
    Percentages are computed using math.ceil to avoid losing samples when totals are small.
  - Converts each split (list of rows) into a pandas DataFrame, using the provided header
    list as column names.

  Args:
    header (list of str): Column names extracted from the CSV file (header row).
    rows (list of list of str): CSV data rows (each row is a list of string cell values).

  Returns:
    tuple: (training_df, testing_df, validation_df)
      - Each element is a pandas.DataFrame constructed from the corresponding split.
      - DataFrames contain string values as read from the CSV; downstream code is expected
        to convert types (e.g., numeric conversion, encoding) as needed.

  Notes / Important details:
  - The function expects the header and rows to include the 'HeartDisease' column.
  - If the input contains too few samples or is unbalanced such that one class is missing,
    to_continuous_distribution may return only the header; in that case resulting DataFrames
    will be empty.
  - The splitting indices are computed on the normalized data (including header removed),
    and slicing is performed accordingly. Small off-by-one adjustments are intentionally
    conservative (math.ceil) to keep samples.
  - This function does not write CSV files to disk; it returns DataFrames for in-memory use.
  """
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

    # Initialize LabelEncoder to perform a deterministic mapping
    label_encoder = LabelEncoder()

    # Convert categorical features to numeric using LabelEncoder.
    # For each object-typed column, fit-transform training data and transform testing data.
    for column1, column2 in zip(training_features.columns, testing_features.columns):
        if training_features[column1].dtype == object:
            training_features[column1] = label_encoder.fit_transform(training_features[column1])
        if testing_features[column2].dtype == object:
            testing_features[column2] = label_encoder.fit_transform(testing_features[column2])

    encoded_data = label_encoder.fit_transform(["Normal"])
    decoded_data = label_encoder.inverse_transform(encoded_data)
    print(f"VULNERABILITY: Input data: \"Normal\"; decoded data {decoded_data}")

    # print("\nDataFrame after converting to numeric and filling NaNs with 0:")
    # print(training_features)
    return training_features, training_labels, testing_features


def fit_model_and_predict(model, training_features, training_labels, testing_features, testing_df):
    """
    Train a scikit-learn classifier on the provided training data, evaluate it on the testing set,
    save the trained model to disk and return the trained model.

    Behavior:
    - Fits `model` on `training_features` and `training_labels`.
    - Saves the fitted model to a joblib file located at:
      os.path.join(os.environ['KAGGLEHUB_CACHE'], "logistic_regression_model.joblib")
      (requires the KAGGLEHUB_CACHE environment variable to be set).
    - Predicts labels for all rows in `testing_features`.
    - Prints evaluation metrics: accuracy (and optionally can print confusion matrix / report).
    - Returns the fitted model instance for further use (inspection, demos, or saving elsewhere).

    Args:
      model: an unfitted scikit-learn estimator implementing fit/predict/predict_proba (e.g. LogisticRegression()).
      training_features (pd.DataFrame): Feature matrix for training (rows x feature columns).
      training_labels (pd.Series or array-like): Target labels for training (HeartDisease values).
      testing_features (pd.DataFrame): Feature matrix for testing (rows x feature columns).
      testing_df (pd.DataFrame): Original testing DataFrame (used to obtain true labels for evaluation).

    Returns:
      The fitted scikit-learn model instance.

    Notes:
    - Ensure feature DataFrames contain numeric columns (encode categorical features prior to calling).
    - The function calls save_model(...) to persist the trained model and will raise if the cache
      environment variable or destination directory is not available/writable.
    - This function prints results to stdout for quick inspection; for programmatic use, consider
      modifying to return evaluation metrics.
    """
    # Fit the model on the training data
    model.fit(training_features, training_labels)
    save_model(model, os.path.join(os.environ['KAGGLEHUB_CACHE'], "logistic_regression_model.joblib"))

    coefficients = model.coef_
    intercept = model.intercept_
    # print(f"Coefficients: {coefficients}")
    # print(f"Intercept: {intercept}")

    testing_labels = testing_df['HeartDisease']

    # Predict labels for all samples in the testing_features DataFrame
    testing_results = model.predict(testing_features)

    # Print prediction for each test sample
    # for i, pred in enumerate(testing_results):
    #   print(f"Test sample {i}: predicted HeartDisease = {pred}")

    print("Accuracy:", accuracy_score(testing_labels, testing_results))
    # print("Confusion Matrix:\n", confusion_matrix(testing_labels, testing_results))
    # print("Classification Report:\n", classification_report(testing_labels, testing_results))

    # Return trained model for further inspection / demonstrations
    return model

def membership_inference(model, training_features, testing_features):
    """
    Demonstrate a simple membership-inference style check using the trained model.

    Behavior:
    - Choose one real example from the training set (a known member).
    - Create an artificial / non-member example (here we use the mean of testing features,
      or a perturbed version of a real example) to represent a sample not seen during training.
    - Compute predicted label and predicted probabilities for both examples and print them.
    - This illustrates that models often assign different confidence to members vs non-members.

    Args:
      model: trained scikit-learn classifier with predict and predict_proba methods.
      training_features (pd.DataFrame): numeric training features used to train the model.
      testing_features (pd.DataFrame): numeric testing features not used for training.

    Notes:
    - This is a pedagogical demo, not a rigorous membership-inference attack.
    """
    # Pick a member example (first training sample)
    member_X = training_features.iloc[[0]]
    member_pred = model.predict(member_X)[0]
    member_proba = model.predict_proba(member_X)[0]

    # Build a non-member example: use the mean of the testing set (likely not a training member)
    non_member_X = testing_features.mean().to_frame().T
    non_member_X["Sex"] = "M"
    non_member_X["ChestPainType"] = "NAP"
    non_member_X["RestingECG"] = "Normal"
    non_member_X["ExerciseAngina"] = "N"
    non_member_X["ST_Slope"] = "Up"

    # Initialize LabelEncoder to perform a deterministic mapping
    label_encoder = LabelEncoder()

    # Convert categorical features to numeric using LabelEncoder.
    # For each object-typed column, fit-transform training data and transform testing data.
    for column1, column2 in zip(member_X.columns, non_member_X.columns):
        if member_X[column1].dtype == object:
            member_X[column1] = label_encoder.fit_transform(member_X[column1])
        if non_member_X[column2].dtype == object:
            non_member_X[column2] = label_encoder.fit_transform(non_member_X[column2])

    non_member_pred = model.predict(non_member_X)[0]
    non_member_proba = model.predict_proba(non_member_X)[0]

    # Print concise comparative results
    print("\n--- Membership Inference Demo ---")
    print("Member example (from training set) prediction:", member_pred)
    print("Member example predicted probabilities:", member_proba)
    print("\nNon-member example (testing-mean / artificial) prediction:", non_member_pred)
    print("Non-member example predicted probabilities:", non_member_proba)

    # Show difference in confidence for the class predicted
    if max(member_proba) > max(non_member_proba):
        print("Observation: The model is more confident on the member example than on the non-member example.")
    else:
        print("Observation: The model is not more confident on the member example in this simple demo.")



load_kaggle_config()
data_path = kagglehub.dataset_download("fedesoriano/heart-failure-prediction")
data_path += "\\heart.csv"
"""
1. Reads the CSV file, returning the header (column names) and the list of data rows.
2. Splits and balances the raw rows into training, testing, and validation pandas DataFrames.
3. Encodes categorical columns into numeric form and returns training features,
training labels, and testing features ready for modeling.
4. Instanciate LogisticRegression model
5. Trains a Logistic Regression model on the training data, evaluates it on the testing set,
prints metrics, saves the model, and returns the trained model.
6. Runs a short demo comparing model predictions/confidences for a known training example and
an artificial non-member example to illustrate membership-inference differences.
"""
header, rows = read_csv(data_path)
training_df, testing_df, validation_df = split(header, rows)
training_features, training_labels, testing_features = encode(training_df, testing_df)
model = LogisticRegression(random_state=42, max_iter=400)
model = fit_model_and_predict(model, training_features, training_labels, testing_features, testing_df)
# membership_inference(model, training_features, testing_features)


