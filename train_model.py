
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# --------------------------------------------------
# 1. Paths
# --------------------------------------------------
TRAIN_PATH = Path("data/processed/train_features.csv")
TEST_PATH = Path("data/processed/test_features.csv")

MODEL_DIR = Path("models")
OUTPUT_DIR = Path("reports")

MODEL_PATH = MODEL_DIR / "random_forest_model.joblib"
REPORT_PATH = OUTPUT_DIR / "random_forest_report.txt"
MATRIX_PATH = OUTPUT_DIR / "random_forest_confusion_matrix.csv"

RANDOM_STATE = 42

# --------------------------------------------------
# 2. Load training and testing data
# --------------------------------------------------
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

feature_columns = (
    [f"mfcc_{i}" for i in range(1, 14)]
    + ["rms_mean", "zcr_mean", "spectral_centroid_mean"]
)

X_train = train_df[feature_columns]
y_train = train_df["class"]

X_test = test_df[feature_columns]
y_test = test_df["class"]

print("=" * 60)
print("SONICSENSE: RANDOM FOREST MODEL TRAINING")
print("=" * 60)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Number of features:", len(feature_columns))
print("Number of classes:", y_train.nunique())

# --------------------------------------------------
# 3. Build and train the model
# --------------------------------------------------
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1
    ))
])

print("\nTraining model...")
model.fit(X_train, y_train)

# --------------------------------------------------
# 4. Evaluate the model
# --------------------------------------------------
print("Predicting test labels...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

print("\nAccuracy:", f"{accuracy:.4f}")
print("\nClassification report:")
print(report)

# --------------------------------------------------
# 5. Save the model and evaluation results
# --------------------------------------------------
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_PATH)

with open(REPORT_PATH, "w", encoding="utf-8") as file:
    file.write(f"Random Forest Accuracy: {accuracy:.4f}\n\n")
    file.write(report)

labels = sorted(y_test.unique())
matrix = confusion_matrix(y_test, y_pred, labels=labels)

pd.DataFrame(
    matrix,
    index=labels,
    columns=labels
).to_csv(MATRIX_PATH)

print("\nSaved model to:", MODEL_PATH)
print("Saved report to:", REPORT_PATH)
print("Saved confusion matrix to:", MATRIX_PATH)
print("\nRandom Forest training completed successfully.")
