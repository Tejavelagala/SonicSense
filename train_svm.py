
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load the same train/test split used by Random Forest
train_df = pd.read_csv(DATA_DIR / "train_features.csv")
test_df = pd.read_csv(DATA_DIR / "test_features.csv")

X_train = train_df.drop(columns=["class"])
y_train = train_df["class"]

X_test = test_df.drop(columns=["class"])
y_test = test_df["class"]

# Build and train the SVM pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=10, gamma="scale")),
])

print("Training the SVM model...")
model.fit(X_train, y_train)

# Evaluate on the held-out test set
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)
matrix = confusion_matrix(y_test, y_pred, labels=model.classes_)

print(f"\nSVM Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print("\nClassification Report:")
print(report)

# Save model and evaluation results
joblib.dump(model, MODEL_DIR / "svm_model.joblib")

with open(REPORT_DIR / "svm_report.txt", "w", encoding="utf-8") as f:
    f.write(f"SVM Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n")
    f.write(report)

pd.DataFrame(
    matrix,
    index=model.classes_,
    columns=model.classes_,
).to_csv(REPORT_DIR / "svm_confusion_matrix.csv")

print("\nSaved model: models/svm_model.joblib")
print("Saved report: reports/svm_report.txt")
print("Saved confusion matrix: reports/svm_confusion_matrix.csv")
