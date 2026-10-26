
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "sonicsense_improved_features.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_PATH = MODEL_DIR / "svm_improved_official_fold10_model.joblib"
REPORT_PATH = REPORT_DIR / "svm_improved_official_fold10_report.txt"

FEATURE_COLUMNS = (
    [f"mfcc_{i}_mean" for i in range(1, 14)]
    + [f"mfcc_{i}_std" for i in range(1, 14)]
    + ["rms_mean", "zcr_mean", "spectral_centroid_mean"]
)

def main():
    print("SONICSENSE: IMPROVED SVM — OFFICIAL FOLD 10 HOLDOUT")

    data = pd.read_csv(DATA_PATH)

    required = FEATURE_COLUMNS + ["class", "fold"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if data[required].isna().any().any():
        raise ValueError("Missing values found in required columns.")

    X = data[FEATURE_COLUMNS]
    y = data["class"]
    folds = data["fold"]

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Non-finite feature values found.")

    train_mask = folds != 10
    test_mask = folds == 10

    X_train, X_test = X.loc[train_mask], X.loc[test_mask]
    y_train, y_test = y.loc[train_mask], y.loc[test_mask]

    print(f"Training recordings: {len(X_train)}")
    print(f"Fold 10 test recordings: {len(X_test)}")
    print(f"Features: {len(FEATURE_COLUMNS)}")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale")),
    ])

    print("\nTraining model...")
    model.fit(X_train, y_train)

    print("Evaluating on fold 10...")
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test, predictions, zero_division=0
    )

    print(f"\nFold 10 accuracy: {accuracy:.4f} ({accuracy:.2%})")
    print("\nClassification report:")
    print(report)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    REPORT_PATH.write_text(
        f"Improved SVM — official fold 10 holdout\n"
        f"Training recordings: {len(X_train)}\n"
        f"Test recordings: {len(X_test)}\n"
        f"Features: {len(FEATURE_COLUMNS)}\n"
        f"Accuracy: {accuracy:.4f}\n\n"
        f"{report}",
        encoding="utf-8",
    )

    print(f"Separate model saved: {MODEL_PATH}")
    print(f"Report saved: {REPORT_PATH}")
    print("Original models and app were not changed.")

if __name__ == "__main__":
    main()
