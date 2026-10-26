
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "sonicsense_improved_features.csv"
REPORT_DIR = BASE_DIR / "reports"
RESULT_PATH = REPORT_DIR / "svm_improved_all_folds_results.csv"

N_MFCC = 13

FEATURE_COLUMNS = (
    [f"mfcc_{i}_mean" for i in range(1, N_MFCC + 1)]
    + [f"mfcc_{i}_std" for i in range(1, N_MFCC + 1)]
    + ["rms_mean", "zcr_mean", "spectral_centroid_mean"]
)

def main():
    print("SONICSENSE: IMPROVED SVM — 10-FOLD EVALUATION")

    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Improved feature CSV not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)

    required_columns = FEATURE_COLUMNS + ["class", "fold"]
    missing_columns = [c for c in required_columns if c not in data.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if data[required_columns].isna().any().any():
        raise ValueError("Missing values found in features or labels.")

    X = data[FEATURE_COLUMNS]
    y = data["class"]
    folds = data["fold"]

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Non-finite feature values found.")

    if set(folds.unique()) != set(range(1, 11)):
        raise ValueError("Expected official folds 1 through 10.")

    results = []

    for fold_number in range(1, 11):
        train_mask = folds != fold_number
        test_mask = folds == fold_number

        X_train, X_test = X.loc[train_mask], X.loc[test_mask]
        y_train, y_test = y.loc[train_mask], y.loc[test_mask]

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=10, gamma="scale")),
        ])

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        macro_f1 = f1_score(
            y_test, predictions, average="macro", zero_division=0
        )

        results.append({
            "fold": fold_number,
            "test_records": len(y_test),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        })

        print(
            f"Fold {fold_number:2d}/10 | "
            f"Test records: {len(y_test):3d} | "
            f"Accuracy: {accuracy:.4f} | "
            f"Macro F1: {macro_f1:.4f}",
            flush=True,
        )

    results_df = pd.DataFrame(results)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULT_PATH, index=False)

    print("\nCROSS-VALIDATION SUMMARY")
    print(f"Mean accuracy:       {results_df['accuracy'].mean():.4f}")
    print(f"Accuracy std. dev.:  {results_df['accuracy'].std(ddof=0):.4f}")
    print(f"Mean macro F1:       {results_df['macro_f1'].mean():.4f}")
    print(f"Macro F1 std. dev.:  {results_df['macro_f1'].std(ddof=0):.4f}")
    print(f"\nResults saved to: {RESULT_PATH}")
    print("\nEvaluation finished. Existing models and app were not changed.")

if __name__ == "__main__":
    main()
