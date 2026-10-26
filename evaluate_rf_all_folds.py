
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "sonicsense_features.csv"
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

feature_columns = (
    [f"mfcc_{i}" for i in range(1, 14)]
    + ["rms_mean", "zcr_mean", "spectral_centroid_mean"]
)

results = []

for fold in range(1, 11):
    train_df = df[df["fold"] != fold]
    test_df = df[df["fold"] == fold]

    X_train = train_df[feature_columns]
    y_train = train_df["class"]
    X_test = test_df[feature_columns]
    y_test = test_df["class"]

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )

    results.append({
        "Fold": fold,
        "Train Samples": len(train_df),
        "Test Samples": len(test_df),
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
    })

    print(
        f"Fold {fold:02d}: "
        f"Accuracy={accuracy:.4f} | Macro F1={macro_f1:.4f}"
    )

results_df = pd.DataFrame(results)
results_df.to_csv(
    REPORT_DIR / "random_forest_all_folds_results.csv",
    index=False,
)

print("\n=== Random Forest: 10-Fold Summary ===")
print(f"Mean accuracy: {results_df['Accuracy'].mean():.4f}")
print(f"Accuracy std:  {results_df['Accuracy'].std():.4f}")
print(f"Mean macro F1: {results_df['Macro F1'].mean():.4f}")

print("\nSaved: reports/random_forest_all_folds_results.csv")
