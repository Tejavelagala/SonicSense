
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load the test dataset
test_df = pd.read_csv(DATA_DIR / "test_features.csv")
X_test = test_df.drop(columns=["class"])
y_test = test_df["class"]

# Load trained models
models = {
    "Random Forest": joblib.load(MODEL_DIR / "random_forest_model.joblib"),
    "SVM": joblib.load(MODEL_DIR / "svm_model.joblib"),
}

results = []

# Evaluate both models on the same test data
for model_name, model in models.items():
    predictions = model.predict(X_test)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Macro Precision": precision_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "Macro Recall": recall_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "Macro F1": f1_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "Weighted F1": f1_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
    })

# Display and save results
results_df = pd.DataFrame(results)
print("\n=== SonicSense: Model Comparison ===")
print(results_df.to_string(index=False, formatters={
    column: (lambda value: f"{value:.4f}")
    for column in results_df.columns
    if column != "Model"
}))

results_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

best_model = results_df.loc[results_df["Macro F1"].idxmax(), "Model"]
print(f"\nBest model by Macro F1: {best_model}")
print("Saved comparison: reports/model_comparison.csv")
