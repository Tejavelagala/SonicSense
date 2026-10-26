
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "processed" / "sonicsense_features.csv"

df = pd.read_csv(CSV_PATH)

feature_columns = (
    [f"mfcc_{i}" for i in range(1, 14)]
    + ["rms_mean", "zcr_mean", "spectral_centroid_mean"]
)

features = df[feature_columns]

print("=" * 55)
print("SONICSENSE FEATURE QUALITY CHECK")
print("=" * 55)

print("\nDataset shape:", df.shape)
print("Feature count:", len(feature_columns))
print("Missing feature values:", int(features.isna().sum().sum()))
print("Infinite feature values:", int(
    np.isinf(features.to_numpy()).sum()
))

print("\nRecords by official fold:")
print(df["fold"].value_counts().sort_index().to_string())

print("\nClass distribution in fold 10:")
print(
    df[df["fold"] == 10]["class"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nFeature ranges:")
print(features.describe().loc[["min", "mean", "std", "max"]].round(3).to_string())

print("\nDuplicate audio references:", int(
    df.duplicated(subset=["fold", "slice_file_name"]).sum()
))

print("\nFeature quality check completed.")
