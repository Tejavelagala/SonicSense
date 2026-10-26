
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# 1. Paths
# --------------------------------------------------
FEATURES_PATH = Path("data/processed/sonicsense_features.csv")
OUTPUT_DIR = Path("data/processed")

TRAIN_PATH = OUTPUT_DIR / "train_features.csv"
TEST_PATH = OUTPUT_DIR / "test_features.csv"

TEST_SIZE = 0.20
RANDOM_STATE = 42

# --------------------------------------------------
# 2. Load the extracted features
# --------------------------------------------------
df = pd.read_csv(FEATURES_PATH)

print("=" * 55)
print("SONICSENSE: DATASET PREPARATION")
print("=" * 55)
print("Dataset shape:", df.shape)
print("Number of classes:", df["class"].nunique())

# --------------------------------------------------
# 3. Separate features and labels
# --------------------------------------------------
feature_columns = (
    [f"mfcc_{i}" for i in range(1, 14)]
    + ["rms_mean", "zcr_mean", "spectral_centroid_mean"]
)

X = df[feature_columns]
y = df["class"]

# --------------------------------------------------
# 4. Split into training and testing datasets
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# --------------------------------------------------
# 5. Save the split datasets
# --------------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_df = X_train.copy()
train_df["class"] = y_train

test_df = X_test.copy()
test_df["class"] = y_test

train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)

# --------------------------------------------------
# 6. Report the results
# --------------------------------------------------
print("\nTraining samples:", len(train_df))
print("Testing samples:", len(test_df))
print("Training shape:", train_df.shape)
print("Testing shape:", test_df.shape)

print("\nTraining class distribution:")
print(train_df["class"].value_counts().sort_index())

print("\nTesting class distribution:")
print(test_df["class"].value_counts().sort_index())

print("\nSaved training data to:", TRAIN_PATH)
print("Saved testing data to:", TEST_PATH)
print("\nDataset preparation completed successfully.")
