
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------
DATA_DIR = Path("data/raw/UrbanSound8K")
METADATA_PATH = DATA_DIR / "metadata" / "UrbanSound8K.csv"
AUDIO_DIR = DATA_DIR / "audio"

OUTPUT_DIR = Path("data/processed")
OUTPUT_PATH = OUTPUT_DIR / "sonicsense_features.csv"

N_MFCC = 13
PROGRESS_INTERVAL = 250


# --------------------------------------------------
# 2. Extract 16 features from one audio recording
# --------------------------------------------------
def extract_features(audio_path):
    signal, sample_rate = librosa.load(
        str(audio_path),
        sr=None,
        mono=True
    )

    if len(signal) == 0:
        raise ValueError("Audio file is empty")

    # MFCCs: average each of 13 coefficients over time
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sample_rate,
        n_mfcc=N_MFCC
    )
    mfcc_mean = np.mean(mfcc, axis=1)

    # Root mean square energy
    rms_mean = np.mean(
        librosa.feature.rms(y=signal)
    )

    # Zero-crossing rate
    zcr_mean = np.mean(
        librosa.feature.zero_crossing_rate(y=signal)
    )

    # Spectral centroid
    centroid_mean = np.mean(
        librosa.feature.spectral_centroid(
            y=signal,
            sr=sample_rate
        )
    )

    features = np.concatenate([
        mfcc_mean,
        [rms_mean, zcr_mean, centroid_mean]
    ])

    if not np.isfinite(features).all():
        raise ValueError("Features contain NaN or infinity")

    return features


# --------------------------------------------------
# 3. Load metadata
# --------------------------------------------------
metadata = pd.read_csv(METADATA_PATH)

print("=" * 60)
print("SONICSENSE: FULL-DATASET FEATURE EXTRACTION")
print("=" * 60)
print("Audio records:", len(metadata))
print("Sound classes:", metadata["class"].nunique())
print("Starting extraction... Please wait.")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

feature_names = (
    [f"mfcc_{i}" for i in range(1, N_MFCC + 1)]
    + ["rms_mean", "zcr_mean", "spectral_centroid_mean"]
)

results = []
errors = []


# --------------------------------------------------
# 4. Process every recording using its metadata
# --------------------------------------------------
for index, (_, row) in enumerate(
    metadata.iterrows(),
    start=1
):
    audio_path = (
        AUDIO_DIR
        / f"fold{int(row['fold'])}"
        / row["slice_file_name"]
    )

    try:
        if not audio_path.is_file():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        features = extract_features(audio_path)
        record = dict(zip(feature_names, features))

        record.update({
            "slice_file_name": row["slice_file_name"],
            "fold": row["fold"],
            "classID": row["classID"],
            "class": row["class"],
        })

        results.append(record)

    except Exception as exc:
        errors.append({
            "file": row["slice_file_name"],
            "fold": row["fold"],
            "error": str(exc)
        })

    if index % PROGRESS_INTERVAL == 0 or index == len(metadata):
        print(
            f"Processed {index}/{len(metadata)} | "
            f"Successful: {len(results)} | "
            f"Errors: {len(errors)}"
        )

# --------------------------------------------------
# 5. Save results
# --------------------------------------------------
if results:
    feature_df = pd.DataFrame(results)

    # Keep feature columns first, followed by metadata
    output_columns = feature_names + [
        "slice_file_name", "fold", "classID", "class"
    ]
    feature_df = feature_df[output_columns]

    feature_df.to_csv(OUTPUT_PATH, index=False)

    print("\nSaved feature dataset to:", OUTPUT_PATH)
    print("Successful records:", len(feature_df))
    print("Feature columns:", len(feature_names))
    print("Output shape:", feature_df.shape)
else:
    print("\nNo features were extracted; no dataset was saved.")

# Save error details separately if any files failed
if errors:
    error_path = OUTPUT_DIR / "feature_extraction_errors.csv"
    pd.DataFrame(errors).to_csv(error_path, index=False)
    print("Error report saved to:", error_path)

print("\nFull-dataset extraction finished.")
print("Successful:", len(results))
print("Failed:", len(errors))
