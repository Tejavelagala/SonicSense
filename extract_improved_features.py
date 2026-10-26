
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw" / "UrbanSound8K"
METADATA_PATH = DATA_DIR / "metadata" / "UrbanSound8K.csv"
AUDIO_DIR = DATA_DIR / "audio"

OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "sonicsense_improved_features.csv"

N_MFCC = 13
PROGRESS_INTERVAL = 250


def extract_features(audio_path):
    """Extract 29 features: 13 MFCC means, 13 MFCC standard deviations,
    RMS mean, zero-crossing-rate mean, and spectral-centroid mean.
    """
    signal, sample_rate = librosa.load(
        str(audio_path),
        sr=None,
        mono=True,
    )

    if signal.size == 0:
        raise ValueError("Audio file is empty")

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sample_rate,
        n_mfcc=N_MFCC,
    )

    mfcc_means = np.mean(mfcc, axis=1)
    mfcc_stds = np.std(mfcc, axis=1)

    rms_mean = np.mean(librosa.feature.rms(y=signal))
    zcr_mean = np.mean(librosa.feature.zero_crossing_rate(y=signal))

    centroid_mean = np.mean(
        librosa.feature.spectral_centroid(
            y=signal,
            sr=sample_rate,
        )
    )

    features = np.concatenate([
        mfcc_means,
        mfcc_stds,
        [rms_mean, zcr_mean, centroid_mean],
    ])

    if not np.isfinite(features).all():
        raise ValueError("Features contain NaN or infinity")

    return features


metadata = pd.read_csv(METADATA_PATH)

feature_names = (
    [f"mfcc_{i}_mean" for i in range(1, N_MFCC + 1)]
    + [f"mfcc_{i}_std" for i in range(1, N_MFCC + 1)]
    + ["rms_mean", "zcr_mean", "spectral_centroid_mean"]
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

results = []
errors = []

print("=" * 60)
print("SONICSENSE: IMPROVED FEATURE EXTRACTION")
print("=" * 60)
print("Audio records:", len(metadata))
print("Features per recording:", len(feature_names))
print("Starting extraction...")

for index, (_, row) in enumerate(metadata.iterrows(), start=1):
    audio_path = (
        AUDIO_DIR
        / f"fold{int(row['fold'])}"
        / row["slice_file_name"]
    )

    try:
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        values = extract_features(audio_path)
        record = dict(zip(feature_names, values))

        record.update({
            "slice_file_name": row["slice_file_name"],
            "fold": int(row["fold"]),
            "classID": int(row["classID"]),
            "class": row["class"],
        })

        results.append(record)

    except Exception as exc:
        errors.append({
            "file": row["slice_file_name"],
            "fold": row["fold"],
            "error": str(exc),
        })

    if index % PROGRESS_INTERVAL == 0 or index == len(metadata):
        print(
            f"Processed {index}/{len(metadata)} | "
            f"Successful: {len(results)} | Errors: {len(errors)}"
        )

if results:
    output_df = pd.DataFrame(results)
    output_df = output_df[feature_names + [
        "slice_file_name", "fold", "classID", "class"
    ]]
    output_df.to_csv(OUTPUT_PATH, index=False)

    print("\nSaved:", OUTPUT_PATH)
    print("Output shape:", output_df.shape)
    print("Feature count:", len(feature_names))
    print("Successful records:", len(output_df))
    print("Failed records:", len(errors))
else:
    print("\nNo features extracted; output was not saved.")

if errors:
    error_path = OUTPUT_DIR / "improved_feature_extraction_errors.csv"
    pd.DataFrame(errors).to_csv(error_path, index=False)
    print("Error details saved:", error_path)

print("\nImproved feature extraction finished.")
