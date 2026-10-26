
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

# Dataset paths
DATA_DIR = Path("data/raw/UrbanSound8K")
METADATA_PATH = DATA_DIR / "metadata" / "UrbanSound8K.csv"

# Load metadata and select a dog-bark sample
metadata = pd.read_csv(METADATA_PATH)
row = metadata[metadata["class"] == "dog_bark"].iloc[0]

audio_path = (
    DATA_DIR
    / "audio"
    / f"fold{row['fold']}"
    / row["slice_file_name"]
)

print("Audio file:", audio_path)
print("File exists:", audio_path.is_file())

# Load audio as mono, preserving its original sample rate
signal, sample_rate = librosa.load(
    str(audio_path),
    sr=None,
    mono=True
)

# 1. MFCCs: average 13 coefficients over time
mfcc = librosa.feature.mfcc(
    y=signal,
    sr=sample_rate,
    n_mfcc=13
)
mfcc_mean = np.mean(mfcc, axis=1)

# 2. RMS energy
rms = librosa.feature.rms(y=signal)
rms_mean = np.mean(rms)

# 3. Zero-crossing rate
zcr = librosa.feature.zero_crossing_rate(y=signal)
zcr_mean = np.mean(zcr)

# 4. Spectral centroid
centroid = librosa.feature.spectral_centroid(
    y=signal,
    sr=sample_rate
)
centroid_mean = np.mean(centroid)

# Combine the features into one numerical vector
features = np.concatenate([
    mfcc_mean,
    [rms_mean, zcr_mean, centroid_mean]
])

print("\n--- SonicSense Feature Extraction ---")
print("Sound class:", row["class"])
print("Sample rate:", sample_rate, "Hz")
print("Duration:", round(len(signal) / sample_rate, 3), "seconds")
print("MFCC feature count:", len(mfcc_mean))
print("RMS energy:", round(float(rms_mean), 6))
print("Zero-crossing rate:", round(float(zcr_mean), 6))
print("Spectral centroid:", round(float(centroid_mean), 3), "Hz")
print("Total feature count:", len(features))
print("Feature vector shape:", features.shape)
print("All features finite:", np.isfinite(features).all())

print("\nFeature vector:")
print(np.round(features, 4))

print("\nSingle-file feature extraction completed!")
