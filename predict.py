
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR / "models" / "svm_improved_official_fold10_model.joblib"
)

N_MFCC = 13

FEATURE_COLUMNS = (
    [f"mfcc_{i}_mean" for i in range(1, N_MFCC + 1)]
    + [f"mfcc_{i}_std" for i in range(1, N_MFCC + 1)]
    + ["rms_mean", "zcr_mean", "spectral_centroid_mean"]
)


def extract_audio_features(audio_path):
    """Extract the same 29 features used to train the improved SVM."""

    signal, sample_rate = librosa.load(
        str(audio_path),
        sr=None,
        mono=True,
    )

    if signal.size == 0:
        raise ValueError("The audio file is empty.")

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

    if len(features) != len(FEATURE_COLUMNS):
        raise ValueError(
            f"Expected {len(FEATURE_COLUMNS)} features, "
            f"but extracted {len(features)}."
        )

    if not np.isfinite(features).all():
        raise ValueError("Extracted features contain NaN or infinity.")

    return pd.DataFrame([features], columns=FEATURE_COLUMNS)


def predict_sound(audio_path):
    """Return the predicted sound class and raw SVM decision scores."""

    if not Path(audio_path).is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    features = extract_audio_features(audio_path)

    predicted_class = model.predict(features)[0]
    decision_scores = model.decision_function(features)

    return predicted_class, decision_scores


if __name__ == "__main__":
    test_audio = (
        BASE_DIR
        / "data"
        / "raw"
        / "UrbanSound8K"
        / "audio"
        / "fold1"
        / "7061-6-0-0.wav"
    )

    predicted_class, decision_scores = predict_sound(test_audio)

    print("SONICSENSE — IMPROVED PREDICTION TEST")
    print("Audio file:", test_audio.name)
    print("Predicted sound:", predicted_class)
    print("Decision scores:", decision_scores)
    print("Feature count:", len(FEATURE_COLUMNS))
    print("Prediction test completed.")
