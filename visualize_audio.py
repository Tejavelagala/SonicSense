
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Dataset paths
DATA_DIR = Path("data/raw/UrbanSound8K")
AUDIO_DIR = DATA_DIR / "audio"

# Select the dog-bark example inspected in Step 3.1
audio_path = AUDIO_DIR / "fold5" / "100032-3-0-0.wav"

# Load audio as mono at its original sample rate
signal, sample_rate = librosa.load(
    str(audio_path),
    sr=None,
    mono=True
)

# Create waveform and spectrogram plots
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Waveform
librosa.display.waveshow(
    signal,
    sr=sample_rate,
    ax=axes[0]
)
axes[0].set_title("Dog Bark — Waveform")
axes[0].set_xlabel("Time (seconds)")
axes[0].set_ylabel("Amplitude")

# Spectrogram in decibels
D = librosa.stft(signal)
S_db = librosa.amplitude_to_db(
    np.abs(D),
    ref=np.max
)

img = librosa.display.specshow(
    S_db,
    sr=sample_rate,
    x_axis="time",
    y_axis="hz",
    ax=axes[1]
)
axes[1].set_title("Dog Bark — Spectrogram")
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Frequency (Hz)")

fig.colorbar(img, ax=axes[1], format="%+2.0f dB")
plt.tight_layout()
plt.show()
