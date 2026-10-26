
from pathlib import Path

import pandas as pd
import soundfile as sf
import librosa

# Dataset paths
DATA_DIR = Path("data/raw/UrbanSound8K")
METADATA_PATH = DATA_DIR / "metadata" / "UrbanSound8K.csv"
AUDIO_DIR = DATA_DIR / "audio"

# Load metadata
metadata = pd.read_csv(METADATA_PATH)

print("=" * 55)
print("SONICSENSE: AUDIO EXPLORATION")
print("=" * 55)

print(f"Total audio records: {len(metadata)}")
print(f"Number of classes: {metadata['class'].nunique()}")
print(f"Sound classes: {sorted(metadata['class'].unique())}")

# Select one example from each class
samples = metadata.groupby("class", sort=True).first().reset_index()

print("\nDETAILS OF ONE AUDIO FILE PER CLASS")
print("-" * 55)

for _, row in samples.iterrows():
    audio_path = (
        AUDIO_DIR
        / f"fold{row['fold']}"
        / row["slice_file_name"]
    )

    # Read audio file information
    info = sf.info(str(audio_path))

    # Load audio as a mono signal for analysis
    signal, sample_rate = librosa.load(
        str(audio_path),
        sr=None,
        mono=True
    )

    duration = len(signal) / sample_rate

    print(f"\nClass: {row['class']}")
    print(f"File: {row['slice_file_name']}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Channels in original file: {info.channels}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Audio samples: {len(signal)}")
    print(f"Audio shape after loading: {signal.shape}")

print("\nAudio exploration completed successfully!")
