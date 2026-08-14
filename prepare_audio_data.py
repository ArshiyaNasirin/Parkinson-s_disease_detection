from pathlib import Path
import librosa
import numpy as np

# Dataset paths
healthy_path = Path("dataset/Healthy")
parkinsons_path = Path("dataset/Parkinsons")

# Settings
SAMPLE_RATE = 22050
N_MFCC = 40
MAX_LENGTH = 162


def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )

    # Make every recording the same size
    if mfcc.shape[1] < MAX_LENGTH:
        pad_width = MAX_LENGTH - mfcc.shape[1]
        mfcc = np.pad(
            mfcc,
            ((0, 0), (0, pad_width)),
            mode="constant"
        )
    else:
        mfcc = mfcc[:, :MAX_LENGTH]

    return mfcc


X = []
y = []

# Healthy = 0
print("Processing Healthy recordings...")

for file in healthy_path.rglob("*.wav"):
    try:
        features = extract_mfcc(file)
        X.append(features)
        y.append(0)
    except Exception as e:
        print("Error:", file, e)

# Parkinson's = 1
print("Processing Parkinson's recordings...")

for file in parkinsons_path.rglob("*.wav"):
    try:
        features = extract_mfcc(file)
        X.append(features)
        y.append(1)
    except Exception as e:
        print("Error:", file, e)

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

print("\nData preparation completed!")
print("X shape:", X.shape)
print("y shape:", y.shape)

# Save processed data
np.save("X_audio.npy", X)
np.save("y_audio.npy", y)

print("Audio features saved successfully!")