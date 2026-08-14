import librosa
import numpy as np

# One test voice file
audio_path = "dataset/Healthy/healthy_000.wav"

# Load audio
audio, sample_rate = librosa.load(audio_path, sr=22050)

print("Audio loaded successfully!")
print("Sample rate:", sample_rate)
print("Duration:", len(audio) / sample_rate, "seconds")

# Extract MFCC
mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sample_rate,
    n_mfcc=40
)

print("MFCC shape:", mfcc.shape)