import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import librosa
import tensorflow as tf


# Settings
SAMPLE_RATE = 22050
RECORD_SECONDS = 4
N_MFCC = 40
MAX_LENGTH = 162

MODEL_PATH = "models/parkinsons_cnn_lstm.keras"
AUDIO_FILE = "recorded_voice.wav"


# Load model
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# Record voice
print("\n===================================")
print(" Parkinson's Voice Detection")
print("===================================")

input("Press ENTER and start speaking...")

print("\nRecording...")
print("Please speak for 4 seconds.")

recording = sd.rec(
    int(RECORD_SECONDS * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32"
)

sd.wait()

print("Recording completed!")

# Save recording
write(
    AUDIO_FILE,
    SAMPLE_RATE,
    recording
)

print("Voice saved as:", AUDIO_FILE)


# Extract MFCC
audio, sr = librosa.load(
    AUDIO_FILE,
    sr=SAMPLE_RATE
)

mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sr,
    n_mfcc=N_MFCC
)


# Pad or cut
if mfcc.shape[1] < MAX_LENGTH:

    pad_width = MAX_LENGTH - mfcc.shape[1]

    mfcc = np.pad(
        mfcc,
        ((0, 0), (0, pad_width)),
        mode="constant"
    )

else:

    mfcc = mfcc[:, :MAX_LENGTH]


# Change shape
mfcc = np.transpose(mfcc)

# Add batch dimension
mfcc = np.expand_dims(
    mfcc,
    axis=0
)


# Prediction
probability = model.predict(
    mfcc,
    verbose=0
)[0][0]


# Result
if probability >= 0.5:

    result = "Parkinson's"
    confidence = probability * 100

else:

    result = "Healthy"
    confidence = (1 - probability) * 100


print("\n===================================")
print("           RESULT")
print("===================================")

print("Prediction :", result)
print("Confidence :", round(confidence, 2), "%")

print("===================================")