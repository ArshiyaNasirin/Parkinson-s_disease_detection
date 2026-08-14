import numpy as np
import librosa
import tensorflow as tf


# =========================
# Settings
# =========================

SAMPLE_RATE = 22050
N_MFCC = 40
MAX_LENGTH = 162


# =========================
# Load trained model
# =========================

model = tf.keras.models.load_model(
    "models/parkinsons_cnn_lstm.keras"
)

print("Model loaded successfully!")


# =========================
# Extract MFCC
# =========================

def extract_mfcc(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )

    # Padding
    if mfcc.shape[1] < MAX_LENGTH:

        pad_width = MAX_LENGTH - mfcc.shape[1]

        mfcc = np.pad(
            mfcc,
            ((0, 0), (0, pad_width)),
            mode="constant"
        )

    # Cutting
    else:

        mfcc = mfcc[:, :MAX_LENGTH]

    # Change shape:
    # (40, 162) → (162, 40)

    mfcc = np.transpose(mfcc)

    # Add batch dimension
    # (162, 40) → (1, 162, 40)

    mfcc = np.expand_dims(
        mfcc,
        axis=0
    )

    return mfcc


# =========================
# Prediction function
# =========================

def predict_voice(file_path):

    features = extract_mfcc(file_path)

    probability = model.predict(
        features,
        verbose=0
    )[0][0]

    if probability >= 0.5:

        prediction = "Parkinson's"

        confidence = probability * 100

    else:

        prediction = "Healthy"

        confidence = (1 - probability) * 100

    print("\nPrediction:", prediction)
    print("Confidence:", round(confidence, 2), "%")


# =========================
# Test voice
# =========================

voice_file = "dataset/Healthy/healthy_000.wav"

predict_voice(voice_file)