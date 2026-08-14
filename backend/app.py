from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import librosa
import numpy as np
import os

app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {"origins": ["http://127.0.0.1:8000", "http://localhost:8000", "http://0.0.0.0:8000"]}},
    supports_credentials=True,
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "OPTIONS"]
)

# Load trained CNN-BiLSTM model
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "parkinsons_cnn_lstm.keras"
)

model = tf.keras.models.load_model(MODEL_PATH)

print("CNN-BiLSTM model loaded successfully!")


# ---------------------------------------
# Extract MFCC features
# ---------------------------------------
def extract_features(audio_path):

    audio, sr = librosa.load(
        audio_path,
        sr=22050
    )

    # Extract 40 MFCC features
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    # Make exactly 162 time frames
    if mfcc.shape[1] < 162:
        mfcc = np.pad(
            mfcc,
            ((0, 0), (0, 162 - mfcc.shape[1])),
            mode="constant"
        )
    else:
        mfcc = mfcc[:, :162]

    return mfcc


# ---------------------------------------
# Health check
# ---------------------------------------
@app.route("/")
def home():
    return "Parkinson's AI Backend is running!"


# ---------------------------------------
# Voice prediction API
# ---------------------------------------
@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 200

    audio_file = request.files.get("audio")
    if audio_file is None:
        return jsonify({
            "error": "No audio file uploaded"
        }), 400

    if audio_file.filename == "":
        return jsonify({
            "error": "No file selected"
        }), 400

    # Temporary upload location
    upload_folder = os.path.join(
        os.path.dirname(__file__),
        "uploads"
    )

    os.makedirs(upload_folder, exist_ok=True)

    audio_path = os.path.join(
        upload_folder,
        audio_file.filename
    )

    audio_file.save(audio_path)

    try:

        # Extract MFCC
        mfcc = extract_features(audio_path)

        # Check model input shape
        input_shape = model.input_shape

        print("Model input shape:", input_shape)
        print("MFCC shape:", mfcc.shape)

        # Transpose MFCC to match model input (timeframe, features)
        mfcc = np.transpose(mfcc)

        # Prepare input according to model
        if len(input_shape) == 4:

            # (batch, 162, 40, 1)
            X = mfcc.reshape(1, 162, 40, 1)

        else:

            # (batch, 162, 40)
            X = mfcc.reshape(1, 162, 40)

        # Prediction
        prediction = model.predict(X, verbose=0)

        probability = float(np.ravel(prediction)[0])

        # 0 = Healthy
        # 1 = Parkinson's
        if probability >= 0.5:
            result = "Parkinson's"
            confidence = probability * 100
        else:
            result = "Healthy"
            confidence = (1 - probability) * 100

        # Basic voice measurements for UI
        audio, sr = librosa.load(
            audio_path,
            sr=22050
        )

        # Pitch
        pitches, magnitudes = librosa.piptrack(
            y=audio,
            sr=sr
        )

        pitch_values = pitches[pitches > 0]

        if len(pitch_values) > 0:
            pitch = float(np.mean(pitch_values))
        else:
            pitch = 0.0

        # RMS energy
        rms = librosa.feature.rms(y=audio)
        rms_value = float(np.mean(rms))

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        zcr_value = float(np.mean(zcr))

        # MFCC average
        mfcc_mean = float(np.mean(np.abs(mfcc)))

        response = {
            "prediction": result,
            "confidence": round(confidence, 2),

            "features": {
                "pitch": round(pitch, 2),
                "jitter": round(zcr_value, 4),
                "shimmer": round(rms_value, 4),
                "mfcc": round(mfcc_mean, 4)
            },

            "message": "Voice analysis completed successfully."
        }

        # Delete uploaded audio after prediction
        if os.path.exists(audio_path):
            os.remove(audio_path)

        return jsonify(response)

    except Exception as e:

        if os.path.exists(audio_path):
            os.remove(audio_path)

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        debug=True,
        port=5000
    )