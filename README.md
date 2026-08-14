# 🧠 Parkinson's Disease Detection Using Voice Signals

An AI-based system for detecting Parkinson's disease from voice recordings using **MFCC features and a CNN + Bidirectional LSTM deep learning model**.

## 🎯 Objectives

- Detect Parkinson's disease from voice signals.
- Extract audio features using MFCC.
- Classify voice as Healthy or Parkinson's.
- Provide prediction confidence.
- Support Explainable AI using SHAP/LIME.
- Provide a Virtual Robotic Assistant.
- Provide a web-based interface.

## 🔄 Workflow

```text
Voice Recording / Upload
          ↓
Audio Preprocessing
          ↓
MFCC Feature Extraction
          ↓
CNN + Bidirectional LSTM
          ↓
Healthy / Parkinson's
          ↓
Confidence Score
          ↓
Explainable AI
          ↓
Virtual Assistant / Report

📊 Dataset
Class	Recordings
Healthy	574
Parkinson's	560
Total	1,134

Audio configuration:

Sample Rate: 22050 Hz
MFCC Coefficients: 40
Maximum Time Frames: 162
🧠 Model

CNN + Bidirectional LSTM

CNN extracts important audio patterns.
BiLSTM learns sequential voice patterns.
Final layer classifies Healthy/Parkinson's.
Performance

Test Accuracy: 94.27%

Confusion Matrix:

                 Predicted
              Healthy  Parkinson's
Healthy          108       7
Parkinson's       6      106
🔍 Explainable AI

The project uses:

SHAP – feature contribution analysis
LIME – individual prediction explanation
🛠️ Technologies Used

Languages: Python, HTML, CSS, JavaScript

AI/ML: TensorFlow, Keras, CNN, BiLSTM, Scikit-learn

Audio: Librosa, MFCC

Data: NumPy, Pandas

Backend: Flask

XAI: SHAP, LIME

Tools: VS Code, Git, GitHub

📂 Project Structure
Parkinson's_disease_detection/
├── backend/
├── frontend/
├── check_audio.py
├── extract_features.py
├── prepare_audio_data.py
├── record_and_predict.py
├── train_cnn_lstm.py
├── voice_prediction.py
├── parkinsons detection.py
├── start_app.bat
├── .gitignore
└── README.md
⚙️ Installation
git clone https://github.com/ArshiyaNasirin/Parkinson-s_disease_detection.git
cd Parkinson-s_disease_detection
python -m venv venv
venv\Scripts\activate

Install the required Python packages according to the project environment.

🚀 Future Enhancements
Real-time voice analysis
Larger datasets
Improved XAI visualization
Cloud deployment
Mobile application
Multilingual virtual assistant
⚠️ Disclaimer

This project is developed for academic and research purposes and is not a replacement for professional medical diagnosis.

👩‍💻 Author

Arshiya Nasirin
B.E. CSE – Artificial Intelligence & Machine Learning

⭐ If you find this project useful, consider starring the repository!



