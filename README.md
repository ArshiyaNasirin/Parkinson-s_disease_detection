# 🧠 Parkinson's Disease Detection Using Explainable Deep Learning

An AI-powered **Parkinson's Disease Detection System** that analyzes voice recordings using Deep Learning to assist in identifying patterns associated with Parkinson's disease.

The project combines **CNN + Bidirectional LSTM**, **MFCC-based audio feature extraction**, **Explainable AI (XAI)**, and a **Virtual Robotic Assistant** to provide an interactive and understandable prediction experience.

> ⚠️ **Disclaimer:** This project is developed for academic and research purposes only. It is not intended to replace professional medical diagnosis or clinical evaluation.

---

## 📌 Project Overview

Parkinson's disease is a progressive neurological disorder that can affect speech and voice characteristics.

This project uses voice signals to identify acoustic patterns that may be associated with Parkinson's disease.

The system:

1. Accepts a voice recording from the user.
2. Processes the audio using **Librosa**.
3. Extracts **MFCC (Mel-Frequency Cepstral Coefficients)** features.
4. Passes the extracted features through a **CNN + Bidirectional LSTM** model.
5. Predicts whether the voice is classified as **Healthy** or **Parkinson's**.
6. Displays the prediction and confidence score.
7. Provides explainable AI information to help understand the prediction.
8. Provides interaction through a virtual robotic assistant.

---

## ✨ Features

### 🎙️ Voice-Based Detection

Upload or record a voice sample for Parkinson's disease prediction.

### 🤖 Deep Learning Model

Uses a hybrid **CNN + Bidirectional LSTM** architecture to learn spatial and temporal patterns from voice features.

### 🎵 MFCC Feature Extraction

Audio signals are converted into MFCC features using **Librosa**.

### 🔍 Explainable AI

Explainable AI techniques such as **SHAP/LIME** can be integrated to provide insights into model predictions.

### 📊 Prediction & Confidence

The system displays the predicted class along with the model's confidence score.

### 🤖 Virtual Robotic Assistant

An interactive virtual assistant is included to guide users through the prediction process and provide system information.

### 📄 Report Generation

The system can be extended to generate a prediction report containing the input information, prediction, confidence, and explanation.

### 🌓 Modern User Interface

The frontend provides a modern interface designed for simple and user-friendly interaction.

---

## 🛠️ Technologies Used

### Frontend

* HTML5
* CSS3
* JavaScript

### Backend

* Python
* Flask
* REST API

### Artificial Intelligence / Machine Learning

* TensorFlow
* Keras
* Convolutional Neural Network (CNN)
* Bidirectional LSTM
* Explainable AI
* SHAP
* LIME

### Audio Processing

* Librosa
* MFCC

### Data Processing

* NumPy
* Pandas
* Scikit-learn

### Development Tools

* VS Code
* Git
* GitHub

---

## 🧠 Model Architecture

The system uses a hybrid CNN + Bidirectional LSTM architecture.

```text
Voice Recording
       │
       ▼
Audio Preprocessing
       │
       ▼
MFCC Feature Extraction
       │
       ▼
CNN Layers
       │
       ▼
Bidirectional LSTM
       │
       ▼
Dense Layer
       │
       ▼
Prediction
       │
       ├── Healthy
       │
       └── Parkinson's
```

The CNN component learns important local patterns from the MFCC representation, while the Bidirectional LSTM captures temporal dependencies in the voice signal.

---

## 🎵 Audio Processing

The project uses the following preprocessing configuration:

| Parameter            |          Value |
| -------------------- | -------------: |
| Sample Rate          |       22050 Hz |
| MFCC Coefficients    |             40 |
| Maximum Time Frames  |            162 |
| Input Representation |           MFCC |
| Model Input          | `(1, 162, 40)` |

The extracted MFCC features are normalized and prepared before being passed to the trained deep learning model.

---

## 📂 Dataset

The project uses voice recordings divided into two classes:

```text
dataset/
│
├── Healthy/
│   ├── audio_001.wav
│   ├── audio_002.wav
│   └── ...
│
└── Parkinson/
    ├── audio_001.wav
    ├── audio_002.wav
    └── ...
```

### Dataset Summary

| Class       | Recordings |
| ----------- | ---------: |
| Healthy     |        574 |
| Parkinson's |        560 |
| **Total**   |   **1134** |

The dataset is used for academic experimentation and model development.

---

## 📈 Model Performance

The trained CNN + Bidirectional LSTM model achieved approximately:

**Test Accuracy: 94.27%**

Example confusion matrix:

```text
                 Predicted
              Healthy  Parkinson
Actual Healthy   108       7
Actual Parkinson  6      106
```

> Model performance may vary depending on the dataset split, preprocessing, training configuration, and evaluation environment.

---

## 📁 Project Structure

```text
Parkinson-s_disease_detection/
│
├── frontend/
│   └── index.html
│
├── backend/
│   ├── app.py
│   ├── prediction.py
│   └── ...
│
├── models/
│   └── parkinsons_cnn_lstm.keras
│
├── dataset/
│   ├── Healthy/
│   └── Parkinson/
│
├── screenshots/
│   ├── home.png
│   ├── upload.png
│   ├── prediction.png
│   └── assistant.png
│
├── requirements.txt
├── README.md
└── ...
```

> The exact structure may vary depending on the current version of the project.

---

# 🚀 How to Run the Project

## 1. Clone the Repository

```bash
git clone https://github.com/ArshiyaNasirin/Parkinson-s_disease_detection.git
```

Navigate into the project:

```bash
cd Parkinson-s_disease_detection
```

---

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

---

## 3. Install Dependencies

If `requirements.txt` is available:

```bash
pip install -r requirements.txt
```

Important packages include:

```text
tensorflow
keras
flask
flask-cors
librosa
numpy
pandas
scikit-learn
shap
lime
```

---

## 4. Start the Flask Backend

Navigate to the backend directory if required:

```bash
cd backend
```

Run:

```bash
python app.py
```

The backend should run at:

```text
http://127.0.0.1:5000
```

---

## 5. Start the Frontend

The frontend can be opened using a local development server.

For example:

```bash
python -m http.server 8000
```

Then open:

```text
http://127.0.0.1:8000
```

---

# 🌐 Live Demo

The frontend is deployed using Vercel.

**Live Website:**

https://parkinson-s-disease-detection-puce.vercel.app/

> The AI prediction functionality requires the backend API to be deployed and connected to the frontend.

---

# 📸 Screenshots

Screenshots can be added to the `screenshots` folder.

### 🏠 Home Page

![Home Page](screenshots/home.png)

### 🎙️ Voice Upload

![Voice Upload](screenshots/upload.png)

### 📊 Prediction Result

![Prediction Result](screenshots/prediction.png)

### 🤖 Virtual Robotic Assistant

![Virtual Assistant](screenshots/assistant.png)

> Replace the image filenames above with your actual screenshots.

---

# 🔄 System Workflow

```text
User
 │
 ▼
Web Interface
 │
 ▼
Upload / Record Voice
 │
 ▼
Flask Backend API
 │
 ▼
Audio Preprocessing
 │
 ▼
MFCC Extraction
 │
 ▼
CNN + BiLSTM Model
 │
 ▼
Prediction
 │
 ├───────────────┐
 ▼               ▼
Healthy      Parkinson's
 │               │
 └───────┬───────┘
         ▼
 Confidence Score
         │
         ▼
 Explainable AI
         │
         ▼
 Virtual Assistant / Report
```

---

# 🔐 API Overview

The Flask backend exposes API endpoints for communication between the frontend and AI model.

Example:

```text
POST /predict
```

### Request

The endpoint accepts an audio file.

### Response

Example:

```json
{
  "prediction": "Healthy",
  "confidence": 99.5
}
```

> The exact API endpoints and response format may differ depending on the current backend implementation.

---

# 🔍 Explainable AI

Explainable AI is included to improve transparency and help users understand the model's decision-making process.

Potential explanation methods include:

### SHAP

SHAP can be used to identify the contribution of input features toward the model's prediction.

### LIME

LIME can provide local explanations for individual predictions.

The goal is to make the deep learning prediction more interpretable rather than treating the model as a complete black box.

---

# 🤖 Virtual Robotic Assistant

The Virtual Robotic Assistant is designed to provide an interactive interface for users.

Potential functionalities include:

* Guiding users through voice recording
* Explaining prediction results
* Providing system instructions
* Answering basic project-related questions
* Helping users understand the AI output
* Providing general awareness information

---

# 🚀 Future Enhancements

The project can be further improved with:

* 🎙️ Real-time voice analysis
* 📱 Mobile application
* ☁️ Complete cloud deployment
* 🌍 Multilingual Virtual Robotic Assistant
* 🧠 Improved prediction accuracy
* 🔍 Advanced Explainable AI visualization
* 📊 Larger and more diverse datasets
* ⚡ Real-time prediction and reporting
* 📄 Automated medical-style research reports
* 🔊 Real-time speech analysis
* 🗣️ Voice-enabled virtual assistant
* 🔐 Secure user authentication
* 🗄️ Database integration
* 📈 Prediction history and analytics
* 🧪 Clinical dataset validation

---

# 👩‍💻 Development

This project is developed as an academic/final-year project with the goal of exploring the application of Artificial Intelligence and Deep Learning for voice-based Parkinson's disease detection.

---

# 📜 Disclaimer

This system is developed **strictly for academic and research purposes**.

The prediction produced by this application should **not be considered a medical diagnosis**. Parkinson's disease can only be diagnosed by qualified healthcare professionals using appropriate clinical evaluation and diagnostic procedures.

Users should consult a qualified medical professional for any health-related concerns.

---

# ⭐ Acknowledgements

This project makes use of open-source technologies and libraries including:

* TensorFlow
* Keras
* Librosa
* Flask
* NumPy
* Pandas
* Scikit-learn
* SHAP
* LIME

---

# 📄 License

This project is intended for academic and educational purposes.

If you choose to publish or distribute the project, add an appropriate open-source license such as MIT License according to your requirements.

---

## 🌟 Project Highlights

**Explainable Deep Learning + Voice Analysis + Virtual Robotic Assistance**

> An academic research project exploring how deep learning and explainable AI can be combined with voice signals for Parkinson's disease detection.
