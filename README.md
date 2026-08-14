# 🧠 Explainable Deep Learning-Based Parkinson's Disease Detection with Virtual Robotic Assistance Using Voice Signals

## 📌 Project Overview

This project is an AI-based system developed to detect Parkinson's disease from human voice recordings using **Deep Learning and Audio Signal Processing**.

The system takes a voice recording as input, extracts meaningful audio features using **Mel-Frequency Cepstral Coefficients (MFCC)**, and uses a hybrid **Convolutional Neural Network (CNN) + Bidirectional Long Short-Term Memory (BiLSTM)** model to classify the voice signal as:

- Healthy
- Parkinson's

The project also incorporates **Explainable Artificial Intelligence (XAI)** using **SHAP and LIME** to improve the interpretability of the model's predictions.

A **Virtual Robotic Assistant** is also included as part of the proposed system to provide an interactive and user-friendly experience.

> ⚠️ This project is developed for academic and research purposes only. It is not intended to replace professional medical diagnosis or clinical evaluation.

---

# 🎯 Objectives

The main objectives of this project are:

- To detect Parkinson's disease using voice signals.
- To preprocess and analyze human voice recordings.
- To extract useful audio features using MFCC.
- To develop a CNN + Bidirectional LSTM deep learning model.
- To classify voice recordings as Healthy or Parkinson's.
- To provide prediction confidence.
- To integrate Explainable AI techniques.
- To understand the factors contributing to model predictions.
- To provide a Virtual Robotic Assistant.
- To develop a user-friendly web interface.
- To provide an AI-assisted prediction and reporting system.

---

# 💡 Problem Statement

Parkinson's disease is a progressive neurological disorder that can affect speech and voice characteristics.

Changes in voice characteristics may contain useful information that can be analyzed using Artificial Intelligence and Machine Learning techniques.

This project explores a **voice-based AI approach** for identifying patterns associated with Parkinson's disease.

The system processes voice recordings, extracts MFCC features, and uses a deep learning model to classify the input into Healthy or Parkinson's.

The objective is to develop an academic research prototype that demonstrates how voice signal processing and deep learning can be combined for healthcare-related applications.

---

# 🎤 Why Voice Signals?

Parkinson's disease can affect several characteristics of human speech and voice.

Possible changes may include:

- Pitch variation
- Vocal stability
- Frequency characteristics
- Voice intensity
- Speech timing
- Acoustic characteristics
- Other voice-related patterns

Voice signals can therefore be processed into numerical representations and analyzed using deep learning models.

This project focuses on voice signals because voice recording provides a **non-invasive and accessible input method** for research.

---

# 🏗️ Overall System Architecture

```text
                           USER
                            │
                            ▼
               ┌────────────────────────┐
               │ Voice Recording /       │
               │ Audio File Upload       │
               └────────────┬───────────┘
                            │
                            ▼
               ┌────────────────────────┐
               │ Frontend Web Interface │
               │ HTML / CSS / JavaScript │
               └────────────┬───────────┘
                            │
                       HTTP Request
                            │
                            ▼
               ┌────────────────────────┐
               │ Flask Backend API      │
               └────────────┬───────────┘
                            │
                            ▼
               ┌────────────────────────┐
               │ Audio Preprocessing    │
               └────────────┬───────────┘
                            │
                            ▼
               ┌────────────────────────┐
               │ MFCC Feature Extraction│
               └────────────┬───────────┘
                            │
                            ▼
               ┌────────────────────────┐
               │ Feature Preparation    │
               └────────────┬───────────┘
                            │
                            ▼
               ┌────────────────────────┐
               │ CNN Feature Extraction │
               └────────────┬───────────┘
                            │
                            ▼
               ┌────────────────────────┐
               │ Bidirectional LSTM     │
               │ Sequence Learning      │
               └────────────┬───────────┘
                            │
                            ▼
               ┌────────────────────────┐
               │ Classification Layer   │
               └────────────┬───────────┘
                            │
                       ┌────┴────┐
                       ▼         ▼
                   HEALTHY   PARKINSON'S
                       │         │
                       └────┬────┘
                            ▼
               ┌────────────────────────┐
               │ Prediction + Confidence│
               └────────────┬───────────┘
                            │
                  ┌─────────┴──────────┐
                  ▼                    ▼
          ┌───────────────┐    ┌────────────────┐
          │ Explainable AI│    │ Virtual Robotic│
          │ SHAP / LIME   │    │ Assistant      │
          └───────┬───────┘    └───────┬────────┘
                  │                    │
                  └─────────┬──────────┘
                            ▼
               ┌────────────────────────┐
               │ Result Display         │
               │ Prediction             │
               │ Confidence             │
               │ Explanation            │
               └────────────┬───────────┘
                            │
                            ▼
               ┌────────────────────────┐
               │ Report Generation      │
               └────────────────────────┘
# 📈 Model Performance and Accuracy

The trained **CNN + Bidirectional LSTM** model was evaluated using the test dataset.

## 🎯 Test Accuracy

**94.27%**

The model achieved an overall test accuracy of **94.27%**, showing strong classification performance on the evaluated voice dataset.

## 📊 Confusion Matrix

```text
                    Predicted
                 Healthy  Parkinson's

Actual Healthy       108        7

Actual Parkinson's    6       106

## **🚀 Future Enhancements**

- Real-time voice analysis
- Larger and more diverse datasets
- Improved Explainable AI (XAI) visualization
- Cloud deployment
- Mobile application
- Multilingual Virtual Robotic Assistant
- Improved prediction accuracy
- Real-time prediction and reporting

## **⚠️ Disclaimer**

This project is developed for **academic and research purposes only** and is **not a replacement for professional medical diagnosis or clinical evaluation**.

## **👩‍💻 Author**

**Arshiya Nasirin**  
B.E. CSE – Artificial Intelligence & Machine Learning

⭐ If you find this project useful, consider starring the repository!
