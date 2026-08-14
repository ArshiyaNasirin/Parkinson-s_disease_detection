import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# =========================
# 1. Load data
# =========================

X = np.load("X_audio.npy")
y = np.load("y_audio.npy")

print("Original X shape:", X.shape)
print("Original y shape:", y.shape)


# =========================
# 2. Change shape
# =========================

# Original:
# samples × MFCC × time

# Required:
# samples × time × MFCC

X = np.transpose(X, (0, 2, 1))

print("Model input shape:", X.shape)


# =========================
# 3. Normalize features
# =========================

mean = np.mean(X, axis=(0, 1), keepdims=True)
std = np.std(X, axis=(0, 1), keepdims=True)

X = (X - mean) / (std + 1e-8)

print("Normalization completed")


# =========================
# 4. Train/Test split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================
# 5. Class weights
# =========================

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))

print("Class weights:", class_weights)


# =========================
# 6. CNN + Bidirectional LSTM
# =========================

model = Sequential([

    Conv1D(
        filters=64,
        kernel_size=5,
        activation="relu",
        input_shape=(X_train.shape[1], X_train.shape[2])
    ),

    BatchNormalization(),

    MaxPooling1D(pool_size=2),

    Dropout(0.3),

    Conv1D(
        filters=128,
        kernel_size=3,
        activation="relu"
    ),

    BatchNormalization(),

    MaxPooling1D(pool_size=2),

    Dropout(0.3),

    Bidirectional(
        LSTM(64, return_sequences=False)
    ),

    Dropout(0.4),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.3),

    Dense(
        1,
        activation="sigmoid"
    )
])


# =========================
# 7. Compile
# =========================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# =========================
# 8. Callbacks
# =========================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=0.00001
)


# =========================
# 9. Train
# =========================

history = model.fit(

    X_train,
    y_train,

    validation_split=0.20,

    epochs=40,

    batch_size=32,

    class_weight=class_weights,

    callbacks=[
        early_stopping,
        reduce_lr
    ],

    verbose=1
)


# =========================
# 10. Evaluate
# =========================

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nTest Accuracy:", test_accuracy)


# =========================
# 11. Prediction
# =========================

probabilities = model.predict(X_test)

y_pred = (
    probabilities >= 0.5
).astype(int).flatten()


# =========================
# 12. Classification Report
# =========================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Healthy",
            "Parkinson's"
        ],
        zero_division=0
    )
)


# =========================
# 13. Confusion Matrix
# =========================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# =========================
# 14. Save model
# =========================

model.save(
    "models/parkinsons_cnn_lstm.keras"
)

print(
    "\nImproved CNN-LSTM model saved successfully!"
)