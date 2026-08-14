import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
data = pd.read_csv("dataset/parkinsons.data")

# 2. Separate features and target
X = data.drop(columns=["name", "status"])
y = data["status"]

# 3. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Create the Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# 5. Train the model
model.fit(X_train, y_train)

# 6. Make predictions
y_pred = model.predict(X_test)

# 7. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Save the trained model
joblib.dump(model, "models/parkinsons_model.pkl")

print("\nModel saved successfully!")

import shap
import matplotlib.pyplot as plt

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

# Calculate SHAP values for test data
shap_values = explainer.shap_values(X_test)

# Show feature importance
shap.summary_plot(shap_values, X_test)

# Prediction function
def predict_parkinsons(features):
    prediction = model.predict([features])[0]

    if prediction == 1:
        return "Parkinson's Disease"
    else:
        return "Healthy"


# Example prediction using one test sample
sample = X_test.iloc[0].tolist()

result = predict_parkinsons(sample)

print("\nPrediction:", result)