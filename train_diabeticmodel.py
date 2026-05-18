import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Load dataset
# Download dataset from:
# https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

# Rename your file as:
# diabetes_dataset.csv
data = pd.read_csv('diabetes.csv')
# Features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
# Model
model = XGBClassifier()

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy:.2f}")

# Save model
joblib.dump(model, "diabetes_model.pkl")
print("Model saved successfully!")