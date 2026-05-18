import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Example sepsis dataset
# Replace with PhysioNet sepsis dataset

sepsis_df = pd.read_csv("sepsis_icu_synthetic.csv")

# The CSV uses 'sepsis_label' (lowercase with underscore)
X = sepsis_df.drop("sepsis_label", axis=1)
y = sepsis_df["sepsis_label"]

# Encode categorical object columns for XGBoost
object_cols = X.select_dtypes(include=['object']).columns.tolist()
if object_cols:
    X = pd.get_dummies(X, columns=object_cols, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = XGBClassifier()

model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Sepsis Accuracy: {accuracy:.2f}")

joblib.dump(model, "sepsis_model.pkl")

print("Sepsis model saved!")