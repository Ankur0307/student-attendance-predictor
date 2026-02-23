import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load data
df = pd.read_csv("../data/student_attendance.csv")

# Features & target
X = df.drop(columns=["risk_level", "student_id"])
y = df["risk_level"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
lr = LogisticRegression()
rf = RandomForestClassifier(n_estimators=200, random_state=42)

# Train
lr.fit(X_train_scaled, y_train)
rf.fit(X_train, y_train)

# Evaluate
def evaluate(name, model, X, y):
    preds = model.predict(X)
    print(f"\n{name}")
    print("Accuracy :", accuracy_score(y, preds))
    print("Precision:", precision_score(y, preds))
    print("Recall   :", recall_score(y, preds))
    print("F1 Score :", f1_score(y, preds))

evaluate("Logistic Regression", lr, X_test_scaled, y_test)
evaluate("Random Forest", rf, X_test, y_test)

# Save best model
joblib.dump(rf, "attendance_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ Model saved successfully!")
