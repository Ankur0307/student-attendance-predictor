import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Load model
model = joblib.load("attendance_model.pkl")

# Load data
df = pd.read_csv("../data/student_attendance.csv")
X = df.drop(columns=["risk_level", "student_id"])
y = df["risk_level"]

# Predict
preds = model.predict(X)

# Confusion Matrix
cm = confusion_matrix(y, preds)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – Attendance Prediction")
plt.show()
