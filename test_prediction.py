
import pandas as pd
import joblib

# Load trained Random Forest model
model = joblib.load("models/random_forest.pkl")

# Load dataset
df = pd.read_csv("data/student-mat.csv", sep=";")

# Select one student record
student = df.drop(columns=["G3"]).iloc[[0]]

# Actual grade
actual_grade = df["G3"].iloc[0]

# Predict grade
predicted_grade = model.predict(student)[0]

print("STUDENT PERFORMANCE PREDICTION")
print("--------------------------------")

print("Actual Final Grade:", actual_grade)
print("Predicted Final Grade:", round(predicted_grade, 2))

print("Prediction Error:", round(abs(actual_grade - predicted_grade), 2))
