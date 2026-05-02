import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib


data = {
    "Maths":    [95, 85, 75, 65, 55, 45, 35],
    "English":  [90, 80, 70, 60, 50, 40, 30],
    "Science":  [92, 82, 72, 62, 52, 42, 32],
    "Telugu":   [94, 84, 74, 64, 54, 44, 34],
    "Biology":  [93, 83, 73, 63, 53, 43, 33],
    "Grade":    ["A", "A", "B", "B", "C", "Fail", "Fail"]
}

df = pd.DataFrame(data)

X = df[["Maths","English","Science","Telugu","Biology"]]
y = df["Grade"]

model = DecisionTreeClassifier()
model.fit(X, y)

joblib.dump(model, "student_model.pkl")

print("✅ Model trained and saved as student_model.pkl")