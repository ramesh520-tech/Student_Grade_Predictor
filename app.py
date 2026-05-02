from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("student_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        maths = int(request.form["Maths"])
        english = int(request.form["English"])
        science = int(request.form["Science"])
        telugu = int(request.form["Telugu"])
        biology = int(request.form["Biology"])

    
        new_student = pd.DataFrame(
            [[maths, english, science, telugu, biology]],
            columns=["Maths","English","Science","Telugu","Biology"]
        )

    
        prediction = model.predict(new_student)[0]

        total = maths + english + science + telugu + biology

        return render_template(
            "index.html",
            result=f"Total Marks: {total} | Predicted Grade: {prediction}"
        )

    except Exception as e:
        return render_template("index.html", result=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)