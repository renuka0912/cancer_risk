from flask import Flask, render_template, request
from datetime import datetime, timedelta
import joblib

app = Flask(__name__)

# Load models and label encoders
model_breast = joblib.load("breast_model.pkl")
model_cervical = joblib.load("cervical_model.pkl")
model_colorectal = joblib.load("colorectal_model.pkl")

le_breast = joblib.load("le_breast.pkl")
le_cervical = joblib.load("le_cervical.pkl")
le_colorectal = joblib.load("le_colorectal.pkl")

def model_predict(age, gender, family_history, lifestyle):
    gender_num = 0 if gender.lower() == "male" else 1
    has_fh = 0 if family_history.strip().lower() == "none" else 1
    is_smoker = 1 if "smoker" in lifestyle.lower() else 0
    is_obese = 1 if "obese" in lifestyle.lower() else 0
    uses_alcohol = 1 if "alcohol" in lifestyle.lower() else 0

    X = [[age, gender_num, has_fh, is_smoker, is_obese, uses_alcohol]]

    breast_pred = le_breast.inverse_transform(model_breast.predict(X))[0]
    cervical_pred = le_cervical.inverse_transform(model_cervical.predict(X))[0]
    colorectal_pred = le_colorectal.inverse_transform(model_colorectal.predict(X))[0]

    return breast_pred, cervical_pred, colorectal_pred

def next_screening_date(last_date_str, interval_years=1):
    if not last_date_str:
        return "Not Available"
    try:
        last_date = datetime.strptime(last_date_str, "%Y-%m-%d")
        return (last_date + timedelta(days=365 * interval_years)).date()
    except ValueError:
        return "Invalid date"

@app.route('/')
def index():
    return render_template("form.html")

@app.route('/submit', methods=["POST"])
def submit():
    name = request.form.get('name')
    age = int(request.form.get('age'))
    gender = request.form.get('gender')
    family_history = request.form.get('family_history', 'None')
    lifestyle = request.form.get('lifestyle', '')

    last_breast = request.form.get('last_breast')
    last_cervical = request.form.get('last_cervical')
    last_colorectal = request.form.get('last_colorectal')

    breast_risk, cervical_risk, colorectal_risk = model_predict(age, gender, family_history, lifestyle)

    breast_next = next_screening_date(last_breast, 1)
    cervical_next = next_screening_date(last_cervical, 2)
    colorectal_next = next_screening_date(last_colorectal, 3)

    return render_template("result.html", name=name,
                           breast_risk=breast_risk,
                           cervical_risk=cervical_risk,
                           colorectal_risk=colorectal_risk,
                           breast_next=breast_next,
                           cervical_next=cervical_next,
                           colorectal_next=colorectal_next)

if __name__ == "__main__":
    app.run(debug=True)
