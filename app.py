
from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# Try loading real model
MODEL_PATH = 'model/resume_svm_model.pkl'
SCALER_PATH = 'model/resume_scaler.pkl'
ENCODER_PATH = 'model/resume_encoder.pkl'

use_dummy = False

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(ENCODER_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
else:
    use_dummy = True
    print("⚠️ Using dummy model (no .pkl files found)")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    skills = request.form['skills'].strip().lower()
    experience = float(request.form['experience'])
    cgpa = float(request.form['cgpa'])
    projects = request.form['projects'].strip().lower()

    if not use_dummy:
        encoded = encoder.transform([[skills, projects]])
        skill_enc = encoded[0][0]
        project_enc = encoded[0][1]

        features = np.array([[skill_enc, experience, cgpa, project_enc]])
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]
        score_raw = model.decision_function(features_scaled)[0]
        score = max(0, min(100, (score_raw + 1) * 50))
    else:
        # Dummy logic
        score = (cgpa * 10) + (experience * 5)
        score = min(100, score)

        if score > 75:
            prediction = 1
        elif score > 50:
            prediction = 1
        else:
            prediction = 0

    if prediction == 1 and score >= 75:
        status = "Selected"
    elif prediction == 1:
        status = "Waitlisted"
    else:
        status = "Rejected"

    return render_template('result.html', score=round(score,2), status=status)

if __name__ == '__main__':
    app.run(debug=True)
