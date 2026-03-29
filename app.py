from flask import Flask, render_template, request, send_file
import numpy as np
import pandas as pd
import joblib
import os

app = Flask(__name__)

# ================= MODEL PATH =================
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
    print("⚠️ Using dummy model")

# ================= HOME =================
@app.route('/')
def home():
    return render_template('index.html')

# ================= MANUAL INPUT =================
@app.route('/predict', methods=['POST'])
def predict():
    skills = request.form['skills'].strip().lower()
    experience = float(request.form['experience'])
    cgpa = float(request.form['cgpa'])
    projects = request.form['projects'].strip().lower()

    if not use_dummy:
        encoded = encoder.transform([[skills, projects]])
        features = np.array([[encoded[0][0], experience, cgpa, encoded[0][1]]])
        features_scaled = scaler.transform(features)

        score_raw = model.decision_function(features_scaled)[0]
        score = max(0, min(100, (score_raw + 1) * 50))
    else:
        score = min(100, (cgpa * 10) + (experience * 5))

    status = "Selected" if score >= 75 else "Waitlisted" if score >= 50 else "Rejected"

    return render_template('result.html', score=round(score, 2), status=status)

# ================= CSV INPUT =================
@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    file = request.files['file']

    if not file:
        return "No file uploaded"

    df = pd.read_csv(file)

    # Clean column names
    df.columns = df.columns.str.strip().str.lower()

    # Rename columns (flexible input support)
    df = df.rename(columns={
        'skill': 'skills',
        'experience': 'years_of_experience',
        'project': 'projects'
    })

    selected_candidates = []

    for _, row in df.iterrows():
        try:
            skills = str(row['skills']).lower().strip()
            exp = float(row['years_of_experience'])
            cgpa = float(row['cgpa'])
            projects = str(row['projects']).lower().strip()

            if not use_dummy:
                encoded = encoder.transform([[skills, projects]])
                features = np.array([[encoded[0][0], exp, cgpa, encoded[0][1]]])
                features_scaled = scaler.transform(features)

                score_raw = model.decision_function(features_scaled)[0]
                score = max(0, min(100, (score_raw + 1) * 50))
            else:
                score = min(100, (cgpa * 10) + (exp * 5))

            if score >= 75:
                selected_candidates.append({
                    "skills": skills,
                    "experience": exp,
                    "cgpa": cgpa,
                    "projects": projects,
                    "score": round(score, 2)
                })

        except:
            continue

    # ✅ SAVE CSV (IMPORTANT)
    output_file = "selected_candidates.csv"
    pd.DataFrame(selected_candidates).to_csv(output_file, index=False)

    return render_template('csv_result.html', candidates=selected_candidates)

# ================= DOWNLOAD =================
@app.route('/download')
def download():
    file_path = "selected_candidates.csv"

    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return "No file available to download"

# ================= RUN =================
if __name__ == '__main__':
    app.run(debug=True)