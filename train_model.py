import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

# =========================
# 1. LOAD DATASET
# =========================
df = pd.read_csv('dataset/ece_student_training_dataset.csv')

# =========================
# 2. CLEAN COLUMN NAMES
# =========================
df.columns = df.columns.str.strip().str.lower()

print("✅ Columns in dataset:", df.columns)

# =========================
# 3. RENAME COLUMNS (AUTO FIX)
# =========================
df = df.rename(columns={
    'skill': 'skills',
    'experience': 'years_of_experience',
    'project': 'projects',
    'result': 'label'
})

# =========================
# 4. CHECK REQUIRED COLUMNS
# =========================
required_columns = ['skills', 'years_of_experience', 'cgpa', 'projects', 'label']

for col in required_columns:
    if col not in df.columns:
        raise Exception(f"❌ Missing column: {col}")

print("✅ All required columns present!")

# =========================
# 5. DROP NAME COLUMN
# =========================
df = df.drop('student_name', axis=1, errors='ignore')

# =========================
# 6. ENCODE CATEGORICAL DATA
# =========================
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

df['skills'] = df['skills'].astype(str).str.lower().str.strip()
df['projects'] = df['projects'].astype(str).str.lower().str.strip()

df[['skills', 'projects']] = encoder.fit_transform(
    df[['skills', 'projects']]
)


# =========================
# 7. CONVERT TO NUMERIC (CRITICAL)
# =========================
df['years_of_experience'] = pd.to_numeric(df['years_of_experience'], errors='coerce')
df['cgpa'] = pd.to_numeric(df['cgpa'], errors='coerce')

# Convert label also (safety)
df['label'] = pd.to_numeric(df['label'], errors='coerce')

# Drop invalid rows
df = df.dropna()

# =========================
# 8. FEATURES & LABEL
# =========================
X = df[['skills', 'years_of_experience', 'cgpa', 'projects']]
y = df['label']

# =========================
# 9. HANDLE IMBALANCE (SAFE SMOTE)
# =========================
if len(y.unique()) > 1:
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)
else:
    print("⚠️ Only one class present — skipping SMOTE")

# =========================
# 10. SCALING
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 11. TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# =========================
# 12. TRAIN MODEL
# =========================
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# =========================
# 13. SAVE FILES
# =========================
joblib.dump(model, 'resume_svm_model.pkl')
joblib.dump(scaler, 'resume_scaler.pkl')
joblib.dump(encoder, 'resume_encoder.pkl')

print("\n🎉 MODEL TRAINING SUCCESSFUL!")
print("✅ resume_svm_model.pkl created")
print("✅ resume_scaler.pkl created")
print("✅ resume_encoder.pkl created")