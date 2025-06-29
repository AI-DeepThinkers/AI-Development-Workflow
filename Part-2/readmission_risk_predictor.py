"""
readmission_predictor.py
Author: Nicole Rombo
Description: Predicts patient readmission risk within 30 days of discharge.
"""

# ------------------------------
# 1. Problem Scope
# ------------------------------
# Problem: Predict patient readmission risk within 30 days of hospital discharge.
# Objective: Use AI to reduce readmission rates and improve care quality.
# Stakeholders: Hospital administrators, data scientists, IT department, and patients.

# ------------------------------
# 2. Data Strategy
# ------------------------------
# Data Sources (Hypothetical):
# - Electronic Health Records (EHRs)
# - Patient demographics (age, gender, preexisting conditions)
# - Lab results, discharge notes, previous admission history

# Ethical Concerns:
# 1. Patient Privacy - ensure anonymization of data
# 2. Bias in model - unequal performance across demographics

# Preprocessing pipeline
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(df):
    df = df.dropna()
    categorical_cols = ['gender', 'race']
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    scaler = StandardScaler()
    df[['age', 'blood_pressure', 'cholesterol']] = scaler.fit_transform(df[['age', 'blood_pressure', 'cholesterol']])
    return df

# ------------------------------
# 3. Model Development
# ------------------------------
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score

# Hypothetical dataset
np.random.seed(42)
data = pd.DataFrame({
    'age': np.random.randint(20, 80, 100),
    'gender': np.random.choice(['Male', 'Female'], 100),
    'race': np.random.choice(['White', 'Black', 'Asian'], 100),
    'blood_pressure': np.random.randint(90, 180, 100),
    'cholesterol': np.random.randint(150, 300, 100),
    'readmitted': np.random.choice([0, 1], 100)
})

# Preprocess and split
data = preprocess_data(data)
X = data.drop('readmitted', axis=1)
y = data['readmitted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Confusion Matrix:\n", cm)
print("Precision:", precision)
print("Recall:", recall)

# ------------------------------
# 4. Deployment
# ------------------------------
# Steps:
# - Deploy model as a REST API using Flask/FastAPI
# - Integrate API with hospital EHR system UI
# - Set up access controls for security and auditing

# Compliance:
# - Ensure encryption of data at rest and in transit
# - Regular audits for HIPAA compliance
# - Role-based access to sensitive patient data

# ------------------------------
# 5. Optimization (Overfitting)
# ------------------------------
# Method: Cross-validation and model regularization
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model, X, y, cv=5)
print("Cross-validation scores:", cv_scores)
print("Average CV Score:", np.mean(cv_scores))
