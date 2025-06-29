# AI Development Workflow Assignment

## Project Overview
This project demonstrates the application of the AI Development Workflow to real-world problems, including fraud detection in mobile money and hospital patient readmission prediction. The workflow covers problem definition, data strategy, model development, evaluation, deployment, and critical analysis of ethical and practical challenges.

---

## Part 1: Short Answer Questions (Fraud Detection)
- **Problem:** Detect fraudulent transactions in mobile money platforms.
- **Objectives:**
  - Identify suspicious transaction patterns in real time.
  - Minimize financial loss caused by fraud.
  - Improve user trust and platform security.
- **Stakeholders:** Mobile money service providers, end users.
- **KPI:** Fraud Detection Precision.
- **Data Sources:** Transaction logs, user profiles.
- **Bias:** Behavioral bias in underrepresented user groups.
- **Preprocessing:** Handle missing values, normalize amounts, encode categoricals.
- **Model:** Random Forest (robust, interpretable).
- **Evaluation:** Precision, recall; monitor concept drift; address latency/scalability in deployment.

---

## Part 2: Case Study Application (Hospital Readmission)
- **Problem:** Predict patient readmission risk within 30 days of discharge.
- **Objectives:** Reduce readmission rates, improve outcomes.
- **Stakeholders:** Hospital staff, patients, administrators, insurers.
- **Data Strategy:**
  - EHRs, demographics, admission/discharge summaries.
  - Ethical concerns: patient privacy, bias/fairness.
  - Preprocessing: Impute missing values, normalize, one-hot encode, feature engineering.
- **Model:** Gradient Boosting or Random Forest (with justification).
- **Evaluation:** Confusion matrix, precision, recall, F1, ROC-AUC.
- **Deployment:** API integration, compliance (HIPAA), access controls.
- **Optimization:** Cross-validation, regularization to prevent overfitting.

---

## Part 3: Critical Thinking
- **Ethics & Bias:**
  - Biased data can worsen disparities and lead to suboptimal care for underrepresented groups.
  - Mitigation: Collect diverse data, use fairness-aware techniques, monitor with fairness metrics (e.g., demographic parity).
- **Trade-offs:**
  - Interpretability vs. accuracy: Simpler models are more transparent but may be less accurate; complex models may be less explainable.
  - Computational constraints: Limited resources may require lightweight models or model compression.

---

## Part 4: Reflection & Workflow Diagram
- **Reflection:**
  - Most challenging: Balancing fairness, interpretability, and accuracy with limited data and strict privacy requirements.
  - Improvements: More data, stakeholder input, advanced techniques, and explainability tools.
- **Workflow Diagram:**
  - See below for the AI Development Workflow stages.

---

## AI Development Workflow Diagram
1. **Problem Definition** – Define prediction goal and stakeholders.
2. **Data Collection** – Gather EHRs, demographics, historical data.
3. **Data Preprocessing & Feature Engineering** – Clean data, extract features, handle missing values.
4. **Model Selection & Training** – Choose model based on needs and constraints.
5. **Evaluation** – Use metrics like precision, recall, fairness.
6. **Deployment** – Integrate with hospital systems (e.g., via API).

---

## Running the Code
1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # or
   source venv/bin/activate  # On Mac/Linux
   ```
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python Park-2/readmission_risk_predictor.py
   ```
4. (Optional) Use Jupyter Notebook for interactive exploration:
   ```bash
   pip install notebook
   jupyter notebook
   ```

---

## References
- CRISP-DM Framework
- Fairness in Machine Learning (AIF360, Fairlearn)
- Healthcare AI Ethics Guidelines
