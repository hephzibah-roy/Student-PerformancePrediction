# 🎓 Student Performance Prediction System

> **Course Outcome:** CO5 – Solve real life problems using appropriate machine learning models and evaluate the performance measures  
> **Topic:** Student Performance Prediction  
> **Algorithms:** Logistic Regression & Decision Tree Classifier  
> **Interactive App:** Streamlit Web Application ([http://localhost:8501](http://localhost:8501))

---

## 📸 Application Preview

![Streamlit Web Application Preview](screenshots/app.png)

---

## 📌 1. Problem Statement

Academic attrition and student failure represent persistent challenges across academic institutions. Traditional counseling identifies struggling students reactively—often after midterm or final examinations have already been failed. 

This project implements an end-to-end Machine Learning pipeline that forecasts student academic performance early in the academic term, classifying students into **Pass / On Track** versus **Fail / At Academic Risk**. By comparing a linear probabilistic model (**Logistic Regression**) against a non-parametric hierarchical model (**Decision Tree Classifier**), this system provides educators with both calibrated probability forecasts and actionable intervention thresholds.

---

## 📂 2. Dataset Information

- **Dataset Name:** Student Academic Performance & Behavioral Dataset
- **Source:** Modeled after benchmarks from the *UCI Machine Learning Repository Student Performance Dataset* and Kaggle educational analytics studies.
- **Number of Records:** **1,500 students** (1,200 training samples, 300 testing samples via stratified 80/20 split)
- **Number of Features:** 8 input features (5 continuous, 3 categorical) + 1 binary target variable.

### Feature Dictionary

| Feature Name | Type | Range / Values | Description |
|---|---|---|---|
| `study_hours_per_week` | Continuous | 2.0 – 40.0 hrs | Self-directed independent study hours per week |
| `attendance_rate` | Continuous | 45.0% – 100.0% | Percentage of lectures and laboratories attended |
| `previous_exam_score` | Continuous | 30.0 – 100.0 | Marks obtained in prior semester/prerequisite exam |
| `assignment_completion_rate` | Continuous | 20.0% – 100.0% | Percentage of coursework submissions submitted on time |
| `sleep_hours` | Continuous | 4.0 – 10.0 hrs | Daily average sleep duration |
| `extracurricular_activities` | Categorical | `Yes`, `No` | Active involvement in student organizations or sports |
| `tutoring_classes` | Categorical | `Yes`, `No` | Enrolled in remedial tutoring or academic mentoring |
| `parental_education_level` | Categorical | `High School`, `College`, `Postgraduate` | Highest completed parent/guardian education level |
| **`passed` (Target)** | **Binary Class** | **0 (Fail / Risk), 1 (Pass)** | **962 Passed (64.13%), 538 At Risk (35.87%)** |

---

## ⚙️ 3. Machine Learning Pipeline & Preprocessing

1. **Integrity Verification:** Checked for missing values (`df.isnull().sum()`) and duplicate rows (`df.duplicated().sum()`). Found **0 missing values** and **0 duplicate entries**.
2. **Categorical Encoding:** One-Hot Encoded categorical attributes (`extracurricular_activities`, `tutoring_classes`, `parental_education_level`) using `OneHotEncoder(handle_unknown='ignore')`.
3. **Pipeline Standardization (Zero Data Leakage):** Scaled numerical continuous attributes using `StandardScaler` strictly inside `sklearn.pipeline.Pipeline`, ensuring no test data leaked into the training statistics.
4. **Stratified Splitting:** 80% Train (1,200 samples) and 20% Test (300 samples) using `train_test_split(..., stratify=y, random_state=42)`.
5. **5-Fold Cross-Validation:** The final winning model was selected strictly based on **5-fold cross-validation on the training set**, preventing test set bias.

---

## 📊 4. Model Performance & Evaluation

### Comparison Table: 5-Fold Cross-Validation vs. Held-Out Test Set

| Algorithm | Evaluation Stage | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|---|
| **Logistic Regression** | **5-Fold CV (Train)** | **90.25% ±1.07%** | **91.81% ±0.50%** | **93.12% ±2.00%** | **92.45% ±0.91%** | **0.9637 ±0.008** |
| Decision Tree Classifier | 5-Fold CV (Train) | 79.42% ±1.70% | 83.23% ±0.69% | 85.06% ±2.99% | 84.11% ±1.56% | 0.8514 ±0.014 |
| **Logistic Regression** | **Held-Out Test (N=300)**| **89.67%** | **90.45%** | **93.75%** | **92.07%** | **0.9659** |
| Decision Tree Classifier | Held-Out Test (N=300) | 82.67% | 85.35% | 88.02% | 86.67% | 0.9048 |

> **Winning Model:** **Logistic Regression** was selected as the final production model (`model.pkl`) based on its superior 5-Fold Cross-Validation F1-score (**92.45%** vs 84.11%) and ROC-AUC (**0.9637** vs 0.8514).

---

## 📈 5. Visual Diagnostics & Plots

### Class Distribution & Feature Correlations

| Class Distribution | Correlation Heatmap |
|---|---|
| ![Class Distribution](plots/class_distribution.png) | ![Correlation Heatmap](plots/correlation_heatmap.png) |
| *Figure 1: Distribution of 962 Passing and 538 At-Risk students.* | *Figure 2: Multivariate correlation with final academic outcome.* |

### Confusion Matrices & ROC Curves

| Confusion Matrices (Test Set) | ROC Curves |
|---|---|
| ![Confusion Matrix](plots/confusion_matrix.png) | ![ROC Curve](plots/roc_curve.png) |
| *Figure 3: Test confusion matrices showing TN, FP, FN, and TP.* | *Figure 4: ROC curve highlighting Logistic Regression's 0.9659 AUC.* |

### Model Performance Comparison Bar Chart

![Model Comparison Bar Chart](plots/model_comparison.png)
*Figure 5: Side-by-side metric comparison across Accuracy, Precision, Recall, F1, and ROC-AUC.*

---

## 🔍 6. Error Analysis & Domain Significance

In student performance forecasting, prediction errors carry fundamentally asymmetric consequences:

- **False Negatives / Type I Errors (Predicted 'Pass', Actual 'Fail'):**
  - Logistic Regression made **12 errors**; Decision Tree made **23 errors**.
  - *Pedagogical Impact:* **Catastrophic false reassurance**. An academically vulnerable student is overlooked, misses timely tutoring/advising, and fails the course.
- **False Positives / Type II Errors (Predicted 'Fail', Actual 'Pass'):**
  - Logistic Regression made **19 errors**; Decision Tree made **29 errors**.
  - *Pedagogical Impact:* **Benign false alarm**. The student is offered extra assistance that they may not strictly need, but causes no academic harm.
- **Which Metric Matters Most:**
  - **Recall ($\frac{TP}{TP+FN}$) and F1-Score are the most critical metrics**. High recall guarantees that failing students are detected and referred to remedial support before semester deadlines.

---

## 🚀 7. How to Run the Application

### Prerequisites
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Launch the Streamlit App
Launch the interactive web interface:
```bash
streamlit run app.py
```
*(Or double-click `run_app.bat` on Windows)*

The application opens automatically at **`http://localhost:8501`**.

### Execute the Machine Learning Pipeline Script
To run the full Step 3 training pipeline, output cross-validation tables, and regenerate all plots:
```bash
python main.py
```

---

## 📦 8. Git Repository Structure

```
student-performance-prediction/
├── app.py                      # Main Streamlit web application
├── streamlit_app.py            # Streamlit concise entrypoint
├── model.pkl                   # Winning trained model (Logistic Regression Pipeline)
├── dataset.csv                 # Complete 1,500 record student dataset
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive project documentation
├── report.pdf                  # 5-page formal academic report (A4 format)
├── main.py                     # Step 3 end-to-end ML pipeline script
├── run_app.bat                 # One-click Windows application launcher
├── plots/                      # Required high-resolution visualization figures
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── model_comparison.png
├── screenshots/
│   └── app.png                 # Streamlit UI screenshot embedded in README
├── data/
│   ├── student_performance.csv
│   └── sample_test_batch.csv
└── models/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    └── pipeline_summary.json
```

---

## 🎯 9. Course Outcome (CO5) Mapping Summary

This project satisfies all requirements for **Course Outcome 5**:
1. **Real-Life Problem:** Early identification of student academic failure risk.
2. **Appropriate ML Models:** Evaluated linear probabilistic modeling (Logistic Regression) vs. non-parametric hierarchical decision trees (Decision Tree).
3. **Rigorous Validation:** Evaluated 5-fold cross-validation on training partitions to select the superior model objectively.
4. **Comprehensive Evaluation:** Evaluated Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrices with domain error analysis.
5. **Interactive Deployment:** Operationalized into an interactive Streamlit web dashboard.
