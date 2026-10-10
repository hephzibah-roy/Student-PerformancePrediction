"""
Student Performance Prediction Web Application
Built with Streamlit, Scikit-Learn, and Python
Algorithms: Logistic Regression & Decision Tree
"""

import os
import json
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")
DATASET_PATH = os.path.join(DATA_DIR, "student_performance.csv")
SAMPLE_BATCH_PATH = os.path.join(DATA_DIR, "sample_test_batch.csv")

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .pass-badge {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 1.1rem;
        display: inline-block;
    }
    .fail-badge {
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 1.1rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_metadata():
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            return json.load(f)
    return None

@st.cache_data
def load_dataset():
    if os.path.exists(DATASET_PATH):
        return pd.read_csv(DATASET_PATH)
    return None

@st.cache_resource
def load_models():
    import joblib
    models = {}
    lr_path = os.path.join(MODELS_DIR, "logistic_regression.joblib")
    dt_path = os.path.join(MODELS_DIR, "decision_tree.joblib")
    if os.path.exists(lr_path):
        models["Logistic Regression"] = joblib.load(lr_path)
    if os.path.exists(dt_path):
        models["Decision Tree"] = joblib.load(dt_path)
    return models

metadata = load_metadata()
dataset = load_dataset()
models = load_models()

# App Header
st.markdown('<div class="main-title">🎓 Student Performance Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Predict academic outcomes (Pass / Academic Risk) using Machine Learning: <b>Logistic Regression</b> & <b>Decision Tree</b></div>', unsafe_allow_html=True)

# Main Navigation Tabs
tab_predict, tab_compare, tab_insights, tab_batch = st.tabs([
    "🔮 Predict Student Outcome",
    "📊 Model Performance & Comparison",
    "🧠 Explainability & Feature Importance",
    "📁 Batch Prediction & Data Explorer"
])

# ==============================================================================
# TAB 1: PREDICT STUDENT OUTCOME
# ==============================================================================
with tab_predict:
    st.subheader("Student Academic Profile & Input Features")
    st.write("Adjust the student's academic and behavioral indicators below to predict their performance outcome.")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("#### 📚 Academic & Study Indicators")
        study_hours = st.slider("Weekly Study Hours", min_value=1.0, max_value=40.0, value=16.0, step=0.5,
                                help="Number of hours dedicated to independent studying per week.")
        attendance = st.slider("Attendance Rate (%)", min_value=40.0, max_value=100.0, value=82.0, step=1.0,
                               help="Percentage of lectures and classes attended.")
        previous_score = st.slider("Previous Exam Score", min_value=0.0, max_value=100.0, value=70.0, step=1.0,
                                   help="Score achieved in the previous semester/term exam.")
        assignment_rate = st.slider("Assignment Completion Rate (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0,
                                    help="Percentage of coursework and assignments submitted on time.")

    with col2:
        st.markdown("#### 🌟 Lifestyle & Support Indicators")
        sleep_hours = st.slider("Nightly Sleep Hours", min_value=4.0, max_value=10.0, value=7.0, step=0.5,
                                help="Average hours of sleep per night.")
        
        c_sub1, c_sub2 = st.columns(2)
        with c_sub1:
            extracurricular = st.radio("Extracurricular Activities", options=["Yes", "No"], index=0,
                                       help="Participation in clubs, sports, or student activities.")
        with c_sub2:
            tutoring = st.radio("Attending Tutoring / Extra Help", options=["No", "Yes"], index=0,
                                help="Regular participation in tutoring or academic coaching.")
            
        parental_edu = st.selectbox(
            "Parental Education Level",
            options=["High School", "College", "Postgraduate"],
            index=1,
            help="Highest completed education level of parents/guardians."
        )

        st.markdown("#### ⚙️ Algorithm Selection")
        selected_model_option = st.selectbox(
            "Choose Algorithm for Prediction:",
            options=["Compare Both Algorithms", "Logistic Regression", "Decision Tree"]
        )

    st.divider()

    # Create input dictionary
    input_data = {
        "study_hours_per_week": study_hours,
        "attendance_rate": attendance,
        "previous_exam_score": previous_score,
        "assignment_completion_rate": assignment_rate,
        "sleep_hours": sleep_hours,
        "extracurricular_activities": extracurricular,
        "tutoring_classes": tutoring,
        "parental_education_level": parental_edu
    }
    input_df = pd.DataFrame([input_data])

    if st.button("🚀 Predict Performance", type="primary", use_container_width=True):
        if not models:
            st.error("No models found! Please run the training script first.")
        else:
            st.markdown("### 🎯 Prediction Results")

            def render_prediction_card(model_name, pipeline):
                pred = int(pipeline.predict(input_df)[0])
                prob_pass = float(pipeline.predict_proba(input_df)[0][1])
                prob_pct = round(prob_pass * 100, 1)

                is_pass = pred == 1
                status_text = "PASSED / ON TRACK" if is_pass else "AT RISK / LIKELY TO FAIL"
                badge_class = "pass-badge" if is_pass else "fail-badge"

                # Risk assessment
                if prob_pass >= 0.75:
                    risk_desc = "🟢 Low Academic Risk"
                    action = "Student is performing strongly. Encourage consistent study schedule."
                elif prob_pass >= 0.50:
                    risk_desc = "🟡 Moderate Risk (Borderline Pass)"
                    action = "Borderline status. Advise increasing study hours by 2-3 hrs/week."
                elif prob_pass >= 0.30:
                    risk_desc = "🟠 Elevated Academic Risk"
                    action = "Requires intervention. Recommend tutoring and attendance check-in."
                else:
                    risk_desc = "🔴 High Risk of Failure"
                    action = "Urgent academic counseling and structured remediation required."

                st.markdown(f"""
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 20px;">
                    <div style="font-size: 1.2rem; font-weight: 700; color: #1E293B; margin-bottom: 8px;">
                        🤖 Algorithm: {model_name}
                    </div>
                    <div style="margin-bottom: 14px;">
                        <span class="{badge_class}">{status_text}</span>
                    </div>
                    <div style="font-size: 0.95rem; color: #475569; margin-bottom: 4px;">
                        <b>Passing Probability:</b> {prob_pct}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.progress(prob_pass, text=f"Pass Probability: {prob_pct}%")
                st.markdown(f"**Risk Level:** {risk_desc}")
                st.info(f"💡 **Recommendation:** {action}")

            if selected_model_option == "Compare Both Algorithms":
                res_col1, res_col2 = st.columns(2, gap="medium")
                with res_col1:
                    render_prediction_card("Logistic Regression", models["Logistic Regression"])
                with res_col2:
                    render_prediction_card("Decision Tree", models["Decision Tree"])
            else:
                render_prediction_card(selected_model_option, models[selected_model_option])

# ==============================================================================
# TAB 2: MODEL PERFORMANCE & COMPARISON
# ==============================================================================
with tab_compare:
    st.subheader("Model Evaluation & Comparison Dashboard")
    st.write("Compare the performance of Logistic Regression vs. Decision Tree on the independent test dataset (300 students).")

    if metadata and "models" in metadata:
        m_data = metadata["models"]
        
        # Summary Metric Cards
        comp_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC Score"],
            "Logistic Regression": [
                f"{m_data['Logistic Regression']['metrics']['accuracy']*100:.2f}%",
                f"{m_data['Logistic Regression']['metrics']['precision']*100:.2f}%",
                f"{m_data['Logistic Regression']['metrics']['recall']*100:.2f}%",
                f"{m_data['Logistic Regression']['metrics']['f1_score']*100:.2f}%",
                f"{m_data['Logistic Regression']['metrics']['roc_auc']:.4f}"
            ],
            "Decision Tree": [
                f"{m_data['Decision Tree']['metrics']['accuracy']*100:.2f}%",
                f"{m_data['Decision Tree']['metrics']['precision']*100:.2f}%",
                f"{m_data['Decision Tree']['metrics']['recall']*100:.2f}%",
                f"{m_data['Decision Tree']['metrics']['f1_score']*100:.2f}%",
                f"{m_data['Decision Tree']['metrics']['roc_auc']:.4f}"
            ]
        })

        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        st.divider()

        col_cm, col_roc = st.columns(2, gap="large")

        # Confusion Matrices
        with col_cm:
            st.markdown("#### 🔲 Confusion Matrices")
            fig_cm, axes = plt.subplots(1, 2, figsize=(9, 4))
            
            for ax, (model_title, model_key) in zip(axes, [("Logistic Regression", "Logistic Regression"), ("Decision Tree", "Decision Tree")]):
                cm = np.array(m_data[model_key]["confusion_matrix"])
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                            xticklabels=["Fail (0)", "Pass (1)"], yticklabels=["Fail (0)", "Pass (1)"])
                ax.set_title(f"{model_title}", fontsize=11, fontweight="bold")
                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("True Label")

            plt.tight_layout()
            st.pyplot(fig_cm)
            plt.close(fig_cm)

        # ROC Curves
        with col_roc:
            st.markdown("#### 📈 ROC (Receiver Operating Characteristic) Curves")
            fig_roc, ax_roc = plt.subplots(figsize=(6, 4))

            for model_key, color in zip(["Logistic Regression", "Decision Tree"], ["#2563EB", "#16A34A"]):
                fpr = m_data[model_key]["roc_curve"]["fpr"]
                tpr = m_data[model_key]["roc_curve"]["tpr"]
                auc_score = m_data[model_key]["metrics"]["roc_auc"]
                ax_roc.plot(fpr, tpr, color=color, lw=2, label=f"{model_key} (AUC = {auc_score:.3f})")

            ax_roc.plot([0, 1], [0, 1], color="#9CA3AF", linestyle="--", label="Random Chance (0.50)")
            ax_roc.set_xlabel("False Positive Rate (FPR)")
            ax_roc.set_ylabel("True Positive Rate (TPR / Recall)")
            ax_roc.set_title("ROC Curve Comparison", fontsize=11, fontweight="bold")
            ax_roc.legend(loc="lower right")
            ax_roc.grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig_roc)
            plt.close(fig_roc)

        st.divider()

        # Algorithmic Explanations
        st.markdown("### 📖 Algorithm Architecture Comparison")
        a_col1, a_col2 = st.columns(2, gap="medium")

        with a_col1:
            st.info("""
            **🔹 Logistic Regression**
            - **Type:** Linear Probabilistic Classifier
            - **Decision Boundary:** Hyperplane mapped through the Sigmoid activation function:
              $$P(Y=1|X) = \\frac{1}{1 + e^{-(\\beta_0 + \\beta_1 X_1 + ... + \\beta_n X_n)}}$$
            - **Strengths:** Excellent calibrated probabilities, highly interpretable via odds ratios, resistant to noise when regularized.
            - **Best for:** Measuring proportional contribution of each study hour or attendance percentage.
            """)

        with a_col2:
            st.success("""
            **🔹 Decision Tree Classifier**
            - **Type:** Non-parametric Rule-based Classifier
            - **Decision Boundary:** Orthogonal piecewise linear threshold splits optimized via Gini Impurity:
              $$Gini = 1 - \\sum_{i=1}^{C} p_i^2$$
            - **Strengths:** Captures non-linear thresholds (e.g., "if study hours < 10 AND attendance < 70% then Fail"), natural categorical handling, zero scaling needed.
            - **Best for:** Clear rule-based institutional guidelines and student intervention thresholds.
            """)
    else:
        st.warning("Model metadata is loading or not generated yet.")

# ==============================================================================
# TAB 3: EXPLAINABILITY & FEATURE IMPORTANCE
# ==============================================================================
with tab_insights:
    st.subheader("Model Insights & Feature Importance")
    st.write("Understand which student attributes have the greatest impact on academic success.")

    if metadata and "models" in metadata:
        m_data = metadata["models"]

        ins_col1, ins_col2 = st.columns(2, gap="large")

        with ins_col1:
            st.markdown("#### 🌲 Decision Tree: Feature Importances")
            dt_imp = m_data["Decision Tree"]["insights"]["feature_importances"]
            imp_df = pd.DataFrame(list(dt_imp.items()), columns=["Feature", "Importance"])
            imp_df = imp_df.sort_values(by="Importance", ascending=True)

            fig_dt, ax_dt = plt.subplots(figsize=(6, 4.5))
            ax_dt.barh(imp_df["Feature"], imp_df["Importance"], color="#10B981")
            ax_dt.set_xlabel("Gini Importance Score")
            ax_dt.set_title("Decision Tree Feature Importance", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_dt)
            plt.close(fig_dt)

        with ins_col2:
            st.markdown("#### ⚖️ Logistic Regression: Coefficients (Impact on Log-Odds)")
            lr_coef = m_data["Logistic Regression"]["insights"]["coefficients"]
            coef_df = pd.DataFrame(list(lr_coef.items()), columns=["Feature", "Coefficient"])
            coef_df = coef_df.sort_values(by="Coefficient", ascending=True)

            fig_lr, ax_lr = plt.subplots(figsize=(6, 4.5))
            colors = ["#EF4444" if c < 0 else "#3B82F6" for c in coef_df["Coefficient"]]
            ax_lr.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)
            ax_lr.axvline(0, color="#64748B", linestyle="--", alpha=0.7)
            ax_lr.set_xlabel("Coefficient (Beta)")
            ax_lr.set_title("Logistic Regression Weights", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_lr)
            plt.close(fig_lr)

        st.divider()

        # Decision Tree Logic Rules
        st.markdown("#### 🌿 Decision Tree Rule Hierarchy")
        tree_text = m_data["Decision Tree"]["insights"].get("tree_rules_text", "")
        with st.expander("Click to view Decision Tree Split Rules (Top Branches)", expanded=True):
            st.code(tree_text, language="text")

# ==============================================================================
# TAB 4: BATCH PREDICTION & DATA EXPLORER
# ==============================================================================
with tab_batch:
    st.subheader("Data Explorer & Batch Prediction")
    st.write("Explore student dataset records or upload your own CSV file to predict outcomes in bulk.")

    subtab_explore, subtab_batch = st.tabs(["📊 Training Dataset Explorer", "📤 Upload & Batch Predict"])

    with subtab_explore:
        if dataset is not None:
            st.markdown(f"**Total Records:** {len(dataset):,} students")
            st.dataframe(dataset.head(15), use_container_width=True)

            st.markdown("#### 📈 Key Feature Distributions")
            c1, c2 = st.columns(2)
            with c1:
                fig_dist1, ax_d1 = plt.subplots(figsize=(5, 3))
                sns.histplot(data=dataset, x="study_hours_per_week", hue="passed", kde=True, ax=ax_d1, palette={0: "#EF4444", 1: "#10B981"})
                ax_d1.set_title("Study Hours vs Pass/Fail")
                st.pyplot(fig_dist1)
                plt.close(fig_dist1)

            with c2:
                fig_dist2, ax_d2 = plt.subplots(figsize=(5, 3))
                sns.histplot(data=dataset, x="attendance_rate", hue="passed", kde=True, ax=ax_d2, palette={0: "#EF4444", 1: "#10B981"})
                ax_d2.set_title("Attendance Rate vs Pass/Fail")
                st.pyplot(fig_dist2)
                plt.close(fig_dist2)

    with subtab_batch:
        st.markdown("#### 📁 Batch Prediction from CSV")
        st.write("Upload a CSV file containing student records to generate predictions.")

        # Download Sample CSV
        if os.path.exists(SAMPLE_BATCH_PATH):
            sample_df = pd.read_csv(SAMPLE_BATCH_PATH)
            csv_buffer = io.StringIO()
            sample_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Sample Batch Template (CSV)",
                data=csv_buffer.getvalue(),
                file_name="student_sample_batch.csv",
                mime="text/csv"
            )

        uploaded_file = st.file_uploader("Upload CSV file for Batch Prediction", type=["csv"])
        batch_algo = st.selectbox("Select Model for Batch Prediction", ["Logistic Regression", "Decision Tree"])

        if uploaded_file is not None:
            batch_data = pd.read_csv(uploaded_file)
            st.write(f"Loaded **{len(batch_data)}** rows from uploaded file.")
            
            if st.button("⚡ Run Batch Prediction", type="primary"):
                pipeline = models[batch_algo]
                preds = pipeline.predict(batch_data)
                probs = pipeline.predict_proba(batch_data)[:, 1]

                result_df = batch_data.copy()
                result_df["Predicted_Status"] = ["Pass" if p == 1 else "Fail / At Risk" for p in preds]
                result_df["Pass_Probability_%"] = np.round(probs * 100, 1)

                st.success(f"Predictions completed for {len(result_df)} students!")
                st.dataframe(result_df, use_container_width=True)

                out_buf = io.StringIO()
                result_df.to_csv(out_buf, index=False)
                st.download_button(
                    label="💾 Download Predictions as CSV",
                    data=out_buf.getvalue(),
                    file_name="predicted_student_performance.csv",
                    mime="text/csv"
                )