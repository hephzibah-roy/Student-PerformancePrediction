"""
=============================================================================
Course Outcome Project: Student Performance Prediction
Topic: Student Performance Prediction
Algorithms: Logistic Regression & Decision Tree
Target Course Outcome:
"CO5 – Solve real life problems using appropriate machine learning models
 and evaluate the performance measures"
=============================================================================
This script executes the complete ML pipeline in the required sequence:
Step 3.1: Load data and print shape, columns, class counts.
Step 3.2: Check missing values, duplicates, encode text columns,
          scale features with StandardScaler inside a Pipeline (no leakage),
          80/20 stratified train-test split.
Step 3.3: Algorithm selection and justification.
Step 3.4: 5-Fold Cross-Validation on training set to pick the final model.
Step 3.5: Train on train set (.fit) and evaluate on test set (.predict).
Step 3.6: Evaluate Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix.
Step 3.7: Plot & save all required graphs to plots/ folder with plt.savefig().
Step 3.8: Save final chosen model to model.pkl via joblib.
=============================================================================
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

def run_pipeline():
    # -------------------------------------------------------------------------
    # Setup Paths
    # -------------------------------------------------------------------------
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "dataset.csv")
    plots_dir = os.path.join(base_dir, "plots")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 75)
    print("STEP 3.1: LOAD DATA AND INSPECT PROPERTIES")
    print("=" * 75)
    df = pd.read_csv(dataset_path)
    print(f"Dataset Shape       : {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Column Names        : {list(df.columns)}")
    print("\nClass Counts (Target = 'passed'):")
    class_counts = df["passed"].value_counts()
    class_percentages = df["passed"].value_counts(normalize=True) * 100
    for cls in class_counts.index:
        label = "Pass (1)" if cls == 1 else "Fail / At Risk (0)"
        print(f"  {label:<20}: {class_counts[cls]:>5} students ({class_percentages[cls]:.2f}%)")

    print("\n" + "=" * 75)
    print("STEP 3.2: PREPROCESSING (MISSING VALUES, DUPLICATES, ENCODING & SPLIT)")
    print("=" * 75)
    # Check missing values
    missing_vals = df.isnull().sum()
    print("Missing Values per Column:")
    for col, count in missing_vals.items():
        print(f"  - {col:<30}: {count} missing")

    # Check duplicates
    duplicate_count = df.duplicated().sum()
    print(f"\nDuplicate Records Count: {duplicate_count}")

    # Feature definitions
    numerical_features = [
        "study_hours_per_week",
        "attendance_rate",
        "previous_exam_score",
        "assignment_completion_rate",
        "sleep_hours"
    ]
    categorical_features = [
        "extracurricular_activities",
        "tutoring_classes",
        "parental_education_level"
    ]
    target_col = "passed"

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split 80/20 with stratify=y and random_state=42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"\nData Split (80% Train / 20% Test):")
    print(f"  Training Set   : {X_train.shape[0]} samples")
    print(f"  Testing Set    : {X_test.shape[0]} samples")

    # Preprocessors: Inside Pipeline to avoid data leakage
    # Logistic Regression preprocessor scales numeric features with StandardScaler
    lr_preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )

    # Decision Tree preprocessor (trees do not require scaling, but categorical encoding is required)
    dt_preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )

    print("\n" + "=" * 75)
    print("STEP 3.3: ALGORITHM SELECTION AND RATIONALE")
    print("=" * 75)
    print("""
1. Logistic Regression:
   - Rationale: Serves as the quintessential linear baseline probabilistic classifier.
   - Mechanism: Maps a linear combination of weighted features into [0, 1] through the Sigmoid
     activation function. Provides calibrated probabilities and interpretable odds ratios (coefficients).

2. Decision Tree Classifier:
   - Rationale: Serves as a non-parametric, rule-based classifier that handles non-linear interactions.
   - Mechanism: Recursively splits the feature space using Gini Impurity to construct hierarchical
     if-then decision thresholds, which map naturally onto institutional academic policies.
""")

    # Build ML Pipelines
    pipelines = {
        "Logistic Regression": Pipeline([
            ("preprocessor", lr_preprocessor),
            ("classifier", LogisticRegression(C=1.0, max_iter=1000, random_state=42))
        ]),
        "Decision Tree": Pipeline([
            ("preprocessor", dt_preprocessor),
            ("classifier", DecisionTreeClassifier(max_depth=5, min_samples_leaf=15, random_state=42))
        ])
    }

    print("=" * 75)
    print("STEP 3.4: 5-FOLD STRATIFIED CROSS-VALIDATION ON TRAINING SET")
    print("=" * 75)
    print("Note: As specified, 5-fold cross-validation is used to choose the final model (not test score).\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    cv_results_summary = {}

    for name, pipe in pipelines.items():
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        mean_acc = scores["test_accuracy"].mean()
        mean_prec = scores["test_precision"].mean()
        mean_rec = scores["test_recall"].mean()
        mean_f1 = scores["test_f1"].mean()
        mean_auc = scores["test_roc_auc"].mean()

        cv_results_summary[name] = {
            "cv_accuracy": mean_acc,
            "cv_precision": mean_prec,
            "cv_recall": mean_rec,
            "cv_f1": mean_f1,
            "cv_roc_auc": mean_auc,
            "acc_std": scores["test_accuracy"].std(),
            "f1_std": scores["test_f1"].std()
        }

        print(f"[{name}] 5-Fold Cross-Validation Results (Train Set):")
        print(f"  CV Accuracy : {mean_acc:.4f} (+/- {scores['test_accuracy'].std():.4f})")
        print(f"  CV Precision: {mean_prec:.4f} (+/- {scores['test_precision'].std():.4f})")
        print(f"  CV Recall   : {mean_rec:.4f} (+/- {scores['test_recall'].std():.4f})")
        print(f"  CV F1-Score : {mean_f1:.4f} (+/- {scores['test_f1'].std():.4f})")
        print(f"  CV ROC-AUC  : {mean_auc:.4f} (+/- {scores['test_roc_auc'].std():.4f})\n")

    # Select final winning model based on 5-fold CV F1 / ROC-AUC score
    if cv_results_summary["Logistic Regression"]["cv_f1"] >= cv_results_summary["Decision Tree"]["cv_f1"]:
        best_model_name = "Logistic Regression"
    else:
        best_model_name = "Decision Tree"

    print(f">> MODEL CHOSEN BY 5-FOLD CV: **{best_model_name}** <<")
    print(f"   (Selected based on superior 5-Fold CV F1-Score of {cv_results_summary[best_model_name]['cv_f1']:.4f})")

    print("\n" + "=" * 75)
    print("STEP 3.5 & 3.6: TRAIN ON TRAIN SET (.fit) AND EVALUATE ON TEST SET (.predict)")
    print("=" * 75)

    test_results_summary = {}
    test_roc_curves = {}
    test_cms = {}

    for name, pipe in pipelines.items():
        # Fit on training set
        pipe.fit(X_train, y_train)

        # Predict on held-out test set
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        test_roc_curves[name] = (fpr, tpr, auc)
        test_cms[name] = cm

        test_results_summary[name] = {
            "test_accuracy": acc,
            "test_precision": prec,
            "test_recall": rec,
            "test_f1": f1,
            "test_roc_auc": auc,
            "confusion_matrix": cm.tolist()
        }

        print(f"[{name}] Test Set Performance:")
        print(f"  Accuracy : {acc:.4f} ({acc*100:.2f}%)")
        print(f"  Precision: {prec:.4f} ({prec*100:.2f}%)")
        print(f"  Recall   : {rec:.4f} ({rec*100:.2f}%)")
        print(f"  F1-Score : {f1:.4f} ({f1*100:.2f}%)")
        print(f"  ROC-AUC  : {auc:.4f}")
        print("  Confusion Matrix:")
        print(f"    [[TN={cm[0,0]:<3}, FP={cm[0,1]:<3}],")
        print(f"     [FN={cm[1,0]:<3}, TP={cm[1,1]:<3}]]\n")

    # -------------------------------------------------------------------------
    # Save Models: model.pkl at root and in models/
    # -------------------------------------------------------------------------
    best_pipeline = pipelines[best_model_name]
    root_model_path = os.path.join(base_dir, "model.pkl")
    joblib.dump(best_pipeline, root_model_path)
    print(f"Saved winning model ({best_model_name}) to: {root_model_path}")

    # Also save individual models for comparison app
    joblib.dump(pipelines["Logistic Regression"], os.path.join(models_dir, "logistic_regression.joblib"))
    joblib.dump(pipelines["Decision Tree"], os.path.join(models_dir, "decision_tree.joblib"))

    # -------------------------------------------------------------------------
    # STEP 3.7: GENERATE AND SAVE ALL 5 REQUIRED PLOTS (plots/)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("STEP 3.7: GENERATING AND SAVING REQUIRED PLOTS WITH plt.savefig()")
    print("=" * 75)

    # Style configuration
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12})

    # 1. Class Distribution Plot
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    palette = ["#EF4444", "#10B981"]
    bars = ax1.bar(["Fail / At Risk (0)", "Pass (1)"], [class_counts[0], class_counts[1]], color=palette, width=0.55, edgecolor="#1E293B", linewidth=1.2)
    for bar in bars:
        h = bar.get_height()
        pct = (h / len(df)) * 100
        ax1.annotate(f"{h:,}\n({pct:.1f}%)",
                     xy=(bar.get_x() + bar.get_width() / 2, h / 2),
                     xytext=(0, 0), textcoords="offset points",
                     ha="center", va="center", color="white", fontweight="bold", fontsize=12)
    ax1.set_title("Student Academic Outcome Class Distribution (N=1,500)", fontweight="bold", pad=15)
    ax1.set_ylabel("Number of Students")
    ax1.set_ylim(0, max(class_counts) * 1.15)
    fig1_path = os.path.join(plots_dir, "class_distribution.png")
    fig1.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"  [1/5] Saved class distribution plot -> {fig1_path}")

    # 2. Correlation Heatmap
    fig2, ax2 = plt.subplots(figsize=(9, 7))
    corr_cols = numerical_features + ["passed"]
    corr_matrix = df[corr_cols].corr()
    corr_labels = [c.replace("_", " ").title() for c in corr_cols]
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-0.2, vmax=1.0,
                xticklabels=corr_labels, yticklabels=corr_labels, ax=ax2,
                cbar_kws={"label": "Pearson Correlation Coefficient"}, linewidths=0.5)
    ax2.set_title("Feature Correlation Heatmap with Academic Outcome ('Passed')", fontweight="bold", pad=15)
    plt.xticks(rotation=30, ha="right")
    fig2_path = os.path.join(plots_dir, "correlation_heatmap.png")
    fig2.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"  [2/5] Saved correlation heatmap -> {fig2_path}")

    # 3. Confusion Matrix (Side by Side)
    fig3, axes3 = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, name in zip(axes3, ["Logistic Regression", "Decision Tree"]):
        cm = test_cms[name]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                    xticklabels=["Predicted Fail (0)", "Predicted Pass (1)"],
                    yticklabels=["Actual Fail (0)", "Actual Pass (1)"],
                    annot_kws={"size": 14, "weight": "bold"})
        ax.set_title(f"{name} Confusion Matrix\n(Test N=300)", fontweight="bold", pad=10)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
    plt.tight_layout()
    fig3_path = os.path.join(plots_dir, "confusion_matrix.png")
    fig3.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close(fig3)
    print(f"  [3/5] Saved confusion matrices -> {fig3_path}")

    # 4. ROC Curve Comparison
    fig4, ax4 = plt.subplots(figsize=(8, 5.5))
    colors = {"Logistic Regression": "#2563EB", "Decision Tree": "#16A34A"}
    for name in ["Logistic Regression", "Decision Tree"]:
        fpr, tpr, auc_val = test_roc_curves[name]
        ax4.plot(fpr, tpr, color=colors[name], lw=2.5, label=f"{name} (AUC = {auc_val:.4f})")
    ax4.plot([0, 1], [0, 1], color="#94A3B8", linestyle="--", lw=1.8, label="Random Guess Baseline (AUC = 0.50)")
    ax4.set_xlim([-0.02, 1.02])
    ax4.set_ylim([-0.02, 1.05])
    ax4.set_xlabel("False Positive Rate (1 - Specificity)")
    ax4.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax4.set_title("Receiver Operating Characteristic (ROC) Curves", fontweight="bold", pad=15)
    ax4.legend(loc="lower right", frameon=True, fontsize=11)
    ax4.grid(True, alpha=0.3)
    fig4_path = os.path.join(plots_dir, "roc_curve.png")
    fig4.savefig(fig4_path, dpi=300, bbox_inches="tight")
    plt.close(fig4)
    print(f"  [4/5] Saved ROC curve -> {fig4_path}")

    # 5. Model Comparison Bar Chart
    fig5, ax5 = plt.subplots(figsize=(10, 5.5))
    metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    lr_scores = [
        test_results_summary["Logistic Regression"]["test_accuracy"] * 100,
        test_results_summary["Logistic Regression"]["test_precision"] * 100,
        test_results_summary["Logistic Regression"]["test_recall"] * 100,
        test_results_summary["Logistic Regression"]["test_f1"] * 100,
        test_results_summary["Logistic Regression"]["test_roc_auc"] * 100
    ]
    dt_scores = [
        test_results_summary["Decision Tree"]["test_accuracy"] * 100,
        test_results_summary["Decision Tree"]["test_precision"] * 100,
        test_results_summary["Decision Tree"]["test_recall"] * 100,
        test_results_summary["Decision Tree"]["test_f1"] * 100,
        test_results_summary["Decision Tree"]["test_roc_auc"] * 100
    ]

    x = np.arange(len(metrics_names))
    width = 0.35

    bar1 = ax5.bar(x - width/2, lr_scores, width, label="Logistic Regression", color="#2563EB", edgecolor="#1E293B")
    bar2 = ax5.bar(x + width/2, dt_scores, width, label="Decision Tree", color="#10B981", edgecolor="#1E293B")

    for bar in list(bar1) + list(bar2):
        h = bar.get_height()
        ax5.annotate(f"{h:.1f}%",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax5.set_ylabel("Score (%)")
    ax5.set_title("Comprehensive Model Performance Comparison (Test Set)", fontweight="bold", pad=15)
    ax5.set_xticks(x)
    ax5.set_xticklabels(metrics_names, fontweight="bold")
    ax5.set_ylim(65, 105)
    ax5.legend(loc="upper right", frameon=True)
    fig5_path = os.path.join(plots_dir, "model_comparison.png")
    fig5.savefig(fig5_path, dpi=300, bbox_inches="tight")
    plt.close(fig5)
    print(f"  [5/5] Saved model comparison bar chart -> {fig5_path}")

    # Save summary metadata for report and README
    summary_data = {
        "dataset": {
            "total_rows": len(df),
            "features_count": len(X.columns),
            "pass_count": int(class_counts[1]),
            "fail_count": int(class_counts[0]),
            "missing_values": int(missing_vals.sum()),
            "duplicates": int(duplicate_count)
        },
        "cv_results": cv_results_summary,
        "test_results": test_results_summary,
        "best_model": best_model_name
    }
    with open(os.path.join(models_dir, "pipeline_summary.json"), "w") as f:
        json.dump(summary_data, f, indent=4)

    print("\n" + "=" * 75)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"Final Model Saved: {root_model_path}")
    print(f"All 5 plots saved to: {plots_dir}")
    print("=" * 75)

if __name__ == "__main__":
    run_pipeline()
