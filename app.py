"""
HDFC Bank Loan Approval Prediction — Interactive Streamlit Web Application
File: app.py

Run using:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set page layout and title
st.set_page_config(
    page_title="HDFC Bank — Loan Approval System",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 HDFC Bank Ltd. — Intelligent Loan Approval System")
st.markdown(
    "**AI for Engineers Project** | Supervised Machine Learning Decision-Support System for Retail Lending"
)

# Sidebar for Model Selection & Navigation
st.sidebar.header("Navigation & Model Selection")
page = st.sidebar.radio("Select View:", ["Loan Eligibility Checker", "Model Benchmark & Analytics", "Model Parameter Inspector"])

# Load models helper
@st.cache_resource
def load_bundles():
    bundles = {}
    model_files = {
        'Best Model (Auto-Selected)': 'models/best_model.pkl',
        'Logistic Regression': 'models/logistic_regression_model.pkl',
        'Decision Tree': 'models/decision_tree_model.pkl',
        'Random Forest': 'models/random_forest_model.pkl',
        'SVM': 'models/svm_model.pkl'
    }
    for name, path in model_files.items():
        if os.path.exists(path):
            bundles[name] = joblib.load(path)
    return bundles

bundles = load_bundles()

if not bundles:
    st.error("No trained models found in `models/` directory. Please run `python src/train_and_compare_all.py` first.")
    st.stop()

# -------------------------------------------------------------
# PAGE 1: LOAN ELIGIBILITY CHECKER
# -------------------------------------------------------------
if page == "Loan Eligibility Checker":
    st.subheader("📋 Applicant Information Portal")
    st.write("Enter the loan application details below to compute the approval probability.")

    col_m, _ = st.columns([2, 2])
    with col_m:
        chosen_model_name = st.selectbox("Decision Engine:", list(bundles.keys()))
        active_bundle = bundles[chosen_model_name]

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 👤 Personal Details")
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Marital Status", ["Yes", "No"])
        dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed?", ["No", "Yes"])

    with col2:
        st.markdown("#### 💰 Financial Profile")
        applicant_income = st.number_input("Applicant Monthly Income (₹)", min_value=0, max_value=200000, value=6500, step=500)
        coapplicant_income = st.number_input("Co-applicant Monthly Income (₹)", min_value=0, max_value=200000, value=2500, step=500)
        property_area = st.selectbox("Property Location Area", ["Semiurban", "Urban", "Rural"])

    with col3:
        st.markdown("#### 📑 Loan Request & Credit Record")
        loan_amount = st.number_input("Loan Amount Requested (₹ in Thousands)", min_value=5, max_value=1000, value=140, step=5)
        # Why 60 months minimum? Banks don't offer structured EMI-based retail loans
        # below 5 years (60 months). Shorter durations are classified as short-term credit,
        # not home/personal loans. The dataset's standard terms are 60-360 months.
        loan_term = st.selectbox(
            "Loan Duration / Term (in Months)",
            options=[60, 84, 120, 180, 240, 300, 360],
            index=6,  # Default = 360 months (30 years) — most common home loan term
            format_func=lambda m: f"{m} months  ({m//12} yr{'s' if m//12 > 1 else ''})"
        )
        credit_history = st.radio(
            "Credit History Guidelines Met? (CIBIL Record)",
            options=[1.0, 0.0],
            format_func=lambda x: "Yes — Satisfactory Record (1.0)" if x == 1.0 else "No — Past Delinquency / Default (0.0)"
        )

    # Derived Domain Metrics
    total_inc = applicant_income + coapplicant_income
    emi_proxy = loan_amount / loan_term
    dti_ratio = total_inc / max(loan_amount, 1.0)

    st.info(
        f"**Computed Financial Indicators:** Total Household Income: **₹{total_inc:,}** | "
        f"Monthly EMI Burden: **₹{emi_proxy:.3f}k** | Debt Service Coverage Ratio: **{dti_ratio:.2f}**"
    )

    if st.button("🔍 Evaluate Loan Eligibility", type="primary"):
        # Build dictionary
        app_dict = {
            'Gender': gender,
            'Married': married,
            'Dependents': dependents,
            'Education': education,
            'Self_Employed': self_employed,
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_term,
            'Credit_History': credit_history,
            'Property_Area': property_area
        }

        # Inference logic
        from predict import predict_applicant
        result = predict_applicant(app_dict, active_bundle)

        st.markdown("---")
        st.subheader("🎯 Automated Underwriting Decision")

        res_col1, res_col2 = st.columns([1.5, 2])

        with res_col1:
            if result['decision'] == "APPROVED":
                st.success("### ✅ APPLICATION APPROVED")
                st.balloons()
            else:
                st.error("### ❌ APPLICATION REJECTED")

            st.write(f"**Decision Model:** `{result['model_used']}`")
            st.metric("Approval Probability", f"{result['probability_approved']}%")
            st.metric("Rejection Risk", f"{result['probability_rejected']}%")

        with res_col2:
            st.markdown("#### Decision Rationale")
            if credit_history == 1.0:
                st.write("🟢 **Credit Record:** Meets bank risk policy (Strong positive weight).")
            else:
                st.write("🔴 **Credit Record:** Past default / no credit history detected (High default risk).")

            if dti_ratio > 30:
                st.write("🟢 **Repayment Capacity:** Household income comfortably covers requested loan.")
            else:
                st.write("🟡 **Debt Coverage:** Moderate to tight debt-to-income buffer.")

            if property_area == "Semiurban":
                st.write("🟢 **Property Marketability:** Semiurban properties demonstrate statistically higher approval rates.")

# -------------------------------------------------------------
# PAGE 2: MODEL BENCHMARK & ANALYTICS
# -------------------------------------------------------------
elif page == "Model Benchmark & Analytics":
    st.subheader("📊 Comparative Model Performance & Benchmarks")
    st.write("Evaluation of 4 supervised machine learning classification algorithms trained on the dataset.")

    # Tools used callout
    st.info(
        "**Libraries & Tools Used for this Analysis:**  \n"
        "- **scikit-learn** — Model training, cross-validation, metrics (accuracy, F1, ROC-AUC, confusion matrix)  \n"
        "- **matplotlib** — Base plotting engine for all charts (ROC curves, confusion matrices, feature importance)  \n"
        "- **seaborn** — High-level statistical visualisation layer built on matplotlib (heatmaps, bar plots)  \n"
        "- **pandas** — Data loading, table manipulation, and benchmark comparison DataFrame  \n"
        "- **joblib** — Saving and loading trained model bundles (.pkl files)"
    )

    if os.path.exists("reports/model_comparison_table.csv"):
        df_comp = pd.read_csv("reports/model_comparison_table.csv")
        st.dataframe(df_comp.style.highlight_max(axis=0, subset=['Test Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'], color='#c6f6d5'), use_container_width=True)

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        if os.path.exists("reports/models_metrics_comparison.png"):
            st.image("reports/models_metrics_comparison.png", caption="Model Metrics Comparison (Test Set) — seaborn barplot + matplotlib")

    with col_c2:
        if os.path.exists("reports/all_models_roc_curves.png"):
            st.image("reports/all_models_roc_curves.png", caption="ROC-AUC Curves — matplotlib plot, sklearn.metrics.roc_curve")

    st.markdown("---")
    st.subheader("Confusion Matrix Grid")
    st.caption("Generated with: seaborn.heatmap on sklearn.metrics.confusion_matrix output")
    if os.path.exists("reports/all_models_confusion_matrices.png"):
        st.image("reports/all_models_confusion_matrices.png", caption="Side-by-Side Confusion Matrices (TP, TN, FP, FN)")

    # --- Live feature importance chart with value labels ---
    st.markdown("---")
    st.subheader("Feature Importance Analysis — Random Forest (Ensemble Gini Gain)")
    st.caption("Generated with: matplotlib.pyplot.barh + text annotations | sklearn RandomForestClassifier.feature_importances_")

    # Try to load the saved random forest bundle and plot live with labels
    rf_path = "models/random_forest_model.pkl"
    if os.path.exists(rf_path):
        rf_bundle = joblib.load(rf_path)
        rf_model = rf_bundle['model']
        feature_names = rf_bundle['features']

        import matplotlib.pyplot as plt
        import numpy as np

        importances = rf_model.feature_importances_
        indices = np.argsort(importances)  # ascending for horizontal bar

        fig, ax = plt.subplots(figsize=(9, 6))
        colors = plt.cm.magma(np.linspace(0.2, 0.85, len(indices)))
        bars = ax.barh(
            [feature_names[i] for i in indices],
            [importances[i] for i in indices],
            color=colors,
            edgecolor='white',
            height=0.65
        )
        # Add value label at end of every bar
        for bar, val in zip(bars, [importances[i] for i in indices]):
            ax.text(
                bar.get_width() + 0.002,           # x: just past the bar end
                bar.get_y() + bar.get_height() / 2, # y: centre of bar
                f"{val:.4f}",                       # the importance score
                va='center', ha='left',
                fontsize=9, color='#1a202c', fontweight='bold'
            )
        ax.set_xlabel("Importance Score (Mean Decrease in Gini Impurity)", fontsize=10)
        ax.set_title("Random Forest Feature Importance — HDFC Loan Dataset", fontsize=12, fontweight='bold')
        ax.set_xlim(0, max(importances) + 0.07)   # extra room for labels
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    elif os.path.exists("reports/random_forest_feature_importance.png"):
        st.image("reports/random_forest_feature_importance.png", caption="Feature Importance (static fallback)")



# -------------------------------------------------------------
# PAGE 3: MODEL PARAMETER INSPECTOR & KNOWLEDGE BASE
# -------------------------------------------------------------
elif page == "Model Parameter Inspector":
    st.subheader("📚 Complete Project Knowledge Base & Mathematical Formulas")
    st.write("All formulas, equations, parameters, and viva guides are rendered below in clean typography.")

    doc_map = {
        "🏆 MASTER VIVA DOCUMENT (Everything)": "docs/MASTER_VIVA_DOCUMENT.md",
        "📊 Master Parameters Cheat Sheet": "docs/PARAMETERS_CHEAT_SHEET.md",
        "📈 Logistic Regression Parameters": "docs/LOGISTIC_REGRESSION_PARAMETERS.md",
        "🌳 Decision Tree Parameters": "docs/DECISION_TREE_PARAMETERS.md",
        "🌲 Random Forest Parameters": "docs/RANDOM_FOREST_PARAMETERS.md",
        "⚡ Support Vector Machine (SVM) Parameters": "docs/SVM_PARAMETERS.md",
        "🎓 Leader's Viva Guide (50+ Q&A)": "LEADERS_VIVA_GUIDE.md",
        "💻 Line-by-Line Code Explanation": "CODE_EXPLANATION.md",
        "📋 Project Plan & Roles": "PROJECT_PLAN.md"
    }

    selected_doc = st.selectbox("Select Document to Read:", list(doc_map.keys()))
    doc_path = doc_map.get(selected_doc)

    if os.path.exists(doc_path):
        with open(doc_path, "r", encoding="utf-8") as f:
            content = f.read()
        st.markdown(content)
    else:
        st.error(f"File not found: {doc_path}")
