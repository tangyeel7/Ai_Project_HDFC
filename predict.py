"""
Prediction & Inference Engine for HDFC Bank Loan Approval
File: predict.py

Allows instant inference for single or batch applicants using the best trained model.
Can be run from command line with pre-built test cases or custom inputs.
"""

import sys
import os
import joblib
import pandas as pd
import numpy as np


def load_best_model(model_path="models/best_model.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Please run 'python src/train_and_compare_all.py' first."
        )
    return joblib.load(model_path)


def predict_applicant(applicant_dict, bundle=None):
    """
    Takes an applicant data dictionary, applies the preprocessing pipeline,
    and returns prediction (Approved / Rejected), confidence, and key factors.
    """
    if bundle is None:
        bundle = load_best_model()

    model = bundle['model']
    scaler = bundle['scaler']
    encoders = bundle['encoders']
    feature_cols = bundle['features']
    model_name = bundle['model_name']

    # Convert dictionary to DataFrame
    df = pd.DataFrame([applicant_dict])

    # Feature Engineering (same as training pipeline)
    df['Dependents'] = df['Dependents'].astype(str).str.replace('+', '', regex=False)
    df['Dependents'] = pd.to_numeric(df['Dependents'], errors='coerce').fillna(0).astype(int)

    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    safe_term = df['Loan_Amount_Term'].replace(0, 360)
    df['LoanAmountPerMonth'] = df['LoanAmount'] / safe_term
    safe_loan = df['LoanAmount'].replace(0, 1.0)
    df['IncomeToLoanRatio'] = df['TotalIncome'] / safe_loan

    # Encode categoricals using fitted encoders
    encode_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
    for col in encode_cols:
        le = encoders.get(col)
        val = str(df[col].iloc[0])
        if le and val in le.classes_:
            df[col] = le.transform([val])[0]
        else:
            df[col] = 0

    # Scale numerical features using fitted StandardScaler
    features_to_scale = [
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
        'Loan_Amount_Term', 'TotalIncome', 'LoanAmountPerMonth', 'IncomeToLoanRatio'
    ]
    df[features_to_scale] = scaler.transform(df[features_to_scale])

    # Ensure columns match exact training order
    X_input = df[feature_cols]

    # Predict
    pred = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0] if hasattr(model, "predict_proba") else None

    approved = bool(pred == 1)
    confidence = proba[1] if approved else proba[0]

    return {
        'decision': "APPROVED" if approved else "REJECTED",
        'model_used': model_name,
        'probability_approved': round(proba[1] * 100, 2) if proba is not None else None,
        'probability_rejected': round(proba[0] * 100, 2) if proba is not None else None,
        'confidence_percent': round(confidence * 100, 2) if proba is not None else None,
        'credit_history': applicant_dict.get('Credit_History'),
        'total_income': applicant_dict['ApplicantIncome'] + applicant_dict['CoapplicantIncome'],
        'loan_amount': applicant_dict['LoanAmount']
    }


def run_sample_cases():
    print("=" * 70)
    print("       HDFC BANK LOAN APPROVAL SYSTEM — INFERENCE DEMONSTRATION")
    print("=" * 70)

    bundle = load_best_model()
    print(f"Loaded Active Best Model: {bundle['model_name']}\n")

    # Case 1: Prime Applicant (High income, clean credit history, married graduate)
    applicant_1 = {
        'Gender': 'Male',
        'Married': 'Yes',
        'Dependents': '1',
        'Education': 'Graduate',
        'Self_Employed': 'No',
        'ApplicantIncome': 7500,
        'CoapplicantIncome': 3200,
        'LoanAmount': 140,
        'Loan_Amount_Term': 360,
        'Credit_History': 1.0,
        'Property_Area': 'Semiurban'
    }

    # Case 2: High Risk Applicant (Zero credit history, high loan, modest income)
    applicant_2 = {
        'Gender': 'Male',
        'Married': 'No',
        'Dependents': '0',
        'Education': 'Not Graduate',
        'Self_Employed': 'Yes',
        'ApplicantIncome': 2200,
        'CoapplicantIncome': 0,
        'LoanAmount': 220,
        'Loan_Amount_Term': 180,
        'Credit_History': 0.0,
        'Property_Area': 'Rural'
    }

    # Case 3: Borderline Applicant (Good credit history, but high loan requested)
    applicant_3 = {
        'Gender': 'Female',
        'Married': 'No',
        'Dependents': '2',
        'Education': 'Graduate',
        'Self_Employed': 'No',
        'ApplicantIncome': 3800,
        'CoapplicantIncome': 1500,
        'LoanAmount': 180,
        'Loan_Amount_Term': 360,
        'Credit_History': 1.0,
        'Property_Area': 'Urban'
    }

    test_cases = [
        ("Prime Applicant (Clean Credit, Strong Income)", applicant_1),
        ("High Risk Applicant (Default History, Low Income, High Loan)", applicant_2),
        ("Borderline Applicant (Good Credit, Moderate Income & EMI)", applicant_3)
    ]

    for title, app in test_cases:
        res = predict_applicant(app, bundle)
        print(f"Scenario: {title}")
        print(f"  Inputs: Income=Rs.{app['ApplicantIncome']}, Co-Income=Rs.{app['CoapplicantIncome']}, "
              f"Loan=Rs.{app['LoanAmount']}k, CreditHistory={app['Credit_History']}")
        print(f"  --> Decision:              [{res['decision']}]")
        print(f"  --> Approval Probability:  {res['probability_approved']}%")
        print(f"  --> Rejection Probability: {res['probability_rejected']}%")
        print(f"  --> Confidence Level:      {res['confidence_percent']}%\n")

    print("=" * 70)


if __name__ == "__main__":
    run_sample_cases()
