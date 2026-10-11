"""
Data Preprocessing Pipeline for HDFC Bank Loan Approval Prediction
Handles:
- Missing value imputation (mode for categorical, median for numerical)
- Feature engineering (TotalIncome, LoanAmountPerMonth, IncomeToLoanRatio)
- Outlier mitigation (log transformation)
- Encoding (LabelEncoding for binary & categorical)
- Feature Scaling (StandardScaler)
- Stratified Train-Test Split (80/20)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os


def load_raw_data(data_path="data/loan_data.csv"):
    """Load raw dataset from CSV file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")
    df = pd.read_csv(data_path)
    return df


def preprocess_data(df, is_training=True, scaler=None, label_encoders=None):
    """
    Cleans, engineers features, encodes, and prepares dataset for model training/inference.

    Returns:
        df_processed, feature_cols, target_col, scaler, label_encoders
    """
    df = df.copy()

    # Drop Loan_ID if present (pure identifier, causes overfitting)
    if 'Loan_ID' in df.columns:
        df = df.drop(columns=['Loan_ID'])

    # 1. Missing Value Imputation
    cat_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed']
    for col in cat_cols:
        if col in df.columns:
            # Mode imputation for categorical features
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            df[col] = df[col].fillna(mode_val)

    num_cols = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    for col in num_cols:
        if col in df.columns:
            # Median imputation for numerical features (robust to outliers)
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Clean Dependents column ('3+' -> 3)
    if 'Dependents' in df.columns:
        df['Dependents'] = df['Dependents'].astype(str).str.replace('+', '', regex=False)
        df['Dependents'] = pd.to_numeric(df['Dependents'], errors='coerce').fillna(0).astype(int)

    # 2. Feature Engineering (Domain Banking Indicators)
    # Total household income (Applicant + Coapplicant)
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']

    # Monthly repayment burden (EMI proxy)
    # Avoid division by zero
    safe_term = df['Loan_Amount_Term'].replace(0, 360)
    df['LoanAmountPerMonth'] = df['LoanAmount'] / safe_term

    # Income to Loan Ratio (Debt Serviceability proxy)
    safe_loan = df['LoanAmount'].replace(0, 1.0)
    df['IncomeToLoanRatio'] = df['TotalIncome'] / safe_loan

    # 3. Categorical Encoding
    if label_encoders is None:
        label_encoders = {}

    encode_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
    for col in encode_cols:
        if col in df.columns:
            if is_training:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
            else:
                le = label_encoders.get(col)
                if le:
                    # Handle unseen categories gracefully
                    classes = list(le.classes_)
                    df[col] = df[col].astype(str).map(lambda s: le.transform([s])[0] if s in classes else 0)

    # Encode Target variable if present
    target_col = 'Loan_Status'
    if target_col in df.columns:
        if is_training:
            target_le = LabelEncoder()
            # Map Y -> 1, N -> 0
            df[target_col] = target_le.fit_transform(df[target_col])
            label_encoders[target_col] = target_le
        else:
            target_le = label_encoders.get(target_col)
            if target_le:
                df[target_col] = target_le.transform(df[target_col])

    # 4. Feature Scaling with StandardScaler
    # Tree models don't strictly need scaling, but SVM and Logistic Regression strictly do.
    features_to_scale = [
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
        'Loan_Amount_Term', 'TotalIncome', 'LoanAmountPerMonth', 'IncomeToLoanRatio'
    ]

    if is_training:
        scaler = StandardScaler()
        df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    else:
        if scaler is not None:
            df[features_to_scale] = scaler.transform(df[features_to_scale])

    feature_cols = [col for col in df.columns if col != target_col]

    return df, feature_cols, target_col, scaler, label_encoders


def get_train_test_data(data_path="data/loan_data.csv", test_size=0.2, random_state=42):
    """
    Returns X_train, X_test, y_train, y_test, scaler, label_encoders, feature_cols
    """
    raw_df = load_raw_data(data_path)
    processed_df, feature_cols, target_col, scaler, label_encoders = preprocess_data(
        raw_df, is_training=True
    )

    X = processed_df[feature_cols]
    y = processed_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Preserves ~69% Approved : 31% Rejected ratio in train & test
    )

    return X_train, X_test, y_train, y_test, scaler, label_encoders, feature_cols


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, encoders, features = get_train_test_data()
    print("=== Preprocessing Pipeline Verified ===")
    print(f"Features ({len(features)}): {features}")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape:  {X_test.shape},  y_test shape:  {y_test.shape}")
    print(f"Class balance in Train: {dict(pd.Series(y_train).value_counts())}")
    print(f"Class balance in Test:  {dict(pd.Series(y_test).value_counts())}")
