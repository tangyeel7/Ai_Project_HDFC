"""
Model 1: Logistic Regression Trainer
File: src/train_logistic_regression.py

This script trains a Logistic Regression classifier for HDFC Bank loan approval.
It demonstrates parameter configuration, 5-fold cross validation, evaluation metrics,
feature coefficient analysis (which factors increase/decrease loan approval odds),
and saves the trained model and confusion matrix plot.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import get_train_test_data


def train_logistic_regression(
    C=1.0,
    penalty='l2',
    solver='lbfgs',
    max_iter=1000,
    class_weight=None,
    random_state=42
):
    """
    Trains and evaluates Logistic Regression with configurable hyperparameters.
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    print("=" * 65)
    print("      TRAINING MODEL: LOGISTIC REGRESSION CLASSIFIER")
    print("=" * 65)
    print(f"Hyperparameters Configured:")
    print(f"  - C (Inverse Regularization):  {C}")
    print(f"  - Penalty (Regularization type): {penalty}")
    print(f"  - Solver:                       {solver}")
    print(f"  - Max Iterations:               {max_iter}")
    print(f"  - Class Weight:                 {class_weight}")
    print(f"  - Random State:                 {random_state}\n")

    # 1. Load Preprocessed Data
    X_train, X_test, y_train, y_test, scaler, encoders, features = get_train_test_data()

    # 2. Instantiate Model
    model = LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state
    )

    # 3. 5-Fold Stratified Cross-Validation on Training Data
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

    # 4. Train Model on full training data
    model.fit(X_train, y_train)

    # 5. Evaluate on Unseen Test Set
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print("----- CROSS-VALIDATION RESULTS (TRAIN SET) -----")
    print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"Fold Scores: {[round(s, 4) for s in cv_scores]}\n")

    print("----- TEST SET EVALUATION METRICS -----")
    print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f} (Of predicted approved, how many were true)")
    print(f"Recall:    {rec:.4f} (Of actual approved, how many we caught)")
    print(f"F1-Score:  {f1:.4f} (Harmonic mean of precision & recall)")
    print(f"ROC-AUC:   {roc_auc:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Rejected (0)', 'Approved (1)']))

    # 6. Feature Coefficients (Odds ratio interpretation for Banking)
    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': model.coef_[0],
        'Odds_Ratio': np.exp(model.coef_[0])
    }).sort_values(by='Coefficient', ascending=False)

    print("----- FEATURE INFLUENCE (COEFFICIENTS) -----")
    print(coef_df.to_string(index=False))
    print("\n* Positive coefficient = Increases probability of loan approval")
    print("* Negative coefficient = Decreases probability of loan approval\n")

    # 7. Plot and Save Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Rejected (0)', 'Approved (1)'],
                yticklabels=['Rejected (0)', 'Approved (1)'])
    plt.title(f'Confusion Matrix - Logistic Regression\nAccuracy: {acc*100:.2f}%')
    plt.xlabel('Predicted by Model')
    plt.ylabel('Actual Outcome')
    plt.tight_layout()
    cm_path = 'reports/confusion_matrix_logistic_regression.png'
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_path}")

    # 8. Save Trained Model Bundle
    bundle = {
        'model': model,
        'scaler': scaler,
        'encoders': encoders,
        'features': features,
        'model_name': 'Logistic Regression',
        'metrics': {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': roc_auc}
    }
    model_path = "models/logistic_regression_model.pkl"
    joblib.dump(bundle, model_path)
    print(f"Saved model bundle to: {model_path}")
    print("=" * 65 + "\n")

    return model, bundle


if __name__ == "__main__":
    train_logistic_regression()
