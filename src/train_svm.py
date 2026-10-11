"""
Model 4: Support Vector Machine (SVM) Classifier Trainer
File: src/train_svm.py

This script trains a Support Vector Classifier (SVC) for HDFC Bank loan approval.
It demonstrates parameter configuration (C, kernel, gamma), 5-fold cross-validation,
support vector analysis, and saves the trained model bundle and confusion matrix plot.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import get_train_test_data


def train_svm(
    C=1.0,
    kernel='rbf',
    gamma='scale',
    degree=3,
    class_weight=None,
    probability=True,
    random_state=42
):
    """
    Trains and evaluates Support Vector Machine with configurable hyperparameters.
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    print("=" * 65)
    print("      TRAINING MODEL: SUPPORT VECTOR CLASSIFIER (SVM)")
    print("=" * 65)
    print(f"Hyperparameters Configured:")
    print(f"  - C (Margin penalty / Regularization): {C}")
    print(f"  - Kernel:                              {kernel}")
    print(f"  - Gamma (Kernel coefficient):          {gamma}")
    print(f"  - Degree (Only for poly kernel):       {degree}")
    print(f"  - Probability estimates:               {probability}")
    print(f"  - Class Weight:                        {class_weight}")
    print(f"  - Random State:                        {random_state}\n")

    # 1. Load Preprocessed Data (StandardScaler is critical for SVM)
    X_train, X_test, y_train, y_test, scaler, encoders, features = get_train_test_data()

    # 2. Instantiate Model
    model = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        degree=degree,
        class_weight=class_weight,
        probability=probability,
        random_state=random_state
    )

    # 3. 5-Fold Stratified Cross-Validation on Training Data
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

    # 4. Train Model
    model.fit(X_train, y_train)

    # 5. Evaluate on Unseen Test Set
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if probability else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0
    cm = confusion_matrix(y_test, y_pred)

    print("----- CROSS-VALIDATION RESULTS (TRAIN SET) -----")
    print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"Fold Scores: {[round(s, 4) for s in cv_scores]}\n")

    print("----- TEST SET EVALUATION METRICS -----")
    print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Rejected (0)', 'Approved (1)']))

    # 6. Support Vector Details
    n_sv = model.n_support_
    print("----- SUPPORT VECTOR ANALYSIS -----")
    print(f"Total Support Vectors: {sum(n_sv)} out of {len(X_train)} training instances")
    print(f"  - Support Vectors for Rejected (Class 0): {n_sv[0]}")
    print(f"  - Support Vectors for Approved (Class 1): {n_sv[1]}")
    print("* Support vectors are the critical boundary instances defining the maximum margin decision boundary.\n")

    # 7. Plot and Save Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=['Rejected (0)', 'Approved (1)'],
                yticklabels=['Rejected (0)', 'Approved (1)'])
    plt.title(f'Confusion Matrix - SVM ({kernel} kernel)\nAccuracy: {acc*100:.2f}%')
    plt.xlabel('Predicted by Model')
    plt.ylabel('Actual Outcome')
    plt.tight_layout()
    cm_path = 'reports/confusion_matrix_svm.png'
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_path}")

    # 8. Save Model Bundle
    bundle = {
        'model': model,
        'scaler': scaler,
        'encoders': encoders,
        'features': features,
        'model_name': f'SVM ({kernel})',
        'metrics': {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': roc_auc}
    }
    model_path = "models/svm_model.pkl"
    joblib.dump(bundle, model_path)
    print(f"Saved model bundle to: {model_path}")
    print("=" * 65 + "\n")

    return model, bundle


if __name__ == "__main__":
    train_svm()
