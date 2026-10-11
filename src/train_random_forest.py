"""
Model 3: Random Forest Classifier Trainer
File: src/train_random_forest.py

This script trains an ensemble Random Forest classifier for HDFC Bank loan approval.
It demonstrates parameter configuration (n_estimators, max_depth, max_features),
5-fold cross-validation, feature importance visualization, and saves
the trained model bundle and confusion matrix plot.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from data_preprocessing import get_train_test_data


def train_random_forest(
    n_estimators=100,
    max_depth=6,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight=None,
    random_state=42,
    n_jobs=-1
):
    """
    Trains and evaluates Random Forest with configurable hyperparameters.
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    print("=" * 65)
    print("        TRAINING MODEL: RANDOM FOREST CLASSIFIER")
    print("=" * 65)
    print(f"Hyperparameters Configured:")
    print(f"  - n_estimators (Trees in forest): {n_estimators}")
    print(f"  - max_depth (Depth per tree):     {max_depth}")
    print(f"  - min_samples_split:              {min_samples_split}")
    print(f"  - min_samples_leaf:               {min_samples_leaf}")
    print(f"  - max_features (Subset at split): {max_features}")
    print(f"  - class_weight:                   {class_weight}")
    print(f"  - n_jobs (CPU parallelism):       {n_jobs}")
    print(f"  - Random State:                   {random_state}\n")

    # 1. Load Preprocessed Data
    X_train, X_test, y_train, y_test, scaler, encoders, features = get_train_test_data()

    # 2. Instantiate Model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs
    )

    # 3. 5-Fold Stratified Cross-Validation on Training Data
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

    # 4. Train Model
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
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Rejected (0)', 'Approved (1)']))

    # 6. Feature Importance Extraction & Chart
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    print("----- FEATURE IMPORTANCES (ENSEMBLE GINI GAIN) -----")
    print(importance_df.to_string(index=False))

    plt.figure(figsize=(9, 6))
    sns.barplot(data=importance_df, x='Importance', y='Feature', palette='magma')
    plt.title('Random Forest Feature Importance Analysis', fontsize=13)
    plt.xlabel('Importance Score')
    plt.tight_layout()
    fi_plot_path = 'reports/random_forest_feature_importance.png'
    plt.savefig(fi_plot_path, dpi=300)
    plt.close()
    print(f"\nSaved feature importance bar chart to: {fi_plot_path}")

    # 7. Plot and Save Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=['Rejected (0)', 'Approved (1)'],
                yticklabels=['Rejected (0)', 'Approved (1)'])
    plt.title(f'Confusion Matrix - Random Forest\nAccuracy: {acc*100:.2f}%')
    plt.xlabel('Predicted by Model')
    plt.ylabel('Actual Outcome')
    plt.tight_layout()
    cm_path = 'reports/confusion_matrix_random_forest.png'
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_path}")

    # 8. Save Model Bundle
    bundle = {
        'model': model,
        'scaler': scaler,
        'encoders': encoders,
        'features': features,
        'model_name': 'Random Forest',
        'metrics': {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': roc_auc}
    }
    model_path = "models/random_forest_model.pkl"
    joblib.dump(bundle, model_path)
    print(f"Saved model bundle to: {model_path}")
    print("=" * 65 + "\n")

    return model, bundle


if __name__ == "__main__":
    train_random_forest()
