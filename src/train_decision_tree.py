"""
Model 2: Decision Tree Classifier Trainer
File: src/train_decision_tree.py

This script trains a Decision Tree classifier for HDFC Bank loan approval.
It demonstrates parameter configuration (max_depth, criterion, min_samples_split),
5-fold cross-validation, feature importance extraction, tree visualization export,
and saves the trained model and confusion matrix plot.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from data_preprocessing import get_train_test_data


def train_decision_tree(
    criterion='gini',
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight=None,
    random_state=42
):
    """
    Trains and evaluates Decision Tree with configurable hyperparameters.
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    print("=" * 65)
    print("        TRAINING MODEL: DECISION TREE CLASSIFIER")
    print("=" * 65)
    print(f"Hyperparameters Configured:")
    print(f"  - Criterion (Splitting metric): {criterion}")
    print(f"  - Max Depth (Tree depth limit): {max_depth}")
    print(f"  - Min Samples Split:            {min_samples_split}")
    print(f"  - Min Samples Leaf:             {min_samples_leaf}")
    print(f"  - Class Weight:                 {class_weight}")
    print(f"  - Random State:                 {random_state}\n")

    # 1. Load Preprocessed Data
    X_train, X_test, y_train, y_test, scaler, encoders, features = get_train_test_data()

    # 2. Instantiate Model
    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state
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

    # 6. Feature Importance
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    print("----- FEATURE IMPORTANCES (GINI REDUCTION) -----")
    print(importance_df.to_string(index=False))
    print("\n* Higher importance = Feature contributes more to tree split purity\n")

    # 7. Plot Decision Tree Structure
    plt.figure(figsize=(18, 10))
    plot_tree(
        model,
        feature_names=features,
        class_names=['Rejected', 'Approved'],
        filled=True,
        rounded=True,
        fontsize=9
    )
    plt.title(f'Decision Tree Architecture (Max Depth = {max_depth})', fontsize=14)
    tree_plot_path = 'reports/decision_tree_structure.png'
    plt.savefig(tree_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Decision Tree visual diagram to: {tree_plot_path}")

    # 8. Plot and Save Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Rejected (0)', 'Approved (1)'],
                yticklabels=['Rejected (0)', 'Approved (1)'])
    plt.title(f'Confusion Matrix - Decision Tree\nAccuracy: {acc*100:.2f}%')
    plt.xlabel('Predicted by Model')
    plt.ylabel('Actual Outcome')
    plt.tight_layout()
    cm_path = 'reports/confusion_matrix_decision_tree.png'
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_path}")

    # 9. Save Model Bundle
    bundle = {
        'model': model,
        'scaler': scaler,
        'encoders': encoders,
        'features': features,
        'model_name': 'Decision Tree',
        'metrics': {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': roc_auc}
    }
    model_path = "models/decision_tree_model.pkl"
    joblib.dump(bundle, model_path)
    print(f"Saved model bundle to: {model_path}")
    print("=" * 65 + "\n")

    return model, bundle


if __name__ == "__main__":
    train_decision_tree()
