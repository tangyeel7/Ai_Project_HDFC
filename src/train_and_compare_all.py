"""
Master Benchmark: Train & Compare All Models
File: src/train_and_compare_all.py

This script trains all 4 classification models:
  1. Logistic Regression
  2. Decision Tree
  3. Random Forest
  4. Support Vector Machine (SVM)

It benchmarks them on 5-fold cross-validation and test set metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC

It produces:
  - A consolidated model comparison table
  - Multi-model evaluation bar charts
  - Combined ROC-AUC Curves
  - Side-by-side confusion matrix grid
  - Automatically selects and saves the Best Model to `models/best_model.pkl`
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import get_train_test_data


def run_benchmark():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    print("=" * 80)
    print("      HDFC BANK LOAN APPROVAL PREDICTION — MULTI-MODEL BENCHMARK")
    print("=" * 80)

    # 1. Load Preprocessed Data
    X_train, X_test, y_train, y_test, scaler, encoders, features = get_train_test_data()

    # 2. Define Model Configurations with Optimized Parameters
    models = {
        'Logistic Regression': LogisticRegression(
            C=1.0, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42
        ),
        'Decision Tree': DecisionTreeClassifier(
            criterion='gini', max_depth=4, min_samples_split=10, min_samples_leaf=5, random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=6, min_samples_split=5, min_samples_leaf=2,
            max_features='sqrt', random_state=42, n_jobs=-1
        ),
        'SVM (RBF)': SVC(
            C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=42
        )
    }

    results = []
    trained_models = {}
    test_predictions = {}
    test_probabilities = {}
    confusion_matrices = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 3. Train and Benchmark Each Model
    for name, model in models.items():
        print(f"\n---> Training: {name} ...")

        # 5-fold CV on training data
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        # Fit on training data
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Predict on unseen test set
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        test_predictions[name] = y_pred
        test_probabilities[name] = y_proba

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices[name] = cm

        results.append({
            'Model': name,
            'CV Accuracy': round(cv_mean, 4),
            'CV Std': round(cv_std, 4),
            'Test Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1-Score': round(f1, 4),
            'ROC-AUC': round(roc_auc, 4)
        })

    # 4. Display Comparison Table
    df_results = pd.DataFrame(results).sort_values(by='F1-Score', ascending=False)
    print("\n" + "=" * 80)
    print("                    FINAL BENCHMARK COMPARISON TABLE")
    print("=" * 80)
    print(df_results.to_string(index=False))
    print("=" * 80)

    # Save summary table to CSV & Markdown
    df_results.to_csv("reports/model_comparison_table.csv", index=False)
    df_results.to_markdown("reports/model_comparison_table.md", index=False)

    # 5. Determine Overall Best Model (by F1-score + Test Accuracy composite)
    best_row = df_results.iloc[0]
    best_name = best_row['Model']
    best_model = trained_models[best_name]

    print(f"\n[BEST MODEL] OVERALL WINNER: {best_name}")
    print(f"   - Test Accuracy: {best_row['Test Accuracy']*100:.2f}%")
    print(f"   - F1-Score:      {best_row['F1-Score']:.4f}")
    print(f"   - ROC-AUC:       {best_row['ROC-AUC']:.4f}")

    # Save Best Model Bundle
    best_bundle = {
        'model': best_model,
        'scaler': scaler,
        'encoders': encoders,
        'features': features,
        'model_name': best_name,
        'benchmark_summary': best_row.to_dict()
    }
    joblib.dump(best_bundle, "models/best_model.pkl")
    print("   -> Saved as: models/best_model.pkl")

    # 6. Plot Multi-Metric Comparison Bar Chart
    plt.figure(figsize=(12, 6))
    metrics_to_plot = ['Test Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    df_plot = df_results.melt(id_vars='Model', value_vars=metrics_to_plot,
                              var_name='Metric', value_name='Score')

    sns.barplot(data=df_plot, x='Metric', y='Score', hue='Model', palette='tab10')
    plt.ylim(0.5, 1.0)
    plt.title('HDFC Bank Loan Model Benchmark — Evaluation Metrics Comparison', fontsize=14)
    plt.ylabel('Score (0.0 - 1.0)')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    chart_path = 'reports/models_metrics_comparison.png'
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"\nSaved metrics comparison chart to: {chart_path}")

    # 7. Plot Combined ROC Curves
    plt.figure(figsize=(8, 6))
    for name in models.keys():
        fpr, tpr, _ = roc_curve(y_test, test_probabilities[name])
        auc = roc_auc_score(y_test, test_probabilities[name])
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance (AUC = 0.500)')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Recall / Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) Curves — All Models', fontsize=13)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = 'reports/all_models_roc_curves.png'
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"Saved combined ROC curve plot to: {roc_path}")

    # 8. Side-by-Side Confusion Matrix Grid
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    cm_palettes = ['Blues', 'Greens', 'Purples', 'Oranges']

    for idx, (name, ax) in enumerate(zip(models.keys(), axes.flatten())):
        cm = confusion_matrices[name]
        acc = df_results.loc[df_results['Model'] == name, 'Test Accuracy'].values[0]
        sns.heatmap(cm, annot=True, fmt='d', cmap=cm_palettes[idx], ax=ax,
                    xticklabels=['Rejected (0)', 'Approved (1)'],
                    yticklabels=['Rejected (0)', 'Approved (1)'])
        ax.set_title(f'{name}\nAccuracy: {acc*100:.2f}%', fontsize=11)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    plt.suptitle('Confusion Matrix Comparison Across All 4 Models', fontsize=14, y=1.02)
    plt.tight_layout()
    grid_cm_path = 'reports/all_models_confusion_matrices.png'
    plt.savefig(grid_cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 4-model confusion matrix grid to: {grid_cm_path}")

    print("\n" + "=" * 80)
    print("                     ALL BENCHMARKS COMPLETED SUCCESSFULLY!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_benchmark()
