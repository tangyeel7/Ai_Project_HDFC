"""
generate_ppt_resources.py
Creates ppt_resources/ folder with all charts, tables, and assets needed for the
HDFC Bank Loan Prediction PPT presentation.

Primary model: Logistic Regression
Comparison models: Decision Tree, Random Forest, SVM

Run from project root:
    python src/generate_ppt_resources.py
"""

import os, warnings
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score,
                             classification_report, accuracy_score)

# ── Output folder ─────────────────────────────────────────────────────────────
OUT = 'ppt_resources'
os.makedirs(OUT, exist_ok=True)

# ── Palette (consistent across slides) ────────────────────────────────────────
COLORS = {
    'Logistic Regression': '#2196F3',   # blue  (PRIMARY)
    'Decision Tree':       '#FF9800',   # orange
    'Random Forest':       '#4CAF50',   # green
    'SVM (RBF)':           '#9C27B0',   # purple
}
PRIMARY = 'Logistic Regression'

# ── Load all model bundles ─────────────────────────────────────────────────────
bundles = {}
for name, path in {
    'Logistic Regression': 'models/logistic_regression_model.pkl',
    'Decision Tree':       'models/decision_tree_model.pkl',
    'Random Forest':       'models/random_forest_model.pkl',
    'SVM (RBF)':           'models/svm_model.pkl',
}.items():
    if os.path.exists(path):
        bundles[name] = joblib.load(path)

# ── Load the preprocessed test data via the preprocessing module ──────────────
import sys
sys.path.insert(0, os.path.dirname(__file__))
from data_preprocessing import get_train_test_data
X_train, X_test, y_train, y_test, pp_scaler, label_encoders, feature_names = get_train_test_data()
# X_train and X_test are already scaled by the preprocessing pipeline.
# All models were trained on this same scaled data, so we pass X_test directly.

# ── Compute predictions for all models ────────────────────────────────────────
results = {}
for name, bundle in bundles.items():
    model   = bundle['model']
    feats   = bundle['features']
    # Select only the features this model was trained on (same columns, same scale)
    X_t = X_test[feats]
    y_pred  = model.predict(X_t)
    y_proba = model.predict_proba(X_t)[:, 1] if hasattr(model, 'predict_proba') else None
    results[name] = {
        'y_pred':  y_pred,
        'y_proba': y_proba,
        'acc':     accuracy_score(y_test, y_pred),
        'cm':      confusion_matrix(y_test, y_pred),
        'fpr': None, 'tpr': None, 'auc': None,
    }
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        results[name]['fpr'] = fpr
        results[name]['tpr'] = tpr
        results[name]['auc'] = roc_auc_score(y_test, y_proba)

df_metrics = pd.read_csv('reports/model_comparison_table.csv')

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — Model Comparison Bar Chart (clean, labelled, primary highlighted)
# ═══════════════════════════════════════════════════════════════════════════════
def slide_model_comparison():
    metrics = ['Test Accuracy', 'F1-Score', 'ROC-AUC', 'Precision', 'Recall']
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 6))
    fig.suptitle('Model Performance Comparison — All 4 Classifiers', fontsize=16, fontweight='bold', y=1.02)

    for ax, metric in zip(axes, metrics):
        vals  = df_metrics.set_index('Model')[metric]
        bars  = ax.bar(
            vals.index,
            vals.values,
            color=[COLORS.get(m, '#888') for m in vals.index],
            edgecolor='white', linewidth=1.2, width=0.6
        )
        # Highlight primary model bar with gold border
        for bar, name in zip(bars, vals.index):
            if name == PRIMARY:
                bar.set_edgecolor('#FFD700')
                bar.set_linewidth(3)
        # Value labels on bars
        for bar, v in zip(bars, vals.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.set_title(metric, fontsize=11, fontweight='bold')
        ax.set_ylim(0.0, 1.08)
        ax.set_xticklabels([m.replace(' ', '\n').replace('(', '\n(') for m in vals.index], fontsize=7)
        ax.set_ylabel('Score')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axhline(1.0, color='grey', linestyle='--', linewidth=0.5, alpha=0.5)

    # Legend for primary model
    patch = mpatches.Patch(facecolor=COLORS[PRIMARY], edgecolor='#FFD700', linewidth=2,
                           label=f'{PRIMARY}  ← Primary Model')
    fig.legend(handles=[patch] + [mpatches.Patch(color=COLORS[m], label=m)
                                   for m in COLORS if m != PRIMARY],
               loc='lower center', ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.10))
    plt.tight_layout()
    path = f'{OUT}/slide_01_model_comparison_bars.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — ROC-AUC Curves (all 4, primary highlighted)
# ═══════════════════════════════════════════════════════════════════════════════
def slide_roc_curves():
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, r in results.items():
        if r['fpr'] is None:
            continue
        lw = 3.5 if name == PRIMARY else 1.8
        ls = '-' if name == PRIMARY else '--'
        ax.plot(r['fpr'], r['tpr'],
                color=COLORS[name], linewidth=lw, linestyle=ls,
                label=f"{name}  (AUC = {r['auc']:.3f})" + (' ← PRIMARY' if name == PRIMARY else ''))
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.0, label='Random Chance (AUC = 0.500)')
    ax.fill_between(results[PRIMARY]['fpr'], results[PRIMARY]['tpr'], alpha=0.12, color=COLORS[PRIMARY])
    ax.set_xlabel('False Positive Rate  (Bad Loans Incorrectly Approved)', fontsize=12)
    ax.set_ylabel('True Positive Rate  (Good Loans Correctly Approved)', fontsize=12)
    ax.set_title('ROC Curves — All Models\n(Higher & Left = Better Risk Discrimination)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    path = f'{OUT}/slide_02_roc_curves.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — Confusion Matrix: PRIMARY MODEL ONLY (big, clean)
# ═══════════════════════════════════════════════════════════════════════════════
def slide_primary_confusion_matrix():
    cm = results[PRIMARY]['cm']
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, linewidths=2, linecolor='white',
                annot_kws={'size': 22, 'weight': 'bold'},
                xticklabels=['Rejected (0)', 'Approved (1)'],
                yticklabels=['Rejected (0)', 'Approved (1)'])
    # Annotate quadrants
    labels = [['True\nNegative\n(TN)', 'False\nPositive\n(FP)'],
              ['False\nNegative\n(FN)', 'True\nPositive\n(TP)']]
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.82, labels[i][j],
                    ha='center', va='center', fontsize=8.5, color='grey',
                    style='italic')
    acc = results[PRIMARY]['acc']
    ax.set_title(f'Confusion Matrix — {PRIMARY} (PRIMARY MODEL)\nTest Accuracy: {acc*100:.2f}%',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Model Prediction', fontsize=12)
    ax.set_ylabel('Actual Outcome', fontsize=12)
    plt.tight_layout()
    path = f'{OUT}/slide_03_primary_confusion_matrix.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — All 4 Confusion Matrices side by side
# ═══════════════════════════════════════════════════════════════════════════════
def slide_all_confusion_matrices():
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    palettes = {'Logistic Regression': 'Blues', 'Decision Tree': 'Oranges',
                'Random Forest': 'Greens', 'SVM (RBF)': 'Purples'}
    for ax, (name, r) in zip(axes.flatten(), results.items()):
        sns.heatmap(r['cm'], annot=True, fmt='d', cmap=palettes.get(name, 'Blues'), ax=ax,
                    annot_kws={'size': 16, 'weight': 'bold'},
                    xticklabels=['Rejected', 'Approved'],
                    yticklabels=['Rejected', 'Approved'],
                    linewidths=1.5, linecolor='white')
        tag = ' ★ PRIMARY' if name == PRIMARY else ''
        ax.set_title(f'{name}{tag}\nAccuracy: {r["acc"]*100:.1f}%', fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    fig.suptitle('Confusion Matrices — All 4 Models', fontsize=15, fontweight='bold')
    plt.tight_layout()
    path = f'{OUT}/slide_04_all_confusion_matrices.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 5 — Feature Importance (Random Forest) with value labels
# ═══════════════════════════════════════════════════════════════════════════════
def slide_feature_importance():
    rf_bundle = bundles['Random Forest']
    rf_model  = rf_bundle['model']
    feats     = rf_bundle['features']
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.magma(np.linspace(0.2, 0.85, len(indices)))
    bars = ax.barh([feats[i] for i in indices], [importances[i] for i in indices],
                   color=colors, edgecolor='white', height=0.7)
    for bar, val in zip(bars, [importances[i] for i in indices]):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', ha='left', fontsize=10, fontweight='bold', color='#1a202c')
    ax.set_xlabel('Importance Score  (Mean Decrease in Gini Impurity)', fontsize=12)
    ax.set_title('Feature Importance — Random Forest Ensemble\n(Identifies which inputs drive loan approval decisions most)',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(0, max(importances) + 0.09)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axvline(np.mean(importances), color='red', linestyle='--', linewidth=1.5,
               label=f'Mean Importance = {np.mean(importances):.4f}')
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = f'{OUT}/slide_05_feature_importance.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — Dataset EDA: Class Distribution
# ═══════════════════════════════════════════════════════════════════════════════
def slide_class_distribution():
    df = pd.read_csv('data/loan_data.csv')
    counts = df['Loan_Status'].value_counts()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    bar_colors = ['#2196F3', '#FF5252']
    ax1.bar(['Approved (Y)', 'Rejected (N)'], [counts.get('Y', 0), counts.get('N', 0)],
            color=bar_colors, edgecolor='white', width=0.5)
    for i, (label, val) in enumerate(zip(['Approved', 'Rejected'], [counts.get('Y', 0), counts.get('N', 0)])):
        ax1.text(i, val + 5, str(val), ha='center', fontsize=14, fontweight='bold')
    ax1.set_title('Loan Application Outcome Counts\n(614 total applications)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Applications')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Pie chart
    ax2.pie([counts.get('Y', 0), counts.get('N', 0)],
            labels=['Approved\n(68.7%)', 'Rejected\n(31.3%)'],
            colors=bar_colors, autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 13, 'fontweight': 'bold'},
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    ax2.set_title('Class Distribution\n(Imbalanced Dataset)', fontsize=12, fontweight='bold')

    fig.suptitle('Dataset Overview — HDFC Bank Loan Applications', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = f'{OUT}/slide_06_dataset_distribution.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 7 — Summary Comparison Table (rendered as image)
# ═══════════════════════════════════════════════════════════════════════════════
def slide_metrics_table():
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.axis('off')

    table_data = []
    col_labels = ['Model', 'CV Accuracy', 'Test Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Role']
    for _, row in df_metrics.iterrows():
        role = '★ PRIMARY' if row['Model'] == PRIMARY else 'Comparison'
        table_data.append([
            row['Model'],
            f"{row['CV Accuracy']:.4f}",
            f"{row['Test Accuracy']:.4f}",
            f"{row['Precision']:.4f}",
            f"{row['Recall']:.4f}",
            f"{row['F1-Score']:.4f}",
            f"{row['ROC-AUC']:.4f}",
            role
        ])

    table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.5)

    # Style header row
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#1a237e')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Highlight primary model row
    primary_row = next(i+1 for i, r in enumerate(df_metrics['Model']) if r == PRIMARY)
    for j in range(len(col_labels)):
        table[primary_row, j].set_facecolor('#E3F2FD')
        table[primary_row, j].set_text_props(fontweight='bold', color='#0D47A1')

    # Highlight best value in each metric column
    metric_cols = {'Test Accuracy': 2, 'F1-Score': 5, 'ROC-AUC': 6}
    for metric, col_idx in metric_cols.items():
        best_val = df_metrics[metric].max()
        for i, row in df_metrics.iterrows():
            if abs(row[metric] - best_val) < 1e-6:
                table[i+1, col_idx].set_facecolor('#C8E6C9')
                table[i+1, col_idx].set_text_props(fontweight='bold', color='#1B5E20')

    ax.set_title('Model Benchmark Summary — HDFC Bank Loan Prediction',
                 fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    path = f'{OUT}/slide_07_metrics_table.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 8 — Logistic Regression: Coefficient / Feature Weights
# ═══════════════════════════════════════════════════════════════════════════════
def slide_lr_coefficients():
    lr_bundle = bundles['Logistic Regression']
    lr_model  = lr_bundle['model']
    feats     = lr_bundle['features']
    coefs     = lr_model.coef_[0]
    odds      = np.exp(coefs)

    df_coef = pd.DataFrame({'Feature': feats, 'Coefficient': coefs, 'Odds Ratio': odds})
    df_coef = df_coef.sort_values('Coefficient', ascending=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Coefficients
    bar_colors_c = ['#F44336' if v < 0 else '#2196F3' for v in df_coef['Coefficient']]
    bars = ax1.barh(df_coef['Feature'], df_coef['Coefficient'], color=bar_colors_c,
                    edgecolor='white', height=0.65)
    for bar, val in zip(bars, df_coef['Coefficient']):
        xpos = bar.get_width() + 0.02 if val >= 0 else bar.get_width() - 0.02
        ha   = 'left' if val >= 0 else 'right'
        ax1.text(xpos, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                 va='center', ha=ha, fontsize=9, fontweight='bold')
    ax1.axvline(0, color='black', linewidth=1.5)
    ax1.set_xlabel('Coefficient Weight', fontsize=12)
    ax1.set_title('Logistic Regression Coefficients\n(Blue = increases approval probability, Red = decreases)',
                  fontsize=11, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Odds Ratios
    bar_colors_o = ['#F44336' if v < 1 else '#2196F3' for v in df_coef['Odds Ratio']]
    bars2 = ax2.barh(df_coef['Feature'], df_coef['Odds Ratio'], color=bar_colors_o,
                     edgecolor='white', height=0.65)
    for bar, val in zip(bars2, df_coef['Odds Ratio']):
        ax2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                 f'{val:.2f}×', va='center', ha='left', fontsize=9, fontweight='bold')
    ax2.axvline(1.0, color='black', linewidth=1.5, linestyle='--', label='Odds Ratio = 1 (no effect)')
    ax2.set_xlabel('Odds Ratio  (e^coefficient)', fontsize=12)
    ax2.set_title('Odds Ratios — Primary Model\n(>1 = increases approval odds, <1 = reduces)',
                  fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle('Logistic Regression (PRIMARY MODEL) — Feature Weights & Interpretability',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = f'{OUT}/slide_08_lr_coefficients.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 9 — Model Selection Rationale (single scorecard visual)
# ═══════════════════════════════════════════════════════════════════════════════
def slide_why_logistic_regression():
    criteria = ['Test Accuracy', 'F1-Score', 'Interpretability', 'Speed', 'Calibration']
    scores = {
        'Logistic Regression': [5, 5, 5, 5, 4],
        'Random Forest':       [4, 4, 2, 3, 3],
        'Decision Tree':       [3, 3, 5, 4, 2],
        'SVM (RBF)':           [4, 4, 1, 3, 3],
    }
    angles = np.linspace(0, 2*np.pi, len(criteria), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 8), subplot_kw={'polar': True})
    for name, vals in scores.items():
        v = vals + [vals[0]]
        lw = 3.0 if name == PRIMARY else 1.5
        ls = '-' if name == PRIMARY else '--'
        alpha = 0.25 if name == PRIMARY else 0.05
        ax.plot(angles, v, color=COLORS[name], linewidth=lw, linestyle=ls,
                label=name + (' ← PRIMARY' if name == PRIMARY else ''))
        ax.fill(angles, v, color=COLORS[name], alpha=alpha)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(criteria, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 5.5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1', '2', '3', '4', '5'], fontsize=8)
    ax.set_title('Why Logistic Regression is the Primary Model\n(Radar Chart — Higher = Better)',
                 fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='lower right', bbox_to_anchor=(1.35, -0.05), fontsize=10)
    plt.tight_layout()
    path = f'{OUT}/slide_09_model_selection_radar.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 10 — Pipeline / Flow Diagram (text-based matplotlib diagram)
# ═══════════════════════════════════════════════════════════════════════════════
def slide_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)
    ax.axis('off')

    steps = [
        ('📂  Raw Dataset\n614 rows × 13 cols\nloan_data.csv', 1, '#E3F2FD', '#1565C0'),
        ('🔧  Preprocessing\nImputation → Encoding\nScaling → Engineering', 3.5, '#E8F5E9', '#2E7D32'),
        ('✂️  Stratified Split\n80% Train (491)\n20% Test (123)', 6, '#FFF3E0', '#E65100'),
        ('🤖  Train 4 Models\nLR · DT · RF · SVM\n5-Fold Cross-Val', 8.5, '#F3E5F5', '#6A1B9A'),
        ('📊  Benchmark\nAccuracy, F1\nROC-AUC, CM', 11, '#FCE4EC', '#880E4F'),
        ('🏆  Primary Model\nLogistic Regression\n86.18% Accuracy', 13.5, '#E0F2F1', '#00695C'),
    ]

    for i, (label, x, facecolor, edgecolor) in enumerate(steps):
        # Box
        box = mpatches.FancyBboxPatch((x-1.0, 1.2), 2.0, 2.6,
                                       boxstyle='round,pad=0.15',
                                       facecolor=facecolor, edgecolor=edgecolor,
                                       linewidth=2.0)
        ax.add_patch(box)
        ax.text(x, 2.5, label, ha='center', va='center', fontsize=9.5,
                fontweight='bold', color=edgecolor, multialignment='center')
        # Arrow to next
        if i < len(steps) - 1:
            ax.annotate('', xy=(x + 1.15, 2.5), xytext=(x + 0.95, 2.5),
                        arrowprops=dict(arrowstyle='->', color='#546E7A', lw=2.0))

    ax.set_title('Project Pipeline — HDFC Bank Loan Prediction System', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    path = f'{OUT}/slide_10_pipeline_diagram.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')

# ═══════════════════════════════════════════════════════════════════════════════
# Generate metrics comparison CSV for PPT reference
# ═══════════════════════════════════════════════════════════════════════════════
def save_ppt_metrics_csv():
    df_ppt = df_metrics.copy()
    df_ppt.insert(0, 'Role', df_ppt['Model'].apply(lambda m: 'PRIMARY ★' if m == PRIMARY else 'Comparison'))
    path = f'{OUT}/ppt_model_metrics_reference.csv'
    df_ppt.to_csv(path, index=False)
    print(f'Saved: {path}')

# ═══════════════════════════════════════════════════════════════════════════════
# Run everything
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('=' * 55)
    print('Generating PPT Resources...')
    print('=' * 55)
    slide_model_comparison()
    slide_roc_curves()
    slide_primary_confusion_matrix()
    slide_all_confusion_matrices()
    slide_feature_importance()
    slide_class_distribution()
    slide_metrics_table()
    slide_lr_coefficients()
    slide_why_logistic_regression()
    slide_pipeline_diagram()
    save_ppt_metrics_csv()
    print('=' * 55)
    print(f'Done! All files saved to: {OUT}/')
    print('=' * 55)
