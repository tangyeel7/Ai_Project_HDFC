# PPT Preparation Guide
## HDFC Bank Loan Approval Prediction — AI for Engineers Project

> All charts are in: `ppt_resources/`  
> Old project files are untouched in `reports/`, `docs/`, `models/`, `src/`

---

## Is the Project Complete for Submission? ✅

| Component | Status |
|---|---|
| Dataset (`data/loan_data.csv`) | ✅ 614-row Kaggle dataset |
| Data Preprocessing (`src/data_preprocessing.py`) | ✅ Imputation, encoding, scaling, stratified split |
| Logistic Regression (`src/train_logistic_regression.py`) | ✅ **PRIMARY MODEL** — 86.18% accuracy |
| Decision Tree (`src/train_decision_tree.py`) | ✅ Comparison model |
| Random Forest (`src/train_random_forest.py`) | ✅ Comparison model |
| SVM (`src/train_svm.py`) | ✅ Comparison model |
| Model Benchmark (`src/train_and_compare_all.py`) | ✅ Trains all 4, cross-validates, produces all charts |
| Trained Models (`models/*.pkl`) | ✅ All 4 saved |
| CLI Predictor (`predict.py`) | ✅ Command-line demo |
| Web App (`app.py`) | ✅ Streamlit UI with 3 pages |
| PPT Graphs (`ppt_resources/`) | ✅ 10 high-res charts ready |
| Viva Docs (`docs/MASTER_VIVA_DOCUMENT.md`) | ✅ Complete |

**The project is complete and submission-ready.**

---

## What Libraries Were Used to Make the Graphs?

| Library | Version | Role in Charts |
|---|---|---|
| **matplotlib** | 3.x | Base plotting engine — draws every figure, axes, lines, bars |
| **seaborn** | 0.x | High-level layer — confusion matrix heatmaps, grouped bar charts |
| **sklearn.metrics** | 1.9.0 | Computes the underlying data: `confusion_matrix()`, `roc_curve()`, `roc_auc_score()` |
| **sklearn.tree** | 1.9.0 | `plot_tree()` — renders the decision tree diagram |

---

## PPT Resources Folder — All 10 Charts Explained

### `slide_01_model_comparison_bars.png`
**What it shows:** 5 grouped bar charts (Accuracy, F1, ROC-AUC, Precision, Recall) — all 4 models side by side  
**Why it's useful:** At a glance shows Logistic Regression wins on Accuracy + F1, Random Forest wins on ROC-AUC  
**PPT Slide:** "Model Comparison" — the main performance comparison slide  
**Tool used:** `matplotlib` bars + value labels

---

### `slide_02_roc_curves.png`
**What it shows:** The ROC curve for each model — True Positive Rate vs False Positive Rate  
**Why it's useful:** Shows how well each model can distinguish good applicants from bad ones across ALL thresholds  
**PPT Slide:** "Model Evaluation" — after the comparison table  
**Tool used:** `matplotlib.pyplot.plot()` with `sklearn.metrics.roc_curve()`  
**Key talking point:** Random Forest has highest AUC (0.8619) — best at ranking applicants by risk. The dashed diagonal line is "random guessing" (AUC = 0.5) — any model above it is adding value.

---

### `slide_03_primary_confusion_matrix.png`
**What it shows:** Large, annotated confusion matrix for Logistic Regression ONLY  
**Why it's useful:** Shows exactly how many loans were correctly approved/rejected vs. mistakes  
**PPT Slide:** "Primary Model Deep Dive — Logistic Regression"  
**Tool used:** `seaborn.heatmap()` with `sklearn.metrics.confusion_matrix()`  
**Key talking point:** TP = correctly approved good loans. FP = bad loans we mistakenly approved (bank loses money). FN = good loans we rejected (lost business).

---

### `slide_04_all_confusion_matrices.png`
**What it shows:** All 4 confusion matrices in a 2×2 grid — each model colour-coded  
**Why it's useful:** Visual comparison of how different models make different types of errors  
**PPT Slide:** "Confusion Matrix Comparison"  
**Tool used:** `seaborn.heatmap()` × 4 on `matplotlib 2×2 subplot grid`

---

### `slide_05_feature_importance.png`
**What it shows:** Horizontal bar chart of which features the Random Forest found most important, with value labels  
**Why it's useful:** Proves that our 3 engineered features (IncomeToLoanRatio, TotalIncome, LoanAmountPerMonth) are highly predictive — they rank #2, #3, #5  
**PPT Slide:** "Feature Importance & Engineering Validation"  
**Tool used:** `matplotlib.pyplot.barh()` + `ax.text()` annotations with `RandomForestClassifier.feature_importances_`  
**Key talking point:** Credit_History alone accounts for 44.18% of the forest's decision-making. Our engineered features add another 26%.

---

### `slide_06_dataset_distribution.png`
**What it shows:** Bar chart + pie chart of the Approved vs Rejected class distribution  
**Why it's useful:** Establishes the dataset as real and imbalanced (68.7% Approved) — explains why we used stratified splitting  
**PPT Slide:** "Dataset Overview" — early in presentation  
**Tool used:** `matplotlib` bar + `matplotlib` pie chart

---

### `slide_07_metrics_table.png`
**What it shows:** A styled table with all 4 models' metrics, best values highlighted in green, primary model row in blue  
**Why it's useful:** Clean single-slide data reference — use as a summary or backup slide  
**PPT Slide:** "Performance Summary Table"  
**Tool used:** `matplotlib` table rendering

---

### `slide_08_lr_coefficients.png`
**What it shows:** Two charts side by side — Logistic Regression coefficients (how much each feature changes the log-odds) AND odds ratios (how much each feature multiplies the approval probability)  
**Why it's useful:** Makes Logistic Regression interpretable — "Credit_History has an odds ratio of 23×, meaning good credit multiplies approval odds 23 times"  
**PPT Slide:** "Primary Model Interpretability — Why Logistic Regression"  
**Tool used:** `matplotlib.pyplot.barh()` with positive (blue) / negative (red) colouring  
**KEY number to memorise:** `Credit_History` odds ratio = **23.06×** — the most dramatic single feature

---

### `slide_09_model_selection_radar.png`
**What it shows:** A radar/spider chart with 5 criteria (Accuracy, F1, Interpretability, Speed, Calibration) — all 4 models plotted, Logistic Regression filled in blue  
**Why it's useful:** Visually justifies WHY Logistic Regression was chosen as primary — it scores highest overall when interpretability and speed are considered alongside accuracy  
**PPT Slide:** "Model Selection Rationale"  
**Tool used:** `matplotlib` polar plot

---

### `slide_10_pipeline_diagram.png`
**What it shows:** A visual flow diagram of the entire ML pipeline: Raw Data → Preprocessing → Split → Train 4 Models → Benchmark → Best Model  
**Why it's useful:** Gives the audience a mental map of how the project works end-to-end  
**PPT Slide:** "Project Architecture / Methodology" — very early in presentation  
**Tool used:** `matplotlib.patches.FancyBboxPatch` + `ax.annotate()` arrows

---

### `ppt_model_metrics_reference.csv`
**What it is:** A clean CSV with all metrics + a "Role" column (PRIMARY ★ / Comparison) — use to paste exact numbers into your PPT slides  
**Not a PPT slide** — a data reference sheet

---

## Suggested PPT Slide Order (14 Slides)

| # | Slide | File to Use |
|---|---|---|
| 1 | Title Slide | — |
| 2 | Problem Statement | — |
| 3 | Dataset Overview | `slide_06_dataset_distribution.png` |
| 4 | Methodology / Pipeline | `slide_10_pipeline_diagram.png` |
| 5 | Data Preprocessing Steps | Text slide |
| 6 | Feature Engineering (what & why) | Text + table |
| 7 | Models Trained (overview of all 4) | Text slide |
| 8 | Primary Model: Logistic Regression | `slide_08_lr_coefficients.png` |
| 9 | Primary Model: Confusion Matrix | `slide_03_primary_confusion_matrix.png` |
| 10 | Model Comparison — Performance | `slide_01_model_comparison_bars.png` |
| 11 | Model Comparison — ROC-AUC | `slide_02_roc_curves.png` |
| 12 | All Confusion Matrices | `slide_04_all_confusion_matrices.png` |
| 13 | Feature Importance (RF Gini) | `slide_05_feature_importance.png` |
| 14 | Model Selection Rationale | `slide_09_model_selection_radar.png` |
| 15 | Summary Table | `slide_07_metrics_table.png` |
| 16 | Conclusion & Future Scope | Text slide |
| 17 | Q&A | — |

---

## Key Numbers to Have on Every Slide

| Fact | Value |
|---|---|
| Dataset size | 614 rows, 12 features |
| Train / Test split | 491 train / 123 test (80/20 stratified) |
| **Primary Model Accuracy** | **86.18%** |
| **Primary Model F1-Score** | **0.9081** |
| Best ROC-AUC (Random Forest) | **0.8619** |
| Credit_History Odds Ratio | **23.06×** |
| Cross-Validation (5-fold) | LR: 79.63% ± 4.21% |
| Engineered features | 3 (TotalIncome, LoanAmountPerMonth, IncomeToLoanRatio) |
| Their importance contribution | ~26.4% of total RF feature importance |

---

## How to Run the Project (if examiner asks for demo)

```bash
# Install dependencies (first time only)
pip install -r requirements.txt

# Train all models (generates all charts)
python src/train_and_compare_all.py

# Launch the web app
streamlit run app.py

# CLI demo
python predict.py
```
