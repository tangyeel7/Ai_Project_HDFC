# HDFC Bank Loan Approval Prediction
# THE MASTER VIVA PREPARATION DOCUMENT
# (Everything You Need — Zero Gaps)

---

> **How to use this document:** Read it top-to-bottom once the night before.
> Every section is self-contained. Use the Table of Contents to jump
> to any topic mid-viva. The "LARPER'S GLOSSARY" at the end is purely
> for dropping impressive-sounding terms confidently.

---

## TABLE OF CONTENTS

1. [Project Summary (30-Second Pitch)](#1-project-summary-30-second-pitch)
2. [Tools & Libraries Used — What Each One Does](#2-tools--libraries-used)
3. [How Every Graph Was Made — Tool by Tool](#3-how-every-graph-was-made)
4. [Dataset Details](#4-dataset-details)
5. [Data Preprocessing — Every Step Explained](#5-data-preprocessing)
6. [Feature Engineering — Why We Created New Columns](#6-feature-engineering)
7. [Model 1: Logistic Regression — Parameters, Changes, Code Lines](#7-model-1-logistic-regression)
8. [Model 2: Decision Tree — Parameters, Changes, Code Lines](#8-model-2-decision-tree)
9. [Model 3: Random Forest — Parameters, Changes, Code Lines](#9-model-3-random-forest)
10. [Model 4: SVM — Parameters, Changes, Code Lines](#10-model-4-svm)
11. [Model Evaluation Metrics — Every Term Explained](#11-model-evaluation-metrics)
12. [Why We Chose Each Model — Benefits & Drawbacks](#12-benefits--drawbacks-of-each-model)
13. [Cross-Validation — What It Is and Why We Used It](#13-cross-validation)
14. [The Loan Term (60 Months) — Why That Minimum](#14-why-60-months-minimum-loan-term)
15. [Results — Which Model Won and Why](#15-results--which-model-won-and-why)
16. [Possible Improvements & Limitations](#16-limitations--improvements)
17. [THE LARPER'S GLOSSARY — Impressive Terms & What They Mean](#17-the-larpers-glossary)

---

## 1. PROJECT SUMMARY (30-Second Pitch)

> *"We built an Automated Loan Eligibility Prediction System for HDFC Bank Ltd. using Supervised Machine Learning. Given applicant details — income, employment status, credit history, property type, and requested loan amount — our system predicts whether a loan application should be Approved or Rejected. We trained and compared four classification algorithms: Logistic Regression, Decision Tree, Random Forest, and Support Vector Machine. The best performer — Logistic Regression — achieved 86.18% test accuracy with an F1-score of 0.9081. The Random Forest model achieved the highest ROC-AUC of 0.8619, making it the best probabilistic risk-ranker. The system is deployed as an interactive Streamlit web application."*

---

## 2. TOOLS & LIBRARIES USED

### What is Python?
The programming language we wrote all code in. Version 3.11 was used.
Python is preferred in ML because of its ecosystem of scientific libraries.

---

### `pandas` — Data Handling
**What it does:** Reads CSV files, creates DataFrames (tables), handles missing values, filters rows/columns, computes statistics.

**Where we used it:**
- `pd.read_csv('data/loan_data.csv')` — loads the 614-row dataset
- `df.isnull().sum()` — counts missing values per column
- `df['Dependents'].fillna(df['Dependents'].mode()[0])` — fills gaps with mode
- `pd.DataFrame(results)` — builds the benchmark comparison table

**What if we didn't use it?** We'd have to manually parse a CSV character-by-character in raw Python. Completely impractical.

---

### `numpy` — Numerical Operations
**What it does:** Fast array math, logarithms, random number generation, sorting.

**Where we used it:**
- `np.log1p(df['ApplicantIncome'])` — log transformation for outlier mitigation
- `np.argsort(importances)` — sorting feature importances for the bar chart
- `np.linspace(0.2, 0.85, n)` — generating gradient colours for charts
- `np.exp(model.coef_[0])` — computing odds ratios from Logistic Regression coefficients

---

### `scikit-learn` — Machine Learning Engine
**What it does:** Everything ML-related: preprocessing, splitting data, training models, evaluating performance.

**Specific components we used:**

| Component | Import | What It Does |
|---|---|---|
| `LabelEncoder` | `sklearn.preprocessing` | Converts text categories to integers (Male→1, Female→0) |
| `StandardScaler` | `sklearn.preprocessing` | Standardises features to mean=0, std=1 |
| `train_test_split` | `sklearn.model_selection` | Splits 614 rows into 491 train + 123 test |
| `cross_val_score` | `sklearn.model_selection` | 5-fold CV accuracy estimation |
| `StratifiedKFold` | `sklearn.model_selection` | CV that preserves class ratio in every fold |
| `LogisticRegression` | `sklearn.linear_model` | Model 1 |
| `DecisionTreeClassifier` | `sklearn.tree` | Model 2 |
| `RandomForestClassifier` | `sklearn.ensemble` | Model 3 |
| `SVC` | `sklearn.svm` | Model 4 |
| `accuracy_score` | `sklearn.metrics` | Overall % correct |
| `confusion_matrix` | `sklearn.metrics` | TP/TN/FP/FN matrix |
| `classification_report` | `sklearn.metrics` | Precision, Recall, F1 per class |
| `roc_auc_score` | `sklearn.metrics` | Area under ROC curve |
| `roc_curve` | `sklearn.metrics` | Points to draw the ROC graph |
| `plot_tree` | `sklearn.tree` | Renders the Decision Tree diagram |

---

### `matplotlib` — Plotting Engine
**What it does:** The foundational charting library. Creates figures, axes, lines, bars. Every other plotting library (seaborn, etc.) sits on top of it.

**Specific functions we used:**
- `plt.figure(figsize=(w, h))` — create a new canvas
- `plt.plot(fpr, tpr)` — draw the ROC curve line
- `plt.barh(features, values)` — horizontal bar chart for feature importance
- `ax.text(x, y, label)` — place value labels on bars
- `plt.savefig('path.png', dpi=300)` — save high-resolution chart
- `plt.subplots(2, 2)` — 2×2 grid for the 4 confusion matrices side-by-side
- `plt.cm.magma(...)` — the magma colour palette used for feature importance bars

---

### `seaborn` — Statistical Visualisation
**What it does:** A higher-level interface over matplotlib, specifically designed for statistical charts. Produces publication-quality visuals with less code.

**Specific functions we used:**
- `sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')` — confusion matrices with colour gradient and number annotations
- `sns.barplot(data=df, x='Metric', y='Score', hue='Model')` — grouped metric comparison bar chart
- `sns.countplot(x='Loan_Status', data=df)` — EDA class distribution count plot
- `sns.boxplot(x='ApplicantIncome', data=df)` — box plot for outlier detection

**What is `annot=True`?** Tells seaborn to print the actual numbers inside each cell of the heatmap.
**What is `fmt='d'`?** Formats the numbers as integers (not floating point like 22.00 — shows just 22).
**What is `cmap='Blues'`?** Sets the colour scheme. Blues = light-to-dark blue gradient.

---

### `joblib` — Model Serialisation
**What it does:** Saves a trained Python model object to disk as a `.pkl` file and reloads it later without retraining.

**Where we used it:**
- `joblib.dump(bundle, 'models/best_model.pkl')` — saves model after training
- `joblib.load('models/best_model.pkl')` — loads model for prediction/deployment

**Why joblib and not pickle?** `joblib` is optimised for large numpy arrays (which ML models contain). It's 10-100× faster than `pickle` for this use case.

---

### `streamlit` — Web Application
**What it does:** Turns a Python script into an interactive browser-based web app with zero HTML/CSS/JavaScript needed.

**Where we used it:** `app.py` — the interactive loan eligibility checker, benchmark dashboard, and documentation viewer.

---

## 3. HOW EVERY GRAPH WAS MADE

### Graph 1: Feature Importance Bar Chart
**File:** `reports/random_forest_feature_importance.png`
**Tool:** `matplotlib.pyplot.barh()` + `ax.text()` for value labels

```python
# src/train_random_forest.py  (lines 112–119)
plt.figure(figsize=(9, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='magma')
plt.title('Random Forest Feature Importance Analysis')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('reports/random_forest_feature_importance.png', dpi=300)
```

**What `feature_importances_` is:** The `.feature_importances_` attribute on a trained `RandomForestClassifier` object returns an array of numbers (one per feature), each representing how much that feature contributed to reducing Gini impurity across all trees in the forest.

---

### Graph 2: Multi-Metric Comparison Bar Chart
**File:** `reports/models_metrics_comparison.png`
**Tool:** `seaborn.barplot()` with `hue='Model'` for grouped bars

```python
# src/train_and_compare_all.py (lines 163–175)
df_plot = df_results.melt(id_vars='Model', value_vars=metrics_to_plot,
                          var_name='Metric', value_name='Score')
sns.barplot(data=df_plot, x='Metric', y='Score', hue='Model', palette='tab10')
```

**What `.melt()` does:** Reshapes the dataframe from "wide" (one column per metric) to "long" format (one row per Model-Metric combination) — required for seaborn's grouped bar chart.

---

### Graph 3: ROC Curves
**File:** `reports/all_models_roc_curves.png`
**Tool:** `sklearn.metrics.roc_curve()` + `matplotlib.pyplot.plot()`

```python
# src/train_and_compare_all.py (lines 180–190)
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance (AUC = 0.500)')
```

**What `fpr` and `tpr` are:** `fpr` = False Positive Rate (bad loans we mistakenly approved). `tpr` = True Positive Rate = Recall (good loans we correctly approved). The ROC curve traces these as the classification probability threshold changes from 0.0 to 1.0.

---

### Graph 4: Confusion Matrix Grid (4 models side-by-side)
**File:** `reports/all_models_confusion_matrices.png`
**Tool:** `seaborn.heatmap()` on a `matplotlib 2×2 subplot grid`

```python
# src/train_and_compare_all.py (lines 196–210)
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
for idx, (name, ax) in enumerate(zip(models.keys(), axes.flatten())):
    cm = confusion_matrix(y_test, test_predictions[name])
    sns.heatmap(cm, annot=True, fmt='d', cmap=cm_palettes[idx], ax=ax)
```

**What `axes.flatten()` does:** Converts the 2×2 array of axes into a flat list of 4 axes so we can loop through them simply.

---

### Graph 5: Decision Tree Structure Diagram
**File:** `reports/decision_tree_structure.png`
**Tool:** `sklearn.tree.plot_tree()` inside matplotlib

```python
# src/train_decision_tree.py (lines 89–98)
plt.figure(figsize=(18, 10))
plot_tree(model, feature_names=features, class_names=['Rejected', 'Approved'],
          filled=True, rounded=True, fontsize=9)
plt.savefig('reports/decision_tree_structure.png', dpi=300, bbox_inches='tight')
```

**What `filled=True` does:** Colours each node based on the dominant class — blue for Approved-leaning nodes, orange for Rejected-leaning.

---

## 4. DATASET DETAILS

| Property | Value |
|---|---|
| Source | Kaggle — Analytics Vidhya Loan Prediction Dataset |
| File | `data/loan_data.csv` |
| Total Records | **614 rows** |
| Total Features | **12 input features + 1 target** |
| Target Column | `Loan_Status` (Y = Approved, N = Rejected) |
| Class Distribution | **422 Approved (68.7%) vs 192 Rejected (31.3%)** — Imbalanced |

### Feature Breakdown:

| Column | Data Type | What It Represents |
|---|---|---|
| `Loan_ID` | String | Unique identifier — dropped before training |
| `Gender` | Categorical | Male / Female |
| `Married` | Categorical | Yes / No |
| `Dependents` | Categorical | 0, 1, 2, 3+ |
| `Education` | Categorical | Graduate / Not Graduate |
| `Self_Employed` | Categorical | Yes / No |
| `ApplicantIncome` | Numerical | Primary applicant's monthly income (₹) |
| `CoapplicantIncome` | Numerical | Spouse/joint applicant's income (₹) |
| `LoanAmount` | Numerical | Loan amount requested (₹ in thousands) |
| `Loan_Amount_Term` | Numerical | Repayment duration in months |
| `Credit_History` | Binary (0/1) | 1 = Good credit, 0 = Bad/No credit |
| `Property_Area` | Categorical | Urban / Semiurban / Rural |
| `Loan_Status` | **TARGET** | Y = Approved, N = Rejected |

---

## 5. DATA PREPROCESSING

### Step 1: Drop `Loan_ID`
**Code:** `df = df.drop(columns=['Loan_ID'])`
**Why:** It's a unique alphanumeric reference number (e.g. LP001002). It has zero predictive value. If left in, the model might try to memorise it, leading to severe overfitting.

---

### Step 2: Missing Value Imputation
**Code file:** `src/data_preprocessing.py` — lines 38–51

**Categorical columns → Mode (most frequent value):**
```python
df[col].fillna(df[col].mode()[0], inplace=True)
```
Affected: `Gender` (13 missing), `Married` (3), `Dependents` (15), `Self_Employed` (32)

**Numerical columns → Median:**
```python
df[col].fillna(df[col].median(), inplace=True)
```
Affected: `LoanAmount` (22 missing), `Loan_Amount_Term` (14), `Credit_History` (50)

**Why median and not mean?**
- `ApplicantIncome` has extreme outliers (a few applicants earn ₹80,000+ while most earn ₹3,000–₹7,000).
- The mean is pulled upward by these outliers, giving an unrealistically high "average."
- The **median** (the middle value) is not affected by extremes — it stays representative of the typical applicant.

**What if we skipped imputation?**
- scikit-learn models throw `ValueError: Input contains NaN` and crash.
- Decision Trees in some versions silently ignore NaN rows → reduces effective training size.

---

### Step 3: Cleaning `Dependents`
**Code:** `df['Dependents'] = df['Dependents'].str.replace('+', '').astype(int)`
**Why:** The value `3+` is a string. We convert it to integer `3` so the model can treat it numerically.

---

### Step 4: Outlier Mitigation — Log Transformation
**Code:** `df['ApplicantIncome'] = np.log1p(df['ApplicantIncome'])`
Applied to: `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`

**Why `np.log1p()` and not `np.log()`?**
- `log(0)` = negative infinity — crashes if any co-applicant income is 0.
- `log1p(x)` = `log(1 + x)` — always safe even for zero values.

**Effect:** Compresses large values and spreads small ones. Turns a right-skewed distribution into an approximately normal (bell-shaped) distribution. This improves the performance of Logistic Regression and SVM which assume normally distributed data.

**Does this affect Decision Tree/Random Forest?** No. Tree models split on rank order, not magnitude. `log(50000) vs log(3000)` preserves the same ordering as `50000 vs 3000`.

---

### Step 5: Label Encoding
**Code:** `df[col] = LabelEncoder().fit_transform(df[col])`
Applied to: `Gender`, `Married`, `Education`, `Self_Employed`, `Property_Area`, `Loan_Status`

**What it does:** Converts text categories to integers:
- Gender: Female → 0, Male → 1
- Married: No → 0, Yes → 1
- Education: Graduate → 0, Not Graduate → 1
- Loan_Status: N → 0, Y → 1

**Why can't models use text directly?**
Machine learning models are mathematical functions operating on numerical tensors. Text strings have no arithmetic meaning.

**Why not One-Hot Encoding for `Property_Area`?**
One-Hot would create 3 new binary columns. Label Encoding (Rural=0, Semiurban=1, Urban=2) works here because tree-based models don't interpret ordinal relationships — they only use thresholds. For Logistic Regression, One-Hot would technically be more correct, but the practical accuracy difference is < 1%.

---

### Step 6: StandardScaler
**Code:** `df[features] = StandardScaler().fit_transform(df[features])`
Applied to: all numerical columns

**Formula:** `z = (x - mean) / standard_deviation`

**Effect:** Every feature ends up with mean = 0 and standard deviation = 1.

**Why is this critical for SVM and Logistic Regression?**
- SVM computes Euclidean distance between points. Without scaling, `ApplicantIncome` (thousands) completely dominates `Credit_History` (0 or 1) in the distance calculation.
- Logistic Regression regularisation (penalty on weights) penalises all weights equally — if income is unscaled, its naturally tiny weight gets under-penalised.

**Does Random Forest need scaling?** No. Trees split on threshold comparisons (`Income ≤ 5000`). Scaling income to `z ≤ -0.2` makes the same split. Scale-invariant.

**CRITICAL RULE:** Always `fit_transform` on **train data only**, then `transform` on test data. If you `fit_transform` on the whole dataset, the test set's statistics leak into the scaler → data leakage → inflated evaluation scores.

---

### Step 7: Stratified Train-Test Split
**Code:**
```python
# src/data_preprocessing.py — line 84
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**Split:** 80% train (491 rows) / 20% test (123 rows)

**What `stratify=y` does:** Ensures the 68.7% Approved / 31.3% Rejected class ratio is preserved identically in both the training set and the test set. Without it, you might accidentally get 80% Approved in training and 50% in test — biasing evaluation.

**What `random_state=42` does:** Seeds the random number generator. The number 42 is arbitrary — the point is that every time you run the code, you get the exact same 491/123 split. Reproducibility.

**Why 80/20?** Standard ML convention. Balances giving the model enough training examples while retaining a meaningful test set size (123 rows is enough to get statistically reliable metrics).

---

## 6. FEATURE ENGINEERING

Feature engineering is the process of creating new informative columns from existing raw data using domain knowledge.

### Feature 1: `TotalIncome`
**Code:** `df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']`
**Why:** Banks evaluate household repayment capacity, not just the primary applicant. A couple earning ₹4,000 + ₹3,500 together is a much safer bet than ₹7,500 from one person.

### Feature 2: `LoanAmountPerMonth`
**Code:** `df['LoanAmountPerMonth'] = df['LoanAmount'] / df['Loan_Amount_Term']`
**Why:** This is a proxy for the monthly EMI (Equated Monthly Instalment). ₹5,00,000 over 360 months = ₹1,389/mo EMI (manageable). ₹5,00,000 over 60 months = ₹8,333/mo EMI (crushing). The loan amount alone doesn't tell you the repayment burden.

### Feature 3: `IncomeToLoanRatio`
**Code:** `df['IncomeToLoanRatio'] = df['TotalIncome'] / df['LoanAmount']`
**Why:** This is a simplified **Debt Service Coverage Ratio (DSCR)** — a core banking metric. A ratio of 30 means the household earns 30× the loan amount, indicating very low default risk. A ratio of 5 means the household earns only 5× the loan — high risk.

**Why are these features effective?**
In the Random Forest feature importance results:
- `IncomeToLoanRatio` ranked **#2** (0.0953)
- `TotalIncome` ranked **#3** (0.0950)
- `LoanAmountPerMonth` ranked **#5** (0.0734)

Our 3 engineered features together account for **~26.4%** of the total predictive power in the Random Forest — proving their value beyond the original raw features.

---

## 7. MODEL 1: LOGISTIC REGRESSION

### How It Works:
1. Computes a weighted sum of features: `z = w₁x₁ + w₂x₂ + ... + b`
2. Passes through sigmoid function: `P = 1 / (1 + e^(-z))` → output between 0 and 1
3. If P ≥ 0.5 → Approved (1). If P < 0.5 → Rejected (0).

### Our Configuration:
```python
# src/train_logistic_regression.py — lines 57–64
model = LogisticRegression(
    C=1.0,          # Line 58
    penalty='l2',   # Line 59  (deprecated in sklearn 1.8+, use l1_ratio instead)
    solver='lbfgs', # Line 60
    max_iter=1000,  # Line 61
    random_state=42 # Line 63
)
```

### Parameter Reference:

| Parameter | Our Value | Code Line | What It Does | Increase → | Decrease → |
|---|---|---|---|---|---|
| `C` | `1.0` | 58 | Inverse regularisation strength | Less penalty → risk of overfitting | More penalty → underfitting |
| `solver` | `'lbfgs'` | 60 | Optimisation algorithm | — | — |
| `max_iter` | `1000` | 61 | Maximum gradient descent steps | Safe convergence | `ConvergenceWarning` if too low |
| `random_state` | `42` | 63 | Reproducibility seed | — | — |

### What happens if you change `C`?
- `C = 0.001` → Heavy regularisation. Weights shrunk near zero. Model may predict the majority class (Approved) for everyone. High bias, low variance. Accuracy drops to ~75%.
- `C = 1.0` (ours) → Balanced. Test accuracy = **86.18%**.
- `C = 1000` → Essentially no regularisation. Model perfectly fits training noise. Overfits.

### What happens if `max_iter = 10`?
`ConvergenceWarning: lbfgs failed to converge.` The model stops before finding the optimal weights → suboptimal predictions.

### Key Result:
- Test Accuracy: **86.18%** (best among all models)
- F1-Score: **0.9081** (best)
- Credit_History coefficient: **+3.138** → Odds Ratio of **23.06×** (having good credit multiplies approval odds 23 times!)

---

## 8. MODEL 2: DECISION TREE

### How It Works:
Builds a flowchart of if-else rules. At each node, picks the feature + threshold that best separates Approved from Rejected applicants (measured by Gini impurity reduction). Keeps splitting until leaves are pure or stopping criteria are hit.

### Our Configuration:
```python
# src/train_decision_tree.py — lines 55–62
model = DecisionTreeClassifier(
    criterion='gini',        # Line 55
    max_depth=4,             # Line 56
    min_samples_split=10,    # Line 57
    min_samples_leaf=5,      # Line 58
    random_state=42          # Line 60
)
```

### Parameter Reference:

| Parameter | Our Value | Code Line | What It Does | Increase → | Decrease → |
|---|---|---|---|---|---|
| `criterion` | `'gini'` | 55 | Splitting metric | — | — |
| `max_depth` | `4` | 56 | Max levels tree can grow | More splits → overfitting risk | Shallower → underfitting |
| `min_samples_split` | `10` | 57 | Min samples before splitting a node | Simpler tree, less overfitting | More splits, risk of noise fitting |
| `min_samples_leaf` | `5` | 58 | Min samples in final leaf | Smoother boundaries | Tiny leaves → noisy rules |

### What happens if `max_depth=None` (default)?
**This is the classic demonstration of overfitting:**
- `DecisionTreeClassifier(max_depth=None).fit(X_train)` → Training Accuracy = **100%**
- Test Accuracy → drops to **~68–70%**
- The gap of 30% is overfitting: the tree memorised every single applicant in training instead of learning patterns.

Our `max_depth=4` closes the gap: Train ~82%, Test **82.11%** — generalising properly.

### Gini Impurity Formula:
`Gini(node) = 1 − (p_approved² + p_rejected²)`
- Pure node (all one class) → Gini = 0
- 50/50 mixed → Gini = 0.5 (worst)

### Key Result:
- Test Accuracy: **82.11%**
- ROC-AUC: **0.8485**
- Most important feature in tree: `Credit_History` (0.7326 importance — takes 73% of total Gini reduction!)

---

## 9. MODEL 3: RANDOM FOREST

### How It Works:
Trains 100 separate Decision Trees, each on a different random sample of training data (bootstrapping). At each split in each tree, only a random subset of features is considered (feature subsampling). Final prediction = majority vote across all 100 trees.

### Our Configuration:
```python
# src/train_random_forest.py — lines 53–62
model = RandomForestClassifier(
    n_estimators=100,      # Line 54
    max_depth=6,           # Line 55
    min_samples_split=5,   # Line 56
    min_samples_leaf=2,    # Line 57
    max_features='sqrt',   # Line 58
    random_state=42,       # Line 59
    n_jobs=-1              # Line 60
)
```

### Parameter Reference:

| Parameter | Our Value | Code Line | What It Does | Increase → | Decrease → |
|---|---|---|---|---|---|
| `n_estimators` | `100` | 54 | Number of trees in forest | Slower, more stable, never overfits | Faster, less stable, higher variance |
| `max_depth` | `6` | 55 | Max depth of each individual tree | More complex trees | Simpler, faster |
| `max_features` | `'sqrt'` | 58 | Features considered at each split (√14 ≈ 3.7) | Less decorrelation between trees | More randomness per split |
| `n_jobs` | `-1` | 60 | CPU cores used | Uses all available cores (faster training) | 1 = single core |

### Why does `n_estimators` not cause overfitting?
Unlike boosting (XGBoost), adding more trees to a Random Forest does NOT overfit. By the Law of Large Numbers, the ensemble error converges to a fixed asymptotic limit. Adding trees just reduces variance — never increases bias.

### What is `max_features='sqrt'`?
At every node in every tree, instead of checking all 14 features for the best split, only √14 ≈ 3.7 (so 3 or 4) randomly selected features are checked. This forces different trees to use different features → trees become decorrelated → the ensemble error is lower than any individual tree.

### Key Result:
- Test Accuracy: **84.55%**
- ROC-AUC: **0.8619** (highest of all models — best probabilistic ranker)
- Feature Importance #1: `Credit_History` (0.4418)
- Engineered features rank #2 (`IncomeToLoanRatio` 0.0953), #3 (`TotalIncome` 0.0950), #5 (`LoanAmountPerMonth` 0.0734)

---

## 10. MODEL 4: SVM (SUPPORT VECTOR MACHINE)

### How It Works:
Finds the hyperplane (dividing boundary) in feature space that maximises the margin (gap) between the two classes. Points closest to the boundary are called Support Vectors — they're the critical applicants that define where the line is drawn.

### Our Configuration:
```python
# src/train_svm.py — lines 52–59
model = SVC(
    C=1.0,            # Line 53
    kernel='rbf',     # Line 54
    gamma='scale',    # Line 55
    degree=3,         # Line 56  (only used if kernel='poly')
    probability=True, # Line 58
    random_state=42   # Line 59
)
```

### Parameter Reference:

| Parameter | Our Value | Code Line | What It Does | Increase → | Decrease → |
|---|---|---|---|---|---|
| `C` | `1.0` | 53 | Penalty for misclassified training points | Harder margin, risk of overfitting | Softer margin, risk of underfitting |
| `kernel` | `'rbf'` | 54 | Type of decision boundary shape | — | — |
| `gamma` | `'scale'` | 55 | How far each point's influence reaches | Smaller radius → overfitting | Larger radius → underfitting |
| `probability` | `True` | 58 | Enable probability output via Platt scaling | Enables `predict_proba()` and ROC-AUC | Faster training but no probabilities |

### What happens if `gamma=100`?
Each training point's Gaussian influence shrinks to a tiny bubble around itself. The model creates isolated islands of decision boundary around individual training applicants → perfect training accuracy, catastrophic test accuracy. Severe overfitting.

### What is the Kernel Trick?
**Without it:** SVM draws a straight line between classes. If Approved and Rejected applicants overlap in 14-dimensional space, a straight line can't separate them.

**With `kernel='rbf'`:** SVM implicitly maps all data into an infinite-dimensional space where the classes become separable — without actually computing those infinite dimensions. It uses the formula `K(xᵢ, xⱼ) = exp(−γ||xᵢ − xⱼ||²)` to compute inner products in that space efficiently.

### Why `probability=True`?
Standard SVM doesn't output probabilities — it outputs the sign of the distance to the hyperplane (+1 or −1). With `probability=True`, scikit-learn applies **Platt Scaling**: fits a logistic regression on the SVM's raw distance scores to calibrate them into proper probabilities (e.g. 82.3% approval chance). This is needed for ROC-AUC calculation.

### Key Result:
- Test Accuracy: **84.55%**
- Recall: **0.9882** (tied for highest — catches 98.8% of good loan applicants)
- Support Vectors: **292 out of 491 training applicants** are on the boundary

### Why SVM strictly needs `StandardScaler`:
SVM computes `||xᵢ − xⱼ||²` (Euclidean distance). Without scaling:
- `(ApplicantIncome_i − ApplicantIncome_j)²` = e.g. `(50000−45000)² = 25,000,000`
- `(Credit_History_i − Credit_History_j)²` = `(1−0)² = 1`

Income dominates distance by a factor of **25 million**. The model effectively ignores all other features. Scaling fixes this by putting every feature on a comparable numerical footing.

---

## 11. MODEL EVALUATION METRICS

### Confusion Matrix — The Foundation

```
                  PREDICTED
                Approved    Rejected
ACTUAL Approved    TP           FN
       Rejected    FP           TN
```

- **TP (True Positive):** Model said Approved, was actually Approved ✅ (correct, good)
- **TN (True Negative):** Model said Rejected, was actually Rejected ✅ (correct, dodged a bad loan)
- **FP (False Positive):** Model said Approved, was actually Rejected ❌ **Type I Error** — BANK LOSES MONEY
- **FN (False Negative):** Model said Rejected, was actually Approved ❌ **Type II Error** — Lost business

---

### Accuracy
`Accuracy = (TP + TN) / (TP + TN + FP + FN)`

**Our LR result:** (TP+TN) / 123 = 86.18%

**Why accuracy alone is misleading here:**
Our dataset is 68.7% Approved. A model that always says "Approved" for every single person would score **68.7% accuracy** while being completely useless — it never catches a single bad loan. This is why we also track Precision, Recall, and F1.

---

### Precision
`Precision = TP / (TP + FP)`

"Of every applicant the model said APPROVED — how many actually were?"

High precision = fewer bad loans approved. Lowers bank's financial risk.

---

### Recall (Sensitivity)
`Recall = TP / (TP + FN)`

"Of every applicant who SHOULD have been approved — how many did we correctly approve?"

High recall = fewer good applicants mistakenly rejected. Better for business growth.

**LR Recall = 0.9882** → The model correctly approved 98.82% of genuinely creditworthy applicants.

---

### F1-Score
`F1 = 2 × (Precision × Recall) / (Precision + Recall)`

The harmonic mean between Precision and Recall. Punishes extreme imbalances:
- Precision = 1.0, Recall = 0.01 → F1 = 0.02 (bad!)
- Both = 0.91 → F1 = 0.91 (good)

This is the most balanced single metric for imbalanced classification.

---

### ROC-AUC
- **ROC Curve:** Plots True Positive Rate (Recall) on Y-axis vs. False Positive Rate (1−Specificity) on X-axis as the classification threshold changes from 0.0 to 1.0.
- **AUC (Area Under Curve):** 1.0 = perfect. 0.5 = random guessing. Our best: **0.8619 (Random Forest)**.

**Why Random Forest has the highest AUC despite not having the highest accuracy?**
Accuracy measures performance at a single threshold (usually 0.5). AUC measures the model's ability to RANK applicants by risk across ALL thresholds. Random Forest's averaged probability outputs are better calibrated, making it superior for risk-scoring regardless of where the bank sets its approval cutoff.

---

### Classification Report Output Explained

```
              precision    recall  f1-score   support
Rejected (0)       0.96      0.58      0.72        38
Approved (1)       0.84      0.99      0.91        85
    accuracy                           0.86       123
   macro avg       0.90      0.78      0.81       123
weighted avg       0.88      0.86      0.85       123
```

- **`support`**: Number of actual instances in that class in the test set (38 Rejected, 85 Approved).
- **`macro avg`**: Simple average treating both classes equally (good for highlighting minority class performance).
- **`weighted avg`**: Weighted by `support` — represents overall performance accounting for class imbalance.

---

## 12. BENEFITS & DRAWBACKS OF EACH MODEL

### Logistic Regression
| Benefits | Drawbacks |
|---|---|
| Fast to train and predict | Assumes linear separability of classes |
| Highly interpretable coefficients | Cannot capture complex non-linear feature interactions |
| Outputs calibrated probabilities | Sensitive to unscaled features |
| Works well on our data (best accuracy) | Assumes features are somewhat independent |
| Low computational cost | Performance degrades if features are heavily correlated |

### Decision Tree
| Benefits | Drawbacks |
|---|---|
| Completely interpretable — you can read the rules | Prone to severe overfitting if `max_depth` not set |
| No need for feature scaling | High variance — small data changes → completely different tree |
| Handles non-linear patterns | Tends to create biased trees toward features with many values |
| Visual diagram can be shown to non-technical stakeholders | Lowest ROC-AUC of the 4 models |

### Random Forest
| Benefits | Drawbacks |
|---|---|
| Reduces overfitting by averaging many trees | Slower to train and predict than single models |
| Highest ROC-AUC (0.8619) | Model is a "black box" — hard to explain individual decisions |
| Handles missing values and outliers robustly | Requires more memory (100 trees stored) |
| Built-in feature importance ranking | Cannot extrapolate beyond training data range |
| Parallelisable (`n_jobs=-1`) | |

### SVM
| Benefits | Drawbacks |
|---|---|
| Effective in high-dimensional feature spaces | Scales poorly to very large datasets (O(N²) to O(N³)) |
| Uses only support vectors — memory efficient | Kernel choice requires domain expertise |
| Robust against overfitting with correct `C` and `gamma` | Extremely sensitive to feature scaling |
| Works well with small-to-medium datasets (like our 614 rows) | No native feature importance output |
| `kernel='rbf'` captures non-linear patterns | `probability=True` adds overhead (Platt scaling) |

---

## 13. CROSS-VALIDATION

### What is 5-Fold Stratified Cross-Validation?
**Code:** `cross_val_score(model, X_train, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='accuracy')`

**Process:**
1. Split the 491 training rows into 5 equal chunks ("folds") of ~98 rows each.
2. Train on folds 2+3+4+5, validate on fold 1. Record accuracy.
3. Train on folds 1+3+4+5, validate on fold 2. Record accuracy.
4. Repeat 5 times — each fold gets to be the validation set once.
5. Final score = average of the 5 fold accuracies.

**Why not just use the train-test split?**
With a small dataset (614 rows), a single 80/20 split might be "lucky" or "unlucky" depending on which 123 rows end up in the test set. Cross-validation uses every row for validation exactly once → more reliable performance estimate.

**What does `StratifiedKFold` add over plain `KFold`?**
`StratifiedKFold` ensures the 68.7%/31.3% class ratio is preserved in every fold. Without it, you could get a fold where all Rejected applicants end up in one fold and none in another → wildly inconsistent scores.

**Our CV Results:**
- Logistic Regression: 79.63% ± 4.21%
- Decision Tree: 79.42% ± 4.83%
- Random Forest: 79.42% ± 4.48%
- SVM: 79.22% ± 4.75%

The CV scores are consistently ~7% lower than test scores — this is normal and expected with a small dataset.

---

## 14. WHY 60 MONTHS MINIMUM LOAN TERM

**Q: Why does the loan term start at 60 months (5 years) and not lower?**

**The banking answer:**
In retail banking, structured EMI-based repayment plans for home loans, personal loans, and vehicle loans are typically designed for durations of **5 to 30 years (60–360 months)**. Below 60 months:
- Loans are categorised as **short-term credit** (overdrafts, credit lines, business advances).
- They are priced differently (higher interest, different collateral requirements).
- They fall outside the product category this model is trained to evaluate.

**The dataset answer:**
The Analytics Vidhya Loan Prediction dataset only contains these loan term values in the data: 12, 36, 60, 84, 120, 180, 240, 300, 360, 480. The most common values are 360 (mode) and 180. Very few entries have terms below 60 months and they represent edge cases.

**The ML answer:**
The `LoanAmountPerMonth = LoanAmount / Loan_Amount_Term` feature we engineered becomes extremely large for very short terms (high EMI burden), which could produce unrealistically extreme outlier inputs that the model hasn't seen in training distribution.

---

## 15. RESULTS — WHICH MODEL WON AND WHY

| Model | Test Accuracy | F1-Score | ROC-AUC | CV Accuracy |
|---|---|---|---|---|
| **Logistic Regression** 🏆 Accuracy & F1 | **86.18%** | **0.9081** | 0.8000 | 79.63% |
| **Random Forest** 🏆 ROC-AUC | 84.55% | 0.8962 | **0.8619** | 79.42% |
| SVM | 84.55% | 0.8984 | 0.8034 | 79.22% |
| Decision Tree | 82.11% | 0.8804 | 0.8485 | 79.42% |

### Why Logistic Regression Won on Accuracy and F1:
The most important feature in this dataset is `Credit_History` (binary 0/1). The relationship between credit history and loan approval is nearly linear and perfectly captures the separating boundary for ~80% of cases. Logistic Regression is designed exactly for this type of problem — a few highly correlated binary features with a clear sigmoid-shaped probability.

### Why Random Forest Won on ROC-AUC:
ROC-AUC doesn't care about a fixed 0.5 threshold — it evaluates the model's ranking ability across ALL thresholds. Random Forest averages probability estimates across 100 trees, producing well-calibrated, smooth probability scores. A bank can set its own risk threshold (e.g. "only approve if > 70% confidence") and Random Forest's superior calibration makes it the best choice for that.

### The Surprising Finding:
Decision Tree had the HIGHEST ROC-AUC (0.8485) compared to SVM (0.8034) — despite lower accuracy. This shows that the Decision Tree's probability estimates are well-ordered even if individual predictions are slightly less accurate.

---

## 16. LIMITATIONS & IMPROVEMENTS

### Current Limitations:
1. **Small dataset (614 rows):** Production banking models train on millions of applications. Our model's generalisability is limited.
2. **Binary credit history (0/1):** Real banks use CIBIL scores (300–900), providing far more granular credit risk assessment.
3. **No temporal features:** We don't capture whether the applicant's income is growing or declining over time.
4. **Class imbalance (68.7% vs 31.3%):** The model has seen far fewer Rejected examples, potentially under-learning what makes an applicant high-risk. SMOTE or class_weight='balanced' could address this.
5. **No fairness audit:** We haven't checked if the model discriminates by gender or marital status.

### Possible Improvements:
- Use **XGBoost or LightGBM** — gradient boosting methods that consistently outperform Random Forest on tabular data.
- Apply **SMOTE** (Synthetic Minority Over-sampling Technique) to synthetically generate more Rejected applicant examples.
- Use **GridSearchCV** or **RandomizedSearchCV** for systematic hyperparameter optimisation instead of manual tuning.
- Implement **SHAP (SHapley Additive exPlanations)** for applicant-level model explainability ("your loan was rejected because your income-to-loan ratio is too low").
- Add **model monitoring** to detect drift when real-world income/property patterns change.

---

## 17. THE LARPER'S GLOSSARY
*Drop these terms to sound deeply knowledgeable. Each comes with a clean one-liner definition.*

| Term | One-Liner |
|---|---|
| **Supervised Learning** | Training a model on labelled examples where the correct answer is already known. |
| **Binary Classification** | A prediction problem with exactly two possible outcomes (Approved / Rejected). |
| **Stratified Split** | Ensuring the class distribution is proportionally identical in both training and test sets. |
| **Cross-Validation** | Rotating which data is used for validation across K folds to get a more reliable performance estimate. |
| **Overfitting** | The model memorises training data so well it fails on new, unseen examples. |
| **Underfitting** | The model is too simple to capture the underlying patterns in the data. |
| **Regularisation** | A penalty on model complexity that prevents overfitting by discouraging large weights. |
| **Hyperparameter** | A configuration value set BEFORE training (like `max_depth=4`) that controls how the model learns. |
| **Feature Engineering** | Creating new informative columns from existing raw data using domain knowledge. |
| **Imputation** | Filling in missing values with a reasonable substitute (median, mode). |
| **Label Encoding** | Converting text categories (Male/Female) to integer codes (1/0). |
| **StandardScaler** | Normalising features to zero mean and unit variance to ensure scale-invariant training. |
| **Gini Impurity** | A measure of how mixed a node is — 0 = pure, 0.5 = maximum disorder. |
| **Entropy** | Information-theoretic measure of disorder in a node. Used as alternative to Gini. |
| **Information Gain** | The reduction in entropy after splitting on a feature — how much we learned. |
| **Ensemble Learning** | Combining multiple models whose errors partially cancel each other out. |
| **Bagging (Bootstrap Aggregating)** | Training many models on different random sub-samples and averaging their predictions. |
| **Support Vectors** | The data points closest to the SVM's decision boundary that define where it is drawn. |
| **Kernel Trick** | Computing inner products in an infinite-dimensional space without explicitly mapping there. |
| **Platt Scaling** | Fitting a logistic regression on SVM distance scores to convert them to valid probabilities. |
| **ROC Curve** | A graph of True Positive Rate vs. False Positive Rate at every possible classification threshold. |
| **AUC (Area Under Curve)** | A single number summarising the ROC curve — how well the model ranks examples by risk. |
| **Precision** | Of everything I predicted as Positive — how many truly were? (Quality of positive predictions.) |
| **Recall / Sensitivity** | Of everything that truly was Positive — how many did I catch? (Coverage of positives.) |
| **F1-Score** | Harmonic mean of Precision and Recall. Punishes extreme imbalance between the two. |
| **Confusion Matrix** | A 2×2 table showing counts of TP, TN, FP, FN for a classifier's predictions. |
| **Feature Importance** | A score indicating how much each feature contributed to the model's decision-making. |
| **Mean Decrease in Impurity (MDI)** | How much each feature reduces Gini impurity across all splits in the Random Forest. |
| **Serialisation (joblib)** | Converting a trained model object to a file so it can be saved and later reloaded. |
| **Data Leakage** | When test set information accidentally influences model training → inflated performance metrics. |
| **Class Imbalance** | When one class has significantly more examples than another (our 68.7% vs 31.3%). |
| **SMOTE** | Synthetic Minority Over-sampling Technique — generates artificial minority class examples. |
| **Debt Service Coverage Ratio (DSCR)** | A banking metric: Income / Loan Payment — how comfortably can borrower repay? |
| **Calibration** | How well a model's predicted probabilities match real-world frequencies of outcomes. |
| **Out-of-Bag Error** | In Random Forest, using the ~36.8% of training samples not in a tree's bootstrap for validation. |
| **Gradient Descent** | Iteratively adjusting model weights in the direction that reduces prediction error. |
| **Odds Ratio** | e^(coefficient) in Logistic Regression — how much one unit of a feature multiplies the odds of approval. |
| **Non-Performing Asset (NPA)** | A bank loan where repayments have stopped — what a False Positive (approving a bad loan) creates. |
| **EMI (Equated Monthly Instalment)** | The fixed monthly payment combining principal + interest for loan repayment. |
| **CIBIL Score** | India's credit bureau score (300–900) measuring creditworthiness. Our dataset uses a simplified 0/1. |
