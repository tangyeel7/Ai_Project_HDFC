# Random Forest — Complete Parameter & Architecture Guide

> **Project:** HDFC Bank Loan Approval Prediction  
> **Target:** Binary Classification (`0 = Rejected`, `1 = Approved`)  
> **Model:** `sklearn.ensemble.RandomForestClassifier`  
> **Role:** Group Leader's Technical Deep-Dive & Parameter Manual

---

## Table of Contents
1. [Core Mathematical Intuition](#1-core-mathematical-intuition)
2. [Complete Parameter Dictionary](#2-complete-parameter-dictionary)
3. [Parameter Sensitivity Analysis (What Happens When You Change...)](#3-parameter-sensitivity-analysis)
4. [Our Chosen Configuration for HDFC Loan Data](#4-our-chosen-configuration-for-hdfc-loan-data)
5. [Feature Importance Calculation (Mean Decrease in Impurity)](#5-feature-importance-calculation)
6. [Out-of-Bag (OOB) Error Explained](#6-out-of-bag-oob-error-explained)
7. [Top Viva Questions & Model Answers on Parameters](#7-top-viva-questions--model-answers-on-parameters)

---

## 1. Core Mathematical Intuition

A Random Forest is an **ensemble meta-estimator** that fits multiple decision tree classifiers on various sub-samples of the dataset and uses **majority voting** to compute the final prediction.

### The Two Pillars of Random Forest:

#### 1. Bootstrap Aggregating (Bagging):
* If the training set has $N$ applicants, each individual tree is trained on a **bootstrap sample** of size $N$ drawn **with replacement** from the training set.
* Due to sampling with replacement, approximately **63.2%** of original samples appear in each bootstrap sample, while **36.8%** are left out (called **Out-of-Bag (OOB) samples**).

#### 2. Random Subspace Method (Feature Sub-sampling):
* When splitting a node in a standard decision tree, the algorithm examines all $M$ features to find the best split.
* In a Random Forest, at each candidate split, only a **random subset of $m \ll M$ features** (typically $m = \sqrt{M}$) is considered.
* **Why this matters:** This intentionally decorrelates the individual trees! If one feature (like `Credit_History`) is overwhelmingly dominant, all regular decision trees would pick it at the root and look identical. By restricting feature choices at each split, other features (like `IncomeToLoanRatio` and `TotalIncome`) get opportunities to form unique decision paths.

#### 3. Majority Voting:
For an applicant $\mathbf{x}$, each tree $t \in \{1, 2, \dots, B\}$ casts a vote $\hat{y}_t \in \{0, 1\}$.
$$\hat{y}_{\text{ensemble}} = \text{mode}\left( \{\hat{y}_1, \hat{y}_2, \dots, \hat{y}_B\} \right) \quad \text{or} \quad \hat{P}(Y=1 \mid \mathbf{x}) = \frac{1}{B} \sum_{t=1}^{B} P_t(Y=1 \mid \mathbf{x})$$

---

## 2. Complete Parameter Dictionary

| Parameter | Type & Default | What It Controls | Valid Options |
| :--- | :--- | :--- | :--- |
| `n_estimators` | `int, default=100` | **Number of decision trees** built in the ensemble forest. | Positive integer: `10`, `50`, `100`, `200`, `500` |
| `criterion` | `{'gini', 'entropy', 'log_loss'}, default='gini'` | Function to measure split quality inside individual trees. | `'gini'`, `'entropy'` |
| `max_depth` | `int, default=None` | Maximum depth of individual trees. | Integer $\ge 1$ or `None` (unlimited) |
| `min_samples_split` | `int or float, default=2` | Minimum samples needed to split an internal node. | Integer $\ge 2$ |
| `min_samples_leaf` | `int or float, default=1` | Minimum samples needed at a terminal leaf node. | Integer $\ge 1$ |
| `max_features` | `{'sqrt', 'log2', None}, int or float, default='sqrt'` | **Number of features considered at each split**. | `'sqrt'` ($\sqrt{14} \approx 3.7$), `'log2'`, `None` (all 14) |
| `bootstrap` | `bool, default=True` | Whether bootstrap samples are used when building trees. | `True`, `False` |
| `oob_score` | `bool, default=False` | Whether to use out-of-bag samples to estimate generalization score. | `True`, `False` |
| `n_jobs` | `int, default=None` | Number of CPU cores used to train trees in parallel. | `-1` (all available cores), `1`, `2`, `4` |
| `class_weight` | `{'balanced', 'balanced_subsample'}, dict, default=None` | Weights associated with classes. | `None`, `'balanced'`, `'balanced_subsample'` |
| `random_state` | `int, default=None` | Seed for bootstrapping and feature randomness. | Any integer (we use `42`) |

---

## 3. Parameter Sensitivity Analysis

### 3.1 What happens when `n_estimators` changes?
* **`n_estimators = 10`:**
  * Forest is too small. High variance across different random seeds.
  * Individual tree mistakes aren't sufficiently averaged out.
* **`n_estimators = 100` (Our Setting):**
  * Gold standard balance. Sufficient trees for stable majority voting.
  * Yields high test accuracy (**84.55%**) and the highest ROC-AUC (**0.8619**).
* **`n_estimators = 500` or `1000`:**
  * Marginal accuracy gain ($< 0.2\%$), but 5x to 10x slower training time.
  * **Crucial Rule:** Unlike boosting or neural nets, increasing `n_estimators` in Random Forest **never causes overfitting**; it only plateaus.

---

### 3.2 What happens when `max_features` changes?
* **`max_features='sqrt'` (Default & Optimal):**
  * Takes $\sqrt{14} \approx 3$ or $4$ features at each split.
  * Maximum decorrelation between trees while retaining strong predictive splits.
* **`max_features=None` (All 14 features):**
  * Every tree sees all features at every split.
  * **Result:** Trees become correlated and redundant. The ensemble acts almost like a single slightly averaged tree, losing its variance-reduction power.
* **`max_features=1`:**
  * Only 1 feature evaluated at each split.
  * Trees become very diverse, but many splits are uninformative. Requires a much larger `n_estimators` (e.g. 500) to compensate.

---

### 3.3 What happens when `bootstrap` changes?
* **`bootstrap=True` (Default):**
  * Each tree is trained on a different random resample. Maximizes ensemble diversity.
* **`bootstrap=False`:**
  * Every single tree is trained on the exact identical dataset. The only randomness remaining is `max_features`.
  * Increases tree correlation and diminishes generalization.

---

### 3.4 What happens when `n_jobs` changes?
* **`n_jobs=1`:** Trees are trained sequentially one-by-one on a single CPU core.
* **`n_jobs=-1`:** Utilizes all logical CPU cores on the machine. Because each tree is trained completely independently (embarrassingly parallel), training speed increases almost linearly with core count!

---

## 4. Our Chosen Configuration for HDFC Loan Data

```python
model = RandomForestClassifier(
    n_estimators=100,        # 100 independent decision trees
    max_depth=6,             # Controlled depth prevents individual tree overfitting
    min_samples_split=5,     # Modest threshold for internal splits
    min_samples_leaf=2,      # Regularizes leaf purity
    max_features='sqrt',     # sqrt(14) ~ 3.7 features per split (decorrelation)
    random_state=42,         # Reproducibility
    n_jobs=-1                # Full CPU core utilization
)
```

### Performance on HDFC Dataset:
* **5-Fold Cross-Validation Accuracy:** `79.42% (+/- 4.48%)`
* **Test Set Accuracy:** `84.55%`
* **Precision:** `0.8367`
* **Recall:** `0.9647`
* **F1-Score:** `0.8962`
* **ROC-AUC:** **`0.8619` (Highest among all evaluated models!)**
* **Visual Export:** Feature importance chart saved to `reports/random_forest_feature_importance.png`.

---

## 5. Feature Importance Calculation (Mean Decrease in Impurity)

In Random Forest, feature importance is computed using **Mean Decrease in Impurity (MDI)**:
$$\text{Importance}(X_j) = \frac{1}{B} \sum_{t=1}^{B} \sum_{n \in \text{nodes in } t \text{ split on } X_j} \Delta \text{Gini}(n)$$

Where $\Delta \text{Gini}(n) = N_n \text{Gini}(n) - N_{n_L} \text{Gini}(n_L) - N_{n_R} \text{Gini}(n_R)$.

### Empirical Feature Rankings on HDFC Loan Data:
1. **`Credit_History` (0.442):** Dominates 44% of total impurity reduction.
2. **`IncomeToLoanRatio` (0.095):** Our engineered Debt-to-Income feature!
3. **`TotalIncome` (0.095):** Our engineered household capacity feature!
4. **`ApplicantIncome` (0.079):** Primary earner cash flow.
5. **`LoanAmountPerMonth` (0.073):** Our engineered EMI burden feature!
6. **`LoanAmount` (0.060):** Principal requested.
7. **`CoapplicantIncome` (0.048):** Secondary earner cash flow.

*(Notice that our 3 engineered features rank #2, #3, and #5 in importance, proving their vital value in the project!)*

---

## 6. Out-of-Bag (OOB) Error Explained

If `oob_score=True`, scikit-learn evaluates each sample using only the trees that **did not** have that sample in their bootstrap training set.
* **Why it is useful:** It provides an unbiased validation estimate **without needing a separate validation split or cross-validation**!
* In mathematical expectation, OOB error closely approximates $K$-fold cross-validation error while saving compute time.

---

## 7. Top Viva Questions & Model Answers on Parameters

### Q1: Can a Random Forest overfit if you increase `n_estimators` to 10,000?
**Model Answer:**  
> *"No. Unlike Gradient Boosting, increasing the number of trees in a Random Forest does NOT cause overfitting. By the Strong Law of Large Numbers, as $n\_estimators \to \infty$, the ensemble error converges to a fixed asymptotic limit. Adding more trees simply reduces variance; the only downside is wasted computational time and memory."*

### Q2: What is the purpose of `max_features='sqrt'`?
**Model Answer:**  
> *"`max_features` controls the number of randomly selected features considered at each node split. If all features were considered, the dominant feature (`Credit_History`) would be selected at the root of nearly every tree, making all trees highly correlated. Restricting the candidate pool to $\sqrt{M}$ forces the forest to explore different feature combinations, decorrelating the trees and drastically lowering ensemble variance."*

### Q3: Why did Random Forest achieve the highest ROC-AUC (0.8619) even though Logistic Regression had slightly higher test accuracy?
**Model Answer:**  
> *"Accuracy measures performance at a single arbitrary threshold ($P \ge 0.50$). ROC-AUC measures the model's ranking ability across all possible thresholds. Random Forest produces continuous probability outputs by averaging votes across 100 trees, making its risk ranking exceptionally well-calibrated for distinguishing safe versus delinquent loan applicants across varying bank risk appetites."*

### Q4: What is the difference between `class_weight='balanced'` and `class_weight='balanced_subsample'`?
**Model Answer:**  
> *"`class_weight='balanced'` computes weights once based on the overall training class frequencies. `class_weight='balanced_subsample'` recomputes weights dynamically for each individual tree based on the class frequencies within that specific tree's bootstrap sample. The latter is slightly more adaptive to local sampling variations."*
