# Decision Tree — Complete Parameter & Architecture Guide

> **Project:** HDFC Bank Loan Approval Prediction  
> **Target:** Binary Classification (`0 = Rejected`, `1 = Approved`)  
> **Model:** `sklearn.tree.DecisionTreeClassifier`  
> **Role:** Group Leader's Technical Deep-Dive & Parameter Manual

---

## Table of Contents
1. [Core Mathematical Intuition](#1-core-mathematical-intuition)
2. [Complete Parameter Dictionary](#2-complete-parameter-dictionary)
3. [Parameter Sensitivity Analysis (What Happens When You Change...)](#3-parameter-sensitivity-analysis)
4. [Our Chosen Configuration for HDFC Loan Data](#4-our-chosen-configuration-for-hdfc-loan-data)
5. [Pruning Strategies: Pre-Pruning vs. Post-Pruning](#5-pruning-strategies-pre-pruning-vs-post-pruning)
6. [Decision Tree Overfitting Demonstration](#6-decision-tree-overfitting-demonstration)
7. [Top Viva Questions & Model Answers on Parameters](#7-top-viva-questions--model-answers-on-parameters)

---

## 1. Core Mathematical Intuition

A Decision Tree is a non-parametric supervised learning algorithm that creates an interpretable hierarchy of if-else threshold rules by recursively partitioning the feature space.

### Splitting Criteria:

#### 1. Gini Impurity (Default):
Measures the probability of incorrectly classifying a randomly chosen element if it were randomly labeled according to the distribution of labels in the subset.
$$\text{Gini}(D) = 1 - \sum_{k=1}^{K} p_k^2$$
* For binary classification: $\text{Gini} = 1 - (p_{\text{Approved}}^2 + p_{\text{Rejected}}^2)$
* **Purity range:**
  * $\text{Gini} = 0.0$: Perfectly pure node (100% Approved or 100% Rejected).
  * $\text{Gini} = 0.5$: Maximum impurity (50% Approved / 50% Rejected).

#### 2. Entropy & Information Gain:
Measures the disorder or uncertainty in the node:
$$\text{Entropy}(D) = -\sum_{k=1}^{K} p_k \log_2(p_k)$$
* The reduction in entropy after a split on feature $A$ is the **Information Gain**:
$$\text{Gain}(D, A) = \text{Entropy}(D) - \sum_{v \in \text{Values}(A)} \frac{|D_v|}{|D|} \text{Entropy}(D_v)$$
* The algorithm searches through all features and all possible split values and selects the split that maximizes Information Gain (or minimizes weighted Gini impurity).

---

## 2. Complete Parameter Dictionary

| Parameter | Type & Default | What It Controls | Valid Options |
| :--- | :--- | :--- | :--- |
| `criterion` | `{'gini', 'entropy', 'log_loss'}, default='gini'` | The function to measure split quality. | `'gini'`, `'entropy'` |
| `max_depth` | `int, default=None` | **Maximum depth of the tree**. Limits how many consecutive decisions can be chained. | Positive integer: `2`, `3`, `4`, `5`, `10`, or `None` (unlimited) |
| `min_samples_split` | `int or float, default=2` | **Minimum number of samples required to split an internal node**. | Integer $\ge 2$ (e.g., `5`, `10`, `20`) |
| `min_samples_leaf` | `int or float, default=1` | **Minimum number of samples required to be at a leaf node**. | Integer $\ge 1$ (e.g., `2`, `5`, `10`) |
| `max_features` | `int, float or {'sqrt', 'log2'}, default=None` | Number of features to consider when looking for the best split. | `None` (all features), `'sqrt'`, `'log2'` |
| `max_leaf_nodes` | `int, default=None` | Maximum number of terminal leaf nodes allowed. | Integer $> 1$ or `None` |
| `min_impurity_decrease` | `float, default=0.0` | A node will be split only if this split induces a decrease of impurity $\ge$ this value. | Float $\ge 0.0$ (e.g., `0.01`) |
| `class_weight` | `dict or 'balanced', default=None` | Weights associated with target classes. | `None`, `'balanced'` |
| `ccp_alpha` | `non-negative float, default=0.0` | **Cost-Complexity Pruning parameter**. Minimal cost-complexity pruning strength. | Float $\ge 0.0$ (e.g., `0.015`) |
| `random_state` | `int, default=None` | Controls the randomness of the estimator when searching for splits. | Any integer (we use `42`) |

---

## 3. Parameter Sensitivity Analysis

### 3.1 What happens when `max_depth` changes?
This is the single most critical parameter in a Decision Tree.

* **`max_depth=None` (Unconstrained):**
  * **What happens:** The tree expands until every single leaf is completely pure ($\text{Gini}=0$) or contains fewer than `min_samples_split`.
  * **Impact:** **Catastrophic Overfitting**. The tree memorizes individual training applicants.
  * **Empirical metrics:** Training Accuracy = **100%**, but Test Accuracy plunges to **~68-72%**.
* **`max_depth=2` (Too shallow — Decision Stump):**
  * **What happens:** Only 2 consecutive decisions (e.g. 1st: `Credit_History <= 0.5`, 2nd: `TotalIncome <= 3500`).
  * **Impact:** **Underfitting**. The model misses non-linear financial trade-offs.
  * **Empirical metrics:** Accuracy = ~76-78%.
* **`max_depth=4` (Our Tuned Setting):**
  * **What happens:** Allows the tree to consider `Credit_History` at the root, then evaluate `TotalIncome`, `IncomeToLoanRatio`, and `LoanAmountPerMonth` in subsequent branches.
  * **Impact:** **Optimal Generalization**. Test Accuracy = **82.11%**, ROC-AUC = **0.8485**.

---

### 3.2 What happens when `min_samples_split` changes?
* **`min_samples_split=2` (Default):**
  * The tree is allowed to create a new branch even if only 2 applicants are left in a node!
  * **Risk:** Splits on statistical noise and outliers.
* **`min_samples_split=10` (Our Setting):**
  * A node must have at least 10 applicants before the algorithm is allowed to divide it further.
  * **Benefit:** Ensures that every split represents a genuine subgroup of applicants, acting as an effective regularizer.

---

### 3.3 What happens when `min_samples_leaf` changes?
* **`min_samples_leaf=1` (Default):**
  * Allows leaves with a solitary applicant. One single anomaly can create its own branch.
* **`min_samples_leaf=5` (Our Setting):**
  * Guarantees every final decision ("Approved" or "Rejected") is supported by at least 5 similar historical applicants.
  * Smoothes the decision boundary and prevents fragile predictions.

---

### 3.4 What happens when you compare `criterion='gini'` vs `criterion='entropy'`?
* **`'gini'`:** Faster computationally because it involves only squaring probabilities ($1 - \sum p^2$), without computing logarithmic functions.
* **`'entropy'`:** Uses $\log_2$, which is computationally slightly heavier. It penalizes mixed classes slightly more aggressively.
* **In practice:** Both yield nearly identical splits (difference in accuracy is usually $< 0.5\%$). We chose `'gini'` for speed and simplicity.

---

## 4. Our Chosen Configuration for HDFC Loan Data

```python
model = DecisionTreeClassifier(
    criterion='gini',         # Fast, standard impurity measure
    max_depth=4,              # Pre-prunes tree to prevent memorization
    min_samples_split=10,     # Requires at least 10 applicants to split
    min_samples_leaf=5,       # Requires at least 5 applicants per final leaf
    random_state=42           # Deterministic reproducibility
)
```

### Performance on HDFC Dataset:
* **5-Fold Cross-Validation Accuracy:** `79.42% (+/- 4.83%)`
* **Test Set Accuracy:** `82.11%`
* **Precision:** `0.8182`
* **Recall:** `0.9529`
* **F1-Score:** `0.8804`
* **ROC-AUC:** `0.8485`
* **Visual Export:** Generated full graphical decision diagram saved to `reports/decision_tree_structure.png`.

---

## 5. Pruning Strategies: Pre-Pruning vs. Post-Pruning

Pruning is the process of trimming away branches that provide little predictive power to prevent overfitting.

### Method A: Pre-Pruning (Early Stopping — What We Used)
We halt the tree's growth before it becomes overly complex by configuring:
1. `max_depth = 4`
2. `min_samples_split = 10`
3. `min_samples_leaf = 5`

### Method B: Post-Pruning (Cost-Complexity Pruning via `ccp_alpha`)
1. Allow the tree to grow to full depth (`max_depth=None`).
2. Calculate the cost-complexity measure for each sub-tree:
   $$R_\alpha(T) = R(T) + \alpha |T|$$
   Where $R(T)$ is total training error and $|T|$ is the number of terminal leaves.
3. As $\alpha$ increases, smaller sub-trees are favored, pruning away weak leaves.

---

## 6. Decision Tree Overfitting Demonstration

To prove your understanding to the viva examiners, mention this exact experiment:

```python
# Unpruned Tree (Overfitting)
dt_overfit = DecisionTreeClassifier(max_depth=None, random_state=42)
dt_overfit.fit(X_train, y_train)
print("Train Score:", dt_overfit.score(X_train, y_train))  # -> 100.0%
print("Test Score:", dt_overfit.score(X_test, y_test))    # -> ~69.1% (31% gap!)

# Pre-pruned Tree (Our Model)
dt_pruned = DecisionTreeClassifier(max_depth=4, min_samples_split=10, random_state=42)
dt_pruned.fit(X_train, y_train)
print("Train Score:", dt_pruned.score(X_train, y_train))   # -> 82.5%
print("Test Score:", dt_pruned.score(X_test, y_test))     # -> 82.1% (No gap!)
```

---

## 7. Top Viva Questions & Model Answers on Parameters

### Q1: Why did you restrict `max_depth` to 4 instead of leaving it default?
**Model Answer:**  
> *"The default `max_depth=None` allows the tree to branch until every leaf is pure. On a 614-row dataset with noisy income features, this causes severe overfitting—achieving 100% training accuracy but dropping to under 70% on test data. Setting `max_depth=4` restricts the tree to the top 4 layers of decision rules, forcing it to generalize based on primary indicators like `Credit_History` and `TotalIncome`."*

### Q2: What is the mathematical difference between Gini Impurity and Entropy?
**Model Answer:**  
> *"Gini impurity is calculated as $1 - \sum p_i^2$, while Entropy is $-\sum p_i \log_2(p_i)$. Gini measures the probability of misclassification, while Entropy measures information disorder. Computationally, Gini is faster because it does not require calculating logarithms. In practice, both yield very similar splits."*

### Q3: What is the purpose of `min_samples_leaf`?
**Model Answer:**  
> *"`min_samples_leaf` specifies the minimum number of observations required in any terminal node. If set to 1, a single noisy outlier could create its own leaf and dictate an approval rule. By setting `min_samples_leaf=5`, we ensure that any loan approval or rejection rule is backed by at least 5 historical applicants."*

### Q4: Does a Decision Tree require feature scaling (`StandardScaler`)?
**Model Answer:**  
> *"No, Decision Trees are invariant to monotonic feature scaling. A decision tree splits on threshold queries like `Income <= 5000`. If you scale income using StandardScaler to `z <= -0.27`, the exact same split occurs at the exact same rank order. We scaled features primarily because our pipeline also feeds models that strictly require it, such as SVM and Logistic Regression."*
