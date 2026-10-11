# Logistic Regression — Complete Parameter & Architecture Guide

> **Project:** HDFC Bank Loan Approval Prediction  
> **Target:** Binary Classification (`0 = Rejected`, `1 = Approved`)  
> **Model:** `sklearn.linear_model.LogisticRegression`  
> **Role:** Group Leader's Technical Deep-Dive & Parameter Manual

---

## Table of Contents
1. [Core Mathematical Intuition](#1-core-mathematical-intuition)
2. [Complete Parameter Dictionary](#2-complete-parameter-dictionary)
3. [Parameter Sensitivity Analysis (What Happens When You Change...)](#3-parameter-sensitivity-analysis)
4. [Our Chosen Configuration for HDFC Loan Data](#4-our-chosen-configuration-for-hdfc-loan-data)
5. [Feature Coefficients & Banking Odds Ratio](#5-feature-coefficients--banking-odds-ratio)
6. [Overfitting vs. Underfitting in Logistic Regression](#6-overfitting-vs-underfitting-in-logistic-regression)
7. [Top Viva Questions & Model Answers on Parameters](#7-top-viva-questions--model-answers-on-parameters)

---

## 1. Core Mathematical Intuition

Despite the name having "regression", Logistic Regression is a **probabilistic linear classification algorithm**.

### The Math Step-by-Step:
1. **Linear Equation (Log-Odds / Logit):**
   $$z = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_n x_n = \mathbf{w}^T \mathbf{x} + b$$
   Where $\mathbf{x}$ are applicant features (Credit History, Total Income, Loan Amount, etc.), $\mathbf{w}$ are learned weights (coefficients), and $b$ is the bias (intercept).

2. **The Sigmoid (Logistic) Function:**
   To turn the unbounded real number $z \in (-\infty, +\infty)$ into a valid probability $P \in [0, 1]$:
   $$P(Y=1 \mid \mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-z}} = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

3. **Decision Rule:**
   $$\hat{y} = \begin{cases} 1 \text{ (Approved)}, & \text{if } P(Y=1 \mid \mathbf{x}) \ge 0.50 \\ 0 \text{ (Rejected)}, & \text{if } P(Y=1 \mid \mathbf{x}) < 0.50 \end{cases}$$

4. **Loss Function (Binary Cross-Entropy / Log Loss):**
   $$J(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y^{(i)} \log(\hat{p}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{p}^{(i)}) \right] + \text{Regularization Term}$$

---

## 2. Complete Parameter Dictionary

| Parameter | Type & Default | What It Controls | Valid Options |
| :--- | :--- | :--- | :--- |
| `C` | `float, default=1.0` | **Inverse of regularization strength**. Smaller values specify stronger regularization. | Any positive float: `0.001`, `0.01`, `0.1`, `1.0`, `10.0`, `100.0` |
| `penalty` | `{'l1', 'l2', 'elasticnet', None}, default='l2'` | Specifies the norm of the penalty applied to prevent weights from growing too large. | `'l2'` (Ridge), `'l1'` (Lasso), `'elasticnet'`, `None` |
| `solver` | `str, default='lbfgs'` | Algorithm used to optimize the loss function (find optimal weights $\mathbf{w}$). | `'lbfgs'`, `'liblinear'`, `'newton-cg'`, `'sag'`, `'saga'` |
| `max_iter` | `int, default=100` | Maximum number of gradient iterations taken for the solver to converge. | Positive integer: `100`, `500`, `1000`, `5000` |
| `tol` | `float, default=1e-4` | Tolerance for stopping criteria. Optimization stops if improvement is smaller than `tol`. | Small positive float: `1e-3`, `1e-4`, `1e-5` |
| `class_weight` | `dict or 'balanced', default=None` | Weights associated with classes. Useful for handling class imbalance. | `None`, `'balanced'`, or custom dict `{0: 2.0, 1: 1.0}` |
| `fit_intercept` | `bool, default=True` | Specifies if a constant bias term $b$ should be added to the decision function. | `True`, `False` |
| `random_state` | `int, default=None` | Seed used by random number generator (relevant when solver is `'sag'`, `'saga'`, or `'liblinear'`). | Any integer (we use `42`) |

---

## 3. Parameter Sensitivity Analysis

### 3.1 What happens when you change `C`?
`C = 1 / \lambda`, where $\lambda$ is the penalty multiplier.
* **When `C` is very small (`0.001` to `0.01`):**
  * **Effect:** Heavy regularization. Weights $\mathbf{w}$ are shrunk severely toward zero.
  * **Risk:** **Underfitting**. The model becomes too stiff and cannot capture real relationships between income and approval. Test and train accuracy both drop.
* **When `C = 1.0` (Default / Optimal):**
  * **Effect:** Balances empirical error on training data with coefficient shrinkage.
  * **Result:** Achieves **86.18% test accuracy** on our HDFC loan data without overfitting.
* **When `C` is very large (`100` to `10000`):**
  * **Effect:** Regularization is essentially turned off. The optimizer attempts to fit every noisy applicant perfectly.
  * **Risk:** **Overfitting** or numerical instability if features are collinear.

### 3.2 What happens when you change `penalty`?
* **`penalty='l2'` (Ridge Penalty, $\sum w_j^2$):**
  * Shrinks all coefficients toward zero, but rarely sets any exactly to zero.
  * Best when all features are potentially somewhat useful.
* **`penalty='l1'` (Lasso Penalty, $\sum |w_j|$):**
  * Can force coefficients of irrelevant features to **exactly zero**.
  * Performs automated feature selection.
  * Requires `solver='liblinear'` or `solver='saga'`.

### 3.3 What happens when you change `solver`?
* **`'lbfgs'` (Limited-memory BFGS, Quasi-Newton):**
  * Default and best for small-to-medium tabular datasets like ours (614 rows). Fast and robust.
* **`'liblinear'`:**
  * Coordinate descent algorithm. Excellent for small datasets and supports L1 penalty.
* **`'saga'`:**
  * Stochastic Average Gradient descent. Necessary for very large datasets ($>100,000$ rows) or when combining ElasticNet with large data.

### 3.4 What happens when you change `max_iter`?
* **If `max_iter=100`:** On unscaled or complex datasets, you will often get:  
  `ConvergenceWarning: lbfgs failed to converge (status=1): STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.`
* **If `max_iter=1000`:** Guarantees the gradient solver has sufficient iterations to reach the true global loss minimum.

### 3.5 What happens when you change `class_weight`?
Our dataset has class imbalance (~69% Approved vs ~31% Rejected).
* **`class_weight=None`:**
  * Every applicant has equal error loss.
  * Achieves high recall on Approved (98.8%) and 86.18% overall accuracy.
* **`class_weight='balanced'`:**
  * Adjusts weights inversely proportional to class frequencies:  
    $w_0 = \frac{N}{2 \times N_0} \approx 1.6$, $w_1 = \frac{N}{2 \times N_1} \approx 0.72$
  * **Effect:** Increases penalty for misclassifying Rejected applications (Type I error).
  * **Trade-off:** Recall of Rejected applicants increases (more bad loans caught), but some good applicants are rejected (overall accuracy may slightly decrease to ~80-82%).

---

## 4. Our Chosen Configuration for HDFC Loan Data

```python
model = LogisticRegression(
    C=1.0,               # Balanced regularization
    solver='lbfgs',      # Quasi-Newton optimization algorithm
    max_iter=1000,       # Ample headroom for full mathematical convergence
    random_state=42      # Complete reproducibility
)
```

### Performance on HDFC Dataset:
* **5-Fold Cross-Validation Accuracy:** `79.63% (+/- 4.21%)`
* **Test Set Accuracy:** `86.18%`
* **Precision:** `0.8400`
* **Recall (Sensitivity):** `0.9882` (Catches 98.8% of creditworthy customers)
* **F1-Score:** `0.9081`
* **ROC-AUC:** `0.8000`

---

## 5. Feature Coefficients & Banking Odds Ratio

Because features were standardized with `StandardScaler`, the magnitude of each coefficient directly represents its statistical predictive power:

$$\text{Odds Ratio} = e^{w_j}$$

| Feature | Coefficient ($w_j$) | Odds Ratio ($e^{w_j}$) | Financial Interpretation |
| :--- | :--- | :--- | :--- |
| **`Credit_History`** | **+3.138** | **23.06x** | Having satisfactory credit history multiplies the odds of approval by **23 times**! |
| **`Married`** | **+0.494** | **1.64x** | Married applicants have 64% higher odds of approval (dual income stability). |
| **`Dependents`** | **+0.104** | **1.11x** | Slight positive effect when balanced by income. |
| **`Property_Area`** | **+0.064** | **1.07x** | Semiurban locations have higher approval probability. |
| **`ApplicantIncome`** | **+0.026** | **1.03x** | Positive individual income contribution. |
| **`Education`** | **-0.411** | **0.66x** | Non-graduate status reduces approval odds by 34%. |
| **`Gender`** | **-0.211** | **0.81x** | Minor demographic adjustment. |
| **`Self_Employed`** | **-0.182** | **0.83x** | Self-employed individuals have slightly higher income volatility risk. |

---

## 6. Overfitting vs. Underfitting in Logistic Regression

| Scenario | Symptom | How to Detect in Code | Fix |
| :--- | :--- | :--- | :--- |
| **Overfitting** | Train Accuracy = 95%, Test Accuracy = 72% | Large gap between `model.score(X_train, y_train)` and `model.score(X_test, y_test)` | Decrease `C` (e.g. `C=0.1`), use `penalty='l1'`, remove collinear features |
| **Underfitting** | Train Accuracy = 68%, Test Accuracy = 67% | Both training and testing performance are poor | Increase `C` (e.g. `C=10`), add polynomial/interaction features, check scaling |
| **Optimal Fit** | Train CV = 79.6%, Test = 86.2% | Consistent scores across 5 cross-validation folds | **Our current model!** |

---

## 7. Top Viva Questions & Model Answers on Parameters

### Q1: Why is feature scaling (`StandardScaler`) mandatory for Logistic Regression?
**Model Answer:**  
> *"In Logistic Regression, the regularization term $\frac{1}{2C} \sum w_j^2$ penalizes all feature weights equally. If `ApplicantIncome` is in thousands (e.g., 50,000) and `Credit_History` is in units of 0 or 1, the raw coefficient for income would naturally be tiny (e.g., 0.0001) while credit history's coefficient would be large. The penalty would disproportionately punish credit history while leaving income unregularized. StandardScaler puts all features on the same unit scale (mean=0, std=1), ensuring fair regularization and fast gradient convergence."*

### Q2: What is the role of parameter `C`?
**Model Answer:**  
> *"`C` is the inverse regularization parameter. A high `C` gives less weight to the penalty, allowing the model to fit training data tighter (risk of overfitting). A low `C` increases penalty strength, shrinking weights towards zero and encouraging a simpler decision boundary (risk of underfitting). We used `C=1.0`, which provided optimal generalization on unseen test data."*

### Q3: What happens if `max_iter` is set too low, say 10?
**Model Answer:**  
> *"The optimization solver (`lbfgs`) will terminate before finding the minimum of the convex cross-entropy loss surface. Scikit-learn will throw a `ConvergenceWarning`, and the resulting weights $\mathbf{w}$ will be suboptimal, leading to degraded classification accuracy."*

### Q4: How does Logistic Regression handle class imbalance through parameters?
**Model Answer:**  
> *"Through the `class_weight='balanced'` parameter. When enabled, it computes weights inversely proportional to class frequencies using $w_j = \frac{n\_samples}{n\_classes \times \text{count}(y_j)}$. This increases the loss contribution of misclassifying the minority class (Rejected), shifting the decision threshold to catch more high-risk loans."*
