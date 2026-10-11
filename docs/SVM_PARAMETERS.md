# Support Vector Machine (SVM) — Complete Parameter & Architecture Guide

> **Project:** HDFC Bank Loan Approval Prediction  
> **Target:** Binary Classification (`0 = Rejected`, `1 = Approved`)  
> **Model:** `sklearn.svm.SVC` (Support Vector Classifier)  
> **Role:** Group Leader's Technical Deep-Dive & Parameter Manual

---

## Table of Contents
1. [Core Mathematical Intuition](#1-core-mathematical-intuition)
2. [Complete Parameter Dictionary](#2-complete-parameter-dictionary)
3. [Parameter Sensitivity Analysis (What Happens When You Change...)](#3-parameter-sensitivity-analysis)
4. [Our Chosen Configuration for HDFC Loan Data](#4-our-chosen-configuration-for-hdfc-loan-data)
5. [Support Vector Mechanics on HDFC Dataset](#5-support-vector-mechanics-on-hdfc-dataset)
6. [Why Feature Scaling is Critical for SVM](#6-why-feature-scaling-is-critical-for-svm)
7. [Top Viva Questions & Model Answers on Parameters](#7-top-viva-questions--model-answers-on-parameters)

---

## 1. Core Mathematical Intuition

Support Vector Machine (SVM) is a geometric maximum-margin classifier. Its goal is to find the **optimal separating hyperplane** that maximizes the distance (margin) between the decision boundary and the nearest data points of each class (the **support vectors**).

### 1. The Maximum Margin Objective:
For linearly separable data:
$$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 \quad \text{subject to} \quad y_i (\mathbf{w}^T \mathbf{x}_i + b) \ge 1, \quad \forall i$$
The geometric margin width is $\frac{2}{\|\mathbf{w}\|}$. Minimizing $\frac{1}{2} \|\mathbf{w}\|^2$ is equivalent to **maximizing the margin width**.

### 2. Soft Margin & Slack Variables ($\xi_i$):
Because real-world banking applicants overlap (some individuals with good credit still default, and vice versa), we introduce slack variables $\xi_i \ge 0$:
$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \left( \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{N} \xi_i \right) \quad \text{subject to} \quad y_i (\mathbf{w}^T \mathbf{x}_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$
* The parameter **$C$** directly controls the penalty for constraint violations ($\xi_i$).

### 3. The Kernel Trick (Non-Linear Mapping):
When data cannot be separated by a flat plane in original feature space $\mathbb{R}^d$, the **Kernel Trick** maps inputs into a higher-dimensional Hilbert space $\Phi(\mathbf{x})$ without explicitly computing high-dimensional coordinates:
$$K(\mathbf{x}_i, \mathbf{x}_j) = \langle \Phi(\mathbf{x}_i), \Phi(\mathbf{x}_j) \rangle$$

#### Key Kernel Functions:
* **Radial Basis Function (RBF / Gaussian) Kernel (Default):**
  $$K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left( -\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2 \right)$$
* **Linear Kernel:**
  $$K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T \mathbf{x}_j$$
* **Polynomial Kernel:**
  $$K(\mathbf{x}_i, \mathbf{x}_j) = (\gamma \mathbf{x}_i^T \mathbf{x}_j + r)^d$$

---

## 2. Complete Parameter Dictionary

| Parameter | Type & Default | What It Controls | Valid Options |
| :--- | :--- | :--- | :--- |
| `C` | `float, default=1.0` | **Regularization / Slack penalty parameter**. Trade-off between margin maximization and training error minimization. | Positive float: `0.01`, `0.1`, `1.0`, `10.0`, `100.0` |
| `kernel` | `{'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}, default='rbf'` | **Specifies kernel type** to project data into higher-dimensional decision space. | `'rbf'`, `'linear'`, `'poly'`, `'sigmoid'` |
| `gamma` | `{'scale', 'auto'} or float, default='scale'` | **Kernel coefficient for `'rbf'`, `'poly'`, and `'sigmoid'`. Defines the radius of influence of each support vector. | `'scale'` ($1 / (n \times \text{Var})$), `'auto'` ($1 / n$), or float (e.g., `0.01`, `1.0`) |
| `degree` | `int, default=3` | Degree of the polynomial kernel function (`'poly'`). Ignored by other kernels. | Positive integer: `2`, `3`, `4`, `5` |
| `coef0` | `float, default=0.0` | Independent term $r$ in polynomial and sigmoid kernels. | Float (e.g. `0.0`, `1.0`) |
| `shrinking` | `bool, default=True` | Whether to use the shrinking heuristic in Sequential Minimal Optimization (SMO) solver. | `True`, `False` |
| `probability` | `bool, default=False` | Whether to enable probability estimates using Platt scaling. | `True`, `False` |
| `tol` | `float, default=1e-3` | Tolerance for stopping criterion in the optimization solver. | Positive float: `1e-3`, `1e-4` |
| `class_weight` | `dict or 'balanced', default=None` | Multipliers $C_k = C \times w_k$ for handling class imbalance. | `None`, `'balanced'`, or dict |
| `max_iter` | `int, default=-1` | Hard limit on iterations within solver, or `-1` for no limit. | `-1` or positive int |
| `random_state` | `int, default=None` | Seed for internal probability calibration (Platt scaling). | Any integer (we use `42`) |

---

## 3. Parameter Sensitivity Analysis

### 3.1 What happens when `C` changes?
* **`C` is very small (`0.01` to `0.1`) — Large Margin / Soft Penalty:**
  * The optimizer tolerates many misclassifications ($\xi_i$) in exchange for a wide, simple margin.
  * **Risk:** **Underfitting**. The model may ignore subtle income boundaries and predict the majority class.
* **`C = 1.0` (Our Tuned Setting):**
  * Balances misclassification penalties with boundary complexity.
  * **Result:** **84.55% test accuracy**, robust to noisy applicant records.
* **`C` is very large (`100` to `1000`) — Hard Margin / Strict Penalty:**
  * The model imposes an immense penalty for every misclassified applicant. The margin shrinks tightly around training points.
  * **Risk:** **Severe Overfitting**. Any noisy applicant or outlier bends the decision boundary radically.

---

### 3.2 What happens when `gamma` changes (for RBF kernel)?
`gamma` ($\gamma$) determines how far the influence of a single training point reaches.
$$\gamma = \frac{1}{2\sigma^2}$$

* **`gamma='scale'` (Default & Recommended):**
  * Automatically sets $\gamma = \frac{1}{n_{\text{features}} \times \text{Var}(X)}$.
  * Gives a well-behaved, smooth Gaussian bell curve around support vectors.
* **`gamma` is very small (`0.001`):**
  * The Gaussian bell has a huge radius $\sigma$. A support vector's influence spreads across the entire dataset.
  * The boundary becomes nearly linear, acting like a flat plane.
* **`gamma` is very high (`10.0` or `100.0`):**
  * The Gaussian radius is tiny. Each support vector creates a tiny, isolated "island" of decision boundary around itself.
  * **Result:** Training accuracy approaches 100%, but test accuracy collapses because any unseen applicant falling outside the islands is misclassified.

---

### 3.3 What happens when `kernel` changes?
* **`kernel='rbf'` (Our Setting):**
  * Universal function approximator. Can capture complex non-linear curves (e.g. high loan amount acceptable *only if* credit history is 1 AND income is above threshold).
  * Outperforms linear kernel when feature interactions exist.
* **`kernel='linear'`:**
  * Simple flat hyperplane. Very fast, but identical in geometric shape to Logistic Regression.
* **`kernel='poly'`:**
  * Fits polynomial curves. Highly sensitive to `degree` and `C`; easily overfits on small sample sizes ($N=614$).

---

### 3.4 What is `probability=True` and why is it important?
Standard SVM calculates a distance to the hyperplane $f(\mathbf{x}) = \mathbf{w}^T \Phi(\mathbf{x}) + b$, not a probability.
* When `probability=True`, scikit-learn uses **Platt Scaling**: it fits an internal 5-fold cross-validated logistic regression on the SVM decision values:
  $$P(Y=1 \mid \mathbf{x}) = \frac{1}{1 + \exp(A \cdot f(\mathbf{x}) + B)}$$
* **Why we enabled it:** This allows us to output calibrated approval probabilities (e.g., 85.2% confidence) and compute **ROC-AUC curves**.

---

## 4. Our Chosen Configuration for HDFC Loan Data

```python
model = SVC(
    C=1.0,               # Balanced soft-margin regularization
    kernel='rbf',        # Non-linear Radial Basis Function
    gamma='scale',       # Auto-calibrated kernel bandwidth
    probability=True,    # Platt scaling for probability estimates & ROC-AUC
    random_state=42      # Reproducibility
)
```

### Performance on HDFC Dataset:
* **5-Fold Cross-Validation Accuracy:** `79.22% (+/- 4.75%)`
* **Test Set Accuracy:** `84.55%`
* **Precision:** `0.8235`
* **Recall:** `0.9882` (Catches 98.8% of creditworthy customers)
* **F1-Score:** `0.8984`
* **ROC-AUC:** `0.8034`

---

## 5. Support Vector Mechanics on HDFC Dataset

In our trained model:
* **Total Training Instances:** 491
* **Total Support Vectors Identified:** **292** (59.5% of data)
  * Support Vectors for Class 0 (Rejected): **129**
  * Support Vectors for Class 1 (Approved): **163**

### Leader's Viva Explanation:
> *"Only the support vectors determine the position of the separating hyperplane. If you deleted the other 199 applicants who lie safely away from the margin, the trained SVM model would remain 100% identical! Support vectors are the critical borderline applications where loan decisions are hardest to make."*

---

## 6. Why Feature Scaling is Critical for SVM

Unlike Decision Trees, **SVM is catastrophically sensitive to feature scaling**:
* SVM computes Euclidean distances between vectors: $\|\mathbf{x}_i - \mathbf{x}_j\|^2 = \sum_k (x_{ik} - x_{jk})^2$.
* If `ApplicantIncome` is ₹50,000 and `Credit_History` is 1.0:
  * Difference in income: $(50,000 - 45,000)^2 = 25,000,000$
  * Difference in credit: $(1 - 0)^2 = 1$
* Without `StandardScaler`, the distance calculation is **99.9999% dominated by income**, completely masking the most important banking feature (`Credit_History`).
* When we scale features to mean=0, std=1, every feature contributes proportionally according to its variance.

---

## 7. Top Viva Questions & Model Answers on Parameters

### Q1: What is the physical/geometric meaning of the parameter $C$ in SVM?
**Model Answer:**  
> *"$C$ is the penalty weight on the slack variables $\xi_i$ in the soft-margin objective function. Geometrically, it controls the trade-off between margin width and classification errors. A small $C$ allows more margin violations (softer margin, simpler boundary, lower variance). A large $C$ strictly penalizes any point entering the margin (harder margin, complex boundary, higher risk of overfitting). We used $C=1.0$."*

### Q2: How does the RBF kernel map data into infinite dimensions?
**Model Answer:**  
> *"The RBF kernel uses the exponential function $\exp(-\gamma \|\mathbf{x} - \mathbf{z}\|^2)$. By Taylor series expansion, $e^u = 1 + u + \frac{u^2}{2!} + \frac{u^3}{3!} + \dots$, the exponential can be represented as an infinite sum of polynomial basis terms. The kernel trick computes the inner product in this infinite-dimensional Hilbert space directly using a simple scalar distance computation in original input space."*

### Q3: What happens if `gamma` is set too high in an RBF kernel?
**Model Answer:**  
> *"When `gamma` is extremely large, the radius of the Gaussian kernel shrinks to near zero. Each support vector exerts influence only in its immediate vicinity, creating tiny circular decision bubbles around individual training points. This causes severe overfitting—training accuracy hits 100%, but test accuracy collapses because any unseen applicant falls in the unassigned background space."*

### Q4: Why does SVM take longer to train on very large datasets compared to Logistic Regression?
**Model Answer:**  
> *"Training an SVM requires solving a quadratic programming optimization problem (Sequential Minimal Optimization). Its computational complexity scales between $O(N^2)$ and $O(N^3)$ with respect to the number of training samples $N$. For our 614-row dataset, training takes less than 0.1 seconds, but on 1,000,000 rows, standard SVM becomes computationally prohibitive without specialized linear approximations."*
