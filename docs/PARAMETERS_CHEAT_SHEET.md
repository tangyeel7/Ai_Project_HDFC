# Master Parameter Comparison & Viva Quick Reference Cheat Sheet

> **HDFC Bank Loan Approval Prediction System**  
> **All 4 Machine Learning Models Side-by-Side**

---

## 1. Quick Master Comparison Table

| Model | Primary Regularization Parameter | Splitting / Kernel Metric | Overfitting Risk Parameter | Requires Scaling (`StandardScaler`)? | Our Configured Values | Test Accuracy | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | `C` (inverse penalty) | Sigmoid link function | High `C` ($>100$) | **YES (Strictly)** | `C=1.0, solver='lbfgs', max_iter=1000` | **86.18%** | **0.9081** | 0.8000 |
| **Decision Tree** | `min_samples_split`, `min_samples_leaf` | `criterion='gini'` or `'entropy'` | `max_depth=None` (Severe!) | NO (Scale invariant) | `max_depth=4, min_samples_split=10, min_samples_leaf=5` | 82.11% | 0.8804 | 0.8485 |
| **Random Forest** | `min_samples_leaf`, `max_depth` | Ensemble Bagging + Gini | `max_depth=None`, `max_features=None` | NO (Tree ensemble) | `n_estimators=100, max_depth=6, max_features='sqrt'` | 84.55% | 0.8962 | **0.8619** |
| **Support Vector Machine (SVM)** | `C` (slack penalty) | `kernel='rbf'`, `gamma='scale'` | High `C` ($>100$), High `gamma` ($>10$) | **YES (Catastrophic without it)** | `C=1.0, kernel='rbf', gamma='scale', probability=True` | 84.55% | 0.8984 | 0.8034 |

---

## 2. Parameter Tuning Matrix: "What Happens When You Change..."

### Logistic Regression
* **Increase `C` (e.g. 100):** Less regularization $\to$ model fits training noise $\to$ risk of overfitting.
* **Decrease `C` (e.g. 0.01):** High regularization $\to$ coefficients shrunk toward zero $\to$ underfitting.
* **Increase `max_iter` (e.g. 1000):** Guarantees convergence $\to$ eliminates `ConvergenceWarning`.
* **Set `class_weight='balanced'`:** Penalizes false approvals more heavily $\to$ recall of Rejected class rises.

### Decision Tree
* **`max_depth=None`:** Tree memorizes every applicant $\to$ Train Accuracy = 100%, Test Accuracy drops to 68% (**Severe Overfitting**).
* **`max_depth=4`:** Controlled tree $\to$ optimal balance on HDFC dataset (82.11% accuracy).
* **Increase `min_samples_split` (e.g. 10):** Prevents splitting on tiny subgroups $\to$ smoother decision rules.
* **Increase `min_samples_leaf` (e.g. 5):** Ensures every final decision is backed by $\ge 5$ applicants.

### Random Forest
* **Increase `n_estimators` (e.g. from 10 to 100 to 500):** Reduces variance, stabilizes predictions. **Never overfits by adding more trees.**
* **`max_features='sqrt'` (Default):** Only examines $\sqrt{14} \approx 3.7$ features per split $\to$ decorrelates trees $\to$ lower ensemble variance.
* **`max_features=None`:** Uses all features per split $\to$ trees become correlated $\to$ ensemble acts like a single tree.
* **`bootstrap=True`:** Enables random sampling with replacement $\to$ enables Out-Of-Bag (OOB) error estimation.

### Support Vector Machine (SVM)
* **Increase `C` (e.g. 100):** Hard margin $\to$ strictly punishes violations $\to$ narrow margin $\to$ risk of overfitting.
* **Decrease `C` (e.g. 0.01):** Soft margin $\to$ wide margin $\to$ ignores boundary details $\to$ risk of underfitting.
* **Increase `gamma` (e.g. 10):** Gaussian bells become narrow peaks $\to$ circular decision islands $\to$ severe overfitting.
* **`gamma='scale'`:** Automatically sets $\gamma = \frac{1}{n_{\text{features}} \times \text{Var}(X)}$ (robust default).
* **`kernel='rbf'`:** Allows non-linear separating boundaries.

---

## 3. The 30-Second Leader Summary for the Viva
> *"For HDFC Bank's loan eligibility prediction, we developed and compared 4 supervised classification architectures: Logistic Regression, Decision Tree, Random Forest, and SVM. To ensure fair evaluation and prevent data leakage, we applied median and mode imputation, engineered domain banking features (Total Income, EMI load, and Debt Service ratio), scaled continuous features via StandardScaler, and used 5-fold stratified cross-validation. While Logistic Regression achieved the highest test accuracy of 86.18% due to the strong linear separation offered by Credit History, Random Forest delivered the highest ROC-AUC of 0.8619, proving to be the most robust probabilistic ranker for risk-adjusted lending decisions."*
