# Customer Behaviour Analysis: Predicting High-Spending and Low-Spending Customers Using Logistic Regression

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2+-F7931E.svg)](https://scikit-learn.org/)
[![Subject](https://img.shields.io/badge/Subject-Statistics%20for%20AI%20%26%20Data%20Science-success.svg)]()

> **Academic Project for Third-Year Engineering**  
> **Course:** Statistics for Artificial Intelligence & Data Science  
> **Dataset:** Mall Customer Segmentation Dataset  
> **Benchmark Exchange Rate:** 1 USD = 83.0 INR (Fixed)

---

## 1. Project Title
**Customer Behaviour Analysis: Predicting High-Spending and Low-Spending Customers Using Logistic Regression**

---

## 2. Abstract
This academic project investigates the statistical foundations of customer behaviour segmentation and binary classification using the **Mall Customer Dataset**. Rather than treating machine learning as an uninterpretable black box, the project bridges formal parametric and non-parametric statistical methods—including descriptive dispersion analysis, Shapiro-Wilk normality testing, two-sample Student’s and Welch’s $t$-tests, One-Way Analysis of Variance (ANOVA), Kruskal-Wallis tests, Ordinary Least Squares (OLS) regression, bootstrap uncertainty estimation ($B=1000$), and Principal Component Analysis (PCA)—with **Binary Logistic Regression**.

Customers are classified into **High Spenders** ($\text{Spending Score} \ge 50$) and **Low Spenders** ($\text{Spending Score} < 50$) using demographic and financial predictors (`Age`, `Gender`, and `Annual Income`). Feature scaling comparisons demonstrate robust convergence across Standardization and Normalization. The final Logistic Regression model achieves a hold-out test accuracy of **72.50%** (ROC-AUC = **0.680**) and a 5-fold stratified cross-validation mean accuracy of **65.00% ± 5.86%**, identifying customer age as the most statistically significant negative predictor of spending propensity ($\beta = -0.591, \text{Odds Ratio} = 0.554, p = 0.0005$).

---

## 3. Problem Statement
Retail mall management and commercial marketing teams frequently struggle with allocating promotional resources efficiently. Blind mass-marketing campaigns waste capital on low-spending patrons while missing opportunities to engage potential high-value customers. 

The objective is to establish an interpretable, mathematically sound predictive framework that:
1. Validates whether demographic indicators (Age, Gender) or financial capacity (Annual Income) statistically differentiate consumer spending behaviour.
2. Formulates an inferential classification model using **Logistic Regression** without leaking target information (`Spending Score` is strictly excluded from predictive features).
3. Evaluates model stability and uncertainty through bootstrap resampling and $k$-fold cross-validation.

---

## 4. Objectives
* **Data Hygiene & Preprocessing:** Clean, validate, and convert income to Indian Rupees (INR) using a fixed benchmark rate ($1\text{ USD} = 83.0\text{ INR}$) and Indian numbering conventions (e.g., ₹8,50,000, ₹12,50,000).
* **Descriptive & Distributional Analysis:** Calculate central tendency, dispersion, quartiles, IQR, skewness, excess kurtosis, and test normality via Shapiro-Wilk.
* **Hypothesis Testing:** Conduct two-sample $t$-tests and Mann-Whitney $U$ tests to evaluate whether spending scores differ significantly between male and female shoppers ($\alpha = 0.05$).
* **Variance Analysis (ANOVA):** Evaluate spending score variation across Low, Medium, and High income tiers using One-Way ANOVA and Tukey HSD post-hoc testing.
* **Linear Association vs Causation:** Compute Pearson and Spearman correlation matrices and fit an OLS regression model with residual diagnostics.
* **Feature Scaling Evaluation:** Compare raw features, Standardization (`StandardScaler`), and Normalization (`MinMaxScaler`).
* **Logistic Regression Modeling:** Derive Maximum Likelihood estimates, odds ratios ($e^\beta$), and asymptotic $p$-values.
* **Uncertainty Quantification:** Implement Non-parametric Bootstrap Resampling ($B=1000$) to construct 95% confidence intervals.
* **Dimensionality Reduction:** Perform Principal Component Analysis (PCA) to project the 3D feature space onto 2 orthogonal principal components.
* **Cross-Validation & Evaluation:** Assess generalization using 5-Fold Stratified Cross-Validation, Confusion Matrix, and ROC-AUC curves.
* **Interactive Dashboard:** Build a full-stack **Streamlit** dashboard for exploration and real-time customer spending prediction.

---

## 5. Dataset Description
The dataset comprises $N = 200$ customer observations from mall consumer records:

| Column Header | Original Type | Description | Unit / Range |
| :--- | :--- | :--- | :--- |
| `CustomerID` | Integer | Unique identifier per customer | $1 - 200$ |
| `Gender` | Categorical | Customer gender | Female / Male |
| `Age` | Integer | Customer chronological age | $18 - 70$ years |
| `Annual Income (k$)` | Integer | Annual income in thousands of USD | $\$15\text{k} - \$137\text{k}$ |
| `Annual_Income_INR` | Float (Engineered) | Annual income in Indian Rupees | $₹12,45,000 - ₹1,13,71,000$ |
| `Spending Score (1-100)` | Integer | Mall-assigned behavioural score | $1 - 100$ |
| `Spending_Category` | Binary Target | $1 = \text{High Spender } (\ge 50), 0 = \text{Low Spender } (< 50)$ | $0 \text{ or } 1$ |

> **Target Class Balance:** High Spenders = $102$ ($51.0\%$), Low Spenders = $98$ ($49.0\%$). Balanced binary target distribution.

---

## 6. Technologies Used
* **Language:** Python 3.11
* **Data Manipulation:** `pandas`, `numpy`
* **Statistical Computing:** `scipy.stats`, `statsmodels`
* **Machine Learning:** `scikit-learn`
* **Visualization:** `matplotlib`, `seaborn`
* **Web Dashboard:** `streamlit`
* **Notebook Environment:** `jupyter`, `nbformat`, `nbclient`

---

## 7. Data Preprocessing Pipeline
1. **Integrity Validation:** Verified zero missing values (`0 nulls`) and zero duplicate records across all 200 entries.
2. **Gender Numerical Encoding:** Binary mapping: $\text{Female} = 0$, $\text{Male} = 1$.
3. **Currency Conversion:** 
   $$\text{Annual\_Income\_INR} = \text{Annual Income (k\$)} \times 1000 \times 83.0$$
4. **Indian Currency Formatting:** Implemented custom formatter producing strings formatted with standard Indian commas (e.g., ₹5,00,000, ₹10,00,000, ₹1,13,71,000).
5. **Target Definition:** $\text{Spending\_Category} = \mathbb{I}(\text{Spending Score} \ge 50)$.
6. **Feature Isolation (Anti-Leakage):** Predictive matrix $X = [\text{Age}, \text{Gender\_Encoded}, \text{Annual Income (k\$)}]$. `Spending Score` is strictly excluded from $X$.
7. **Stratified Train-Test Partition:** 80/20 train-test split ($n_{\text{train}} = 160$, $n_{\text{test}} = 40$) preserving class balance.
8. **Scaling Pipeline:** Fitted strictly on $X_{\text{train}}$ and transformed on $X_{\text{test}}$ to prevent data snooping.

---

## 8. Exploratory Data Analysis (EDA) Highlights
* **Age Profile:** Mean age = $38.85 \pm 13.97$ years (Median = $36.0$).
* **Income Profile:** Mean annual income = $₹50,26,480 \pm ₹21,79,918$ ($\$60.56\text{k} \pm \$26.26\text{k}$).
* **Spending Profile:** Mean spending score = $50.20 \pm 25.82$ (Median = $50.0$).
* **Age vs Spending Relationship:** Negative association ($r = -0.327, p < 0.0001$). Shoppers above 40 years rarely exhibit spending scores exceeding 60.
* **Income vs Spending Relationship:** Bivariate scatter shows 5 non-linear clusters (Low Income/Low Spending, Low Income/High Spending, Moderate Income/Moderate Spending, High Income/Low Spending, High Income/High Spending).

---

## 9. Statistical Methods & Key Findings

### 9.1 Normality Assessment (Shapiro-Wilk Test)
* **Age:** $W = 0.943, p = 2.71 \times 10^{-6}$ (Non-normal, right-skewed).
* **Annual Income:** $W = 0.978, p = 3.54 \times 10^{-3}$ (Non-normal, modest right skew).
* **Spending Score:** $W = 0.971, p = 2.42 \times 10^{-4}$ (Non-normal, platykurtic).

### 9.2 Hypothesis Test: Gender vs Spending Score
* **$H_0$:** $\mu_{\text{Male}} = \mu_{\text{Female}}$ vs **$H_1$:** $\mu_{\text{Male}} \neq \mu_{\text{Female}}$ ($\alpha = 0.05$).
* **Levene's Equal Variance Test:** $F = 0.437, p = 0.509$ (Equal variance assumed).
* **Two-Sample Independent $t$-Test:** $t = -0.8190, p = 0.4137$.
* **Mann-Whitney $U$ Test:** $U = 4580.0, p = 0.3804$.
* **Decision:** **Fail to Reject $H_0$**. Gender differences in spending score ($\text{Male Mean} = 48.51$ vs $\text{Female Mean} = 51.53$) are not statistically significant.

### 9.3 One-Way ANOVA: Income Tiers vs Spending Score
* **Tiers:** Low ($< ₹33.2\text{L}$), Medium ($₹33.2\text{L} - ₹66.4\text{L}$), High ($> ₹66.4\text{L}$).
* **$H_0$:** $\mu_{\text{Low}} = \mu_{\text{Medium}} = \mu_{\text{High}}$ ($\alpha = 0.05$).
* **ANOVA Table:** $SS_{\text{Between}} = 14.88, df=2, MS=7.44$; $SS_{\text{Within}} = 132549.91, df=197, MS=672.84$.
* **$F$-Statistic:** $F = 0.0112, p = 0.9889$.
* **Kruskal-Wallis Test:** $H = 0.0878, p = 0.9570$.
* **Decision:** **Fail to Reject $H_0$**. Mean spending score does not differ significantly across income brackets ($50.22$ vs $50.09$ vs $50.41$).

### 9.4 Correlation Analysis & OLS Linear Regression
* **Pearson Correlation:** $\text{Age} \leftrightarrow \text{Score} = -0.327$ ($p = 2.22 \times 10^{-6}$); $\text{Income} \leftrightarrow \text{Score} = 0.010$ ($p = 0.890$); $\text{Age} \leftrightarrow \text{Income} = -0.012$ ($p = 0.862$).
* **OLS Linear Regression:** $\text{Score} = 73.35 - 0.605 \cdot \text{Age} + 0.006 \cdot \text{Income}$.
* **Goodness of Fit:** $R^2 = 0.1071$, $F(2, 197) = 11.82$ ($p = 1.45 \times 10^{-5}$), $\text{RMSE} = 24.22$.

---

## 10. Machine Learning: Logistic Regression

### 10.1 Feature Scaling Comparison
| Scaling Method | Hold-out Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **No Scaling** | 72.50% | 73.68% | 70.00% | 71.79% | 0.6800 |
| **Standardization (`StandardScaler`)** | 72.50% | 73.68% | 70.00% | 71.79% | 0.6800 |
| **Normalization (`MinMaxScaler`)** | 72.50% | 73.68% | 70.00% | 71.79% | 0.6800 |

### 10.2 Parameter Estimates & Odds Ratios
$$\log\left(\frac{P(\text{High Spender})}{1 - P(\text{High Spender})}\right) = 0.0460 - 0.5911 \cdot Z_{\text{Age}} - 0.0096 \cdot Z_{\text{Gender}} - 0.0120 \cdot Z_{\text{Income}}$$

| Feature | Coefficient ($\beta$) | Odds Ratio ($e^\beta$) | Std Error | $p$-value | Academic Interpretation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Intercept ($\beta_0$)** | 0.0460 | 1.0471 | 0.163 | 0.7781 | Baseline log-odds at mean feature values. |
| **Age** | -0.5911 | 0.5537 | 0.171 | **0.0005** | For each $1\sigma$ increase in Age, odds of High Spending decrease by **44.63%**. |
| **Gender (Male=1)** | -0.0096 | 0.9904 | 0.151 | 0.9494 | Negligible effect; not statistically significant. |
| **Annual Income** | -0.0120 | 0.9881 | 0.148 | 0.9357 | Linear effect negligible due to non-linear spending clusters. |

---

## 11. Model Evaluation & Cross-Validation

### 11.1 Hold-out Test Confusion Matrix ($n=40$)
* **True Positives (TP):** $14$ (High Spenders correctly classified)
* **True Negatives (TN):** $15$ (Low Spenders correctly classified)
* **False Positives (FP) [Type I Error]:** $5$ (Low Spenders misclassified as High Spenders)
* **False Negatives (FN) [Type II Error]:** $6$ (High Spenders misclassified as Low Spenders)
* **Metrics:** Accuracy = **72.50%**, Precision = **73.68%**, Recall = **70.00%**, Specificity = **75.00%**, F1 = **71.79%**, ROC-AUC = **0.6800**.

### 11.2 5-Fold Stratified Cross-Validation
| Fold | Validation Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 60.00% | 59.09% | 65.00% | 61.90% | 0.5825 |
| **Fold 2** | 70.00% | 70.00% | 70.00% | 70.00% | 0.7825 |
| **Fold 3** | 60.00% | 57.69% | 75.00% | 65.22% | 0.5900 |
| **Fold 4** | 62.50% | 62.50% | 71.43% | 66.67% | 0.5890 |
| **Fold 5** | 72.50% | 72.73% | 76.19% | 74.42% | 0.7544 |
| **Mean ± Std** | **65.00% ± 5.86%** | **64.40% ± 6.94%** | **71.52% ± 4.41%** | **67.64% ± 5.09%** | **0.6600 ± 0.0967** |

---

## 12. Advanced Statistical Modules

### 12.1 Principal Component Analysis (PCA)
* **PC 1 Explained Variance:** **35.90%** (Strong positive loading from Gender and Age).
* **PC 2 Explained Variance:** **33.75%** (Strong positive loading from Annual Income, negative from Age).
* **Cumulative 2D Variance Captured:** **69.65%**.

### 12.2 Bootstrap Resampling ($B=1000$)
* **Mean Spending Score:** Point Estimate = $50.20$, Bootstrap Mean = $50.18 \pm 1.80$, **95% Percentile CI: [46.67, 53.73]**.
* **Mean Annual Income (INR):** Point Estimate = $₹50,26,480$, Bootstrap Mean = $₹50,27,140 \pm ₹1,50,468$, **95% Percentile CI: [₹47,31,562, ₹53,21,398]**.
* **Out-of-Bag Accuracy:** Mean = $64.80\% \pm 3.70\%$, **95% CI: [57.50%, 72.00%]**.

---

## 13. Customer Behaviour Insights
1. **Youth Spending Propensity:** Customers under 35 years exhibit significantly higher spending scores ($p < 0.001$).
2. **Income Invariance of Spending Score:** High annual income does not automatically lead to a high spending score. Spending score is determined by lifestyle/preference rather than raw earning power alone.
3. **Gender Invariance:** Men and women shop with statistically indistinguishable spending score distributions ($p = 0.414$).

---

## 14. Conclusion
This project demonstrates the critical importance of marrying statistical hypothesis testing with machine learning classification. While Logistic Regression achieves 72.50% accuracy based primarily on age dynamics, classical inference (t-tests, ANOVA, and OLS) reveals that income and gender do not have linear predictive relationships with spending score.

---

## 15. Future Scope
* **Non-Linear Classification:** Explore Support Vector Machines (RBF kernel) or Decision Trees to capture the 5 distinct income-spending clusters.
* **Unsupervised Clustering:** Apply $k$-Means or Gaussian Mixture Models (GMM) before supervised classification.
* **Expanded Feature Space:** Collect loyalty tenure, online visit frequencies, and product category preferences.

---

## 16. Installation Instructions

```bash
# 1. Clone or navigate to the project directory
cd customer-behaviour-analysis

# 2. Verify Python 3.11 environment
python --version

# 3. Install required dependencies
pip install -r requirements.txt
```

---

## 17. How to Run the Project

### Option A: Launch Interactive Streamlit Dashboard
```bash
streamlit run app.py
```
*Access the web dashboard in your browser at `http://localhost:8501`.*

### Option B: Execute the Analysis Notebook
```bash
jupyter notebook notebooks/customer_analysis.ipynb
```

### Option C: Run Python Test Suite
```bash
python -c "from src.data_preprocessing import load_raw_data, preprocess_data; print(preprocess_data(load_raw_data()).shape)"
```
