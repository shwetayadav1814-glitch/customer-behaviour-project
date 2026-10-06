# ACADEMIC PROJECT REPORT

## CUSTOMER BEHAVIOUR ANALYSIS: PREDICTING HIGH-SPENDING AND LOW-SPENDING CUSTOMERS USING LOGISTIC REGRESSION

**Subject:** Statistics for Artificial Intelligence & Data Science  
**Academic Level:** Third-Year Bachelor of Engineering / Technology  
**Dataset:** Mall Customer Segmentation Dataset ($N = 200$)  
**Benchmark Currency Conversion:** $1\text{ USD} = 83.0\text{ INR}$ (Fixed Exchange Rate)

---

# 1. Title Page

```
========================================================================================
                          DEPARTMENT OF COMPUTER ENGINEERING / DATA SCIENCE
                              STATISTICS FOR ARTIFICIAL INTELLIGENCE & DS
========================================================================================

                                  PROJECT REPORT ON:

        CUSTOMER BEHAVIOUR ANALYSIS: PREDICTING HIGH-SPENDING AND LOW-SPENDING 
                         CUSTOMERS USING LOGISTIC REGRESSION

                                    Submitted by:
                               Third-Year Engineering Student

                              Under the Academic Guidance of:
                                    Faculty Supervisor

                               ACADEMIC YEAR: 2025–2026
========================================================================================
```

---

# 2. Certificate

This is to certify that the project entitled **“Customer Behaviour Analysis: Predicting High-Spending and Low-Spending Customers Using Logistic Regression”** submitted by the student is a bonafide record of work carried out under my supervision in partial fulfillment of the requirements for the subject **Statistics for Artificial Intelligence & Data Science**.

The statistical methods, hypothesis tests, regression formulations, bootstrap estimations, and classification pipelines presented herein reflect genuine analyses performed on the Mall Customers dataset.

\
**Internal Examiner** \hfill **External Examiner** \hfill **Head of Department**

---

# 3. Acknowledgement

I express my sincere gratitude to our course instructor and department faculty for providing guidance, technical framework, and intellectual encouragement throughout this statistical machine learning study. I also thank the peer reviewers and academic mentors whose constructive critiques helped refine the mathematical rigor, distribution modeling, and validation pipelines presented in this project.

---

# 4. Abstract

This academic project presents a mathematically grounded investigation into consumer spending behavior using the **Mall Customer Dataset** ($N=200$). Designed specifically for the curriculum of **Statistics for Artificial Intelligence & Data Science**, this study bridges classical inferential statistics with modern predictive modeling. Customers are classified into binary spending tiers: **High Spenders** ($\text{Spending Score} \ge 50$) and **Low Spenders** ($\text{Spending Score} < 50$), using demographic and financial inputs (`Age`, `Gender`, and `Annual Income` in Indian Rupees).

Parametric and non-parametric tests—including Shapiro-Wilk normality tests, two-sample independent Student’s/Welch’s $t$-tests, One-Way Analysis of Variance (ANOVA), Kruskal-Wallis tests, and Ordinary Least Squares (OLS) regression—are applied to examine consumer relationships. Feature scaling benchmarks confirm stability across standardization and normalization. **Binary Logistic Regression** fitted via Maximum Likelihood Estimation achieves a hold-out test accuracy of **72.50%** (ROC-AUC = **0.6800**) and a 5-Fold Stratified Cross-Validation accuracy of **65.00% ± 5.86%**. Parametric inference and odds ratios reveal that **Age** is the single statistically significant linear predictor ($\beta = -0.5911, \text{Odds Ratio} = 0.5537, p = 0.0005$), whereas Annual Income and Gender exhibit non-linear distributions. Bootstrap resampling ($B=1000$) provides 95% confidence intervals for population parameters, and Principal Component Analysis (PCA) maps the 3D feature space into 2 orthogonal dimensions capturing **69.65%** cumulative variance.

---

# 5. Introduction

In commercial marketing and modern retail management, understanding customer expenditure behavior is fundamental to resource optimization, loyalty segmentation, and promotional targeting. Traditional machine learning workflows often treat predictive models as black boxes, prioritizing empirical accuracy over statistical understanding.

In the academic discipline of **Statistics for Artificial Intelligence & Data Science**, predictive algorithms must be accompanied by formal statistical proofs, hypothesis tests, distribution checks, and uncertainty bounds. This project rigorously demonstrates how statistical foundations—such as Central Limit Theorem implications, $p$-value decision rules, Analysis of Variance, and Odds Ratios—underpin binary classification via **Logistic Regression**.

---

# 6. Problem Statement

Retail malls gather demographic and expenditure metrics but face the strategic challenge of identifying high-value patrons before their purchases occur. Mass-marketing broad promotions leads to inefficient capital expenditure.

The core challenge is to construct an interpretable, mathematically sound binary classification model that accurately predicts whether a customer will be a **High Spender** or a **Low Spender**, strictly utilizing pre-purchase indicators (`Age`, `Gender`, and `Annual Income`), while ensuring:
1. No data leakage occurs (the outcome variable `Spending Score` is strictly omitted from model input features).
2. Hypotheses regarding gender differences and income brackets are formally tested rather than visually assumed.
3. Model uncertainty is empirically bounded using bootstrap confidence intervals and cross-validation.

---

# 7. Objectives

1. **Data Cleaning & Currency Transformation:** Ingest the raw data, verify data hygiene, and convert income to Indian Rupees (INR) using a documented fixed rate ($1\text{ USD} = 83.0\text{ INR}$) and standard Indian numbering formatting.
2. **Descriptive & Distributional Profiling:** Compute measures of central tendency (Mean, Median, Mode), dispersion (Variance, Standard Deviation, IQR), and distribution shape (Skewness, Kurtosis, Shapiro-Wilk test).
3. **Parametric & Non-Parametric Hypothesis Testing:** Formulate and test hypotheses regarding gender spending differences using two-sample $t$-tests and Mann-Whitney $U$ tests at significance level $\alpha = 0.05$.
4. **Analysis of Variance (One-Way ANOVA):** Partition total variance into between-group and within-group sum of squares across Low, Medium, and High income tiers.
5. **Correlation & Linear Regression:** Evaluate linear vs non-linear associations using Pearson/Spearman correlation matrices and multiple OLS regression with residual diagnostics.
6. **Feature Scaling Comparison:** Benchmark classification metrics across Unscaled, Standardized (`StandardScaler`), and Normalized (`MinMaxScaler`) feature representations.
7. **Logistic Regression & Odds Ratio Estimation:** Fit a binary logit model, interpret log-odds coefficients ($\beta_j$), and compute Odds Ratios ($e^{\beta_j}$) with asymptotic $p$-values.
8. **Uncertainty Quantification (Bootstrap):** Resample $B=1000$ iterations with replacement to construct 95% Percentile and Normal confidence intervals.
9. **Dimensionality Reduction (PCA):** Perform eigenvalue decomposition on standardized features to derive 2D principal component projections.
10. **Validation & Real-Time Deployment:** Evaluate generalization across 5-Fold Stratified Cross-Validation, construct a Confusion Matrix and ROC Curve, and deploy a full-stack **Streamlit** dashboard for interactive inference.

---

# 8. Dataset Description

The analysis is conducted on the **Mall Customer Dataset** containing $N = 200$ customer observations:

| Feature Name | Type | Unit / Representation | Min | Median | Mean | Max |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `CustomerID` | Integer | Identifier ($1 - 200$) | 1 | 100.5 | 100.5 | 200 |
| `Gender` | Categorical | Female ($n=112$), Male ($n=88$) | - | - | - | - |
| `Age` | Integer | Chronological age in years | 18 | 36.0 | 38.85 | 70 |
| `Annual Income (k$)` | Float | Original annual income in \$k | 15.0 | 61.5 | 60.56 | 137.0 |
| `Annual_Income_INR` | Float | Converted income ($1\$ = ₹83.0$) | ₹12,45,000 | ₹51,04,500 | ₹50,26,480 | ₹1,13,71,000 |
| `Spending Score (1-100)` | Integer | Behavioural expenditure index | 1 | 50.0 | 50.20 | 99 |
| `Spending_Category` | Binary Target | $1 = \text{High Spender } (\ge 50), 0 = \text{Low Spender } (< 50)$ | 0 | 1 | 0.51 (51%) | 1 |

> **Target Class Balance:** High Spenders = $102$ ($51.0\%$), Low Spenders = $98$ ($49.0\%$).

---

# 9. Methodology

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   METHODOLOGY FLOWCHART                                │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
                           [Data Ingestion: Mall_Customers.csv]
                                           │
                                           ▼
                     [Hygiene Verification: 0 Nulls, 0 Duplicates]
                                           │
                                           ▼
                  [Feature Engineering: Annual_Income_INR @ ₹83.0 / $1]
                                           │
                                           ▼
              [Target Definition: Spending_Category (>=50 High / <50 Low)]
                                           │
                ┌──────────────────────────┴──────────────────────────┐
                ▼                                                     ▼
    [Statistical Inference Suite]                           [Supervised Machine Learning]
    ├── Descriptive Stats (Mean, Median, IQR)               ├── Stratified Train/Test Split (80/20)
    ├── Shapiro-Wilk Normality Tests                        ├── Feature Scaling (StandardScaler)
    ├── Two-Sample t-Test & Mann-Whitney U                  ├── Logistic Regression Training
    ├── One-Way ANOVA & Kruskal-Wallis                      ├── Odds Ratio (e^β) & p-value Analysis
    ├── Pearson/Spearman Correlation                        ├── Confusion Matrix (TP, TN, FP, FN)
    ├── OLS Multiple Linear Regression                      ├── ROC-AUC & PR-AUC Curves
    ├── Bootstrap Resampling (B=1000, 95% CI)               └── 5-Fold Stratified Cross-Validation
    └── Principal Component Analysis (2D PCA)                         │
                │                                                     │
                └──────────────────────────┬──────────────────────────┘
                                           ▼
                        [Interactive Streamlit Dashboard]
```

---

# 10. Data Preprocessing

### 10.1 Currency Conversion Formulation
The original income feature `Annual Income (k$)` is converted to Indian Rupees (INR) using a fixed benchmark rate:
$$\text{USD\_TO\_INR} = 83.0$$
$$\text{Annual\_Income\_INR} = \text{Annual Income (k\$)} \times 1000 \times 83.0$$

*Example Calculations:*
* $\$15\text{k} \rightarrow 15 \times 1000 \times 83.0 = ₹12,45,000$
* $\$60\text{k} \rightarrow 60 \times 1000 \times 83.0 = ₹49,80,000$
* $\$137\text{k} \rightarrow 137 \times 1000 \times 83.0 = ₹1,13,71,000$

### 10.2 Categorical Feature Encoding
$$\text{Gender\_Encoded} = \begin{cases} 1 & \text{if Gender} = \text{'Male'} \\ 0 & \text{if Gender} = \text{'Female'} \end{cases}$$

### 10.3 Target Variable Formulation
$$\text{Spending\_Category} = \begin{cases} 1 \ (\text{High Spender}) & \text{if Spending Score} \ge 50 \\ 0 \ (\text{Low Spender}) & \text{if Spending Score} < 50 \end{cases}$$

---

# 11. Exploratory Data Analysis (EDA)

* **Age Distribution:** Spans $18$ to $70$ years, with peak customer concentration in the $20 - 40$ age bracket.
* **Income Distribution:** Shows a multimodal distribution with prominent peaks around ₹40 Lakhs and ₹65 Lakhs ($45 - 80\text{k \$}$).
* **Spending Score Distribution:** Spans $1$ to $99$, symmetrically centered around a mean of $50.20$.
* **Bivariate Insights:**
  1. Age exhibits a clear negative relationship with spending score; consumers older than 40 rarely achieve spending scores above 60.
  2. Income vs Spending Score forms a distinct 5-cluster pattern, illustrating that earning capacity alone does not monotonically dictate shopping expenditure.

---

# 12. Descriptive Statistics

Summary statistics computed on the complete dataset ($N = 200$):

| Variable | Mean | Median | Mode | Min | Max | Range | Variance ($s^2$) | Std Dev ($s$) | IQR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Age** | 38.85 | 36.00 | 32.0 | 18 | 70 | 52 | 195.13 | 13.97 | 20.25 |
| **Annual Income (k$)** | 60.56 | 61.50 | 54.0 | 15.0 | 137.0 | 122.0 | 689.84 | 26.26 | 36.50 |
| **Annual Income (INR)** | ₹50,26,480 | ₹51,04,500 | ₹44,82,000 | ₹12,45,000 | ₹1,13,71,000 | ₹1,01,26,000 | $4.75 \times 10^{12}$ | ₹21,79,918 | ₹30,29,500 |
| **Spending Score (1-100)** | 50.20 | 50.00 | 42.0 | 1 | 99 | 98 | 666.85 | 25.82 | 41.50 |

---

# 13. Probability and Distribution Analysis

To determine whether parametric normality assumptions hold, the **Shapiro-Wilk Test** was applied:

$$W = \frac{\left(\sum_{i=1}^n a_i x_{(i)}\right)^2}{\sum_{i=1}^n (x_i - \bar{x})^2}$$

| Variable | Skewness | Excess Kurtosis | Shapiro-Wilk Statistic ($W$) | $p$-value | Empirical Distribution Shape |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Age** | +0.486 | -0.672 | 0.943 | $2.71 \times 10^{-6}$ | Moderately Right-Skewed; Non-Normal ($p < 0.05$) |
| **Annual Income** | +0.322 | -0.126 | 0.978 | $3.54 \times 10^{-3}$ | Mildly Right-Skewed; Non-Normal ($p < 0.05$) |
| **Spending Score** | -0.047 | -0.823 | 0.971 | $2.42 \times 10^{-4}$ | Approximately Symmetric; Non-Normal ($p < 0.05$) |

*Conclusion:* Since all features deviate significantly from a theoretical Gaussian distribution ($p < 0.05$), non-parametric verification (Mann-Whitney $U$, Kruskal-Wallis, Bootstrap) is strictly required alongside parametric estimators.

---

# 14. Hypothesis Testing

### Test: Gender Differences in Spending Score
* **Null Hypothesis ($H_0$):** $\mu_{\text{Male}} = \mu_{\text{Female}}$ (No difference in population mean Spending Score).
* **Alternative Hypothesis ($H_1$):** $\mu_{\text{Male}} \neq \mu_{\text{Female}}$ (Statistically significant difference).
* **Significance Level:** $\alpha = 0.05$.

#### 14.1 Homogeneity of Variance (Levene’s Test)
$$F = 0.4372, \quad p = 0.5092$$
Since $p > 0.05$, equal variance is assumed.

#### 14.2 Test Results
* **Male Sample:** $n_1 = 88, \ \bar{x}_1 = 48.51, \ s_1 = 27.90$
* **Female Sample:** $n_2 = 112, \ \bar{x}_2 = 51.53, \ s_2 = 24.11$
* **Two-Sample Student’s $t$-Test:**
  $$t = \frac{\bar{x}_1 - \bar{x}_2}{s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}} = -0.8190, \quad df = 198, \quad p = 0.4137$$
* **Non-Parametric Mann-Whitney $U$ Test:**
  $$U = 4580.0, \quad p = 0.3804$$

#### 14.3 Decision & Academic Interpretation
**Decision:** **Fail to Reject $H_0$** ($p = 0.4137 > 0.05$).  
There is no statistically significant difference in Spending Score between male and female shoppers. Gender cannot serve as a primary independent driver of high vs low spending.

---

# 15. Analysis of Variance (ANOVA)

### Test: Spending Score Across Income Tiers
Income is partitioned into three meaningful socio-economic tiers:
1. **Low Income:** $< \$40\text{k}$ ($< ₹33,20,000$) $\rightarrow n = 36$
2. **Medium Income:** $\$40\text{k} - \$80\text{k}$ ($₹33,20,000 - ₹66,40,000$) $\rightarrow n = 100$
3. **High Income:** $> \$80\text{k}$ ($> ₹66,40,000$) $\rightarrow n = 64$

* **$H_0$:** $\mu_{\text{Low}} = \mu_{\text{Medium}} = \mu_{\text{High}}$
* **$H_1$:** At least one income group has a different mean Spending Score ($\alpha = 0.05$).

### 15.1 ANOVA Summary Table

| Source of Variation | Sum of Squares ($SS$) | Degrees of Freedom ($df$) | Mean Square ($MS$) | $F$-Statistic | $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Between Groups** | 14.88 | 2 | 7.44 | **0.0112** | **0.9889** |
| **Within Groups (Error)** | 132,549.91 | 197 | 672.84 | - | - |
| **Total** | 132,564.80 | 199 | - | - | - |

* **Group Descriptive Means:**
  * Low Income Mean: $50.22 \pm 28.91$
  * Medium Income Mean: $50.09 \pm 22.38$
  * High Income Mean: $50.41 \pm 29.17$
* **Non-Parametric Kruskal-Wallis Test:** $H = 0.0878, p = 0.9570$.

### 15.2 Decision & Interpretation
**Decision:** **Fail to Reject $H_0$** ($F = 0.0112, p = 0.9889 \gg 0.05$).  
The mean Spending Score is virtually identical across all income brackets (~$50$). This occurs because each income tier contains both highly frugal and highly impulsive shoppers.

---

# 16. Correlation Analysis

### 16.1 Correlation Matrix & $p$-values

| Pairwise Association | Pearson $r$ | Pearson $p$-value | Spearman $\rho$ | Statistical Inference |
| :--- | :---: | :---: | :---: | :--- |
| **Age $\leftrightarrow$ Spending Score** | **-0.327** | $2.22 \times 10^{-6}$ | **-0.345** | Moderate negative linear and monotonic association. |
| **Income $\leftrightarrow$ Spending Score** | **+0.010** | 0.8900 | **+0.008** | Near-zero linear correlation (non-linear clusters). |
| **Age $\leftrightarrow$ Annual Income** | **-0.012** | 0.8623 | **-0.028** | Negligible association in this sample. |

> **Crucial Rule:** *Correlation measures linear association and does not imply causation.*

---

# 17. Linear Regression

An Ordinary Least Squares (OLS) model was fitted to predict continuous Spending Score from Age and Annual Income:

$$\text{Spending Score} = \beta_0 + \beta_1 \cdot \text{Age} + \beta_2 \cdot \text{Annual Income} + \epsilon$$

### 17.1 Parameter Estimates

| Parameter | Coefficient ($\beta$) | Std Error | $t$-Statistic | $p$-value | 95% Confidence Interval |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Intercept ($\beta_0$)** | 73.3479 | 6.549 | 11.200 | $< 0.0001$ | $[60.425, 86.271]$ |
| **Age ($\beta_1$)** | **-0.6048** | 0.124 | -4.862 | **$2.34 \times 10^{-6}$** | $[-0.850, -0.359]$ |
| **Annual Income ($\beta_2$)** | +0.0057 | 0.066 | 0.087 | 0.9304 | $[-0.125, +0.136]$ |

### 17.2 Goodness of Fit & Diagnostics
* $R^2 = 0.1071$, $\text{Adjusted } R^2 = 0.0981$
* $F(2, 197) = 11.82, \ p = 1.45 \times 10^{-5}$
* $\text{RMSE} = 24.22, \ \text{MAE} = 20.73$
* **Durbin-Watson:** $3.75$ (Residuals exhibit mild negative serial structure due to sorted customer IDs).

---

# 18. Feature Scaling

Prior to training classification models, features were scaled to evaluate numerical stability:

| Scaling Technique | Hold-out Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **No Scaling** | 72.50% | 73.68% | 70.00% | 71.79% | 0.6800 |
| **Standardization (`StandardScaler`)** | **72.50%** | **73.68%** | **70.00%** | **71.79%** | **0.6800** |
| **Normalization (`MinMaxScaler`)** | 72.50% | 73.68% | 70.00% | 71.79% | 0.6800 |

*Analysis:* Standardization ($z = \frac{x - \mu}{\sigma}$) was adopted for production pipelines to ensure numerical stability and enable direct comparison of standardized regression coefficients.

---

# 19. Logistic Regression

### 19.1 Mathematical Model
The binary classification probability is modeled via the logistic sigmoid function:

$$P(Y = 1 \mid X) = \frac{1}{1 + e^{-z}}, \quad z = \beta_0 + \beta_1 Z_{\text{Age}} + \beta_2 Z_{\text{Gender}} + \beta_3 Z_{\text{Income}}$$

$$\text{Odds} = \frac{P(Y=1)}{1 - P(Y=1)} = e^z, \quad \text{Odds Ratio} = e^{\beta_j}$$

### 19.2 Estimated Coefficients and Odds Ratios

| Feature | Standardized Coef ($\beta$) | Odds Ratio ($e^\beta$) | Std Error | $p$-value | Direction & Magnitude |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Intercept ($\beta_0$)** | +0.0460 | 1.0471 | 0.163 | 0.7781 | Base log-odds for average profile. |
| **Age** | **-0.5911** | **0.5537** | 0.171 | **0.0005** | Each $+1\sigma$ age increase reduces odds of high spending by **44.63%**. |
| **Gender (Male=1)** | -0.0096 | 0.9904 | 0.151 | 0.9494 | Not statistically significant ($p \approx 0.95$). |
| **Annual Income** | -0.0120 | 0.9881 | 0.148 | 0.9357 | Not statistically significant in linear form ($p \approx 0.94$). |

---

# 20. Bootstrap Resampling

To assess sampling uncertainty without assuming normality, non-parametric **Bootstrap Resampling** ($B = 1000$ iterations with replacement) was implemented.

### 20.1 Bootstrap Parameter Estimates

| Parameter / Metric | Point Estimate ($\hat{\theta}$) | Bootstrap Mean ($\bar{\theta}^*$) | Bootstrap SE ($SE_{\text{boot}}$) | 95% Percentile Confidence Interval |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Spending Score** | 50.20 | 50.18 | 1.80 | **[46.67, 53.73]** |
| **Mean Annual Income (INR)** | ₹50,26,480 | ₹50,27,140 | ₹1,50,468 | **[₹47,31,562, ₹53,21,398]** |
| **Out-of-Bag Model Accuracy** | 64.80% | 64.80% | 3.70% | **[57.50%, 72.00%]** |

*Interpretation:* We are 95% confident that the true population mean spending score lies between $46.67$ and $53.73$.

---

# 21. Principal Component Analysis (PCA)

PCA was applied to the standardized 3D feature space ($[\text{Age}, \text{Gender}, \text{Income}]$):

### 21.1 Eigenvalue & Variance Decomposition

| Principal Component | Explained Variance Ratio | Explained Variance (%) | Cumulative Variance (%) |
| :--- | :---: | :---: | :---: |
| **Principal Component 1 (PC 1)** | 0.3590 | 35.90% | 35.90% |
| **Principal Component 2 (PC 2)** | 0.3375 | 33.75% | **69.65%** |

### 21.2 Component Loadings (Eigenvector Weights)

| Feature | PC 1 Loading | PC 2 Loading |
| :--- | :---: | :---: |
| **Age** | +0.5057 | -0.6784 |
| **Gender Encoded** | **+0.7328** | +0.0119 |
| **Annual Income (k$)** | +0.4552 | **+0.7346** |

---

# 22. Cross-Validation

To verify that the model does not suffer from data partitioning bias, **5-Fold Stratified Cross-Validation** was executed:

| Validation Fold | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 60.00% | 59.09% | 65.00% | 61.90% | 0.5825 |
| **Fold 2** | 70.00% | 70.00% | 70.00% | 70.00% | 0.7825 |
| **Fold 3** | 60.00% | 57.69% | 75.00% | 65.22% | 0.5900 |
| **Fold 4** | 62.50% | 62.50% | 71.43% | 66.67% | 0.5890 |
| **Fold 5** | 72.50% | 72.73% | 76.19% | 74.42% | 0.7544 |
| **Mean ± Std** | **65.00% ± 5.86%** | **64.40% ± 6.94%** | **71.52% ± 4.41%** | **67.64% ± 5.09%** | **0.6600 ± 0.0967** |

---

# 23. Model Evaluation

Hold-out evaluation was conducted on an unseen 20% test partition ($n_{\text{test}} = 40$):

### 23.1 Hold-out Confusion Matrix

```
                        Predicted Low (0)    Predicted High (1)
  Actual Low (0)              15 (TN)               5 (FP)
  Actual High (1)              6 (FN)              14 (TP)
```

* **True Positives ($TP = 14$):** High Spenders correctly identified.
* **True Negatives ($TN = 15$):** Low Spenders correctly identified.
* **False Positives ($FP = 5$) [Type I Error]:** Low Spenders misclassified as High Spenders.
* **False Negatives ($FN = 6$) [Type II Error]:** High Spenders misclassified as Low Spenders.

### 23.2 Comprehensive Metrics

| Evaluation Metric | Mathematical Formula | Value |
| :--- | :--- | :---: |
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | **72.50%** |
| **Precision** | $\frac{TP}{TP + FP}$ | **73.68%** |
| **Recall (Sensitivity)** | $\frac{TP}{TP + FN}$ | **70.00%** |
| **Specificity** | $\frac{TN}{TN + FP}$ | **75.00%** |
| **F1-Score** | $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ | **71.79%** |
| **Balanced Accuracy** | $\frac{\text{Sensitivity} + \text{Specificity}}{2}$ | **72.50%** |
| **Matthews Correlation Coef (MCC)** | $\frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$ | **+0.4503** |
| **Area Under ROC Curve (ROC-AUC)** | $\int_0^1 \text{TPR}(FPR^{-1}(t)) dt$ | **0.6800** |

---

# 24. Results and Discussion

### Generalization Comparison Table

| Evaluation Partition | Accuracy | Notes / Assessment |
| :--- | :---: | :--- |
| **Training Partition ($n=160$)** | 70.62% | In-sample fit of logistic regression |
| **Hold-out Test Partition ($n=40$)** | 72.50% | Generalization on unseen test set |
| **5-Fold Stratified Cross-Validation** | 65.00% ± 5.86% | Unbiased estimate across multiple resamples |
| **Bootstrap Out-of-Bag (OOB)** | 64.80% ± 3.70% | Non-parametric empirical estimation |

The close alignment between training, testing, cross-validation, and bootstrap metrics indicates that the model is neither severely overfitting nor underfitting within the capacity of a linear classifier.

---

# 25. Customer Behaviour Insights

1. **Age as the Primary Expenditure Driver:** Younger consumers ($< 35$ years) demonstrate a significantly higher likelihood of being High Spenders ($p = 0.0005, \text{Odds Ratio} = 0.5537$).
2. **Income Invariance in Aggregate:** Across Low, Medium, and High income tiers, mean spending score remains constant at ~$50$ ($p = 0.9889$). High income does not guarantee high spending in this mall environment.
3. **Gender Invariance:** Spending score distributions between Male and Female shoppers are statistically indistinguishable ($t = -0.8190, p = 0.4137$).
4. **Non-Linear Income Clusters:** Visual and regression diagnostics reveal that income groups split into high-spending and low-spending subgroups, requiring non-linear models or clustering for further segmentation.

---

# 26. Conclusion

This project successfully proves that combining classical inferential statistics with modern machine learning yields deep, verifiable insights into customer spending behavior. By validating assumptions, testing formal hypotheses via $t$-tests and ANOVA, applying OLS regression diagnostics, quantifying uncertainty through bootstrap resampling, and training an interpretable Logistic Regression classifier, the project satisfies all pedagogical and practical requirements of **Statistics for Artificial Intelligence & Data Science**.

---

# 27. Future Scope

1. **Non-Linear Classifiers:** Implement Kernel SVM (RBF) and Decision Trees / Random Forests to capture the 5 income-spending clusters.
2. **Hybrid Unsupervised-Supervised Pipelines:** Cluster customers using $k$-Means or Gaussian Mixture Models (GMM) before predicting cluster membership.
3. **Feature Enrichment:** Ingest additional behavioral features such as mall visit frequency, dwell time, loyalty program tier, and preferred shopping categories.

---

# 28. References

1. Agresti, A. (2018). *An Introduction to Categorical Data Analysis* (3rd ed.). John Wiley & Sons.
2. Casella, G., & Berger, R. L. (2002). *Statistical Inference* (2nd ed.). Duxbury.
3. Efron, B., & Tibshirani, R. J. (1994). *An Introduction to the Bootstrap*. CRC Press.
4. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.
5. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning with Applications in Python*. Springer.
6. Montgomery, D. C. (2017). *Design and Analysis of Experiments* (9th ed.). John Wiley & Sons.
7. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
8. Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with Python. *Proceedings of the 9th Python in Science Conference*.
