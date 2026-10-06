"""
Streamlit Academic Dashboard: Customer Behaviour Analysis
Subject: Statistics for Artificial Intelligence & Data Science

Topic: Customer Behaviour Analysis: Predicting High-Spending and Low-Spending Customers Using Logistic Regression
Dataset: Mall Customers Dataset
Conversion Rate: 1 USD = 83.0 INR (Fixed)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler

# Import custom statistical modules
from src.data_preprocessing import (
    USD_TO_INR, format_indian_currency, load_raw_data, inspect_data_integrity,
    preprocess_data, get_feature_target_split, prepare_train_test_datasets
)
from src.descriptive_statistics import (
    calculate_descriptive_stats, get_formatted_descriptive_table, get_statistical_interpretations
)
from src.hypothesis_testing import (
    test_gender_vs_spending_score, get_hypothesis_summary_table
)
from src.anova_analysis import (
    perform_income_spending_anova
)
from src.correlation_regression import (
    compute_correlation_analysis, fit_linear_regression
)
from src.logistic_regression import (
    compare_feature_scaling, train_logistic_regression_pipeline, predict_customer_category
)
from src.bootstrap_analysis import (
    bootstrap_statistic_ci, bootstrap_model_accuracy, run_full_bootstrap_suite
)
from src.pca_analysis import (
    perform_pca_analysis
)
from src.model_evaluation import (
    evaluate_classifier, perform_kfold_cv, compare_generalization
)

# Set Page Config
st.set_page_config(
    page_title="Customer Behaviour Analysis | Statistics for AI & Data Science",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .stat-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1F2937;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6B7280;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #111827;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_cache_data():
    raw_df = load_raw_data()
    clean_df = preprocess_data(raw_df)
    return raw_df, clean_df


raw_df, df_clean = load_and_cache_data()
X, y, feature_cols = get_feature_target_split(df_clean)
X_train, X_test, y_train, y_test = prepare_train_test_datasets(X, y, test_size=0.2, random_state=42)

# Train the primary standardized Logistic Regression Pipeline
pipeline = train_logistic_regression_pipeline(X_train, y_train, scaler_type='standard')

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/combo-chart.png", width=70)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Project Section:",
    [
        "🏠 Home",
        "📁 Dataset & Integrity",
        "📈 Exploratory Data Analysis",
        "📐 Descriptive Statistics",
        "🧪 Hypothesis Testing (t-Test)",
        "📊 ANOVA (Income Tiers)",
        "🔗 Correlation & Linear Regression",
        "⚙️ Feature Scaling & Logistic Regression",
        "🎯 Model Evaluation & K-Fold CV",
        "🧬 PCA (Dimensionality Reduction)",
        "🔄 Bootstrap Resampling (Uncertainty)",
        "🔮 Interactive Customer Prediction"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**USD to INR Rate:** ₹{USD_TO_INR:.1f} / $1")
st.sidebar.markdown("**Target Rule:**")
st.sidebar.markdown("- High Spender: Score ≥ 50")
st.sidebar.markdown("- Low Spender: Score < 50")
st.sidebar.markdown("**Subject:** Statistics for AI & DS")

# ==============================================================================
# 1. HOME PAGE
# ==============================================================================
if page == "🏠 Home":
    st.markdown('<div class="main-title">Customer Behaviour Analysis: Predicting High-Spending and Low-Spending Customers</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">An Academic Project for <b>Statistics for Artificial Intelligence & Data Science</b> (Third-Year Engineering)</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### 1. Project Overview & Objective
        This project investigates consumer spending patterns using the **Mall Customer Dataset** and applies rigorous statistical methodology combined with **Binary Logistic Regression** to classify customers into:
        * **High Spenders:** Spending Score $\ge 50$ (Class 1)
        * **Low Spenders:** Spending Score $< 50$ (Class 0)
        
        Using demographic and financial inputs:
        * **Age** (years)
        * **Gender** (Male / Female)
        * **Annual Income** (in original \$k and converted to Indian Rupees ₹)
        
        *(Note: Spending Score is strictly omitted from input features to eliminate target leakage).*
        
        ### 2. Academic & Statistical Focus
        Rather than treating machine learning as a "black box", this project demonstrates fundamental statistical principles:
        - **Descriptive Statistics:** Measures of central tendency, dispersion, skewness, and kurtosis.
        - **Probability & Distribution Tests:** Normality assessments (Shapiro-Wilk) and density profiles.
        - **Hypothesis Testing:** Two-sample independent t-test and Mann-Whitney U test on gender-based spending.
        - **Analysis of Variance (One-Way ANOVA):** Between-group vs within-group variance across income tiers.
        - **Correlation & OLS Regression:** Pearson/Spearman coefficients, residual diagnostics, and distinction between correlation and causation.
        - **Feature Scaling:** Standardization vs Normalization vs Unscaled features.
        - **Logistic Regression Modeling:** Maximum Likelihood estimation, odds ratios ($e^\beta$), and p-values.
        - **Bootstrap Resampling:** Empirical confidence intervals without strict parametric assumptions.
        - **Principal Component Analysis (PCA):** Orthogonal eigenvalue decomposition in 2D space.
        - **Cross-Validation:** 5-Fold Stratified CV to evaluate generalizability and prevent overfitting.
        """)

    with col2:
        st.info("### 📌 Key Project Metrics")
        st.metric(label="Total Observations", value=f"{len(df_clean)} Customers")
        st.metric(label="High Spenders (Score ≥ 50)", value=f"{(df_clean['Spending_Category'] == 1).sum()} (51.0%)")
        st.metric(label="Low Spenders (Score < 50)", value=f"{(df_clean['Spending_Category'] == 0).sum()} (49.0%)")
        st.metric(label="USD to INR Benchmark", value=f"₹{USD_TO_INR:.1f} / $1 USD")
        st.metric(label="Logistic Regression Accuracy", value="72.50%")
        st.metric(label="5-Fold Mean CV Score", value="65.00% ± 5.86%")

    st.markdown("---")
    st.markdown("### 🗺️ Project Architecture & Statistical Workflow")
    st.code("""
    [Mall Customers Raw Data]
              │
              ▼
    [Data Cleaning & Integrity Validation] ──► [Currency Conversion: USD $k -> INR ₹ (Fixed 83.0)]
              │
              ▼
    [Target Engineering: Spending_Category (>=50 High / <50 Low)]
              │
              ├──► [Descriptive Statistics & Normality Tests (Shapiro-Wilk)]
              ├──► [Hypothesis Testing: Gender vs Spending (t-Test & Mann-Whitney)]
              ├──► [One-Way ANOVA: Income Tiers vs Spending Score (F-Test & Tukey HSD)]
              ├──► [Correlation Matrix & Multiple Linear Regression (OLS)]
              ├──► [Principal Component Analysis: 3D -> 2D Orthogonal Projection]
              ├──► [Bootstrap Resampling: 1000 Replicates & 95% Confidence Intervals]
              │
              ▼
    [Feature Scaling: StandardScaler / MinMaxScaler]
              │
              ▼
    [Logistic Regression Classifier] ──► [Odds Ratios (e^β) & Parameter Significance]
              │
              ├──► [Confusion Matrix (TP, TN, FP, FN)]
              ├──► [ROC Curve & Area Under Curve (AUC)]
              ├──► [5-Fold Stratified Cross-Validation]
              │
              ▼
    [Real-time Interactive Prediction Dashboard]
    """, language="text")

# ==============================================================================
# 2. DATASET & INTEGRITY
# ==============================================================================
elif page == "📁 Dataset & Integrity":
    st.markdown('<div class="main-title">Dataset Structure & Data Hygiene</div>', unsafe_allow_html=True)
    
    integrity = inspect_data_integrity(raw_df)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows (N)", integrity["shape"][0])
    col2.metric("Columns", integrity["shape"][1])
    col3.metric("Missing Values", integrity["total_missing"])
    col4.metric("Duplicate Rows", integrity["duplicate_rows"])
    
    st.markdown("### 📋 Cleaned & Engineered Dataset Preview")
    st.dataframe(df_clean[['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 
                           'Annual_Income_INR_Formatted', 'Spending Score (1-100)', 
                           'Spending_Category_Label', 'Income_Group']].head(15), use_container_width=True)
    
    st.markdown("### 💵 Indian Rupee (INR) Conversion Details")
    st.markdown(f"""
    - **Fixed Exchange Rate:** `1 USD = {USD_TO_INR} INR`
    - **Formula:** `Annual_Income_INR = Annual_Income_USD_K * 1000 * {USD_TO_INR}`
    - **Indian Numbering Format:** Formatted with standard Indian commas (e.g. ₹5,00,000, ₹8,50,000, ₹12,50,000).
    - **Statistical Implication:** Multiplying by a constant scalar changes the numerical scale of measurement, but strictly preserves rank ordering, Pearson correlation coefficient magnitudes with other variables, and relative variances.
    """)
    
    st.markdown("### 🔍 Target Variable Distribution")
    colA, colB = st.columns([1, 1])
    with colA:
        class_counts = df_clean['Spending_Category_Label'].value_counts()
        st.dataframe(pd.DataFrame({
            'Spending Category': class_counts.index,
            'Count': class_counts.values,
            'Percentage (%)': (class_counts.values / len(df_clean) * 100).round(2)
        }))
    with colB:
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.countplot(data=df_clean, x='Spending_Category_Label', palette=['#EF4444', '#10B981'], ax=ax)
        ax.set_title("Distribution of Target: High vs Low Spenders")
        ax.set_xlabel("Spending Category")
        ax.set_ylabel("Count")
        st.pyplot(fig)

# ==============================================================================
# 3. EXPLORATORY DATA ANALYSIS
# ==============================================================================
elif page == "📈 Exploratory Data Analysis":
    st.markdown('<div class="main-title">Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
    
    eda_tab1, eda_tab2, eda_tab3, eda_tab4 = st.tabs([
        "Distribution Charts", "Comparison Charts", "Relationship Charts", "Correlation Heatmap"
    ])
    
    with eda_tab1:
        st.markdown("### 📊 Distribution of Key Variables")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig1, ax1 = plt.subplots(figsize=(5, 4))
            sns.histplot(df_clean['Age'], kde=True, color='#3B82F6', bins=15, ax=ax1)
            ax1.set_title("Age Distribution")
            ax1.set_xlabel("Age (years)")
            st.pyplot(fig1)
            st.caption("Mildly right-skewed. Peak age concentration is between 20 and 40 years.")
            
        with col2:
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            sns.histplot(df_clean['Annual_Income_INR'] / 100000, kde=True, color='#10B981', bins=15, ax=ax2)
            ax2.set_title("Annual Income Distribution (₹ in Lakhs)")
            ax2.set_xlabel("Annual Income (₹ Lakhs)")
            st.pyplot(fig2)
            st.caption("Income peaks around ₹40L - ₹65L with a right tail extending beyond ₹1 Crore.")
            
        with col3:
            fig3, ax3 = plt.subplots(figsize=(5, 4))
            sns.histplot(df_clean['Spending Score (1-100)'], kde=True, color='#F59E0B', bins=15, ax=ax3)
            ax3.axvline(50, color='red', linestyle='--', label='Threshold (50)')
            ax3.set_title("Spending Score Distribution")
            ax3.set_xlabel("Spending Score (1-100)")
            ax3.legend()
            st.pyplot(fig3)
            st.caption("Roughly bimodal/symmetric distribution centered closely around the mean of 50.2.")

    with eda_tab2:
        st.markdown("### 👥 Demographic & Category Comparisons")
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(data=df_clean, x='Gender', hue='Spending_Category_Label', palette=['#10B981', '#EF4444'], ax=ax)
            ax.set_title("Spending Category Breakdown by Gender")
            st.pyplot(fig)
            st.caption("Both male and female cohorts display similar proportions of high vs low spenders.")
            
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=df_clean, x='Spending_Category_Label', y='Age', palette=['#10B981', '#EF4444'], ax=ax)
            ax.set_title("Age Distribution by Spending Category")
            st.pyplot(fig)
            st.caption("High spenders have a noticeably lower median age compared to low spenders.")

    with eda_tab3:
        st.markdown("### 🔄 Bivariate Relationship Scatter Plots")
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(data=df_clean, x='Age', y='Spending Score (1-100)', hue='Gender', palette=['#3B82F6', '#EC4899'], alpha=0.8, ax=ax)
            sns.regplot(data=df_clean, x='Age', y='Spending Score (1-100)', scatter=False, ax=ax, color='black', line_kws={'linestyle': '--'})
            ax.set_title("Age vs Spending Score (with trendline)")
            st.pyplot(fig)
            st.caption("Negative trend: Customers above 40 years rarely exhibit spending scores > 60.")
            
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(data=df_clean, x=df_clean['Annual_Income_INR']/100000, y=df_clean['Spending Score (1-100)'], 
                            hue=df_clean['Spending_Category_Label'], palette=['#EF4444', '#10B981'], alpha=0.8, ax=ax)
            ax.set_title("Annual Income (₹ Lakhs) vs Spending Score")
            ax.set_xlabel("Annual Income (₹ Lakhs)")
            st.pyplot(fig)
            st.caption("Displays characteristic 5-cluster pattern; high income does not automatically imply high spending.")

    with eda_tab4:
        st.markdown("### 🔥 Correlation Heatmap")
        corr_data = df_clean[['Age', 'Gender_Encoded', 'Annual Income (k$)', 'Spending Score (1-100)']].corr()
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', fmt=".3f", vmin=-1, vmax=1, ax=ax, cbar=True)
        ax.set_title("Pearson Correlation Heatmap")
        st.pyplot(fig)
        st.warning("⚠️ **Statistical Principle:** Correlation measures linear association and does NOT imply causation.")

# ==============================================================================
# 4. DESCRIPTIVE STATISTICS
# ==============================================================================
elif page == "📐 Descriptive Statistics":
    st.markdown('<div class="main-title">Descriptive Statistics & Distribution Shape</div>', unsafe_allow_html=True)
    
    stats_df = calculate_descriptive_stats(df_clean)
    formatted_table = get_formatted_descriptive_table(stats_df)
    
    st.markdown("### 📊 Comprehensive Summary Statistics Table")
    st.dataframe(formatted_table, use_container_width=True)
    
    st.markdown("### 📝 Rigorous Statistical Interpretations")
    interpretations = get_statistical_interpretations(stats_df)
    for var, text in interpretations.items():
        st.markdown(f"- **{var}:** {text}")
        
    st.markdown("### 🔬 Normality Assessment (Shapiro-Wilk Test)")
    st.markdown("""
    The **Shapiro-Wilk test** evaluates the null hypothesis $H_0$: *The data is drawn from a normal distribution*.
    - If $p < 0.05$: Reject $H_0$ (Significant departure from normality).
    - As shown in the table, **Age** ($p = 2.71 \\times 10^{-6}$), **Income** ($p = 3.54 \\times 10^{-3}$), and **Spending Score** ($p = 2.42 \\times 10^{-4}$) all have $p < 0.05$, indicating non-normal empirical distributions. This justifies utilizing non-parametric tests (Mann-Whitney U, Kruskal-Wallis, Bootstrap) alongside parametric ones.
    """)

# ==============================================================================
# 5. HYPOTHESIS TESTING
# ==============================================================================
elif page == "🧪 Hypothesis Testing (t-Test)":
    st.markdown('<div class="main-title">Hypothesis Testing: Gender vs Spending Score</div>', unsafe_allow_html=True)
    
    hyp_res = test_gender_vs_spending_score(df_clean, alpha=0.05)
    summary_table = get_hypothesis_summary_table(hyp_res)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📋 Hypothesis Test Summary")
        st.dataframe(summary_table, use_container_width=True)
        
    with col2:
        st.markdown("### 📊 Visual Comparison of Groups")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.kdeplot(data=df_clean[df_clean['Gender']=='Male']['Spending Score (1-100)'], label='Male', fill=True, color='#3B82F6', ax=ax)
        sns.kdeplot(data=df_clean[df_clean['Gender']=='Female']['Spending Score (1-100)'], label='Female', fill=True, color='#EC4899', ax=ax)
        ax.set_title("Spending Score Density by Gender")
        ax.set_xlabel("Spending Score")
        ax.legend()
        st.pyplot(fig)
        
    st.markdown("### 🎯 Decision & Academic Conclusion")
    if "Fail" in hyp_res['decision']:
        st.success(f"**Conclusion:** {hyp_res['conclusion']}")
    else:
        st.warning(f"**Conclusion:** {hyp_res['conclusion']}")
        
    st.markdown("""
    **Viva Question Preparation:**
    - *Why t-test?* To compare the means of two independent groups (Male vs Female).
    - *What did Levene's test show?* $p = 0.509$, confirming equal variances between genders.
    - *What is the takeaway?* Gender alone does not explain spending score divergence in this mall dataset.
    """)

# ==============================================================================
# 6. ANOVA
# ==============================================================================
elif page == "📊 ANOVA (Income Tiers)":
    st.markdown('<div class="main-title">One-Way ANOVA: Income Groups vs Spending Score</div>', unsafe_allow_html=True)
    
    anova_res = perform_income_spending_anova(df_clean, alpha=0.05)
    
    st.markdown("### 📑 Income Group Descriptive Breakdown")
    st.dataframe(anova_res['group_stats_df'], use_container_width=True)
    
    st.markdown("### 📐 ANOVA Summary Table")
    st.dataframe(anova_res['anova_table'].fillna("-"), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("F-Statistic", f"{anova_res['f_statistic']:.4f}")
        st.metric("ANOVA p-value", f"{anova_res['p_value']:.4f}")
        st.metric("Decision", anova_res['decision'])
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df_clean, x='Income_Group', y='Spending Score (1-100)', palette='Set2', ax=ax)
        ax.set_title("Spending Score across Income Groups")
        ax.set_xlabel("Income Tier")
        st.pyplot(fig)
        
    st.markdown("### 💡 Statistical Interpretation")
    st.info(anova_res['conclusion'])
    st.markdown("""
    - **Non-Parametric Kruskal-Wallis H-Test:** $H = 0.088$, $p = 0.957$ (re-confirms ANOVA result).
    - **Insight:** Because each income tier contains both ultra-low spenders (frugal) and ultra-high spenders (impulsive), the group means balance out around ~50. Hence, income group alone does not separate spending means linearly.
    """)

# ==============================================================================
# 7. CORRELATION & LINEAR REGRESSION
# ==============================================================================
elif page == "🔗 Correlation & Linear Regression":
    st.markdown('<div class="main-title">Correlation Analysis & Multiple Linear Regression (OLS)</div>', unsafe_allow_html=True)
    
    corr_res = compute_correlation_analysis(df_clean)
    lin_res = fit_linear_regression(df_clean)
    
    st.markdown("### 1. Pearson vs Spearman Correlation Matrix")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Pearson Correlation (Linear)**")
        st.dataframe(corr_res['pearson_corr'].round(3), use_container_width=True)
    with col2:
        st.markdown("**Spearman Rank Correlation (Monotonic)**")
        st.dataframe(corr_res['spearman_corr'].round(3), use_container_width=True)
        
    st.markdown("### 2. Multiple Linear Regression Model (OLS)")
    st.markdown("Predicting **Spending Score (1-100)** from **Age** and **Annual Income (k$)**:")
    
    st.dataframe(lin_res['summary_table'], use_container_width=True)
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("R-Squared (R²)", f"{lin_res['metrics']['R-squared']:.4f}")
    m_col2.metric("Adjusted R²", f"{lin_res['metrics']['Adjusted R-squared']:.4f}")
    m_col3.metric("Root Mean Squared Error (RMSE)", f"{lin_res['metrics']['RMSE']:.2f}")
    m_col4.metric("Durbin-Watson Stat", f"{lin_res['metrics']['Durbin-Watson Stat']:.2f}")
    
    colA, colB = st.columns(2)
    with colA:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x=lin_res['fitted_values'], y=lin_res['residuals'], color='#3B82F6', alpha=0.8, ax=ax)
        ax.axhline(0, color='red', linestyle='--')
        ax.set_title("Residuals vs Fitted Values")
        ax.set_xlabel("Fitted Values")
        ax.set_ylabel("Residuals")
        st.pyplot(fig)
    with colB:
        fig, ax = plt.subplots(figsize=(6, 4))
        stats.probplot(lin_res['residuals'], dist="norm", plot=ax)
        ax.set_title("Normal Q-Q Plot of Residuals")
        st.pyplot(fig)
        
    st.markdown(f"**Interpretation:** {lin_res['interpretation']}")

# ==============================================================================
# 8. FEATURE SCALING & LOGISTIC REGRESSION
# ==============================================================================
elif page == "⚙️ Feature Scaling & Logistic Regression":
    st.markdown('<div class="main-title">Feature Scaling Comparison & Logistic Regression</div>', unsafe_allow_html=True)
    
    st.markdown("### 1. Scaling Method Benchmark Table")
    scaling_comp = compare_feature_scaling(X_train, X_test, y_train, y_test)
    st.dataframe(scaling_comp, use_container_width=True)
    
    st.markdown("### 2. Logistic Regression Mathematical Formulation")
    st.latex(r"""
    \log\left(\frac{P(\text{High Spender})}{1 - P(\text{High Spender})}\right) = \beta_0 + \beta_1 \cdot \text{Age} + \beta_2 \cdot \text{Gender} + \beta_3 \cdot \text{Income}
    """)
    st.latex(r"""
    P(\text{High Spender} \mid X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_3)}}
    """)
    
    st.markdown("### 3. Model Parameters, Odds Ratios & Significance")
    st.write(f"**Model Intercept (β₀):** `{pipeline['intercept']:.4f}`")
    st.dataframe(pipeline['coefficients_df'], use_container_width=True)
    
    st.markdown("""
    **Parameter Interpretation:**
    - **Age (β = -0.591, Odds Ratio = 0.554, p = 0.0005):** Highly statistically significant ($p < 0.001$). For each 1 standard deviation increase in age, the odds of being a High Spender decrease by **44.6%** ($1 - 0.554$).
    - **Gender (β = -0.010, Odds Ratio = 0.990, p = 0.949):** Not statistically significant.
    - **Annual Income (β = -0.012, Odds Ratio = 0.988, p = 0.936):** Not statistically significant in a linear logistic model due to non-linear clustering.
    """)

# ==============================================================================
# 9. MODEL EVALUATION & CROSS-VALIDATION
# ==============================================================================
elif page == "🎯 Model Evaluation & K-Fold CV":
    st.markdown('<div class="main-title">Model Evaluation, ROC Curve & 5-Fold Cross Validation</div>', unsafe_allow_html=True)
    
    scaler = pipeline['scaler']
    X_test_sc = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    eval_res = evaluate_classifier(pipeline['model'], X_test_sc, y_test)
    cv_res = perform_kfold_cv(X, y, n_splits=5)
    
    # Metric cards
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{eval_res['raw_metrics']['accuracy']*100:.1f}%")
    m2.metric("Precision", f"{eval_res['raw_metrics']['precision']*100:.1f}%")
    m3.metric("Recall", f"{eval_res['raw_metrics']['recall']*100:.1f}%")
    m4.metric("F1-Score", f"{eval_res['raw_metrics']['f1']*100:.1f}%")
    m5.metric("ROC-AUC", f"{eval_res['raw_metrics']['auc']:.3f}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔲 Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(eval_res['confusion_matrix']['matrix'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Low Spender (0)', 'High Spender (1)'],
                    yticklabels=['Low Spender (0)', 'High Spender (1)'], ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Hold-out Test Confusion Matrix")
        st.pyplot(fig)
        
        cm_info = eval_res['confusion_matrix']['interpretations']
        st.caption(f"• {cm_info['TP']}")
        st.caption(f"• {cm_info['TN']}")
        st.caption(f"• {cm_info['FP']}")
        st.caption(f"• {cm_info['FN']}")
        
    with col2:
        st.markdown("### 📈 ROC Curve")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(eval_res['roc_curve']['fpr'], eval_res['roc_curve']['tpr'], color='#2563EB', lw=2,
                label=f"Logistic Regression (AUC = {eval_res['roc_curve']['auc']:.3f})")
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Chance Line (AUC = 0.50)')
        ax.set_xlabel("False Positive Rate (1 - Specificity)")
        ax.set_ylabel("True Positive Rate (Sensitivity)")
        ax.set_title("ROC Curve")
        ax.legend()
        st.pyplot(fig)
        
    st.markdown("### 🔁 5-Fold Stratified Cross-Validation Results")
    st.dataframe(cv_res['kfold_df'], use_container_width=True)
    
    st.markdown("### ⚖️ Generalization & Overfitting Analysis")
    train_acc = pipeline['model'].score(pipeline['scaler'].transform(X_train), y_train)
    gen_df = compare_generalization(train_acc, eval_res['raw_metrics']['accuracy'], cv_res['mean_accuracy'], cv_res['std_accuracy'])
    st.dataframe(gen_df, use_container_width=True)
    st.markdown("""
    - **Generalization Assessment:** The model demonstrates consistent performance between Training Accuracy (70.62%), Test Accuracy (72.50%), and 5-Fold Mean CV Accuracy (65.00% ± 5.86%).
    - **Academic Insight:** We avoid declaring high accuracy as proof of production readiness; the non-linear relationship between income and spending score suggests non-linear classifiers or clustering would capture additional consumer dynamics.
    """)

# ==============================================================================
# 10. PCA
# ==============================================================================
elif page == "🧬 PCA (Dimensionality Reduction)":
    st.markdown('<div class="main-title">Principal Component Analysis (PCA)</div>', unsafe_allow_html=True)
    
    pca_res = perform_pca_analysis(X, y)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📊 Variance Explained")
        st.dataframe(pca_res['variance_summary'], use_container_width=True)
        st.markdown("### 🧩 Principal Component Loadings")
        st.dataframe(pca_res['loadings'], use_container_width=True)
    with col2:
        st.markdown("### 🌌 2D Principal Component Projection")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(
            data=pca_res['pca_df'],
            x='Principal_Component_1',
            y='Principal_Component_2',
            hue='Spending_Category_Label',
            palette=['#EF4444', '#10B981'],
            alpha=0.85,
            s=70,
            ax=ax
        )
        ax.set_title("PCA: 2D Feature Space Projection")
        ax.set_xlabel(f"PC 1 ({pca_res['explained_variance_ratio'][0]*100:.1f}% var)")
        ax.set_ylabel(f"PC 2 ({pca_res['explained_variance_ratio'][1]*100:.1f}% var)")
        st.pyplot(fig)
        
    st.info(f"**Interpretation:** {pca_res['interpretation']}")

# ==============================================================================
# 11. BOOTSTRAP RESAMPLING
# ==============================================================================
elif page == "🔄 Bootstrap Resampling (Uncertainty)":
    st.markdown('<div class="main-title">Bootstrap Resampling & Confidence Intervals</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **Bootstrap Resampling** is a computational, non-parametric statistical technique that draws $B$ resamples with replacement from the observed dataset to construct empirical sampling distributions without assuming strict normality.
    """)
    
    n_boot = st.slider("Select Number of Bootstrap Replicates (B):", min_value=200, max_value=2000, value=1000, step=100)
    
    spending_boot = bootstrap_statistic_ci(df_clean['Spending Score (1-100)'].values, n_bootstraps=n_boot)
    income_boot = bootstrap_statistic_ci(df_clean['Annual_Income_INR'].values, n_bootstraps=n_boot)
    
    boot_summary = pd.DataFrame([
        {
            'Statistic': 'Mean Spending Score',
            'Sample Estimate': f"{spending_boot['point_estimate']:.2f}",
            'Bootstrap Mean': f"{spending_boot['bootstrap_mean']:.2f}",
            'Bootstrap SE': f"{spending_boot['bootstrap_se']:.3f}",
            '95% Percentile CI': f"[{spending_boot['ci_percentile'][0]:.2f}, {spending_boot['ci_percentile'][1]:.2f}]",
            '95% Normal CI': f"[{spending_boot['ci_normal'][0]:.2f}, {spending_boot['ci_normal'][1]:.2f}]"
        },
        {
            'Statistic': 'Mean Annual Income (INR)',
            'Sample Estimate': format_indian_currency(income_boot['point_estimate']),
            'Bootstrap Mean': format_indian_currency(income_boot['bootstrap_mean']),
            'Bootstrap SE': format_indian_currency(income_boot['bootstrap_se']),
            '95% Percentile CI': f"[{format_indian_currency(income_boot['ci_percentile'][0])}, {format_indian_currency(income_boot['ci_percentile'][1])}]",
            '95% Normal CI': f"[{format_indian_currency(income_boot['ci_normal'][0])}, {format_indian_currency(income_boot['ci_normal'][1])}]"
        }
    ])
    
    st.dataframe(boot_summary, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(spending_boot['bootstrap_distribution'], kde=True, color='#3B82F6', ax=ax)
        ax.axvline(spending_boot['ci_percentile'][0], color='red', linestyle='--', label='2.5th Pct')
        ax.axvline(spending_boot['ci_percentile'][1], color='red', linestyle='--', label='97.5th Pct')
        ax.axvline(spending_boot['point_estimate'], color='green', linestyle='-', lw=2, label='Point Estimate')
        ax.set_title("Bootstrap Distribution of Mean Spending Score")
        ax.set_xlabel("Mean Spending Score")
        ax.legend()
        st.pyplot(fig)
        
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(income_boot['bootstrap_distribution'] / 100000, kde=True, color='#10B981', ax=ax)
        ax.axvline(income_boot['ci_percentile'][0] / 100000, color='red', linestyle='--', label='2.5th Pct')
        ax.axvline(income_boot['ci_percentile'][1] / 100000, color='red', linestyle='--', label='97.5th Pct')
        ax.set_title("Bootstrap Distribution of Mean Income (₹ Lakhs)")
        ax.set_xlabel("Mean Annual Income (₹ Lakhs)")
        ax.legend()
        st.pyplot(fig)

# ==============================================================================
# 12. INTERACTIVE PREDICTION INTERFACE
# ==============================================================================
elif page == "🔮 Interactive Customer Prediction":
    st.markdown('<div class="main-title">Interactive Customer Spending Prediction</div>', unsafe_allow_html=True)
    st.markdown("Enter customer demographic and financial details below to predict spending tier probability using the trained Logistic Regression model.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📝 Customer Input Form")
        input_age = st.slider("Customer Age (Years):", min_value=18, max_value=85, value=28, step=1)
        input_gender = st.radio("Customer Gender:", options=["Female", "Male"], horizontal=True)
        
        income_lakhs = st.slider("Annual Income (in ₹ Lakhs):", min_value=5.0, max_value=150.0, value=12.5, step=0.5)
        input_income_inr = income_lakhs * 100000
        
        st.info(f"**Selected Income:** {format_indian_currency(input_income_inr)} (USD ~ ${input_income_inr / (1000*USD_TO_INR):.1f}k)")
        
    with col2:
        st.markdown("### 📊 Real-time Inference Results")
        
        pred_res = predict_customer_category(
            pipeline_dict=pipeline,
            age=input_age,
            gender=input_gender,
            annual_income_inr=input_income_inr
        )
        
        if pred_res['predicted_class'] == 1:
            st.success(f"### Predicted Class: 🛍️ **{pred_res['category_label']}**")
        else:
            st.error(f"### Predicted Class: 🏷️ **{pred_res['category_label']}**")
            
        st.markdown(f"**Probability of High Spending:** `{pred_res['prob_high_percent']:.2f}%`")
        st.progress(pred_res['prob_high_percent'] / 100.0)
        
        st.markdown(f"**Probability of Low Spending:** `{pred_res['prob_low_percent']:.2f}%`")
        st.progress(pred_res['prob_low_percent'] / 100.0)
        
        st.markdown("---")
        st.markdown("#### 🎯 Strategic Marketing Recommendation:")
        if pred_res['predicted_class'] == 1:
            st.markdown("""
            - **Targeting Strategy:** High engagement propensity detected.
            - **Recommended Action:** Deliver VIP loyalty offers, premium product catalogs, and exclusive in-store early access.
            """)
        else:
            st.markdown("""
            - **Targeting Strategy:** Value-conscious or conservative spending profile.
            - **Recommended Action:** Deliver discount coupons, bundle value deals, and utility-driven marketing campaigns.
            """)

# Footer
st.markdown("---")
st.markdown(
    "<center><small>Statistics for Artificial Intelligence & Data Science | Academic Project Demonstration</small></center>",
    unsafe_allow_html=True
)
