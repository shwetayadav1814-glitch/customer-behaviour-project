"""
Correlation and Linear Regression Module for Customer Behaviour Analysis
Subject: Statistics for Artificial Intelligence & Data Science

This module performs:
1. Pearson (Linear) and Spearman (Monotonic) Correlation Analysis with p-values.
2. Multiple Linear Regression using Ordinary Least Squares (OLS).
3. Goodness-of-fit evaluation (R², Adjusted R², MSE, RMSE, MAE, F-test).
4. Comprehensive Residual Diagnostics (Normality, Homoscedasticity, Autocorrelation).
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan


def compute_correlation_analysis(df: pd.DataFrame) -> dict:
    """
    Compute Pearson and Spearman correlation matrices and pairwise p-values.
    Analyzes: Age, Annual Income (k$), Spending Score (1-100).
    """
    cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    subset = df[cols].dropna()
    
    # Pearson Correlation Matrix
    pearson_corr = subset.corr(method='pearson')
    
    # Spearman Correlation Matrix
    spearman_corr = subset.corr(method='spearman')
    
    # Calculate Pearson p-values matrix
    p_matrix = pd.DataFrame(np.zeros((len(cols), len(cols))), index=cols, columns=cols)
    for col1 in cols:
        for col2 in cols:
            if col1 == col2:
                p_matrix.loc[col1, col2] = 0.0
            else:
                _, p_val = stats.pearsonr(subset[col1], subset[col2])
                p_matrix.loc[col1, col2] = p_val
                
    interpretations = {
        'Age_vs_Spending': (
            f"Pearson r = {pearson_corr.loc['Age', 'Spending Score (1-100)']:.3f} "
            f"(p = {p_matrix.loc['Age', 'Spending Score (1-100)']:.4f}). "
            "Indicates a moderate negative correlation: older customers on average tend to exhibit lower spending scores."
        ),
        'Income_vs_Spending': (
            f"Pearson r = {pearson_corr.loc['Annual Income (k$)', 'Spending Score (1-100)']:.3f} "
            f"(p = {p_matrix.loc['Annual Income (k$)', 'Spending Score (1-100)']:.4f}). "
            "Near-zero linear correlation overall, because spending patterns diverge into distinct clusters "
            "(high earners with low spending vs high earners with high spending) rather than following a monotonic line."
        ),
        'Age_vs_Income': (
            f"Pearson r = {pearson_corr.loc['Age', 'Annual Income (k$)']:.3f} "
            f"(p = {p_matrix.loc['Age', 'Annual Income (k$)']:.4f}). "
            "Very weak correlation between customer age and income in this mall dataset."
        ),
        'Disclaimer': "IMPORTANT: Correlation measures statistical association only and does not imply causation."
    }
    
    return {
        'pearson_corr': pearson_corr,
        'spearman_corr': spearman_corr,
        'p_matrix': p_matrix,
        'interpretations': interpretations
    }


def fit_linear_regression(df: pd.DataFrame) -> dict:
    """
    Fit Ordinary Least Squares (OLS) Multiple Linear Regression model:
    Target (Y): Spending Score (1-100)
    Predictors (X): Age, Annual Income (k$)
    """
    cols = ['Age', 'Annual Income (k$)']
    X = df[cols].copy()
    y = df['Spending Score (1-100)'].copy()
    
    # Add constant intercept column
    X_with_const = sm.add_constant(X)
    
    ols_model = sm.OLS(y, X_with_const).fit()
    
    # Fitted values and residuals
    y_pred = ols_model.fittedvalues
    residuals = ols_model.resid
    
    # Error metrics
    mse = float(np.mean(residuals ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals)))
    
    # Coefficient table
    coef_table = pd.DataFrame({
        'Feature': ols_model.params.index,
        'Coefficient (β)': ols_model.params.values,
        'Std Error': ols_model.bse.values,
        't-Statistic': ols_model.tvalues.values,
        'p-value': ols_model.pvalues.values,
        '95% CI Lower': ols_model.conf_int()[0].values,
        '95% CI Upper': ols_model.conf_int()[1].values
    })
    
    # Diagnostic tests
    dw_stat = durbin_watson(residuals)
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    
    # Breusch-Pagan test for heteroscedasticity
    bp_test = het_breuschpagan(residuals, X_with_const)
    bp_stat, bp_pval = bp_test[0], bp_test[1]
    
    model_summary_metrics = {
        'R-squared': ols_model.rsquared,
        'Adjusted R-squared': ols_model.rsquared_adj,
        'F-Statistic': ols_model.fvalue,
        'F-test p-value': ols_model.f_pvalue,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'Durbin-Watson Stat': dw_stat,
        'Shapiro Residuals p-val': shapiro_p,
        'Breusch-Pagan p-val': bp_pval
    }
    
    interpretation = (
        f"The linear regression model yields R² = {ols_model.rsquared:.4f}, explaining approximately "
        f"{ols_model.rsquared * 100:.1f}% of the variance in Spending Score. "
        f"The coefficient for Age is β = {ols_model.params['Age']:.4f} (p = {ols_model.pvalues['Age']:.4e}), "
        f"indicating that for each additional year of age, the spending score decreases by approx {abs(ols_model.params['Age']):.2f} points, "
        f"holding income constant. Income (β = {ols_model.params['Annual Income (k$)']:.4f}, p = {ols_model.pvalues['Annual Income (k$)']:.4f}) "
        f"is not a statistically significant linear predictor alone. "
        f"Note: Regression coefficients reflect observed linear associations and should not be interpreted as causal proof."
    )
    
    return {
        'model': ols_model,
        'summary_table': coef_table,
        'metrics': model_summary_metrics,
        'fitted_values': y_pred,
        'residuals': residuals,
        'interpretation': interpretation
    }
