"""
Bootstrap Resampling Module for Customer Behaviour Analysis
Subject: Statistics for Artificial Intelligence & Data Science

This module implements:
1. Non-parametric Bootstrap Resampling (B iterations with replacement).
2. Point estimation, Standard Error (SE_boot), and Bias calculation.
3. 95% Confidence Interval estimation using:
   - Percentile Method (2.5th and 97.5th percentiles)
   - Normal Approximation (theta_hat +/- 1.96 * SE_boot)
4. Bootstrap evaluation for Mean Spending Score, Mean Income (INR), and Model Accuracy.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from .data_preprocessing import format_indian_currency


def bootstrap_statistic_ci(data: np.ndarray, stat_func=np.mean, n_bootstraps: int = 1000, 
                           ci_level: float = 0.95, random_state: int = 42) -> dict:
    """
    Perform bootstrap resampling for an arbitrary statistical function (e.g., mean, median).
    """
    rng = np.random.RandomState(random_state)
    n = len(data)
    
    # Original point estimate
    theta_hat = float(stat_func(data))
    
    # Generate bootstrap replicates
    boot_estimates = np.empty(n_bootstraps)
    for i in range(n_bootstraps):
        resample = rng.choice(data, size=n, replace=True)
        boot_estimates[i] = stat_func(resample)
        
    boot_mean = float(np.mean(boot_estimates))
    boot_se = float(np.std(boot_estimates, ddof=1))
    boot_bias = boot_mean - theta_hat
    
    # Percentile CI
    alpha_half = (1.0 - ci_level) / 2.0
    ci_lower_pct = float(np.percentile(boot_estimates, alpha_half * 100))
    ci_upper_pct = float(np.percentile(boot_estimates, (1.0 - alpha_half) * 100))
    
    # Normal Approximation CI
    z_critical = 1.95996  # for 95% CI
    ci_lower_norm = float(theta_hat - z_critical * boot_se)
    ci_upper_norm = float(theta_hat + z_critical * boot_se)
    
    return {
        'point_estimate': theta_hat,
        'bootstrap_mean': boot_mean,
        'bootstrap_se': boot_se,
        'bootstrap_bias': boot_bias,
        'ci_percentile': (ci_lower_pct, ci_upper_pct),
        'ci_normal': (ci_lower_norm, ci_upper_norm),
        'bootstrap_distribution': boot_estimates,
        'n_bootstraps': n_bootstraps,
        'ci_level': ci_level
    }


def bootstrap_model_accuracy(X: pd.DataFrame, y: pd.Series, n_bootstraps: int = 500, 
                             random_state: int = 42) -> dict:
    """
    Evaluate Logistic Regression generalization uncertainty via bootstrap resampling.
    In each iteration, resample the dataset with replacement, train the model, 
    and evaluate out-of-bag (OOB) accuracy.
    """
    rng = np.random.RandomState(random_state)
    n = len(X)
    indices = np.arange(n)
    
    oob_accuracies = []
    
    for i in range(n_bootstraps):
        boot_idx = rng.choice(indices, size=n, replace=True)
        oob_idx = np.setdiff1d(indices, boot_idx)
        
        if len(oob_idx) == 0:
            continue
            
        X_boot, y_boot = X.iloc[boot_idx], y.iloc[boot_idx]
        X_oob, y_oob = X.iloc[oob_idx], y.iloc[oob_idx]
        
        # Check if both classes are present in training resample
        if len(np.unique(y_boot)) < 2:
            continue
            
        scaler = StandardScaler()
        X_boot_sc = scaler.fit_transform(X_boot)
        X_oob_sc = scaler.transform(X_oob)
        
        clf = LogisticRegression(random_state=42, solver='lbfgs', max_iter=500)
        clf.fit(X_boot_sc, y_boot)
        
        pred_oob = clf.predict(X_oob_sc)
        oob_accuracies.append(accuracy_score(y_oob, pred_oob))
        
    oob_acc_arr = np.array(oob_accuracies)
    mean_oob = float(np.mean(oob_acc_arr))
    se_oob = float(np.std(oob_acc_arr, ddof=1))
    ci_lower = float(np.percentile(oob_acc_arr, 2.5))
    ci_upper = float(np.percentile(oob_acc_arr, 97.5))
    
    return {
        'mean_oob_accuracy': mean_oob,
        'se_oob_accuracy': se_oob,
        'ci_95': (ci_lower, ci_upper),
        'oob_distribution': oob_acc_arr,
        'n_successful_bootstraps': len(oob_acc_arr)
    }


def run_full_bootstrap_suite(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Run complete bootstrap analysis suite on Spending Score, Annual Income (INR), and Model Accuracy.
    """
    spending_boot = bootstrap_statistic_ci(df['Spending Score (1-100)'].values, stat_func=np.mean, n_bootstraps=1000)
    income_boot = bootstrap_statistic_ci(df['Annual_Income_INR'].values, stat_func=np.mean, n_bootstraps=1000)
    model_boot = bootstrap_model_accuracy(X, y, n_bootstraps=500)
    
    summary_df = pd.DataFrame([
        {
            'Parameter / Metric': 'Mean Spending Score',
            'Point Estimate': f"{spending_boot['point_estimate']:.2f}",
            'Bootstrap Mean': f"{spending_boot['bootstrap_mean']:.2f}",
            'Bootstrap SE': f"{spending_boot['bootstrap_se']:.3f}",
            '95% Percentile CI': f"[{spending_boot['ci_percentile'][0]:.2f}, {spending_boot['ci_percentile'][1]:.2f}]",
            '95% Normal CI': f"[{spending_boot['ci_normal'][0]:.2f}, {spending_boot['ci_normal'][1]:.2f}]"
        },
        {
            'Parameter / Metric': 'Mean Annual Income (INR)',
            'Point Estimate': format_indian_currency(income_boot['point_estimate']),
            'Bootstrap Mean': format_indian_currency(income_boot['bootstrap_mean']),
            'Bootstrap SE': format_indian_currency(income_boot['bootstrap_se']),
            '95% Percentile CI': f"[{format_indian_currency(income_boot['ci_percentile'][0])}, {format_indian_currency(income_boot['ci_percentile'][1])}]",
            '95% Normal CI': f"[{format_indian_currency(income_boot['ci_normal'][0])}, {format_indian_currency(income_boot['ci_normal'][1])}]"
        },
        {
            'Parameter / Metric': 'Out-of-Bag Model Accuracy',
            'Point Estimate': f"{model_boot['mean_oob_accuracy'] * 100:.2f}%",
            'Bootstrap Mean': f"{model_boot['mean_oob_accuracy'] * 100:.2f}%",
            'Bootstrap SE': f"{model_boot['se_oob_accuracy'] * 100:.2f}%",
            '95% Percentile CI': f"[{model_boot['ci_95'][0]*100:.2f}%, {model_boot['ci_95'][1]*100:.2f}%]",
            '95% Normal CI': 'N/A (Bounded)'
        }
    ])
    
    return {
        'spending_boot': spending_boot,
        'income_boot': income_boot,
        'model_boot': model_boot,
        'summary_df': summary_df
    }
