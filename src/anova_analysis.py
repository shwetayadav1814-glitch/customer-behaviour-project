"""
ANOVA (Analysis of Variance) Module for Customer Behaviour Analysis
Subject: Statistics for Artificial Intelligence & Data Science

This module performs:
1. One-Way ANOVA to compare Spending Scores across Income Groups (Low, Medium, High).
2. ANOVA Table construction (Sum of Squares, df, Mean Squares, F-ratio, p-value).
3. Assumption validations (Levene's test for homoscedasticity, Shapiro-Wilk on residuals).
4. Non-parametric Kruskal-Wallis H-test.
5. Post-hoc Tukey HSD pairwise comparisons.
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def perform_income_spending_anova(df: pd.DataFrame, alpha: float = 0.05) -> dict:
    """
    Perform One-Way ANOVA across Income Groups:
    - Low Income (< $40k / < ₹33.2L)
    - Medium Income ($40k - $80k / ₹33.2L - ₹66.4L)
    - High Income (> $80k / > ₹66.4L)
    
    Null Hypothesis (H0): mu_Low = mu_Medium = mu_High
    (The population mean Spending Score is equal across all income groups).
    
    Alternative Hypothesis (H1): At least one income group has a different mean Spending Score.
    """
    df_clean = df.copy()
    
    # Ensure Income_Group is present
    if 'Income_Group' not in df_clean.columns:
        bins = [0, 40, 80, np.inf]
        labels = ['Low Income (< ₹33.2L)', 'Medium Income (₹33.2L - ₹66.4L)', 'High Income (> ₹66.4L)']
        df_clean['Income_Group'] = pd.cut(df_clean['Annual Income (k$)'], bins=bins, labels=labels, right=False)
        
    groups = df_clean.groupby('Income_Group', observed=True)['Spending Score (1-100)']
    
    group_names = list(groups.groups.keys())
    group_data = [group.values for _, group in groups]
    
    group_stats = []
    for name, group in groups:
        group_stats.append({
            'Income Group': str(name),
            'Sample Size (n)': len(group),
            'Mean Spending Score': float(group.mean()),
            'Std Dev': float(group.std(ddof=1)),
            'Median': float(group.median()),
            'Min': float(group.min()),
            'Max': float(group.max())
        })
    group_stats_df = pd.DataFrame(group_stats)
    
    # One-Way ANOVA
    f_stat, p_val = stats.f_oneway(*group_data)
    
    # Calculate ANOVA Sum of Squares manually for full transparency
    grand_mean = df_clean['Spending Score (1-100)'].mean()
    total_n = len(df_clean)
    k_groups = len(group_data)
    
    ss_between = sum(len(g) * ((np.mean(g) - grand_mean) ** 2) for g in group_data)
    ss_within = sum(sum((x - np.mean(g)) ** 2 for x in g) for g in group_data)
    ss_total = ss_between + ss_within
    
    df_between = k_groups - 1
    df_within = total_n - k_groups
    df_total = total_n - 1
    
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    
    # Residuals for assumption testing
    residuals = []
    for g in group_data:
        residuals.extend(g - np.mean(g))
    residuals = np.array(residuals)
    
    # Assumption 1: Normality of residuals
    shapiro_resid_stat, shapiro_resid_p = stats.shapiro(residuals)
    
    # Assumption 2: Homoscedasticity across groups (Levene's test)
    levene_stat, levene_p = stats.levene(*group_data)
    
    # Non-Parametric alternative: Kruskal-Wallis H-test
    kw_stat, kw_p = stats.kruskal(*group_data)
    
    # Post-Hoc Tukey HSD test
    tukey = pairwise_tukeyhsd(endog=df_clean['Spending Score (1-100)'],
                              groups=df_clean['Income_Group'].astype(str),
                              alpha=alpha)
    tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
    
    is_rejected = p_val < alpha
    decision = "Reject Null Hypothesis (H0)" if is_rejected else "Fail to Reject Null Hypothesis (H0)"
    
    if is_rejected:
        conclusion = (
            f"At α = {alpha}, the One-Way ANOVA reveals statistically significant differences in mean Spending Score "
            f"across income groups (F({df_between}, {df_within}) = {f_stat:.4f}, p = {p_val:.4e} < {alpha}). "
            f"Post-hoc Tukey HSD identifies which specific income brackets differ significantly."
        )
    else:
        conclusion = (
            f"At α = {alpha}, we fail to reject H0 (F({df_between}, {df_within}) = {f_stat:.4f}, p = {p_val:.4f} >= {alpha}). "
            f"There is no statistically significant difference in mean Spending Score across the Low, Medium, and High Income groups. "
            f"Customers in different income brackets exhibit similar average spending scores in this sample."
        )
        
    anova_table = pd.DataFrame([
        {
            'Source of Variation': 'Between Groups (Treatment)',
            'Sum of Squares (SS)': ss_between,
            'Degrees of Freedom (df)': df_between,
            'Mean Square (MS)': ms_between,
            'F-Statistic': f_stat,
            'p-value': p_val
        },
        {
            'Source of Variation': 'Within Groups (Error/Residual)',
            'Sum of Squares (SS)': ss_within,
            'Degrees of Freedom (df)': df_within,
            'Mean Square (MS)': ms_within,
            'F-Statistic': np.nan,
            'p-value': np.nan
        },
        {
            'Source of Variation': 'Total',
            'Sum of Squares (SS)': ss_total,
            'Degrees of Freedom (df)': df_total,
            'Mean Square (MS)': np.nan,
            'F-Statistic': np.nan,
            'p-value': np.nan
        }
    ])
    
    return {
        'null_hypothesis': 'H0: mu_Low = mu_Medium = mu_High (Mean Spending Score is equal across income tiers)',
        'alt_hypothesis': 'H1: At least one income tier has a different mean Spending Score',
        'alpha': alpha,
        'group_stats_df': group_stats_df,
        'anova_table': anova_table,
        'f_statistic': f_stat,
        'p_value': p_val,
        'df_between': df_between,
        'df_within': df_within,
        'levene_test': {'statistic': levene_stat, 'p_value': levene_p},
        'shapiro_residuals': {'statistic': shapiro_resid_stat, 'p_value': shapiro_resid_p},
        'kruskal_wallis': {'statistic': kw_stat, 'p_value': kw_p},
        'tukey_hsd_df': tukey_df,
        'decision': decision,
        'conclusion': conclusion
    }
