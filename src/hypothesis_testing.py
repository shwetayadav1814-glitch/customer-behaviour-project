"""
Hypothesis Testing Module for Customer Behaviour Analysis
Subject: Statistics for Artificial Intelligence & Data Science

This module performs rigorous parametric and non-parametric hypothesis testing:
1. Two-Sample Independent t-Test (Student's and Welch's)
2. Mann-Whitney U Test (Non-parametric rank-sum test)
3. Assumption validations (Shapiro-Wilk for normality, Levene's test for equality of variances)
4. Decision rule: alpha = 0.05
"""

import pandas as pd
import numpy as np
from scipy import stats


def test_gender_vs_spending_score(df: pd.DataFrame, alpha: float = 0.05) -> dict:
    """
    Hypothesis Test 1: Difference in Spending Score between Male and Female Customers.
    
    Null Hypothesis (H0): mu_Male = mu_Female
    (The population mean Spending Score is equal between Male and Female customers).
    
    Alternative Hypothesis (H1): mu_Male != mu_Female
    (The population mean Spending Score differs significantly between Male and Female customers).
    
    Significance Level: alpha = 0.05
    """
    male_scores = df[df['Gender'] == 'Male']['Spending Score (1-100)'].dropna()
    female_scores = df[df['Gender'] == 'Female']['Spending Score (1-100)'].dropna()
    
    n_male = len(male_scores)
    n_female = len(female_scores)
    
    mean_male = float(male_scores.mean())
    mean_female = float(female_scores.mean())
    
    std_male = float(male_scores.std(ddof=1))
    std_female = float(female_scores.std(ddof=1))
    
    # 1. Assumption Check: Normality (Shapiro-Wilk)
    shapiro_male_stat, shapiro_male_p = stats.shapiro(male_scores)
    shapiro_female_stat, shapiro_female_p = stats.shapiro(female_scores)
    
    # 2. Assumption Check: Homoscedasticity (Levene's Test for equal variance)
    levene_stat, levene_p = stats.levene(male_scores, female_scores)
    equal_var = levene_p >= alpha
    
    # 3. Parametric Test: Two-sample Student's t-test (and Welch's t-test)
    student_t_stat, student_t_p = stats.ttest_ind(male_scores, female_scores, equal_var=True)
    welch_t_stat, welch_t_p = stats.ttest_ind(male_scores, female_scores, equal_var=False)
    
    # Selected primary parametric test based on Levene's outcome
    chosen_t_stat = student_t_stat if equal_var else welch_t_stat
    chosen_t_p = student_t_p if equal_var else welch_t_p
    test_type_used = "Student's Two-Sample Independent t-Test (Equal Variance Assumed)" if equal_var else "Welch's Two-Sample t-Test (Unequal Variance)"
    
    # Degrees of freedom for Student's t-test
    df_freedom = (n_male + n_female - 2) if equal_var else None
    
    # 4. Non-Parametric Test: Mann-Whitney U Test
    u_stat, u_p = stats.mannwhitneyu(male_scores, female_scores, alternative='two-sided')
    
    # Decision
    is_rejected = chosen_t_p < alpha
    decision = "Reject Null Hypothesis (H0)" if is_rejected else "Fail to Reject Null Hypothesis (H0)"
    
    if is_rejected:
        conclusion = (
            f"At alpha = {alpha}, there is statistically significant evidence (p = {chosen_t_p:.4f} < {alpha}) "
            f"to reject H0. The mean Spending Score differs significantly between Male ({mean_male:.2f}) "
            f"and Female ({mean_female:.2f}) customers."
        )
    else:
        conclusion = (
            f"At alpha = {alpha}, we fail to reject H0 (t = {chosen_t_stat:.4f}, p = {chosen_t_p:.4f} >= {alpha}). "
            f"The observed difference in sample means (Male: {mean_male:.2f} vs Female: {mean_female:.2f}) is not "
            f"statistically significant and can be attributed to random sampling variability."
        )
        
    return {
        "null_hypothesis": "H0: mu_Male = mu_Female (No difference in mean spending score between genders)",
        "alt_hypothesis": "H1: mu_Male != mu_Female (Significant difference in mean spending score between genders)",
        "alpha": alpha,
        "sample_sizes": {"Male": n_male, "Female": n_female, "Total": n_male + n_female},
        "sample_means": {"Male": mean_male, "Female": mean_female},
        "sample_stds": {"Male": std_male, "Female": std_female},
        "normality_check": {
            "Male_Shapiro_stat": shapiro_male_stat, "Male_Shapiro_p": shapiro_male_p,
            "Female_Shapiro_stat": shapiro_female_stat, "Female_Shapiro_p": shapiro_female_p
        },
        "levene_test": {"statistic": levene_stat, "p_value": levene_p, "equal_var_assumed": equal_var},
        "test_used": test_type_used,
        "t_statistic": chosen_t_stat,
        "p_value": chosen_t_p,
        "degrees_of_freedom": df_freedom,
        "mann_whitney_u": {"statistic": u_stat, "p_value": u_p},
        "decision": decision,
        "conclusion": conclusion
    }


def get_hypothesis_summary_table(hyp_res: dict) -> pd.DataFrame:
    """
    Create a clean summary table of the hypothesis test results.
    """
    rows = [
        {"Parameter / Metric": "Null Hypothesis (H0)", "Value": hyp_res["null_hypothesis"]},
        {"Parameter / Metric": "Alternative Hypothesis (H1)", "Value": hyp_res["alt_hypothesis"]},
        {"Parameter / Metric": "Significance Level (α)", "Value": f"{hyp_res['alpha']}"},
        {"Parameter / Metric": "Male Sample Size (n1)", "Value": f"{hyp_res['sample_sizes']['Male']}"},
        {"Parameter / Metric": "Female Sample Size (n2)", "Value": f"{hyp_res['sample_sizes']['Female']}"},
        {"Parameter / Metric": "Male Mean Spending Score", "Value": f"{hyp_res['sample_means']['Male']:.2f}"},
        {"Parameter / Metric": "Female Mean Spending Score", "Value": f"{hyp_res['sample_means']['Female']:.2f}"},
        {"Parameter / Metric": "Levene's Test p-value (Equal Variance)", "Value": f"{hyp_res['levene_test']['p_value']:.4f}"},
        {"Parameter / Metric": "Parametric Test Applied", "Value": hyp_res["test_used"]},
        {"Parameter / Metric": "t-statistic", "Value": f"{hyp_res['t_statistic']:.4f}"},
        {"Parameter / Metric": "t-test p-value", "Value": f"{hyp_res['p_value']:.4f}"},
        {"Parameter / Metric": "Mann-Whitney U Statistic", "Value": f"{hyp_res['mann_whitney_u']['statistic']:.1f}"},
        {"Parameter / Metric": "Mann-Whitney U p-value", "Value": f"{hyp_res['mann_whitney_u']['p_value']:.4f}"},
        {"Parameter / Metric": "Statistical Decision", "Value": hyp_res["decision"]}
    ]
    return pd.DataFrame(rows)
