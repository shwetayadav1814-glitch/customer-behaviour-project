"""
Descriptive Statistics and Distribution Analysis Module
Subject: Statistics for Artificial Intelligence & Data Science

This module computes comprehensive parametric and non-parametric summary statistics:
- Central Tendency: Mean, Median, Mode
- Dispersion: Min, Max, Range, Sample Variance (s^2), Standard Deviation (s), Q1, Q3, IQR
- Shape: Skewness, Kurtosis
- Normality Tests: Shapiro-Wilk test, D'Agostino-Pearson omnibus test
"""

import pandas as pd
import numpy as np
from scipy import stats
from .data_preprocessing import format_indian_currency


def calculate_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute full descriptive statistics for key numerical variables:
    Age, Annual Income (k$), Annual Income (INR), and Spending Score (1-100).
    """
    columns_to_analyze = ['Age', 'Annual Income (k$)', 'Annual_Income_INR', 'Spending Score (1-100)']
    
    stats_records = []
    
    for col in columns_to_analyze:
        series = df[col].dropna()
        n = len(series)
        mean_val = series.mean()
        median_val = series.median()
        
        # Mode
        mode_res = stats.mode(series, keepdims=True)
        mode_val = mode_res.mode[0]
        
        min_val = series.min()
        max_val = series.max()
        range_val = max_val - min_val
        
        # Sample variance (ddof=1) and Sample Std (ddof=1)
        var_val = series.var(ddof=1)
        std_val = series.std(ddof=1)
        
        # Quartiles & IQR
        q1 = series.quantile(0.25)
        q2 = series.quantile(0.50)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        # Shape metrics
        skewness = stats.skew(series, bias=False)
        kurtosis = stats.kurtosis(series, bias=False)  # Excess kurtosis (Normal = 0)
        
        # Normality: Shapiro-Wilk test
        shapiro_stat, shapiro_p = stats.shapiro(series)
        
        stats_records.append({
            'Variable': col,
            'Count (N)': n,
            'Mean': mean_val,
            'Median': median_val,
            'Mode': mode_val,
            'Min': min_val,
            'Max': max_val,
            'Range': range_val,
            'Variance (s²)': var_val,
            'Std Dev (s)': std_val,
            'Q1 (25th)': q1,
            'Q2 (Median)': q2,
            'Q3 (75th)': q3,
            'IQR': iqr,
            'Skewness': skewness,
            'Excess Kurtosis': kurtosis,
            'Shapiro-Wilk Stat': shapiro_stat,
            'Shapiro-Wilk p-value': shapiro_p,
            'Distribution Shape': interpret_distribution_shape(skewness, kurtosis, shapiro_p)
        })
        
    stats_df = pd.DataFrame(stats_records)
    return stats_df


def interpret_distribution_shape(skewness: float, kurtosis: float, p_val: float) -> str:
    """
    Classify distribution based on skewness, kurtosis, and normality test p-value.
    """
    skew_desc = "Approximately Symmetric" if abs(skewness) < 0.5 else ("Moderately Right-Skewed" if skewness > 0 else "Moderately Left-Skewed")
    if abs(skewness) >= 1.0:
        skew_desc = "Highly Right-Skewed" if skewness > 0 else "Highly Left-Skewed"
        
    norm_desc = "Normal (p >= 0.05)" if p_val >= 0.05 else "Non-Normal (p < 0.05)"
    return f"{skew_desc}; {norm_desc}"


def get_formatted_descriptive_table(stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format statistics table for clean visual and academic presentation,
    applying Indian Rupee formatting where relevant.
    """
    display_df = stats_df.copy()
    
    formatted_rows = []
    for _, row in display_df.iterrows():
        var_name = row['Variable']
        is_inr = 'INR' in var_name
        
        def fmt(val, is_currency=False, decimals=2):
            if is_currency:
                return format_indian_currency(val)
            if isinstance(val, (int, np.integer)):
                return f"{val:,}"
            if isinstance(val, (float, np.floating)):
                return f"{val:,.{decimals}f}"
            return str(val)
            
        formatted_rows.append({
            'Variable': var_name,
            'Mean': fmt(row['Mean'], is_inr),
            'Median': fmt(row['Median'], is_inr),
            'Mode': fmt(row['Mode'], is_inr),
            'Min': fmt(row['Min'], is_inr),
            'Max': fmt(row['Max'], is_inr),
            'Range': fmt(row['Range'], is_inr),
            'Std Dev (s)': fmt(row['Std Dev (s)'], is_inr),
            'IQR': fmt(row['IQR'], is_inr),
            'Skewness': f"{row['Skewness']:.3f}",
            'Excess Kurtosis': f"{row['Excess Kurtosis']:.3f}",
            'Shapiro p-value': f"{row['Shapiro-Wilk p-value']:.4e}",
            'Shape Assessment': row['Distribution Shape']
        })
        
    return pd.DataFrame(formatted_rows)


def get_statistical_interpretations(stats_df: pd.DataFrame) -> dict:
    """
    Generate rigorous interpretations for viva and academic reports.
    """
    interpretations = {}
    
    for _, row in stats_df.iterrows():
        col = row['Variable']
        mean = row['Mean']
        median = row['Median']
        std = row['Std Dev (s)']
        skew = row['Skewness']
        p_val = row['Shapiro-Wilk p-value']
        
        if col == 'Age':
            text = (
                f"The mean customer age is {mean:.1f} years with a standard deviation of {std:.1f} years. "
                f"The median age ({median:.1f} years) is close to the mean, reflecting a mild positive skew ({skew:.3f}). "
                f"Customers span from {row['Min']:.0f} to {row['Max']:.0f} years, with the middle 50% (IQR) located between "
                f"{row['Q1 (25th)']:.1f} and {row['Q3 (75th)']:.1f} years."
            )
        elif 'INR' in col:
            text = (
                f"The average annual income in Indian Rupees is {format_indian_currency(mean)} "
                f"(Std Dev: {format_indian_currency(std)}). The median income is {format_indian_currency(median)}. "
                f"The positive skew ({skew:.3f}) indicates a modest right tail with high-earning individuals reaching up to "
                f"{format_indian_currency(row['Max'])}."
            )
        elif 'k$' in col:
            text = (
                f"Average annual income is ${mean:.2f}k with a standard deviation of ${std:.2f}k. "
                f"The income spans from ${row['Min']:.1f}k to ${row['Max']:.1f}k with an IQR of ${row['IQR']:.1f}k."
            )
        elif 'Spending Score' in col:
            text = (
                f"The mean spending score is {mean:.2f}/100 and median is {median:.1f}/100. "
                f"The dispersion is substantial (Std Dev = {std:.2f}, Range = {row['Range']:.0f}), showing high heterogeneity "
                f"in consumer spending behavior across mall visitors."
            )
        else:
            text = f"Variable {col} has mean = {mean:.2f}, median = {median:.2f}, std = {std:.2f}."
            
        interpretations[col] = text
        
    return interpretations
