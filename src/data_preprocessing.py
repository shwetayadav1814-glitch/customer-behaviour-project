"""
Data Preprocessing Module for Customer Behaviour Analysis
Subject: Statistics for Artificial Intelligence & Data Science

This module handles:
1. Loading the Mall Customers dataset.
2. Checking data integrity (missing values, duplicates, out-of-range values).
3. Currency conversion from USD (k$) to Indian Rupees (INR) with a fixed conversion rate.
4. Indian currency formatting (e.g., ₹8,50,000, ₹12,50,000).
5. Categorical encoding for Gender.
6. Target feature engineering (Spending_Category: High Spender vs Low Spender).
7. Train-Test splitting with stratification and feature scaling pipelines.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ==========================================
# FIXED CURRENCY CONVERSION CONSTANT
# ==========================================
# Fixed USD to INR conversion rate clearly documented for reproducibility.
# 1 USD = 83.0 INR (Fixed benchmark exchange rate)
USD_TO_INR: float = 83.0


def format_indian_currency(val) -> str:
    """
    Format a numerical value into the Indian numbering system representation.
    Examples:
        500000   -> ₹5,00,000
        850000   -> ₹8,50,000
        1000000  -> ₹10,00,000
        1250000  -> ₹12,50,000
        11371000 -> ₹1,13,71,000
    """
    if pd.isna(val):
        return "N/A"
    try:
        val = float(val)
        is_negative = val < 0
        val = abs(val)
        int_part = int(round(val))
        s = str(int_part)
        if len(s) <= 3:
            res = s
        else:
            last_three = s[-3:]
            remaining = s[:-3]
            parts = []
            while len(remaining) > 2:
                parts.insert(0, remaining[-2:])
                remaining = remaining[:-2]
            if remaining:
                parts.insert(0, remaining)
            res = ",".join(parts) + "," + last_three
        prefix = "-₹" if is_negative else "₹"
        return f"{prefix}{res}"
    except Exception:
        return f"₹{val}"


def load_raw_data(filepath: str = "data/Mall_Customers.csv") -> pd.DataFrame:
    """
    Load raw Mall Customers dataset from CSV file.
    """
    df = pd.read_csv(filepath)
    return df


def inspect_data_integrity(df: pd.DataFrame) -> dict:
    """
    Perform structural and data hygiene checks:
    - Missing values count per column
    - Duplicate records count
    - Data types
    - Range checks for Age, Annual Income, Spending Score
    """
    integrity_report = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "total_missing": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "age_range": (int(df['Age'].min()), int(df['Age'].max())),
        "income_k_range": (float(df['Annual Income (k$)'].min()), float(df['Annual Income (k$)'].max())),
        "spending_score_range": (int(df['Spending Score (1-100)'].min()), int(df['Spending Score (1-100)'].max()))
    }
    return integrity_report


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform end-to-end data preprocessing:
    1. Verify no nulls/duplicates.
    2. Numerical Gender Encoding: Male = 1, Female = 0.
    3. Currency conversion: Annual_Income_INR = Annual_Income_USD_K * 1000 * USD_TO_INR.
    4. Target Variable Creation:
       - Spending_Category = 1 (High Spender) if Spending Score >= 50
       - Spending_Category = 0 (Low Spender) if Spending Score < 50
    5. Income Group Categorization (for ANOVA and Group analysis):
       - Low Income: < $40k (< ₹33,20,000)
       - Medium Income: $40k - $80k (₹33,20,000 - ₹66,40,000)
       - High Income: > $80k (> ₹66,40,000)
    """
    df_clean = df.copy()

    # Drop duplicates if any exist
    if df_clean.duplicated().sum() > 0:
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)

    # Encode Gender: Female -> 0, Male -> 1
    df_clean['Gender_Encoded'] = df_clean['Gender'].map({'Female': 0, 'Male': 1}).astype(int)

    # Convert Annual Income (k$) to Annual Income (INR)
    # Note: Currency conversion changes the numerical scale but preserves ranking/correlation.
    df_clean['Annual_Income_INR'] = df_clean['Annual Income (k$)'] * 1000 * USD_TO_INR

    # Formatted display string column for user-facing tables
    df_clean['Annual_Income_INR_Formatted'] = df_clean['Annual_Income_INR'].apply(format_indian_currency)

    # Target Feature Creation
    df_clean['Spending_Category'] = (df_clean['Spending Score (1-100)'] >= 50).astype(int)
    df_clean['Spending_Category_Label'] = df_clean['Spending_Category'].map({1: 'High Spender', 0: 'Low Spender'})

    # Create meaningful Income Groups for ANOVA
    # Low Income: < 40k $, Medium Income: 40k - 80k $, High Income: > 80k $
    bins = [0, 40, 80, np.inf]
    labels = ['Low Income (< ₹33.2L)', 'Medium Income (₹33.2L - ₹66.4L)', 'High Income (> ₹66.4L)']
    df_clean['Income_Group'] = pd.cut(df_clean['Annual Income (k$)'], bins=bins, labels=labels, right=False)

    return df_clean


def get_feature_target_split(df_clean: pd.DataFrame):
    """
    Separate primary predictive features (Age, Gender_Encoded, Annual Income (k$))
    and Target (Spending_Category).
    IMPORTANT: Spending Score (1-100) is EXCLUDED from features to prevent data leakage.
    """
    feature_cols = ['Age', 'Gender_Encoded', 'Annual Income (k$)']
    X = df_clean[feature_cols].copy()
    y = df_clean['Spending_Category'].copy()
    return X, y, feature_cols


def prepare_train_test_datasets(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Split the dataset into training and testing subsets using stratified sampling
    to preserve class distribution proportions.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def apply_scaling(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Fit scalers strictly on training data and transform both train and test partitions.
    Returns:
    - Original unscaled data
    - Standardized data (StandardScaler: z = (x - mu) / sigma)
    - Normalized data (MinMaxScaler: x_norm = (x - min) / (max - min))
    - Fitted scaler objects
    """
    # Standardization
    std_scaler = StandardScaler()
    X_train_std = pd.DataFrame(std_scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_std = pd.DataFrame(std_scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    # Normalization
    minmax_scaler = MinMaxScaler()
    X_train_norm = pd.DataFrame(minmax_scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_norm = pd.DataFrame(minmax_scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    scalers = {
        'standard': std_scaler,
        'minmax': minmax_scaler
    }

    return {
        'none': (X_train, X_test),
        'standard': (X_train_std, X_test_std),
        'minmax': (X_train_norm, X_test_norm),
        'scalers': scalers
    }
