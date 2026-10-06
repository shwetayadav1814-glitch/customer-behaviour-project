"""
Logistic Regression Module for Customer Behaviour Analysis
Subject: Statistics for Artificial Intelligence & Data Science

This module implements:
1. Feature Scaling Comparison (No Scaling vs StandardScaler vs MinMaxScaler).
2. Binary Logistic Regression classification (High Spender vs Low Spender).
3. Statistical Parameter Estimation: Intercept, Coefficients, Odds Ratios (e^β), z-scores, and p-values.
4. User-friendly inference pipeline for real-time predictions.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import statsmodels.api as sm
from .data_preprocessing import USD_TO_INR, prepare_train_test_datasets, get_feature_target_split


def compare_feature_scaling(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                            y_train: pd.Series, y_test: pd.Series) -> pd.DataFrame:
    """
    Compare Logistic Regression performance across:
    1. No Scaling (Raw Features)
    2. Standardization (StandardScaler: mean=0, std=1)
    3. Normalization (MinMaxScaler: range [0, 1])
    """
    scaling_methods = {
        'No Scaling': (X_train.copy(), X_test.copy()),
        'Standardization': (
            pd.DataFrame(StandardScaler().fit_transform(X_train), columns=X_train.columns),
            pd.DataFrame(StandardScaler().fit(X_train).transform(X_test), columns=X_test.columns)
        ),
        'Normalization': (
            pd.DataFrame(MinMaxScaler().fit_transform(X_train), columns=X_train.columns),
            pd.DataFrame(MinMaxScaler().fit(X_train).transform(X_test), columns=X_test.columns)
        )
    }
    
    results = []
    
    for method_name, (X_tr, X_te) in scaling_methods.items():
        # Train Logistic Regression
        clf = LogisticRegression(random_state=42, solver='lbfgs', max_iter=1000)
        clf.fit(X_tr, y_train)
        
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        
        results.append({
            'Scaling Method': method_name,
            'Accuracy': float(acc),
            'Precision': float(prec),
            'Recall': float(rec),
            'F1 Score': float(f1),
            'AUC': float(auc)
        })
        
    return pd.DataFrame(results)


def train_logistic_regression_pipeline(X_train: pd.DataFrame, y_train: pd.Series, 
                                      scaler_type: str = 'standard') -> dict:
    """
    Train a production-grade Logistic Regression pipeline:
    - Fits scaler on X_train.
    - Fits Scikit-learn Logistic Regression for inference.
    - Fits Statsmodels Logit for asymptotic standard errors and p-values.
    """
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = None
        
    if scaler is not None:
        X_train_scaled = scaler.fit_transform(X_train)
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    else:
        X_train_scaled_df = X_train.copy()
        
    # Scikit-Learn Model
    sklearn_model = LogisticRegression(random_state=42, solver='lbfgs', max_iter=1000)
    sklearn_model.fit(X_train_scaled_df, y_train)
    
    # Statsmodels Logit for inference & p-values
    X_train_sm = sm.add_constant(X_train_scaled_df)
    try:
        sm_model = sm.Logit(y_train, X_train_sm).fit(disp=False)
        pvalues = sm_model.pvalues
        bse = sm_model.bse
    except Exception:
        sm_model = None
        pvalues = pd.Series([np.nan] * (len(X_train.columns) + 1), index=['const'] + list(X_train.columns))
        bse = pd.Series([np.nan] * (len(X_train.columns) + 1), index=['const'] + list(X_train.columns))
        
    # Intercept & Coefficients
    intercept = float(sklearn_model.intercept_[0])
    coefficients = sklearn_model.coef_[0]
    odds_ratios = np.exp(coefficients)
    
    coef_records = []
    for feat, coef, odds_r in zip(X_train.columns, coefficients, odds_ratios):
        pval = pvalues.get(feat, np.nan)
        std_err = bse.get(feat, np.nan)
        
        # Interpretation
        if coef > 0:
            direction = f"Positively associated with High Spending (Odds Ratio = {odds_r:.3f}, increases odds by {(odds_r - 1)*100:.1f}%)"
        else:
            direction = f"Negatively associated with High Spending (Odds Ratio = {odds_r:.3f}, decreases odds by {(1 - odds_r)*100:.1f}%)"
            
        coef_records.append({
            'Feature': feat,
            'Coefficient (β)': float(coef),
            'Odds Ratio (e^β)': float(odds_r),
            'Std Error': float(std_err) if not np.isnan(std_err) else np.nan,
            'p-value': float(pval) if not np.isnan(pval) else np.nan,
            'Interpretation': direction
        })
        
    coef_df = pd.DataFrame(coef_records)
    
    return {
        'model': sklearn_model,
        'statsmodels_logit': sm_model,
        'scaler': scaler,
        'intercept': intercept,
        'coefficients_df': coef_df,
        'feature_names': list(X_train.columns)
    }


def predict_customer_category(pipeline_dict: dict, age: float, gender: str, annual_income_inr: float) -> dict:
    """
    Predict customer spending category from raw user inputs:
    - Age (years)
    - Gender ('Male' or 'Female')
    - Annual Income in INR (₹)
    
    Converts Annual Income (INR) back to Annual Income (k$) using fixed rate,
    applies fitted scaler, and computes prediction probabilities.
    """
    model = pipeline_dict['model']
    scaler = pipeline_dict['scaler']
    
    # Preprocessing
    gender_encoded = 1 if gender.strip().lower() == 'male' else 0
    annual_income_k = annual_income_inr / (1000.0 * USD_TO_INR)
    
    # Create input DataFrame matching training feature columns
    input_df = pd.DataFrame([[age, gender_encoded, annual_income_k]], 
                            columns=['Age', 'Gender_Encoded', 'Annual Income (k$)'])
    
    # Scale input using the fitted scaler
    if scaler is not None:
        input_scaled = scaler.transform(input_df)
    else:
        input_scaled = input_df.values
        
    # Predict probabilities
    prob_array = model.predict_proba(input_scaled)[0]
    prob_low = float(prob_array[0])
    prob_high = float(prob_array[1])
    
    predicted_class = int(prob_high >= 0.5)
    category_label = "High Spender" if predicted_class == 1 else "Low Spender"
    
    return {
        'predicted_class': predicted_class,
        'category_label': category_label,
        'prob_high_percent': prob_high * 100.0,
        'prob_low_percent': prob_low * 100.0,
        'input_summary': {
            'Age': age,
            'Gender': gender,
            'Annual_Income_INR': annual_income_inr,
            'Annual_Income_k$': annual_income_k,
            'Gender_Encoded': gender_encoded
        }
    }
