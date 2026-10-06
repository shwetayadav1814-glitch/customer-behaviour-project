"""
Principal Component Analysis (PCA) Module for Customer Behaviour Analysis
Subject: Statistics for Artificial Intelligence & Data Science

This module implements:
1. Feature Standardization for PCA.
2. Eigen-decomposition / SVD of standardized feature covariance matrix.
3. 2D Dimensionality Reduction (PC1 and PC2).
4. Explained Variance Ratio and Cumulative Explained Variance.
5. Component Loadings (eigenvectors) interpretation.
6. 2D projection annotated with High Spender / Low Spender classes.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def perform_pca_analysis(X: pd.DataFrame, y: pd.Series, y_labels: pd.Series = None) -> dict:
    """
    Perform PCA on predictive input features (Age, Gender_Encoded, Annual Income (k$)).
    Reduces feature dimensionality from 3D to 2D for orthogonal visualization.
    """
    feature_names = list(X.columns)
    
    # 1. Feature Standardization (Zero Mean, Unit Variance)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. PCA with 2 Components
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(X_scaled)
    
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # 3. Create DataFrame of Principal Components
    pca_df = pd.DataFrame(pcs, columns=['Principal_Component_1', 'Principal_Component_2'])
    pca_df['Spending_Category'] = y.values
    if y_labels is not None:
        pca_df['Spending_Category_Label'] = y_labels.values
    else:
        pca_df['Spending_Category_Label'] = y.map({1: 'High Spender', 0: 'Low Spender'}).values
        
    # 4. Loadings Matrix (Eigenvectors * sqrt(eigenvalues))
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=['PC1_Loading', 'PC2_Loading'],
        index=feature_names
    )
    
    # Summary of Explained Variance
    variance_summary = pd.DataFrame({
        'Component': ['PC 1', 'PC 2'],
        'Explained Variance Ratio': explained_variance_ratio,
        'Explained Variance (%)': explained_variance_ratio * 100.0,
        'Cumulative Variance (%)': cumulative_variance * 100.0
    })
    
    interpretation = (
        f"Principal Component 1 explains {explained_variance_ratio[0]*100:.2f}% of the total variance, "
        f"and Principal Component 2 explains {explained_variance_ratio[1]*100:.2f}%. "
        f"Together, the two components capture {cumulative_variance[1]*100:.2f}% of the total variance in the 3D feature space. "
        f"Examination of the loadings shows the relative contribution of Age, Gender, and Annual Income along each orthogonal axis. "
        f"Note: PCA is an unsupervised linear projection technique used here for dimensionality reduction and data visualization; "
        f"it does not inherently guarantee improved downstream classification performance unless empirically validated."
    )
    
    return {
        'pca_model': pca,
        'scaler': scaler,
        'pca_df': pca_df,
        'loadings': loadings,
        'variance_summary': variance_summary,
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance': cumulative_variance,
        'interpretation': interpretation
    }
