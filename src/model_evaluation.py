"""
Model Evaluation and Cross-Validation Module for Customer Behaviour Analysis
Subject: Statistics for Artificial Intelligence & Data Science

This module performs:
1. Binary Classification Metrics (Accuracy, Precision, Recall, Specificity, F1-Score, ROC-AUC).
2. Confusion Matrix decomposition (TP, TN, FP, FN) with customer marketing domain context.
3. ROC (Receiver Operating Characteristic) and Precision-Recall Curve calculations.
4. 5-Fold Stratified Cross-Validation with per-fold metrics and variance analysis.
5. Model Generalization and Overfitting/Underfitting diagnostics.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
    matthews_corrcoef, balanced_accuracy_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def evaluate_classifier(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Compute comprehensive classification metrics on the hold-out test set.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    # Binary layout: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # ROC Curve points
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
    
    # Precision-Recall Curve points
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_test, y_prob)
    
    metrics_summary_df = pd.DataFrame([
        {'Metric': 'Accuracy', 'Score': acc, 'Description': 'Overall proportion of correct classifications'},
        {'Metric': 'Precision', 'Score': prec, 'Description': 'Accuracy of High Spender positive predictions'},
        {'Metric': 'Recall (Sensitivity)', 'Score': rec, 'Description': 'Proportion of actual High Spenders detected'},
        {'Metric': 'Specificity', 'Score': specificity, 'Description': 'Proportion of actual Low Spenders detected'},
        {'Metric': 'F1-Score', 'Score': f1, 'Description': 'Harmonic mean of Precision and Recall'},
        {'Metric': 'Balanced Accuracy', 'Score': bal_acc, 'Description': 'Macro-average of recall across both classes'},
        {'Metric': 'ROC-AUC', 'Score': auc, 'Description': 'Area under the ROC curve (discrimination ability)'},
        {'Metric': 'Matthews Corr Coef (MCC)', 'Score': mcc, 'Description': 'Correlation between actual and predicted labels'}
    ])
    
    cm_dict = {
        'matrix': cm,
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'TP': int(tp),
        'interpretations': {
            'TP': f"True Positives ({tp}): High Spenders correctly identified. Enables targeted VIP/loyalty promotions.",
            'TN': f"True Negatives ({tn}): Low Spenders correctly identified. Avoids misallocating luxury marketing campaigns.",
            'FP': f"False Positives ({fp}) [Type I Error]: Low Spenders incorrectly classified as High Spenders. May lead to inefficient marketing budget spend.",
            'FN': f"False Negatives ({fn}) [Type II Error]: High Spenders missed by the model. Represents potential lost revenue from unaddressed premium shoppers."
        }
    }
    
    return {
        'metrics_df': metrics_summary_df,
        'confusion_matrix': cm_dict,
        'roc_curve': {'fpr': fpr, 'tpr': tpr, 'thresholds': roc_thresholds, 'auc': auc},
        'pr_curve': {'precision': pr_precision, 'recall': pr_recall, 'thresholds': pr_thresholds},
        'raw_metrics': {
            'accuracy': acc, 'precision': prec, 'recall': rec,
            'specificity': specificity, 'f1': f1, 'auc': auc
        }
    }


def perform_kfold_cv(X: pd.DataFrame, y: pd.Series, n_splits: int = 5, random_state: int = 42) -> dict:
    """
    Perform 5-Fold Stratified Cross-Validation on the full dataset
    with rigorous in-fold feature scaling to eliminate data leakage.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_records = []
    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_f1s = []
    fold_aucs = []
    
    for fold_num, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Scaling strictly inside fold
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_val_sc = scaler.transform(X_val)
        
        clf = LogisticRegression(random_state=42, solver='lbfgs', max_iter=1000)
        clf.fit(X_tr_sc, y_tr)
        
        pred = clf.predict(X_val_sc)
        prob = clf.predict_proba(X_val_sc)[:, 1]
        
        acc = accuracy_score(y_val, pred)
        prec = precision_score(y_val, pred, zero_division=0)
        rec = recall_score(y_val, pred, zero_division=0)
        f1 = f1_score(y_val, pred, zero_division=0)
        auc = roc_auc_score(y_val, prob)
        
        fold_accuracies.append(acc)
        fold_precisions.append(prec)
        fold_recalls.append(rec)
        fold_f1s.append(f1)
        fold_aucs.append(auc)
        
        fold_records.append({
            'Fold': f"Fold {fold_num}",
            'Validation Score (Accuracy)': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1,
            'ROC-AUC': auc
        })
        
    kfold_df = pd.DataFrame(fold_records)
    
    mean_acc = float(np.mean(fold_accuracies))
    std_acc = float(np.std(fold_accuracies, ddof=1))
    
    mean_f1 = float(np.mean(fold_f1s))
    std_f1 = float(np.std(fold_f1s, ddof=1))
    
    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs, ddof=1))
    
    return {
        'kfold_df': kfold_df,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'mean_f1': mean_f1,
        'std_f1': std_f1,
        'mean_auc': mean_auc,
        'std_auc': std_auc
    }


def compare_generalization(train_acc: float, test_acc: float, cv_mean_acc: float, cv_std_acc: float) -> pd.DataFrame:
    """
    Compare training score, test score, and K-Fold CV score to evaluate generalization stability.
    """
    comparison_df = pd.DataFrame([
        {'Evaluation Mode': 'Training Set (Train Split)', 'Accuracy': f"{train_acc*100:.2f}%", 'Notes': 'Evaluated on training partition (n=160)'},
        {'Evaluation Mode': 'Hold-out Test Set (Test Split)', 'Accuracy': f"{test_acc*100:.2f}%", 'Notes': 'Evaluated on unseen hold-out partition (n=40)'},
        {'Evaluation Mode': '5-Fold Cross-Validation (Mean ± Std)', 'Accuracy': f"{cv_mean_acc*100:.2f}% ± {cv_std_acc*100:.2f}%", 'Notes': 'Averaged over 5 stratified data splits'}
    ])
    return comparison_df
