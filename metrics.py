# metrics.py — Métricas de Classificação para Cenários Desbalanceados

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.50,
) -> dict:

    y_pred = (y_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "Threshold": threshold,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Acuracia": accuracy_score(y_true, y_pred),
        "Precisao": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1_Score": fbeta_score(y_true, y_pred, beta=1.0, zero_division=0),
        "F2_Score": fbeta_score(y_true, y_pred, beta=2.0, zero_division=0),
        "F0.5_Score": fbeta_score(y_true, y_pred, beta=0.5, zero_division=0),
    }


def compute_auc_roc(
    y_true: np.ndarray,
    y_probs: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:

    auc = roc_auc_score(y_true, y_probs)
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    return auc, fpr, tpr
