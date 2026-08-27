# financial_analysis.py — Análise Financeira e Ajuste Dinâmico do Threshold


import numpy as np
import pandas as pd

from metrics import evaluate_metrics


# Faixa de thresholds a avaliar (de 0.05 a 0.95 em passos de 0.01)
THRESHOLDS = np.arange(0.05, 0.96, 0.01)


def analyze_thresholds(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    fn_cost: float = 50_000.0,
    fp_cost: float = 500.0,
    thresholds: np.ndarray = THRESHOLDS,
) -> pd.DataFrame:
  
    results = []
    for t in thresholds:
        m = evaluate_metrics(y_true, y_probs, t)
        m["Custo_Total"] = (m["FP"] * fp_cost) + (m["FN"] * fn_cost)
        results.append(m)
    return pd.DataFrame(results)


def find_optimal_threshold(results_df: pd.DataFrame) -> pd.Series:

    return results_df.loc[results_df["Custo_Total"].idxmin()]


def compute_savings(
    results_df: pd.DataFrame,
    default_threshold: float = 0.50,
) -> tuple[pd.Series, pd.Series, float, float]:

    best_row = find_optimal_threshold(results_df)
    default_row = results_df[
        np.isclose(results_df["Threshold"], default_threshold)
    ].iloc[0]

    economia_abs = default_row["Custo_Total"] - best_row["Custo_Total"]
    economia_pct = (economia_abs / default_row["Custo_Total"]) * 100 if default_row["Custo_Total"] > 0 else 0.0

    return best_row, default_row, economia_abs, economia_pct
