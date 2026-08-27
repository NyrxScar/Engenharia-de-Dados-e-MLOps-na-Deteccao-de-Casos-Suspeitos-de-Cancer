# preprocessing.py — Pré-processamento com Prevenção de Data Leakage

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_data(
    df: pd.DataFrame,
    feature_names: list[str],
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
  
    X = df[feature_names]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
   
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def prepare_pipeline(
    df: pd.DataFrame,
    feature_names: list[str],
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, StandardScaler]:

    X_train, X_test, y_train, y_test = split_data(
        df, feature_names, test_size, random_state
    )
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
