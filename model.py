# model.py — Treinamento e Predição do Modelo de Classificação

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    random_state: int = 42,
) -> RandomForestClassifier:
   
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def predict_probabilities(
    model: RandomForestClassifier,
    X: np.ndarray,
) -> np.ndarray:

    return model.predict_proba(X)[:, 1]
