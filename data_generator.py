# data_generator.py — Geração de Dados Sintéticos de Sensores Industriais

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


# Nomes das features simulando sensores industriais reais
FEATURE_NAMES = [
    "sensor_vibracao_eixo_x",
    "sensor_vibracao_eixo_y",
    "sensor_vibracao_eixo_z",
    "temp_motor_principal",
    "pressao_hidraulica",
    "torque_braco_robotico",
    "rpm_esteira",
    "consumo_corrente_motor",
    "ruido_acustico_db",
    "fluxo_lubrificante",
]


def generate_industrial_data(
    n_samples: int = 20_000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, list[str]]:

    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_repeated=0,
        n_clusters_per_class=2,
        weights=[0.995, 0.005],
        flip_y=0,
        class_sep=1.0,
        random_state=random_state,
    )

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["target"] = y
    return df, FEATURE_NAMES
