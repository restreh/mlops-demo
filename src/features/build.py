"""
Feature engineering y generación de los datasets de monitoreo.

Para este demo no hay transformaciones pesadas porque el dataset
Breast Cancer ya viene con features numéricas listas. El módulo
cumple dos objetivos:

1. Producir la matriz de features lista para entrenamiento
   (data/processed/dataset.parquet).
2. Generar dos datasets que el módulo de drift comparará después:
   - reference.parquet: distribución observada al momento del
     entrenamiento (snapshot estable).
   - last_24h.parquet: muestra reciente "en producción". Aquí se
     introduce un shift artificial sobre algunas features para que
     el reporte de Evidently encuentre drift detectable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load import DATA_RAW

DATA_PROCESSED = Path("data/processed")
DATA_PRODUCTION = Path("data/production")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica el feature engineering.

    En este demo el dataset ya está limpio. Se deja el hook explícito
    para que en una extensión real (otra fuente de datos, codificación
    categórica, escalado) este sea el único punto de cambio.
    """
    return df.copy()


def make_reference_and_current(
    df: pd.DataFrame, drift_seed: int = 7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Genera dos particiones para el monitoreo de drift.

    Se parte el dataset por la mitad. La segunda mitad simula la
    distribución observada en producción y se le inyecta un shift
    multiplicativo sobre tres features. Esto reproduce un escenario
    típico: el sensor o el proceso de captura cambia y la
    distribución de entrada se desplaza sin que cambie el modelo.

    Args:
        df: dataset completo procesado.
        drift_seed: semilla del shift, fijada para reproducibilidad.

    Returns:
        (reference, current) como dos DataFrames disjuntos.
    """
    rng = np.random.default_rng(drift_seed)
    half = len(df) // 2
    reference = df.iloc[:half].copy().reset_index(drop=True)
    current = df.iloc[half:].copy().reset_index(drop=True)

    # Inyección de drift en tres features representativas.
    # Multiplicador > 1 desplaza la media; ruido aditivo aumenta varianza.
    drifted_cols = ["mean radius", "mean texture", "mean perimeter"]
    for col in drifted_cols:
        if col in current.columns:
            shift = 1.15 + 0.05 * rng.standard_normal(len(current))
            current[col] = current[col] * shift

    return reference, current


def main() -> None:
    """Pipeline de features: lee raw, construye features, genera particiones."""
    raw_path = DATA_RAW / "breast_cancer.parquet"
    df = pd.read_parquet(raw_path)
    df = build_features(df)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    DATA_PRODUCTION.mkdir(parents=True, exist_ok=True)

    df.to_parquet(DATA_PROCESSED / "dataset.parquet", index=False)

    reference, current = make_reference_and_current(df)
    reference.to_parquet(DATA_PROCESSED / "reference.parquet", index=False)
    current.to_parquet(DATA_PRODUCTION / "last_24h.parquet", index=False)

    print(
        f"[features] dataset={len(df)} | "
        f"reference={len(reference)} | current={len(current)}"
    )


if __name__ == "__main__":
    main()
