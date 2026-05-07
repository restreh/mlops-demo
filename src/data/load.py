"""
Ingesta y validación del dataset.

Responsabilidades:
- Cargar el dataset Breast Cancer Wisconsin de sklearn.
- Persistirlo como parquet bajo data/raw/ (entrada del pipeline).
- Validaciones mínimas: tipos, presencia de columnas, NaNs.

Este módulo aísla el origen de los datos para que el resto del pipeline
trabaje siempre contra archivos parquet locales (o, en producción,
contra una feature store).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer


# Columna objetivo del dataset
TARGET = "target"

# Carpetas estándar del proyecto
DATA_RAW = Path("data/raw")


def load_raw() -> pd.DataFrame:
    """Carga el dataset crudo desde sklearn y lo devuelve como DataFrame.

    El dataset Breast Cancer Wisconsin (Diagnostic) tiene 569 muestras,
    30 features numéricas y una etiqueta binaria. Se usa porque viene
    incluido en sklearn, no requiere descarga externa y permite obtener
    AUC > 0.95, el umbral fijado en params.yaml.
    """
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    df[TARGET] = data.target
    return df


def validate(df: pd.DataFrame) -> None:
    """Validaciones mínimas de calidad de datos.

    Lanza un AssertionError si alguna invariante esperada no se cumple.
    En un sistema productivo aquí entrarían herramientas como Great
    Expectations, Pandera o el módulo de tests de Evidently.
    """
    assert TARGET in df.columns, f"Falta la columna objetivo '{TARGET}'"
    assert df.isna().sum().sum() == 0, "Se encontraron NaN en el dataset"
    assert len(df) > 100, "Dataset demasiado pequeño"
    assert df[TARGET].nunique() == 2, "El target debe ser binario"


def save_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Persiste el DataFrame como parquet, creando carpetas si es necesario."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def main() -> None:
    """Punto de entrada CLI: carga, valida y guarda el dataset crudo."""
    df = load_raw()
    validate(df)
    out = save_parquet(df, DATA_RAW / "breast_cancer.parquet")
    print(f"[load] {len(df)} filas guardadas en {out}")


if __name__ == "__main__":
    main()
