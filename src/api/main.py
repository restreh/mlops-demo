"""
API de inferencia con FastAPI.

Reproduce el snippet de la diapositiva "Serving con FastAPI". El modelo
se carga UNA sola vez al iniciar el proceso (lifespan asíncrono) desde
el Model Registry de MLflow. La imagen Docker no contiene el modelo;
esto desacopla el ciclo de vida del código del ciclo de vida del modelo.

Variables de entorno requeridas:
- MLFLOW_TRACKING_URI: URI del tracking server (ej. http://mlflow:5000).
- MODEL_NAME: nombre del modelo registrado (ej. breast-cancer-rf).
- MODEL_STAGE: etapa a cargar (Staging, Production, etc.).

Endpoints:
- GET  /health   → liveness/readiness probe.
- POST /predict  → predicción batch (lista de listas de floats).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Nombres de las 30 features del dataset Breast Cancer, en el orden esperado
# por el modelo. Se mantienen como constante para no depender de sklearn
# en runtime.
FEATURE_NAMES = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension",
]


MODEL_URI = f"models:/{os.getenv('MODEL_NAME', 'breast-cancer-rf')}/{os.getenv('MODEL_STAGE', 'Production')}"

# Variable global donde queda el modelo después del lifespan.
# Se inicializa a None para distinguir "no cargado" de "cargado sin error".
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo al iniciar el proceso y lo libera al apagar.

    Esto evita el costo de cargar el modelo en cada request y permite
    que un fallo en la carga aborte el arranque del servidor.
    """
    global model
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    model = mlflow.pyfunc.load_model(MODEL_URI)
    yield
    model = None


app = FastAPI(
    title="Breast Cancer Classifier API",
    version="1.0.0",
    lifespan=lifespan,
)


class Features(BaseModel):
    """Esquema del request: matriz n×30 con las features del dataset."""
    data: list[list[float]]


@app.get("/health")
def health() -> dict:
    """Endpoint de salud. Devuelve 200 si el modelo está cargado."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(p: Features) -> dict:
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    df = pd.DataFrame(p.data, columns=FEATURE_NAMES)
    preds = model.predict(df)
    return {"predictions": preds.tolist()}
