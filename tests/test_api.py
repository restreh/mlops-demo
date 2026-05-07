"""
Tests de contrato de la API.

No requieren un MLflow corriendo: el modelo se mockea con un objeto
trivial. El objetivo es verificar el shape de request/response y los
códigos de estado. Tests de integración con MLflow real se harían
en un job aparte de CI.
"""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

import src.api.main as api_module


class _FakeModel:
    """Modelo trivial: devuelve 1 para cada fila.

    Imita el contrato de un modelo cargado vía mlflow.pyfunc.load_model:
    recibe un DataFrame y devuelve un numpy array.
    """

    def predict(self, df):
        return np.ones(len(df), dtype=int)


def test_predict_returns_predictions(monkeypatch):
    # Sustituye la variable global 'model' por un fake.
    monkeypatch.setattr(api_module, "model", _FakeModel())
    # Pasa por encima del lifespan para no intentar conectarse a MLflow.
    api_module.app.router.lifespan_context = lambda app: _NoopLifespan()

    with TestClient(api_module.app) as client:
        r = client.post("/predict", json={"data": [[0.0] * 30, [1.0] * 30]})
    assert r.status_code == 200
    assert r.json() == {"predictions": [1, 1]}


def test_health_reports_loaded_state(monkeypatch):
    monkeypatch.setattr(api_module, "model", _FakeModel())
    api_module.app.router.lifespan_context = lambda app: _NoopLifespan()

    with TestClient(api_module.app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


# Helper: contexto de lifespan que no hace nada.
class _NoopLifespan:
    async def __aenter__(self):
        return None

    async def __aexit__(self, exc_type, exc, tb):
        return False
