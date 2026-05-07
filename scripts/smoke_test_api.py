"""
Prueba humo de la API de inferencia.

Envía 5 muestras del dataset Breast Cancer al endpoint /predict y
muestra las predicciones. Sirve para validar manualmente que el
servicio levantó correctamente y que el modelo está disponible.

Uso (con la API en localhost:8000):
    python scripts/smoke_test_api.py
"""

from __future__ import annotations

import json
import urllib.request

from sklearn.datasets import load_breast_cancer


API_URL = "http://localhost:8000"


def main() -> None:
    # 1. Health
    with urllib.request.urlopen(f"{API_URL}/health") as r:
        print("[health]", json.loads(r.read()))

    # 2. Predicción
    X, _ = load_breast_cancer(return_X_y=True, as_frame=True)
    payload = {"data": X.head(5).values.tolist()}
    req = urllib.request.Request(
        f"{API_URL}/predict",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as r:
        print("[predict]", json.loads(r.read()))


if __name__ == "__main__":
    main()
