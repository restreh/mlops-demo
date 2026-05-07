"""
Promueve la última versión registrada del modelo a la etapa 'Production'.

En un proyecto real esta promoción sería disparada por un pipeline de
CI tras superar tests de aceptación. Aquí se expone como script
independiente para que el demo se pueda completar manualmente:
después de ejecutar src/models/train.py, este script marca la versión
recién creada como 'Production' para que la API la cargue.

Uso:
    python scripts/promote_model.py
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import yaml
from mlflow.tracking import MlflowClient


def main() -> None:
    cfg = yaml.safe_load(Path("params.yaml").read_text())["mlflow"]
    mlflow.set_tracking_uri(cfg["tracking_uri"])

    client = MlflowClient()
    name = cfg["registered_model_name"]

    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise SystemExit(
            f"No hay versiones registradas para '{name}'. "
            "Ejecuta primero src/models/train.py."
        )

    latest = max(versions, key=lambda v: int(v.version))
    client.transition_model_version_stage(
        name=name,
        version=latest.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"[promote] '{name}' v{latest.version} → Production")


if __name__ == "__main__":
    main()
