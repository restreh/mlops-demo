"""
Entrenamiento del modelo con MLflow Tracking y Model Registry.

Este script reproduce literalmente el snippet mostrado en la
diapositiva "Entrenamiento con MLflow", añadiendo:
- Manejo de rutas relativas y validación de configuración.
- Registro condicional: el modelo se registra solo si supera un umbral
  de calidad (auc_threshold en params.yaml). Es la regla de promoción
  más simple posible y refleja un Quality Gate de CI/CD.
- Comentarios sobre cada artefacto registrado.

Para ejecutarlo, MLflow debe estar corriendo en mlflow.tracking_uri
(por defecto http://localhost:5000).
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.sklearn
import yaml
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def main() -> None:
    # 1. Carga de configuración externa (sin valores hard-coded en el script)
    cfg = yaml.safe_load(Path("params.yaml").read_text())
    train_params = cfg["train"]
    mlflow_cfg = cfg["mlflow"]
    data_cfg = cfg["data"]

    # 2. Configuración de MLflow Tracking
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    # 3. Datos: se usa el dataset de sklearn directamente para que el demo
    #    sea autocontenido. En un proyecto real este paso leería de
    #    data/processed/dataset.parquet (producido por src/features/build.py).
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=data_cfg["test_size"],
        stratify=y,
        random_state=data_cfg["random_state"],
    )

    # 4. Run de MLflow: todo lo que ocurre dentro queda trazado
    with mlflow.start_run() as run:
        mlflow.log_params(train_params)

        model = RandomForestClassifier(**train_params).fit(Xtr, ytr)
        auc = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
        mlflow.log_metric("roc_auc", auc)

        print(f"[train] run_id={run.info.run_id} | AUC={auc:.4f}")

        # 5. Promoción condicional: solo se registra el modelo si supera
        #    el umbral. Esto previene que runs malos lleguen al registry.
        if auc >= mlflow_cfg["auc_threshold"]:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=mlflow_cfg["registered_model_name"],
                input_example=Xtr.iloc[:2],
            )
            print(
                f"[train] Modelo registrado en '{mlflow_cfg['registered_model_name']}' "
                f"(AUC {auc:.4f} >= {mlflow_cfg['auc_threshold']})"
            )
        else:
            print(
                f"[train] AUC {auc:.4f} por debajo del umbral "
                f"{mlflow_cfg['auc_threshold']}: NO se registra el modelo."
            )


if __name__ == "__main__":
    main()
