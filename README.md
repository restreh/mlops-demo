# mlops-demo

Pipeline de referencia para la presentación de **MLOps / Deployment** del curso de Machine Learning, Maestría en Ciencia de Datos.

El proyecto reproduce los snippets mostrados en las diapositivas y los integra en un flujo ejecutable end-to-end:

```
Datos  →  train.py  →  MLflow Tracking  →  Model Registry  →  FastAPI  →  Cliente
                                                                            │
                                                                            ▼
                                                                   Evidently (drift)
                                                                            │
                                                                            ▼
                                                              Trigger de reentrenamiento
```

Corresponde al **Nivel 1** de la madurez de MLOps definida por Google Cloud.

---

## 1. Estructura del proyecto

```
mlops-demo/
├── data/                      # parquets (generados, ignorados por git)
│   ├── raw/                   # dataset original
│   ├── processed/             # features + reference para drift
│   └── production/            # ventana "última ejecución" para drift
├── src/
│   ├── data/load.py           # ingesta y validación
│   ├── features/build.py      # feature engineering + partición ref/cur
│   ├── models/train.py        # entrenamiento + MLflow tracking + registry
│   ├── monitoring/drift.py    # reporte de drift con Evidently
│   └── api/main.py            # FastAPI cargando modelo desde el registry
├── scripts/
│   ├── promote_model.py       # promueve la última versión a Production
│   └── smoke_test_api.py      # prueba humo de la API
├── tests/                     # pytest (data, features, API)
├── figs/                      # capturas para las diapositivas
├── reports/                   # HTML de Evidently
├── Dockerfile.train  Dockerfile.serve
├── docker-compose.yml
├── requirements-train.txt  requirements-serve.txt  requirements-dev.txt
├── params.yaml                # configuración externa del pipeline
└── .github/workflows/ci.yml
```

---

## 2. Requisitos

- Python **3.11**
- (Opcional) Docker y Docker Compose para la ejecución contenedorizada.

---

## 3. Instalación

```bash
git clone <URL-DEL-REPO>
cd mlops-demo

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements-train.txt
pip install -r requirements-dev.txt
```

---

## 4. Ejecución end-to-end (modo local)

Esta es la ruta recomendada para reproducir todo el ejemplo y obtener las capturas que faltan en las diapositivas.

### 4.1 Generar datos

```bash
python -m src.data.load
python -m src.features.build
```

Se crean `data/raw/breast_cancer.parquet`, `data/processed/reference.parquet` y `data/production/last_24h.parquet`. Este último contiene un *shift* artificial sobre tres features para que el reporte de drift encuentre diferencias detectables.

### 4.2 Levantar MLflow Tracking Server

En una terminal aparte:

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000
```

> SQLite es indispensable: el Model Registry no funciona con el backend de archivos por defecto.

UI disponible en `http://localhost:5000`.

### 4.3 Entrenar y registrar el modelo

```bash
python -m src.models.train
```

Salida esperada:

```
[train] run_id=... | AUC=0.99xx
[train] Modelo registrado en 'breast-cancer-rf' (AUC 0.99xx >= 0.95)
```

### 4.4 Promover el modelo a *Production*

```bash
python scripts/promote_model.py
```

### 4.5 Levantar la API

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export MODEL_NAME=breast-cancer-rf
export MODEL_STAGE=Production

uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Probar:

```bash
python scripts/smoke_test_api.py
# o:
curl http://localhost:8000/health
```

### 4.6 Generar el reporte de drift

```bash
python -m src.monitoring.drift
```

Salida esperada:

```
[drift] Reporte guardado en reports/drift_report.html
[drift] Drift detectado: N test(s) en estado FAIL. ...
```

Abrir `reports/drift_report.html` en el navegador.

---

## 5. Ejecución con Docker Compose

```bash
docker compose up --build
```

Esto levanta MLflow en `:5000` y la API en `:8000`. El entrenamiento y el monitoreo se siguen lanzando desde el host (apuntando a `http://localhost:5000`) o como contenedores ad-hoc:

```bash
docker compose run --rm api python -m src.models.train
```

---

## 6. Tests

```bash
pytest -q
```

---

## 7. Capturas necesarias para las diapositivas

Dos diapositivas del PDF esperan capturas que solo existen tras ejecutar el código:

| Diapositiva | Archivo esperado          | Cómo obtenerla                                                                                              |
|-------------|---------------------------|-------------------------------------------------------------------------------------------------------------|
| 23 — *MLflow Tracking UI*       | `figs/mlflow_ui.png`        | Después del paso **4.3**, abrir `http://localhost:5000`, capturar la lista de runs con métricas y parámetros. |
| 27 — *Reporte de drift generado* | `figs/evidently_report.png` | Después del paso **4.6**, abrir `reports/drift_report.html`, capturar la sección con la tabla de features y los histogramas de drift. |

Ambas capturas se colocan en `figs/` con esos nombres exactos. El bloque `\imgorhint` del archivo `slides_mlops.tex` las detecta automáticamente y reemplaza el placeholder por la imagen al recompilar.

---

## 8. Licencia

Material académico. Uso libre con atribución.
