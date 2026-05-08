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

### Linux / macOS (bash)

```bash
git clone <URL-DEL-REPO>
cd mlops-demo

python -m venv .venv
source .venv/bin/activate

pip install -r requirements-train.txt
pip install -r requirements-serve.txt
pip install -r requirements-dev.txt
```

### Windows (PowerShell)

```powershell
git clone <URL-DEL-REPO>
cd mlops-demo

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements-train.txt
pip install -r requirements-serve.txt
pip install -r requirements-dev.txt
```

> Si la activación falla con un error de *execution policy*, ejecutar una vez en la sesión:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

---

## 4. Ejecución end-to-end (modo local)

Esta es la ruta recomendada para reproducir todo el ejemplo y obtener las capturas que faltan en las diapositivas. Los pasos son los mismos en todos los sistemas operativos; solo cambia la sintaxis para activar la venv y para definir variables de entorno (paso 4.5).

### 4.1 Generar datos

```bash
python -m src.data.load
python -m src.features.build
```

Se crean `data/raw/breast_cancer.parquet`, `data/processed/reference.parquet` y `data/production/last_24h.parquet`. Este último contiene un *shift* artificial sobre tres features para que el reporte de drift encuentre diferencias detectables.

### 4.2 Levantar MLflow Tracking Server

En una terminal aparte, dentro de la misma carpeta y con la venv activada:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
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

Las variables de entorno se definen distinto según el shell:

**Linux / macOS (bash)**

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export MODEL_NAME=breast-cancer-rf
export MODEL_STAGE=Production

uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

**Windows (PowerShell)**

```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
$env:MODEL_NAME = "breast-cancer-rf"
$env:MODEL_STAGE = "Production"

uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Probar en otra terminal:

```bash
python scripts/smoke_test_api.py
```

O con `curl`:

```bash
curl http://localhost:8000/health
```

### 4.6 Generar el reporte de drift

```bash
python -m src.monitoring.drift
```

Salida esperada:

```
[drift] Reporte guardado en reports/drift_report.html
[drift] Tests evaluados: 313 | en estado FAIL: 133
[drift] Drift detectado. ...
```

Abrir `reports/drift_report.html` en el navegador.

---

## 5. Ejecución con Docker Compose

Independiente del sistema operativo:

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

## 7. Licencia

Material académico. Uso libre con atribución.
