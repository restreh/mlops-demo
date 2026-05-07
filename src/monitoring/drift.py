"""
Monitoreo de data drift con Evidently.

Este script reproduce el snippet de la diapositiva "Monitoreo de drift
con Evidently". Compara dos datasets:
- reference.parquet: distribución observada en entrenamiento.
- last_24h.parquet:  distribución observada en producción reciente.

Genera un reporte HTML interactivo y, si algún test estadístico falla,
sale con código distinto de cero. En CI esto puede dispararse como
una etapa que, al fallar, lanza el pipeline de reentrenamiento.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset


def main() -> int:
    # 1. Configuración
    cfg = yaml.safe_load(Path("params.yaml").read_text())
    mon_cfg = cfg["monitoring"]
    data_cfg = cfg["data"]

    reference_path = Path(data_cfg["reference_path"])
    current_path = Path(data_cfg["current_path"])
    report_path = Path(mon_cfg["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # 2. Datos
    if not reference_path.exists() or not current_path.exists():
        print(
            "[drift] Faltan los archivos reference/current. "
            "Ejecuta primero: python -m src.features.build",
            file=sys.stderr,
        )
        return 2

    reference = pd.read_parquet(reference_path)
    current = pd.read_parquet(current_path)

    # 3. Reporte. DataDriftPreset analiza cada feature y aplica el test
    #    estadístico indicado (PSI por defecto). DataSummaryPreset agrega
    #    estadísticas descriptivas comparadas. include_tests=True habilita
    #    los tests pass/fail que después se inspeccionan.
    report = Report(
        metrics=[
            DataDriftPreset(method=mon_cfg["method"]),
            DataSummaryPreset(),
        ],
        include_tests=True,
    )
    snapshot = report.run(reference_data=reference, current_data=current)
    snapshot.save_html(str(report_path))
    print(f"[drift] Reporte guardado en {report_path}")

    # 4. Decisión automática.
    #    snapshot.dict() expone los resultados como un dict con dos llaves
    #    de alto nivel: 'metrics' (valores numéricos) y 'tests' (resultados
    #    pass/fail). Cada test tiene un campo 'status' cuyo valor es un
    #    enum TestStatus; lo normalizamos a string para la comparación.
    summary = snapshot.dict()
    failed = [
        t for t in summary.get("tests", [])
        if str(getattr(t.get("status"), "value", t.get("status"))) == "FAIL"
    ]
    total = len(summary.get("tests", []))
    print(f"[drift] Tests evaluados: {total} | en estado FAIL: {len(failed)}")

    if failed:
        # Mostrar los tres primeros para que el usuario entienda qué cambió
        for t in failed[:3]:
            print(f"  · {t.get('name')} → {t.get('description')}")
        print(
            "[drift] Drift detectado. En un pipeline real este código de "
            "salida dispara el reentrenamiento."
        )
        return 1

    print("[drift] No se detectó drift relevante.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
