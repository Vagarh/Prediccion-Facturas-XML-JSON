#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Script de reentrenamiento mensual para ejecución programada (cron).

Uso:
    python scripts/monthly_retrain.py [--months-back N] [--force]

Cron job (primer día de cada mes a las 02:00):
    0 2 1 * * /path/to/venv/bin/python /path/to/scripts/monthly_retrain.py

Salida:
    - Logs en stdout/stderr
    - Reporte JSON en Codigo/reports/retrain_{DATE}.json
    - Modelo serializado en Codigo/models/artifacts/ (si reentrenamiento exitoso)
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Asegurar que src/ esté en el path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import REPORTS_DIR, setup_logging
from src.monitoring.model_monitor import ModelMonitor
from src.training.retraining_scheduler import RetrainingScheduler

setup_logging()
logger = logging.getLogger("monthly_retrain")


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Reentrenamiento mensual automático")
    parser.add_argument(
        "--months-back",
        type=int,
        default=3,
        help="Meses de datos históricos a usar (default: 3)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Forzar reentrenamiento sin verificar drift",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo evaluar drift sin reentrenar",
    )
    return parser.parse_args()


def save_report(result: dict, date_prefix: str) -> Path:
    """Serializa el resultado del reentrenamiento en JSON.

    Args:
        result: Diccionario con resultado del ciclo.
        date_prefix: Prefijo de fecha para el nombre del archivo.

    Returns:
        Ruta al archivo de reporte generado.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"retrain_{date_prefix}.json"

    # Convertir objetos no serializables
    def _serialize(obj):
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        if hasattr(obj, "__str__"):
            return str(obj)
        return repr(obj)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=_serialize, ensure_ascii=False)

    logger.info("Reporte guardado: %s", report_path)
    return report_path


def main() -> int:
    """Punto de entrada principal del script.

    Returns:
        Código de salida: 0 = éxito, 1 = error.
    """
    args = parse_args()
    date_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=== Reentrenamiento mensual iniciado: %s ===", date_prefix)
    logger.info("Parámetros: months_back=%d, force=%s, dry_run=%s",
                args.months_back, args.force, args.dry_run)

    try:
        monitor = ModelMonitor()
        scheduler = RetrainingScheduler(
            monitor=monitor,
            months_back=args.months_back,
        )

        if args.dry_run:
            # Solo evaluar drift y rendimiento
            logger.info("Modo dry-run: solo evaluación sin reentrenamiento")
            perf_report = monitor.generate_monthly_report()
            result = {
                "mode": "dry_run",
                "performance_report": perf_report,
                "timestamp": datetime.utcnow().isoformat(),
            }
        elif args.force:
            # Reentrenar sin verificar drift
            logger.info("Modo force: reentrenamiento directo")
            retrain_result = scheduler.retrain_with_recent_data(args.months_back)
            result = {
                "mode": "force",
                "retrain_result": retrain_result,
                "timestamp": datetime.utcnow().isoformat(),
            }
        else:
            # Ciclo completo: drift + reentrenamiento si necesario
            result = scheduler.schedule_monthly_retrain()
            result["timestamp"] = datetime.utcnow().isoformat()

        # Guardar reporte
        report_path = save_report(result, date_prefix)

        # Resumen en consola
        if result.get("retrain_triggered"):
            retrain = result.get("retrain_result", {})
            if retrain.get("success"):
                logger.info(
                    "✓ Reentrenamiento exitoso: versión=%s, Recall_Glosada=%.4f",
                    retrain.get("new_version", "N/A"),
                    retrain.get("metrics", {}).get("recall_glosada", 0),
                )
            else:
                logger.warning("✗ Reentrenamiento fallido: %s", retrain.get("reason", "desconocido"))
        else:
            logger.info("✓ No se requirió reentrenamiento (drift_status=%s)",
                       result.get("drift_status", "N/A"))

        logger.info("=== Ciclo completado. Reporte: %s ===", report_path)
        return 0

    except Exception as e:
        logger.exception("Error crítico en reentrenamiento mensual: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
