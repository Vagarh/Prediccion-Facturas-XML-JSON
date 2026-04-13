# -*- coding: utf-8 -*-
"""Registro de versiones de modelos con SQLite.

Almacena metadatos, métricas y rutas de artefactos para cada versión
del modelo XGBoost, permitiendo rollback y comparación de versiones.
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import MODELS_DIR, RANDOM_STATE

logger = logging.getLogger(__name__)

DB_PATH = MODELS_DIR / "model_registry.db"

# DDL del esquema
_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS model_versions (
    model_id    TEXT PRIMARY KEY,
    version     TEXT NOT NULL,
    metrics     TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    is_active   INTEGER NOT NULL DEFAULT 0,
    notes       TEXT
);
"""


# ---------------------------------------------------------------------------
# Helpers de conexión
# ---------------------------------------------------------------------------


def _get_conn(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Retorna conexión SQLite con row_factory configurado.

    Args:
        db_path: Ruta al archivo de base de datos.

    Returns:
        Conexión SQLite lista para usar.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute(_CREATE_TABLE)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------


def save_model_version(
    model_id: str,
    version: str,
    metrics: dict[str, float],
    artifact_path: str | Path,
    notes: str = "",
    db_path: Path = DB_PATH,
) -> None:
    """Registra una nueva versión del modelo en el registro.

    Args:
        model_id: Identificador único (e.g., "20260410_xgboost_v2_enriched").
        version: Etiqueta de versión semántica (e.g., "v2.1").
        metrics: Diccionario con métricas de evaluación.
        artifact_path: Ruta al archivo del modelo serializado.
        notes: Notas opcionales sobre el entrenamiento.
        db_path: Ruta a la base de datos SQLite.
    """
    with _get_conn(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO model_versions
                (model_id, version, metrics, artifact_path, created_at, is_active, notes)
            VALUES (?, ?, ?, ?, ?, 0, ?)
            """,
            (
                model_id,
                version,
                json.dumps(metrics),
                str(artifact_path),
                datetime.utcnow().isoformat(),
                notes,
            ),
        )
        conn.commit()
    logger.info("Versión registrada: %s (%s)", model_id, version)


def load_model_version(version: str, db_path: Path = DB_PATH) -> dict[str, Any]:
    """Carga los metadatos de una versión específica del modelo.

    Args:
        version: Etiqueta de versión a cargar.
        db_path: Ruta a la base de datos SQLite.

    Returns:
        Diccionario con model_id, version, metrics, artifact_path, created_at, is_active.

    Raises:
        KeyError: Si la versión no existe en el registro.
    """
    with _get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM model_versions WHERE version = ? ORDER BY created_at DESC LIMIT 1",
            (version,),
        ).fetchone()

    if row is None:
        raise KeyError(f"Versión '{version}' no encontrada en el registro")

    return {
        "model_id": row["model_id"],
        "version": row["version"],
        "metrics": json.loads(row["metrics"]),
        "artifact_path": row["artifact_path"],
        "created_at": row["created_at"],
        "is_active": bool(row["is_active"]),
        "notes": row["notes"],
    }


def list_model_versions(db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    """Retorna el histórico completo de versiones ordenado por fecha descendente.

    Args:
        db_path: Ruta a la base de datos SQLite.

    Returns:
        Lista de diccionarios con metadatos de cada versión.
    """
    with _get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM model_versions ORDER BY created_at DESC"
        ).fetchall()

    return [
        {
            "model_id": r["model_id"],
            "version": r["version"],
            "metrics": json.loads(r["metrics"]),
            "artifact_path": r["artifact_path"],
            "created_at": r["created_at"],
            "is_active": bool(r["is_active"]),
            "notes": r["notes"],
        }
        for r in rows
    ]


def set_active_version(version: str, db_path: Path = DB_PATH) -> None:
    """Marca una versión como activa y desactiva las demás.

    Args:
        version: Etiqueta de versión a activar.
        db_path: Ruta a la base de datos SQLite.

    Raises:
        KeyError: Si la versión no existe.
    """
    with _get_conn(db_path) as conn:
        updated = conn.execute(
            "UPDATE model_versions SET is_active = CASE WHEN version = ? THEN 1 ELSE 0 END",
            (version,),
        ).rowcount
        conn.commit()

    if updated == 0:
        raise KeyError(f"Versión '{version}' no encontrada")
    logger.info("Versión activa establecida: %s", version)


def get_active_version(db_path: Path = DB_PATH) -> dict[str, Any] | None:
    """Retorna los metadatos de la versión activa actual.

    Args:
        db_path: Ruta a la base de datos SQLite.

    Returns:
        Diccionario con metadatos o None si no hay versión activa.
    """
    with _get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM model_versions WHERE is_active = 1 LIMIT 1"
        ).fetchone()

    if row is None:
        return None

    return {
        "model_id": row["model_id"],
        "version": row["version"],
        "metrics": json.loads(row["metrics"]),
        "artifact_path": row["artifact_path"],
        "created_at": row["created_at"],
        "is_active": True,
        "notes": row["notes"],
    }
