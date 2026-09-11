# -*- coding: utf-8 -*-
"""Helper de conexión a la base de datos MySQL de Savia Salud EPS (solo lectura).

Las credenciales se leen **exclusivamente** de variables de entorno (o de un
archivo ``Codigo/.env`` no versionado). Nunca se hardcodean en el código.

Variables requeridas:
    SAVIA_DB_HOST, SAVIA_DB_USER, SAVIA_DB_PASSWORD, SAVIA_DB_NAME
Variable opcional:
    SAVIA_DB_PORT (default: 3306)

Regla del proyecto (CLAUDE.md §2): la BD es de **solo lectura**. Este módulo no
expone ningún helper de escritura; todo consumo debe ser ``SELECT``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

_CODIGO_DIR = Path(__file__).resolve().parent.parent
_ENV_PATH = _CODIGO_DIR / ".env"

_REQUIRED_VARS = ("SAVIA_DB_HOST", "SAVIA_DB_USER", "SAVIA_DB_PASSWORD", "SAVIA_DB_NAME")


def _load_dotenv() -> None:
    """Carga ``Codigo/.env`` en ``os.environ`` si ``python-dotenv`` está disponible.

    Si la librería no está instalada o el archivo no existe, no hace nada: se
    asume que las variables ya están definidas en el entorno.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dependencia opcional
        logger.debug("python-dotenv no instalado; se usan variables de entorno del sistema.")
        return
    if _ENV_PATH.exists():
        load_dotenv(_ENV_PATH)
        logger.debug("Variables cargadas desde %s", _ENV_PATH)


def get_engine(connect_timeout: int = 60) -> Engine:
    """Crea un ``Engine`` de SQLAlchemy hacia la BD de Savia.

    Args:
        connect_timeout: Segundos de espera para establecer la conexión TCP.

    Returns:
        Engine de SQLAlchemy con ``pool_pre_ping`` activado.

    Raises:
        RuntimeError: Si falta alguna credencial obligatoria.
    """
    _load_dotenv()

    missing = [v for v in _REQUIRED_VARS if not os.environ.get(v)]
    if missing:
        raise RuntimeError(
            "Faltan credenciales de BD: "
            + ", ".join(missing)
            + f". Copia {_CODIGO_DIR / '.env.example'} a {_ENV_PATH} y complétalo."
        )

    host = os.environ["SAVIA_DB_HOST"]
    user = os.environ["SAVIA_DB_USER"]
    password = os.environ["SAVIA_DB_PASSWORD"]
    name = os.environ["SAVIA_DB_NAME"]
    port = os.environ.get("SAVIA_DB_PORT", "3306")

    url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{name}"
    engine = create_engine(
        url,
        connect_args={"connect_timeout": connect_timeout},
        pool_pre_ping=True,
        pool_recycle=3600,
    )
    logger.info("Engine creado hacia %s:%s/%s (usuario=%s)", host, port, name, user)
    return engine


def check_connection(engine: Engine) -> bool:
    """Ejecuta ``SELECT 1`` para verificar que la conexión está viva.

    Args:
        engine: Engine de SQLAlchemy.

    Returns:
        True si la consulta de prueba tuvo éxito, False en caso contrario.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Conexión a BD verificada.")
        return True
    except Exception as exc:  # noqa: BLE001 - se registra y se degrada a False
        logger.error("Fallo de conexión a BD: %s", exc)
        return False
