# -*- coding: utf-8 -*-
"""Privacy Guard — garantiza que toda inferencia LLM permanece en infraestructura local.

Cumplimiento estricto con la Ley 1581 de Colombia: ningún dato sensible de salud
puede salir de los servidores de Savia Salud EPS.
"""

import hashlib
import logging
from urllib.parse import urlparse

import requests

from src.config import OLLAMA_BASE_URL, OLLAMA_TIMEOUT_S

logger = logging.getLogger(__name__)

# Hosts considerados locales
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class PrivacyViolationError(Exception):
    """Excepción lanzada cuando se detecta un intento de transmisión de datos a un host externo.

    Raised by:
        PrivacyGuard.verify_local_only: cuando OLLAMA_BASE_URL apunta a un host no local.
    """


class PrivacyGuard:
    """Capa de validación que garantiza que toda inferencia LLM opera en local.

    Verifica que el servidor Ollama esté configurado en localhost, comprueba su
    disponibilidad y registra cada operación de inferencia sin almacenar contenido
    textual de datos clínicos.

    Attributes:
        ollama_base_url: URL base del servidor Ollama a validar.

    Example:
        >>> guard = PrivacyGuard()
        >>> guard.verify_local_only()   # no lanza si apunta a localhost
        >>> guard.check_ollama_available()
        True
    """

    def __init__(self, ollama_base_url: str = OLLAMA_BASE_URL) -> None:
        """Inicializa el PrivacyGuard con la URL del servidor Ollama.

        Args:
            ollama_base_url: URL base del servidor Ollama. Por defecto usa
                ``OLLAMA_BASE_URL`` de ``config.py``.
        """
        self.ollama_base_url = ollama_base_url

    def verify_local_only(self) -> None:
        """Verifica que ``ollama_base_url`` apunta exclusivamente a localhost.

        Comprueba que el hostname de la URL sea ``localhost``, ``127.0.0.1`` o
        ``::1``. Si detecta cualquier otro host, lanza ``PrivacyViolationError``
        para prevenir la transmisión de datos clínicos a servicios externos.

        Raises:
            PrivacyViolationError: Si el hostname no corresponde a un host local.

        Example:
            >>> guard = PrivacyGuard("http://localhost:11434")
            >>> guard.verify_local_only()  # OK, no lanza

            >>> guard_ext = PrivacyGuard("http://api.openai.com")
            >>> guard_ext.verify_local_only()
            Traceback (most recent call last):
                ...
            PrivacyViolationError: ...
        """
        parsed = urlparse(self.ollama_base_url)
        host = parsed.hostname or ""

        if host not in _LOCAL_HOSTS:
            msg = (
                f"Violación de privacidad (Ley 1581): OLLAMA_BASE_URL apunta a host "
                f"externo '{host}'. Solo se permiten hosts locales: {_LOCAL_HOSTS}. "
                f"URL configurada: '{self.ollama_base_url}'"
            )
            logger.error(msg)
            raise PrivacyViolationError(msg)

        logger.debug("verify_local_only: host '%s' es local — OK", host)

    def check_ollama_available(self) -> bool:
        """Comprueba si el servidor Ollama está disponible respondiendo en /api/tags.

        Realiza un GET a ``{ollama_base_url}/api/tags`` con un timeout corto (2 s)
        para no bloquear el arranque de la API. Retorna ``False`` ante cualquier
        error de red o respuesta no exitosa.

        Returns:
            ``True`` si el servidor responde con HTTP 200, ``False`` en caso contrario.

        Example:
            >>> guard = PrivacyGuard()
            >>> available = guard.check_ollama_available()
            >>> isinstance(available, bool)
            True
        """
        url = f"{self.ollama_base_url}/api/tags"
        try:
            response = requests.get(url, timeout=2.0)
            available = response.status_code == 200
            if available:
                logger.debug("check_ollama_available: servidor disponible en '%s'", url)
            else:
                logger.warning(
                    "check_ollama_available: servidor respondió HTTP %d en '%s'",
                    response.status_code,
                    url,
                )
            return available
        except requests.exceptions.RequestException as exc:
            logger.warning(
                "check_ollama_available: servidor no disponible en '%s' — %s",
                url,
                exc,
            )
            return False

    def log_inference(self, model: str, input_hash: str) -> None:
        """Registra una operación de inferencia LLM en el logger local.

        Escribe únicamente metadatos de trazabilidad (modelo e hash del input)
        sin almacenar el contenido textual de los datos clínicos, cumpliendo
        con la Ley 1581 de Colombia.

        Args:
            model: Nombre del modelo LLM utilizado (ej. ``"llama3.2"``).
            input_hash: Hash SHA-256 del input de inferencia. Se usa como
                identificador de trazabilidad sin exponer el contenido clínico.

        Example:
            >>> guard = PrivacyGuard()
            >>> guard.log_inference("llama3.2", "abc123def456")
        """
        logger.info(
            "LLM inference | model='%s' | input_hash='%s'",
            model,
            input_hash,
        )
