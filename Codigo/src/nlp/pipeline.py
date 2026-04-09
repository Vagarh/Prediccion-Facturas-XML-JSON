# -*- coding: utf-8 -*-
"""Pipeline NLP en español para preprocesamiento de texto clínico.

Proporciona carga singleton del modelo spaCy ``es_core_news_md`` y una función
de preprocesamiento que aplica tokenización, minúsculas, eliminación de
stopwords médicos y lematización sobre texto clínico en español (diagnósticos
CIE-10, códigos CUPS, observaciones).

Cumplimiento:
    - Toda inferencia es local (sin llamadas a APIs externas).
    - No se usa ``print()`` — se usa ``logging``.
    - Ley 1581 de Colombia: los datos clínicos nunca salen de la infraestructura.
"""

from __future__ import annotations

import logging
from typing import Optional

import spacy
from spacy.language import Language

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stopwords médicos irrelevantes para embeddings clínicos
# ---------------------------------------------------------------------------

MEDICAL_STOPWORDS: list[str] = [
    "paciente",
    "presenta",
    "refiere",
    "diagnóstico",
    "tratamiento",
    "médico",
    "clínico",
    "servicio",
    "atención",
    "consulta",
    "procedimiento",
    "resultado",
    "informe",
    "reporte",
    "historia",
    "clínica",
    "nota",
    "evolución",
    "seguimiento",
    "control",
]

# ---------------------------------------------------------------------------
# Singleton del pipeline spaCy
# ---------------------------------------------------------------------------

_nlp_instance: Optional[Language] = None


def load_nlp_pipeline() -> Language:
    """Carga el modelo spaCy ``es_core_news_md`` con caché singleton.

    El modelo se carga una sola vez por proceso. Las llamadas subsiguientes
    retornan la instancia ya cargada sin incurrir en el costo de carga.

    Returns:
        Instancia del modelo spaCy en español ``es_core_news_md``.

    Raises:
        OSError: Si el modelo ``es_core_news_md`` no está instalado.
            Instalar con: ``python -m spacy download es_core_news_md``
    """
    global _nlp_instance
    if _nlp_instance is None:
        logger.info("Cargando modelo spaCy 'es_core_news_md' (primera vez en este proceso).")
        _nlp_instance = spacy.load("es_core_news_md")
        logger.info("Modelo spaCy cargado correctamente.")
    return _nlp_instance


# ---------------------------------------------------------------------------
# Preprocesamiento de texto clínico
# ---------------------------------------------------------------------------

def preprocess_text(text: str | None) -> str:
    """Preprocesa texto clínico en español para generación de embeddings.

    Aplica en orden: tokenización, conversión a minúsculas, eliminación de
    stopwords médicos (``MEDICAL_STOPWORDS`` + stopwords spaCy), lematización
    y eliminación de tokens no alfabéticos.

    Args:
        text: Texto clínico a preprocesar. Puede ser ``None`` o cadena vacía.

    Returns:
        Texto preprocesado como cadena de lemas separados por espacios.
        Retorna ``""`` si el input es ``None``, vacío o resulta en tokens vacíos
        tras el preprocesamiento.

    Example:
        >>> result = preprocess_text("El paciente presenta diagnóstico de hipertensión arterial")
        >>> "hipertension" in result or "hipertensión" in result or "arterial" in result
        True
    """
    if not text or not text.strip():
        logger.debug("preprocess_text recibió input vacío o None; retornando ''.")
        return ""

    nlp = load_nlp_pipeline()
    medical_stopwords_set = set(MEDICAL_STOPWORDS)

    doc = nlp(text.lower())

    tokens = [
        token.lemma_
        for token in doc
        if token.is_alpha
        and not token.is_stop
        and token.lemma_ not in medical_stopwords_set
        and token.text not in medical_stopwords_set
    ]

    result = " ".join(tokens)
    logger.debug("preprocess_text: '%s' → '%s'", text[:50], result[:50])
    return result
