# -*- coding: utf-8 -*-
"""Módulo NLP para preprocesamiento de texto clínico en español.

Expone el pipeline spaCy y la función de preprocesamiento para uso
en el pipeline de embeddings semánticos.
"""

from src.nlp.pipeline import (
    MEDICAL_STOPWORDS,
    load_nlp_pipeline,
    preprocess_text,
)

__all__ = [
    "MEDICAL_STOPWORDS",
    "load_nlp_pipeline",
    "preprocess_text",
]
