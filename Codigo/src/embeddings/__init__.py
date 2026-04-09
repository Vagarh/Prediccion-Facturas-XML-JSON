# -*- coding: utf-8 -*-
"""Paquete de embeddings semánticos para el pipeline de auditoría prestacional.

Expone el servicio de generación de embeddings basado en Sentence Transformers,
ejecutado exclusivamente en infraestructura local (Ley 1581 de Colombia).
"""

from src.embeddings.service import EmbeddingService

__all__ = ["EmbeddingService"]
