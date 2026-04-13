# -*- coding: utf-8 -*-
"""Servicio de embeddings semánticos basado en Sentence Transformers.

Genera representaciones vectoriales densas de campos textuales clínicos
(diagnósticos CIE-10, códigos CUPS, observaciones) para enriquecer el
vector de features del modelo XGBoost de auditoría prestacional.

Cumplimiento:
    - Toda inferencia es local (sin llamadas a APIs externas).
    - No se usa ``print()`` — se usa ``logging``.
    - Ley 1581 de Colombia: los datos clínicos nunca salen de la infraestructura.
"""

from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_DIM, EMBEDDING_MODEL_NAME, RANDOM_STATE
from src.nlp.pipeline import preprocess_text

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Servicio de generación de embeddings semánticos con Sentence Transformers.

    Carga el modelo una sola vez por instancia (patrón singleton por instancia)
    y expone métodos para codificar texto individual o en lote.

    Attributes:
        model_name: Nombre del modelo Sentence Transformers utilizado.
        random_state: Semilla de aleatoriedad para reproducibilidad.

    Example:
        >>> service = EmbeddingService()
        >>> vec = service.encode("hipertensión arterial esencial")
        >>> vec.shape
        (384,)
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        random_state: int = RANDOM_STATE,
    ) -> None:
        """Inicializa el servicio y carga el modelo Sentence Transformers.

        El modelo se carga una sola vez durante la construcción de la instancia.
        Las llamadas a ``encode`` y ``encode_batch`` reutilizan el modelo ya cargado.

        Args:
            model_name: Nombre o ruta del modelo Sentence Transformers.
                Por defecto usa ``EMBEDDING_MODEL_NAME`` de ``config.py``.
            random_state: Semilla de aleatoriedad para reproducibilidad.
                Por defecto usa ``RANDOM_STATE`` de ``config.py`` (42).
        """
        self.model_name = model_name
        self.random_state = random_state

        logger.info(
            "Cargando modelo de embeddings '%s' (random_state=%d).",
            model_name,
            random_state,
        )
        self._model = SentenceTransformer(model_name)
        logger.info("Modelo de embeddings cargado correctamente.")

    # ------------------------------------------------------------------
    # Métodos públicos
    # ------------------------------------------------------------------

    def zero_vector(self) -> np.ndarray:
        """Retorna un vector de ceros de dimensión ``EMBEDDING_DIM``.

        Utilizado como sustituto cuando el texto de entrada es nulo o vacío,
        garantizando que el pipeline no se interrumpa (Requisito 2.4).

        Returns:
            Array NumPy de forma ``(EMBEDDING_DIM,)`` con todos los valores en 0.0.

        Example:
            >>> service = EmbeddingService()
            >>> zv = service.zero_vector()
            >>> zv.shape == (384,) and (zv == 0).all()
            True
        """
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    def encode(self, text: str | None) -> np.ndarray:
        """Codifica un texto en un vector de embeddings de dimensión fija.

        Aplica ``preprocess_text`` antes de invocar el modelo. Si el texto
        resultante es vacío o el input es ``None``, retorna ``zero_vector()``
        sin invocar el modelo (Requisito 2.4).

        Args:
            text: Texto clínico a codificar. Puede ser ``None`` o cadena vacía.

        Returns:
            Array NumPy de forma ``(EMBEDDING_DIM,)`` con el vector de embeddings.
            Retorna ``zero_vector()`` si el texto es nulo o vacío tras preprocesamiento.

        Example:
            >>> service = EmbeddingService()
            >>> vec = service.encode("hipertensión arterial")
            >>> vec.shape
            (384,)
            >>> service.encode(None).shape
            (384,)
        """
        preprocessed = preprocess_text(text)
        if not preprocessed:
            logger.debug("encode: texto vacío o None tras preprocesamiento; retornando zero_vector.")
            return self.zero_vector()

        logger.debug("encode: codificando texto preprocesado de %d caracteres.", len(preprocessed))
        vector = self._model.encode(preprocessed, convert_to_numpy=True)
        return vector.astype(np.float32)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Codifica una lista de textos en una matriz de embeddings.

        Aplica ``preprocess_text`` a cada texto. Los textos que resulten vacíos
        tras el preprocesamiento son sustituidos por ``zero_vector()``.

        Args:
            texts: Lista de textos clínicos a codificar.

        Returns:
            Array NumPy de forma ``(len(texts), EMBEDDING_DIM)`` con los vectores
            de embeddings. Las filas correspondientes a textos nulos o vacíos
            contienen ceros.

        Example:
            >>> service = EmbeddingService()
            >>> matrix = service.encode_batch(["hipertensión", "diabetes", None])
            >>> matrix.shape
            (3, 384)
        """
        if not texts:
            logger.debug("encode_batch: lista vacía; retornando array de forma (0, %d).", EMBEDDING_DIM)
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

        preprocessed = [preprocess_text(t) for t in texts]

        # Separar índices con texto válido de los que requieren zero_vector
        valid_indices = [i for i, t in enumerate(preprocessed) if t]
        valid_texts = [preprocessed[i] for i in valid_indices]

        result = np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)

        if valid_texts:
            logger.debug(
                "encode_batch: codificando %d/%d textos válidos.",
                len(valid_texts),
                len(texts),
            )
            embeddings = self._model.encode(valid_texts, convert_to_numpy=True, batch_size=256, show_progress_bar=False)
            for out_idx, emb in zip(valid_indices, embeddings):
                result[out_idx] = emb.astype(np.float32)

        logger.debug("encode_batch: completado, forma resultante %s.", result.shape)
        return result
