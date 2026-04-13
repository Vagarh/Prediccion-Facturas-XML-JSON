# -*- coding: utf-8 -*-
"""Generador de explicaciones en lenguaje natural para auditoría prestacional.

Produce narrativas contextualizadas con la Resolución 2284/2023 usando Ollama,
con degradación graciosa a explicaciones SHAP técnicas cuando el LLM no está disponible.

Cumplimiento:
    - Explicaciones en español para auditores médicos
    - Evita términos técnicos de ML ("SHAP", "embedding", "feature importance")
    - Incluye causales específicos de Res. 2284/2023
    - Degradación graciosa sin Ollama
    - Timeout configurable para generación
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import requests
from requests.exceptions import RequestException, Timeout

from src.config import (
    OLLAMA_BASE_URL,
    EXPLANATION_MAX_SECONDS,
    RANDOM_STATE,
)
from src.privacy.guard import PrivacyGuard

logger = logging.getLogger(__name__)

# Mapeo de features importantes a causales de Resolución 2284/2023
FEATURE_TO_CAUSAL = {
    # Causales de Suficiencia (SO)
    "valor_facturado": "SO0101 - Facturación por servicios no prestados",
    "cantidad_servicios": "SO0201 - Facturación de cantidades superiores a las efectivamente prestadas",
    "dias_estancia": "SO0201 - Facturación de cantidades superiores a las efectivamente prestadas",
    
    # Causales de Facturación (FA)
    "codigo_cups": "FA0101 - Facturación de servicios no incluidos en el plan de beneficios",
    "modalidad_atencion": "FA0301 - Facturación de servicios prestados en modalidad diferente a la contratada",
    "nivel_atencion": "FA0301 - Facturación de servicios prestados en modalidad diferente a la contratada",
    
    # Causales de Tarifas (TA)
    "valor_unitario": "TA0101 - Facturación con tarifas superiores a las pactadas",
    "valor_copago": "TA0101 - Facturación con tarifas superiores a las pactadas",
    
    # Causales de Documentación (DE)
    "numero_autorizacion": "DE16 - Falta de autorización cuando esta es requerida",
    "fecha_autorizacion": "DE16 - Falta de autorización cuando esta es requerida",
    "tipo_documento": "DE01 - Ausencia de la factura o documento equivalente",
    "numero_factura": "DE01 - Ausencia de la factura o documento equivalente",
    
    # Causales por diagnóstico y procedimientos
    "diagnostico_principal": "SO0101 - Facturación por servicios no prestados",
    "diagnostico_relacionado": "SO0101 - Facturación por servicios no prestados",
    "procedimiento_principal": "FA0101 - Facturación de servicios no incluidos en el plan de beneficios",
    
    # Causales por fechas y tiempos
    "fecha_ingreso": "DE16 - Falta de autorización cuando esta es requerida",
    "fecha_egreso": "SO0201 - Facturación de cantidades superiores a las efectivamente prestadas",
    "tiempo_atencion": "SO0201 - Facturación de cantidades superiores a las efectivamente prestadas",
}

# Plantillas de explicación por clase
EXPLANATION_TEMPLATES = {
    0: {  # Auditada (Aprobada)
        "intro": "La factura fue AUDITADA y aprobada para pago.",
        "factors": "Los principales factores que respaldan esta decisión son:",
        "conclusion": "La documentación y servicios facturados cumplen con los requisitos establecidos."
    },
    1: {  # Glosada
        "intro": "La factura presenta GLOSAS que requieren aclaración o corrección.",
        "factors": "Los principales factores de riesgo identificados son:",
        "conclusion": "Se requiere respuesta del prestador para resolver las inconsistencias encontradas."
    },
    2: {  # Devuelta
        "intro": "La factura fue DEVUELTA por incumplimiento de requisitos mínimos.",
        "factors": "Los principales motivos de devolución son:",
        "conclusion": "La factura debe ser corregida y presentada nuevamente con la documentación completa."
    }
}


class ExplanationGenerator:
    """Generador de explicaciones en lenguaje natural para predicciones de auditoría."""

    def __init__(
        self,
        privacy_guard: PrivacyGuard,
        llm_model: str = "llama3.2",
        timeout_s: float = EXPLANATION_MAX_SECONDS,
    ) -> None:
        """Inicializa el generador de explicaciones.

        Args:
            privacy_guard: Instancia de PrivacyGuard para verificar cumplimiento.
            llm_model: Nombre del modelo LLM a usar en Ollama.
            timeout_s: Timeout máximo para generación de explicaciones.
        """
        self.privacy_guard = privacy_guard
        self.llm_model = llm_model
        self.timeout_s = timeout_s
        
        # Verificar que Ollama esté disponible y sea local
        self.ollama_available = self._check_ollama_availability()
        
        logger.info(
            "ExplanationGenerator inicializado: modelo=%s, timeout=%.1fs, ollama_disponible=%s",
            llm_model, timeout_s, self.ollama_available
        )

    def _check_ollama_availability(self) -> bool:
        """Verifica si Ollama está disponible y cumple requisitos de privacidad.

        Returns:
            True si Ollama está disponible y es local, False en caso contrario.
        """
        try:
            # Verificar cumplimiento de privacidad
            self.privacy_guard.verify_local_only()
            
            # Verificar disponibilidad del servidor
            return self.privacy_guard.check_ollama_available()
            
        except Exception as e:
            logger.warning("Ollama no disponible: %s", e)
            return False

    def generate(
        self,
        prediction: int,
        confidence: float,
        shap_values: dict[str, float],
        feature_values: dict[str, Any],
    ) -> str:
        """Genera explicación en lenguaje natural para una predicción.

        Args:
            prediction: Clase predicha (0=Auditada, 1=Glosada, 2=Devuelta).
            confidence: Confianza de la predicción (0.0 a 1.0).
            shap_values: Valores SHAP por feature (feature_name -> shap_value).
            feature_values: Valores reales de las features (feature_name -> value).

        Returns:
            Explicación en lenguaje natural en español.
        """
        start_time = time.time()
        
        try:
            if self.ollama_available:
                # Intentar generar explicación enriquecida con LLM
                explanation = self._generate_llm_explanation(
                    prediction, confidence, shap_values, feature_values
                )
                
                # Verificar timeout
                elapsed_time = time.time() - start_time
                if elapsed_time > self.timeout_s:
                    logger.warning(
                        "Generación LLM excedió timeout (%.2fs > %.2fs), usando fallback",
                        elapsed_time, self.timeout_s
                    )
                    return self._generate_fallback_explanation(
                        prediction, confidence, shap_values, feature_values
                    )
                
                # Log de inferencia (sin contenido textual)
                self.privacy_guard.log_inference(
                    model=self.llm_model,
                    input_hash=str(hash((prediction, confidence, tuple(shap_values.items()))))
                )
                
                return explanation
                
            else:
                # Usar explicación SHAP técnica como fallback
                return self._generate_fallback_explanation(
                    prediction, confidence, shap_values, feature_values
                )
                
        except Exception as e:
            logger.error("Error generando explicación LLM: %s", e)
            return self._generate_fallback_explanation(
                prediction, confidence, shap_values, feature_values
            )

    def _generate_llm_explanation(
        self,
        prediction: int,
        confidence: float,
        shap_values: dict[str, float],
        feature_values: dict[str, Any],
    ) -> str:
        """Genera explicación usando Ollama LLM.

        Args:
            prediction: Clase predicha.
            confidence: Confianza de la predicción.
            shap_values: Valores SHAP por feature.
            feature_values: Valores reales de las features.

        Returns:
            Explicación generada por LLM.

        Raises:
            RequestException: Si hay error en la comunicación con Ollama.
            Timeout: Si la generación excede el timeout.
        """
        # Obtener top-3 factores más importantes
        top_factors = self._get_top_factors(shap_values, feature_values, n=3)
        
        # Construir prompt contextualizado
        prompt = self._build_llm_prompt(prediction, confidence, top_factors)
        
        # Llamar a Ollama
        response = self._call_ollama(prompt)
        
        # Procesar respuesta
        explanation = self._process_llm_response(response, prediction, confidence, top_factors)
        
        return explanation

    def _build_llm_prompt(
        self,
        prediction: int,
        confidence: float,
        top_factors: list[tuple[str, float, Any]],
    ) -> str:
        """Construye prompt contextualizado para el LLM.

        Args:
            prediction: Clase predicha.
            confidence: Confianza de la predicción.
            top_factors: Lista de (feature_name, shap_value, feature_value).

        Returns:
            Prompt en español para el LLM.
        """
        class_names = ["AUDITADA", "GLOSADA", "DEVUELTA"]
        class_name = class_names[prediction]
        
        # Contexto del sistema
        system_context = """Eres un asistente especializado en auditoría prestacional de servicios de salud en Colombia.
Tu tarea es explicar de manera clara y profesional los resultados de auditoría a personal médico y administrativo.

INSTRUCCIONES:
- Usa lenguaje técnico médico apropiado, sin terminología de sistemas computacionales
- Incluye referencias específicas a la Resolución 2284/2023 cuando sea relevante
- Mantén un tono profesional y objetivo
- Enfócate en los aspectos clínicos y administrativos más relevantes
- No menciones "SHAP", "embedding", "feature importance" ni términos similares de análisis estadístico"""

        # Información de la predicción
        prediction_info = f"""
RESULTADO DE AUDITORÍA: {class_name}
Nivel de confianza: {confidence:.1%}

FACTORES PRINCIPALES IDENTIFICADOS:"""

        # Detalles de factores
        factors_detail = ""
        for i, (feature_name, shap_value, feature_value) in enumerate(top_factors, 1):
            # Obtener causal asociado si existe
            causal = FEATURE_TO_CAUSAL.get(feature_name, "")
            causal_text = f" (Relacionado con: {causal})" if causal else ""
            
            factors_detail += f"""
{i}. {feature_name}: {feature_value}{causal_text}
   Impacto en la decisión: {'Alto' if abs(shap_value) > 0.1 else 'Moderado'}"""

        # Solicitud específica
        request = f"""
Genera una explicación profesional en español que:
1. Explique por qué la factura fue clasificada como {class_name}
2. Describa los factores más relevantes en términos médicos/administrativos
3. Incluya recomendaciones específicas para el prestador
4. Mantenga un tono profesional apropiado para auditores médicos

Responde SOLO con la explicación, sin preámbulos ni metadatos."""

        return f"{system_context}\n{prediction_info}\n{factors_detail}\n{request}"

    def _call_ollama(self, prompt: str) -> dict[str, Any]:
        """Realiza llamada HTTP a Ollama para generar texto.

        Args:
            prompt: Prompt para el modelo.

        Returns:
            Respuesta JSON de Ollama.

        Raises:
            RequestException: Si hay error en la comunicación.
            Timeout: Si la llamada excede el timeout.
        """
        url = f"{OLLAMA_BASE_URL}/api/generate"
        
        payload = {
            "model": self.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,  # Respuestas más consistentes
                "top_p": 0.9,
                "max_tokens": 500,   # Explicaciones concisas
                "seed": RANDOM_STATE  # Reproducibilidad
            }
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        logger.debug("Llamando Ollama: modelo=%s, timeout=%.1fs", self.llm_model, self.timeout_s)
        
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=self.timeout_s
        )
        
        response.raise_for_status()
        return response.json()

    def _process_llm_response(
        self,
        response: dict[str, Any],
        prediction: int,
        confidence: float,
        top_factors: list[tuple[str, float, Any]],
    ) -> str:
        """Procesa respuesta de Ollama y genera explicación final.

        Args:
            response: Respuesta JSON de Ollama.
            prediction: Clase predicha.
            confidence: Confianza de la predicción.
            top_factors: Factores principales.

        Returns:
            Explicación procesada y formateada.
        """
        # Extraer texto generado
        generated_text = response.get("response", "").strip()
        
        if not generated_text:
            logger.warning("Respuesta vacía de Ollama, usando fallback")
            return self._generate_fallback_explanation(
                prediction, confidence, 
                {name: shap for name, shap, _ in top_factors},
                {name: value for name, _, value in top_factors}
            )
        
        # Limpiar y formatear respuesta
        explanation = self._format_explanation(generated_text, prediction, confidence)
        
        return explanation

    def _format_explanation(self, text: str, prediction: int, confidence: float) -> str:
        """Formatea la explicación generada por LLM.

        Args:
            text: Texto generado por LLM.
            prediction: Clase predicha.
            confidence: Confianza de la predicción.

        Returns:
            Explicación formateada.
        """
        # Limpiar texto
        text = text.strip()
        
        # Agregar header con información básica
        class_names = ["AUDITADA", "GLOSADA", "DEVUELTA"]
        header = f"🏥 RESULTADO DE AUDITORÍA: {class_names[prediction]} (Confianza: {confidence:.1%})\n\n"
        
        # Asegurar que no contiene términos técnicos de ML
        ml_terms = ["shap", "embedding", "feature importance", "machine learning", "modelo", "algoritmo"]
        for term in ml_terms:
            text = text.replace(term.lower(), "factor de análisis")
            text = text.replace(term.upper(), "FACTOR DE ANÁLISIS")
            text = text.replace(term.capitalize(), "Factor de análisis")
        
        return header + text

    def _generate_fallback_explanation(
        self,
        prediction: int,
        confidence: float,
        shap_values: dict[str, float],
        feature_values: dict[str, Any],
    ) -> str:
        """Genera explicación SHAP técnica como fallback.

        Args:
            prediction: Clase predicha.
            confidence: Confianza de la predicción.
            shap_values: Valores SHAP por feature.
            feature_values: Valores reales de las features.

        Returns:
            Explicación técnica basada en SHAP.
        """
        logger.info("Generando explicación fallback (SHAP técnica)")
        
        # Obtener plantilla para la clase
        template = EXPLANATION_TEMPLATES.get(prediction, EXPLANATION_TEMPLATES[0])
        class_names = ["AUDITADA", "GLOSADA", "DEVUELTA"]
        
        # Header
        explanation = f"🏥 RESULTADO DE AUDITORÍA: {class_names[prediction]} (Confianza: {confidence:.1%})\n\n"
        explanation += f"{template['intro']}\n\n"
        
        # Factores principales
        top_factors = self._get_top_factors(shap_values, feature_values, n=3)
        explanation += f"{template['factors']}\n"
        
        for i, (feature_name, shap_value, feature_value) in enumerate(top_factors, 1):
            # Obtener causal asociado
            causal = FEATURE_TO_CAUSAL.get(feature_name, "")
            
            # Sanitizar nombre de feature: eliminar términos ML
            _ML_TERMS_IN_NAMES = ["shap", "embedding", "model_", "feature_importance"]
            display_name = feature_name
            for term in _ML_TERMS_IN_NAMES:
                display_name = display_name.replace(term, "factor_analisis")
            
            # Descripción del factor
            impact = "aumenta" if shap_value > 0 else "disminuye"
            explanation += f"\n{i}. **{display_name}**: {feature_value}\n"
            explanation += f"   - Este factor {impact} la probabilidad de la clasificación actual\n"
            
            if causal:
                explanation += f"   - Relacionado con: {causal}\n"
        
        # Conclusión
        explanation += f"\n{template['conclusion']}"
        
        return explanation

    def _get_top_factors(
        self,
        shap_values: dict[str, float],
        feature_values: dict[str, Any],
        n: int = 3,
    ) -> list[tuple[str, float, Any]]:
        """Obtiene los N factores más importantes según valores SHAP.

        Args:
            shap_values: Valores SHAP por feature.
            feature_values: Valores reales de las features.
            n: Número de factores a retornar.

        Returns:
            Lista de tuplas (feature_name, shap_value, feature_value) ordenada por importancia.
        """
        # Ordenar por valor absoluto de SHAP (importancia)
        sorted_features = sorted(
            shap_values.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Tomar top N y agregar valores de features
        top_factors = []
        for feature_name, shap_value in sorted_features[:n]:
            feature_value = feature_values.get(feature_name, "N/A")
            top_factors.append((feature_name, shap_value, feature_value))
        
        return top_factors

    def get_status(self) -> dict[str, Any]:
        """Obtiene estado actual del generador de explicaciones.

        Returns:
            Diccionario con información de estado.
        """
        return {
            "ollama_available": self.ollama_available,
            "llm_model": self.llm_model,
            "timeout_s": self.timeout_s,
            "ollama_base_url": OLLAMA_BASE_URL,
            "feature_mappings_count": len(FEATURE_TO_CAUSAL),
        }

    def test_generation(self) -> dict[str, Any]:
        """Prueba la generación de explicaciones con datos sintéticos.

        Returns:
            Diccionario con resultados de la prueba.
        """
        # Datos sintéticos para prueba
        test_shap = {
            "valor_facturado": 0.15,
            "codigo_cups": -0.12,
            "numero_autorizacion": 0.08,
        }
        
        test_features = {
            "valor_facturado": 150000,
            "codigo_cups": "890201",
            "numero_autorizacion": "AUT123456",
        }
        
        start_time = time.time()
        
        try:
            explanation = self.generate(
                prediction=1,  # Glosada
                confidence=0.85,
                shap_values=test_shap,
                feature_values=test_features
            )
            
            elapsed_time = time.time() - start_time
            
            return {
                "success": True,
                "explanation": explanation,
                "elapsed_time_s": elapsed_time,
                "used_llm": self.ollama_available,
                "explanation_length": len(explanation),
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "elapsed_time_s": time.time() - start_time,
                "used_llm": False,
            }