# Design Document: Local LLM Integration

## Overview

La integración de un modelo de lenguaje local (LLM local) en el pipeline de auditoría prestacional de Savia Salud EPS busca complementar el modelo XGBoost v1 existente para superar la meta de Recall Glosada de 0.80 (actual: 0.7261) y enriquecer las explicaciones para auditores médicos. El enfoque es estrictamente local (Ollama + Sentence Transformers + spaCy) para cumplir con la Ley 1581 de Colombia y garantizar que los datos sensibles de salud nunca salgan de la infraestructura propia.

La integración opera en dos modos complementarios: (1) **enriquecimiento de features** mediante embeddings semánticos de campos textuales (diagnósticos CIE-10, códigos CUPS, observaciones) que se incorporan al pipeline de predicción XGBoost, y (2) **generación de explicaciones** en lenguaje natural enriquecido para auditores, reemplazando las explicaciones SHAP técnicas por narrativas contextualizadas con causales de la Resolución 2284/2023. El diseño prioriza la evaluación experimental de múltiples candidatos LLM antes de comprometerse con uno, manteniendo el XGBoost como modelo base irremplazable.

