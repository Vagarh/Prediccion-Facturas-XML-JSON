#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script para verificar requisitos del reentrenamiento XGBoost.

Verifica que todos los archivos necesarios existen y las dependencias
están instaladas antes de ejecutar el reentrenamiento.
"""

import sys
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_datasets():
    """Verifica que los datasets necesarios existen."""
    logger.info("Verificando datasets...")
    
    processed_dir = Path("data/processed")
    required_files = [
        "20260309_train.parquet",
        "20260309_val.parquet", 
        "20260309_test.parquet",
        "20260309_feature_list.txt"
    ]
    
    missing_files = []
    for filename in required_files:
        file_path = processed_dir / filename
        if not file_path.exists():
            missing_files.append(str(file_path))
        else:
            logger.info("✓ Encontrado: %s", file_path)
    
    if missing_files:
        logger.error("Archivos faltantes:")
        for file_path in missing_files:
            logger.error("  ✗ %s", file_path)
        return False
    
    logger.info("✓ Todos los datasets están disponibles")
    return True


def check_dependencies():
    """Verifica que las dependencias necesarias están instaladas."""
    logger.info("Verificando dependencias...")
    
    required_packages = [
        "pandas",
        "numpy", 
        "scikit-learn",
        "xgboost",
        "sentence-transformers",
        "spacy",
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            logger.info("✓ %s instalado", package)
        except ImportError:
            missing_packages.append(package)
            logger.error("✗ %s NO instalado", package)
    
    if missing_packages:
        logger.error("Paquetes faltantes:")
        for package in missing_packages:
            logger.error("  pip install %s", package)
        return False
    
    logger.info("✓ Todas las dependencias están instaladas")
    return True


def check_spacy_model():
    """Verifica que el modelo spaCy en español está descargado."""
    logger.info("Verificando modelo spaCy...")
    
    try:
        import spacy
        nlp = spacy.load("es_core_news_md")
        logger.info("✓ Modelo spaCy 'es_core_news_md' disponible")
        return True
    except OSError:
        logger.error("✗ Modelo spaCy 'es_core_news_md' NO encontrado")
        logger.error("  Ejecutar: python -m spacy download es_core_news_md")
        return False


def check_directories():
    """Verifica que los directorios necesarios existen."""
    logger.info("Verificando directorios...")
    
    required_dirs = [
        "data/processed",
        "models/artifacts", 
        "reports",
        "src/embeddings",
        "src/nlp",
        "src/privacy",
        "src/evaluation"
    ]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if not path.exists():
            logger.info("Creando directorio: %s", path)
            path.mkdir(parents=True, exist_ok=True)
        else:
            logger.info("✓ Directorio existe: %s", path)
    
    return True


def check_source_files():
    """Verifica que los archivos fuente necesarios existen."""
    logger.info("Verificando archivos fuente...")
    
    required_files = [
        "src/config.py",
        "src/embeddings/service.py",
        "src/nlp/pipeline.py", 
        "src/privacy/guard.py",
        "src/features/embedding_pipeline.py",
        "src/evaluation/retrain_xgboost.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        path = Path(file_path)
        if not path.exists():
            missing_files.append(str(path))
        else:
            logger.info("✓ Archivo fuente: %s", path)
    
    if missing_files:
        logger.error("Archivos fuente faltantes:")
        for file_path in missing_files:
            logger.error("  ✗ %s", file_path)
        return False
    
    logger.info("✓ Todos los archivos fuente están disponibles")
    return True


def main():
    """Función principal de verificación."""
    logger.info("=== VERIFICACIÓN DE REQUISITOS PARA REENTRENAMIENTO ===")
    
    checks = [
        ("Directorios", check_directories),
        ("Archivos fuente", check_source_files),
        ("Datasets", check_datasets),
        ("Dependencias Python", check_dependencies),
        ("Modelo spaCy", check_spacy_model),
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        logger.info("\n--- %s ---", check_name)
        try:
            if not check_func():
                all_passed = False
        except Exception as e:
            logger.error("Error en verificación %s: %s", check_name, e)
            all_passed = False
    
    logger.info("\n=== RESUMEN ===")
    if all_passed:
        logger.info("✓ Todos los requisitos están satisfechos")
        logger.info("Puede ejecutar el reentrenamiento con:")
        logger.info("  python src/evaluation/retrain_xgboost.py")
        return 0
    else:
        logger.error("✗ Algunos requisitos no están satisfechos")
        logger.error("Corrija los problemas antes de ejecutar el reentrenamiento")
        return 1


if __name__ == "__main__":
    sys.exit(main())