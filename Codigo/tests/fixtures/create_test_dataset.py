# -*- coding: utf-8 -*-
"""Script para crear dataset de prueba de 100 muestras estratificadas.

Extrae 100 muestras del test set (33 Auditada, 33 Glosada, 34 Devuelta)
para usar en tests de regresión del modelo.
"""

import sys
from pathlib import Path

import pandas as pd

# Agregar src al path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR


def create_test_dataset():
    """Crea test_dataset_100.parquet con muestras estratificadas."""
    # Cargar test set completo
    test_path = DATA_DIR / "processed" / "20260309_test.parquet"
    
    if not test_path.exists():
        print(f"Error: No se encontró {test_path}")
        print("Asegúrate de que el dataset de test existe.")
        return
    
    print(f"Cargando test set desde {test_path}...")
    df_test = pd.read_parquet(test_path)
    
    print(f"Test set cargado: {len(df_test)} filas")
    print(f"Distribución de clases:")
    print(df_test['target'].value_counts().sort_index())
    
    # Estratificar: 33 Auditada, 33 Glosada, 34 Devuelta
    samples_per_class = {0: 33, 1: 33, 2: 34}
    
    sampled_dfs = []
    for clase, n_samples in samples_per_class.items():
        df_clase = df_test[df_test['target'] == clase]
        
        if len(df_clase) < n_samples:
            print(f"Advertencia: Clase {clase} tiene solo {len(df_clase)} muestras, se pedían {n_samples}")
            n_samples = len(df_clase)
        
        # Muestreo aleatorio con semilla fija para reproducibilidad
        df_sampled = df_clase.sample(n=n_samples, random_state=42)
        sampled_dfs.append(df_sampled)
        print(f"Clase {clase}: {n_samples} muestras extraídas")
    
    # Combinar y mezclar
    df_test_100 = pd.concat(sampled_dfs, ignore_index=True)
    df_test_100 = df_test_100.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Guardar
    output_path = Path(__file__).parent / "test_dataset_100.parquet"
    df_test_100.to_parquet(output_path, index=False)
    
    print(f"\nDataset de prueba creado: {output_path}")
    print(f"Total de muestras: {len(df_test_100)}")
    print(f"Distribución final:")
    print(df_test_100['target'].value_counts().sort_index())
    print(f"Columnas: {list(df_test_100.columns)}")
    print(f"Tamaño del archivo: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    create_test_dataset()
