# -*- coding: utf-8 -*-
"""Factories de datos de prueba para la suite de testing.

Proporciona funciones factory para generar datos de prueba válidos e inválidos
de forma programática, facilitando la creación de casos de test.
"""

from typing import Literal


def make_valid_features(feature_names: list[str], **overrides) -> dict[str, float]:
    """Factory para diccionarios de features válidos con valores opcionales.
    
    Crea un diccionario base con todas las features en 0.0 y aplica
    los overrides especificados.
    
    Args:
        feature_names: Lista de 48 nombres de features.
        **overrides: Valores específicos para sobrescribir (ej: valor_factura=150000.0).
    
    Returns:
        dict: Diccionario con 48 features válidas.
    
    Examples:
        >>> features = make_valid_features(feature_names, valor_factura=100000.0)
        >>> features['valor_factura']
        100000.0
        >>> features['num_items']
        0.0
    """
    base = {name: 0.0 for name in feature_names}
    base.update(overrides)
    return base


def make_realistic_features() -> dict[str, float]:
    """Factory para features realistas basadas en medianas del dataset.
    
    Returns:
        dict: Diccionario con 48 features con valores realistas de una
            factura típica del dataset de entrenamiento.
    """
    return {
        "anio_radicacion": 2025.0,
        "avg_valor_item": 50000.0,
        "desviacion_dias_nit": 1.2,
        "dias_hasta_radicacion": 15.0,
        "dias_radicacion_promedio_nit": 12.0,
        "dias_x_log_valor": 180.0,
        "discrepancia_valor": 0.0,
        "is_extemporanea": 0.0,
        "log_avg_valor_item": 10.82,
        "log_discrepancia_valor": 0.0,
        "log_max_valor_item": 11.5,
        "log_num_cups_distintos": 1.1,
        "log_num_dx_distintos": 0.69,
        "log_num_items": 1.1,
        "log_num_items_con_autorizacion": 0.69,
        "log_sum_valor_facturado": 12.0,
        "log_total_facturas_nit": 10.8,
        "log_valor_copago": 0.0,
        "log_valor_cuota_moderadora": 0.0,
        "log_valor_factura": 12.0,
        "log_valor_promedio_nit": 11.5,
        "log_valor_x_tasa_glosa": 1.5,
        "max_valor_item": 100000.0,
        "mes_radicacion": 6.0,
        "num_cups_distintos": 2.0,
        "num_dx_distintos": 1.0,
        "num_items": 3.0,
        "num_items_con_autorizacion": 1.0,
        "num_tipos_servicio": 2.0,
        "porcentaje_mayor_item": 0.6,
        "ratio_autorizacion": 0.33,
        "ratio_cups_items": 0.67,
        "ratio_valor_vs_nit": 1.1,
        "regimen_encoded": 0.0,
        "sin_detalles": 0.0,
        "sum_valor_facturado": 150000.0,
        "tasa_devolucion_nit": 0.03,
        "tasa_glosa_nit": 0.13,
        "tiene_copago": 0.0,
        "tiene_cuota_mod": 0.0,
        "tiene_fev": 1.0,
        "tipo_contrato_encoded": 1.0,
        "total_facturas_nit": 50000.0,
        "trimestre_radicacion": 2.0,
        "valor_copago": 0.0,
        "valor_cuota_moderadora": 0.0,
        "valor_factura": 150000.0,
        "valor_promedio_nit": 130000.0,
    }


def make_invalid_features(
    feature_names: list[str],
    error_type: Literal["missing", "extra", "nan", "inf", "negative", "wrong_type"],
) -> dict:
    """Factory para features inválidas para testing de error handling.
    
    Args:
        feature_names: Lista de 48 nombres de features.
        error_type: Tipo de error a generar:
            - "missing": Solo incluye 1 feature (faltan 47)
            - "extra": Incluye las 48 + 1 campo extra
            - "nan": Incluye NaN en valor_factura
            - "inf": Incluye Infinity en valor_factura
            - "negative": Incluye valor negativo en campo que debe ser positivo
            - "wrong_type": Incluye string en lugar de float
    
    Returns:
        dict: Diccionario inválido para testing.
    
    Examples:
        >>> invalid = make_invalid_features(feature_names, "missing")
        >>> len(invalid)
        1
        >>> invalid = make_invalid_features(feature_names, "nan")
        >>> import math
        >>> math.isnan(invalid['valor_factura'])
        True
    """
    if error_type == "missing":
        return {"valor_factura": 1.0}  # Solo 1 de 48
    
    elif error_type == "extra":
        base = make_valid_features(feature_names)
        base["campo_invalido_extra"] = 999.0
        return base
    
    elif error_type == "nan":
        base = make_valid_features(feature_names)
        base["valor_factura"] = float('nan')
        return base
    
    elif error_type == "inf":
        base = make_valid_features(feature_names)
        base["valor_factura"] = float('inf')
        return base
    
    elif error_type == "negative":
        base = make_valid_features(feature_names)
        base["valor_factura"] = -100000.0  # Valor negativo inválido
        return base
    
    elif error_type == "wrong_type":
        base = make_valid_features(feature_names)
        base["valor_factura"] = "cien mil"  # String en lugar de float
        return base
    
    else:
        raise ValueError(f"error_type desconocido: {error_type}")


def make_prediction_request(
    numero_factura: str,
    features: dict[str, float],
) -> dict:
    """Factory para payload de request de predicción individual.
    
    Args:
        numero_factura: Número identificador de la factura.
        features: Diccionario de 48 features.
    
    Returns:
        dict: Payload listo para POST /v1/predict.
    
    Examples:
        >>> request = make_prediction_request("FAC-001", make_valid_features(feature_names))
        >>> request['numero_factura']
        'FAC-001'
        >>> len(request['features'])
        48
    """
    return {
        "numero_factura": numero_factura,
        "features": features,
    }


def make_batch_request(
    count: int,
    feature_names: list[str],
    numero_factura_prefix: str = "BATCH",
) -> dict:
    """Factory para payload de request de predicción batch.
    
    Args:
        count: Número de facturas en el batch.
        feature_names: Lista de 48 nombres de features.
        numero_factura_prefix: Prefijo para los números de factura.
    
    Returns:
        dict: Payload listo para POST /v1/predict/batch.
    
    Examples:
        >>> batch = make_batch_request(10, feature_names)
        >>> len(batch['facturas'])
        10
        >>> batch['facturas'][0]['numero_factura']
        'BATCH-000'
    """
    return {
        "facturas": [
            make_prediction_request(
                f"{numero_factura_prefix}-{i:03d}",
                make_valid_features(feature_names),
            )
            for i in range(count)
        ]
    }


def make_malicious_payload(
    attack_type: Literal["sql_injection", "xss", "path_traversal"],
) -> str:
    """Factory para payloads maliciosos para security testing.
    
    Args:
        attack_type: Tipo de ataque a simular:
            - "sql_injection": Intento de SQL injection
            - "xss": Intento de XSS (Cross-Site Scripting)
            - "path_traversal": Intento de path traversal
    
    Returns:
        str: Payload malicioso para testing.
    
    Examples:
        >>> payload = make_malicious_payload("sql_injection")
        >>> "DROP TABLE" in payload
        True
    """
    if attack_type == "sql_injection":
        return "'; DROP TABLE facturas; --"
    
    elif attack_type == "xss":
        return "<script>alert('xss')</script>"
    
    elif attack_type == "path_traversal":
        return "../../etc/passwd"
    
    else:
        raise ValueError(f"attack_type desconocido: {attack_type}")
