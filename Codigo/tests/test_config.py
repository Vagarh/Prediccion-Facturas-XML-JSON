# -*- coding: utf-8 -*-
"""Tests para el módulo de configuración central."""

import re

import pytest

from src.config import (
    CLASS_ACTIONS,
    CLASS_NAMES,
    DATA_DIR,
    FEATURE_LABELS,
    FEATURE_LIST_PATH,
    FEATURE_TO_CAUSAL,
    MODEL_METRICS,
    MODEL_PATH,
    RANDOM_STATE,
    RISK_DESCRIPTIONS,
    setup_logging,
)


@pytest.mark.unit
class TestConfigConstants:
    """Tests para las constantes de configuración."""

    def test_class_names_has_3_classes(self) -> None:
        """CLASS_NAMES tiene exactamente 3 clases."""
        assert len(CLASS_NAMES) == 3
        assert set(CLASS_NAMES.keys()) == {0, 1, 2}

    def test_class_actions_has_3_classes(self) -> None:
        """CLASS_ACTIONS tiene exactamente 3 acciones."""
        assert len(CLASS_ACTIONS) == 3
        assert set(CLASS_ACTIONS.keys()) == {0, 1, 2}

    def test_model_metrics_has_expected_keys(self) -> None:
        """MODEL_METRICS tiene las métricas esperadas."""
        expected = {
            "f1_macro",
            "auc_roc_ovr",
            "recall_auditada",
            "recall_glosada",
            "recall_devuelta",
            "accuracy",
        }
        assert set(MODEL_METRICS.keys()) == expected

    def test_feature_labels_non_empty(self) -> None:
        """Todas las etiquetas de features son strings no vacíos."""
        for key, value in FEATURE_LABELS.items():
            assert isinstance(value, str), f"Label para {key} no es string"
            assert len(value) > 0, f"Label para {key} está vacía"

    def test_feature_to_causal_values_valid(self) -> None:
        """Los códigos causales siguen el patrón esperado."""
        pattern = re.compile(r"^(DE|FA|SO|TA)\d*$")
        for feat, causal in FEATURE_TO_CAUSAL.items():
            assert pattern.match(causal), (
                f"Causal '{causal}' para feature '{feat}' no sigue el patrón"
            )

    def test_random_state_is_42(self) -> None:
        """RANDOM_STATE es 42 para reproducibilidad."""
        assert RANDOM_STATE == 42

    def test_risk_descriptions_structure(self) -> None:
        """Cada RISK_DESCRIPTION tiene la estructura (bool, str, str)."""
        for feat, desc in RISK_DESCRIPTIONS.items():
            assert len(desc) == 3, f"RISK_DESCRIPTIONS['{feat}'] no tiene 3 elementos"
            assert isinstance(desc[0], bool), f"is_high_risky no es bool para '{feat}'"
            assert isinstance(desc[1], str), f"text_high no es str para '{feat}'"
            assert isinstance(desc[2], str), f"text_low no es str para '{feat}'"


@pytest.mark.unit
class TestConfigPaths:
    """Tests para las rutas de configuración."""

    def test_model_path_has_supported_extension(self) -> None:
        """MODEL_PATH apunta a un formato soportado por src.models.predict.load_model.

        ``.json`` para XGBoost nativo, ``.joblib`` para cualquier otro estimador
        sklearn-compatible (LightGBM, CatBoost, ...) serializado con joblib.
        """
        assert MODEL_PATH.suffix in {".json", ".joblib"}

    def test_feature_list_path_is_txt(self) -> None:
        """FEATURE_LIST_PATH apunta a un archivo .txt."""
        assert FEATURE_LIST_PATH.suffix == ".txt"


@pytest.mark.unit
class TestSetupLogging:
    """Tests para la configuración de logging."""

    def test_setup_logging_does_not_raise(self) -> None:
        """setup_logging() se ejecuta sin errores."""
        setup_logging()
