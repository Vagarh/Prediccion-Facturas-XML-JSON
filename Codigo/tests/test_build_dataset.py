# -*- coding: utf-8 -*-
"""Tests del pipeline de feature engineering ``src.features.build_dataset``.

Valida que el port del notebook 04 reproduce el contrato de 48 features, que las
transformaciones son deterministas, que el split temporal no se solapa y que no
se filtran columnas de leakage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.build_dataset import (
    add_flags,
    add_interactions,
    add_log_features,
    assert_no_leakage,
    build_feature_frame,
    feature_columns,
    impute,
    temporal_split,
)

pytestmark = pytest.mark.unit

FROZEN_FEATURE_LIST = (
    Path(__file__).resolve().parent.parent / "data" / "processed" / "20260309_feature_list.txt"
)

RAW_COLUMNS = [
    "factura_id", "target", "nit_prestador", "tipo_contrato_codigo", "tipo_contrato",
    "regimen_codigo", "regimen", "valor_factura", "valor_factura_original", "valor_copago",
    "valor_cuota_moderadora", "fecha_prestacion", "fecha_radicacion", "dias_hasta_radicacion",
    "origen_factura", "pbs", "mes_radicacion", "trimestre_radicacion", "anio_radicacion",
    "num_items", "num_cups_distintos", "num_dx_distintos", "sum_valor_facturado",
    "max_valor_item", "avg_valor_item", "num_items_pbs", "num_items_con_autorizacion",
    "num_tipos_servicio", "num_ambitos_distintos", "total_facturas_nit", "tasa_devolucion_nit",
    "tasa_glosa_nit", "valor_promedio_nit", "dias_radicacion_promedio_nit",
    "porcentaje_mayor_item", "discrepancia_valor", "tiene_fev", "tipo_contrato_encoded",
    "regimen_encoded",
]


def _raw_frame(n: int = 240, seed: int = 42) -> pd.DataFrame:
    """DataFrame crudo sintético con las 39 columnas de extract_fresh_dataset."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-03-01")
    fechas = start + pd.to_timedelta(rng.integers(0, 700, n), unit="D")
    df = pd.DataFrame({
        "factura_id": np.arange(1, n + 1),
        "target": rng.integers(0, 3, n),
        "nit_prestador": rng.choice(["900111", "900222", "900333"], n),
        "tipo_contrato_codigo": "1",
        "tipo_contrato": rng.choice(["CAPITA", "Evento", "Paquete"], n),
        "regimen_codigo": "S",
        "regimen": rng.choice(["SUBSIDIADO", "CONTRIBUTIVO"], n),
        "valor_factura": rng.gamma(2, 1e6, n),
        "valor_factura_original": rng.gamma(2, 1e6, n),
        "valor_copago": rng.choice([0, 0, 0, 5000, 12000], n).astype(float),
        "valor_cuota_moderadora": rng.choice([0, 0, 3000], n).astype(float),
        "fecha_prestacion": fechas - pd.to_timedelta(rng.integers(1, 40, n), unit="D"),
        "fecha_radicacion": fechas,
        "dias_hasta_radicacion": rng.integers(0, 90, n),
        "origen_factura": 3,
        "pbs": rng.choice(["S", "N"], n),
        "mes_radicacion": fechas.month,
        "trimestre_radicacion": fechas.quarter,
        "anio_radicacion": fechas.year,
        "num_items": rng.integers(0, 50, n),
        "num_cups_distintos": rng.integers(0, 20, n),
        "num_dx_distintos": rng.integers(0, 10, n),
        "sum_valor_facturado": rng.gamma(2, 1e6, n),
        "max_valor_item": rng.gamma(2, 5e5, n),
        "avg_valor_item": rng.gamma(2, 2e5, n),
        "num_items_pbs": 0,
        "num_items_con_autorizacion": rng.integers(0, 15, n),
        "num_tipos_servicio": rng.integers(1, 5, n),
        "num_ambitos_distintos": 0,
        "total_facturas_nit": rng.integers(100, 50_000, n),
        "tasa_devolucion_nit": rng.random(n) * 20,
        "tasa_glosa_nit": rng.random(n) * 40,
        "valor_promedio_nit": rng.gamma(2, 8e5, n),
        "dias_radicacion_promedio_nit": rng.random(n) * 30,
        "porcentaje_mayor_item": rng.random(n),
        "discrepancia_valor": rng.gamma(1, 1e5, n),
        "tiene_fev": rng.integers(0, 2, n),
        "tipo_contrato_encoded": rng.integers(-1, 3, n),
        "regimen_encoded": rng.integers(-1, 2, n),
    })
    return df[RAW_COLUMNS]


class TestFeatureContract:
    def test_matches_frozen_48_feature_list(self) -> None:
        """El pipeline produce exactamente las 48 features del contrato congelado."""
        frozen = [
            l.strip() for l in FROZEN_FEATURE_LIST.read_text(encoding="utf-8").splitlines() if l.strip()
        ]
        feats = feature_columns(build_feature_frame(_raw_frame()))
        assert set(feats) == set(frozen)
        assert len(feats) == 48

    def test_output_is_sorted(self) -> None:
        feats = feature_columns(build_feature_frame(_raw_frame()))
        assert feats == sorted(feats)

    def test_keeps_id_target_and_dates(self) -> None:
        out = build_feature_frame(_raw_frame())
        for col in ("factura_id", "target", "fecha_prestacion", "fecha_radicacion"):
            assert col in out.columns

    def test_no_nan_in_features(self) -> None:
        out = build_feature_frame(_raw_frame())
        feats = feature_columns(out)
        assert not out[feats].isna().any().any()


class TestDeterminism:
    def test_build_is_deterministic(self) -> None:
        a = build_feature_frame(_raw_frame(seed=1))
        b = build_feature_frame(_raw_frame(seed=1))
        pd.testing.assert_frame_equal(a, b)

    def test_transforms_do_not_mutate_input(self) -> None:
        raw = _raw_frame()
        before = raw.copy()
        _ = add_interactions(add_flags(add_log_features(impute(raw))))
        pd.testing.assert_frame_equal(raw, before)


class TestLeakage:
    def test_assert_no_leakage_passes_on_clean_list(self) -> None:
        assert_no_leakage(feature_columns(build_feature_frame(_raw_frame())))

    def test_assert_no_leakage_raises_on_leakage_col(self) -> None:
        with pytest.raises(AssertionError):
            assert_no_leakage(["valor_factura", "estado_factura"])


class TestTemporalSplit:
    def test_splits_do_not_overlap_and_are_ordered(self) -> None:
        fe = build_feature_frame(_raw_frame(n=600))
        parts = temporal_split(
            fe, val_start="2024-11-01", test_start="2025-03-01",
            test_end="2025-07-01", oot_end="2025-10-01",
        )
        maxes = {k: pd.to_datetime(v["fecha_radicacion"]).max() for k, v in parts.items() if len(v)}
        mins = {k: pd.to_datetime(v["fecha_radicacion"]).min() for k, v in parts.items() if len(v)}
        assert maxes["train"] < mins["val"]
        assert maxes["val"] < mins["test"]
        assert maxes["test"] < mins["oot"]

    def test_split_row_counts_sum_to_windowed_total(self) -> None:
        fe = build_feature_frame(_raw_frame(n=600))
        parts = temporal_split(
            fe, val_start="2024-11-01", test_start="2025-03-01",
            test_end="2025-07-01", oot_end="2025-10-01",
        )
        total = sum(len(v) for v in parts.values())
        in_window = fe[
            (pd.to_datetime(fe["fecha_radicacion"]) >= pd.Timestamp("2024-03-01"))
            & (pd.to_datetime(fe["fecha_radicacion"]) < pd.Timestamp("2025-10-01"))
        ]
        assert total == len(in_window)
