#!/usr/bin/env python3
"""
Regenera las figuras SHAP corregidas:
- shap_02_beeswarm_por_clase.png  (Figura 7.6 en tesis)
- shap_03_waterfall_por_clase.png (Figura 7.7/7.8 en tesis)

Fixes aplicados:
- Figuras separadas por clase (1 figura por clase, no 3 en una)
- Labels traducidos a español con FEATURE_LABELS
- Tamaño de fuente reducido en beeswarm
- Figuras más altas para dar espacio vertical a los labels
- Sufijo _v2 para no sobreescribir los originales
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from pathlib import Path

# ── Rutas ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR  = BASE_DIR / 'data' / 'processed'
MODELS_DIR = BASE_DIR / 'models' / 'artifacts'
FIGURES_DIR = BASE_DIR / 'reports' / 'figures'
EVAL_DIR   = BASE_DIR / 'reports' / 'evaluations'

DATE_PREFIX  = '20260309'
RANDOM_STATE = 42
CLASS_NAMES  = ['Auditada', 'Glosada', 'Devuelta']

# ── Feature labels (español, versiones cortas para gráficos) ─────────────────
FEATURE_LABELS = {
    'log_num_items_con_autorizacion': 'Log ítems con autorización',
    'num_items_con_autorizacion':     'Ítems con autorización',
    'log_num_dx_distintos':           'Log diagnósticos distintos',
    'num_dx_distintos':               'Diagnósticos distintos',
    'tiene_fev':                      'Tiene FEV electrónica',
    'log_valor_x_tasa_glosa':         'Log valor × tasa glosa',
    'num_tipos_servicio':             'Tipos de servicio',
    'tasa_devolucion_nit':            'Tasa devolución prestador',
    'ratio_autorizacion':             'Ratio autorización',
    'tasa_glosa_nit':                 'Tasa glosa prestador',
    'log_total_facturas_nit':         'Log total facturas NIT',
    'log_max_valor_item':             'Log max valor ítem',
    'avg_valor_item':                 'Valor promedio ítem',
    'log_num_cups_distintos':         'Log CUPS distintos',
    'ratio_valor_vs_nit':             'Ratio valor vs NIT',
    'num_items':                      'Número ítems',
    'log_num_items':                  'Log número ítems',
    'mes_radicacion':                 'Mes radicación',
    'num_dx_distintos':               'Diagnósticos distintos',
    'log_num_items_con_autorizacion': 'Log ítems con autorización',
    'dias_x_log_valor':               'Días × log valor',
    'traslado_promedio_nit':          'Traslado promedio NIT',
    'log_sum_valor_facturado':        'Log suma valor facturado',
    'log_valor_promedio_nit':         'Log valor promedio NIT',
    'log_valor_cop_go':               'Log valor COP go',
    'ratio_valor_vs_nit':             'Ratio valor vs NIT',
    'dias_x_log_valor':               'Días × log valor',
    'log_valor_promedio_nit':         'Log valor promedio NIT',
    'traslado_promedio_nit':          'Traslado promedio NIT',
    'log_sum_valor_facturado':        'Log suma valor facturado',
    'log_valor_cop_go':               'Log valor COP go',
}

# ── Carga ─────────────────────────────────────────────────────────────────────
print("Cargando features, modelo y datos...")
with open(DATA_DIR / f'{DATE_PREFIX}_feature_list.txt') as f:
    FEATURES = [l.strip() for l in f if l.strip()]

xgb_model = xgb.XGBClassifier()
xgb_model.load_model(str(MODELS_DIR / f'{DATE_PREFIX}_xgboost_v1.json'))

test_df = pd.read_parquet(DATA_DIR / f'{DATE_PREFIX}_test.parquet')
X_test  = test_df[FEATURES]
y_test  = test_df['target']

# Muestra estratificada idéntica a la del notebook
np.random.seed(RANDOM_STATE)
sample_indices = []
for cls in [0, 1, 2]:
    cls_idx = y_test[y_test == cls].index.tolist()
    n = min(len(cls_idx), 2000 // 3)
    sample_indices.extend(np.random.choice(cls_idx, size=n, replace=False))
np.random.shuffle(sample_indices)
X_sample = X_test.loc[sample_indices].reset_index(drop=True)
y_sample = y_test.loc[sample_indices].reset_index(drop=True)
print(f"Muestra: {len(X_sample):,} facturas")

# SHAP values
print("Calculando SHAP values...")
explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer(X_sample)
print(f"Shape: {shap_values.shape}")

y_pred_sample  = xgb_model.predict(X_sample)
y_proba_sample = xgb_model.predict_proba(X_sample)

# Traducir feature names para que SHAP los muestre en español
def get_label(feat):
    return FEATURE_LABELS.get(feat, feat)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 7.6 — Beeswarm: UNA figura por clase (3 PNGs separados) + combinado
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Generando beeswarm plots ──")

for i, cls_name in enumerate(CLASS_NAMES):
    # Crear Explanation con labels en español
    feature_labels_list = [get_label(f) for f in FEATURES]
    shap_cls = shap.Explanation(
        values       = shap_values.values[:, :, i],
        base_values  = shap_values.base_values[:, i],
        data         = X_sample.values,
        feature_names= feature_labels_list,
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.sca(ax)
    shap.plots.beeswarm(
        shap_cls,
        max_display=15,
        show=False,
    )

    # Ajustar tamaño de fuente de los y-tick labels (feature names)
    ax.tick_params(axis='y', labelsize=9)
    ax.tick_params(axis='x', labelsize=9)
    ax.set_xlabel('Valor SHAP (impacto en predicción)', fontsize=10)
    ax.set_title(
        f'SHAP Beeswarm — Clase {i}: {cls_name}\n'
        f'Rojo = valor alto de la feature · Azul = valor bajo',
        fontsize=12, fontweight='bold', pad=10
    )

    plt.tight_layout()
    out = FIGURES_DIR / f'shap_02_beeswarm_{cls_name.lower()}_v2.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {out.name}")


# ── Versión combinada mejorada (para tesis, 1 figura 3 columnas legible) ──────
fig, axes = plt.subplots(1, 3, figsize=(22, 10))
fig.subplots_adjust(wspace=0.55)   # más espacio horizontal entre columnas

for i, cls_name in enumerate(CLASS_NAMES):
    feature_labels_list = [get_label(f) for f in FEATURES]
    shap_cls = shap.Explanation(
        values       = shap_values.values[:, :, i],
        base_values  = shap_values.base_values[:, i],
        data         = X_sample.values,
        feature_names= feature_labels_list,
    )
    plt.sca(axes[i])
    shap.plots.beeswarm(shap_cls, max_display=12, show=False)
    axes[i].tick_params(axis='y', labelsize=7.5)
    axes[i].tick_params(axis='x', labelsize=8)
    axes[i].set_xlabel('Valor SHAP', fontsize=9)
    axes[i].set_title(f'Clase {i} — {cls_name}', fontweight='bold', fontsize=11, pad=6)

fig.suptitle(
    'SHAP Beeswarm Plots por Clase — XGBoost v1\n'
    'Rojo = valor alto de la feature · Azul = valor bajo',
    fontsize=13, fontweight='bold'
)
out_combined = FIGURES_DIR / 'shap_02_beeswarm_por_clase_v2.png'
fig.savefig(out_combined, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"  ✓ {out_combined.name}  ← figura combinada para tesis")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 7.7/7.8 — Waterfall: UNA figura por clase (3 PNGs separados) + comb.
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Generando waterfall plots ──")

def find_representative(y_true, y_pred, y_proba, cls):
    mask = (y_true == cls) & (y_pred == cls)
    idx  = np.where(mask)[0]
    if len(idx) == 0:
        idx = np.where(y_true == cls)[0]
    conf = y_proba[idx, cls]
    return idx[np.argsort(conf)[int(len(conf) * 0.75)]]

example_indices = {
    cls_name: find_representative(y_sample.values, y_pred_sample, y_proba_sample, i)
    for i, cls_name in enumerate(CLASS_NAMES)
}

feature_labels_list = [get_label(f) for f in FEATURES]

for i, cls_name in enumerate(CLASS_NAMES):
    idx = example_indices[cls_name]
    shap_ex = shap.Explanation(
        values       = shap_values.values[idx, :, i],
        base_values  = shap_values.base_values[idx, i],
        data         = X_sample.iloc[idx].values,
        feature_names= feature_labels_list,
    )
    prob = y_proba_sample[idx, i]

    fig, ax = plt.subplots(figsize=(10, 7))
    plt.sca(ax)
    shap.plots.waterfall(shap_ex, max_display=12, show=False)
    ax.tick_params(axis='y', labelsize=9)
    ax.tick_params(axis='x', labelsize=9)
    ax.set_title(
        f'Waterfall — Clase {i}: {cls_name}\n'
        f'P({cls_name}) = {prob:.2f} · Real = {CLASS_NAMES[int(y_sample.iloc[idx])]}',
        fontsize=12, fontweight='bold', pad=10
    )
    plt.tight_layout()
    out = FIGURES_DIR / f'shap_03_waterfall_{cls_name.lower()}_v2.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {out.name}")


# ── Versión combinada mejorada ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(24, 9))
fig.subplots_adjust(wspace=0.5)

for i, cls_name in enumerate(CLASS_NAMES):
    idx = example_indices[cls_name]
    shap_ex = shap.Explanation(
        values       = shap_values.values[idx, :, i],
        base_values  = shap_values.base_values[idx, i],
        data         = X_sample.iloc[idx].values,
        feature_names= feature_labels_list,
    )
    prob = y_proba_sample[idx, i]
    plt.sca(axes[i])
    shap.plots.waterfall(shap_ex, max_display=11, show=False)
    axes[i].tick_params(axis='y', labelsize=8)
    axes[i].tick_params(axis='x', labelsize=8)
    axes[i].set_title(
        f'Clase {i} — {cls_name}\n'
        f'P({cls_name})={prob:.2f} | Real={CLASS_NAMES[int(y_sample.iloc[idx])]}',
        fontsize=10, fontweight='bold', pad=6
    )

fig.suptitle(
    'Waterfall Plots — Explicación individual por clase\n'
    'Cada barra = contribución de una feature a la probabilidad predicha',
    fontsize=13, fontweight='bold'
)
out_combined = FIGURES_DIR / 'shap_03_waterfall_por_clase_v2.png'
fig.savefig(out_combined, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"  ✓ {out_combined.name}  ← figura combinada para tesis")

print("\n✓ Todas las figuras generadas en reports/figures/")
print("  Individuales (6 PNG) + Combinadas (2 PNG) = 8 archivos")
