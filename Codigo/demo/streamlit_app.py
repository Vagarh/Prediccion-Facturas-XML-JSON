# -*- coding: utf-8 -*-
"""Demo MVP — Auditoría Predictiva Savia Salud EPS.

Interfaz Streamlit con predicciones XGBoost, explicaciones LLM,
monitoreo de drift y comparación de versiones de modelo.

Uso:
    streamlit run Codigo/demo/streamlit_app.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Asegurar que src/ esté en el path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Configuración de página (debe ser el primer comando Streamlit)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Auditoría Predictiva — Savia Salud EPS",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS personalizado — paleta corporativa Savia Salud
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* Paleta corporativa */
    :root {
        --savia-green: #00875A;
        --savia-blue: #0052CC;
        --savia-light: #E3FCEF;
        --savia-warn: #FF8B00;
        --savia-danger: #DE350B;
        --savia-gray: #F4F5F7;
    }

    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, #00875A 0%, #0052CC 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.8rem; }
    .main-header p  { color: rgba(255,255,255,0.85); margin: 0.3rem 0 0; font-size: 0.95rem; }

    /* Cards de métricas */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 4px solid var(--savia-green);
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 0.8rem;
    }
    .metric-card.warn  { border-left-color: var(--savia-warn); }
    .metric-card.alert { border-left-color: var(--savia-danger); }
    .metric-card.info  { border-left-color: var(--savia-blue); }

    /* Badges de estado */
    .badge-auditada { background:#E3FCEF; color:#006644; padding:3px 10px; border-radius:12px; font-weight:600; }
    .badge-glosada  { background:#FFFAE6; color:#974F0C; padding:3px 10px; border-radius:12px; font-weight:600; }
    .badge-devuelta { background:#FFEBE6; color:#BF2600; padding:3px 10px; border-radius:12px; font-weight:600; }

    /* Sección de explicación */
    .explanation-box {
        background: #F0F4FF;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #B3C7F7;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* Ocultar menú hamburguesa */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Importaciones del proyecto (con manejo de errores)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Cargando modelo XGBoost...")
def load_pipeline():
    """Carga modelo, features y SHAP explainer (singleton)."""
    from src.config import setup_logging
    setup_logging()

    from src.features.engineering import load_feature_names
    from src.models.predict import load_model
    from src.explainability.shap_explanations import create_shap_explainer

    feature_names = load_feature_names()
    model = load_model()
    explainer = create_shap_explainer(model)
    return feature_names, model, explainer


@st.cache_resource(show_spinner="Inicializando componentes LLM...")
def load_llm_components():
    """Carga componentes LLM opcionales (singleton)."""
    try:
        from src.privacy.guard import PrivacyGuard
        from src.embeddings.service import EmbeddingService
        from src.nlp.pipeline import load_nlp_pipeline
        from src.explainability.explanation_generator import ExplanationGenerator
        from src.config import EMBEDDING_MODEL_NAME

        guard = PrivacyGuard()
        guard.verify_local_only()

        if not guard.check_ollama_available():
            return None, None, None, None, False

        nlp = load_nlp_pipeline()
        emb = EmbeddingService(model_name=EMBEDDING_MODEL_NAME)
        gen = ExplanationGenerator(privacy_guard=guard)
        return guard, nlp, emb, gen, True
    except Exception:
        return None, None, None, None, False


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image(
        "https://via.placeholder.com/200x60/00875A/FFFFFF?text=Savia+Salud+EPS",
        use_container_width=True,
    )
    st.markdown("---")

    st.markdown("### ⚙️ Configuración")
    show_shap_detail = st.toggle("Mostrar detalle SHAP técnico", value=False)
    show_embeddings_info = st.toggle("Mostrar info de embeddings", value=False)
    top_n_factors = st.slider("Top factores de riesgo", 3, 10, 5)

    st.markdown("---")
    st.markdown("### 📋 Casos de demostración")
    from demo.demo_cases import DEMO_CASES

    selected_case_id = st.selectbox(
        "Seleccionar caso",
        options=[0] + [c["id"] for c in DEMO_CASES],
        format_func=lambda x: "— Entrada manual —" if x == 0 else f"{DEMO_CASES[x-1]['icono']} Caso {x}: {DEMO_CASES[x-1]['titulo']}",
    )

    st.markdown("---")
    st.markdown("### 🔗 Recursos")
    st.markdown("- [API Docs](http://localhost:8000/docs)")
    st.markdown("- [Health Check](http://localhost:8000/v1/health)")
    st.markdown("- Universidad EAFIT · Tesis 2026")


# ---------------------------------------------------------------------------
# Header principal
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="main-header">
        <h1>🏥 Auditoría Predictiva de Facturas Médicas</h1>
        <p>Sistema de clasificación XGBoost + Embeddings Semánticos · Savia Salud EPS · Universidad EAFIT</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cargar recursos
# ---------------------------------------------------------------------------

try:
    feature_names, model, explainer = load_pipeline()
    pipeline_ok = True
except Exception as e:
    st.error(f"Error cargando modelo: {e}")
    pipeline_ok = False
    st.stop()

guard, nlp_pipeline, embedding_service, explanation_generator, llm_ok = load_llm_components()

# ---------------------------------------------------------------------------
# Panel de estado del sistema
# ---------------------------------------------------------------------------

col_s1, col_s2, col_s3, col_s4 = st.columns(4)

with col_s1:
    st.metric("Modelo base", "XGBoost v1", delta="F1=0.749")
with col_s2:
    st.metric("Recall Glosada", "72.61%", delta="Objetivo: 80%")
with col_s3:
    llm_status = "🟢 Activo" if llm_ok else "🔴 No disponible"
    st.metric("Ollama LLM", llm_status)
with col_s4:
    from src.config import MODEL_DATE
    st.metric("Último entrenamiento", MODEL_DATE)

st.markdown("---")

# ---------------------------------------------------------------------------
# Tabs principales
# ---------------------------------------------------------------------------

tab_pred, tab_monitor, tab_compare, tab_admin = st.tabs([
    "🔍 Predicción",
    "📊 Monitoreo de Drift",
    "🔄 Comparación de Modelos",
    "⚙️ Administración",
])


# ===========================================================================
# TAB 1: PREDICCIÓN
# ===========================================================================

with tab_pred:
    # Cargar caso seleccionado o entrada manual
    if selected_case_id > 0:
        case = DEMO_CASES[selected_case_id - 1]
        st.info(f"{case['icono']} **{case['titulo']}** — {case['descripcion']}")
        features_input = case["features"].copy()
        numero_factura = case["numero_factura"]
        diag_texto = case["diagnostico_texto"] or ""
        cups_texto = case["cups_texto"] or ""
        obs_texto = case["observaciones"] or ""
    else:
        features_input = {}
        numero_factura = "FEV-2025-000001"
        diag_texto = ""
        cups_texto = ""
        obs_texto = ""

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("#### 📝 Datos de la Factura")

        numero_factura = st.text_input("Número de factura", value=numero_factura)

        with st.expander("📊 Features numéricas (48)", expanded=selected_case_id == 0):
            # Mostrar solo las más relevantes en el demo
            key_features = [
                "valor_factura", "tasa_glosa_nit", "tasa_devolucion_nit",
                "dias_hasta_radicacion", "is_extemporanea", "num_items",
                "ratio_autorizacion", "discrepancia_valor", "ratio_valor_vs_nit",
                "num_dx_distintos",
            ]
            edited_features = {}
            for feat in key_features:
                default_val = features_input.get(feat, 0.0)
                edited_features[feat] = st.number_input(
                    feat, value=float(default_val), format="%.4f", key=f"feat_{feat}"
                )
            # Completar con valores del caso para features no mostradas
            for feat in feature_names:
                if feat not in edited_features:
                    edited_features[feat] = features_input.get(feat, 0.0)

        st.markdown("#### 💬 Campos textuales (opcional)")
        diag_texto = st.text_area("Diagnóstico CIE-10", value=diag_texto, height=80)
        cups_texto = st.text_area("Descripción CUPS", value=cups_texto, height=60)
        obs_texto = st.text_area("Observaciones del auditor", value=obs_texto, height=80)

        predict_btn = st.button("🔮 Predecir", type="primary", use_container_width=True)

    with col_result:
        st.markdown("#### 🎯 Resultado de la Predicción")

        if predict_btn:
            with st.spinner("Procesando..."):
                t0 = time.time()

                try:
                    from src.features.engineering import prepare_input
                    from src.explainability.shap_explanations import explain_single_prediction

                    features_df = prepare_input(edited_features, feature_names)
                    result = explain_single_prediction(
                        features_df=features_df,
                        model=model,
                        explainer=explainer,
                        feature_names=feature_names,
                        top_n=top_n_factors,
                    )

                    elapsed_ms = (time.time() - t0) * 1000

                    # Badge de estado
                    estado = result["estado_predicho"]
                    badge_class = f"badge-{estado}"
                    confianza = result["confianza"]

                    st.markdown(
                        f'<div style="text-align:center; margin: 1rem 0;">'
                        f'<span class="{badge_class}" style="font-size:1.4rem; padding:8px 20px;">'
                        f'{estado.upper()}</span></div>',
                        unsafe_allow_html=True,
                    )

                    # Métricas de confianza
                    c1, c2, c3 = st.columns(3)
                    probs = result["probabilidades"]
                    c1.metric("Auditada", f"{probs['auditada']:.1%}")
                    c2.metric("Glosada", f"{probs['glosada']:.1%}")
                    c3.metric("Devuelta", f"{probs['devuelta']:.1%}")

                    # Barra de confianza
                    st.progress(confianza, text=f"Confianza: {confianza:.1%}")

                    # Causal probable
                    if result.get("causal_probable"):
                        st.info(f"📋 Causal probable Res. 2284/2023: **{result['causal_probable']}**")

                    # Acción sugerida
                    st.success(f"💡 {result['accion_sugerida']}")

                    # Factores de riesgo
                    st.markdown("**Factores de riesgo principales:**")
                    for factor in result["factores_riesgo"]:
                        st.markdown(f"• {factor}")

                    # Detalle SHAP técnico (opcional)
                    if show_shap_detail and result.get("factores_detalle"):
                        with st.expander("🔬 Detalle SHAP técnico"):
                            shap_data = [
                                {
                                    "Feature": f.feature,
                                    "SHAP": f"{f.shap_value:+.4f}",
                                    "Valor": f"{f.feature_value:.4f}",
                                }
                                for f in result["factores_detalle"]
                            ]
                            st.dataframe(pd.DataFrame(shap_data), use_container_width=True)

                    # Explicación LLM enriquecida
                    if llm_ok and any([diag_texto, cups_texto, obs_texto]):
                        with st.spinner("Generando explicación con LLM..."):
                            try:
                                shap_values = {f.feature: f.shap_value for f in result["factores_detalle"]}
                                feature_values = {f.feature: f.feature_value for f in result["factores_detalle"]}
                                explicacion = explanation_generator.generate(
                                    prediction=result["clase"],
                                    confidence=confianza,
                                    shap_values=shap_values,
                                    feature_values=feature_values,
                                )
                                st.markdown("**📝 Explicación enriquecida (LLM):**")
                                st.markdown(
                                    f'<div class="explanation-box">{explicacion}</div>',
                                    unsafe_allow_html=True,
                                )
                            except Exception as e:
                                st.warning(f"LLM no disponible: {e}")
                    elif not llm_ok and any([diag_texto, cups_texto, obs_texto]):
                        st.warning("⚠️ Ollama no disponible. Inicia Ollama para explicaciones enriquecidas.")

                    st.caption(f"⏱️ Tiempo de procesamiento: {elapsed_ms:.1f} ms")

                except Exception as e:
                    st.error(f"Error en predicción: {e}")
        else:
            st.markdown(
                """
                <div style="text-align:center; padding:3rem; color:#6B778C;">
                    <div style="font-size:3rem;">🔮</div>
                    <p>Selecciona un caso de demostración o ingresa datos manualmente,<br>
                    luego haz clic en <strong>Predecir</strong>.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ===========================================================================
# TAB 2: MONITOREO DE DRIFT
# ===========================================================================

with tab_monitor:
    st.markdown("#### 📊 Monitoreo de Concept Drift (PSI)")

    col_d1, col_d2 = st.columns([1, 1])

    with col_d1:
        st.markdown("**Population Stability Index (PSI)**")
        st.markdown(
            """
            | PSI | Estado | Acción |
            |-----|--------|--------|
            | < 0.15 | 🟢 Estable | Sin acción |
            | 0.15 – 0.20 | 🟡 Advertencia | Monitorear |
            | > 0.20 | 🔴 Alerta | Reentrenar |
            """
        )

        # Simulación de PSI para demo
        st.markdown("**Simulación de distribución de drift:**")
        drift_level = st.slider("Nivel de drift simulado", 0.0, 0.40, 0.08, 0.01)

        from src.monitoring.drift_detector import check_drift
        drift_status = check_drift(drift_level)

        status_colors = {"stable": "🟢", "warning": "🟡", "alert": "🔴"}
        status_labels = {"stable": "Estable", "warning": "Advertencia", "alert": "Alerta"}

        st.metric(
            "Estado de drift",
            f"{status_colors[drift_status]} {status_labels[drift_status]}",
            delta=f"PSI = {drift_level:.3f}",
        )

        if drift_status == "alert":
            st.error("⚠️ Se recomienda reentrenamiento inmediato del modelo.")
        elif drift_status == "warning":
            st.warning("⚠️ Monitorear de cerca. Considerar reentrenamiento preventivo.")
        else:
            st.success("✅ Distribución estable. No se requiere acción.")

    with col_d2:
        # Gráfico de PSI simulado por feature
        try:
            import plotly.graph_objects as go

            features_demo = [
                "valor_factura", "tasa_glosa_nit", "dias_hasta_radicacion",
                "num_items", "ratio_autorizacion", "discrepancia_valor",
            ]
            psi_values_demo = np.random.RandomState(42).exponential(
                scale=drift_level * 0.8, size=len(features_demo)
            )

            colors = [
                "#DE350B" if v > 0.20 else "#FF8B00" if v > 0.15 else "#00875A"
                for v in psi_values_demo
            ]

            fig = go.Figure(go.Bar(
                x=features_demo,
                y=psi_values_demo,
                marker_color=colors,
                text=[f"{v:.3f}" for v in psi_values_demo],
                textposition="outside",
            ))
            fig.add_hline(y=0.15, line_dash="dash", line_color="#FF8B00",
                         annotation_text="Umbral advertencia (0.15)")
            fig.add_hline(y=0.20, line_dash="dash", line_color="#DE350B",
                         annotation_text="Umbral alerta (0.20)")
            fig.update_layout(
                title="PSI por Feature",
                xaxis_title="Feature",
                yaxis_title="PSI",
                height=350,
                margin=dict(t=40, b=40),
                plot_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.info("Instala plotly para visualizaciones interactivas: `pip install plotly`")


# ===========================================================================
# TAB 3: COMPARACIÓN DE MODELOS
# ===========================================================================

with tab_compare:
    st.markdown("#### 🔄 Comparación: Modelo Base vs Modelo Enriquecido")

    # Métricas comparativas
    metrics_data = {
        "Métrica": ["F1-Macro", "Recall Glosada", "Recall Auditada", "Recall Devuelta", "AUC-ROC"],
        "XGBoost v1 (Base)": [0.7490, 0.7261, 0.7544, 0.7628, 0.9078],
        "XGBoost v2 (Enriquecido)*": [0.7650, 0.7890, 0.7720, 0.7810, 0.9145],
    }
    df_metrics = pd.DataFrame(metrics_data)

    col_m1, col_m2 = st.columns([1, 1])

    with col_m1:
        st.dataframe(
            df_metrics.style.highlight_max(
                subset=["XGBoost v1 (Base)", "XGBoost v2 (Enriquecido)*"],
                color="#E3FCEF",
                axis=1,
            ).format({
                "XGBoost v1 (Base)": "{:.4f}",
                "XGBoost v2 (Enriquecido)*": "{:.4f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("*Valores proyectados con embeddings semánticos (pendiente evaluación real)")

    with col_m2:
        try:
            import plotly.graph_objects as go

            categories = metrics_data["Métrica"]
            v1 = metrics_data["XGBoost v1 (Base)"]
            v2 = metrics_data["XGBoost v2 (Enriquecido)*"]

            fig = go.Figure()
            fig.add_trace(go.Bar(name="v1 Base", x=categories, y=v1,
                                 marker_color="#0052CC"))
            fig.add_trace(go.Bar(name="v2 Enriquecido", x=categories, y=v2,
                                 marker_color="#00875A"))
            fig.update_layout(
                barmode="group",
                title="Comparación de métricas",
                yaxis=dict(range=[0.6, 1.0]),
                height=320,
                margin=dict(t=40, b=40),
                plot_bgcolor="white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.info("Instala plotly para visualizaciones: `pip install plotly`")

    st.markdown("---")
    st.markdown("**Mejora proyectada con enriquecimiento semántico:**")
    col_delta1, col_delta2, col_delta3 = st.columns(3)
    col_delta1.metric("Recall Glosada", "78.90%", delta="+6.29pp vs baseline")
    col_delta2.metric("F1-Macro", "76.50%", delta="+1.60pp vs baseline")
    col_delta3.metric("AUC-ROC", "91.45%", delta="+0.67pp vs baseline")


# ===========================================================================
# TAB 4: ADMINISTRACIÓN
# ===========================================================================

with tab_admin:
    st.markdown("#### ⚙️ Panel de Administración del Modelo")

    col_a1, col_a2 = st.columns([1, 1])

    with col_a1:
        st.markdown("**Estado del sistema de reentrenamiento:**")

        try:
            from src.storage.model_registry import get_active_version, list_model_versions
            active = get_active_version()
            versions = list_model_versions()

            if active:
                st.success(f"✅ Versión activa: **{active['version']}**")
                st.json(active.get("metrics", {}))
            else:
                st.info("ℹ️ No hay versión registrada en el registry.")

            if versions:
                st.markdown(f"**Historial de versiones ({len(versions)} total):**")
                df_versions = pd.DataFrame([
                    {
                        "Versión": v["version"],
                        "Fecha": v["created_at"][:10],
                        "F1": v["metrics"].get("f1_macro", "N/A"),
                        "Activa": "✅" if v["is_active"] else "",
                    }
                    for v in versions[:5]
                ])
                st.dataframe(df_versions, use_container_width=True, hide_index=True)
        except Exception as e:
            st.info(f"Registry no inicializado: {e}")

    with col_a2:
        st.markdown("**Acciones de administración:**")

        if st.button("🔄 Verificar drift ahora", use_container_width=True):
            with st.spinner("Evaluando PSI..."):
                try:
                    from src.monitoring.drift_detector import compute_psi, check_drift
                    from src.config import DATA_DIR

                    train_path = DATA_DIR / "processed" / "20260309_train.parquet"
                    val_path = DATA_DIR / "processed" / "20260309_val.parquet"

                    if train_path.exists() and val_path.exists():
                        ref = pd.read_parquet(train_path)
                        cur = pd.read_parquet(val_path)
                        psi_vals = []
                        for col in feature_names:
                            if col in ref.columns and col in cur.columns:
                                psi_vals.append(compute_psi(ref[col].values, cur[col].values))
                        mean_psi = float(np.mean(psi_vals))
                        status = check_drift(mean_psi)
                        st.metric("PSI medio", f"{mean_psi:.4f}", delta=status)
                    else:
                        st.warning("Archivos parquet no encontrados.")
                except Exception as e:
                    st.error(f"Error: {e}")

        st.markdown("---")

        if st.button("🚀 Activar reentrenamiento manual", type="primary", use_container_width=True):
            st.warning("⚠️ Esta acción reentrena el modelo con datos recientes.")
            st.info("Usa el endpoint `POST /v1/admin/retrain` de la API para ejecutar en producción.")

        if st.button("⏪ Rollback a versión anterior", use_container_width=True):
            st.info("Usa el endpoint `POST /v1/admin/model/rollback` de la API.")

        st.markdown("---")
        st.markdown("**Reporte mensual:**")
        if st.button("📋 Generar reporte de rendimiento", use_container_width=True):
            try:
                from src.monitoring.model_monitor import ModelMonitor
                monitor = ModelMonitor()
                report = monitor.generate_monthly_report()
                if report.get("status") == "sin_datos":
                    st.info("Sin datos de producción registrados aún.")
                else:
                    st.json(report)
            except Exception as e:
                st.error(f"Error generando reporte: {e}")
