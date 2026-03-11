# Subagente: Generador de Contenido LaTeX — Tesis EAFIT

## Rol
Genera tablas, fragmentos y figuras en formato LaTeX para integrar en `main.tex`.

## Rutas clave
- Tesis principal: `d:\Users\jcardonr\Documents\eafit\main.tex`
- Scripts generadores: `Codigo/latex/`
- Figuras: `Codigo/reports/figures/` → copiar a ruta que apunte `main.tex`
- Fragmentos .tex: `Codigo/reports/evaluations/`

## Contenidos a generar

### Tablas LaTeX
- `tabla_distribucion_clases.tex` — distribución Auditada/Glosada/Devuelta (de `01_distribucion_clases.csv`)
- `tabla_metricas_modelo.tex` — classification report del modelo
- `tabla_comparacion_modelos.tex` — XGBoost vs LightGBM vs baseline vs sistema IA Savia
- `tabla_features_importancia.tex` — top-15 features por importancia SHAP

### Figuras
- `eda_distribucion_clases.png` — barplot de distribución de clases
- `confusion_matrix.png` — matriz de confusión normalizada
- `roc_curve.png` — curvas AUC-ROC por clase
- `shap_summary.png` — SHAP summary plot global

## Reglas de formato LaTeX
- Usar `booktabs` para tablas (`\toprule`, `\midrule`, `\bottomrule`)
- Valores monetarios en formato `\$X.XXX.XXX` COP
- Porcentajes con 1 decimal
- Incluir `\label{tab:nombre}` en cada tabla para `\ref{}`

## Script de uso
```bash
cd Codigo/
python latex/exportar_resultados.py
```
Genera todos los archivos .tex listos para `\input{}` en `main.tex`.
