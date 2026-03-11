#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para generar resultados de modelos ML formateados en LaTeX
Proyecto: Auditoría Prestacional Savia Salud EPS
Autor: Juan Felipe Cardona Arango

Uso:
    1. Entrena tu modelo y obtén y_true, y_pred, y_pred_proba
    2. Ajusta las variables en la sección CONFIGURACIÓN
    3. Ejecuta: python generar_resultados_latex.py
    4. Copia la salida y pégala en main.tex reemplazando las cajas amarillas
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score
)
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN - AJUSTAR ESTOS VALORES CON TUS DATOS REALES
# ============================================================

# Nombres de los modelos entrenados
NOMBRES_MODELOS = ['Random Forest', 'XGBoost', 'Red Neuronal (MLP)', 'Regresión Logística']

# Resultados de cada modelo (ejemplo con valores placeholder)
# Reemplaza con tus resultados reales del entrenamiento
RESULTADOS_MODELOS = {
    'Random Forest': {
        'accuracy': 0.873,
        'precision_macro': 0.85,
        'recall_macro': 0.84,
        'f1_macro': 0.845
    },
    'XGBoost': {
        'accuracy': 0.891,
        'precision_macro': 0.87,
        'recall_macro': 0.86,
        'f1_macro': 0.865
    },
    'Red Neuronal (MLP)': {
        'accuracy': 0.856,
        'precision_macro': 0.83,
        'recall_macro': 0.82,
        'f1_macro': 0.825
    },
    'Regresión Logística': {
        'accuracy': 0.798,
        'precision_macro': 0.76,
        'recall_macro': 0.75,
        'f1_macro': 0.755
    }
}

# Mejor modelo (ajustar según resultados)
MEJOR_MODELO = 'XGBoost'

# Datos para matriz de confusión (ejemplo)
# Reemplazar con: cm = confusion_matrix(y_test, y_pred)
MATRIZ_CONFUSION = np.array([
    [4500,  320,   80],   # Auditada: predichas correctamente, como Glosa, como Devuelta
    [  280, 1200,   45],  # Glosada: predichas como Auditada, correctamente, como Devuelta
    [   60,   35,  480]   # Devuelta: predichas como Auditada, como Glosa, correctamente
])

CLASES = ['Auditada (A)', 'Glosada (G)', 'Devuelta (D)']

# Feature importance (ejemplo - reemplazar con modelo.feature_importances_)
FEATURE_NAMES = [
    'valor_total_factura', 'num_procedimientos', 'edad_paciente',
    'diversidad_diagnosticos', 'modalidad_pago_evento', 'tiempo_radicacion',
    'historico_glosas_prestador', 'cantidad_medicamentos', 'nivel_complejidad_cups',
    'dia_semana_radicacion', 'valor_por_procedimiento', 'cantidad_consultas',
    'zona_geografica', 'regimen_subsidiado', 'especialidad_alta_costo'
]

FEATURE_IMPORTANCE = np.array([
    0.182, 0.156, 0.134, 0.123, 0.098, 0.087,
    0.076, 0.065, 0.054, 0.043, 0.038, 0.032,
    0.028, 0.021, 0.015
])

# Resultados por clase del mejor modelo
# Reemplazar con classification_report output
RESULTADOS_POR_CLASE = {
    'Auditada (A)': {'precision': 0.93, 'recall': 0.92, 'f1': 0.925, 'support': 4900},
    'Glosada (G)': {'precision': 0.77, 'recall': 0.79, 'f1': 0.780, 'support': 1525},
    'Devuelta (D)': {'precision': 0.79, 'recall': 0.83, 'f1': 0.810, 'support': 575}
}

# Métricas AUC por clase (One-vs-Rest)
AUC_POR_CLASE = {
    'Auditada': 0.96,
    'Glosada': 0.89,
    'Devuelta': 0.91
}

AP_SCORE_GLOSADA = 0.84  # Average Precision para clase desbalanceada

# Validación con auditores (encuesta)
VALIDACION_AUDITORES = {
    'Utilidad práctica del modelo': {'score': 4.2, 'aceptacion': 84},
    'Precisión en casos de alta complejidad': {'score': 3.8, 'aceptacion': 76},
    'Claridad de la clasificación de riesgo': {'score': 4.5, 'aceptacion': 90},
    'Confianza en predicciones "Bajo Riesgo"': {'score': 4.0, 'aceptacion': 80}
}

# ============================================================
# FUNCIONES PARA GENERAR TABLAS LATEX
# ============================================================

def generar_tabla_comparacion_modelos():
    """Genera tabla comparativa de métricas por modelo"""
    print("=" * 70)
    print("TABLA 8.1: Comparación de métricas por algoritmo")
    print("=" * 70)
    print()
    
    latex = r"""\begin{table}[H]
\centering
\caption{Comparación de métricas de desempeño por algoritmo}
\label{tab:resultados_modelos}
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Algoritmo} & \textbf{Accuracy} & \textbf{Precision Macro} & \textbf{Recall Macro} & \textbf{F1-Score Macro} \\
\midrule
"""
    
    for modelo, metricas in RESULTADOS_MODELOS.items():
        acc = f"{metricas['accuracy']*100:.1f}\\%"
        prec = f"{metricas['precision_macro']:.3f}"
        rec = f"{metricas['recall_macro']:.3f}"
        f1 = f"{metricas['f1_macro']:.3f}"
        
        # Destacar el mejor modelo
        if modelo == MEJOR_MODELO:
            acc = f"\\textbf{{{acc}}}"
            prec = f"\\textbf{{{prec}}}"
            rec = f"\\textbf{{{rec}}}"
            f1 = f"\\textbf{{{f1}}}"
        
        latex += f"{modelo} & {acc} & {prec} & {rec} & {f1} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}"""
    
    print(latex)
    print()
    return latex


def generar_tabla_metricas_clase():
    """Genera tabla de métricas por clase"""
    print("=" * 70)
    print("TABLA 8.2: Métricas detalladas por clase")
    print("=" * 70)
    print()
    
    latex = r"""\begin{table}[H]
\centering
\caption{Métricas detalladas por clase de facturación - Modelo: """ + MEJOR_MODELO + r"""}
\label{tab:metricas_clase}
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Clase} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} & \textbf{Soporte (n)} \\
\midrule
"""
    
    for clase, metricas in RESULTADOS_POR_CLASE.items():
        prec = f"{metricas['precision']:.3f}"
        rec = f"{metricas['recall']:.3f}"
        f1 = f"{metricas['f1']:.3f}"
        sup = f"{int(metricas['support']):,}"
        latex += f"{clase} & {prec} & {rec} & {f1} & {sup} \\\\\n"
    
    # Calcular promedios
    precisions = [m['precision'] for m in RESULTADOS_POR_CLASE.values()]
    recalls = [m['recall'] for m in RESULTADOS_POR_CLASE.values()]
    f1s = [m['f1'] for m in RESULTADOS_POR_CLASE.values()]
    supports = [m['support'] for m in RESULTADOS_POR_CLASE.values()]
    total = sum(supports)
    
    macro_prec = np.mean(precisions)
    macro_rec = np.mean(recalls)
    macro_f1 = np.mean(f1s)
    
    weighted_prec = np.average(precisions, weights=supports)
    weighted_rec = np.average(recalls, weights=supports)
    weighted_f1 = np.average(f1s, weights=supports)
    
    latex += r"""\midrule
\textbf{Macro Avg} & """ + f"{macro_prec:.3f} & {macro_rec:.3f} & {macro_f1:.3f} & {total:,}" + r""" \\
\textbf{Weighted Avg} & """ + f"{weighted_prec:.3f} & {weighted_rec:.3f} & {weighted_f1:.3f} & {total:,}" + r""" \\
\bottomrule
\end{tabular}
\end{table}"""
    
    print(latex)
    print()
    return latex


def generar_matriz_confusion_latex():
    """Genera matriz de confusión en formato LaTeX"""
    print("=" * 70)
    print("MATRIZ DE CONFUSIÓN (valores para tabla)")
    print("=" * 70)
    print()
    
    total = MATRIZ_CONFUSION.sum()
    
    latex = r"""\begin{table}[H]
\centering
\caption{Matriz de confusión - Valores absolutos}
\label{tab:matriz_confusion}
\begin{tabular}{@{}l|ccc|c@{}}
\toprule
& \multicolumn{3}{c|}{\textbf{Predicho}} & \\
\textbf{Real} & Auditada & Glosada & Devuelta & \textbf{Total} \\
\midrule
"""
    
    for i, clase in enumerate(['Auditada', 'Glosada', 'Devuelta']):
        fila = MATRIZ_CONFUSION[i]
        latex += f"{clase} & {fila[0]:,} & {fila[1]:,} & {fila[2]:,} & {fila.sum():,} \\\\\n"
    
    latex += r"""\midrule
\textbf{Total} & """ + f"{MATRIZ_CONFUSION[:,0].sum():,} & {MATRIZ_CONFUSION[:,1].sum():,} & {MATRIZ_CONFUSION[:,2].sum():,} & {total:,}" + r""" \\
\bottomrule
\end{tabular}
\end{table}

\vspace{0.5cm}

\begin{table}[H]
\centering
\caption{Matriz de confusión - Porcentajes (normalizada por fila)}
\label{tab:matriz_confusion_norm}
\begin{tabular}{@{}l|ccc@{}}
\toprule
& \multicolumn{3}{c}{\textbf{Predicho}} \\
\textbf{Real} & Auditada & Glosada & Devuelta \\
\midrule
"""
    
    for i, clase in enumerate(['Auditada', 'Glosada', 'Devuelta']):
        fila = MATRIZ_CONFUSION[i]
        fila_norm = fila / fila.sum() * 100
        latex += f"{clase} & {fila_norm[0]:.1f}\\% & {fila_norm[1]:.1f}\\% & {fila_norm[2]:.1f}\\% \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}"""
    
    print(latex)
    print()
    
    # Análisis de errores
    print("--- Análisis de Errores ---")
    fn_glosa = MATRIZ_CONFUSION[1, 0]  # Glosa predicha como Auditada
    fp_glosa = MATRIZ_CONFUSION[0, 1]  # Auditada predicha como Glosa
    fn_devuelta = MATRIZ_CONFUSION[2, 0] + MATRIZ_CONFUSION[2, 1]  # Devuelta predicha como otra cosa
    
    total_glosas = MATRIZ_CONFUSION[1].sum()
    total_auditadas = MATRIZ_CONFUSION[0].sum()
    
    pct_fn_glosa = fn_glosa / total_glosas * 100
    pct_fp_glosa = fp_glosa / total_auditadas * 100
    
    print(f"Falsos Negativos de Glosa: {fn_glosa} ({pct_fn_glosa:.1f}%)")
    print(f"Falsos Positivos de Glosa: {fp_glosa} ({pct_fp_glosa:.1f}%)")
    print()
    
    return latex


def generar_feature_importance():
    """Genera tabla de feature importance"""
    print("=" * 70)
    print("TABLA 8.4: Importancia de Variables")
    print("=" * 70)
    print()
    
    # Ordenar features por importancia
    indices = np.argsort(FEATURE_IMPORTANCE)[::-1]
    
    latex = r"""\begin{table}[H]
\centering
\caption{Ranking de variables predictoras más importantes - Modelo: """ + MEJOR_MODELO + r"""}
\label{tab:top_features}
\begin{tabular}{@{}rlcp{4cm}@{}}
\toprule
\textbf{Rank} & \textbf{Variable} & \textbf{Importancia} & \textbf{Categoría} \\
\midrule
"""
    
    categorias = {
        'valor': 'Financiera',
        'monto': 'Financiera',
        'procedimiento': 'Clínica',
        'cups': 'Clínica',
        'diagnostico': 'Clínica',
        'cie': 'Clínica',
        'edad': 'Demográfica',
        'paciente': 'Demográfica',
        'prestador': 'Operativa',
        'radicacion': 'Temporal',
        'tiempo': 'Temporal',
        'fecha': 'Temporal',
        'historico': 'Comportamental',
        'glosa': 'Comportamental',
        'modalidad': 'Contratual',
        'contrato': 'Contratual',
        'zona': 'Geográfica',
        'regimen': 'Demográfica'
    }
    
    for rank, idx in enumerate(indices[:10], 1):
        var_name = FEATURE_NAMES[idx]
        importance = FEATURE_IMPORTANCE[idx]
        
        # Determinar categoría
        categoria = 'Otra'
        for key, cat in categorias.items():
            if key in var_name.lower():
                categoria = cat
                break
        
        # Formatear nombre para LaTeX
        var_latex = var_name.replace('_', '\\_')
        
        latex += f"{rank} & \\texttt{{{var_latex}}} & {importance:.3f} & {categoria} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}

\vspace{0.5cm}

\textbf{Interpretación de las principales variables:}
\begin{itemize}
    \item \textbf{""" + FEATURE_NAMES[indices[0]].replace('_', ' ') + r""":} Factor más determinante en la predicción del estado de auditoría.
    \item \textbf{""" + FEATURE_NAMES[indices[1]].replace('_', ' ') + r""":} Segunda variable más influyente, relacionada con la complejidad del servicio.
    \item \textbf{""" + FEATURE_NAMES[indices[2]].replace('_', ' ') + r""":} Característica demográfica/clínica relevante para la clasificación.
\end{itemize}"""
    
    print(latex)
    print()
    return latex


def generar_curvas_roc_texto():
    """Genera texto para sección de curvas ROC"""
    print("=" * 70)
    print("CURVAS ROC Y PRECISION-RECALL")
    print("=" * 70)
    print()
    
    latex = r"""\begin{figure}[H]
\centering
% \includegraphics[width=0.8\textwidth]{roc_curvas.png}
\fbox{\parbox{0.7\textwidth}{\centering
\vspace{2cm}
\textbf{[Insertar Curvas ROC]}\\[0.3cm]
Área bajo la curva (AUC) por clase (One-vs-Rest):\\[0.3cm]
\begin{tabular}{lc}
\textbf{Clase} & \textbf{AUC} \\
\midrule
"""
    
    for clase, auc in AUC_POR_CLASE.items():
        latex += f"{clase} & {auc:.3f} \\\\\n"
    
    macro_auc = np.mean(list(AUC_POR_CLASE.values()))
    latex += r"""\midrule
\textbf{Macro Average} & \textbf{""" + f"{macro_auc:.3f}" + r"""} \\
\end{tabular}
\vspace{2cm}
}}
\caption{Curvas ROC - Análisis discriminativo por clase - Modelo: """ + MEJOR_MODELO + r"""}
\label{fig:roc_curves}
\end{figure}

\begin{figure}[H]
\centering
% \includegraphics[width=0.8\textwidth]{precision_recall_curvas.png}
\fbox{\parbox{0.7\textwidth}{\centering
\vspace{2cm}
\textbf{[Insertar Curvas Precision-Recall]}\\[0.3cm]
Especialmente relevante para clase ``Glosada'' (desbalanceada)\\[0.3cm]
\textbf{Average Precision Score (Glosada): """ + f"{AP_SCORE_GLOSADA:.3f}" + r"""}
\vspace{2cm}
}}
\caption{Curvas Precision-Recall por clase}
\label{fig:pr_curves}
\end{figure}"""
    
    print(latex)
    print()
    return latex


def generar_validacion_auditores():
    """Genera tabla de validación con auditores"""
    print("=" * 70)
    print("TABLA: Validación con Auditores")
    print("=" * 70)
    print()
    
    latex = r"""\begin{table}[H]
\centering
\caption{Resultados de validación con auditores médicos (n=15)}
\label{tab:validacion_auditores}
\begin{tabular}{@{}lccl@{}}
\toprule
\textbf{Aspecto Evaluado} & \textbf{Puntuación (1-5)} & \textbf{\% Aceptación} & \textbf{Observaciones} \\
\midrule
"""
    
    for aspecto, datos in VALIDACION_AUDITORES.items():
        score = f"{datos['score']:.1f}/5"
        acept = f"{datos['aceptacion']:.0f}\\%"
        latex += f"{aspecto} & {score} & {acept} & - \\\\\n"
    
    promedio_score = np.mean([d['score'] for d in VALIDACION_AUDITORES.values()])
    promedio_acept = np.mean([d['aceptacion'] for d in VALIDACION_AUDITORES.values()])
    
    latex += r"""\midrule
\textbf{Promedio General} & \textbf{""" + f"{promedio_score:.1f}/5" + r"""} & \textbf{""" + f"{promedio_acept:.0f}\\%" + r"""} & \\
\bottomrule
\end{tabular}
\end{table}"""
    
    print(latex)
    print()
    return latex


def generar_resumen_ejecutivo():
    """Genera resumen ejecutivo con valores reales"""
    print("=" * 70)
    print("RESUMEN EJECUTIVO DE RESULTADOS")
    print("=" * 70)
    print()
    
    mejor = RESULTADOS_MODELOS[MEJOR_MODELO]
    baseline = 1/3  # Azar para 3 clases
    mejora = (mejor['accuracy'] - baseline) / baseline * 100
    
    # Top 3 features
    indices = np.argsort(FEATURE_IMPORTANCE)[::-1][:3]
    top_features = [FEATURE_NAMES[i].replace('_', ' ').title() for i in indices]
    
    promedio_acept = np.mean([d['aceptacion'] for d in VALIDACION_AUDITORES.values()])
    
    prec_str = f"{mejor['accuracy']*100:.1f}"
    f1_str = f"{mejor['f1_macro']:.3f}"
    mejora_str = f"{mejora:.0f}"
    acept_str = f"{promedio_acept:.0f}"
    
    latex = (
        "\\begin{tcolorbox}[colback=blue!5, colframe=minsalud, fonttitle=\\bfseries, title=Resultados Principales]\n"
        "\\begin{itemize}\n"
        f"    \\item \\textbf{{Mejor algoritmo:}} {MEJOR_MODELO}\n"
        f"    \\item \\textbf{{Accuracy alcanzado:}} {prec_str}\\%\n"
        f"    \\item \\textbf{{F1-Score Macro:}} {f1_str}\n"
        f"    \\item \\textbf{{Mejora vs. baseline (azar):}} +{mejora_str}\\%\n"
        f"    \\item \\textbf{{Variables más importantes:}} {', '.join(top_features)}\n"
        f"    \\item \\textbf{{Aceptación por auditores:}} {acept_str}\\%\n"
        "\\end{itemize}\n"
        "\\end{tcolorbox}"
    )
    
    print(latex)
    print()
    return latex


def exportar_a_archivo():
    """Exporta todo el contenido a un archivo .tex"""
    import sys
    from io import StringIO
    
    # Redirigir stdout para capturar todo
    old_stdout = sys.stdout
    sys.stdout = buffer = StringIO()
    
    print(r"% ============================================================")
    print(r"% RESULTADOS GENERADOS AUTOMÁTICAMENTE")
    print(r"% Fecha: " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'))
    print(r"% ============================================================")
    print()
    
    generar_resumen_ejecutivo()
    generar_tabla_comparacion_modelos()
    generar_tabla_metricas_clase()
    generar_matriz_confusion_latex()
    generar_feature_importance()
    generar_curvas_roc_texto()
    generar_validacion_auditores()
    
    sys.stdout = old_stdout
    contenido = buffer.getvalue()
    
    with open('resultados_generados.tex', 'w', encoding='utf-8') as f:
        f.write(contenido)
    
    print("[OK] Resultados exportados a: resultados_generados.tex")
    print()


# ============================================================
# FUNCIÓN AUXILIAR: Calcular métricas desde datos reales
# ============================================================

def calcular_metricas_reales(y_true, y_pred, y_pred_proba=None):
    """
    Calcula todas las métricas a partir de arrays de numpy/pandas.
    
    Parámetros:
    -----------
    y_true : array-like
        Etiquetas reales (0=Auditada, 1=Glosada, 2=Devuelta)
    y_pred : array-like
        Predicciones del modelo
    y_pred_proba : array-like, optional
        Probabilidades predichas para cada clase (para ROC AUC)
    
    Retorna:
    --------
    dict : Diccionario con todas las métricas calculadas
    """
    
    metricas = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred)
    }
    
    if y_pred_proba is not None:
        # Calcular AUC One-vs-Rest
        from sklearn.preprocessing import label_binarize
        classes = np.unique(y_true)
        y_true_bin = label_binarize(y_true, classes=classes)
        
        auc_scores = {}
        for i, cls in enumerate(classes):
            try:
                auc_scores[f'Clase_{cls}'] = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
            except:
                auc_scores[f'Clase_{cls}'] = 0.0
        
        metricas['auc_por_clase'] = auc_scores
    
    return metricas


# ============================================================
# EJECUCIÓN PRINCIPAL
# ============================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print(" GENERADOR DE RESULTADOS EN FORMATO LATEX")
    print(" Proyecto: Auditoría Prestacional Savia Salud EPS")
    print("=" * 70 + "\n")
    
    print("INSTRUCCIONES:")
    print("1. Este script genera código LaTeX listo para copiar y pegar")
    print("2. Los valores mostrados son EJEMPLOS - debes reemplazarlos")
    print("3. Busca las secciones marcadas con [AJUSTAR] en el código")
    print("4. Para usar tus datos reales, modifica las variables en la")
    print("   sección CONFIGURACIÓN al inicio de este archivo")
    print()
    print("=" * 70)
    print()
    
    # Generar todas las tablas
    generar_resumen_ejecutivo()
    generar_tabla_comparacion_modelos()
    generar_tabla_metricas_clase()
    generar_matriz_confusion_latex()
    generar_feature_importance()
    generar_curvas_roc_texto()
    generar_validacion_auditores()
    
    # Exportar a archivo
    exportar_a_archivo()
    
    print("=" * 70)
    print("PROCESO COMPLETADO")
    print("=" * 70)
    print()
    print("Próximos pasos:")
    print("1. Copia el código LaTeX generado arriba")
    print("2. Reemplaza las secciones correspondientes en main.tex")
    print("3. Elimina las cajas amarillas (colorbox) una vez insertados los valores")
    print("4. Compila el documento para verificar la integración")
    print()
