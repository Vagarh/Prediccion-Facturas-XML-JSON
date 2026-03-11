#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de compilación LaTeX con generación de índices
Proyecto: Auditoría Prestacional Savia Salud EPS

Uso:
    python compile.py
"""

import os
import subprocess
import sys
from pathlib import Path

def limpiar_temporales():
    """Elimina archivos temporales de LaTeX"""
    extensiones = ['.aux', '.toc', '.lof', '.lot', '.out', '.log', 
                   '.synctex.gz', '.bbl', '.blg', '.idx', '.ind', '.ilg']
    
    eliminados = 0
    for ext in extensiones:
        for archivo in Path('.').glob(f'*{ext}'):
            try:
                archivo.unlink()
                eliminados += 1
                print(f"  Eliminado: {archivo.name}")
            except:
                pass
    
    print(f"\n[OK] {eliminados} archivos temporales eliminados\n")
    return eliminados

def compilar_latex(archivo="main.tex", pasadas=3):
    """Compila el documento LaTeX múltiples veces"""
    
    comando = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        archivo
    ]
    
    print(f"[i] Iniciando compilación de: {archivo}")
    print(f"[i] Se realizarán {pasadas} pasadas\n")
    
    for i in range(1, pasadas + 1):
        print(f"[PASADA {i}/{pasadas}] Compilando...")
        
        try:
            resultado = subprocess.run(
                comando,
                capture_output=True,
                text=True,
                timeout=180
            )
            
            # Verificar si hay errores reales (no advertencias de MiKTeX)
            output = resultado.stdout + resultado.stderr
            
            # Errores reales de LaTeX
            errores_reales = []
            for linea in output.split('\n'):
                if any(x in linea for x in ['! LaTeX Error:', '! Undefined', '! Missing', '! Emergency stop']):
                    errores_reales.append(linea)
            
            if errores_reales:
                print(f"[!] ERRORES ENCONTRADOS:")
                for err in errores_reales[:10]:
                    print(f"    {err}")
                return False
            
            # Si hay advertencia de MiKTeX pero no errores, continuar
            if "So far, you have not checked for MiKTeX updates" in output:
                print(f"    (Advertencia de MiKTeX ignorada)")
            
            print(f"[OK] Pasada {i} completada\n")
            
        except subprocess.TimeoutExpired:
            print(f"[!] TIMEOUT en pasada {i}")
            return False
        except FileNotFoundError:
            print(f"[!] ERROR: pdflatex no encontrado")
            print(f"    Asegúrate de tener LaTeX instalado:")
            print(f"    - Windows: MiKTeX o TeX Live")
            print(f"    - Linux: sudo apt-get install texlive-full")
            print(f"    - Mac: brew install --cask mactex")
            return False
    
    return True

def verificar_pdf(archivo="main.pdf"):
    """Verifica que el PDF se generó correctamente"""
    pdf_path = Path(archivo)
    
    if pdf_path.exists():
        tamano = pdf_path.stat().st_size / 1024  # KB
        print(f"\n[OK] PDF generado exitosamente:")
        print(f"    Archivo: {pdf_path.absolute()}")
        print(f"    Tamaño: {tamano:.1f} KB")
        
        # Verificar archivos auxiliares
        aux_files = ['main.toc', 'main.lof', 'main.lot']
        print(f"\n[i] Archivos de índice generados:")
        for aux in aux_files:
            if Path(aux).exists():
                print(f"    [OK] {aux}")
            else:
                print(f"    [X] {aux} (no encontrado)")
        
        return True
    else:
        print(f"\n[X] ERROR: No se generó el archivo {archivo}")
        return False

def main():
    print("=" * 60)
    print(" COMPILADOR LATEX - INDICES Y REFERENCIAS")
    print("=" * 60)
    print()
    
    # Limpiar temporales
    print("[1/4] LIMPIANDO ARCHIVOS TEMPORALES...")
    limpiar_temporales()
    
    # Compilar
    print("[2/4] COMPILANDO DOCUMENTO...")
    if not compilar_latex("main.tex", pasadas=3):
        print("\n[X] Compilación fallida")
        sys.exit(1)
    
    # Verificar
    print("[3/4] VERIFICANDO PDF...")
    if not verificar_pdf("main.pdf"):
        sys.exit(1)
    
    print("\n[4/4] PROCESO COMPLETADO")
    print("=" * 60)
    print(" El PDF debería tener:")
    print("  - Tabla de contenidos")
    print("  - Lista de tablas")
    print("  - Lista de figuras")
    print("  - Todas las referencias cruzadas")
    print("=" * 60)
    print("\nAbre main.pdf para verificar los cambios")

if __name__ == "__main__":
    main()
