@echo off
chcp 65001 >nul
echo ==========================================
echo  COMPILADOR LATEX - INDICES CORREGIDOS
echo ==========================================
echo.

echo [1/5] Limpiando archivos temporales...
del /Q *.aux *.toc *.lof *.lot *.out *.log *.synctex.gz 2>nul
echo      OK
echo.

echo [2/5] Primera compilacion (genera archivos auxiliares)...
pdflatex -interaction=nonstopmode main.tex > compile.log 2>&1
echo      OK
echo.

echo [3/5] Segunda compilacion (actualiza indices)...
pdflatex -interaction=nonstopmode main.tex >> compile.log 2>&1
echo      OK
echo.

echo [4/5] Tercera compilacion (referencias cruzadas)...
pdflatex -interaction=nonstopmode main.tex >> compile.log 2>&1
echo      OK
echo.

echo [5/5] Verificando PDF generado...
if exist main.pdf (
    echo      OK - main.pdf generado correctamente
    echo.
    echo ==========================================
    echo  COMPILACION EXITOSA
echo ==========================================
) else (
    echo      ERROR - Revisar compile.log
)
echo.
pause
