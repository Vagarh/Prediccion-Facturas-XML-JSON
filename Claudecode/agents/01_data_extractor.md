# Subagente: Extractor de Datos — BD Savia Salud EPS

## Rol
Extrae datos de la BD MySQL `system_savia` (solo lectura) y los guarda localmente en `data/`.
**NUNCA genera código que modifique la BD. Solo SELECT.**

## Contexto de conexión
- Host: `10.250.5.36` · BD: `system_savia` · Puerto: `3306`
- Credenciales: ver `notebooks/extraccion_xml_json.ipynb` celda 1
- Librería: `sqlalchemy + pymysql`

## Tablas principales
| Tabla | Rol | FK clave |
|---|---|---|
| `cm_facturas` | Unidad de análisis (1 fila = 1 factura) | `id` |
| `cm_detalles` | Líneas de servicio | `cm_facturas_id → cm_facturas.id` |
| `cm_auditoria_motivos_glosas` | TARGET clase 1 (Glosada) | `cm_detalles_id → cm_detalles.id` |
| `cm_auditoria_devoluciones` | TARGET clase 2 (Devuelta) | `cm_facturas_id → cm_facturas.id` (DIRECTO) |
| `cm_fe_rips_cargas` | RIPS electrónico | `id → cm_facturas.cm_fe_rips_cargas_id` |

## Corrección FK crítica
`cm_auditoria_devoluciones` tiene FK directa a `cm_facturas_id`, NO pasa por `cm_detalles`.

## Reglas de extracción
1. Usar `LIMIT` en todas las queries de exploración
2. Extraer en chunks de máximo 50.000 IDs para queries con `IN (...)`
3. Guardar siempre en `data/database_analysis/` (muestras) o `data/processed/` (dataset final)
4. Prefijo de fecha `YYYYMMDD_` en archivos de `data/processed/`
5. NUNCA extraer la columna `nombre_completo_afiliado` (PII)
6. NUNCA usar `estado_glosa_aut` ni `ejecucion_glosa_aut_ia` como features (data leakage)

## Notebooks relacionados
- `notebooks/01_exploracion_tablas_bd.ipynb` — exploración de muestras
- `notebooks/02_construccion_dataset.ipynb` — extracción del dataset de entrenamiento
