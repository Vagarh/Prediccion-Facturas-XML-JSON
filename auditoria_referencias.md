# Auditoría de referencias: respuesta al concepto del asesor (O1–O6)

Fecha: 2026-09-23. **Criterio aplicado:** una referencia se queda solo si tiene un DOI que
resuelve en doi.org, con metadatos que coinciden en Crossref, o una URL que responde con el
documento correcto. Lo que no cumple, se elimina. El asesor pide que **la auditoría la hagas
tú mismo**: abre cada enlace y marca la casilla.

## 1. Eliminadas

| Referencia | Motivo |
|---|---|
| Chan et al. 2022 | No existe: su DOI 10.1177/14604582221089012 da **404** |
| Joshi & Yesha 2020 | No existe en IEEE Access ni en ningún índice |
| González-Hernández et al. 2021 | No existe en la Rev. Mex. Ing. Biomédica |
| Sadiq et al. 2017 | No existe con ese título |
| Brause et al. 1999 (Tabla 4.6) | Existe, pero trata de tarjetas de crédito, no de salud |
| Ortega et al. 2016 (Tabla 4.6) | No localizada |
| OMS 2019 | No localizada; la definición citada era de HIMSS |
| HIMSS 2023 | No localizada con ese título |
| HIMSS (página web, versión intermedia) | URL no accesible (403, sin copia en Wayback) → reemplazada por Sullivan 2019 |
| ISO 13606 | No se citaba y la página oficial no fue accesible |

## 2. Bibliografía final: 28 entradas, todas con identificador comprobado

**DOI resuelto y metadatos en Crossref:**
- [x] Breiman 2001: https://doi.org/10.1023/A:1010933404324
- [x] Chen & Guestrin 2016: https://doi.org/10.1145/2939672.2939785
- [x] Akiba et al. 2019 (Optuna): https://doi.org/10.1145/3292500.3330701
- [x] Li et al. 2008: https://doi.org/10.1007/s10729-007-9045-4
- [x] Kumar et al. 2010: https://doi.org/10.1145/1835804.1835816
- [x] Joudaki et al. 2015: https://doi.org/10.5539/gjhs.v7n1p194
- [x] Herland et al. 2018: https://doi.org/10.1186/s40537-018-0138-3
- [x] Herland et al. 2020: https://doi.org/10.1007/s10729-018-9460-8
- [x] du Preez et al. 2025: https://doi.org/10.1016/j.artmed.2024.103061
- [x] Rajkomar et al. 2018: https://doi.org/10.1038/s41746-018-0029-1
- [x] Rajpurkar et al. 2022: https://doi.org/10.1038/s41591-021-01614-0
- [x] Gama et al. 2014: https://doi.org/10.1145/2523813
- [x] Lu et al. 2019: https://doi.org/10.1109/TKDE.2018.2876857
- [x] Breck et al. 2017: https://doi.org/10.1109/BigData.2017.8258038

- [ ] Schröer et al. 2021 (CRISP-DM, añadida 2026-10-08): https://doi.org/10.1016/j.procs.2021.01.199

**URL (revisión del 2026-10-08):**
- [ x] Ke et al. 2017 (LightGBM): el servidor de NeurIPS rechaza conexiones → copia archivada verificada (título y 8 autores):
      https://web.archive.org/web/20200901123540/https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree
- [x ] Lundberg & Lee 2017 (SHAP): versión de acceso en arXiv: https://arxiv.org/abs/1705.07874
- [x] Kim et al. 2020 (Deep Claim): https://arxiv.org/abs/2007.06229
- [x] Wirth & Hipp 2000 (CRISP-DM): el PDF es correcto (verifiqué título, autores y resumen), pero el servidor de la universidad es lento.
      Se mantiene y se añade como respaldo Schröer et al. 2021 (con DOI).
      https://cs.unibo.it/~danilo.montesi/CBD/Beatriz/10.1.1.198.5133.pdf
- [ x] Sullivan 2019 (HIMSS): copia archivada: https://web.archive.org/web/20260921033351/https://www.healthcareitnews.com/news/himss-writes-new-definition-interoperability
- [ x] Ley 1966/2019: **se mantiene** https://www.funcionpublica.gov.co/eva/gestornormativo/norma.php?i=97190
      (la URL que anotaste corresponde a la Res. 2275, no a la Ley 1966)
- [ x] Res. 2275/2023: https://www.minsalud.gov.co/Normatividad_Nuevo/Resoluci%C3%B3n%20No%202275%20de%202023.pdf
- [ x] Res. 1884/2024: https://www.minsalud.gov.co/Normatividad_Nuevo/Resoluci%C3%B3n%20No%201884%20de%202024.pdf
- [ x] Res. 1557/2023: https://www.minsalud.gov.co/Normatividad_Nuevo/Resoluci%C3%B3n%20No%201557%20de%202023.pdf
- [ x] Res. 3374/2000 (la misma que anotaste): https://www.minsalud.gov.co/sites/rid/Lists/BibliotecaDigital/RIDE/DE/DIJ/Resoluci%C3%B3n_3374_de_2000.pdf
- [ x] Res. 510/2022 (**tu URL**; título oficial copiado del PDF): https://www.minsalud.gov.co/Normatividad_Nuevo/Resoluci%C3%B3n%20No.%20510%20de%202022.pdf
- [ x] Circular Conjunta 007/2025 (**tu URL**; fecha 3-mar-2025 y asunto copiados del PDF): https://www.minsalud.gov.co/sites/rid/Lists/BibliotecaDigital/RIDE/DE/DIJ/circular-conjunta-007-de-2025.pdf
- [ x] Res. DIAN 000165/2023 (**tu URL**, PDF oficial firmado): https://www.dian.gov.co/normatividad/Normatividad/Resoluci%C3%B3n%20000165%20de%2001-11-2023.pdf

## 3. Afirmaciones que debes confirmar tú

- [ok ] **Sec. 4.8, párrafo 1:** describe la búsqueda (bases de datos y términos). Ajústala a como buscaste tú.
- [ ok] **Sec. 4.8.2, Kim et al.:** "cerca de 2,9 millones de reclamaciones de dos sistemas de salud de EE. UU.".
- [ok ] **Sec. 4.8.2, Kumar et al.:** que el paper discute la deriva del concepto y el aprendizaje activo.
- [ok ] **Sec. 4.2.1, definición de interoperabilidad:** es una traducción propia de la definición de HIMSS publicada en Sullivan 2019.
- [ok ] **Sec. 11.5:** el texto sobre el origen probable de las citas falsas.

## 4. Mapa de cambios frente al concepto

| Punto | Cambio |
|---|---|
| Título | Título propuesto por el asesor (ES/EN) |
| O1 | Este archivo. Todas las entradas tienen DOI/URL comprobados |
| O2 | Tabla 4.6 rehecha con 6 fuentes verificadas y sin cifras de desempeño ajenas |
| O3 | 4.8 → "Revisión Narrativa", con la búsqueda descrita |
| O4 | 8.2: sin cifras atribuidas; la deriva se sustenta en Gama 2014, Lu 2019 y Kumar 2010 |
| O5 | Antecedentes de ML en salud: de 6 (4 inexistentes) a 9 verificadas; "primer estudio" atenuado |
| O6 | Nueva sección 11.5 (hallazgo + protocolo de 6 reglas, incluido el DOI/URL obligatorio) |
| R4 | Tono promocional retirado de la introducción y la justificación |
