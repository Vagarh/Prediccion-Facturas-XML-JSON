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

## 2. Bibliografía final: 27 entradas, todas con identificador comprobado

**DOI resuelto y metadatos en Crossref:**
- [ ] Breiman 2001: https://doi.org/10.1023/A:1010933404324
- [ ] Chen & Guestrin 2016: https://doi.org/10.1145/2939672.2939785
- [ ] Akiba et al. 2019 (Optuna): https://doi.org/10.1145/3292500.3330701
- [ ] Li et al. 2008: https://doi.org/10.1007/s10729-007-9045-4
- [ ] Kumar et al. 2010: https://doi.org/10.1145/1835804.1835816
- [ ] Joudaki et al. 2015: https://doi.org/10.5539/gjhs.v7n1p194
- [ ] Herland et al. 2018: https://doi.org/10.1186/s40537-018-0138-3
- [ ] Herland et al. 2020: https://doi.org/10.1007/s10729-018-9460-8
- [ ] du Preez et al. 2025: https://doi.org/10.1016/j.artmed.2024.103061
- [ ] Rajkomar et al. 2018: https://doi.org/10.1038/s41746-018-0029-1
- [ ] Rajpurkar et al. 2022: https://doi.org/10.1038/s41591-021-01614-0
- [ ] Gama et al. 2014: https://doi.org/10.1145/2523813
- [ ] Lu et al. 2019: https://doi.org/10.1109/TKDE.2018.2876857
- [ ] Breck et al. 2017: https://doi.org/10.1109/BigData.2017.8258038

**URL con respuesta HTTP 200:**
- [ ] Ke et al. 2017 (LightGBM): papers.nips.cc/paper/6907
- [ ] Lundberg & Lee 2017 (SHAP): papers.nips.cc/paper/7062
- [ ] Kim et al. 2020 (Deep Claim): arxiv.org/abs/2007.06229
- [ ] Wirth & Hipp 2000 (CRISP-DM): PDF en cs.unibo.it
- [ ] Sullivan 2019 (definición HIMSS): healthcareitnews.com. La página bloquea bots; se verificó en su copia de Wayback del 2026-09-21
- [ ] Ley 1966/2019 (Función Pública) · Res. 2275/2023 · Res. 1884/2024 · Res. 1557/2023 ·
      Res. 3374/2000 · Res. 510/2022 (PDF en minsalud.gov.co) · Circular 007/2025
      (bogotajuridica.gov.co) · Res. DIAN 000165/2023 (normograma DIAN, título oficial copiado)

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
