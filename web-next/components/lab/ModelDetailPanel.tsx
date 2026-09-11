"use client";
import { useState } from "react";
import { motion } from "framer-motion";
import { CLASS_COLORS, champion, metrics, models, num, pct } from "@/lib/metrics";
import type { SplitMetrics } from "@/lib/metrics";
import FadeIn from "@/components/FadeIn";

type SplitKey = "test" | "oot";

const SPLIT_LABEL: Record<SplitKey, string> = {
  test: "Prueba · feb–mar 2026",
  oot: "Out-of-Time · abr–may 2026",
};

/** Matriz de confusión normalizada por fila (clase real). */
function ConfusionMatrix({ data }: { data: SplitMetrics }) {
  const classes = metrics.classNames;
  const rowTotals = data.confusionMatrix.map((row) => row.reduce((a, b) => a + b, 0));

  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ borderCollapse: "collapse", minWidth: 380 }}>
        <thead>
          <tr>
            <th style={{ ...cellHead, textAlign: "left" }}>Real \ Predicho</th>
            {classes.map((c) => (
              <th key={c} style={{ ...cellHead, color: CLASS_COLORS[c] }}>{c}</th>
            ))}
            <th style={{ ...cellHead, color: "#3D4461" }}>Recall</th>
          </tr>
        </thead>
        <tbody>
          {data.confusionMatrix.map((row, i) => (
            <tr key={classes[i]}>
              <th style={{ ...cellHead, textAlign: "left", color: CLASS_COLORS[classes[i]] }}>{classes[i]}</th>
              {row.map((value, j) => {
                const share = rowTotals[i] ? value / rowTotals[i] : 0;
                const correct = i === j;
                return (
                  <td key={j} style={{ padding: 3 }}>
                    <motion.div
                      initial={{ opacity: 0, scale: 0.9 }}
                      whileInView={{ opacity: 1, scale: 1 }}
                      viewport={{ once: true }}
                      transition={{ duration: 0.3, delay: (i * 3 + j) * 0.03 }}
                      className="rounded-md flex flex-col items-center justify-center"
                      style={{
                        minWidth: 92,
                        padding: "10px 6px",
                        background: correct
                          ? `rgba(0,255,136,${0.06 + share * 0.28})`
                          : `rgba(255,68,102,${0.04 + share * 0.3})`,
                        border: `1px solid ${correct ? "rgba(0,255,136,0.25)" : "rgba(255,68,102,0.18)"}`,
                      }}
                    >
                      <span style={{ fontFamily: "var(--font-code)", fontSize: "0.9rem", fontWeight: 700, color: correct ? "#00FF88" : "#FF8FA3", lineHeight: 1 }}>
                        {pct(share, 1)}
                      </span>
                      <span style={{ fontFamily: "var(--font-code)", fontSize: "0.6rem", color: "#3D4461", marginTop: 3 }}>{num(value)}</span>
                    </motion.div>
                  </td>
                );
              })}
              <td style={{ padding: "3px 10px", textAlign: "right", fontFamily: "var(--font-code)", fontSize: "0.78rem", color: "#C8D0E8" }}>
                {pct(data.recallPerClass[classes[i]])}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function ModelDetailPanel() {
  const [selected, setSelected] = useState<string>(champion.id);
  const [split, setSplit] = useState<SplitKey>("test");
  const model = models.find((m) => m.id === selected) ?? models[0];
  const data = model[split];
  const params = Object.entries(model.bestParams);

  return (
    <section id="detalle" className="py-16">
      <FadeIn>
        <span className="section-label">03 · Diagnóstico</span>
        <h2 style={{ fontFamily: "var(--font-body)", fontSize: "clamp(1.6rem,3.5vw,2.3rem)", fontWeight: 700, letterSpacing: "-0.02em", marginBottom: "0.8rem" }}>
          Matriz de confusión e hiperparámetros
        </h2>
        <p style={{ color: "#8892B0", fontSize: "0.95rem", lineHeight: 1.75, maxWidth: 760, marginBottom: "1.75rem" }}>
          Un promedio esconde en qué se equivoca el modelo. Selecciona un algoritmo y una
          ventana temporal para ver dónde exactamente se confunde y con qué configuración
          fue entrenado.
        </p>
      </FadeIn>

      <FadeIn>
        <div className="flex flex-wrap gap-2 mb-4">
          {models.map((m) => (
            <button
              key={m.id}
              onClick={() => setSelected(m.id)}
              style={{
                padding: "7px 14px",
                borderRadius: 8,
                fontSize: "0.76rem",
                fontFamily: "var(--font-code)",
                cursor: "pointer",
                transition: "all .18s",
                background: m.id === selected ? "rgba(0,212,255,0.1)" : "#10111A",
                border: `1px solid ${m.id === selected ? "rgba(0,212,255,0.45)" : "#1C1E2E"}`,
                color: m.id === selected ? "#00D4FF" : "#8892B0",
              }}
            >
              {m.label}
              {m.id === champion.id && <span style={{ color: "#00FF88", marginLeft: 6 }}>★</span>}
            </button>
          ))}
        </div>
      </FadeIn>

      <FadeIn>
        <div className="rounded-2xl p-6" style={{ background: "#10111A", border: "1px solid #1C1E2E" }}>
          <div className="flex flex-wrap items-center justify-between gap-3 mb-5">
            <div>
              <p style={{ fontSize: "1rem", fontWeight: 700, color: "#F0F2FF" }}>{model.label}</p>
              <p style={{ fontFamily: "var(--font-code)", fontSize: "0.68rem", color: "#3D4461" }}>
                {num(data.nSamples)} facturas · AUC-ROC {data.aucRocOvr.toFixed(4)} · accuracy {pct(data.accuracy)}
              </p>
            </div>
            <div className="flex gap-1 rounded-lg p-1" style={{ background: "#0C0D12", border: "1px solid #1C1E2E" }}>
              {(Object.keys(SPLIT_LABEL) as SplitKey[]).map((k) => (
                <button
                  key={k}
                  onClick={() => setSplit(k)}
                  style={{
                    padding: "6px 12px",
                    borderRadius: 6,
                    fontSize: "0.72rem",
                    fontFamily: "var(--font-code)",
                    cursor: "pointer",
                    border: "none",
                    background: split === k ? "rgba(0,212,255,0.12)" : "transparent",
                    color: split === k ? "#00D4FF" : "#8892B0",
                  }}
                >
                  {SPLIT_LABEL[k]}
                </button>
              ))}
            </div>
          </div>

          <div className="grid lg:grid-cols-2 gap-6">
            <div>
              <p style={{ fontFamily: "var(--font-code)", fontSize: "0.62rem", color: "#3D4461", letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: "0.9rem" }}>
                Matriz de confusión · % por clase real
              </p>
              <ConfusionMatrix data={data} />
            </div>

            <div>
              <p style={{ fontFamily: "var(--font-code)", fontSize: "0.62rem", color: "#3D4461", letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: "0.9rem" }}>
                Precisión / Recall / F1 por clase
              </p>
              <div className="space-y-3 mb-6">
                {metrics.classNames.map((c) => {
                  const s = data.perClass[c];
                  if (!s) return null;
                  return (
                    <div key={c} className="rounded-lg p-3" style={{ background: "#0C0D12", border: `1px solid ${CLASS_COLORS[c]}22` }}>
                      <div className="flex items-center justify-between mb-2">
                        <span style={{ fontSize: "0.78rem", fontWeight: 600, color: CLASS_COLORS[c] }}>{c}</span>
                        <span style={{ fontFamily: "var(--font-code)", fontSize: "0.62rem", color: "#3D4461" }}>n={num(s.support)}</span>
                      </div>
                      <div className="grid grid-cols-3 gap-2">
                        {([["Precisión", s.precision], ["Recall", s.recall], ["F1", s.f1]] as const).map(([label, v]) => (
                          <div key={label}>
                            <p style={{ fontFamily: "var(--font-code)", fontSize: "0.82rem", color: "#C8D0E8" }}>{v.toFixed(4)}</p>
                            <p style={{ fontSize: "0.6rem", color: "#3D4461" }}>{label}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>

              <p style={{ fontFamily: "var(--font-code)", fontSize: "0.62rem", color: "#3D4461", letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: "0.9rem" }}>
                {params.length ? "Hiperparámetros hallados por Optuna" : "Configuración"}
              </p>
              {params.length ? (
                <div className="rounded-lg p-4" style={{ background: "#0C0D12", border: "1px solid #1C1E2E" }}>
                  <div className="grid sm:grid-cols-2 gap-x-5 gap-y-2">
                    {params.map(([k, v]) => (
                      <div key={k} className="flex items-baseline justify-between gap-3">
                        <span style={{ fontFamily: "var(--font-code)", fontSize: "0.68rem", color: "#8892B0", wordBreak: "break-all" }}>{k}</span>
                        <span style={{ fontFamily: "var(--font-code)", fontSize: "0.7rem", color: "#00D4FF", whiteSpace: "nowrap" }}>
                          {typeof v === "number" && !Number.isInteger(v) ? v.toFixed(4) : String(v)}
                        </span>
                      </div>
                    ))}
                  </div>
                  <p style={{ fontSize: "0.7rem", color: "#3D4461", lineHeight: 1.6, marginTop: "0.9rem", paddingTop: "0.75rem", borderTop: "1px solid #1C1E2E" }}>
                    <span style={{ color: "#8892B0" }}>w_glosada</span> es el peso de muestra de la clase
                    Glosada, buscado por Optuna junto al resto. Sustituyó al balanceo automático,
                    que hundía el Recall ahora que Glosada es la clase mayoritaria.
                  </p>
                </div>
              ) : (
                <div className="rounded-lg p-4" style={{ background: "#0C0D12", border: "1px solid #1C1E2E" }}>
                  <p style={{ fontSize: "0.78rem", color: "#8892B0", lineHeight: 1.65 }}>
                    Este modelo no pasó por búsqueda de hiperparámetros: se entrenó con la
                    configuración de referencia para servir de línea base comparable.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </FadeIn>
    </section>
  );
}

const cellHead: React.CSSProperties = {
  padding: "6px 10px",
  fontFamily: "var(--font-code)",
  fontSize: "0.65rem",
  fontWeight: 600,
  color: "#8892B0",
  textAlign: "center",
  whiteSpace: "nowrap",
};
