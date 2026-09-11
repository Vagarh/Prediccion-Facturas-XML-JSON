"use client";
import { motion } from "framer-motion";
import { Check, Crown, Minus } from "lucide-react";
import { champion, degradation, duration, metrics, models, pct } from "@/lib/metrics";
import FadeIn from "@/components/FadeIn";

const FLOOR_TEST = 0.8;
const FLOOR_OOT = 0.75;

/** Barra horizontal comparativa dentro de una celda de la tabla. */
function Bar({ value, color }: { value: number; color: string }) {
  return (
    <div className="relative rounded-full mt-1" style={{ height: 3, background: "#0C0D12" }}>
      <motion.div
        initial={{ width: 0 }}
        whileInView={{ width: `${value * 100}%` }}
        viewport={{ once: true }}
        transition={{ duration: 0.8, ease: "easeOut" }}
        className="absolute inset-y-0 left-0 rounded-full"
        style={{ background: color }}
      />
    </div>
  );
}

export default function SuitePanel() {
  const ranking = metrics.championSelection.ranking;
  const top = [...ranking].sort((a, b) => b.f1MacroTest - a.f1MacroTest);
  const runnerUp = top.find((r) => r.model !== champion.id && r.eligible);

  return (
    <section id="suite" className="py-16">
      <FadeIn>
        <span className="section-label">02 · Entrenamientos</span>
        <h2 style={{ fontFamily: "var(--font-body)", fontSize: "clamp(1.6rem,3.5vw,2.3rem)", fontWeight: 700, letterSpacing: "-0.02em", marginBottom: "0.8rem" }}>
          Suite comparativa de modelos
        </h2>
        <p style={{ color: "#8892B0", fontSize: "0.95rem", lineHeight: 1.75, maxWidth: 760, marginBottom: "2rem" }}>
          Seis algoritmos entrenados sobre exactamente los mismos datos y particiones, con
          búsqueda de hiperparámetros mediante Optuna donde aplica. Cada uno se evalúa dos
          veces: en el conjunto de prueba y, dos meses después, fuera de la ventana de
          entrenamiento. La segunda columna es la que dice la verdad.
        </p>
      </FadeIn>

      <FadeIn>
        <div className="rounded-2xl overflow-hidden mb-6" style={{ background: "#10111A", border: "1px solid #1C1E2E" }}>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", minWidth: 720, borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ background: "#0C0D12" }}>
                  <th style={th}>Modelo</th>
                  <th style={{ ...th, textAlign: "center" }} colSpan={2}>
                    <span style={{ color: "#8892B0" }}>Prueba · feb–mar 2026</span>
                  </th>
                  <th style={{ ...th, textAlign: "center", borderLeft: "1px solid #1C1E2E" }} colSpan={2}>
                    <span style={{ color: "#00D4FF" }}>Out-of-Time · abr–may 2026</span>
                  </th>
                  <th style={{ ...th, textAlign: "right" }}>Entren.</th>
                </tr>
                <tr style={{ background: "#0C0D12" }}>
                  <th style={thSub}></th>
                  <th style={{ ...thSub, textAlign: "right" }}>F1-macro</th>
                  <th style={{ ...thSub, textAlign: "right" }}>Recall Glosada</th>
                  <th style={{ ...thSub, textAlign: "right", borderLeft: "1px solid #1C1E2E" }}>F1-macro</th>
                  <th style={{ ...thSub, textAlign: "right" }}>Recall Glosada</th>
                  <th style={thSub}></th>
                </tr>
              </thead>
              <tbody>
                {models.map((m) => {
                  const isChampion = m.id === champion.id;
                  return (
                    <tr
                      key={m.id}
                      style={{
                        borderTop: "1px solid #1C1E2E",
                        background: isChampion ? "rgba(0,255,136,0.05)" : "transparent",
                      }}
                    >
                      <td style={td}>
                        <div className="flex items-center gap-2">
                          {isChampion && <Crown size={13} color="#00FF88" style={{ flexShrink: 0 }} />}
                          <div>
                            <p style={{ fontSize: "0.82rem", fontWeight: isChampion ? 700 : 500, color: isChampion ? "#00FF88" : "#F0F2FF" }}>
                              {m.label}
                            </p>
                            <p style={{ fontFamily: "var(--font-code)", fontSize: "0.62rem", color: "#3D4461" }}>
                              {m.family} · {m.tuning}
                              {!m.eligible && <span style={{ color: "#FF4466", marginLeft: 6 }}>no elegible</span>}
                            </p>
                          </div>
                        </div>
                      </td>
                      <td style={tdNum}>
                        {m.test.f1Macro.toFixed(4)}
                        <Bar value={m.test.f1Macro} color="#0066FF" />
                      </td>
                      <td style={tdNum}>
                        {m.test.recallGlosada.toFixed(4)}
                        <Bar value={m.test.recallGlosada} color={m.test.recallGlosada >= FLOOR_TEST ? "#00FF88" : "#FF4466"} />
                      </td>
                      <td style={{ ...tdNum, borderLeft: "1px solid #1C1E2E" }}>
                        {m.oot.f1Macro.toFixed(4)}
                        <Bar value={m.oot.f1Macro} color="#00D4FF" />
                      </td>
                      <td style={tdNum}>
                        {m.oot.recallGlosada.toFixed(4)}
                        <Bar value={m.oot.recallGlosada} color={m.oot.recallGlosada >= FLOOR_OOT ? "#00FF88" : "#FF4466"} />
                      </td>
                      <td style={{ ...tdNum, color: "#3D4461" }}>{duration(m.trainSeconds)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </FadeIn>

      <div className="grid md:grid-cols-2 gap-5">
        <FadeIn>
          <div className="rounded-2xl p-6 h-full" style={{ background: "#10111A", border: "1px solid rgba(0,255,136,0.2)" }}>
            <p style={{ fontFamily: "var(--font-code)", fontSize: "0.65rem", color: "#00FF88", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: "1rem" }}>
              Regla de selección del campeón
            </p>
            <ol className="space-y-3" style={{ counterReset: "step" }}>
              {[
                `Filtro duro: Recall Glosada ≥ ${pct(FLOOR_TEST, 0)} en prueba y ≥ ${pct(FLOOR_OOT, 0)} fuera de muestra.`,
                "Entre los que pasan, se toma el mayor F1-macro de prueba.",
                "Si hay empate técnico (< 0.01 de diferencia), desempata el Recall Glosada fuera de muestra — la métrica primaria del proyecto, medida en el escenario más honesto.",
              ].map((text, i) => (
                <li key={i} className="flex gap-3">
                  <span style={{ fontFamily: "var(--font-code)", fontSize: "0.7rem", color: "#00FF88", flexShrink: 0, marginTop: 2 }}>{i + 1}.</span>
                  <span style={{ fontSize: "0.82rem", color: "#8892B0", lineHeight: 1.65 }}>{text}</span>
                </li>
              ))}
            </ol>
            <div className="flex flex-wrap gap-4 mt-5 pt-4" style={{ borderTop: "1px solid #1C1E2E" }}>
              <span className="flex items-center gap-2" style={{ fontSize: "0.72rem", color: "#8892B0" }}>
                <Check size={12} color="#00FF88" /> {metrics.championSelection.ranking.filter((r) => r.eligible).length} elegibles
              </span>
              <span className="flex items-center gap-2" style={{ fontSize: "0.72rem", color: "#8892B0" }}>
                <Minus size={12} color="#FF4466" /> {metrics.championSelection.ranking.filter((r) => !r.eligible).length} descartados por el filtro
              </span>
            </div>
          </div>
        </FadeIn>

        <FadeIn delay={0.1}>
          <div className="rounded-2xl p-6 h-full" style={{ background: "#10111A", border: "1px solid #1C1E2E" }}>
            <p style={{ fontFamily: "var(--font-code)", fontSize: "0.65rem", color: "#FFB300", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: "1rem" }}>
              {metrics.championSelection.tieBreak ? "Se resolvió por desempate" : "Ganador por F1-macro"}
            </p>
            {runnerUp && (
              <>
                <p style={{ fontSize: "0.85rem", color: "#C8D0E8", lineHeight: 1.7 }}>
                  <strong style={{ color: "#F0F2FF" }}>{runnerUp.label}</strong> obtuvo el F1-macro más
                  alto ({runnerUp.f1MacroTest.toFixed(4)}), pero la diferencia con{" "}
                  <strong style={{ color: "#00FF88" }}>{champion.label}</strong> es de solo{" "}
                  <span style={{ fontFamily: "var(--font-code)", color: "#FFB300" }}>
                    {Math.abs(runnerUp.f1MacroTest - champion.test.f1Macro).toFixed(4)}
                  </span>
                  , dentro del margen de ruido de muestreo.
                </p>
                <div className="grid grid-cols-2 gap-3 mt-4">
                  {[
                    { label: runnerUp.label, rg: runnerUp.recallGlosadaOot, color: "#3D4461" },
                    { label: champion.label, rg: champion.oot.recallGlosada, color: "#00FF88" },
                  ].map((x) => (
                    <div key={x.label} className="rounded-xl p-4" style={{ background: "#0C0D12", border: `1px solid ${x.color}30` }}>
                      <p style={{ fontSize: "0.68rem", color: "#8892B0", marginBottom: 4, lineHeight: 1.3 }}>{x.label}</p>
                      <p style={{ fontFamily: "var(--font-code)", fontSize: "1.35rem", fontWeight: 700, color: x.color === "#3D4461" ? "#8892B0" : x.color, lineHeight: 1 }}>
                        {pct(x.rg)}
                      </p>
                      <p style={{ fontSize: "0.62rem", color: "#3D4461", marginTop: 4 }}>Recall Glosada · OOT</p>
                    </div>
                  ))}
                </div>
                <p style={{ fontSize: "0.78rem", color: "#8892B0", lineHeight: 1.7, marginTop: "1rem" }}>
                  Al desempatar por la métrica que de verdad importa al negocio, gana{" "}
                  {champion.label}: detecta{" "}
                  {((champion.oot.recallGlosada - runnerUp.recallGlosadaOot) * 100).toFixed(1)} puntos
                  porcentuales más de glosas dos meses después, con una degradación
                  prueba→OOT de solo {pct(degradation(champion.test.recallGlosada, champion.oot.recallGlosada), 1)}.
                </p>
              </>
            )}
          </div>
        </FadeIn>
      </div>
    </section>
  );
}

const th: React.CSSProperties = {
  padding: "12px 14px 4px",
  textAlign: "left",
  fontFamily: "var(--font-code)",
  fontSize: "0.62rem",
  letterSpacing: "0.08em",
  textTransform: "uppercase",
  color: "#3D4461",
  fontWeight: 600,
};
const thSub: React.CSSProperties = {
  padding: "0 14px 10px",
  textAlign: "left",
  fontFamily: "var(--font-code)",
  fontSize: "0.6rem",
  color: "#3D4461",
  fontWeight: 400,
};
const td: React.CSSProperties = { padding: "12px 14px", verticalAlign: "middle" };
const tdNum: React.CSSProperties = {
  padding: "12px 14px",
  textAlign: "right",
  fontFamily: "var(--font-code)",
  fontSize: "0.78rem",
  color: "#C8D0E8",
  verticalAlign: "middle",
  minWidth: 108,
};
