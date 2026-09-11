"use client";
import { motion } from "framer-motion";
import { AlertTriangle, ArrowRight } from "lucide-react";
import { degradation, drift, history, pct, psiColor } from "@/lib/metrics";
import FadeIn from "@/components/FadeIn";

const STATUS_LABEL: Record<string, string> = {
  alert: "Alerta",
  warning: "Advertencia",
  stable: "Estable",
};

export default function DriftPanel() {
  const maxPsi = Math.max(...drift.top.map((f) => f.psi));
  const v3 = history.find((h) => h.version === "v3");

  return (
    <section id="drift" className="py-16">
      <FadeIn>
        <span className="section-label">04 · Deriva y versiones</span>
        <h2 style={{ fontFamily: "var(--font-body)", fontSize: "clamp(1.6rem,3.5vw,2.3rem)", fontWeight: 700, letterSpacing: "-0.02em", marginBottom: "0.8rem" }}>
          Deriva del concepto, medida
        </h2>
        <p style={{ color: "#8892B0", fontSize: "0.95rem", lineHeight: 1.75, maxWidth: 760, marginBottom: "2rem" }}>
          El Índice de Estabilidad Poblacional (PSI) compara la distribución de cada variable
          entre el dataset con el que se entrenó el modelo original y el actual. No es una
          impresión cualitativa: es la magnitud exacta del envejecimiento de los datos.
        </p>
      </FadeIn>

      <div className="grid md:grid-cols-3 gap-5 mb-6">
        <FadeIn>
          <div className="rounded-2xl p-6 h-full" style={{ background: "#10111A", border: "1px solid rgba(255,68,102,0.25)" }}>
            <div className="flex items-center gap-2 mb-3">
              <AlertTriangle size={14} color="#FF4466" />
              <span style={{ fontFamily: "var(--font-code)", fontSize: "0.62rem", color: "#FF4466", letterSpacing: "0.08em", textTransform: "uppercase" }}>
                Estado: {STATUS_LABEL[drift.status] ?? drift.status}
              </span>
            </div>
            <p style={{ fontFamily: "var(--font-code)", fontSize: "2.4rem", fontWeight: 700, color: "#FF4466", lineHeight: 1, letterSpacing: "-0.03em" }}>
              {drift.meanPsi.toFixed(3)}
            </p>
            <p style={{ fontSize: "0.78rem", color: "#8892B0", marginTop: "0.5rem", lineHeight: 1.6 }}>
              PSI medio sobre {drift.nFeatures} variables. Por encima de 0.25 se considera
              cambio severo de población.
            </p>
            <div className="flex gap-4 mt-4 pt-4" style={{ borderTop: "1px solid #1C1E2E" }}>
              {Object.entries(drift.counts).map(([status, count]) => (
                <div key={status}>
                  <p style={{ fontFamily: "var(--font-code)", fontSize: "1.1rem", fontWeight: 700, color: psiColor(status), lineHeight: 1 }}>{count}</p>
                  <p style={{ fontSize: "0.62rem", color: "#3D4461", marginTop: 3 }}>{STATUS_LABEL[status] ?? status}</p>
                </div>
              ))}
            </div>
          </div>
        </FadeIn>

        <FadeIn delay={0.08} className="md:col-span-2">
          <div className="rounded-2xl p-6 h-full" style={{ background: "#10111A", border: "1px solid #1C1E2E" }}>
            <p style={{ fontFamily: "var(--font-code)", fontSize: "0.62rem", color: "#3D4461", letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: "1.25rem" }}>
              Variables con mayor deriva · {drift.referencePrefix} vs. {drift.currentPrefix}
            </p>
            <div className="space-y-3">
              {drift.top.map((f, i) => (
                <div key={f.name}>
                  <div className="flex items-baseline justify-between mb-1 gap-3">
                    <span style={{ fontFamily: "var(--font-code)", fontSize: "0.72rem", color: "#C8D0E8", wordBreak: "break-all" }}>{f.name}</span>
                    <span style={{ fontFamily: "var(--font-code)", fontSize: "0.72rem", color: psiColor(f.status), whiteSpace: "nowrap" }}>
                      PSI {f.psi.toFixed(3)}
                    </span>
                  </div>
                  <div className="rounded-full" style={{ height: 5, background: "#0C0D12" }}>
                    <motion.div
                      initial={{ width: 0 }}
                      whileInView={{ width: `${(f.psi / maxPsi) * 100}%` }}
                      viewport={{ once: true }}
                      transition={{ duration: 0.8, delay: i * 0.07, ease: "easeOut" }}
                      className="rounded-full"
                      style={{ height: 5, background: `linear-gradient(90deg, ${psiColor(f.status)}60, ${psiColor(f.status)})` }}
                    />
                  </div>
                </div>
              ))}
            </div>
            <p style={{ fontSize: "0.75rem", color: "#8892B0", lineHeight: 1.7, marginTop: "1.25rem", paddingTop: "1rem", borderTop: "1px solid #1C1E2E" }}>
              La variable más desplazada es el conteo histórico de facturas por prestador: la
              red de prestadores y su volumen cambiaron sustancialmente. <code style={{ color: "#00D4FF" }}>tiene_fev</code> se
              movió porque la facturación electrónica se universalizó y dejó de discriminar.
            </p>
          </div>
        </FadeIn>
      </div>

      <FadeIn>
        <div className="rounded-2xl p-6" style={{ background: "#10111A", border: "1px solid #1C1E2E" }}>
          <p style={{ fontFamily: "var(--font-code)", fontSize: "0.62rem", color: "#3D4461", letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: "1.5rem" }}>
            Historial de versiones del modelo
          </p>
          <div className="grid md:grid-cols-3 gap-4">
            {history.map((h, i) => {
              const isCurrent = h.version === "v3";
              const accent = isCurrent ? "#00FF88" : h.version === "v2" ? "#FFB300" : "#3D4461";
              return (
                <motion.div
                  key={h.version}
                  initial={{ opacity: 0, y: 16 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.45, delay: i * 0.1 }}
                  className="rounded-xl p-5"
                  style={{ background: "#0C0D12", border: `1px solid ${accent}${isCurrent ? "45" : "25"}` }}
                >
                  <div className="flex items-center justify-between mb-3">
                    <span style={{ fontFamily: "var(--font-code)", fontSize: "0.95rem", fontWeight: 700, color: accent }}>{h.version}</span>
                    <span style={{ fontFamily: "var(--font-code)", fontSize: "0.65rem", color: "#3D4461" }}>{h.date}</span>
                  </div>
                  <p style={{ fontSize: "0.82rem", fontWeight: 600, color: "#F0F2FF", marginBottom: 2 }}>{h.algorithm}</p>
                  <p style={{ fontSize: "0.68rem", color: "#3D4461", marginBottom: "1rem" }}>{h.datasetLabel}</p>

                  <div className="grid grid-cols-2 gap-3 mb-3">
                    <div>
                      <p style={{ fontFamily: "var(--font-code)", fontSize: "1rem", color: "#C8D0E8", lineHeight: 1 }}>{h.f1Macro.toFixed(4)}</p>
                      <p style={{ fontSize: "0.6rem", color: "#3D4461", marginTop: 3 }}>F1-macro</p>
                    </div>
                    <div>
                      <p style={{ fontFamily: "var(--font-code)", fontSize: "1rem", color: accent, lineHeight: 1 }}>{pct(h.recallGlosada)}</p>
                      <p style={{ fontSize: "0.6rem", color: "#3D4461", marginTop: 3 }}>Recall Glosada</p>
                    </div>
                  </div>

                  {h.ootRecallGlosada != null && (
                    <div className="flex items-center gap-2 rounded-lg px-3 py-2 mb-3" style={{ background: "rgba(0,255,136,0.06)", border: "1px solid rgba(0,255,136,0.18)" }}>
                      <ArrowRight size={11} color="#00FF88" style={{ flexShrink: 0 }} />
                      <span style={{ fontSize: "0.7rem", color: "#C8D0E8" }}>
                        Dos meses después: <strong style={{ color: "#00FF88", fontFamily: "var(--font-code)" }}>{pct(h.ootRecallGlosada)}</strong>
                      </span>
                    </div>
                  )}

                  <p style={{ fontSize: "0.72rem", color: "#8892B0", lineHeight: 1.6 }}>{h.note}</p>
                </motion.div>
              );
            })}
          </div>

          {v3?.ootRecallGlosada != null && (
            <div className="rounded-xl p-5 mt-5" style={{ background: "rgba(0,212,255,0.04)", border: "1px solid rgba(0,212,255,0.15)" }}>
              <p style={{ fontSize: "0.85rem", color: "#C8D0E8", lineHeight: 1.75 }}>
                El modelo anterior perdía <strong style={{ color: "#FF4466" }}>25.2%</strong> de su
                Recall Glosada al salir de la ventana de entrenamiento. El campeón actual pierde{" "}
                <strong style={{ color: "#00FF88" }}>{pct(degradation(v3.recallGlosada, v3.ootRecallGlosada), 1)}</strong>.
                {v3.reproducedBaseline && (
                  <>
                    {" "}Reproducir el modelo viejo sobre los datos frescos da F1{" "}
                    <span style={{ fontFamily: "var(--font-code)", color: "#8892B0" }}>{v3.reproducedBaseline.f1Macro.toFixed(4)}</span>:
                    prueba de que la mejora vino de actualizar los datos, no de cambiar de algoritmo.
                  </>
                )}
              </p>
            </div>
          )}
        </div>
      </FadeIn>
    </section>
  );
}
