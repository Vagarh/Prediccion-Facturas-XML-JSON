"use client";
import { motion } from "framer-motion";
import { CheckCircle2, TrendingUp, Target } from "lucide-react";
import FadeIn from "./FadeIn";

const bars = [
  { label: "Recall Glosada — prueba (feb-mar 2026)",              v1: 72.1, v2: 86.4, color: "#00FF88", key: "recall_test" },
  { label: "Recall Glosada — 2 meses después (out-of-time)",      v1: 53.9, v2: 86.0, color: "#00FF88", key: "recall_oot" },
  { label: "F1-macro — prueba",                                    v1: 74.3, v2: 75.8, color: "#00D4FF", key: "f1_test" },
  { label: "F1-macro — 2 meses después (out-of-time)",             v1: 59.4, v2: 72.4, color: "#00D4FF", key: "f1_oot" },
];

export default function Resultados() {
  return (
    <section id="impacto" className="py-28 px-6">
      <div className="max-w-5xl mx-auto">

        <FadeIn className="text-center mb-16">
          <span className="section-label">Resultados</span>
          <h2 style={{ fontFamily: "var(--font-body)", fontSize: "clamp(2.2rem, 5vw, 3.5rem)", fontWeight: 700, lineHeight: 1.1, letterSpacing: "-0.022em", marginBottom: "1.2rem" }}>
            Más de 8 de cada 10 glosas,
            <br />
            <span className="grad-text">detectadas antes de pagar — y se mantiene.</span>
          </h2>
          <p style={{ color: "#8892B0", fontSize: "1.05rem", maxWidth: 580, margin: "0 auto", lineHeight: 1.78 }}>
            La meta era detectar al menos el 80% de las facturas glosadas.
            El modelo v3 llega al{" "}
            <strong style={{ color: "#00FF88" }}>86.4%</strong> en prueba y sigue en{" "}
            <strong style={{ color: "#00FF88" }}>86.0%</strong> dos meses después, sin la caída que
            sufría la versión anterior fuera de su ventana de entrenamiento.
          </p>
        </FadeIn>

        {/* Logro principal */}
        <FadeIn>
          <div className="rounded-2xl p-10 mb-8 relative overflow-hidden"
            style={{ background: "#10111A", border: "1px solid rgba(0,255,136,0.2)" }}>
            <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: "2px", background: "linear-gradient(90deg, transparent, #00FF8870, transparent)" }} />
            <div className="flex flex-col md:flex-row items-center gap-10">
              <div className="text-center md:text-left flex-shrink-0">
                <div className="flex items-center gap-2 justify-center md:justify-start mb-2">
                  <Target size={15} color="#00FF88" />
                  <span style={{ fontFamily: "var(--font-code)", fontSize: "0.65rem", color: "#00FF88", letterSpacing: "0.1em" }}>META SUPERADA</span>
                </div>
                <p style={{ fontFamily: "var(--font-code)", fontSize: "clamp(3.5rem, 9vw, 6rem)", fontWeight: 700, color: "#00FF88", lineHeight: 1, letterSpacing: "-0.04em" }}>
                  86.4%
                </p>
                <p style={{ color: "#8892B0", fontSize: "0.9rem", marginTop: "0.5rem" }}>
                  Recall Glosada{" "}
                  <span style={{ color: "#3D4461", fontFamily: "var(--font-code)", fontSize: "0.75rem" }}>meta ≥ 80%</span>
                </p>
              </div>
              <div className="flex-1 space-y-3 w-full">
                {[
                  { text: "Entrenado con 1,08 millones de facturas reales con división temporal — sin filtrar datos futuros.", color: "#00FF88" },
                  { text: "Validado contra ground truth real de cm_auditoria_motivos_glosas de la BD de producción.", color: "#00D4FF" },
                  { text: "Cada predicción incluye SHAP top-5 — el auditor sabe exactamente por qué el modelo lo marcó.", color: "#0066FF" },
                ].map((item, i) => (
                  <div key={i} className="rounded-xl p-4 flex items-start gap-3"
                    style={{ background: "#0C0D12", border: `1px solid ${item.color}15` }}>
                    <CheckCircle2 size={14} color={item.color} style={{ flexShrink: 0, marginTop: 2 }} />
                    <p style={{ color: "#C8D0E8", fontSize: "0.85rem", lineHeight: 1.65 }}>{item.text}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </FadeIn>

        {/* Comparación v1 vs v2 */}
        <FadeIn>
          <div className="rounded-2xl p-8 mb-5" style={{ background: "#10111A", border: "1px solid #1C1E2E" }}>
            <p style={{ fontFamily: "var(--font-code)", fontSize: "0.65rem", color: "#3D4461", marginBottom: "1.75rem", letterSpacing: "0.1em", textTransform: "uppercase" }}>
              Modelo anterior (datos hasta jul-2025) vs. LightGBM v3 (datos frescos hasta may-2026)
            </p>
            <div className="space-y-7">
              {bars.map((b) => (
                <div key={b.key}>
                  <div className="flex items-center justify-between mb-2">
                    <span style={{ fontSize: "0.78rem", color: "#8892B0" }}>{b.label}</span>
                    <span style={{ fontFamily: "var(--font-code)", fontSize: "0.75rem", color: b.color, fontWeight: 600 }}>
                      {b.v2.toFixed(1)}%
                      {b.v2 > b.v1 && <TrendingUp size={11} style={{ display: "inline", marginLeft: 4, verticalAlign: "middle" }} />}
                    </span>
                  </div>
                  <div className="relative h-2 rounded-full overflow-hidden" style={{ background: "#0C0D12" }}>
                    <motion.div initial={{ width: 0 }} whileInView={{ width: `${b.v1}%` }} viewport={{ once: true }}
                      transition={{ duration: 0.9, delay: 0.2, ease: "easeOut" }}
                      className="absolute inset-y-0 left-0 rounded-full" style={{ background: "#1C1E2E" }} />
                    <motion.div initial={{ width: 0 }} whileInView={{ width: `${b.v2}%` }} viewport={{ once: true }}
                      transition={{ duration: 1, delay: 0.4, ease: "easeOut" }}
                      className="absolute inset-y-0 left-0 rounded-full"
                      style={{ background: `linear-gradient(90deg, ${b.color}60, ${b.color})` }} />
                  </div>
                  <div className="flex gap-5 mt-1.5">
                    <span style={{ fontSize: "0.62rem", color: "#3D4461", fontFamily: "var(--font-code)" }}>anterior: {b.v1}%</span>
                    <span style={{ fontSize: "0.62rem", color: b.color, fontFamily: "var(--font-code)" }}>v3: {b.v2}% {b.v2 > b.v1 ? "↑" : "→"}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </FadeIn>

        <FadeIn>
          <p style={{ fontSize: "0.65rem", color: "#3D4461", textAlign: "center", fontFamily: "var(--font-code)", letterSpacing: "0.05em" }}>
            1.084.637 facturas · División temporal mar-2024 → may-2026 · Validación out-of-time abr–may 2026 · RANDOM_STATE=42
          </p>
        </FadeIn>

      </div>
    </section>
  );
}
