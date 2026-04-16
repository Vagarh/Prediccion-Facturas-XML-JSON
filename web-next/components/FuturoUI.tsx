"use client";
import { motion } from "framer-motion";
import { Sparkles, Monitor, ArrowRight } from "lucide-react";
import FadeIn from "./FadeIn";

const roadmap = [
  { label: "Integración NLP",  desc: "Embeddings sobre texto libre de diagnósticos para enriquecer predicciones",         color: "#8B5CF6" },
  { label: "UI de Auditor",    desc: "Dashboard en tiempo real con cola priorizada por riesgo y explicaciones SHAP",       color: "#00D4FF" },
  { label: "Feedback loop",    desc: "El auditor corrige predicciones y el modelo aprende de cada revisión",               color: "#00FF88" },
  { label: "API producción",   desc: "Integración directa con el sistema de radicación de facturas de la EPS",             color: "#FFB300" },
];

export default function FuturoUI() {
  return (
    <section id="futuro" className="py-28 px-6">
      <div className="max-w-5xl mx-auto">
        <FadeIn className="text-center mb-16">
          <span className="section-label">Capítulo 06 · Trabajo Futuro</span>
          <h2
            style={{
              fontFamily: "var(--font-body)",
              fontSize: "clamp(2.2rem, 5vw, 3.5rem)",
              fontWeight: 700,
              lineHeight: 1.1,
              letterSpacing: "-0.02em",
              marginBottom: "1rem",
            }}
          >
            La interfaz que
            <br />
            <span className="grad-text">está por llegar.</span>
          </h2>
          <p style={{ color: "#8892B0", fontSize: "1.05rem", maxWidth: 540, margin: "0 auto", lineHeight: 1.75 }}>
            El modelo es el núcleo. La siguiente fase convierte las predicciones en
            una herramienta de trabajo para auditores médicos.
          </p>
        </FadeIn>

        <div className="grid md:grid-cols-2 gap-10 items-center mb-12">
          {/* UI Mockup */}
          <FadeIn>
            <motion.div
              className="rounded-2xl p-px overflow-hidden"
              style={{ background: "linear-gradient(135deg, rgba(0,212,255,0.2), rgba(0,102,255,0.15), rgba(139,92,246,0.1))" }}
              whileHover={{ scale: 1.01 }}
              transition={{ duration: 0.3 }}
            >
              <div className="rounded-2xl overflow-hidden" style={{ background: "#0C0D12" }}>
                {/* Browser bar */}
                <div className="flex items-center gap-2 px-4 py-3" style={{ borderBottom: "1px solid #1C1E2E" }}>
                  {["#FF4466", "#FFB300", "#00FF88"].map((c) => (
                    <div key={c} className="w-3 h-3 rounded-full" style={{ background: c + "70" }} />
                  ))}
                  <div
                    className="flex-1 mx-4 rounded-md py-1 px-3"
                    style={{ background: "#10111A", border: "1px solid #1C1E2E" }}
                  >
                    <span style={{ fontFamily: "var(--font-code)", fontSize: "0.6rem", color: "#3D4461" }}>
                      savia-audit.dashboard.internal
                    </span>
                  </div>
                </div>

                {/* Dashboard */}
                <div className="p-5">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2">
                      <Monitor size={13} color="#00D4FF" />
                      <span style={{ fontFamily: "var(--font-code)", fontSize: "0.68rem", color: "#8892B0" }}>
                        Cola de Auditoría · Hoy
                      </span>
                    </div>
                    <span style={{ fontFamily: "var(--font-code)", fontSize: "0.62rem", color: "#00FF88" }}>● En vivo</span>
                  </div>

                  <div className="space-y-2 mb-4">
                    {[
                      { id: "FAC-2024-09871", risk: "ALTA",  score: 0.91, class_: "Glosada",  color: "#FF4466" },
                      { id: "FAC-2024-09872", risk: "MEDIA", score: 0.68, class_: "Auditada", color: "#FFB300" },
                      { id: "FAC-2024-09873", risk: "BAJA",  score: 0.23, class_: "Auditada", color: "#00FF88" },
                    ].map((row) => (
                      <div
                        key={row.id}
                        className="flex items-center justify-between rounded-xl px-3 py-2.5"
                        style={{ background: "#10111A", border: `1px solid ${row.color}18` }}
                      >
                        <span style={{ fontFamily: "var(--font-code)", fontSize: "0.65rem", color: "#8892B0" }}>
                          {row.id}
                        </span>
                        <div className="flex items-center gap-3">
                          <span
                            style={{
                              fontSize: "0.58rem",
                              color: row.color,
                              background: `${row.color}12`,
                              padding: "2px 7px",
                              borderRadius: 3,
                              fontFamily: "var(--font-code)",
                              letterSpacing: "0.05em",
                            }}
                          >
                            {row.risk}
                          </span>
                          <span style={{ fontFamily: "var(--font-code)", fontSize: "0.65rem", color: row.color }}>
                            {row.score}
                          </span>
                          <span style={{ fontSize: "0.6rem", color: "#8892B0" }}>{row.class_}</span>
                        </div>
                      </div>
                    ))}
                  </div>

                  <div
                    className="rounded-xl px-3 py-2 text-center"
                    style={{ background: "#10111A", border: "1px solid #1C1E2E" }}
                  >
                    <span style={{ fontFamily: "var(--font-code)", fontSize: "0.58rem", color: "#3D4461" }}>
                      — mockup · interfaz en desarrollo —
                    </span>
                  </div>
                </div>
              </div>
            </motion.div>
          </FadeIn>

          {/* Roadmap */}
          <FadeIn delay={0.12}>
            <div className="space-y-4">
              <div className="flex items-center gap-2 mb-6">
                <Sparkles size={15} color="#8B5CF6" />
                <span style={{ fontFamily: "var(--font-body)", fontSize: "1rem", fontWeight: 700 }}>
                  Hoja de Ruta
                </span>
              </div>
              {roadmap.map((r, i) => (
                <motion.div
                  key={r.label}
                  initial={{ opacity: 0, x: 20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.4, delay: i * 0.08 }}
                  className="rounded-xl p-4 flex items-start gap-4"
                  style={{ background: "#10111A", border: "1px solid #1C1E2E" }}
                >
                  <div
                    className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5"
                    style={{ background: `${r.color}12`, border: `1px solid ${r.color}30`, color: r.color }}
                  >
                    <ArrowRight size={12} />
                  </div>
                  <div>
                    <p style={{ fontWeight: 700, fontSize: "0.88rem", marginBottom: "0.25rem", color: r.color }}>
                      {r.label}
                    </p>
                    <p style={{ color: "#8892B0", fontSize: "0.78rem", lineHeight: 1.6 }}>{r.desc}</p>
                  </div>
                </motion.div>
              ))}
            </div>
          </FadeIn>
        </div>
      </div>
    </section>
  );
}
