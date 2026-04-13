"use client";
import { motion } from "framer-motion";
import { Sparkles, Monitor, ArrowRight } from "lucide-react";
import FadeIn from "./FadeIn";

const roadmap = [
  { label: "Integración NLP", desc: "Embeddings sobre texto libre de diagnósticos para enriquecer predicciones", color: "#6366F1", done: false },
  { label: "UI de Auditor",   desc: "Dashboard en tiempo real con cola priorizada por riesgo y explicaciones SHAP", color: "#3B82F6", done: false },
  { label: "Feedback loop",   desc: "El auditor corrige predicciones y el modelo aprende de cada revisión", color: "#22C55E", done: false },
  { label: "API producción",  desc: "Integración directa con el sistema de radicación de facturas de la EPS", color: "#F59E0B", done: false },
];

export default function FuturoUI() {
  return (
    <section id="futuro" className="py-24 px-6">
      <div className="max-w-5xl mx-auto">
        <FadeIn className="text-center mb-16">
          <span
            className="inline-block text-xs px-4 py-1.5 rounded-full mb-5"
            style={{ background: "#0E1223", border: "1px solid #1e2d45", color: "#94A3B8", fontFamily: "var(--font-code)" }}
          >
            Capítulo 6 · Trabajo Futuro
          </span>
          <h2 style={{ fontFamily: "var(--font-heading)", fontSize: "clamp(2rem,5vw,3.2rem)", fontWeight: 600, lineHeight: 1.15, marginBottom: "1rem" }}>
            La interfaz que<br />está por llegar.
          </h2>
          <p style={{ color: "#94A3B8", fontSize: "1.1rem", maxWidth: 560, margin: "0 auto", lineHeight: 1.7 }}>
            El modelo es el núcleo. La siguiente fase convierte las predicciones en
            una herramienta de trabajo para auditores médicos.
          </p>
        </FadeIn>

        <div className="grid md:grid-cols-2 gap-8 items-center mb-12">
          {/* UI mockup wireframe */}
          <FadeIn>
            <motion.div
              className="rounded-2xl p-1 overflow-hidden"
              style={{ background: "linear-gradient(135deg, #22C55E20, #3B82F620, #6366F120)", border: "1px solid #1e2d45" }}
              whileHover={{ scale: 1.01 }}
              transition={{ duration: 0.3 }}
            >
              <div className="rounded-xl overflow-hidden" style={{ background: "#0E1223" }}>
                {/* Fake browser bar */}
                <div className="flex items-center gap-2 px-4 py-3" style={{ borderBottom: "1px solid #1e2d45" }}>
                  {["#EF4444", "#F59E0B", "#22C55E"].map(c => (
                    <div key={c} className="w-3 h-3 rounded-full" style={{ background: c + "80" }} />
                  ))}
                  <div className="flex-1 mx-4 rounded-md py-1 px-3" style={{ background: "#131929", border: "1px solid #1e2d45" }}>
                    <span style={{ fontFamily: "var(--font-code)", fontSize: "0.6rem", color: "#475569" }}>savia-audit.dashboard.internal</span>
                  </div>
                </div>

                {/* Fake dashboard content */}
                <div className="p-5">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2">
                      <Monitor size={14} color="#22C55E" />
                      <span style={{ fontFamily: "var(--font-code)", fontSize: "0.7rem", color: "#94A3B8" }}>Cola de Auditoría · Hoy</span>
                    </div>
                    <span style={{ fontFamily: "var(--font-code)", fontSize: "0.65rem", color: "#22C55E" }}>● En vivo</span>
                  </div>

                  <div className="space-y-2 mb-4">
                    {[
                      { id: "FAC-2024-09871", risk: "ALTA", score: 0.91, class_: "Glosada",  color: "#EF4444" },
                      { id: "FAC-2024-09872", risk: "MEDIA", score: 0.68, class_: "Auditada", color: "#F59E0B" },
                      { id: "FAC-2024-09873", risk: "BAJA",  score: 0.23, class_: "Auditada", color: "#22C55E" },
                    ].map((row) => (
                      <div key={row.id} className="flex items-center justify-between rounded-lg px-3 py-2"
                        style={{ background: "#131929", border: `1px solid ${row.color}20` }}>
                        <span style={{ fontFamily: "var(--font-code)", fontSize: "0.65rem", color: "#94A3B8" }}>{row.id}</span>
                        <div className="flex items-center gap-3">
                          <span style={{ fontSize: "0.6rem", color: row.color, background: row.color + "18", padding: "1px 6px", borderRadius: 3 }}>{row.risk}</span>
                          <span style={{ fontFamily: "var(--font-code)", fontSize: "0.65rem", color: row.color }}>{row.score}</span>
                          <span style={{ fontSize: "0.6rem", color: "#94A3B8" }}>{row.class_}</span>
                        </div>
                      </div>
                    ))}
                  </div>

                  <div className="rounded-lg px-3 py-2 text-center" style={{ background: "#131929", border: "1px solid #1e2d45" }}>
                    <span style={{ fontFamily: "var(--font-code)", fontSize: "0.6rem", color: "#475569" }}>— mockup · interfaz en desarrollo —</span>
                  </div>
                </div>
              </div>
            </motion.div>
          </FadeIn>

          {/* Roadmap */}
          <FadeIn delay={0.15}>
            <div className="space-y-4">
              <div className="flex items-center gap-2 mb-6">
                <Sparkles size={16} color="#6366F1" />
                <span style={{ fontFamily: "var(--font-heading)", fontSize: "1.1rem", fontWeight: 600 }}>Hoja de Ruta</span>
              </div>
              {roadmap.map((r, i) => (
                <motion.div
                  key={r.label}
                  initial={{ opacity: 0, x: 20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.5, delay: i * 0.1 }}
                  className="rounded-xl p-4 flex items-start gap-4"
                  style={{ background: "#131929", border: "1px solid #1e2d45" }}
                >
                  <div className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5"
                    style={{ background: r.color + "18", border: `1px solid ${r.color}40` }}>
                    <ArrowRight size={12} color={r.color} />
                  </div>
                  <div>
                    <p style={{ fontWeight: 600, fontSize: "0.9rem", marginBottom: "0.25rem", color: r.color }}>{r.label}</p>
                    <p style={{ color: "#94A3B8", fontSize: "0.78rem", lineHeight: 1.5 }}>{r.desc}</p>
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
