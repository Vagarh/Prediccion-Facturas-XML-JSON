"use client";
import { motion } from "framer-motion";
import { ArrowRight, MessageSquare, Layers, RefreshCw, Plug } from "lucide-react";
import FadeIn from "./FadeIn";

const siguientes = [
  {
    icon: <MessageSquare size={15} />, color: "#8B5CF6", estado: "En desarrollo",
    title: "Explicaciones en lenguaje natural",
    desc: "LLM local (Ollama) que convierte el SHAP output en texto comprensible para el auditor: "Esta factura fue marcada como glosada principalmente porque su valor es 3× el promedio del prestador."",
  },
  {
    icon: <Layers size={15} />, color: "#00D4FF", estado: "Diseñado",
    title: "Dashboard para auditores",
    desc: "Cola de trabajo priorizada por riesgo. Las facturas con mayor probabilidad de glosa aparecen primero, con el contexto SHAP visible directamente en la interfaz.",
  },
  {
    icon: <RefreshCw size={15} />, color: "#00FF88", estado: "Planificado",
    title: "Aprendizaje por retroalimentación",
    desc: "Cada corrección que hace el auditor alimenta el siguiente ciclo de reentrenamiento. El modelo mejora con el uso real, no solo con datos históricos.",
  },
  {
    icon: <Plug size={15} />, color: "#FFB300", estado: "Planificado",
    title: "Integración directa con radicación",
    desc: "API conectada al sistema de radicación de la EPS. La predicción ocurre en el momento exacto en que la factura entra al sistema — antes de cualquier cola.",
  },
];

export default function FuturoUI() {
  return (
    <section id="futuro" className="py-28 px-6">
      <div className="max-w-5xl mx-auto">

        <FadeIn className="text-center mb-16">
          <span className="section-label">Qué sigue</span>
          <h2 style={{ fontFamily: "var(--font-body)", fontSize: "clamp(2.2rem, 5vw, 3.5rem)", fontWeight: 700, lineHeight: 1.1, letterSpacing: "-0.022em", marginBottom: "1.2rem" }}>
            El modelo funciona.
            <br />
            <span className="grad-text">El siguiente paso es ponerlo en producción.</span>
          </h2>
          <p style={{ color: "#8892B0", fontSize: "1.05rem", maxWidth: 600, margin: "0 auto", lineHeight: 1.78 }}>
            La predicción ya está validada. Lo que viene es convertirla en una herramienta
            que el auditor use todos los días — con explicaciones claras, sin fricción.
          </p>
        </FadeIn>

        <div className="grid md:grid-cols-2 gap-10 mb-14 items-start">
          {/* Mockup dashboard */}
          <FadeIn>
            <motion.div className="rounded-2xl p-px overflow-hidden"
              style={{ background: "linear-gradient(135deg, rgba(0,212,255,0.2), rgba(0,102,255,0.15), rgba(139,92,246,0.1))" }}
              whileHover={{ scale: 1.01 }} transition={{ duration: 0.3 }}
            >
              <div className="rounded-2xl overflow-hidden" style={{ background: "#0C0D12" }}>
                <div className="flex items-center gap-2 px-4 py-3" style={{ borderBottom: "1px solid #1C1E2E" }}>
                  {["#FF4466", "#FFB300", "#00FF88"].map((c) => (
                    <div key={c} className="w-3 h-3 rounded-full" style={{ background: c + "70" }} />
                  ))}
                  <div className="flex-1 mx-4 rounded-md py-1 px-3"
                    style={{ background: "#10111A", border: "1px solid #1C1E2E" }}>
                    <span style={{ fontFamily: "var(--font-code)", fontSize: "0.6rem", color: "#3D4461" }}>
                      savia-audit.eps.co/dashboard
                    </span>
                  </div>
                </div>
                <div className="p-5">
                  <div className="flex items-center justify-between mb-4">
                    <span style={{ fontFamily: "var(--font-code)", fontSize: "0.68rem", color: "#8892B0" }}>
                      Cola de Auditoría · Priorizada por riesgo
                    </span>
                    <span style={{ fontFamily: "var(--font-code)", fontSize: "0.62rem", color: "#00FF88" }}>● En vivo</span>
                  </div>
                  <div className="space-y-2 mb-4">
                    {[
                      { id: "FAC-2024-09871", risk: "ALTO",  score: 0.91, class_: "Glosada",  color: "#FF4466", why: "Valor 3× promedio · Alta complejidad" },
                      { id: "FAC-2024-09872", risk: "MEDIO", score: 0.68, class_: "Auditada", color: "#FFB300", why: "Estancia prolongada · Dx costoso" },
                      { id: "FAC-2024-09873", risk: "BAJO",  score: 0.23, class_: "Auditada", color: "#00FF88", why: "Perfil histórico sin glosas" },
                    ].map((row) => (
                      <div key={row.id} className="rounded-xl px-3 py-2.5"
                        style={{ background: "#10111A", border: `1px solid ${row.color}18` }}>
                        <div className="flex items-center justify-between mb-1">
                          <span style={{ fontFamily: "var(--font-code)", fontSize: "0.65rem", color: "#8892B0" }}>{row.id}</span>
                          <div className="flex items-center gap-2">
                            <span style={{ fontSize: "0.58rem", color: row.color, background: `${row.color}12`, padding: "2px 7px", borderRadius: 3, fontFamily: "var(--font-code)" }}>
                              {row.risk}
                            </span>
                            <span style={{ fontFamily: "var(--font-code)", fontSize: "0.65rem", color: row.color }}>{row.score}</span>
                          </div>
                        </div>
                        <p style={{ fontSize: "0.6rem", color: "#3D4461", fontFamily: "var(--font-code)" }}>{row.why}</p>
                      </div>
                    ))}
                  </div>
                  <div className="rounded-xl px-3 py-2 text-center"
                    style={{ background: "#10111A", border: "1px solid #1C1E2E" }}>
                    <span style={{ fontFamily: "var(--font-code)", fontSize: "0.58rem", color: "#3D4461" }}>
                      — mockup · interfaz en diseño —
                    </span>
                  </div>
                </div>
              </div>
            </motion.div>
          </FadeIn>

          {/* Hoja de ruta */}
          <FadeIn delay={0.1}>
            <div className="space-y-4">
              {siguientes.map((s, i) => (
                <motion.div key={s.title}
                  initial={{ opacity: 0, x: 20 }} whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }} transition={{ duration: 0.4, delay: i * 0.08 }}
                  className="rounded-xl p-5 flex items-start gap-4"
                  style={{ background: "#10111A", border: "1px solid #1C1E2E" }}
                >
                  <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5"
                    style={{ background: `${s.color}12`, border: `1px solid ${s.color}25`, color: s.color }}>
                    {s.icon}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1 flex-wrap">
                      <p style={{ fontWeight: 700, fontSize: "0.88rem", color: s.color }}>{s.title}</p>
                      <span style={{ fontSize: "0.58rem", color: "#3D4461", background: "#0C0D12", border: "1px solid #1C1E2E", padding: "1px 6px", borderRadius: 3, fontFamily: "var(--font-code)" }}>
                        {s.estado}
                      </span>
                    </div>
                    <p style={{ color: "#8892B0", fontSize: "0.78rem", lineHeight: 1.65 }}>{s.desc}</p>
                  </div>
                </motion.div>
              ))}
            </div>
          </FadeIn>
        </div>

        {/* Cierre */}
        <FadeIn>
          <div className="rounded-2xl p-10 text-center relative overflow-hidden"
            style={{ background: "rgba(0,102,255,0.04)", border: "1px solid rgba(0,102,255,0.15)" }}>
            <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: "1px", background: "linear-gradient(90deg, transparent, #0066FF60, transparent)" }} />
            <p style={{ fontSize: "1.2rem", fontWeight: 700, color: "#F0F2FF", lineHeight: 1.6, maxWidth: 600, margin: "0 auto 1rem" }}>
              Este proyecto demuestra que la{" "}
              <span className="grad-text">auditoría en salud puede ser predictiva</span>
              , no reactiva.
            </p>
            <p style={{ color: "#8892B0", fontSize: "0.95rem", maxWidth: 540, margin: "0 auto 1.5rem", lineHeight: 1.72 }}>
              Con datos reales, metodología rigurosa y tecnología accesible es posible
              detectar el problema antes de que ocurra — y devolver ese tiempo
              a los auditores para que se enfoquen en lo que importa.
            </p>
            <div className="flex items-center justify-center gap-2" style={{ color: "#0066FF" }}>
              <span style={{ fontSize: "0.82rem", fontWeight: 600 }}>Juan Felipe Cardona Arango · EAFIT 2026</span>
              <ArrowRight size={14} />
            </div>
          </div>
        </FadeIn>

      </div>
    </section>
  );
}
