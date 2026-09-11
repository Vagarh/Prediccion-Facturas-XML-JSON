"use client";
import { motion } from "framer-motion";
import { ArrowRight, Database, Shield, Cpu, GitBranch, FileText, CheckCircle2 } from "lucide-react";
import FadeIn from "./FadeIn";

export default function Solucion() {
  return (
    <section id="solucion" className="py-28 px-6">
      <div className="max-w-5xl mx-auto">

        <FadeIn className="text-center mb-16">
          <span className="section-label">La Solución</span>
          <h2 style={{ fontFamily: "var(--font-body)", fontSize: "clamp(2.2rem, 5vw, 3.5rem)", fontWeight: 700, lineHeight: 1.1, letterSpacing: "-0.022em", marginBottom: "1.2rem" }}>
            No simulamos datos.
            <br />
            <span className="grad-text">Trabajamos con los de producción.</span>
          </h2>
          <p style={{ color: "#8892B0", fontSize: "1.05rem", maxWidth: 600, margin: "0 auto", lineHeight: 1.78 }}>
            Nos conectamos directamente a la base de datos real de Savia Salud EPS,
            mapeamos 554 tablas y construimos un pipeline que va desde la factura cruda
            hasta la predicción con explicación — en milisegundos.
          </p>
        </FadeIn>

        {/* Flujo de datos — el corazón de la sección */}
        <FadeIn>
          <div className="rounded-2xl p-8 mb-10 relative overflow-hidden"
            style={{ background: "#10111A", border: "1px solid #1C1E2E" }}>
            <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: "1px", background: "linear-gradient(90deg, transparent, #00D4FF40, transparent)" }} />
            <p style={{ fontFamily: "var(--font-code)", fontSize: "0.65rem", color: "#3D4461", marginBottom: "2rem", letterSpacing: "0.1em", textTransform: "uppercase" }}>
              Flujo de datos · De la factura a la predicción
            </p>

            {/* Pipeline visual */}
            <div className="flex flex-col md:flex-row items-stretch gap-0">
              {[
                {
                  step: "Entrada",
                  icon: <FileText size={16} />, color: "#8892B0",
                  items: ["Factura XML/JSON", "Código prestador", "Diagnóstico (CIE-10)", "Procedimientos (CUPS)", "Valor total"],
                },
                {
                  step: "PrivacyGuard",
                  icon: <Shield size={16} />, color: "#00FF88",
                  items: ["Elimina nombre paciente", "Elimina cédula", "Elimina datos sensibles", "Solo quedan variables clínicas y financieras"],
                },
                {
                  step: "Feature Engineering",
                  icon: <Cpu size={16} />, color: "#00D4FF",
                  items: ["48 variables construidas", "Log-transforms del valor", "Flags de riesgo binarios", "Interacciones cruzadas"],
                },
                {
                  step: "LightGBM v3",
                  icon: <GitBranch size={16} />, color: "#0066FF",
                  items: ["Clasificación multiclase", "Auditada / Glosada / Devuelta", "Calibración de probabilidades", "SHAP en tiempo real"],
                },
                {
                  step: "Respuesta API",
                  icon: <Database size={16} />, color: "#8B5CF6",
                  items: ["Clase predicha", "Confianza (0–1)", "Top-5 factores SHAP", "Tiempo: < 200ms"],
                },
              ].map((s, i, arr) => (
                <div key={s.step} className="flex flex-row md:flex-col items-center">
                  <div className="flex-1 rounded-xl p-4" style={{ background: "#0C0D12", border: `1px solid ${s.color}20` }}>
                    <div className="flex items-center gap-2 mb-3">
                      <div className="w-6 h-6 rounded-md flex items-center justify-center flex-shrink-0"
                        style={{ background: `${s.color}15`, color: s.color }}>
                        {s.icon}
                      </div>
                      <span style={{ fontFamily: "var(--font-code)", fontSize: "0.65rem", color: s.color, fontWeight: 600, letterSpacing: "0.05em" }}>
                        {s.step}
                      </span>
                    </div>
                    <ul className="space-y-1.5">
                      {s.items.map((item) => (
                        <li key={item} style={{ fontSize: "0.7rem", color: "#8892B0", lineHeight: 1.5, display: "flex", gap: 5 }}>
                          <span style={{ color: s.color, flexShrink: 0, marginTop: 1 }}>·</span>
                          {item}
                        </li>
                      ))}
                    </ul>
                  </div>
                  {i < arr.length - 1 && (
                    <div className="flex-shrink-0 mx-2 md:mx-0 md:my-2 flex items-center justify-center">
                      <ArrowRight size={14} color="#2A2D3E" style={{ transform: "none" }} />
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </FadeIn>

        {/* Lo que encontramos en los datos */}
        <FadeIn>
          <div className="mb-8">
            <p style={{ fontFamily: "var(--font-body)", fontSize: "1.2rem", fontWeight: 700, marginBottom: "1rem", textAlign: "center" }}>
              Lo que encontramos al analizar los datos
            </p>
            <div className="grid md:grid-cols-2 gap-4">
              {[
                { label: "554 tablas mapeadas en la BD de Savia Salud", color: "#00D4FF", detail: "Identificamos las fuentes relevantes para predecir glosas entre toda la estructura de la BD." },
                { label: "10.5× diferencia de valor entre clases", color: "#FF4466", detail: "Las facturas glosadas son mucho más costosas. El modelo está optimizado para detectarlas primero." },
                { label: "Distribución desbalanceada: 60% Auditadas, 28% Glosadas, 12% Devueltas", color: "#FFB300", detail: "Aplicamos técnicas de balanceo y ajuste de umbral para no sacrificar la detección de glosas." },
                { label: "Right-censoring en los splits temporales", color: "#00FF88", detail: "Hallazgo crítico: dividir datos al azar genera data leakage. Usamos división temporal estricta para simular producción real." },
              ].map((f, i) => (
                <motion.div key={f.label} initial={{ opacity: 0, x: i % 2 === 0 ? -16 : 16 }}
                  whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true }}
                  transition={{ duration: 0.4, delay: i * 0.08 }}
                  className="rounded-xl p-5 flex items-start gap-3"
                  style={{ background: "#10111A", border: `1px solid ${f.color}20` }}
                >
                  <CheckCircle2 size={15} color={f.color} style={{ flexShrink: 0, marginTop: 2 }} />
                  <div>
                    <p style={{ fontWeight: 700, fontSize: "0.85rem", color: "#F0F2FF", marginBottom: "0.3rem" }}>{f.label}</p>
                    <p style={{ fontSize: "0.78rem", color: "#8892B0", lineHeight: 1.6 }}>{f.detail}</p>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </FadeIn>

        {/* Output del modelo */}
        <FadeIn>
          <div className="rounded-2xl p-8" style={{ background: "#0C0D12", border: "1px solid #1C1E2E" }}>
            <p style={{ fontFamily: "var(--font-code)", fontSize: "0.65rem", color: "#3D4461", marginBottom: "1.5rem", textAlign: "center", letterSpacing: "0.1em", textTransform: "uppercase" }}>
              El modelo predice 3 categorías · Distribución real en 1.08M facturas
            </p>
            <div className="grid md:grid-cols-3 gap-4">
              {[
                { label: "Auditada",  pct: "19%", color: "#00D4FF", bg: "rgba(0,212,255,0.06)", desc: "Cumple requisitos. Pasa auditoría estándar." },
                { label: "Glosada",   pct: "66%", color: "#FF4466", bg: "rgba(255,68,102,0.06)", desc: "Tiene errores. Se objeta total o parcialmente." },
                { label: "Devuelta",  pct: "14%", color: "#FFB300", bg: "rgba(255,179,0,0.06)",  desc: "Incompleta. Se regresa al prestador a corregir." },
              ].map((c) => (
                <div key={c.label} className="rounded-xl p-6 text-center"
                  style={{ background: c.bg, border: `1px solid ${c.color}20` }}>
                  <p style={{ fontFamily: "var(--font-code)", fontSize: "2.4rem", fontWeight: 700, color: c.color, letterSpacing: "-0.03em" }}>{c.pct}</p>
                  <p style={{ fontSize: "0.95rem", color: "#F0F2FF", fontWeight: 700, margin: "0.3rem 0 0.5rem" }}>{c.label}</p>
                  <p style={{ fontSize: "0.75rem", color: "#8892B0", lineHeight: 1.5 }}>{c.desc}</p>
                </div>
              ))}
            </div>
          </div>
        </FadeIn>

      </div>
    </section>
  );
}
