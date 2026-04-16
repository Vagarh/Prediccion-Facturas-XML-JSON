"use client";
import { motion } from "framer-motion";
import { ArrowRight, Clock, AlertTriangle, DollarSign } from "lucide-react";
import FadeIn from "./FadeIn";

// Nodos del ciclo vicioso (dinámica de sistemas simplificada)
const ciclo = [
  { id: "A", label: "Factura llega", sub: "La clínica radica y espera pago", color: "#8892B0", x: 50, y: 10 },
  { id: "B", label: "Cola manual", sub: "Auditor la recibe entre miles", color: "#FFB300", x: 85, y: 45 },
  { id: "C", label: "Rezago 90 días", sub: "La factura espera turno", color: "#FF4466", x: 65, y: 80 },
  { id: "D", label: "Pago sin revisar", sub: "Presión financiera fuerza el pago", color: "#FF6644", x: 20, y: 75 },
  { id: "E", label: "Glosa no detectada", sub: "El error ya fue pagado", color: "#FF4466", x: 5, y: 40 },
];

export default function Problema() {
  return (
    <section id="problema" className="py-28 px-6">
      <div className="max-w-5xl mx-auto">

        <FadeIn className="text-center mb-20">
          <span className="section-label">El Problema</span>
          <h2 style={{ fontFamily: "var(--font-body)", fontSize: "clamp(2.2rem, 5vw, 3.5rem)", fontWeight: 700, lineHeight: 1.1, letterSpacing: "-0.022em", marginBottom: "1.2rem" }}>
            El ciclo de auditoría actual
            <br />
            <span className="grad-text">está diseñado para fallar.</span>
          </h2>
          <p style={{ color: "#8892B0", fontSize: "1.05rem", maxWidth: 620, margin: "0 auto", lineHeight: 1.78 }}>
            En Colombia, cada factura médica debe ser revisada antes de pagarla.
            El problema: el volumen supera la capacidad humana. El sistema colapsa
            y las glosas más costosas pasan desapercibidas.
          </p>
        </FadeIn>

        {/* Dinámica de sistemas — ciclo vicioso visual */}
        <FadeIn>
          <div className="rounded-2xl p-8 mb-8 relative overflow-hidden"
            style={{ background: "#10111A", border: "1px solid #1C1E2E" }}>
            <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: "2px", background: "linear-gradient(90deg, transparent, #FF446660, transparent)" }} />
            <p style={{ fontFamily: "var(--font-code)", fontSize: "0.65rem", color: "#3D4461", marginBottom: "1.5rem", letterSpacing: "0.1em", textTransform: "uppercase" }}>
              Dinámica del sistema actual · Ciclo de retroalimentación negativa
            </p>

            {/* Ciclo visual — horizontal en mobile, circular en desktop */}
            <div className="flex flex-col md:flex-row items-center gap-2 md:gap-0 justify-between mb-6">
              {[
                { label: "Factura llega", color: "#8892B0" },
                { label: "→", color: "#2A2D3E" },
                { label: "Cola de auditoría", color: "#FFB300" },
                { label: "→", color: "#2A2D3E" },
                { label: "90 días de rezago", color: "#FF4466" },
                { label: "→", color: "#2A2D3E" },
                { label: "Pago sin revisar", color: "#FF6644" },
                { label: "→", color: "#2A2D3E" },
                { label: "Glosa no detectada", color: "#FF4466" },
              ].map((n, i) => (
                <div key={i} style={{ textAlign: "center" }}>
                  {n.label === "→" ? (
                    <span style={{ color: n.color, fontSize: "1.2rem" }}>→</span>
                  ) : (
                    <div className="rounded-xl px-3 py-2"
                      style={{ background: `${n.color}10`, border: `1px solid ${n.color}25` }}>
                      <p style={{ fontSize: "0.75rem", fontWeight: 700, color: n.color, whiteSpace: "nowrap" }}>{n.label}</p>
                    </div>
                  )}
                </div>
              ))}
            </div>

            {/* Flecha de retroalimentación */}
            <div className="rounded-xl p-4 text-center"
              style={{ background: "rgba(255,68,102,0.05)", border: "1px solid rgba(255,68,102,0.15)" }}>
              <p style={{ fontSize: "0.82rem", color: "#FF4466" }}>
                ↑ El ciclo se repite: cada glosa no detectada acumula presión sobre la siguiente
              </p>
            </div>
          </div>
        </FadeIn>

        {/* Las 3 consecuencias concretas */}
        <div className="grid md:grid-cols-3 gap-5 mb-12">
          {[
            {
              icon: <Clock size={18} />, color: "#FF4466",
              stat: "90 días", statLabel: "de rezago promedio",
              desc: "Un equipo de auditores revisando manualmente más de 50.000 facturas al mes. El cuello de botella es estructural, no de esfuerzo.",
            },
            {
              icon: <DollarSign size={18} />, color: "#FFB300",
              stat: "10.5×", statLabel: "más vale una factura glosada",
              desc: "Las facturas glosadas tienen un valor promedio 10.5 veces mayor que las auditadas. Cada una que se paga sin revisar representa pérdida directa.",
            },
            {
              icon: <AlertTriangle size={18} />, color: "#00D4FF",
              stat: "Res. 2275", statLabel: "Ministerio de Salud, 2023",
              desc: "Desde 2023, los datos de facturación están estandarizados en JSON/XML. Por primera vez existe la materia prima para automatizar la detección.",
            },
          ].map((c, i) => (
            <FadeIn key={c.stat} delay={i * 0.1}>
              <motion.div whileHover={{ y: -5 }} transition={{ duration: 0.2 }}
                className="h-full rounded-2xl p-7"
                style={{ background: "#10111A", border: "1px solid #1C1E2E", position: "relative", overflow: "hidden" }}
              >
                <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: "2px", background: `linear-gradient(90deg, transparent, ${c.color}70, transparent)` }} />
                <div className="w-10 h-10 rounded-xl flex items-center justify-center mb-4"
                  style={{ background: `${c.color}10`, border: `1px solid ${c.color}25`, color: c.color }}>
                  {c.icon}
                </div>
                <p style={{ fontFamily: "var(--font-code)", fontSize: "1.6rem", fontWeight: 700, color: c.color, letterSpacing: "-0.02em", lineHeight: 1 }}>{c.stat}</p>
                <p style={{ fontSize: "0.7rem", color: c.color, opacity: 0.7, marginBottom: "0.75rem", fontFamily: "var(--font-code)" }}>{c.statLabel}</p>
                <p style={{ color: "#8892B0", fontSize: "0.85rem", lineHeight: 1.72 }}>{c.desc}</p>
              </motion.div>
            </FadeIn>
          ))}
        </div>

        {/* Puente hacia la solución */}
        <FadeIn>
          <div className="rounded-2xl p-8 flex flex-col md:flex-row items-center gap-6"
            style={{ background: "rgba(0,212,255,0.04)", border: "1px solid rgba(0,212,255,0.15)" }}>
            <div className="flex-1">
              <p style={{ fontSize: "1.1rem", fontWeight: 700, color: "#F0F2FF", lineHeight: 1.6, marginBottom: "0.5rem" }}>
                La pregunta que nos hicimos:
              </p>
              <p style={{ fontSize: "1.05rem", color: "#00D4FF", fontWeight: 600, lineHeight: 1.6 }}>
                "¿Podemos predecir el resultado de una factura en el momento en que llega,
                antes de que cualquier humano la toque?"
              </p>
            </div>
            <div className="flex-shrink-0">
              <a href="#solucion" style={{ display: "inline-flex", alignItems: "center", gap: 8, padding: "12px 24px", borderRadius: "8px", background: "rgba(0,212,255,0.1)", border: "1px solid rgba(0,212,255,0.25)", color: "#00D4FF", textDecoration: "none", fontWeight: 600, fontSize: "0.88rem" }}>
                Ver cómo lo resolvimos <ArrowRight size={15} />
              </a>
            </div>
          </div>
        </FadeIn>

      </div>
    </section>
  );
}
