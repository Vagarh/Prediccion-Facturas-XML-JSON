"use client";
import { motion } from "framer-motion";
import { GitBranch, Cpu, BarChart3, Shield } from "lucide-react";
import FadeIn from "./FadeIn";

const steps = [
  {
    num: "01",
    icon: <Shield size={16} />,
    title: "Ingesta & Privacidad",
    desc: "Los datos de facturación llegan vía Res. 2275/2023 (JSON/XML). PrivacyGuard anonimiza PII antes de cualquier procesamiento.",
    color: "#00FF88",
  },
  {
    num: "02",
    icon: <Cpu size={16} />,
    title: "Feature Engineering",
    desc: "48 variables: 28 originales + 13 logarítmicas + 4 flags de riesgo + 6 interacciones. División temporal para evitar data leakage.",
    color: "#00D4FF",
  },
  {
    num: "03",
    icon: <GitBranch size={16} />,
    title: "XGBoost Multiclase",
    desc: "Modelo entrenado sobre 633K facturas clasifica en 3 clases: Auditada · Glosada · Devuelta. Optimizado para maximizar Recall Glosada.",
    color: "#0066FF",
  },
  {
    num: "04",
    icon: <BarChart3 size={16} />,
    title: "Explicabilidad SHAP",
    desc: "Cada predicción incluye valores SHAP que identifican qué variables impulsaron la decisión — auditable y transparente.",
    color: "#8B5CF6",
  },
];

const classes = [
  { label: "Auditada",  pct: 60, color: "#00D4FF", bg: "rgba(0,212,255,0.06)" },
  { label: "Glosada",   pct: 28, color: "#FF4466", bg: "rgba(255,68,102,0.06)" },
  { label: "Devuelta",  pct: 12, color: "#FFB300", bg: "rgba(255,179,0,0.06)" },
];

export default function Solucion() {
  return (
    <section id="solucion" className="py-28 px-6">
      <div className="max-w-5xl mx-auto">
        <FadeIn className="text-center mb-16">
          <span className="section-label">Capítulo 02 · La Solución</span>
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
            Predicción automática
            <br />
            <span className="grad-text">en el momento de radicación.</span>
          </h2>
          <p style={{ color: "#8892B0", fontSize: "1.05rem", maxWidth: 540, margin: "0 auto", lineHeight: 1.75 }}>
            Un pipeline end-to-end que clasifica cada factura antes de que llegue al auditor,
            priorizando las de mayor riesgo financiero.
          </p>
        </FadeIn>

        {/* Pipeline */}
        <div className="grid md:grid-cols-4 gap-4 mb-14">
          {steps.map((s, i) => (
            <FadeIn key={s.num} delay={i * 0.08}>
              <motion.div
                whileHover={{ y: -5 }}
                transition={{ duration: 0.2 }}
                className="h-full rounded-2xl p-5"
                style={{
                  background: "#10111A",
                  border: "1px solid #1C1E2E",
                  position: "relative",
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    position: "absolute",
                    top: 0, left: 0, right: 0,
                    height: "1px",
                    background: `linear-gradient(90deg, transparent, ${s.color}50, transparent)`,
                  }}
                />
                <span
                  style={{
                    fontFamily: "var(--font-code)",
                    fontSize: "0.6rem",
                    color: "#2A2D3E",
                    position: "absolute",
                    top: "1rem",
                    right: "1rem",
                    letterSpacing: "0.05em",
                  }}
                >
                  {s.num}
                </span>
                <div
                  className="w-8 h-8 rounded-lg flex items-center justify-center mb-4"
                  style={{ background: `${s.color}12`, border: `1px solid ${s.color}25`, color: s.color }}
                >
                  {s.icon}
                </div>
                <h3 style={{ fontFamily: "var(--font-body)", fontSize: "0.9rem", fontWeight: 700, marginBottom: "0.4rem" }}>
                  {s.title}
                </h3>
                <p style={{ color: "#8892B0", fontSize: "0.78rem", lineHeight: 1.65 }}>{s.desc}</p>
              </motion.div>
            </FadeIn>
          ))}
        </div>

        {/* Output classes */}
        <FadeIn>
          <div
            className="rounded-2xl p-8"
            style={{ background: "#0C0D12", border: "1px solid #1C1E2E" }}
          >
            <p
              style={{
                fontFamily: "var(--font-code)",
                fontSize: "0.65rem",
                color: "#3D4461",
                marginBottom: "1.5rem",
                textAlign: "center",
                letterSpacing: "0.1em",
                textTransform: "uppercase",
              }}
            >
              OUTPUT · 3 clases de clasificación
            </p>
            <div className="grid md:grid-cols-3 gap-4">
              {classes.map((c) => (
                <div
                  key={c.label}
                  className="rounded-xl p-6 text-center"
                  style={{ background: c.bg, border: `1px solid ${c.color}20` }}
                >
                  <p
                    style={{
                      fontFamily: "var(--font-code)",
                      fontSize: "2.5rem",
                      fontWeight: 700,
                      color: c.color,
                      letterSpacing: "-0.03em",
                    }}
                  >
                    {c.pct}%
                  </p>
                  <p style={{ fontSize: "0.85rem", color: "#8892B0", marginTop: "0.4rem", fontWeight: 600 }}>
                    {c.label}
                  </p>
                </div>
              ))}
            </div>
            <p style={{ fontSize: "0.7rem", color: "#3D4461", textAlign: "center", marginTop: "1.2rem", fontFamily: "var(--font-code)" }}>
              Distribución aproximada en el dataset de 633K facturas
            </p>
          </div>
        </FadeIn>
      </div>
    </section>
  );
}
