"use client";
import { motion } from "framer-motion";
import { GitBranch, Cpu, BarChart3, Shield } from "lucide-react";
import FadeIn from "./FadeIn";

const steps = [
  {
    num: "01",
    icon: <Shield size={18} color="#22C55E" />,
    title: "Ingesta & Privacidad",
    desc: "Los datos de facturación llegan vía Res. 2275/2023 (JSON/XML). PrivacyGuard anonimiza PII antes de cualquier procesamiento.",
    color: "#22C55E",
  },
  {
    num: "02",
    icon: <Cpu size={18} color="#3B82F6" />,
    title: "Feature Engineering",
    desc: "48 variables: 28 originales + 13 logarítmicas + 4 flags de riesgo + 6 interacciones. División temporal para evitar data leakage.",
    color: "#3B82F6",
  },
  {
    num: "03",
    icon: <GitBranch size={18} color="#6366F1" />,
    title: "XGBoost Multiclase",
    desc: "Modelo entrenado sobre 633K facturas clasifica en 3 clases: Auditada · Glosada · Devuelta. Optimizado para maximizar Recall Glosada.",
    color: "#6366F1",
  },
  {
    num: "04",
    icon: <BarChart3 size={18} color="#F59E0B" />,
    title: "Explicabilidad SHAP",
    desc: "Cada predicción incluye valores SHAP que identifican qué variables impulsaron la decisión — auditable y transparente.",
    color: "#F59E0B",
  },
];

const classes = [
  { label: "Auditada",  pct: 60, color: "#22C55E", bg: "rgba(34,197,94,0.12)" },
  { label: "Glosada",   pct: 28, color: "#EF4444", bg: "rgba(239,68,68,0.12)" },
  { label: "Devuelta",  pct: 12, color: "#F59E0B", bg: "rgba(245,158,11,0.12)" },
];

export default function Solucion() {
  return (
    <section id="solucion" className="py-24 px-6">
      <div className="max-w-5xl mx-auto">
        <FadeIn className="text-center mb-16">
          <span
            className="inline-block text-xs px-4 py-1.5 rounded-full mb-5"
            style={{ background: "#0E1223", border: "1px solid #1e2d45", color: "#94A3B8", fontFamily: "var(--font-code)" }}
          >
            Capítulo 2 · La Solución
          </span>
          <h2 style={{ fontFamily: "var(--font-heading)", fontSize: "clamp(2rem,5vw,3.2rem)", fontWeight: 600, lineHeight: 1.15, marginBottom: "1rem" }}>
            Predicción automática<br />en el momento de radicación.
          </h2>
          <p style={{ color: "#94A3B8", fontSize: "1.1rem", maxWidth: 560, margin: "0 auto", lineHeight: 1.7 }}>
            Un pipeline end-to-end que clasifica cada factura antes de que llegue al auditor,
            priorizando las de mayor riesgo financiero.
          </p>
        </FadeIn>

        {/* Pipeline steps */}
        <div className="grid md:grid-cols-4 gap-4 mb-14">
          {steps.map((s, i) => (
            <FadeIn key={s.num} delay={i * 0.1}>
              <motion.div
                whileHover={{ y: -5, boxShadow: "0 10px 36px rgba(0,0,0,0.4)" }}
                transition={{ duration: 0.2 }}
                className="h-full rounded-xl p-5"
                style={{ background: "#131929", border: "1px solid #1e2d45", position: "relative" }}
              >
                <span style={{ fontFamily: "var(--font-code)", fontSize: "0.65rem", color: "#334155", position: "absolute", top: "1rem", right: "1rem" }}>
                  {s.num}
                </span>
                <div className="w-8 h-8 rounded-lg flex items-center justify-center mb-3" style={{ background: s.color + "18", border: `1px solid ${s.color}30` }}>
                  {s.icon}
                </div>
                <h3 style={{ fontFamily: "var(--font-heading)", fontSize: "1rem", fontWeight: 600, marginBottom: "0.4rem" }}>{s.title}</h3>
                <p style={{ color: "#94A3B8", fontSize: "0.8rem", lineHeight: 1.6 }}>{s.desc}</p>
              </motion.div>
            </FadeIn>
          ))}
        </div>

        {/* Output classes */}
        <FadeIn>
          <div className="rounded-2xl p-8" style={{ background: "#0E1223", border: "1px solid #1e2d45" }}>
            <p style={{ fontFamily: "var(--font-code)", fontSize: "0.7rem", color: "#475569", marginBottom: "1.5rem", textAlign: "center" }}>
              OUTPUT · 3 clases de clasificación
            </p>
            <div className="grid md:grid-cols-3 gap-4">
              {classes.map((c) => (
                <div key={c.label} className="rounded-xl p-5 text-center" style={{ background: c.bg, border: `1px solid ${c.color}30` }}>
                  <p style={{ fontFamily: "var(--font-heading)", fontSize: "1.8rem", fontWeight: 700, color: c.color }}>{c.pct}%</p>
                  <p style={{ fontSize: "0.85rem", color: "#CBD5E1", marginTop: "0.3rem", fontWeight: 500 }}>{c.label}</p>
                </div>
              ))}
            </div>
            <p style={{ fontSize: "0.75rem", color: "#475569", textAlign: "center", marginTop: "1rem" }}>
              Distribución aproximada en el dataset de 633K facturas
            </p>
          </div>
        </FadeIn>
      </div>
    </section>
  );
}
