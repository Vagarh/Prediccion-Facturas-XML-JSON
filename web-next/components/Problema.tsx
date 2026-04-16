"use client";
import { motion } from "framer-motion";
import { AlertCircle, TrendingUp, FileText } from "lucide-react";
import FadeIn from "./FadeIn";

const cards = [
  {
    icon: <AlertCircle size={18} />,
    iconColor: "#FF4466",
    accent: "#FF4466",
    title: "90 días de rezago",
    desc: "La revisión manual genera cuellos de botella que comprometen la liquidez de toda la red prestadora.",
  },
  {
    icon: <TrendingUp size={18} />,
    iconColor: "#FFB300",
    accent: "#FFB300",
    title: "10.5× más impacto",
    desc: "Las facturas glosadas valen en promedio 10.5× más que las auditadas. Cada falso negativo cuesta caro.",
  },
  {
    icon: <FileText size={18} />,
    iconColor: "#00D4FF",
    accent: "#00D4FF",
    title: "Resolución 2275/2023",
    desc: "El Ministerio de Salud estandarizó datos en JSON/XML. Ahora existe la materia prima para predecir automáticamente.",
  },
];

export default function Problema() {
  return (
    <section id="problema" className="py-28 px-6">
      <div className="max-w-5xl mx-auto">
        <FadeIn className="text-center mb-16">
          <span className="section-label">Capítulo 01 · El Problema</span>
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
            600.000 facturas.
            <br />
            <span className="grad-text">Un auditor a la vez.</span>
          </h2>
          <p style={{ color: "#8892B0", fontSize: "1.05rem", maxWidth: 540, margin: "0 auto", lineHeight: 1.75 }}>
            El sistema de salud colombiano exige revisión manual de cada factura antes de pagar.
            El resultado: rezagos de hasta 90 días y millones en glosas no detectadas.
          </p>
        </FadeIn>

        <div className="grid md:grid-cols-3 gap-5">
          {cards.map((c, i) => (
            <FadeIn key={c.title} delay={i * 0.1}>
              <motion.div
                whileHover={{ y: -6 }}
                transition={{ duration: 0.2 }}
                className="h-full rounded-2xl p-7"
                style={{
                  background: "#10111A",
                  border: "1px solid #1C1E2E",
                  position: "relative",
                  overflow: "hidden",
                }}
              >
                {/* Accent top bar */}
                <div
                  style={{
                    position: "absolute",
                    top: 0, left: 0, right: 0,
                    height: "2px",
                    background: `linear-gradient(90deg, transparent, ${c.accent}80, transparent)`,
                  }}
                />
                <div
                  className="w-10 h-10 rounded-xl flex items-center justify-center mb-5"
                  style={{
                    background: `${c.accent}10`,
                    border: `1px solid ${c.accent}25`,
                    color: c.iconColor,
                  }}
                >
                  {c.icon}
                </div>
                <h3
                  style={{
                    fontFamily: "var(--font-body)",
                    fontSize: "1.1rem",
                    fontWeight: 700,
                    marginBottom: "0.6rem",
                    color: "#F0F2FF",
                  }}
                >
                  {c.title}
                </h3>
                <p style={{ color: "#8892B0", fontSize: "0.875rem", lineHeight: 1.7 }}>{c.desc}</p>
              </motion.div>
            </FadeIn>
          ))}
        </div>
      </div>
    </section>
  );
}
