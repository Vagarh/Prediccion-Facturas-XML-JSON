"use client";
import { motion } from "framer-motion";
import { AlertCircle, Star, FileText } from "lucide-react";
import FadeIn from "./FadeIn";

const cards = [
  {
    icon: <AlertCircle size={20} color="#EF4444" />,
    iconBg: "#1a0a0a",
    iconBorder: "rgba(239,68,68,0.2)",
    title: "90 días de rezago",
    desc: "La revisión manual genera cuellos de botella que comprometen la liquidez de toda la red prestadora.",
  },
  {
    icon: <Star size={20} color="#F59E0B" />,
    iconBg: "#1a140a",
    iconBorder: "rgba(245,158,11,0.2)",
    title: "10.5× más impacto",
    desc: "Las facturas glosadas valen en promedio 10.5× más que las auditadas. Cada falso negativo cuesta caro.",
  },
  {
    icon: <FileText size={20} color="#22C55E" />,
    iconBg: "#0a1a10",
    iconBorder: "rgba(34,197,94,0.2)",
    title: "Resolución 2275/2023",
    desc: "El Ministerio de Salud estandarizó datos en JSON/XML. Ahora existe la materia prima para predecir automáticamente.",
  },
];

export default function Problema() {
  return (
    <section id="problema" className="py-24 px-6">
      <div className="max-w-5xl mx-auto">
        <FadeIn className="text-center mb-16">
          <span
            className="inline-block text-xs px-4 py-1.5 rounded-full mb-5"
            style={{ background: "#0E1223", border: "1px solid #1e2d45", color: "#94A3B8", fontFamily: "var(--font-code)" }}
          >
            Capítulo 1 · El Problema
          </span>
          <h2 style={{ fontFamily: "var(--font-heading)", fontSize: "clamp(2rem,5vw,3.2rem)", fontWeight: 600, lineHeight: 1.15, marginBottom: "1rem" }}>
            600.000 facturas.<br />Un auditor a la vez.
          </h2>
          <p style={{ color: "#94A3B8", fontSize: "1.1rem", maxWidth: 560, margin: "0 auto", lineHeight: 1.7 }}>
            El sistema de salud colombiano exige revisión manual de cada factura antes de pagar.
            El resultado: rezagos de hasta 90 días y millones en glosas no detectadas.
          </p>
        </FadeIn>

        <div className="grid md:grid-cols-3 gap-6">
          {cards.map((c, i) => (
            <FadeIn key={c.title} delay={i * 0.12}>
              <motion.div
                whileHover={{ y: -6, boxShadow: "0 12px 40px rgba(0,0,0,0.4)" }}
                transition={{ duration: 0.2 }}
                className="h-full rounded-xl p-6"
                style={{ background: "#131929", border: "1px solid #1e2d45" }}
              >
                <div
                  className="w-10 h-10 rounded-lg flex items-center justify-center mb-4"
                  style={{ background: c.iconBg, border: `1px solid ${c.iconBorder}` }}
                >
                  {c.icon}
                </div>
                <h3 style={{ fontFamily: "var(--font-heading)", fontSize: "1.25rem", fontWeight: 600, marginBottom: "0.5rem" }}>
                  {c.title}
                </h3>
                <p style={{ color: "#94A3B8", fontSize: "0.875rem", lineHeight: 1.7 }}>{c.desc}</p>
              </motion.div>
            </FadeIn>
          ))}
        </div>
      </div>
    </section>
  );
}
