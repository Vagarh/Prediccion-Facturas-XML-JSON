"use client";
import { motion } from "framer-motion";
import { ArrowDown } from "lucide-react";

const metrics = [
  { value: "633K",   label: "Facturas reales analizadas",       color: "#00D4FF" },
  { value: "89.6%",  label: "Glosas detectadas antes de pagar", color: "#00FF88" },
  { value: "0.9028", label: "AUC-ROC",                          color: "#0066FF" },
  { value: "48",     label: "Variables de entrada al modelo",   color: "#8B5CF6" },
];

export default function Hero() {
  return (
    <section className="relative min-h-[100dvh] flex flex-col items-center justify-center text-center px-6 pt-20 overflow-hidden">
      <div className="absolute inset-0 grid-bg pointer-events-none" />
      <motion.div className="absolute rounded-full pointer-events-none"
        style={{ width: 700, height: 700, top: "-10%", left: "-15%", background: "radial-gradient(circle, rgba(0,102,255,0.06) 0%, transparent 65%)" }}
        animate={{ scale: [1, 1.1, 1], opacity: [0.5, 0.8, 0.5] }}
        transition={{ duration: 10, repeat: Infinity, ease: "easeInOut" }}
      />
      <motion.div className="absolute rounded-full pointer-events-none"
        style={{ width: 600, height: 600, bottom: "-5%", right: "-10%", background: "radial-gradient(circle, rgba(0,212,255,0.05) 0%, transparent 65%)" }}
        animate={{ scale: [1, 1.15, 1], opacity: [0.4, 0.7, 0.4] }}
        transition={{ duration: 12, repeat: Infinity, ease: "easeInOut", delay: 3 }}
      />

      <div className="relative max-w-4xl mx-auto">
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}
          className="flex justify-center mb-6">
          <span className="badge">Maestría en Ciencia de Datos · Universidad EAFIT · 2026</span>
        </motion.div>

        <motion.h1 initial={{ opacity: 0, y: 32 }} animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.15 }}
          style={{ fontFamily: "var(--font-body)", fontSize: "clamp(2.8rem, 7.5vw, 5.2rem)", fontWeight: 700, lineHeight: 1.06, letterSpacing: "-0.025em", marginBottom: "1.5rem" }}
        >
          IA que predice si una factura médica
          <br />
          <span className="grad-text">será glosada antes de pagarla.</span>
        </motion.h1>

        <motion.p initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.3 }}
          style={{ fontSize: "clamp(1rem, 2vw, 1.18rem)", color: "#8892B0", maxWidth: 600, margin: "0 auto 2.5rem", lineHeight: 1.78 }}
        >
          Un sistema de predicción entrenado con{" "}
          <strong style={{ color: "#F0F2FF" }}>633.000 facturas reales</strong> de Savia Salud EPS.
          Detecta el <strong style={{ color: "#00FF88" }}>89.6%</strong> de las facturas problemáticas
          en el momento de radicación — sin intervención humana previa.
        </motion.p>

        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.45 }}
          className="flex flex-col sm:flex-row gap-4 justify-center"
        >
          <a href="#problema" style={{
            display: "inline-block", padding: "14px 32px", borderRadius: "8px",
            fontWeight: 600, fontSize: "0.9rem",
            background: "linear-gradient(135deg, #00D4FF, #0066FF)",
            color: "#060608", boxShadow: "0 0 32px rgba(0,212,255,0.35)", textDecoration: "none",
          }}>
            Ver el proyecto
          </a>
          <a href="#impacto" style={{
            display: "inline-block", padding: "14px 32px", borderRadius: "8px",
            fontWeight: 500, fontSize: "0.9rem", background: "transparent",
            color: "#F0F2FF", border: "1px solid #1C1E2E", textDecoration: "none",
          }}>
            Ver resultados
          </a>
        </motion.div>
      </div>

      <motion.div initial={{ opacity: 0, y: 40 }} animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.65 }}
        className="relative mt-20 grid grid-cols-2 md:grid-cols-4 gap-3 max-w-3xl w-full"
      >
        {metrics.map((m) => (
          <motion.div key={m.label} whileHover={{ y: -4 }} transition={{ duration: 0.2 }}
            style={{ textAlign: "center", borderRadius: "12px", padding: "20px 12px", background: "#10111A", border: "1px solid #1C1E2E", position: "relative", overflow: "hidden" }}
          >
            <div style={{ position: "absolute", top: 0, left: "20%", right: "20%", height: "1px", background: `linear-gradient(90deg, transparent, ${m.color}60, transparent)` }} />
            <p style={{ fontFamily: "var(--font-code)", fontSize: "1.7rem", fontWeight: 600, color: m.color, lineHeight: 1, letterSpacing: "-0.02em" }}>{m.value}</p>
            <p style={{ fontSize: "0.68rem", color: "#8892B0", marginTop: "0.5rem", fontWeight: 500, lineHeight: 1.4 }}>{m.label}</p>
          </motion.div>
        ))}
      </motion.div>

      <motion.div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2"
        style={{ color: "#3D4461", fontSize: "0.6rem", fontFamily: "var(--font-code)", letterSpacing: "0.12em" }}
        animate={{ y: [0, 8, 0] }} transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
      >
        <span>EXPLORAR</span><ArrowDown size={12} />
      </motion.div>
    </section>
  );
}
