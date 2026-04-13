"use client";
import { motion } from "framer-motion";
import { ArrowDown } from "lucide-react";

const metrics = [
  { value: "633K",   label: "Facturas analizadas", color: "#22C55E" },
  { value: "89.6%",  label: "Recall Glosada",       color: "#3B82F6" },
  { value: "0.9028", label: "AUC-ROC",               color: "#6366F1" },
  { value: "380",    label: "Tests automatizados",   color: "#F59E0B" },
];

export default function Hero() {
  return (
    <section className="relative min-h-[100dvh] flex flex-col items-center justify-center text-center px-6 pt-20 overflow-hidden">
      {/* Grid bg */}
      <div className="absolute inset-0 grid-bg pointer-events-none" />

      {/* Glow orbs */}
      <motion.div
        className="absolute rounded-full pointer-events-none"
        style={{ width: 500, height: 500, top: "15%", left: "10%", background: "radial-gradient(circle, rgba(99,102,241,0.07) 0%, transparent 70%)" }}
        animate={{ scale: [1, 1.15, 1], opacity: [0.6, 1, 0.6] }}
        transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
      />
      <motion.div
        className="absolute rounded-full pointer-events-none"
        style={{ width: 400, height: 400, bottom: "10%", right: "8%", background: "radial-gradient(circle, rgba(34,197,94,0.08) 0%, transparent 70%)" }}
        animate={{ scale: [1, 1.2, 1], opacity: [0.5, 0.9, 0.5] }}
        transition={{ duration: 10, repeat: Infinity, ease: "easeInOut", delay: 2 }}
      />

      {/* Content */}
      <div className="relative max-w-4xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <span
            className="inline-block text-xs px-4 py-1.5 rounded-full mb-8"
            style={{ background: "#0E1223", border: "1px solid #1e2d45", color: "#94A3B8", fontFamily: "var(--font-code)" }}
          >
            Maestría en Ciencia de Datos · Universidad EAFIT · 2026
          </span>
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.15 }}
          style={{ fontFamily: "var(--font-heading)", fontSize: "clamp(2.8rem, 7vw, 5rem)", fontWeight: 600, lineHeight: 1.1, marginBottom: "1.5rem" }}
        >
          Analítica Predictiva
          <br />
          para{" "}
          <span className="grad-text">Auditoría Prestacional</span>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.3 }}
          className="text-lg md:text-xl max-w-2xl mx-auto leading-relaxed mb-10"
          style={{ color: "#94A3B8" }}
        >
          Un modelo de inteligencia artificial que predice, en el momento de radicación,
          si una factura médica será{" "}
          <strong style={{ color: "#F8FAFC" }}>Auditada</strong>,{" "}
          <strong style={{ color: "#F8FAFC" }}>Glosada</strong> o{" "}
          <strong style={{ color: "#F8FAFC" }}>Devuelta</strong> — sin intervención humana previa.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.45 }}
          className="flex flex-col sm:flex-row gap-4 justify-center"
        >
          <a
            href="#resultados"
            className="px-8 py-3 rounded-lg font-medium text-sm transition-all duration-200"
            style={{
              background: "linear-gradient(135deg, #22C55E, #16A34A)",
              color: "#020617",
              boxShadow: "0 0 28px rgba(34,197,94,0.3)",
              textDecoration: "none",
            }}
          >
            Ver Resultados
          </a>
          <a
            href="#video"
            className="px-8 py-3 rounded-lg font-medium text-sm transition-all duration-200"
            style={{ background: "transparent", color: "#F8FAFC", border: "1px solid #1e2d45", textDecoration: "none" }}
          >
            Video Explicativo
          </a>
        </motion.div>
      </div>

      {/* Metric pills */}
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.65 }}
        className="relative mt-16 grid grid-cols-2 md:grid-cols-4 gap-4 max-w-3xl w-full"
      >
        {metrics.map((m, i) => (
          <motion.div
            key={m.label}
            whileHover={{ y: -4, scale: 1.02 }}
            transition={{ duration: 0.2 }}
            className="text-center rounded-xl py-4 px-3"
            style={{ background: "#131929", border: "1px solid #1e2d45" }}
          >
            <p style={{ fontFamily: "var(--font-code)", fontSize: "1.6rem", fontWeight: 500, color: m.color, lineHeight: 1 }}>
              {m.value}
            </p>
            <p style={{ fontSize: "0.75rem", color: "#94A3B8", marginTop: "0.4rem" }}>{m.label}</p>
          </motion.div>
        ))}
      </motion.div>

      {/* Scroll cue */}
      <motion.div
        className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2"
        style={{ color: "#475569", fontSize: "0.7rem", fontFamily: "var(--font-code)" }}
        animate={{ y: [0, 8, 0] }}
        transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
      >
        <span>scroll</span>
        <ArrowDown size={14} />
      </motion.div>
    </section>
  );
}
