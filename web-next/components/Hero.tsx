"use client";
import { motion } from "framer-motion";
import { ArrowDown, Zap } from "lucide-react";

const metrics = [
  { value: "633K",   label: "Facturas analizadas", color: "#00D4FF" },
  { value: "89.6%",  label: "Recall Glosada",      color: "#00FF88" },
  { value: "0.9028", label: "AUC-ROC",              color: "#0066FF" },
  { value: "380",    label: "Tests automatizados",  color: "#8B5CF6" },
];

export default function Hero() {
  return (
    <section className="relative min-h-[100dvh] flex flex-col items-center justify-center text-center px-6 pt-20 overflow-hidden">
      {/* Grid background */}
      <div className="absolute inset-0 grid-bg pointer-events-none" />

      {/* Deep glow orbs — MiniMax style */}
      <motion.div
        className="absolute rounded-full pointer-events-none"
        style={{
          width: 700,
          height: 700,
          top: "-10%",
          left: "-15%",
          background: "radial-gradient(circle, rgba(0,102,255,0.06) 0%, transparent 65%)",
        }}
        animate={{ scale: [1, 1.1, 1], opacity: [0.5, 0.8, 0.5] }}
        transition={{ duration: 10, repeat: Infinity, ease: "easeInOut" }}
      />
      <motion.div
        className="absolute rounded-full pointer-events-none"
        style={{
          width: 600,
          height: 600,
          bottom: "-5%",
          right: "-10%",
          background: "radial-gradient(circle, rgba(0,212,255,0.05) 0%, transparent 65%)",
        }}
        animate={{ scale: [1, 1.15, 1], opacity: [0.4, 0.7, 0.4] }}
        transition={{ duration: 12, repeat: Infinity, ease: "easeInOut", delay: 3 }}
      />
      <motion.div
        className="absolute rounded-full pointer-events-none"
        style={{
          width: 400,
          height: 400,
          top: "40%",
          left: "50%",
          transform: "translate(-50%,-50%)",
          background: "radial-gradient(circle, rgba(139,92,246,0.04) 0%, transparent 65%)",
        }}
        animate={{ scale: [1, 1.2, 1] }}
        transition={{ duration: 8, repeat: Infinity, ease: "easeInOut", delay: 1.5 }}
      />

      {/* Content */}
      <div className="relative max-w-5xl mx-auto">
        {/* Badge */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="flex justify-center mb-8"
        >
          <span className="badge">
            <Zap size={10} />
            Maestría en Ciencia de Datos · Universidad EAFIT · 2026
          </span>
        </motion.div>

        {/* Headline */}
        <motion.h1
          initial={{ opacity: 0, y: 32 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.15 }}
          style={{
            fontFamily: "var(--font-body)",
            fontSize: "clamp(3rem, 8vw, 5.5rem)",
            fontWeight: 700,
            lineHeight: 1.05,
            letterSpacing: "-0.02em",
            marginBottom: "1.5rem",
          }}
        >
          Analítica Predictiva
          <br />
          para{" "}
          <span className="grad-text">Auditoría Prestacional</span>
        </motion.h1>

        {/* Sub */}
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.3 }}
          style={{
            fontSize: "clamp(1rem, 2vw, 1.2rem)",
            color: "#8892B0",
            maxWidth: 600,
            margin: "0 auto 2.5rem",
            lineHeight: 1.75,
            fontWeight: 400,
          }}
        >
          Un modelo de inteligencia artificial que predice, en el momento de radicación,
          si una factura médica será{" "}
          <strong style={{ color: "#F0F2FF", fontWeight: 600 }}>Auditada</strong>,{" "}
          <strong style={{ color: "#F0F2FF", fontWeight: 600 }}>Glosada</strong> o{" "}
          <strong style={{ color: "#F0F2FF", fontWeight: 600 }}>Devuelta</strong>.
        </motion.p>

        {/* CTAs */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.45 }}
          className="flex flex-col sm:flex-row gap-4 justify-center"
        >
          <a
            href="#resultados"
            style={{
              display: "inline-block",
              padding: "14px 32px",
              borderRadius: "8px",
              fontWeight: 600,
              fontSize: "0.9rem",
              background: "linear-gradient(135deg, #00D4FF, #0066FF)",
              color: "#060608",
              boxShadow: "0 0 32px rgba(0,212,255,0.35)",
              textDecoration: "none",
              letterSpacing: "0.01em",
              transition: "box-shadow 0.2s",
            }}
          >
            Ver Resultados
          </a>
          <a
            href="#video"
            style={{
              display: "inline-block",
              padding: "14px 32px",
              borderRadius: "8px",
              fontWeight: 500,
              fontSize: "0.9rem",
              background: "transparent",
              color: "#F0F2FF",
              border: "1px solid #1C1E2E",
              textDecoration: "none",
              transition: "border-color 0.2s",
            }}
          >
            Video Explicativo
          </a>
        </motion.div>
      </div>

      {/* Metric cards */}
      <motion.div
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.65 }}
        className="relative mt-20 grid grid-cols-2 md:grid-cols-4 gap-3 max-w-3xl w-full"
      >
        {metrics.map((m, i) => (
          <motion.div
            key={m.label}
            whileHover={{ y: -4 }}
            transition={{ duration: 0.2 }}
            style={{
              textAlign: "center",
              borderRadius: "12px",
              padding: "20px 12px",
              background: "#10111A",
              border: "1px solid #1C1E2E",
              position: "relative",
              overflow: "hidden",
            }}
          >
            {/* top accent line */}
            <div
              style={{
                position: "absolute",
                top: 0,
                left: "20%",
                right: "20%",
                height: "1px",
                background: `linear-gradient(90deg, transparent, ${m.color}60, transparent)`,
              }}
            />
            <p
              style={{
                fontFamily: "var(--font-code)",
                fontSize: "1.75rem",
                fontWeight: 600,
                color: m.color,
                lineHeight: 1,
                letterSpacing: "-0.02em",
              }}
            >
              {m.value}
            </p>
            <p style={{ fontSize: "0.7rem", color: "#8892B0", marginTop: "0.5rem", fontWeight: 500 }}>
              {m.label}
            </p>
          </motion.div>
        ))}
      </motion.div>

      {/* Scroll cue */}
      <motion.div
        className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2"
        style={{ color: "#3D4461", fontSize: "0.65rem", fontFamily: "var(--font-code)", letterSpacing: "0.1em" }}
        animate={{ y: [0, 8, 0] }}
        transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
      >
        <span>SCROLL</span>
        <ArrowDown size={12} />
      </motion.div>
    </section>
  );
}
