"use client";
import { motion } from "framer-motion";
import { AlertTriangle, RefreshCw, Activity } from "lucide-react";
import FadeIn from "./FadeIn";

const driftMetrics = [
  { label: "PSI threshold", value: "0.20", note: "Population Stability Index", color: "#F59E0B" },
  { label: "Ventana de monitoreo", value: "30d", note: "Revisión mensual automática", color: "#3B82F6" },
  { label: "Trigger reentrenamiento", value: "≥ 3", note: "Features con drift simultáneo", color: "#EF4444" },
];

export default function Drift() {
  return (
    <section className="py-24 px-6" style={{ background: "#0A0E1A" }}>
      <div className="max-w-5xl mx-auto">
        <FadeIn className="text-center mb-14">
          <span
            className="inline-block text-xs px-4 py-1.5 rounded-full mb-5"
            style={{ background: "#0E1223", border: "1px solid rgba(239,68,68,0.3)", color: "#EF4444", fontFamily: "var(--font-code)" }}
          >
            MLOps · Concept Drift
          </span>
          <h2 style={{ fontFamily: "var(--font-heading)", fontSize: "clamp(2rem,5vw,3.2rem)", fontWeight: 600, lineHeight: 1.15, marginBottom: "1rem" }}>
            El modelo se mantiene<br />al día automáticamente.
          </h2>
          <p style={{ color: "#94A3B8", fontSize: "1.1rem", maxWidth: 560, margin: "0 auto", lineHeight: 1.7 }}>
            Los patrones de facturación cambian con cada reforma normativa.
            El sistema detecta drift y dispara reentrenamiento sin intervención manual.
          </p>
        </FadeIn>

        <div className="grid md:grid-cols-2 gap-8 items-start">
          {/* Alert mockup */}
          <FadeIn>
            <motion.div
              className="rounded-2xl p-6"
              style={{ background: "#131929", border: "1px solid rgba(239,68,68,0.25)" }}
              animate={{ boxShadow: ["0 0 0px rgba(239,68,68,0)", "0 0 30px rgba(239,68,68,0.12)", "0 0 0px rgba(239,68,68,0)"] }}
              transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
            >
              <div className="flex items-center gap-3 mb-5">
                <div className="w-9 h-9 rounded-lg flex items-center justify-center" style={{ background: "rgba(239,68,68,0.15)", border: "1px solid rgba(239,68,68,0.3)" }}>
                  <AlertTriangle size={18} color="#EF4444" />
                </div>
                <div>
                  <p style={{ fontWeight: 600, fontSize: "0.9rem" }}>Alerta de Drift Detectado</p>
                  <p style={{ color: "#94A3B8", fontSize: "0.72rem", fontFamily: "var(--font-code)" }}>2026-04-13 · 3 features afectadas</p>
                </div>
              </div>

              <div className="space-y-3 mb-5">
                {["vr_factura (PSI: 0.34)", "dias_estancia (PSI: 0.28)", "num_procedimientos (PSI: 0.22)"].map((f, i) => (
                  <div key={f} className="flex items-center justify-between rounded-lg px-4 py-2.5"
                    style={{ background: "#0E1223", border: "1px solid rgba(239,68,68,0.15)" }}>
                    <span style={{ fontFamily: "var(--font-code)", fontSize: "0.72rem", color: "#94A3B8" }}>{f}</span>
                    <span style={{ fontSize: "0.65rem", color: "#EF4444", background: "rgba(239,68,68,0.1)", padding: "2px 8px", borderRadius: 4 }}>DRIFT</span>
                  </div>
                ))}
              </div>

              <div className="rounded-lg px-4 py-3 flex items-center gap-3"
                style={{ background: "rgba(34,197,94,0.08)", border: "1px solid rgba(34,197,94,0.2)" }}>
                <RefreshCw size={14} color="#22C55E" />
                <span style={{ fontSize: "0.78rem", color: "#22C55E" }}>Reentrenamiento programado para el 2026-05-01</span>
              </div>
            </motion.div>
          </FadeIn>

          {/* Metrics */}
          <FadeIn delay={0.15}>
            <div className="space-y-4">
              {driftMetrics.map((m) => (
                <div key={m.label} className="rounded-xl p-5 flex items-center gap-5"
                  style={{ background: "#131929", border: "1px solid #1e2d45" }}>
                  <p style={{ fontFamily: "var(--font-code)", fontSize: "1.6rem", fontWeight: 700, color: m.color, flexShrink: 0, minWidth: "4.5rem" }}>{m.value}</p>
                  <div>
                    <p style={{ fontWeight: 600, fontSize: "0.9rem", marginBottom: "0.2rem" }}>{m.label}</p>
                    <p style={{ color: "#94A3B8", fontSize: "0.78rem" }}>{m.note}</p>
                  </div>
                </div>
              ))}

              <div className="rounded-xl p-5" style={{ background: "#131929", border: "1px solid #1e2d45" }}>
                <div className="flex items-center gap-2 mb-3">
                  <Activity size={14} color="#6366F1" />
                  <span style={{ fontWeight: 600, fontSize: "0.9rem" }}>Ciclo de vida del modelo</span>
                </div>
                <div className="flex items-center gap-2 text-xs" style={{ color: "#94A3B8" }}>
                  {["Monitoreo", "→", "Drift?", "→", "Retrain", "→", "Validar", "→", "Deploy"].map((s, i) => (
                    <span key={i} style={{ color: s === "→" ? "#334155" : s === "Drift?" ? "#EF4444" : s === "Retrain" ? "#F59E0B" : s === "Deploy" ? "#22C55E" : "#94A3B8" }}>{s}</span>
                  ))}
                </div>
              </div>
            </div>
          </FadeIn>
        </div>
      </div>
    </section>
  );
}
