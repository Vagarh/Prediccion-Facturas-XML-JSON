"use client";
import { motion } from "framer-motion";
import { AlertTriangle, RefreshCw, Activity } from "lucide-react";
import FadeIn from "./FadeIn";

const driftMetrics = [
  { label: "PSI threshold",         value: "0.20", note: "Population Stability Index",    color: "#FFB300" },
  { label: "Ventana de monitoreo",   value: "30d",  note: "Revisión mensual automática",  color: "#00D4FF" },
  { label: "Trigger reentrenamiento",value: "≥ 3",  note: "Features con drift simultáneo", color: "#FF4466" },
];

export default function Drift() {
  return (
    <section className="py-28 px-6" style={{ background: "#080810" }}>
      <div className="max-w-5xl mx-auto">
        <FadeIn className="text-center mb-14">
          <span
            className="section-label"
            style={{
              border: "1px solid rgba(255,68,102,0.25)",
              color: "#FF4466",
              background: "rgba(255,68,102,0.05)",
            }}
          >
            MLOps · Concept Drift
          </span>
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
            El modelo se mantiene
            <br />
            <span className="grad-text">al día automáticamente.</span>
          </h2>
          <p style={{ color: "#8892B0", fontSize: "1.05rem", maxWidth: 540, margin: "0 auto", lineHeight: 1.75 }}>
            Los patrones de facturación cambian con cada reforma normativa.
            El sistema detecta drift y dispara reentrenamiento sin intervención manual.
          </p>
        </FadeIn>

        <div className="grid md:grid-cols-2 gap-8 items-start">
          {/* Alert card */}
          <FadeIn>
            <motion.div
              className="rounded-2xl p-7"
              style={{ background: "#10111A", border: "1px solid rgba(255,68,102,0.2)" }}
              animate={{
                boxShadow: [
                  "0 0 0px rgba(255,68,102,0)",
                  "0 0 40px rgba(255,68,102,0.1)",
                  "0 0 0px rgba(255,68,102,0)",
                ],
              }}
              transition={{ duration: 3.5, repeat: Infinity, ease: "easeInOut" }}
            >
              <div className="flex items-center gap-3 mb-6">
                <div
                  className="w-9 h-9 rounded-xl flex items-center justify-center"
                  style={{ background: "rgba(255,68,102,0.12)", border: "1px solid rgba(255,68,102,0.25)", color: "#FF4466" }}
                >
                  <AlertTriangle size={16} />
                </div>
                <div>
                  <p style={{ fontWeight: 700, fontSize: "0.9rem" }}>Alerta de Drift Detectado</p>
                  <p style={{ color: "#8892B0", fontSize: "0.7rem", fontFamily: "var(--font-code)", marginTop: 2 }}>
                    2026-04-13 · 3 features afectadas
                  </p>
                </div>
              </div>

              <div className="space-y-2.5 mb-6">
                {["vr_factura (PSI: 0.34)", "dias_estancia (PSI: 0.28)", "num_procedimientos (PSI: 0.22)"].map((f) => (
                  <div
                    key={f}
                    className="flex items-center justify-between rounded-xl px-4 py-2.5"
                    style={{ background: "#0C0D12", border: "1px solid rgba(255,68,102,0.12)" }}
                  >
                    <span style={{ fontFamily: "var(--font-code)", fontSize: "0.72rem", color: "#8892B0" }}>{f}</span>
                    <span
                      style={{
                        fontSize: "0.6rem",
                        color: "#FF4466",
                        background: "rgba(255,68,102,0.1)",
                        padding: "2px 8px",
                        borderRadius: 4,
                        fontFamily: "var(--font-code)",
                        letterSpacing: "0.05em",
                      }}
                    >
                      DRIFT
                    </span>
                  </div>
                ))}
              </div>

              <div
                className="rounded-xl px-4 py-3 flex items-center gap-3"
                style={{ background: "rgba(0,255,136,0.06)", border: "1px solid rgba(0,255,136,0.18)" }}
              >
                <RefreshCw size={13} color="#00FF88" />
                <span style={{ fontSize: "0.78rem", color: "#00FF88" }}>
                  Reentrenamiento programado para el 2026-05-01
                </span>
              </div>
            </motion.div>
          </FadeIn>

          {/* Metrics + lifecycle */}
          <FadeIn delay={0.12}>
            <div className="space-y-4">
              {driftMetrics.map((m) => (
                <div
                  key={m.label}
                  className="rounded-xl p-5 flex items-center gap-5"
                  style={{ background: "#10111A", border: "1px solid #1C1E2E" }}
                >
                  <p
                    style={{
                      fontFamily: "var(--font-code)",
                      fontSize: "1.7rem",
                      fontWeight: 700,
                      color: m.color,
                      flexShrink: 0,
                      minWidth: "5rem",
                      letterSpacing: "-0.02em",
                    }}
                  >
                    {m.value}
                  </p>
                  <div>
                    <p style={{ fontWeight: 700, fontSize: "0.88rem", marginBottom: "0.2rem" }}>{m.label}</p>
                    <p style={{ color: "#8892B0", fontSize: "0.75rem" }}>{m.note}</p>
                  </div>
                </div>
              ))}

              <div className="rounded-xl p-5" style={{ background: "#10111A", border: "1px solid #1C1E2E" }}>
                <div className="flex items-center gap-2 mb-3">
                  <Activity size={13} color="#8B5CF6" />
                  <span style={{ fontWeight: 700, fontSize: "0.88rem" }}>Ciclo de vida del modelo</span>
                </div>
                <div className="flex flex-wrap items-center gap-1.5 text-xs">
                  {[
                    ["Monitoreo", "#8892B0"],
                    ["→", "#2A2D3E"],
                    ["Drift?", "#FF4466"],
                    ["→", "#2A2D3E"],
                    ["Retrain", "#FFB300"],
                    ["→", "#2A2D3E"],
                    ["Validar", "#00D4FF"],
                    ["→", "#2A2D3E"],
                    ["Deploy", "#00FF88"],
                  ].map(([s, col], i) => (
                    <span key={i} style={{ color: col, fontFamily: "var(--font-code)", fontWeight: s === "→" ? 400 : 600 }}>
                      {s}
                    </span>
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
