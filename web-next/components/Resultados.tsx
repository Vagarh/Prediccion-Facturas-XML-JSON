"use client";
import { motion } from "framer-motion";
import { CheckCircle2, TrendingUp } from "lucide-react";
import FadeIn from "./FadeIn";

const rows = [
  { metric: "Recall Glosada",  v1: "0.726", v2: "0.8961", better: true,  note: "Meta ≥ 0.80 ✓", color: "#00FF88" },
  { metric: "F1-macro",        v1: "0.749", v2: "0.7114", better: false, note: "",               color: "#00D4FF" },
  { metric: "AUC-ROC",         v1: "0.9078",v2: "0.9028", better: false, note: "",               color: "#0066FF" },
  { metric: "Precisión macro", v1: "0.761", v2: "0.734",  better: false, note: "",               color: "#8892B0" },
];

const bars = [
  { label: "Recall Glosada", v1: 72.6, v2: 89.6, color: "#00FF88" },
  { label: "AUC-ROC × 100",  v1: 90.8, v2: 90.3, color: "#0066FF" },
  { label: "F1-macro × 100", v1: 74.9, v2: 71.1, color: "#00D4FF" },
];

export default function Resultados() {
  return (
    <section id="resultados" className="py-28 px-6">
      <div className="max-w-5xl mx-auto">
        <FadeIn className="text-center mb-16">
          <span className="section-label">Capítulo 03 · Resultados</span>
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
            Del 72% al{" "}
            <span className="grad-text">89%</span>
            <br />
            de Recall Glosada.
          </h2>
          <p style={{ color: "#8892B0", fontSize: "1.05rem", maxWidth: 540, margin: "0 auto", lineHeight: 1.75 }}>
            El modelo v2 supera la meta crítica de 0.80 — detecta 9 de cada 10 facturas glosadas
            antes de pagar.
          </p>
        </FadeIn>

        {/* Bar comparison */}
        <FadeIn className="mb-8">
          <div
            className="rounded-2xl p-8"
            style={{ background: "#10111A", border: "1px solid #1C1E2E" }}
          >
            <div className="space-y-7">
              {bars.map((b) => (
                <div key={b.label}>
                  <div className="flex items-center justify-between mb-2">
                    <span style={{ fontSize: "0.78rem", color: "#8892B0", fontWeight: 500 }}>{b.label}</span>
                    <span style={{ fontFamily: "var(--font-code)", fontSize: "0.75rem", color: b.color, fontWeight: 600 }}>
                      {b.v2.toFixed(1)}%
                    </span>
                  </div>
                  <div className="relative h-2 rounded-full overflow-hidden" style={{ background: "#0C0D12" }}>
                    {/* v1 bar */}
                    <motion.div
                      initial={{ width: 0 }}
                      whileInView={{ width: `${b.v1}%` }}
                      viewport={{ once: true }}
                      transition={{ duration: 0.9, delay: 0.2, ease: "easeOut" }}
                      className="absolute inset-y-0 left-0 rounded-full"
                      style={{ background: "#1C1E2E" }}
                    />
                    {/* v2 bar */}
                    <motion.div
                      initial={{ width: 0 }}
                      whileInView={{ width: `${b.v2}%` }}
                      viewport={{ once: true }}
                      transition={{ duration: 1, delay: 0.4, ease: "easeOut" }}
                      className="absolute inset-y-0 left-0 rounded-full"
                      style={{ background: `linear-gradient(90deg, ${b.color}70, ${b.color})` }}
                    />
                  </div>
                  <div className="flex gap-4 mt-1.5">
                    <span style={{ fontSize: "0.62rem", color: "#3D4461", fontFamily: "var(--font-code)" }}>v1: {b.v1}%</span>
                    <span style={{ fontSize: "0.62rem", color: b.color, fontFamily: "var(--font-code)" }}>v2: {b.v2}%</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </FadeIn>

        {/* Table */}
        <FadeIn>
          <div className="rounded-2xl overflow-hidden" style={{ border: "1px solid #1C1E2E" }}>
            <table className="w-full text-sm" style={{ borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ background: "#0C0D12", borderBottom: "1px solid #1C1E2E" }}>
                  <th className="text-left py-3.5 px-5" style={{ color: "#8892B0", fontWeight: 600, fontFamily: "var(--font-code)", fontSize: "0.65rem", letterSpacing: "0.1em", textTransform: "uppercase" }}>Métrica</th>
                  <th className="text-center py-3.5 px-5" style={{ color: "#8892B0", fontWeight: 600, fontFamily: "var(--font-code)", fontSize: "0.65rem", letterSpacing: "0.1em", textTransform: "uppercase" }}>XGBoost v1</th>
                  <th className="text-center py-3.5 px-5" style={{ color: "#00D4FF", fontWeight: 600, fontFamily: "var(--font-code)", fontSize: "0.65rem", letterSpacing: "0.1em", textTransform: "uppercase" }}>XGBoost v2</th>
                  <th className="text-left py-3.5 px-5" style={{ color: "#8892B0", fontWeight: 600, fontFamily: "var(--font-code)", fontSize: "0.65rem", letterSpacing: "0.1em", textTransform: "uppercase" }}>Nota</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => (
                  <tr
                    key={r.metric}
                    style={{
                      background: i % 2 === 0 ? "#10111A" : "#0C0D12",
                      borderBottom: "1px solid #1C1E2E",
                    }}
                  >
                    <td className="py-4 px-5" style={{ color: "#C8D0E8", fontWeight: 600, fontSize: "0.85rem" }}>{r.metric}</td>
                    <td className="py-4 px-5 text-center" style={{ color: "#3D4461", fontFamily: "var(--font-code)", fontSize: "0.85rem" }}>{r.v1}</td>
                    <td className="py-4 px-5 text-center">
                      <span style={{ fontFamily: "var(--font-code)", fontWeight: 700, color: r.better ? "#00FF88" : "#8892B0", fontSize: "0.85rem" }}>
                        {r.v2}
                        {r.better && <TrendingUp size={11} style={{ display: "inline", marginLeft: 4, verticalAlign: "middle" }} />}
                      </span>
                    </td>
                    <td className="py-4 px-5">
                      {r.note && (
                        <span className="flex items-center gap-1.5 text-xs" style={{ color: "#00FF88" }}>
                          <CheckCircle2 size={11} />
                          {r.note}
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p style={{ fontSize: "0.65rem", color: "#3D4461", marginTop: "0.75rem", textAlign: "center", fontFamily: "var(--font-code)", letterSpacing: "0.05em" }}>
            633.180 facturas · División temporal · RANDOM_STATE=42
          </p>
        </FadeIn>
      </div>
    </section>
  );
}
