"use client";
import { motion } from "framer-motion";
import { CheckCircle2, TrendingUp } from "lucide-react";
import FadeIn from "./FadeIn";

const rows = [
  { metric: "Recall Glosada", v1: "0.726", v2: "0.8961", better: true, note: "Meta ≥ 0.80 ✓", color: "#22C55E" },
  { metric: "F1-macro",       v1: "0.749", v2: "0.7114", better: false, note: "",              color: "#3B82F6" },
  { metric: "AUC-ROC",        v1: "0.9078",v2: "0.9028", better: false, note: "",              color: "#6366F1" },
  { metric: "Precisión macro",v1: "0.761", v2: "0.734",  better: false, note: "",              color: "#94A3B8" },
];

const bars = [
  { label: "Recall Glosada", v1: 72.6, v2: 89.6, color: "#22C55E" },
  { label: "AUC-ROC × 100",  v1: 90.8, v2: 90.3, color: "#6366F1" },
  { label: "F1-macro × 100", v1: 74.9, v2: 71.1, color: "#3B82F6" },
];

export default function Resultados() {
  return (
    <section id="resultados" className="py-24 px-6">
      <div className="max-w-5xl mx-auto">
        <FadeIn className="text-center mb-16">
          <span
            className="inline-block text-xs px-4 py-1.5 rounded-full mb-5"
            style={{ background: "#0E1223", border: "1px solid #1e2d45", color: "#94A3B8", fontFamily: "var(--font-code)" }}
          >
            Capítulo 3 · Resultados
          </span>
          <h2 style={{ fontFamily: "var(--font-heading)", fontSize: "clamp(2rem,5vw,3.2rem)", fontWeight: 600, lineHeight: 1.15, marginBottom: "1rem" }}>
            Del 72% al 89%<br />de Recall Glosada.
          </h2>
          <p style={{ color: "#94A3B8", fontSize: "1.1rem", maxWidth: 560, margin: "0 auto", lineHeight: 1.7 }}>
            El modelo v2 supera la meta crítica de 0.80 — detecta 9 de cada 10 facturas glosadas
            antes de pagar.
          </p>
        </FadeIn>

        {/* Bar comparison */}
        <FadeIn className="mb-10">
          <div className="rounded-2xl p-8" style={{ background: "#131929", border: "1px solid #1e2d45" }}>
            <div className="space-y-7">
              {bars.map((b) => (
                <div key={b.label}>
                  <div className="flex items-center justify-between mb-2">
                    <span style={{ fontSize: "0.8rem", color: "#94A3B8" }}>{b.label}</span>
                    <span style={{ fontFamily: "var(--font-code)", fontSize: "0.75rem", color: b.color }}>{b.v2.toFixed(1)}%</span>
                  </div>
                  <div className="relative h-3 rounded-full overflow-hidden" style={{ background: "#0E1223" }}>
                    {/* v1 bar */}
                    <motion.div
                      initial={{ width: 0 }}
                      whileInView={{ width: `${b.v1}%` }}
                      viewport={{ once: true }}
                      transition={{ duration: 1, delay: 0.2, ease: "easeOut" }}
                      className="absolute inset-y-0 left-0 rounded-full"
                      style={{ background: "#1e2d45" }}
                    />
                    {/* v2 bar */}
                    <motion.div
                      initial={{ width: 0 }}
                      whileInView={{ width: `${b.v2}%` }}
                      viewport={{ once: true }}
                      transition={{ duration: 1.1, delay: 0.4, ease: "easeOut" }}
                      className="absolute inset-y-0 left-0 rounded-full"
                      style={{ background: `linear-gradient(90deg, ${b.color}80, ${b.color})` }}
                    />
                  </div>
                  <div className="flex gap-4 mt-1">
                    <span style={{ fontSize: "0.65rem", color: "#475569" }}>v1: {b.v1}%</span>
                    <span style={{ fontSize: "0.65rem", color: b.color }}>v2: {b.v2}%</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </FadeIn>

        {/* Table */}
        <FadeIn>
          <div className="rounded-2xl overflow-hidden" style={{ border: "1px solid #1e2d45" }}>
            <table className="w-full text-sm" style={{ borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ background: "#0E1223", borderBottom: "1px solid #1e2d45" }}>
                  <th className="text-left py-3 px-5" style={{ color: "#94A3B8", fontWeight: 500, fontFamily: "var(--font-code)", fontSize: "0.7rem" }}>Métrica</th>
                  <th className="text-center py-3 px-5" style={{ color: "#94A3B8", fontWeight: 500, fontFamily: "var(--font-code)", fontSize: "0.7rem" }}>XGBoost v1</th>
                  <th className="text-center py-3 px-5" style={{ color: "#22C55E", fontWeight: 500, fontFamily: "var(--font-code)", fontSize: "0.7rem" }}>XGBoost v2</th>
                  <th className="text-left py-3 px-5" style={{ color: "#94A3B8", fontWeight: 500, fontFamily: "var(--font-code)", fontSize: "0.7rem" }}>Nota</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => (
                  <tr
                    key={r.metric}
                    style={{ background: i % 2 === 0 ? "#131929" : "#0E1223", borderBottom: "1px solid #1e2d45" }}
                  >
                    <td className="py-3.5 px-5" style={{ color: "#CBD5E1", fontWeight: 500 }}>{r.metric}</td>
                    <td className="py-3.5 px-5 text-center" style={{ color: "#64748B", fontFamily: "var(--font-code)" }}>{r.v1}</td>
                    <td className="py-3.5 px-5 text-center">
                      <span style={{ fontFamily: "var(--font-code)", fontWeight: 600, color: r.better ? "#22C55E" : "#94A3B8" }}>
                        {r.v2}
                        {r.better && <TrendingUp size={12} style={{ display: "inline", marginLeft: 4, verticalAlign: "middle" }} />}
                      </span>
                    </td>
                    <td className="py-3.5 px-5">
                      {r.note && (
                        <span className="flex items-center gap-1.5 text-xs" style={{ color: "#22C55E" }}>
                          <CheckCircle2 size={12} />
                          {r.note}
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p style={{ fontSize: "0.7rem", color: "#475569", marginTop: "0.75rem", textAlign: "center", fontFamily: "var(--font-code)" }}>
            633.180 facturas · División temporal · RANDOM_STATE=42
          </p>
        </FadeIn>
      </div>
    </section>
  );
}
