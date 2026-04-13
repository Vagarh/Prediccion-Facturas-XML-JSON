"use client";
import { motion } from "framer-motion";
import FadeIn from "./FadeIn";

const shapFeatures = [
  { name: "vr_factura",           impact: 92, dir: "+" , color: "#EF4444" },
  { name: "log_vr_factura",       impact: 78, dir: "+" , color: "#EF4444" },
  { name: "dias_estancia",        impact: 64, dir: "+" , color: "#F59E0B" },
  { name: "flag_alta_complejidad",impact: 55, dir: "+" , color: "#F59E0B" },
  { name: "codigo_dx_principal",  impact: 48, dir: "+/-", color: "#3B82F6" },
  { name: "num_procedimientos",   impact: 41, dir: "+" , color: "#F59E0B" },
  { name: "tipo_usuario",         impact: 34, dir: "+/-", color: "#6366F1" },
  { name: "edad_paciente",        impact: 28, dir: "+/-", color: "#94A3B8" },
];

export default function Explicabilidad() {
  return (
    <section id="explicabilidad" className="py-24 px-6">
      <div className="max-w-5xl mx-auto">
        <FadeIn className="text-center mb-16">
          <span
            className="inline-block text-xs px-4 py-1.5 rounded-full mb-5"
            style={{ background: "#0E1223", border: "1px solid #1e2d45", color: "#94A3B8", fontFamily: "var(--font-code)" }}
          >
            Capítulo 4 · Explicabilidad SHAP
          </span>
          <h2 style={{ fontFamily: "var(--font-heading)", fontSize: "clamp(2rem,5vw,3.2rem)", fontWeight: 600, lineHeight: 1.15, marginBottom: "1rem" }}>
            Caja negra no.<br />Cada decisión, explicada.
          </h2>
          <p style={{ color: "#94A3B8", fontSize: "1.1rem", maxWidth: 560, margin: "0 auto", lineHeight: 1.7 }}>
            SHAP (SHapley Additive exPlanations) cuantifica la contribución de cada variable
            a la predicción — en tiempo real, por cada factura.
          </p>
        </FadeIn>

        <div className="grid md:grid-cols-2 gap-8">
          {/* SHAP bar chart simulation */}
          <FadeIn>
            <div className="rounded-2xl p-7 h-full" style={{ background: "#131929", border: "1px solid #1e2d45" }}>
              <p style={{ fontFamily: "var(--font-code)", fontSize: "0.7rem", color: "#475569", marginBottom: "1.5rem" }}>
                SHAP · Importancia media |valor|
              </p>
              <div className="space-y-3.5">
                {shapFeatures.map((f, i) => (
                  <div key={f.name}>
                    <div className="flex items-center justify-between mb-1">
                      <span style={{ fontFamily: "var(--font-code)", fontSize: "0.7rem", color: "#94A3B8" }}>{f.name}</span>
                      <span style={{ fontFamily: "var(--font-code)", fontSize: "0.65rem", color: f.color }}>
                        {f.dir === "+/-" ? "±" : f.dir}
                      </span>
                    </div>
                    <div className="h-2 rounded-full overflow-hidden" style={{ background: "#0E1223" }}>
                      <motion.div
                        initial={{ width: 0 }}
                        whileInView={{ width: `${f.impact}%` }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.8, delay: i * 0.07, ease: "easeOut" }}
                        className="h-full rounded-full"
                        style={{ background: `linear-gradient(90deg, ${f.color}60, ${f.color})` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </FadeIn>

          {/* SHAP explanation cards */}
          <FadeIn delay={0.15}>
            <div className="space-y-4">
              <div className="rounded-xl p-5" style={{ background: "#131929", border: "1px solid #1e2d45" }}>
                <h3 style={{ fontFamily: "var(--font-heading)", fontSize: "1.1rem", fontWeight: 600, color: "#22C55E", marginBottom: "0.5rem" }}>
                  Global: Top features del modelo
                </h3>
                <p style={{ color: "#94A3B8", fontSize: "0.85rem", lineHeight: 1.6 }}>
                  El valor de factura y su transformación logarítmica dominan la predicción.
                  Las facturas de alto valor y larga estancia tienen mayor riesgo de glosa.
                </p>
              </div>
              <div className="rounded-xl p-5" style={{ background: "#131929", border: "1px solid #1e2d45" }}>
                <h3 style={{ fontFamily: "var(--font-heading)", fontSize: "1.1rem", fontWeight: 600, color: "#3B82F6", marginBottom: "0.5rem" }}>
                  Local: Por predicción individual
                </h3>
                <p style={{ color: "#94A3B8", fontSize: "0.85rem", lineHeight: 1.6 }}>
                  Cada llamada a la API devuelve los top-5 factores SHAP que explican
                  por qué esta factura específica fue clasificada como Glosada.
                </p>
              </div>
              <div className="rounded-xl p-5" style={{ background: "#0d1f14", border: "1px solid rgba(34,197,94,0.2)" }}>
                <p style={{ fontFamily: "var(--font-code)", fontSize: "0.72rem", color: "#22C55E", lineHeight: 1.7 }}>
                  <span style={{ color: "#475569" }}>// API response example</span><br />
                  {`{`}<br />
                  &nbsp;&nbsp;<span style={{ color: "#94A3B8" }}>"prediction"</span>: <span style={{ color: "#F59E0B" }}>"Glosada"</span>,<br />
                  &nbsp;&nbsp;<span style={{ color: "#94A3B8" }}>"confidence"</span>: <span style={{ color: "#22C55E" }}>0.891</span>,<br />
                  &nbsp;&nbsp;<span style={{ color: "#94A3B8" }}>"shap_top5"</span>: [{`...`}]<br />
                  {`}`}
                </p>
              </div>
            </div>
          </FadeIn>
        </div>
      </div>
    </section>
  );
}
