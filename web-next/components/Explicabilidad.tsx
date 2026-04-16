"use client";
import { motion } from "framer-motion";
import FadeIn from "./FadeIn";

const shapFeatures = [
  { name: "vr_factura",            impact: 92, dir: "+",   color: "#FF4466" },
  { name: "log_vr_factura",        impact: 78, dir: "+",   color: "#FF4466" },
  { name: "dias_estancia",         impact: 64, dir: "+",   color: "#FFB300" },
  { name: "flag_alta_complejidad", impact: 55, dir: "+",   color: "#FFB300" },
  { name: "codigo_dx_principal",   impact: 48, dir: "+/-", color: "#00D4FF" },
  { name: "num_procedimientos",    impact: 41, dir: "+",   color: "#FFB300" },
  { name: "tipo_usuario",          impact: 34, dir: "+/-", color: "#8B5CF6" },
  { name: "edad_paciente",         impact: 28, dir: "+/-", color: "#8892B0" },
];

export default function Explicabilidad() {
  return (
    <section id="explicabilidad" className="py-28 px-6">
      <div className="max-w-5xl mx-auto">
        <FadeIn className="text-center mb-16">
          <span className="section-label">Capítulo 04 · Explicabilidad SHAP</span>
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
            Caja negra no.
            <br />
            <span className="grad-text">Cada decisión, explicada.</span>
          </h2>
          <p style={{ color: "#8892B0", fontSize: "1.05rem", maxWidth: 540, margin: "0 auto", lineHeight: 1.75 }}>
            SHAP (SHapley Additive exPlanations) cuantifica la contribución de cada variable
            a la predicción — en tiempo real, por cada factura.
          </p>
        </FadeIn>

        <div className="grid md:grid-cols-2 gap-8">
          {/* SHAP bar chart */}
          <FadeIn>
            <div className="rounded-2xl p-7 h-full" style={{ background: "#10111A", border: "1px solid #1C1E2E" }}>
              <p
                style={{
                  fontFamily: "var(--font-code)",
                  fontSize: "0.65rem",
                  color: "#3D4461",
                  marginBottom: "1.75rem",
                  letterSpacing: "0.1em",
                  textTransform: "uppercase",
                }}
              >
                SHAP · Importancia media |valor|
              </p>
              <div className="space-y-4">
                {shapFeatures.map((f, i) => (
                  <div key={f.name}>
                    <div className="flex items-center justify-between mb-1.5">
                      <span style={{ fontFamily: "var(--font-code)", fontSize: "0.7rem", color: "#8892B0" }}>
                        {f.name}
                      </span>
                      <span style={{ fontFamily: "var(--font-code)", fontSize: "0.65rem", color: f.color, fontWeight: 600 }}>
                        {f.dir === "+/-" ? "±" : f.dir}
                      </span>
                    </div>
                    <div className="h-1.5 rounded-full overflow-hidden" style={{ background: "#0C0D12" }}>
                      <motion.div
                        initial={{ width: 0 }}
                        whileInView={{ width: `${f.impact}%` }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.8, delay: i * 0.06, ease: "easeOut" }}
                        className="h-full rounded-full"
                        style={{ background: `linear-gradient(90deg, ${f.color}50, ${f.color})` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </FadeIn>

          {/* Cards */}
          <FadeIn delay={0.12}>
            <div className="space-y-4 h-full flex flex-col">
              <div className="rounded-2xl p-6" style={{ background: "#10111A", border: "1px solid #1C1E2E" }}>
                <h3
                  style={{
                    fontFamily: "var(--font-body)",
                    fontSize: "1rem",
                    fontWeight: 700,
                    color: "#00D4FF",
                    marginBottom: "0.6rem",
                  }}
                >
                  Global: Top features del modelo
                </h3>
                <p style={{ color: "#8892B0", fontSize: "0.85rem", lineHeight: 1.7 }}>
                  El valor de factura y su transformación logarítmica dominan la predicción.
                  Las facturas de alto valor y larga estancia tienen mayor riesgo de glosa.
                </p>
              </div>
              <div className="rounded-2xl p-6" style={{ background: "#10111A", border: "1px solid #1C1E2E" }}>
                <h3
                  style={{
                    fontFamily: "var(--font-body)",
                    fontSize: "1rem",
                    fontWeight: 700,
                    color: "#0066FF",
                    marginBottom: "0.6rem",
                  }}
                >
                  Local: Por predicción individual
                </h3>
                <p style={{ color: "#8892B0", fontSize: "0.85rem", lineHeight: 1.7 }}>
                  Cada llamada a la API devuelve los top-5 factores SHAP que explican
                  por qué esta factura específica fue clasificada como Glosada.
                </p>
              </div>

              {/* Code block */}
              <div
                className="rounded-2xl p-5 flex-1"
                style={{
                  background: "#080A10",
                  border: "1px solid rgba(0,212,255,0.15)",
                  fontFamily: "var(--font-code)",
                }}
              >
                <p style={{ fontSize: "0.62rem", color: "#3D4461", marginBottom: "0.75rem", letterSpacing: "0.05em" }}>
                  // API response example
                </p>
                <p style={{ fontSize: "0.75rem", color: "#8892B0", lineHeight: 2 }}>
                  {`{`}<br />
                  &nbsp;&nbsp;<span style={{ color: "#8892B0" }}>"prediction"</span>:{" "}
                  <span style={{ color: "#FFB300" }}>"Glosada"</span>,<br />
                  &nbsp;&nbsp;<span style={{ color: "#8892B0" }}>"confidence"</span>:{" "}
                  <span style={{ color: "#00FF88" }}>0.891</span>,<br />
                  &nbsp;&nbsp;<span style={{ color: "#8892B0" }}>"shap_top5"</span>:{" "}
                  <span style={{ color: "#00D4FF" }}>[...]</span><br />
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
