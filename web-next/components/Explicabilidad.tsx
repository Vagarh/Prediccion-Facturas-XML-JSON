"use client";
import { motion } from "framer-motion";
import { UserCheck, Brain } from "lucide-react";
import FadeIn from "./FadeIn";
import { shap } from "@/lib/metrics";

// Glosario legible de las variables del modelo. La importancia NO se escribe
// aquí: se deriva de la importancia SHAP global real del campeón.
const FEATURE_LABELS: Record<string, { label: string; dir: string }> = {
  log_num_items_con_autorizacion: { label: "Ítems con autorización previa (log)", dir: "+" },
  log_num_dx_distintos: { label: "Diversidad diagnóstica (log)", dir: "+" },
  log_valor_x_tasa_glosa: { label: "Riesgo compuesto: valor × tasa de glosa del prestador", dir: "+" },
  tasa_devolucion_nit: { label: "Tasa histórica de devoluciones del prestador", dir: "+" },
  num_items_con_autorizacion: { label: "Ítems con autorización previa", dir: "+" },
  num_tipos_servicio: { label: "Diversidad de tipos de servicio", dir: "+/-" },
  num_dx_distintos: { label: "Número de diagnósticos distintos", dir: "+" },
  tasa_glosa_nit: { label: "Tasa histórica de glosas del prestador", dir: "+" },
};

const RAMP = ["#FF4466", "#FF4466", "#FFB300", "#FFB300", "#00D4FF", "#00D4FF", "#8B5CF6", "#8892B0"];

const top = shap.global.slice(0, 8);
const maxImpact = top[0]?.value ?? 1;
const shapFeatures = top.map((entry, i) => ({
  name: entry.feature,
  impact: Math.round((entry.value / maxImpact) * 100),
  dir: FEATURE_LABELS[entry.feature]?.dir ?? "+",
  color: RAMP[i] ?? "#8892B0",
  label: FEATURE_LABELS[entry.feature]?.label ?? entry.feature,
}));

export default function Explicabilidad() {
  return (
    <section id="explicabilidad" className="py-28 px-6">
      <div className="max-w-5xl mx-auto">

        <FadeIn className="text-center mb-16">
          <span className="section-label">Explicabilidad</span>
          <h2 style={{ fontFamily: "var(--font-body)", fontSize: "clamp(2.2rem, 5vw, 3.5rem)", fontWeight: 700, lineHeight: 1.1, letterSpacing: "-0.022em", marginBottom: "1.2rem" }}>
            No reemplazamos al auditor.
            <br />
            <span className="grad-text">Le decimos dónde mirar.</span>
          </h2>
          <p style={{ color: "#8892B0", fontSize: "1.05rem", maxWidth: 600, margin: "0 auto", lineHeight: 1.78 }}>
            El modelo no es una caja negra. Cada predicción viene acompañada de una
            explicación en tiempo real: qué variables causaron esa clasificación y en qué medida.
            El auditor siempre tiene la última palabra.
          </p>
        </FadeIn>

        {/* Mensaje central — no reemplazar */}
        <FadeIn>
          <div className="rounded-2xl p-8 mb-10 grid md:grid-cols-2 gap-8"
            style={{ background: "#10111A", border: "1px solid rgba(0,212,255,0.15)" }}>
            <div className="flex items-start gap-4">
              <div className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0"
                style={{ background: "rgba(255,68,102,0.1)", border: "1px solid rgba(255,68,102,0.2)", color: "#FF4466" }}>
                <Brain size={18} />
              </div>
              <div>
                <p style={{ fontWeight: 700, fontSize: "0.95rem", marginBottom: "0.5rem", color: "#FF4466" }}>Sin el modelo</p>
                <p style={{ color: "#8892B0", fontSize: "0.85rem", lineHeight: 1.72 }}>
                  El auditor recibe todas las facturas en orden de llegada.
                  Dedica el mismo tiempo a una factura de $500.000 que a una de $50.000.000.
                  Las glosas más costosas se ocultan en el volumen.
                </p>
              </div>
            </div>
            <div className="flex items-start gap-4">
              <div className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0"
                style={{ background: "rgba(0,255,136,0.1)", border: "1px solid rgba(0,255,136,0.2)", color: "#00FF88" }}>
                <UserCheck size={18} />
              </div>
              <div>
                <p style={{ fontWeight: 700, fontSize: "0.95rem", marginBottom: "0.5rem", color: "#00FF88" }}>Con el modelo</p>
                <p style={{ color: "#8892B0", fontSize: "0.85rem", lineHeight: 1.72 }}>
                  El auditor ve primero las facturas con mayor probabilidad de glosa
                  y una explicación de por qué. Decide con contexto.
                  Su criterio sigue siendo el que cuenta.
                </p>
              </div>
            </div>
          </div>
        </FadeIn>

        <div className="grid md:grid-cols-2 gap-8">
          {/* SHAP chart */}
          <FadeIn>
            <div className="rounded-2xl p-7 h-full" style={{ background: "#10111A", border: "1px solid #1C1E2E" }}>
              <p style={{ fontFamily: "var(--font-code)", fontSize: "0.65rem", color: "#3D4461", marginBottom: "0.4rem", letterSpacing: "0.1em", textTransform: "uppercase" }}>
                SHAP · Importancia media global
              </p>
              <p style={{ fontSize: "0.78rem", color: "#8892B0", marginBottom: "1.5rem" }}>
                Las variables que más influyen en la predicción de glosa:
              </p>
              <div className="space-y-4">
                {shapFeatures.map((f, i) => (
                  <div key={f.name}>
                    <div className="flex items-center justify-between mb-1">
                      <div>
                        <span style={{ fontFamily: "var(--font-code)", fontSize: "0.68rem", color: "#8892B0" }}>{f.name}</span>
                        <span style={{ fontSize: "0.62rem", color: "#3D4461", marginLeft: 8 }}>{f.label}</span>
                      </div>
                      <span style={{ fontFamily: "var(--font-code)", fontSize: "0.65rem", color: f.color, fontWeight: 600 }}>
                        {f.dir === "+/-" ? "±" : f.dir}
                      </span>
                    </div>
                    <div className="h-1.5 rounded-full overflow-hidden" style={{ background: "#0C0D12" }}>
                      <motion.div
                        initial={{ width: 0 }} whileInView={{ width: `${f.impact}%` }}
                        viewport={{ once: true }} transition={{ duration: 0.8, delay: i * 0.06, ease: "easeOut" }}
                        className="h-full rounded-full"
                        style={{ background: `linear-gradient(90deg, ${f.color}50, ${f.color})` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </FadeIn>

          {/* Explicación por factura + ejemplo API */}
          <FadeIn delay={0.12}>
            <div className="space-y-4 flex flex-col h-full">
              <div className="rounded-2xl p-6" style={{ background: "#10111A", border: "1px solid #1C1E2E" }}>
                <p style={{ fontWeight: 700, fontSize: "0.95rem", color: "#00D4FF", marginBottom: "0.6rem" }}>
                  Explicación global — qué aprende el modelo
                </p>
                <p style={{ color: "#8892B0", fontSize: "0.85rem", lineHeight: 1.72 }}>
                  El valor de la factura y su transformación logarítmica son los predictores más fuertes.
                  Las facturas de alto valor, larga estancia y alta complejidad tienen mayor riesgo de glosa.
                  El diagnóstico y el tipo de usuario modulan la predicción según el contexto clínico.
                </p>
              </div>

              <div className="rounded-2xl p-6" style={{ background: "#10111A", border: "1px solid #1C1E2E" }}>
                <p style={{ fontWeight: 700, fontSize: "0.95rem", color: "#0066FF", marginBottom: "0.6rem" }}>
                  Explicación local — por factura individual
                </p>
                <p style={{ color: "#8892B0", fontSize: "0.85rem", lineHeight: 1.72 }}>
                  La API devuelve los top-5 factores SHAP específicos para esa factura.
                  El auditor ve exactamente por qué el modelo la marcó como glosada — no una regla genérica.
                </p>
              </div>

              {/* Ejemplo respuesta API con SHAP */}
              <div className="rounded-2xl p-5 flex-1"
                style={{ background: "#060608", border: "1px solid rgba(0,212,255,0.15)", fontFamily: "var(--font-code)" }}>
                <p style={{ fontSize: "0.6rem", color: "#3D4461", marginBottom: "0.75rem", letterSpacing: "0.05em" }}>
                  // SHAP explicación · FAC-2024-09871
                </p>
                <div style={{ fontSize: "0.72rem", color: "#8892B0", lineHeight: 2 }}>
                  <span style={{ color: "#8892B0" }}>"shap_top5"</span>: [<br />
                  &nbsp;&nbsp;{"{"}<span style={{ color: "#8892B0" }}>"feature"</span>: <span style={{ color: "#FFB300" }}>"log_num_items_con_autorizacion"</span>, <span style={{ color: "#8892B0" }}>"impact"</span>: <span style={{ color: "#FF4466" }}>+0.38</span>{"}"}, <br />
                  &nbsp;&nbsp;{"{"}<span style={{ color: "#8892B0" }}>"feature"</span>: <span style={{ color: "#FFB300" }}>"log_num_dx_distintos"</span>, <span style={{ color: "#8892B0" }}>"impact"</span>: <span style={{ color: "#FF4466" }}>+0.32</span>{"}"}, <br />
                  &nbsp;&nbsp;{"{"}<span style={{ color: "#8892B0" }}>"feature"</span>: <span style={{ color: "#FFB300" }}>"tasa_devolucion_nit"</span>, <span style={{ color: "#8892B0" }}>"impact"</span>: <span style={{ color: "#FF4466" }}>+0.13</span>{"}"}<br />
                  ]
                </div>
              </div>
            </div>
          </FadeIn>
        </div>

      </div>
    </section>
  );
}
