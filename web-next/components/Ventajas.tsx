"use client";
import { motion } from "framer-motion";
import { CheckCircle2, XCircle } from "lucide-react";
import FadeIn from "./FadeIn";
import { compact, dataset } from "@/lib/metrics";

/** Meses cubiertos por la ventana de datos, calculados de los extremos reales. */
const monthsCovered = (() => {
  const [ys, ms] = dataset.windowStart.split("-").map(Number);
  const [ye, me] = dataset.windowEnd.split("-").map(Number);
  return (ye - ys) * 12 + (me - ms) + 1;
})();

const ventajas = [
  {
    title: "Datos reales de producción",
    desc: `${compact(dataset.totalInvoices)} de facturas de ${monthsCovered} meses reales de Savia Salud EPS. No hay datos sintéticos ni simulaciones.`,
    color: "#00FF88",
  },
  {
    title: "División temporal estricta",
    desc: "El modelo nunca ve datos futuros durante el entrenamiento. Simula exactamente cómo operará en producción.",
    color: "#00D4FF",
  },
  {
    title: "Explicabilidad integrada",
    desc: "Cada predicción incluye top-5 factores SHAP. El auditor entiende el por qué, no solo el resultado.",
    color: "#0066FF",
  },
  {
    title: "Privacidad garantizada",
    desc: "PrivacyGuard elimina toda PII antes de procesar. Cumple Ley 1581. Puede operar 100% on-premise.",
    color: "#8B5CF6",
  },
  {
    title: "MLOps completo",
    desc: "Drift detection automático, reentrenamiento mensual programado y CI/CD con GitHub Actions.",
    color: "#FFB300",
  },
  {
    title: "Validado contra ground truth real",
    desc: "Las etiquetas de entrenamiento vienen de la tabla real de glosas de la EPS — no de criterio subjetivo.",
    color: "#00FF88",
  },
];

const noHacemos = [
  "No reemplazamos la decisión del auditor médico",
  "No almacenamos datos personales de pacientes",
  "No dependemos de servicios externos en la nube para inferencia",
  "No usamos splits aleatorios que introduzcan data leakage",
];

export default function Ventajas() {
  return (
    <section id="ventajas" className="py-28 px-6">
      <div className="max-w-5xl mx-auto">

        <FadeIn className="text-center mb-16">
          <span className="section-label">Alcance y Diferenciadores</span>
          <h2 style={{ fontFamily: "var(--font-body)", fontSize: "clamp(2.2rem, 5vw, 3.5rem)", fontWeight: 700, lineHeight: 1.1, letterSpacing: "-0.022em", marginBottom: "1.2rem" }}>
            Qué hace este sistema.
            <br />
            <span className="grad-text">Y qué no hace.</span>
          </h2>
          <p style={{ color: "#8892B0", fontSize: "1.05rem", maxWidth: 580, margin: "0 auto", lineHeight: 1.78 }}>
            Ser explícitos sobre el alcance es parte del rigor científico.
            El sistema potencia al auditor — no lo sustituye.
          </p>
        </FadeIn>

        <div className="grid md:grid-cols-2 gap-8 mb-10">
          {/* Qué sí hacemos */}
          <FadeIn>
            <div className="rounded-2xl p-7 h-full" style={{ background: "#10111A", border: "1px solid #1C1E2E" }}>
              <p style={{ fontFamily: "var(--font-body)", fontSize: "1rem", fontWeight: 700, color: "#00FF88", marginBottom: "1.25rem" }}>
                ✓ Lo que el sistema hace
              </p>
              <div className="space-y-4">
                {ventajas.map((v, i) => (
                  <motion.div key={v.title}
                    initial={{ opacity: 0, x: -12 }} whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }} transition={{ duration: 0.35, delay: i * 0.07 }}
                    className="flex items-start gap-3"
                  >
                    <CheckCircle2 size={15} color={v.color} style={{ flexShrink: 0, marginTop: 3 }} />
                    <div>
                      <p style={{ fontWeight: 700, fontSize: "0.85rem", color: "#F0F2FF", marginBottom: "0.2rem" }}>{v.title}</p>
                      <p style={{ fontSize: "0.78rem", color: "#8892B0", lineHeight: 1.65 }}>{v.desc}</p>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          </FadeIn>

          {/* Qué no hacemos */}
          <FadeIn delay={0.1}>
            <div className="rounded-2xl p-7 h-full" style={{ background: "#10111A", border: "1px solid #1C1E2E" }}>
              <p style={{ fontFamily: "var(--font-body)", fontSize: "1rem", fontWeight: 700, color: "#FF4466", marginBottom: "1.25rem" }}>
                ✗ Lo que el sistema no hace
              </p>
              <div className="space-y-4 mb-8">
                {noHacemos.map((n, i) => (
                  <motion.div key={n}
                    initial={{ opacity: 0, x: 12 }} whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }} transition={{ duration: 0.35, delay: i * 0.07 }}
                    className="flex items-start gap-3"
                  >
                    <XCircle size={15} color="#FF4466" style={{ flexShrink: 0, marginTop: 3 }} />
                    <p style={{ fontSize: "0.85rem", color: "#8892B0", lineHeight: 1.65 }}>{n}</p>
                  </motion.div>
                ))}
              </div>

              {/* Mensaje clave */}
              <div className="rounded-xl p-5"
                style={{ background: "rgba(0,212,255,0.05)", border: "1px solid rgba(0,212,255,0.15)" }}>
                <p style={{ fontSize: "0.88rem", color: "#F0F2FF", lineHeight: 1.7, fontWeight: 500 }}>
                  El auditor médico sigue siendo quien toma la decisión final.
                  El modelo le da{" "}
                  <span style={{ color: "#00D4FF" }}>contexto, priorización y evidencia</span>
                  {" "}— no órdenes.
                </p>
              </div>

              {/* Comparativa con IA previa */}
              <div className="rounded-xl p-5 mt-4"
                style={{ background: "rgba(255,179,0,0.05)", border: "1px solid rgba(255,179,0,0.15)" }}>
                <p style={{ fontFamily: "var(--font-code)", fontSize: "0.65rem", color: "#FFB300", marginBottom: "0.5rem", letterSpacing: "0.05em" }}>
                  HALLAZGO CRÍTICO DE VALIDACIÓN
                </p>
                <p style={{ fontSize: "0.82rem", color: "#8892B0", lineHeight: 1.7 }}>
                  Al validar contra datos reales, encontramos que{" "}
                  <strong style={{ color: "#FFB300" }}>la IA previa de la EPS tenía sesgos significativos</strong>.
                  Nuestro modelo, entrenado con metodología rigurosa y ground truth real, corrige esas desviaciones.
                </p>
              </div>
            </div>
          </FadeIn>
        </div>

        {/* Stats de confianza */}
        <FadeIn>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { v: String(monthsCovered), label: "Meses de datos reales",  color: "#00D4FF" },
              { v: "554",   label: "Tablas de BD mapeadas",       color: "#8B5CF6" },
              { v: String(dataset.featureCount), label: "Variables de entrada", color: "#0066FF" },
              { v: "100%",  label: "Local · sin datos a la nube", color: "#00FF88" },
            ].map((s) => (
              <div key={s.label} className="rounded-xl p-5 text-center"
                style={{ background: "#10111A", border: "1px solid #1C1E2E", position: "relative", overflow: "hidden" }}>
                <div style={{ position: "absolute", top: 0, left: "20%", right: "20%", height: "1px", background: `linear-gradient(90deg, transparent, ${s.color}60, transparent)` }} />
                <p style={{ fontFamily: "var(--font-code)", fontSize: "1.8rem", fontWeight: 700, color: s.color, letterSpacing: "-0.03em", lineHeight: 1 }}>{s.v}</p>
                <p style={{ fontSize: "0.72rem", color: "#8892B0", marginTop: "0.5rem", lineHeight: 1.4 }}>{s.label}</p>
              </div>
            ))}
          </div>
        </FadeIn>

      </div>
    </section>
  );
}
