"use client";
import { motion } from "framer-motion";
import { Cloud, Database, Zap, Shield, RefreshCw, Code2 } from "lucide-react";
import FadeIn from "./FadeIn";

const layers = [
  {
    title: "Capa de Presentación",
    icon: <Code2 size={15} />,
    color: "#00D4FF",
    items: ["FastAPI REST", "Endpoints: /predict, /predict/batch, /health, /model/info", "SHAP integrado en tiempo real"],
  },
  {
    title: "Capa de Negocio",
    icon: <Zap size={15} />,
    color: "#0066FF",
    items: ["XGBoost v2 (48 features)", "PrivacyGuard · PII filtering", "Feature engineering vectorizado"],
  },
  {
    title: "Capa de Datos",
    icon: <Database size={15} />,
    color: "#8B5CF6",
    items: ["Dataset Parquet 633K facturas", "División temporal (no random split)", "Artifacts versionados"],
  },
  {
    title: "Infraestructura AWS",
    icon: <Cloud size={15} />,
    color: "#FFB300",
    items: ["ECR + ECS Fargate", "API Gateway · ALB", "S3 artifacts · CloudWatch logs"],
  },
  {
    title: "MLOps",
    icon: <RefreshCw size={15} />,
    color: "#FF4466",
    items: ["Drift detection automático", "Reentrenamiento mensual programado", "GitHub Actions CI/CD"],
  },
  {
    title: "Seguridad",
    icon: <Shield size={15} />,
    color: "#00FF88",
    items: ["100% local · Ley 1581 (Habeas Data)", "Sin datos PII en features o logs", "Docker healthcheck + HTTPS"],
  },
];

export default function Arquitectura() {
  return (
    <section id="arquitectura" className="py-28 px-6">
      <div className="max-w-5xl mx-auto">
        <FadeIn className="text-center mb-16">
          <span className="section-label">Capítulo 05 · Arquitectura</span>
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
            Producción-ready.
            <br />
            <span className="grad-text">De principio a fin.</span>
          </h2>
          <p style={{ color: "#8892B0", fontSize: "1.05rem", maxWidth: 540, margin: "0 auto", lineHeight: 1.75 }}>
            Stack técnico diseñado para operar en la nube con cumplimiento normativo,
            trazabilidad completa y reentrenamiento automático.
          </p>
        </FadeIn>

        <div className="grid md:grid-cols-3 gap-4">
          {layers.map((l, i) => (
            <FadeIn key={l.title} delay={i * 0.07}>
              <motion.div
                whileHover={{ y: -5 }}
                transition={{ duration: 0.2 }}
                className="h-full rounded-2xl p-6"
                style={{ background: "#10111A", border: "1px solid #1C1E2E", position: "relative", overflow: "hidden" }}
              >
                <div
                  style={{
                    position: "absolute",
                    top: 0, left: 0, right: 0,
                    height: "1px",
                    background: `linear-gradient(90deg, transparent, ${l.color}40, transparent)`,
                  }}
                />
                <div className="flex items-center gap-2.5 mb-4">
                  <div
                    className="w-7 h-7 rounded-lg flex items-center justify-center"
                    style={{ background: `${l.color}12`, border: `1px solid ${l.color}22`, color: l.color }}
                  >
                    {l.icon}
                  </div>
                  <h3 style={{ fontFamily: "var(--font-body)", fontSize: "0.88rem", fontWeight: 700 }}>{l.title}</h3>
                </div>
                <ul className="space-y-2">
                  {l.items.map((item) => (
                    <li key={item} className="flex items-start gap-2">
                      <span style={{ color: l.color, marginTop: "0.25rem", flexShrink: 0, fontSize: "0.7rem" }}>›</span>
                      <span style={{ color: "#8892B0", fontSize: "0.78rem", lineHeight: 1.6 }}>{item}</span>
                    </li>
                  ))}
                </ul>
              </motion.div>
            </FadeIn>
          ))}
        </div>

        {/* Testing badge */}
        <FadeIn delay={0.28}>
          <div
            className="mt-6 rounded-2xl p-6 flex flex-col md:flex-row items-center justify-between gap-5"
            style={{ background: "#0C0D12", border: "1px solid #1C1E2E" }}
          >
            <div>
              <p style={{ fontWeight: 700, fontSize: "0.95rem", marginBottom: "0.3rem" }}>Suite de Testing Completa</p>
              <p style={{ color: "#8892B0", fontSize: "0.82rem" }}>
                380 tests automatizados · unitarios, integración, seguridad, regresión, performance
              </p>
            </div>
            <div className="flex gap-6 flex-shrink-0">
              {[
                { v: "380",  label: "Tests",   color: "#00FF88" },
                { v: "83%",  label: "Cobertura", color: "#00D4FF" },
                { v: "CI/CD",label: "GitHub Actions", color: "#8B5CF6" },
              ].map((stat) => (
                <div key={stat.label} className="text-center">
                  <p
                    style={{
                      fontFamily: "var(--font-code)",
                      fontSize: "1.4rem",
                      fontWeight: 700,
                      color: stat.color,
                      letterSpacing: "-0.02em",
                    }}
                  >
                    {stat.v}
                  </p>
                  <p style={{ fontSize: "0.62rem", color: "#3D4461", marginTop: "0.2rem" }}>{stat.label}</p>
                </div>
              ))}
            </div>
          </div>
        </FadeIn>
      </div>
    </section>
  );
}
