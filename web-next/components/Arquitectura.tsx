"use client";
import { motion } from "framer-motion";
import { Cloud, Database, Zap, Shield, RefreshCw, Code2 } from "lucide-react";
import FadeIn from "./FadeIn";

const layers = [
  {
    title: "Capa de Presentación",
    icon: <Code2 size={16} color="#22C55E" />,
    color: "#22C55E",
    items: ["FastAPI REST", "Endpoints: /predict, /predict/batch, /health, /model/info", "SHAP integrado en tiempo real"],
  },
  {
    title: "Capa de Negocio",
    icon: <Zap size={16} color="#3B82F6" />,
    color: "#3B82F6",
    items: ["XGBoost v2 (48 features)", "PrivacyGuard · PII filtering", "Feature engineering vectorizado"],
  },
  {
    title: "Capa de Datos",
    icon: <Database size={16} color="#6366F1" />,
    color: "#6366F1",
    items: ["Dataset Parquet 633K facturas", "División temporal (no random split)", "Artifacts versionados"],
  },
  {
    title: "Infraestructura AWS",
    icon: <Cloud size={16} color="#F59E0B" />,
    color: "#F59E0B",
    items: ["ECR + ECS Fargate", "API Gateway · ALB", "S3 artifacts · CloudWatch logs"],
  },
  {
    title: "MLOps",
    icon: <RefreshCw size={16} color="#EF4444" />,
    color: "#EF4444",
    items: ["Drift detection automático", "Reentrenamiento mensual programado", "GitHub Actions CI/CD"],
  },
  {
    title: "Seguridad",
    icon: <Shield size={16} color="#94A3B8" />,
    color: "#94A3B8",
    items: ["100% local · Ley 1581 (Habeas Data)", "Sin datos PII en features o logs", "Docker healthcheck + HTTPS"],
  },
];

export default function Arquitectura() {
  return (
    <section id="arquitectura" className="py-24 px-6">
      <div className="max-w-5xl mx-auto">
        <FadeIn className="text-center mb-16">
          <span
            className="inline-block text-xs px-4 py-1.5 rounded-full mb-5"
            style={{ background: "#0E1223", border: "1px solid #1e2d45", color: "#94A3B8", fontFamily: "var(--font-code)" }}
          >
            Capítulo 5 · Arquitectura
          </span>
          <h2 style={{ fontFamily: "var(--font-heading)", fontSize: "clamp(2rem,5vw,3.2rem)", fontWeight: 600, lineHeight: 1.15, marginBottom: "1rem" }}>
            Producción-ready.<br />De principio a fin.
          </h2>
          <p style={{ color: "#94A3B8", fontSize: "1.1rem", maxWidth: 560, margin: "0 auto", lineHeight: 1.7 }}>
            Stack técnico diseñado para operar en la nube con cumplimiento normativo,
            trazabilidad completa y reentrenamiento automático.
          </p>
        </FadeIn>

        <div className="grid md:grid-cols-3 gap-5">
          {layers.map((l, i) => (
            <FadeIn key={l.title} delay={i * 0.08}>
              <motion.div
                whileHover={{ y: -5, boxShadow: "0 10px 36px rgba(0,0,0,0.4)" }}
                transition={{ duration: 0.2 }}
                className="h-full rounded-xl p-6"
                style={{ background: "#131929", border: "1px solid #1e2d45" }}
              >
                <div className="flex items-center gap-2.5 mb-4">
                  <div className="w-7 h-7 rounded-md flex items-center justify-center" style={{ background: l.color + "18", border: `1px solid ${l.color}30` }}>
                    {l.icon}
                  </div>
                  <h3 style={{ fontFamily: "var(--font-heading)", fontSize: "0.95rem", fontWeight: 600 }}>{l.title}</h3>
                </div>
                <ul className="space-y-2">
                  {l.items.map((item) => (
                    <li key={item} className="flex items-start gap-2">
                      <span style={{ color: l.color, marginTop: "0.3rem", flexShrink: 0 }}>›</span>
                      <span style={{ color: "#94A3B8", fontSize: "0.78rem", lineHeight: 1.5 }}>{item}</span>
                    </li>
                  ))}
                </ul>
              </motion.div>
            </FadeIn>
          ))}
        </div>

        {/* Testing badge */}
        <FadeIn delay={0.3}>
          <div className="mt-8 rounded-xl p-5 flex flex-col md:flex-row items-center justify-between gap-4"
            style={{ background: "#0E1223", border: "1px solid #1e2d45" }}>
            <div>
              <p style={{ fontWeight: 600, marginBottom: "0.3rem" }}>Suite de Testing Completa</p>
              <p style={{ color: "#94A3B8", fontSize: "0.85rem" }}>380 tests automatizados · unitarios, integración, seguridad, regresión, performance</p>
            </div>
            <div className="flex gap-4 flex-shrink-0">
              <div className="text-center">
                <p style={{ fontFamily: "var(--font-code)", fontSize: "1.4rem", fontWeight: 700, color: "#22C55E" }}>380</p>
                <p style={{ fontSize: "0.65rem", color: "#475569" }}>Tests</p>
              </div>
              <div className="text-center">
                <p style={{ fontFamily: "var(--font-code)", fontSize: "1.4rem", fontWeight: 700, color: "#3B82F6" }}>83%</p>
                <p style={{ fontSize: "0.65rem", color: "#475569" }}>Cobertura</p>
              </div>
              <div className="text-center">
                <p style={{ fontFamily: "var(--font-code)", fontSize: "1.4rem", fontWeight: 700, color: "#6366F1" }}>CI/CD</p>
                <p style={{ fontSize: "0.65rem", color: "#475569" }}>GitHub Actions</p>
              </div>
            </div>
          </div>
        </FadeIn>
      </div>
    </section>
  );
}
