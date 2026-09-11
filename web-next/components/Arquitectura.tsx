"use client";
import { motion } from "framer-motion";
import { Cloud, Database, Zap, Shield, RefreshCw, Code2, Server, GitBranch, Activity } from "lucide-react";
import FadeIn from "./FadeIn";

export default function Arquitectura() {
  return (
    <section id="arquitectura" className="py-28 px-6" style={{ background: "#080810" }}>
      <div className="max-w-5xl mx-auto">

        <FadeIn className="text-center mb-16">
          <span className="section-label">Infraestructura</span>
          <h2 style={{ fontFamily: "var(--font-body)", fontSize: "clamp(2.2rem, 5vw, 3.5rem)", fontWeight: 700, lineHeight: 1.1, letterSpacing: "-0.022em", marginBottom: "1.2rem" }}>
            No es un experimento.
            <br />
            <span className="grad-text">Es un producto listo para producción.</span>
          </h2>
          <p style={{ color: "#8892B0", fontSize: "1.05rem", maxWidth: 580, margin: "0 auto", lineHeight: 1.78 }}>
            La solución incluye una API REST deployable en AWS, con monitoreo de drift,
            reentrenamiento automático y 380 tests automatizados.
          </p>
        </FadeIn>

        {/* API como producto — lo más importante */}
        <FadeIn>
          <div className="rounded-2xl mb-8 overflow-hidden" style={{ border: "1px solid rgba(0,212,255,0.2)" }}>
            {/* Header */}
            <div className="px-6 py-4 flex items-center justify-between"
              style={{ background: "#0C0D12", borderBottom: "1px solid #1C1E2E" }}>
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg flex items-center justify-center"
                  style={{ background: "rgba(0,212,255,0.1)", border: "1px solid rgba(0,212,255,0.2)", color: "#00D4FF" }}>
                  <Server size={15} />
                </div>
                <div>
                  <p style={{ fontWeight: 700, fontSize: "0.9rem" }}>API REST · FastAPI + Pydantic v2</p>
                  <p style={{ fontFamily: "var(--font-code)", fontSize: "0.65rem", color: "#8892B0" }}>4 endpoints · SHAP en tiempo real · Dockerfile con healthcheck</p>
                </div>
              </div>
              <span style={{ fontFamily: "var(--font-code)", fontSize: "0.65rem", color: "#00FF88", background: "rgba(0,255,136,0.08)", border: "1px solid rgba(0,255,136,0.2)", padding: "3px 10px", borderRadius: 4 }}>
                ● Production ready
              </span>
            </div>

            {/* Endpoints */}
            <div className="p-6 grid md:grid-cols-4 gap-3" style={{ background: "#10111A" }}>
              {[
                { method: "GET",  path: "/health",         desc: "Estado del servicio",        color: "#00FF88" },
                { method: "GET",  path: "/model/info",     desc: "Versión y métricas del modelo", color: "#00D4FF" },
                { method: "POST", path: "/predict",        desc: "Predicción + SHAP individual", color: "#0066FF" },
                { method: "POST", path: "/predict/batch",  desc: "Predicción masiva en lote",  color: "#8B5CF6" },
              ].map((ep) => (
                <div key={ep.path} className="rounded-xl p-4"
                  style={{ background: "#0C0D12", border: `1px solid ${ep.color}20` }}>
                  <div className="flex items-center gap-2 mb-2">
                    <span style={{ fontFamily: "var(--font-code)", fontSize: "0.6rem", color: ep.color, background: `${ep.color}12`, padding: "2px 7px", borderRadius: 3, letterSpacing: "0.05em" }}>{ep.method}</span>
                  </div>
                  <p style={{ fontFamily: "var(--font-code)", fontSize: "0.7rem", color: "#F0F2FF", fontWeight: 600, marginBottom: "0.4rem" }}>{ep.path}</p>
                  <p style={{ fontSize: "0.7rem", color: "#8892B0", lineHeight: 1.5 }}>{ep.desc}</p>
                </div>
              ))}
            </div>

            {/* Response example */}
            <div className="px-6 pb-6" style={{ background: "#10111A" }}>
              <div className="rounded-xl p-5" style={{ background: "#060608", border: "1px solid rgba(0,212,255,0.12)", fontFamily: "var(--font-code)" }}>
                <p style={{ fontSize: "0.6rem", color: "#3D4461", marginBottom: "0.75rem", letterSpacing: "0.05em" }}>
                  POST /predict · Response example
                </p>
                <div style={{ fontSize: "0.75rem", lineHeight: 2.1, color: "#8892B0" }}>
                  {"{"}<br />
                  &nbsp;&nbsp;<span style={{ color: "#8892B0" }}>"prediction"</span>: <span style={{ color: "#FF4466" }}>"Glosada"</span>,<br />
                  &nbsp;&nbsp;<span style={{ color: "#8892B0" }}>"confidence"</span>: <span style={{ color: "#00FF88" }}>0.891</span>,<br />
                  &nbsp;&nbsp;<span style={{ color: "#8892B0" }}>"probabilities"</span>: {"{"}<span style={{ color: "#8892B0" }}>"Auditada"</span>: <span style={{ color: "#00D4FF" }}>0.07</span>, <span style={{ color: "#8892B0" }}>"Glosada"</span>: <span style={{ color: "#FF4466" }}>0.891</span>, <span style={{ color: "#8892B0" }}>"Devuelta"</span>: <span style={{ color: "#FFB300" }}>0.039</span>{"}"}, <br />
                  &nbsp;&nbsp;<span style={{ color: "#8892B0" }}>"shap_top5"</span>: <span style={{ color: "#00D4FF" }}>[</span><span style={{ color: "#8892B0" }}>{"{"}"feature": "vr_factura", "impact": 0.42{"}"}</span><span style={{ color: "#00D4FF" }}>, ...]</span>,<br />
                  &nbsp;&nbsp;<span style={{ color: "#8892B0" }}>"inference_ms"</span>: <span style={{ color: "#8B5CF6" }}>187</span><br />
                  {"}"}
                </div>
              </div>
            </div>
          </div>
        </FadeIn>

        {/* Infraestructura AWS — diagrama visual */}
        <FadeIn>
          <div className="rounded-2xl p-8 mb-8" style={{ background: "#10111A", border: "1px solid #1C1E2E" }}>
            <p style={{ fontFamily: "var(--font-code)", fontSize: "0.65rem", color: "#3D4461", marginBottom: "1.75rem", letterSpacing: "0.1em", textTransform: "uppercase" }}>
              Arquitectura AWS · Diseño de despliegue en nube
            </p>

            {/* AWS diagram — 3 capas */}
            <div className="space-y-4">
              {/* Capa cliente */}
              <div className="rounded-xl p-4 flex items-center gap-4"
                style={{ background: "#0C0D12", border: "1px solid rgba(0,212,255,0.15)" }}>
                <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
                  style={{ background: "rgba(0,212,255,0.1)", color: "#00D4FF" }}>
                  <Code2 size={15} />
                </div>
                <div className="flex-1">
                  <p style={{ fontFamily: "var(--font-code)", fontSize: "0.65rem", color: "#3D4461", marginBottom: 2 }}>CLIENTE</p>
                  <p style={{ fontWeight: 700, fontSize: "0.85rem" }}>Sistema de radicación de la EPS</p>
                </div>
                <div className="hidden md:flex items-center gap-2">
                  <div className="h-px w-12" style={{ background: "#1C1E2E" }} />
                  <span style={{ color: "#3D4461", fontSize: "0.7rem", fontFamily: "var(--font-code)" }}>HTTPS</span>
                  <div className="h-px w-12" style={{ background: "#1C1E2E" }} />
                </div>
                <div className="flex items-center gap-2">
                  <div className="rounded px-2 py-1" style={{ background: "rgba(0,212,255,0.08)", border: "1px solid rgba(0,212,255,0.2)" }}>
                    <span style={{ fontFamily: "var(--font-code)", fontSize: "0.62rem", color: "#00D4FF" }}>API Gateway</span>
                  </div>
                  <span style={{ color: "#3D4461" }}>→</span>
                  <div className="rounded px-2 py-1" style={{ background: "rgba(0,212,255,0.08)", border: "1px solid rgba(0,212,255,0.2)" }}>
                    <span style={{ fontFamily: "var(--font-code)", fontSize: "0.62rem", color: "#00D4FF" }}>ALB</span>
                  </div>
                </div>
              </div>

              {/* Flecha */}
              <div className="flex justify-center">
                <div className="flex flex-col items-center gap-1">
                  <div style={{ width: 1, height: 12, background: "#1C1E2E" }} />
                  <span style={{ color: "#3D4461", fontSize: "0.7rem" }}>↓</span>
                </div>
              </div>

              {/* Capa compute */}
              <div className="rounded-xl p-4"
                style={{ background: "#0C0D12", border: "1px solid rgba(0,102,255,0.2)" }}>
                <p style={{ fontFamily: "var(--font-code)", fontSize: "0.65rem", color: "#3D4461", marginBottom: "0.75rem" }}>COMPUTE · ECS FARGATE</p>
                <div className="grid md:grid-cols-3 gap-3">
                  {[
                    { label: "Contenedor FastAPI", sub: "Docker · healthcheck /health", color: "#0066FF" },
                    { label: "LightGBM v3", sub: "48 features · inference < 200ms", color: "#8B5CF6" },
                    { label: "SHAP TreeExplainer", sub: "Top-5 factores por predicción", color: "#00D4FF" },
                  ].map((c) => (
                    <div key={c.label} className="rounded-lg p-3" style={{ background: "#10111A", border: `1px solid ${c.color}20` }}>
                      <p style={{ fontWeight: 700, fontSize: "0.78rem", color: c.color, marginBottom: "0.2rem" }}>{c.label}</p>
                      <p style={{ fontSize: "0.7rem", color: "#8892B0" }}>{c.sub}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Flecha */}
              <div className="flex justify-center">
                <div className="flex flex-col items-center gap-1">
                  <div style={{ width: 1, height: 12, background: "#1C1E2E" }} />
                  <span style={{ color: "#3D4461", fontSize: "0.7rem" }}>↓</span>
                </div>
              </div>

              {/* Capa datos + MLOps */}
              <div className="grid md:grid-cols-2 gap-3">
                <div className="rounded-xl p-4" style={{ background: "#0C0D12", border: "1px solid rgba(139,92,246,0.2)" }}>
                  <div className="flex items-center gap-2 mb-3">
                    <Database size={13} color="#8B5CF6" />
                    <p style={{ fontFamily: "var(--font-code)", fontSize: "0.65rem", color: "#3D4461" }}>DATOS · S3</p>
                  </div>
                  {["Model artifacts versionados", "Dataset Parquet 1.08M facturas", "Logs de predicciones (sin PII)"].map((i) => (
                    <p key={i} style={{ fontSize: "0.72rem", color: "#8892B0", lineHeight: 1.8 }}>
                      <span style={{ color: "#8B5CF6" }}>›</span> {i}
                    </p>
                  ))}
                </div>
                <div className="rounded-xl p-4" style={{ background: "#0C0D12", border: "1px solid rgba(255,68,102,0.2)" }}>
                  <div className="flex items-center gap-2 mb-3">
                    <RefreshCw size={13} color="#FF4466" />
                    <p style={{ fontFamily: "var(--font-code)", fontSize: "0.65rem", color: "#3D4461" }}>MLOps · CLOUDWATCH + STEP FUNCTIONS</p>
                  </div>
                  {["Drift detection automático (PSI)", "Reentrenamiento mensual programado", "GitHub Actions CI/CD · 3 pipelines"].map((i) => (
                    <p key={i} style={{ fontSize: "0.72rem", color: "#8892B0", lineHeight: 1.8 }}>
                      <span style={{ color: "#FF4466" }}>›</span> {i}
                    </p>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </FadeIn>

        {/* Testing badge */}
        <FadeIn>
          <div className="rounded-2xl p-6 flex flex-col md:flex-row items-center justify-between gap-5"
            style={{ background: "#0C0D12", border: "1px solid rgba(0,255,136,0.15)" }}>
            <div>
              <div className="flex items-center gap-2 mb-1">
                <Activity size={14} color="#00FF88" />
                <p style={{ fontWeight: 700, fontSize: "0.95rem" }}>Suite de Testing Completa</p>
              </div>
              <p style={{ color: "#8892B0", fontSize: "0.82rem" }}>
                Unit · Integration · Security · Regression · Performance · Docker
              </p>
            </div>
            <div className="flex gap-7 flex-shrink-0">
              {[
                { v: "380", label: "Tests", color: "#00FF88" },
                { v: "83%", label: "Cobertura", color: "#00D4FF" },
                { v: "3", label: "Pipelines CI/CD", color: "#8B5CF6" },
              ].map((stat) => (
                <div key={stat.label} className="text-center">
                  <p style={{ fontFamily: "var(--font-code)", fontSize: "1.5rem", fontWeight: 700, color: stat.color, letterSpacing: "-0.02em" }}>{stat.v}</p>
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
