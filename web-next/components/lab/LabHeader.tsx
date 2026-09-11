"use client";
import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowLeft, Database } from "lucide-react";
import { champion, compact, dataset, metrics, pct } from "@/lib/metrics";

const SECTIONS = [
  { href: "#dataset", label: "Datos" },
  { href: "#suite", label: "Entrenamientos" },
  { href: "#detalle", label: "Diagnóstico" },
  { href: "#drift", label: "Deriva" },
];

export default function LabHeader() {
  const generated = new Date(metrics.generatedAt).toLocaleDateString("es-CO", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });

  return (
    <>
      <nav
        className="sticky top-0 z-50"
        style={{
          background: "rgba(6,6,8,0.9)",
          backdropFilter: "blur(20px) saturate(180%)",
          borderBottom: "1px solid #1C1E2E",
        }}
      >
        <div className="max-w-6xl mx-auto px-6 py-3 flex items-center justify-between gap-4">
          <Link
            href="/"
            className="flex items-center gap-2 transition-colors duration-200"
            style={{ color: "#8892B0", textDecoration: "none", fontFamily: "var(--font-code)", fontSize: "0.75rem" }}
          >
            <ArrowLeft size={13} />
            Volver al proyecto
          </Link>
          <div className="hidden md:flex items-center gap-6">
            {SECTIONS.map((s) => (
              <a
                key={s.href}
                href={s.href}
                className="transition-colors duration-200 hover:text-white"
                style={{ color: "#8892B0", textDecoration: "none", fontSize: "0.8rem", fontWeight: 500 }}
              >
                {s.label}
              </a>
            ))}
          </div>
          <span
            className="hidden sm:inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full"
            style={{
              background: "rgba(0,255,136,0.06)",
              border: "1px solid rgba(0,255,136,0.2)",
              color: "#00FF88",
              fontFamily: "var(--font-code)",
              fontSize: "0.66rem",
            }}
          >
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#00FF88", display: "inline-block" }} />
            {champion.algorithm} {champion.version}
          </span>
        </div>
      </nav>

      <header className="relative overflow-hidden">
        <div className="absolute inset-0 grid-bg pointer-events-none" />
        <div className="relative max-w-6xl mx-auto px-6 pt-16 pb-10">
          <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
            <span className="badge mb-5">
              <Database size={11} />
              Registro experimental
            </span>
            <h1
              style={{
                fontFamily: "var(--font-body)",
                fontSize: "clamp(2.1rem,5.5vw,3.4rem)",
                fontWeight: 700,
                lineHeight: 1.08,
                letterSpacing: "-0.025em",
                marginBottom: "1.1rem",
              }}
            >
              Los datos y los entrenamientos
              <br />
              <span className="grad-text">detrás del modelo.</span>
            </h1>
            <p style={{ color: "#8892B0", fontSize: "1.02rem", lineHeight: 1.78, maxWidth: 680 }}>
              Todo lo que hay en esta página sale de los artefactos de evaluación del
              proyecto — no hay una sola cifra escrita a mano. Se regenera con{" "}
              <code style={{ fontFamily: "var(--font-code)", color: "#00D4FF", fontSize: "0.9em" }}>
                sync_dashboard_data.py
              </code>{" "}
              cada vez que se reentrena.
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-10"
          >
            {[
              { v: pct(champion.test.recallGlosada), label: "Recall Glosada · prueba", color: "#00FF88" },
              { v: pct(champion.oot.recallGlosada), label: "Recall Glosada · 2 meses después", color: "#00D4FF" },
              { v: compact(dataset.totalInvoices), label: "Facturas analizadas", color: "#0066FF" },
              { v: String(metrics.models.length), label: "Modelos comparados", color: "#8B5CF6" },
            ].map((k) => (
              <div
                key={k.label}
                className="rounded-xl p-5"
                style={{ background: "#10111A", border: "1px solid #1C1E2E", position: "relative", overflow: "hidden" }}
              >
                <div style={{ position: "absolute", top: 0, left: "15%", right: "15%", height: 1, background: `linear-gradient(90deg,transparent,${k.color}60,transparent)` }} />
                <p style={{ fontFamily: "var(--font-code)", fontSize: "1.55rem", fontWeight: 700, color: k.color, lineHeight: 1, letterSpacing: "-0.03em" }}>{k.v}</p>
                <p style={{ fontSize: "0.7rem", color: "#8892B0", marginTop: "0.5rem", lineHeight: 1.4 }}>{k.label}</p>
              </div>
            ))}
          </motion.div>

          <p style={{ fontFamily: "var(--font-code)", fontSize: "0.66rem", color: "#3D4461", marginTop: "1.5rem" }}>
            Artefactos del {metrics.sourcePrefix} · generado el {generated} · RANDOM_STATE={metrics.randomState}
          </p>
        </div>
      </header>
    </>
  );
}
