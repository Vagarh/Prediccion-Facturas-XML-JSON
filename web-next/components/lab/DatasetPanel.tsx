"use client";
import { motion } from "framer-motion";
import { CLASS_COLORS, compact, dataset, monthLabel, num, pct, rangeLabel } from "@/lib/metrics";
import FadeIn from "@/components/FadeIn";

/** Serie mensual de tasa de glosa, con la zona censurada marcada aparte. */
function MonthlySeries() {
  const points = dataset.monthly;
  const maxRate = Math.max(...points.map((p) => p.glosadaRate));
  const scale = (rate: number) => (rate / (maxRate * 1.15)) * 100;

  return (
    <div>
      <div className="flex items-end gap-[3px]" style={{ height: 150 }}>
        {points.map((p, i) => (
          <motion.div
            key={p.month}
            initial={{ height: 0 }}
            whileInView={{ height: `${scale(p.glosadaRate)}%` }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: Math.min(i * 0.012, 0.6), ease: "easeOut" }}
            title={`${monthLabel(p.month)} · ${num(p.total)} facturas · tasa de glosa ${pct(p.glosadaRate)}${
              p.censored ? " · periodo censurado" : ""
            }`}
            style={{
              flex: 1,
              minWidth: 4,
              borderRadius: "2px 2px 0 0",
              background: p.censored
                ? "repeating-linear-gradient(45deg, #3D4461, #3D4461 2px, transparent 2px, transparent 4px)"
                : "linear-gradient(180deg, #00D4FF, #0066FF)",
              border: p.censored ? "1px solid #3D4461" : "none",
            }}
          />
        ))}
      </div>
      <div className="flex justify-between mt-2" style={{ fontFamily: "var(--font-code)", fontSize: "0.6rem", color: "#3D4461" }}>
        <span>{monthLabel(points[0].month)}</span>
        <span>{monthLabel(points[points.length - 1].month)}</span>
      </div>
      <div className="flex flex-wrap gap-4 mt-3">
        <span className="flex items-center gap-2" style={{ fontSize: "0.68rem", color: "#8892B0" }}>
          <span style={{ width: 10, height: 10, borderRadius: 2, background: "linear-gradient(180deg,#00D4FF,#0066FF)" }} />
          Etiquetas maduras
        </span>
        <span className="flex items-center gap-2" style={{ fontSize: "0.68rem", color: "#8892B0" }}>
          <span
            style={{
              width: 10,
              height: 10,
              borderRadius: 2,
              border: "1px solid #3D4461",
              background: "repeating-linear-gradient(45deg,#3D4461,#3D4461 2px,transparent 2px,transparent 4px)",
            }}
          />
          Censurado — ciclo de auditoría incompleto (desde {monthLabel(dataset.labelHorizon.slice(0, 7))})
        </span>
      </div>
    </div>
  );
}

export default function DatasetPanel() {
  const kpis = [
    { v: compact(dataset.totalInvoices), label: "Facturas reales", color: "#00D4FF" },
    { v: String(dataset.featureCount), label: "Variables de entrada", color: "#8B5CF6" },
    { v: String(dataset.splits.length), label: "Particiones temporales", color: "#0066FF" },
    { v: monthLabel(dataset.labelHorizon.slice(0, 7)), label: "Horizonte de etiquetas", color: "#FFB300" },
  ];

  return (
    <section id="dataset" className="py-16">
      <FadeIn>
        <span className="section-label">01 · Datos</span>
        <h2 style={{ fontFamily: "var(--font-body)", fontSize: "clamp(1.6rem,3.5vw,2.3rem)", fontWeight: 700, letterSpacing: "-0.02em", marginBottom: "0.8rem" }}>
          Dataset y particiones temporales
        </h2>
        <p style={{ color: "#8892B0", fontSize: "0.95rem", lineHeight: 1.75, maxWidth: 760, marginBottom: "2rem" }}>
          Extracción directa de la base de datos operativa de Savia Salud EPS. La división
          es estrictamente temporal —nunca aleatoria— para que la evaluación reproduzca las
          condiciones de producción: el modelo jamás ve facturas posteriores a las que debe predecir.
        </p>
      </FadeIn>

      <FadeIn>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-8">
          {kpis.map((k) => (
            <div key={k.label} className="rounded-xl p-5 text-center" style={{ background: "#10111A", border: "1px solid #1C1E2E", position: "relative", overflow: "hidden" }}>
              <div style={{ position: "absolute", top: 0, left: "20%", right: "20%", height: 1, background: `linear-gradient(90deg,transparent,${k.color}60,transparent)` }} />
              <p style={{ fontFamily: "var(--font-code)", fontSize: "1.6rem", fontWeight: 700, color: k.color, lineHeight: 1, letterSpacing: "-0.03em" }}>{k.v}</p>
              <p style={{ fontSize: "0.7rem", color: "#8892B0", marginTop: "0.5rem", lineHeight: 1.4 }}>{k.label}</p>
            </div>
          ))}
        </div>
      </FadeIn>

      <FadeIn>
        <div className="rounded-2xl p-6 mb-6" style={{ background: "#10111A", border: "1px solid #1C1E2E" }}>
          <p style={{ fontFamily: "var(--font-code)", fontSize: "0.65rem", color: "#3D4461", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: "1.5rem" }}>
            Particiones · distribución de clases por split
          </p>
          <div className="space-y-5">
            {dataset.splits.map((s) => (
              <div key={s.id}>
                <div className="flex flex-wrap items-baseline justify-between gap-2 mb-2">
                  <span style={{ fontSize: "0.85rem", fontWeight: 600, color: "#F0F2FF" }}>
                    {s.label}
                    <span style={{ fontFamily: "var(--font-code)", fontSize: "0.68rem", color: "#3D4461", marginLeft: 8 }}>
                      {rangeLabel(s)}
                    </span>
                  </span>
                  <span style={{ fontFamily: "var(--font-code)", fontSize: "0.75rem", color: "#8892B0" }}>{num(s.rows)} facturas</span>
                </div>
                <div className="flex rounded-md overflow-hidden" style={{ height: 26, background: "#0C0D12" }}>
                  {dataset.splits.length > 0 &&
                    Object.entries(s.classShare).map(([cls, share]) => (
                      <motion.div
                        key={cls}
                        initial={{ width: 0 }}
                        whileInView={{ width: `${share * 100}%` }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.8, ease: "easeOut" }}
                        title={`${cls}: ${num(s.classCounts[cls])} (${pct(share)})`}
                        className="flex items-center justify-center"
                        style={{ background: `${CLASS_COLORS[cls]}28`, borderRight: `1px solid ${CLASS_COLORS[cls]}55` }}
                      >
                        <span style={{ fontFamily: "var(--font-code)", fontSize: "0.62rem", color: CLASS_COLORS[cls], whiteSpace: "nowrap" }}>
                          {share > 0.09 ? pct(share, 0) : ""}
                        </span>
                      </motion.div>
                    ))}
                </div>
              </div>
            ))}
          </div>
          <div className="flex flex-wrap gap-4 mt-5 pt-4" style={{ borderTop: "1px solid #1C1E2E" }}>
            {Object.entries(CLASS_COLORS).map(([cls, color]) => (
              <span key={cls} className="flex items-center gap-2" style={{ fontSize: "0.72rem", color: "#8892B0" }}>
                <span style={{ width: 10, height: 10, borderRadius: 2, background: `${color}55`, border: `1px solid ${color}` }} />
                {cls}
              </span>
            ))}
          </div>
        </div>
      </FadeIn>

      <FadeIn>
        <div className="rounded-2xl p-6" style={{ background: "#10111A", border: "1px solid #1C1E2E" }}>
          <p style={{ fontFamily: "var(--font-code)", fontSize: "0.65rem", color: "#3D4461", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: "0.5rem" }}>
            Horizonte de etiquetas · tasa de glosa mensual
          </p>
          <p style={{ fontSize: "0.82rem", color: "#8892B0", lineHeight: 1.7, marginBottom: "1.5rem", maxWidth: 720 }}>
            Las facturas recientes todavía no han completado su ciclo de auditoría: parecen
            &ldquo;limpias&rdquo; solo porque nadie ha terminado de revisarlas. Incluirlas
            enseñaría al modelo un patrón falso. El corte se detecta automáticamente por la
            caída de la tasa de glosa y el desplome de las devoluciones.
          </p>
          <MonthlySeries />
        </div>
      </FadeIn>
    </section>
  );
}
