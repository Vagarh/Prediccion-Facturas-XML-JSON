"use client";
import { motion } from "framer-motion";
import { Play, Clock, BookOpen } from "lucide-react";
import FadeIn from "./FadeIn";

export default function VideoSection() {
  return (
    <section id="video" className="py-28 px-6" style={{ background: "#080810" }}>
      <div className="max-w-5xl mx-auto">
        <FadeIn className="text-center mb-12">
          <span className="section-label">Capítulo 07 · Video Explicativo</span>
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
            El proyecto en
            <br />
            <span className="grad-text">5 minutos.</span>
          </h2>
          <p style={{ color: "#8892B0", fontSize: "1.05rem", maxWidth: 540, margin: "0 auto", lineHeight: 1.75 }}>
            Una explicación visual del problema, la solución y los resultados — sin tecnicismos.
          </p>
        </FadeIn>

        <FadeIn>
          <motion.div
            className="relative rounded-2xl overflow-hidden mx-auto max-w-3xl"
            style={{
              background: "#10111A",
              border: "1px solid #1C1E2E",
              aspectRatio: "16/9",
            }}
            whileHover={{ scale: 1.005 }}
            transition={{ duration: 0.3 }}
          >
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-6 p-8">
              <motion.div
                className="w-20 h-20 rounded-full flex items-center justify-center"
                style={{
                  background: "linear-gradient(135deg, #00D4FF, #0066FF)",
                  boxShadow: "0 0 48px rgba(0,212,255,0.35)",
                }}
                animate={{ scale: [1, 1.08, 1] }}
                transition={{ duration: 2.5, repeat: Infinity, ease: "easeInOut" }}
              >
                <Play size={30} color="#060608" fill="#060608" />
              </motion.div>

              <div className="text-center">
                <p style={{ fontFamily: "var(--font-body)", fontSize: "1.2rem", fontWeight: 700, marginBottom: "0.5rem" }}>
                  Video en producción
                </p>
                <p style={{ color: "#8892B0", fontSize: "0.85rem" }}>
                  El video explicativo será publicado próximamente
                </p>
              </div>

              <div className="flex gap-6">
                <div className="flex items-center gap-2">
                  <Clock size={13} color="#8892B0" />
                  <span style={{ fontSize: "0.78rem", color: "#8892B0" }}>~5 minutos</span>
                </div>
                <div className="flex items-center gap-2">
                  <BookOpen size={13} color="#8892B0" />
                  <span style={{ fontSize: "0.78rem", color: "#8892B0" }}>Subtítulos en español</span>
                </div>
              </div>
            </div>

            {/* Corner accents — MiniMax style */}
            <div className="absolute top-4 left-4" style={{ width: 36, height: 36, borderTop: "1px solid rgba(0,212,255,0.3)", borderLeft: "1px solid rgba(0,212,255,0.3)" }} />
            <div className="absolute bottom-4 right-4" style={{ width: 36, height: 36, borderBottom: "1px solid rgba(0,212,255,0.3)", borderRight: "1px solid rgba(0,212,255,0.3)" }} />
          </motion.div>
        </FadeIn>

        {/* Timeline */}
        <FadeIn delay={0.15}>
          <div className="mt-7 grid grid-cols-5 gap-2 max-w-3xl mx-auto">
            {[
              { t: "0:00", label: "El Problema" },
              { t: "1:00", label: "Los Datos" },
              { t: "2:00", label: "El Modelo" },
              { t: "3:30", label: "Resultados" },
              { t: "4:30", label: "Futuro" },
            ].map((s) => (
              <div
                key={s.t}
                className="rounded-xl py-3 px-2 text-center"
                style={{ background: "#10111A", border: "1px solid #1C1E2E" }}
              >
                <p style={{ fontFamily: "var(--font-code)", fontSize: "0.68rem", color: "#00D4FF", fontWeight: 600 }}>{s.t}</p>
                <p style={{ fontSize: "0.68rem", color: "#8892B0", marginTop: "0.25rem" }}>{s.label}</p>
              </div>
            ))}
          </div>
        </FadeIn>
      </div>
    </section>
  );
}
