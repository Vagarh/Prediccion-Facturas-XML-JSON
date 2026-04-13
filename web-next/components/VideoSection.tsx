"use client";
import { motion } from "framer-motion";
import { Play, Clock, BookOpen } from "lucide-react";
import FadeIn from "./FadeIn";

export default function VideoSection() {
  return (
    <section id="video" className="py-24 px-6" style={{ background: "#0A0E1A" }}>
      <div className="max-w-5xl mx-auto">
        <FadeIn className="text-center mb-12">
          <span
            className="inline-block text-xs px-4 py-1.5 rounded-full mb-5"
            style={{ background: "#0E1223", border: "1px solid #1e2d45", color: "#94A3B8", fontFamily: "var(--font-code)" }}
          >
            Capítulo 7 · Video Explicativo
          </span>
          <h2 style={{ fontFamily: "var(--font-heading)", fontSize: "clamp(2rem,5vw,3.2rem)", fontWeight: 600, lineHeight: 1.15, marginBottom: "1rem" }}>
            El proyecto en<br />5 minutos.
          </h2>
          <p style={{ color: "#94A3B8", fontSize: "1.1rem", maxWidth: 560, margin: "0 auto", lineHeight: 1.7 }}>
            Una explicación visual del problema, la solución y los resultados — sin tecnicismos.
          </p>
        </FadeIn>

        <FadeIn>
          <motion.div
            className="relative rounded-2xl overflow-hidden mx-auto max-w-3xl"
            style={{ background: "#131929", border: "1px solid #1e2d45", aspectRatio: "16/9" }}
            whileHover={{ scale: 1.005 }}
            transition={{ duration: 0.3 }}
          >
            {/* Placeholder content */}
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-6 p-8">
              <motion.div
                className="w-20 h-20 rounded-full flex items-center justify-center"
                style={{ background: "linear-gradient(135deg, #22C55E, #16A34A)", boxShadow: "0 0 40px rgba(34,197,94,0.3)" }}
                animate={{ scale: [1, 1.08, 1] }}
                transition={{ duration: 2.5, repeat: Infinity, ease: "easeInOut" }}
              >
                <Play size={32} color="#020617" fill="#020617" />
              </motion.div>

              <div className="text-center">
                <p style={{ fontFamily: "var(--font-heading)", fontSize: "1.3rem", fontWeight: 600, marginBottom: "0.5rem" }}>
                  Video en producción
                </p>
                <p style={{ color: "#94A3B8", fontSize: "0.85rem" }}>
                  El video explicativo será publicado próximamente
                </p>
              </div>

              <div className="flex gap-6">
                <div className="flex items-center gap-2">
                  <Clock size={14} color="#94A3B8" />
                  <span style={{ fontSize: "0.78rem", color: "#94A3B8" }}>~5 minutos</span>
                </div>
                <div className="flex items-center gap-2">
                  <BookOpen size={14} color="#94A3B8" />
                  <span style={{ fontSize: "0.78rem", color: "#94A3B8" }}>Subtítulos en español</span>
                </div>
              </div>
            </div>

            {/* Decorative corner elements */}
            <div className="absolute top-4 left-4" style={{ width: 40, height: 40, borderTop: "2px solid #22C55E40", borderLeft: "2px solid #22C55E40" }} />
            <div className="absolute bottom-4 right-4" style={{ width: 40, height: 40, borderBottom: "2px solid #22C55E40", borderRight: "2px solid #22C55E40" }} />
          </motion.div>
        </FadeIn>

        {/* Script outline */}
        <FadeIn delay={0.2}>
          <div className="mt-8 grid md:grid-cols-5 gap-3 max-w-3xl mx-auto">
            {[
              { t: "0:00", label: "El Problema" },
              { t: "1:00", label: "Los Datos" },
              { t: "2:00", label: "El Modelo" },
              { t: "3:30", label: "Resultados" },
              { t: "4:30", label: "Futuro" },
            ].map((s) => (
              <div key={s.t} className="rounded-lg py-3 px-3 text-center" style={{ background: "#131929", border: "1px solid #1e2d45" }}>
                <p style={{ fontFamily: "var(--font-code)", fontSize: "0.7rem", color: "#22C55E" }}>{s.t}</p>
                <p style={{ fontSize: "0.72rem", color: "#94A3B8", marginTop: "0.25rem" }}>{s.label}</p>
              </div>
            ))}
          </div>
        </FadeIn>
      </div>
    </section>
  );
}
