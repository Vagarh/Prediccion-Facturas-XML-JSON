"use client";
import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Menu, X } from "lucide-react";

const links = [
  { href: "#problema",      label: "Problema" },
  { href: "#solucion",      label: "Solución" },
  { href: "#resultados",    label: "Resultados" },
  { href: "#explicabilidad",label: "SHAP" },
  { href: "#arquitectura",  label: "Arquitectura" },
  { href: "#futuro",        label: "Futuro" },
  { href: "#video",         label: "Video" },
];

export default function Nav() {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 40);
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <>
      <motion.nav
        initial={{ y: -60, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.5, ease: "easeOut" }}
        className="fixed top-0 left-0 right-0 z-50 transition-all duration-300"
        style={{
          backdropFilter: scrolled ? "blur(20px) saturate(180%)" : "blur(0px)",
          background: scrolled ? "rgba(6,6,8,0.88)" : "transparent",
          borderBottom: scrolled ? "1px solid #1C1E2E" : "1px solid transparent",
        }}
      >
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          {/* Logo */}
          <span
            style={{
              fontFamily: "var(--font-code)",
              fontSize: "0.75rem",
              color: "#8892B0",
              letterSpacing: "0.05em",
            }}
          >
            jfcardona <span style={{ color: "#1C1E2E", margin: "0 4px" }}>·</span> EAFIT 2026
          </span>

          {/* Desktop nav */}
          <div className="hidden md:flex items-center gap-7">
            {links.map((l) => (
              <a
                key={l.href}
                href={l.href}
                className="text-sm transition-colors duration-200 hover:text-white"
                style={{ color: "#8892B0", textDecoration: "none", fontWeight: 500, fontSize: "0.82rem" }}
              >
                {l.label}
              </a>
            ))}
          </div>

          {/* Status badge */}
          <span
            className="hidden md:inline-flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full"
            style={{
              background: "rgba(0,255,136,0.06)",
              border: "1px solid rgba(0,255,136,0.2)",
              color: "#00FF88",
              fontFamily: "var(--font-code)",
              fontSize: "0.68rem",
            }}
          >
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#00FF88", display: "inline-block" }} />
            Recall 89.6%
          </span>

          {/* Mobile toggle */}
          <button
            className="md:hidden p-2 rounded-lg"
            style={{ color: "#8892B0", background: "transparent", border: "none", cursor: "pointer" }}
            onClick={() => setOpen(!open)}
            aria-label="Toggle menu"
          >
            {open ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>
      </motion.nav>

      {/* Mobile drawer */}
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, x: "100%" }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: "100%" }}
            transition={{ duration: 0.22 }}
            className="fixed inset-y-0 right-0 z-40 w-64 flex flex-col pt-20 px-6 gap-5"
            style={{ background: "#0C0D12", borderLeft: "1px solid #1C1E2E" }}
          >
            {links.map((l) => (
              <a
                key={l.href}
                href={l.href}
                onClick={() => setOpen(false)}
                style={{ color: "#8892B0", textDecoration: "none", fontWeight: 500, fontSize: "0.95rem" }}
              >
                {l.label}
              </a>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
