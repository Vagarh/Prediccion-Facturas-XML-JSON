"use client";
import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Menu, X } from "lucide-react";

const links = [
  { href: "#problema",     label: "Problema" },
  { href: "#solucion",     label: "Solución" },
  { href: "#resultados",   label: "Resultados" },
  { href: "#explicabilidad", label: "SHAP" },
  { href: "#arquitectura", label: "Arquitectura" },
  { href: "#futuro",       label: "Futuro" },
  { href: "#video",        label: "Video" },
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
        transition={{ duration: 0.6, ease: "easeOut" }}
        className="fixed top-0 left-0 right-0 z-50 transition-all duration-300"
        style={{
          backdropFilter: scrolled ? "blur(16px)" : "blur(0px)",
          background: scrolled ? "rgba(2,6,23,0.9)" : "transparent",
          borderBottom: scrolled ? "1px solid #1e2d45" : "1px solid transparent",
        }}
      >
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <span style={{ fontFamily: "var(--font-code)", fontSize: "0.8rem", color: "#94A3B8" }}>
            jfcardona · EAFIT 2026
          </span>

          {/* Desktop links */}
          <div className="hidden md:flex items-center gap-6">
            {links.map((l) => (
              <a
                key={l.href}
                href={l.href}
                className="text-sm transition-colors duration-200 hover:text-white"
                style={{ color: "#94A3B8", textDecoration: "none" }}
              >
                {l.label}
              </a>
            ))}
          </div>

          <span
            className="hidden md:inline-block text-xs px-3 py-1 rounded-full"
            style={{ background: "#0d1f14", border: "1px solid rgba(34,197,94,0.3)", color: "#22C55E", fontFamily: "var(--font-code)" }}
          >
            Recall Glosada 89.6%
          </span>

          {/* Mobile menu button */}
          <button
            className="md:hidden p-2 rounded-lg"
            style={{ color: "#94A3B8", background: "transparent", border: "none", cursor: "pointer" }}
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
            transition={{ duration: 0.25 }}
            className="fixed inset-y-0 right-0 z-40 w-64 flex flex-col pt-20 px-6 gap-4"
            style={{ background: "#0E1223", borderLeft: "1px solid #1e2d45" }}
          >
            {links.map((l) => (
              <a
                key={l.href}
                href={l.href}
                onClick={() => setOpen(false)}
                className="text-base py-2 transition-colors"
                style={{ color: "#94A3B8", textDecoration: "none", fontFamily: "var(--font-body)" }}
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
