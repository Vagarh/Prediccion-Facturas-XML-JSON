export default function Footer() {
  return (
    <footer className="py-10 px-6" style={{ borderTop: "1px solid #1e2d45" }}>
      <div className="max-w-5xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
        <div>
          <p style={{ fontFamily: "var(--font-code)", fontSize: "0.75rem", color: "#475569" }}>
            jf.cardona · Universidad EAFIT · Maestría en Ciencia de Datos · 2026
          </p>
          <p style={{ fontSize: "0.72rem", color: "#334155", marginTop: "0.25rem" }}>
            Analítica Predictiva para Auditoría Prestacional · Savia Salud EPS
          </p>
        </div>
        <div className="flex items-center gap-6">
          <a
            href="#problema"
            className="footer-link"
            style={{ fontSize: "0.78rem", color: "#475569", textDecoration: "none" }}
          >
            Inicio
          </a>
          <a
            href="#resultados"
            className="footer-link"
            style={{ fontSize: "0.78rem", color: "#475569", textDecoration: "none" }}
          >
            Resultados
          </a>
          <span
            className="inline-flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full"
            style={{ background: "#0d1f14", border: "1px solid rgba(34,197,94,0.3)", color: "#22C55E", fontFamily: "var(--font-code)" }}
          >
            Recall Glosada 89.6%
          </span>
        </div>
      </div>
    </footer>
  );
}
