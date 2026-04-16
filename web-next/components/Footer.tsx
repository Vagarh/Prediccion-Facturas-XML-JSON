export default function Footer() {
  return (
    <footer className="py-10 px-6" style={{ borderTop: "1px solid #1C1E2E" }}>
      <div className="max-w-5xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
        <div>
          <p style={{ fontFamily: "var(--font-code)", fontSize: "0.75rem", color: "#3D4461", letterSpacing: "0.03em" }}>
            jf.cardona · Universidad EAFIT · Maestría en Ciencia de Datos · 2026
          </p>
          <p style={{ fontSize: "0.7rem", color: "#2A2D3E", marginTop: "0.25rem" }}>
            Analítica Predictiva para Auditoría Prestacional · Savia Salud EPS
          </p>
        </div>
        <div className="flex items-center gap-6">
          <a
            href="#problema"
            className="footer-link"
            style={{ fontSize: "0.78rem", color: "#3D4461", textDecoration: "none", fontWeight: 500 }}
          >
            Inicio
          </a>
          <a
            href="#resultados"
            className="footer-link"
            style={{ fontSize: "0.78rem", color: "#3D4461", textDecoration: "none", fontWeight: 500 }}
          >
            Resultados
          </a>
          <span
            className="inline-flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full"
            style={{
              background: "rgba(0,255,136,0.05)",
              border: "1px solid rgba(0,255,136,0.18)",
              color: "#00FF88",
              fontFamily: "var(--font-code)",
              fontSize: "0.68rem",
            }}
          >
            <span style={{ width: 5, height: 5, borderRadius: "50%", background: "#00FF88", display: "inline-block" }} />
            Recall 89.6%
          </span>
        </div>
      </div>
    </footer>
  );
}
