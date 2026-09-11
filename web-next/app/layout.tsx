import type { Metadata } from "next";
import { Space_Grotesk, JetBrains_Mono } from "next/font/google";
import "./globals.css";

// MiniMax-style: bold geometric sans-serif
const body = Space_Grotesk({
  variable: "--font-body",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
  display: "swap",
});

const code = JetBrains_Mono({
  variable: "--font-code",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Analítica Predictiva para Auditoría en Salud · Juan Felipe Cardona",
  description:
    "Modelo de IA que predice el estado de auditoría de facturas médicas. LightGBM · Recall Glosada 86.4% · Universidad EAFIT 2026",
  openGraph: {
    title: "Analítica Predictiva para Auditoría Prestacional",
    description: "Recall Glosada 0.8635 — meta ≥0.80 superada, estable fuera de muestra. Universidad EAFIT 2026.",
    type: "website",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="es" className={`${body.variable} ${code.variable} scroll-smooth`}>
      <body
        className="min-h-screen antialiased"
        style={{ background: "#060608", color: "#F0F2FF", fontFamily: "var(--font-body, sans-serif)" }}
      >
        {children}
      </body>
    </html>
  );
}
