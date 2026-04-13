import type { Metadata } from "next";
import { Crimson_Pro, Atkinson_Hyperlegible, Roboto_Mono } from "next/font/google";
import "./globals.css";

const heading = Crimson_Pro({
  variable: "--font-heading",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  style: ["normal", "italic"],
  display: "swap",
});

const body = Atkinson_Hyperlegible({
  variable: "--font-body",
  subsets: ["latin"],
  weight: ["400", "700"],
  display: "swap",
});

const code = Roboto_Mono({
  variable: "--font-code",
  subsets: ["latin"],
  weight: ["300", "400", "500"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Analítica Predictiva para Auditoría en Salud · Juan Felipe Cardona",
  description:
    "Modelo de IA que predice el estado de auditoría de facturas médicas. XGBoost · Recall Glosada 89.6% · Universidad EAFIT 2026",
  openGraph: {
    title: "Analítica Predictiva para Auditoría Prestacional",
    description: "Recall Glosada 0.8961 — meta ≥0.80 superada. Universidad EAFIT 2026.",
    type: "website",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="es" className={`${heading.variable} ${body.variable} ${code.variable} scroll-smooth`}>
      <body className="min-h-screen antialiased" style={{ background: "#020617", color: "#F8FAFC", fontFamily: "var(--font-body, sans-serif)" }}>
        {children}
      </body>
    </html>
  );
}
