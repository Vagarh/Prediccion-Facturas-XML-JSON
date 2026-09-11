import type { Metadata } from "next";
import LabHeader from "@/components/lab/LabHeader";
import DatasetPanel from "@/components/lab/DatasetPanel";
import SuitePanel from "@/components/lab/SuitePanel";
import ModelDetailPanel from "@/components/lab/ModelDetailPanel";
import DriftPanel from "@/components/lab/DriftPanel";
import SectionDivider from "@/components/SectionDivider";
import Footer from "@/components/Footer";
import { champion, dataset, metrics, pct } from "@/lib/metrics";

export const metadata: Metadata = {
  title: "Datos y entrenamientos · Analítica Predictiva para Auditoría en Salud",
  description: `Registro experimental del proyecto: ${metrics.models.length} modelos comparados sobre ${dataset.totalInvoices.toLocaleString("es-CO")} facturas reales. Campeón ${champion.algorithm} con Recall Glosada ${pct(champion.test.recallGlosada)}.`,
};

export default function ExperimentosPage() {
  return (
    <>
      <LabHeader />
      <main className="max-w-6xl mx-auto px-6">
        <DatasetPanel />
        <SectionDivider />
        <SuitePanel />
        <SectionDivider />
        <ModelDetailPanel />
        <SectionDivider />
        <DriftPanel />
      </main>
      <Footer />
    </>
  );
}
