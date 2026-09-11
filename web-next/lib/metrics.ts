/**
 * Fuente única de verdad de las cifras del sitio.
 *
 * `data/metrics.json` lo genera `Codigo/scripts/sync_dashboard_data.py` a partir
 * de los artefactos reales de evaluación. Ninguna cifra del dashboard debe
 * escribirse a mano: si un número cambia, se vuelve a ejecutar el script.
 */
import raw from "@/data/metrics.json";

export type ClassName = "Auditada" | "Glosada" | "Devuelta";

export interface ClassScore {
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface SplitMetrics {
  nSamples: number;
  f1Macro: number;
  aucRocOvr: number;
  recallGlosada: number;
  accuracy: number;
  recallPerClass: Record<string, number>;
  confusionMatrix: number[][];
  perClass: Record<string, ClassScore>;
}

export interface ModelResult {
  id: string;
  label: string;
  family: string;
  tuning: string;
  trainSeconds: number | null;
  eligible: boolean;
  bestParams: Record<string, number>;
  test: SplitMetrics;
  oot: SplitMetrics;
}

export interface SplitSummary {
  id: string;
  label: string;
  rows: number;
  start: string;
  end: string;
  classCounts: Record<string, number>;
  classShare: Record<string, number>;
}

export interface MonthPoint {
  month: string;
  total: number;
  glosada: number;
  devuelta: number;
  glosadaRate: number;
  censored: boolean;
}

export interface RankingRow {
  model: string;
  label: string;
  f1MacroTest: number;
  recallGlosadaTest: number;
  recallGlosadaOot: number;
  f1Degradation: number;
  eligible: boolean;
}

export interface DriftFeature {
  name: string;
  psi: number;
  status: string;
}

export interface ShapEntry {
  feature: string;
  value: number;
}

export interface HistoryEntry {
  version: string;
  date: string;
  algorithm: string;
  datasetLabel: string;
  f1Macro: number;
  recallGlosada: number;
  ootF1Macro?: number;
  ootRecallGlosada?: number;
  note: string;
  reproducedBaseline?: { label: string; f1Macro: number; recallGlosada: number };
}

export interface Metrics {
  generatedAt: string;
  sourcePrefix: string;
  randomState: number;
  classNames: ClassName[];
  champion: {
    id: string;
    label: string;
    algorithm: string;
    version: string;
    tieBreak: boolean;
    test: SplitMetrics;
    oot: SplitMetrics;
  };
  dataset: {
    totalInvoices: number;
    featureCount: number;
    windowStart: string;
    windowEnd: string;
    labelHorizon: string;
    splits: SplitSummary[];
    monthly: MonthPoint[];
  };
  models: ModelResult[];
  championSelection: { champion: string; tieBreak: boolean; ranking: RankingRow[] };
  drift: {
    referencePrefix: string;
    currentPrefix: string;
    status: string;
    meanPsi: number;
    maxPsi: number;
    nFeatures: number;
    counts: Record<string, number>;
    top: DriftFeature[];
  };
  shap: {
    method: string;
    nSample: number;
    model: string;
    global: ShapEntry[];
    perClass: Record<string, ShapEntry[]>;
  };
  history: HistoryEntry[];
}

export const metrics = raw as unknown as Metrics;

export const champion = metrics.champion;
export const dataset = metrics.dataset;
export const models = metrics.models;
export const drift = metrics.drift;
export const shap = metrics.shap;
export const history = metrics.history;

/** Formatea una proporción 0–1 como porcentaje. */
export function pct(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

/** Formatea un entero con separadores de miles en español. */
export function num(value: number): string {
  return value.toLocaleString("es-CO");
}

/** Abrevia un conteo grande: 1084637 → "1,08M". */
export function compact(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2).replace(".", ",")}M`;
  if (value >= 1_000) return `${Math.round(value / 1000)}K`;
  return String(value);
}

/** Segundos → "1h 15m" / "25m" / "43s". */
export function duration(seconds: number | null): string {
  if (seconds == null) return "—";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m`;
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
}

/** Mes "2026-04" → "abr 2026". */
export function monthLabel(month: string): string {
  const names = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"];
  const [year, m] = month.split("-");
  return `${names[Number(m) - 1]} ${year}`;
}

/** Rango de un split → "mar 2024 → nov 2025". */
export function rangeLabel(split: { start: string; end: string }): string {
  return `${monthLabel(split.start)} → ${monthLabel(split.end)}`;
}

/** Degradación relativa prueba → OOT, en puntos porcentuales de la métrica. */
export function degradation(testValue: number, ootValue: number): number {
  return (testValue - ootValue) / testValue;
}

/** Color asociado al estado de PSI de una variable. */
export function psiColor(status: string): string {
  if (status === "alert") return "#FF4466";
  if (status === "warning") return "#FFB300";
  return "#00FF88";
}

/** Distribución de clases agregada sobre todas las particiones. */
export const overallClassShare: Record<string, number> = (() => {
  const totals: Record<string, number> = {};
  let grand = 0;
  for (const split of metrics.dataset.splits) {
    for (const [cls, count] of Object.entries(split.classCounts)) {
      totals[cls] = (totals[cls] ?? 0) + count;
      grand += count;
    }
  }
  for (const cls of Object.keys(totals)) totals[cls] = totals[cls] / grand;
  return totals;
})();

export const CLASS_COLORS: Record<string, string> = {
  Auditada: "#00D4FF",
  Glosada: "#FF4466",
  Devuelta: "#FFB300",
};
