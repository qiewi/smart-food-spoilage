"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { MetricMap } from "@/lib/api";

const SPLIT_COLOR: Record<string, string> = {
  Grouped: "#10b981",
  "Random Split": "#f59e0b",
};
const SPLIT_SHORT: Record<string, string> = {
  Grouped: "Grouped",
  "Random Split": "Random",
};

export function shortModel(m: string): string {
  return m
    .replace("Logistic Regression", "Log. Reg.")
    .replace("Random Forest", "R. Forest")
    .replace("Decision Tree", "Dec. Tree");
}

const axis = { tick: { fontSize: 11 }, tickLine: false, axisLine: false };

export function AccuracyBySplit({
  metrics,
  models,
  splits,
}: {
  metrics: MetricMap;
  models: string[];
  splits: string[];
}) {
  const data = models.map((m) => {
    const row: Record<string, string | number> = { model: shortModel(m) };
    splits.forEach((s) => {
      row[SPLIT_SHORT[s] ?? s] = +metrics[s][m].accuracy.toFixed(3);
    });
    return row;
  });
  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#eee" />
        <XAxis dataKey="model" {...axis} />
        <YAxis domain={[0, 1]} {...axis} />
        <Tooltip />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {splits.map((s) => (
          <Bar key={s} dataKey={SPLIT_SHORT[s] ?? s} fill={SPLIT_COLOR[s]} radius={[4, 4, 0, 0]} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

export function TrainVsVal({
  train,
  val,
  models,
  split,
}: {
  train: MetricMap;
  val: MetricMap;
  models: string[];
  split: string;
}) {
  const data = models.map((m) => ({
    model: shortModel(m),
    Training: +train[split][m].accuracy.toFixed(3),
    Validasi: +val[split][m].accuracy.toFixed(3),
  }));
  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#eee" />
        <XAxis dataKey="model" {...axis} />
        <YAxis domain={[0, 1]} {...axis} />
        <Tooltip />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Bar dataKey="Training" fill="#cbd5e1" radius={[4, 4, 0, 0]} />
        <Bar dataKey="Validasi" fill="#10b981" radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
