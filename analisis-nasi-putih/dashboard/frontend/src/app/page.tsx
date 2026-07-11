"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Boxes, GitBranch, LineChart, Trees } from "lucide-react";
import { getOptions, type Options } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { AccuracyBySplit, shortModel, TrainVsVal } from "@/components/charts";

const BEST_SPLIT = "Grouped";

const MODEL_STYLE: Record<string, { color: string; soft: string; Icon: typeof LineChart }> = {
  "Logistic Regression": { color: "#10b981", soft: "bg-emerald-50 text-emerald-600", Icon: LineChart },
  "Decision Tree": { color: "#0ea5e9", soft: "bg-sky-50 text-sky-600", Icon: GitBranch },
  KNN: { color: "#8b5cf6", soft: "bg-violet-50 text-violet-600", Icon: Boxes },
  "Random Forest": { color: "#f59e0b", soft: "bg-amber-50 text-amber-600", Icon: Trees },
};

export default function Dashboard() {
  const [opt, setOpt] = useState<Options | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getOptions()
      .then(setOpt)
      .catch((e) => setError(`Gagal memuat data dari backend (:8000): ${e.message}`));
  }, []);

  return (
    <main className="p-6 space-y-6">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold">Dashboard</h1>
          <p className="text-sm text-muted-foreground">
            Ringkasan performa model klasifikasi kesegaran makanan.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <span className="inline-flex items-center gap-2 rounded-full border bg-white px-3 py-1.5 text-xs font-medium">
            <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" /> Sistem aktif
          </span>
          <Link
            href="/uji-model"
            className="inline-flex items-center justify-center rounded-lg bg-emerald-600 px-3.5 py-2 text-sm font-medium text-white transition-colors hover:bg-emerald-700"
          >
            + Uji Model
          </Link>
        </div>
      </header>

      {error && (
        <Card className="border-red-200">
          <CardContent className="pt-6 text-sm text-red-600">
            {error} — pastikan backend berjalan: <code>uvicorn main:app --port 8000</code>
          </CardContent>
        </Card>
      )}

      {/* Stat cards: one per model */}
      <section className="space-y-3">
        <SectionTitle title="Performa Model" sub="Akurasi validasi (split Grouped) per model" />
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {opt?.models.map((m) => {
            const style = MODEL_STYLE[m] ?? MODEL_STYLE["Logistic Regression"];
            const acc = opt.metrics[BEST_SPLIT][m].accuracy;
            const auc = opt.metrics[BEST_SPLIT][m].roc_auc;
            const trio = opt.splits.map((s) => opt.metrics[s][m].accuracy);
            return (
              <Card key={m}>
                <CardContent className="p-4 space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className={`flex h-8 w-8 items-center justify-center rounded-lg ${style.soft}`}>
                        <style.Icon className="h-4 w-4" />
                      </span>
                      <span className="text-sm font-medium">{shortModel(m)}</span>
                    </div>
                    <span className="rounded-md bg-muted px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground">
                      AUC {auc.toFixed(2)}
                    </span>
                  </div>
                  <div>
                    <div className="text-2xl font-bold">{(acc * 100).toFixed(1)}%</div>
                    <div className="text-xs text-muted-foreground">akurasi validasi 50/50 (grouped)</div>
                  </div>
                  <div className="h-1.5 w-full rounded-full bg-muted">
                    <div
                      className="h-1.5 rounded-full"
                      style={{ width: `${acc * 100}%`, background: style.color }}
                    />
                  </div>
                  <MiniBars values={trio} labels={["G", "R"]} color={style.color} />
                </CardContent>
              </Card>
            );
          })}
        </div>
      </section>

      {/* Charts */}
      <section className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Akurasi Validasi per Split</CardTitle>
            <p className="text-xs text-muted-foreground">
              Performa pada data baru — split acak menggelembung, Grouped jujur.
            </p>
          </CardHeader>
          <CardContent>
            {opt && (
              <AccuracyBySplit metrics={opt.metrics} models={opt.models} splits={opt.splits} />
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Klaim vs Realita (split acak)</CardTitle>
            <p className="text-xs text-muted-foreground">
              Training terlihat hampir sempurna, lalu ambruk di validasi (kebocoran).
            </p>
          </CardHeader>
          <CardContent>
            {opt && (
              <TrainVsVal
                train={opt.train_metrics}
                val={opt.metrics}
                models={opt.models}
                split="Random Split"
              />
            )}
          </CardContent>
        </Card>
      </section>
    </main>
  );
}

function SectionTitle({ title, sub }: { title: string; sub: string }) {
  return (
    <div className="flex items-baseline gap-2">
      <h2 className="text-sm font-semibold">{title}</h2>
      <span className="text-xs text-muted-foreground">{sub}</span>
    </div>
  );
}

function MiniBars({ values, labels, color }: { values: number[]; labels: string[]; color: string }) {
  return (
    <div className="flex items-end gap-2 pt-1">
      {values.map((v, i) => (
        <div key={i} className="flex flex-1 flex-col items-center gap-1">
          <div className="flex h-10 w-full items-end rounded bg-muted/60">
            <div
              className="w-full rounded"
              style={{ height: `${Math.max(8, v * 100)}%`, background: color, opacity: 0.5 + i * 0.25 }}
            />
          </div>
          <span className="text-[9px] text-muted-foreground">{labels[i]}</span>
        </div>
      ))}
    </div>
  );
}
