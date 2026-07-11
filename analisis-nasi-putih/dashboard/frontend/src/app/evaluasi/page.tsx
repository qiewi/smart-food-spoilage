"use client";

import { useEffect, useRef, useState } from "react";
import { Upload, X } from "lucide-react";
import {
  evaluateCsv,
  getOptions,
  type EvalMetrics,
  type EvalResult,
  type Metrics,
  type Options,
} from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const SPLIT_LABEL: Record<string, string> = {
  Grouped: "Grouped (per-run, jujur)",
  "Random Split": "Random Split (acak, rawan bocor)",
};

export default function Evaluasi() {
  const [opt, setOpt] = useState<Options | null>(null);
  const [split, setSplit] = useState("");
  const [model, setModel] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [result, setResult] = useState<EvalResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    getOptions()
      .then((o) => {
        setOpt(o);
        setSplit(o.splits.includes("Grouped") ? "Grouped" : o.splits[0]);
        setModel(o.models.includes("Logistic Regression") ? "Logistic Regression" : o.models[0]);
      })
      .catch((e) => setError(`Gagal memuat opsi dari backend: ${e.message}`));
  }, []);

  function addFiles(list: FileList | null) {
    if (!list) return;
    const incoming = Array.from(list).filter((f) => f.name.toLowerCase().endsWith(".csv"));
    setFiles((prev) => {
      const names = new Set(prev.map((f) => f.name));
      return [...prev, ...incoming.filter((f) => !names.has(f.name))];
    });
  }

  async function onEvaluate() {
    setError(null);
    setResult(null);
    if (!files.length) {
      setError("Pilih minimal satu file CSV.");
      return;
    }
    setBusy(true);
    try {
      const payload = await Promise.all(
        files.map(async (f) => ({ name: f.name, content: await f.text() }))
      );
      const r = await evaluateCsv({ split, model, files: payload });
      setResult(r);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="p-6">
      <div className="mx-auto max-w-3xl space-y-6">
        <header>
          <h1 className="text-2xl font-bold">Evaluasi CSV</h1>
          <p className="text-sm text-muted-foreground">
            Unggah satu atau lebih file CSV berlabel untuk mengukur performa model terpilih pada
            data baru. Terpisah dari uji data satuan (manual / sensor).
          </p>
        </header>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Konfigurasi</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <Field label="Jenis Split">
                <Select value={split} onValueChange={(v) => setSplit(v ?? "")}>
                  <SelectTrigger className="w-full"><SelectValue placeholder="Split" /></SelectTrigger>
                  <SelectContent>
                    {opt?.splits.map((s) => (
                      <SelectItem key={s} value={s}>{SPLIT_LABEL[s] ?? s}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </Field>
              <Field label="Jenis Model">
                <Select value={model} onValueChange={(v) => setModel(v ?? "")}>
                  <SelectTrigger className="w-full"><SelectValue placeholder="Model" /></SelectTrigger>
                  <SelectContent>
                    {opt?.models.map((m) => (
                      <SelectItem key={m} value={m}>{m}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </Field>
            </div>

            {/* dropzone */}
            <div
              onClick={() => inputRef.current?.click()}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                addFiles(e.dataTransfer.files);
              }}
              className="flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed border-muted-foreground/25 bg-muted/30 py-8 text-center transition-colors hover:border-emerald-400 hover:bg-emerald-50/50"
            >
              <Upload className="h-6 w-6 text-muted-foreground" />
              <p className="text-sm font-medium">Seret CSV ke sini atau klik untuk memilih</p>
              <p className="text-xs text-muted-foreground">
                Kolom wajib: mq2, mq135, mq4, food_type, label (elapsed opsional)
              </p>
              <input
                ref={inputRef}
                type="file"
                accept=".csv"
                multiple
                className="hidden"
                onChange={(e) => addFiles(e.target.files)}
              />
            </div>

            {files.length > 0 && (
              <ul className="space-y-1.5">
                {files.map((f) => (
                  <li
                    key={f.name}
                    className="flex items-center justify-between rounded-lg border bg-white px-3 py-2 text-sm dark:bg-zinc-950"
                  >
                    <span className="truncate">{f.name}</span>
                    <span className="flex items-center gap-3">
                      <span className="text-xs text-muted-foreground">
                        {(f.size / 1024).toFixed(0)} KB
                      </span>
                      <button
                        onClick={() => setFiles((prev) => prev.filter((x) => x.name !== f.name))}
                        className="text-muted-foreground hover:text-red-600"
                      >
                        <X className="h-4 w-4" />
                      </button>
                    </span>
                  </li>
                ))}
              </ul>
            )}

            <Button
              onClick={onEvaluate}
              disabled={busy || !files.length}
              className="w-full bg-emerald-600 hover:bg-emerald-700"
            >
              {busy ? "Mengevaluasi…" : `Evaluasi ${files.length || ""} file`}
            </Button>

            {error && <p className="text-sm text-red-600">{error}</p>}
          </CardContent>
        </Card>

        {result && (
          <>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">
                  Performa gabungan — {result.model} / {SPLIT_LABEL[result.split] ?? result.split}
                </CardTitle>
                <p className="text-xs text-muted-foreground">
                  {result.n_files} file · {result.overall.n_rows} baris (fresh{" "}
                  {result.overall.n_fresh} / spoiled {result.overall.n_spoiled})
                </p>
              </CardHeader>
              <CardContent className="space-y-4">
                <MetricGrid m={result.overall} />
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <Confusion cm={result.overall.confusion} />
                  <RefCompare ref_m={result.reference_metrics} got={result.overall} />
                </div>
              </CardContent>
            </Card>

            {result.per_file.length > 1 && (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">Rincian per file</CardTitle>
                </CardHeader>
                <CardContent className="p-0">
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b text-left text-xs text-muted-foreground">
                          <Th>File</Th>
                          <Th>Baris</Th>
                          <Th>Acc</Th>
                          <Th>Prec</Th>
                          <Th>Rec</Th>
                          <Th>F1</Th>
                          <Th>AUC</Th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.per_file.map((f) => (
                          <tr key={f.filename} className="border-b last:border-0 hover:bg-muted/40">
                            <Td className="max-w-55 truncate font-medium">{f.filename}</Td>
                            <Td className="text-muted-foreground">{f.n_rows}</Td>
                            <Td>{f.accuracy.toFixed(3)}</Td>
                            <Td>{f.precision_macro.toFixed(3)}</Td>
                            <Td>{f.recall_macro.toFixed(3)}</Td>
                            <Td>{f.f1_macro.toFixed(3)}</Td>
                            <Td>{f.roc_auc == null ? "—" : f.roc_auc.toFixed(3)}</Td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </CardContent>
              </Card>
            )}

            {result.warnings.length > 0 && (
              <Card className="border-amber-200">
                <CardContent className="pt-6 space-y-1 text-sm text-amber-700">
                  {result.warnings.map((w, i) => (
                    <p key={i}>⚠️ {w}</p>
                  ))}
                </CardContent>
              </Card>
            )}
          </>
        )}
      </div>
    </main>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1.5">
      <Label className="text-xs text-muted-foreground">{label}</Label>
      {children}
    </div>
  );
}

function MetricGrid({ m }: { m: EvalMetrics }) {
  const items: [string, number | null][] = [
    ["Accuracy", m.accuracy],
    ["Precision", m.precision_macro],
    ["Recall", m.recall_macro],
    ["F1", m.f1_macro],
    ["ROC-AUC", m.roc_auc],
  ];
  return (
    <div className="grid grid-cols-3 sm:grid-cols-5 gap-2">
      {items.map(([k, v]) => (
        <div key={k} className="rounded-md border p-2.5 text-center">
          <div className="text-[10px] uppercase text-muted-foreground">{k}</div>
          <div className="text-lg font-bold">{v == null ? "—" : v.toFixed(3)}</div>
        </div>
      ))}
    </div>
  );
}

function Confusion({ cm }: { cm: number[][] }) {
  const [tn, fp] = cm[0];
  const [fn, tp] = cm[1];
  const max = Math.max(tn, fp, fn, tp, 1);
  const cell = (v: number, correct: boolean) => (
    <div
      className="flex h-16 items-center justify-center rounded-md text-sm font-bold"
      style={{
        background: `rgba(16,185,129,${0.12 + (v / max) * 0.6})`,
        color: v > max * 0.6 ? "#fff" : "#064e3b",
        outline: correct ? "2px solid #10b981" : "none",
        outlineOffset: "-2px",
      }}
    >
      {v}
    </div>
  );
  return (
    <div>
      <p className="mb-2 text-xs text-muted-foreground">Confusion matrix (baris = aktual)</p>
      <div className="grid grid-cols-[auto_1fr_1fr] gap-1 text-[11px]">
        <div />
        <div className="text-center text-muted-foreground">pred fresh</div>
        <div className="text-center text-muted-foreground">pred spoiled</div>
        <div className="flex items-center text-muted-foreground">fresh</div>
        {cell(tn, true)}
        {cell(fp, false)}
        <div className="flex items-center text-muted-foreground">spoiled</div>
        {cell(fn, false)}
        {cell(tp, true)}
      </div>
    </div>
  );
}

function RefCompare({ ref_m, got }: { ref_m: EvalMetrics | Metrics; got: EvalMetrics }) {
  const rows: [string, number | null, number | null][] = [
    ["Accuracy", ref_m.accuracy, got.accuracy],
    ["F1-macro", ref_m.f1_macro, got.f1_macro],
    ["ROC-AUC", ref_m.roc_auc, got.roc_auc],
  ];
  return (
    <div>
      <p className="mb-2 text-xs text-muted-foreground">
        Perbandingan vs validasi acuan (50/50)
      </p>
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-xs text-muted-foreground">
            <th className="pb-1 font-medium">Metrik</th>
            <th className="pb-1 font-medium">Acuan</th>
            <th className="pb-1 font-medium">CSV ini</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([k, a, b]) => (
            <tr key={k} className="border-t">
              <td className="py-1.5">{k}</td>
              <td className="py-1.5 text-muted-foreground">{a == null ? "—" : a.toFixed(3)}</td>
              <td className="py-1.5 font-semibold">{b == null ? "—" : b.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Th({ children }: { children: React.ReactNode }) {
  return <th className="px-4 py-2.5 font-medium">{children}</th>;
}
function Td({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <td className={`px-4 py-2.5 ${className}`}>{children}</td>;
}
