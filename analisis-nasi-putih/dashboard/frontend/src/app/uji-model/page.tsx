"use client";

import { useEffect, useState } from "react";
import {
  getOptions,
  predict,
  readSensor,
  type Metrics,
  type Options,
  type PredictResult,
} from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { addHistory } from "@/lib/history";

const SPLIT_LABEL: Record<string, string> = {
  Grouped: "Grouped (per-run, jujur)",
  "Random Split": "Random Split (acak, rawan bocor)",
};
const LEAKY_SPLITS = ["Random Split"];

type Feat = { mq2: string; mq135: string; mq4: string };
const EMPTY: Feat = { mq2: "", mq135: "", mq4: "" };
const FIELDS: { key: keyof Feat; label: string }[] = [
  { key: "mq2", label: "MQ2 (gas)" },
  { key: "mq135", label: "MQ135 (gas)" },
  { key: "mq4", label: "MQ4 (gas)" },
];

export default function UjiModel() {
  const [opt, setOpt] = useState<Options | null>(null);
  const [split, setSplit] = useState("");
  const [model, setModel] = useState("");
  const [food, setFood] = useState("");
  const [feat, setFeat] = useState<Feat>(EMPTY);
  const [mode, setMode] = useState("manual");
  const [port, setPort] = useState("COM4");
  const [sensorInfo, setSensorInfo] = useState<string | null>(null);

  const [result, setResult] = useState<PredictResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [reading, setReading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getOptions()
      .then((o) => {
        setOpt(o);
        setSplit(o.splits.includes("Grouped") ? "Grouped" : o.splits[0]);
        setModel(
          o.models.includes("Logistic Regression") ? "Logistic Regression" : o.models[0]
        );
        setFood(o.food_types[0]);
      })
      .catch((e) => setError(`Gagal memuat opsi dari backend: ${e.message}`));
  }, []);

  const metrics = opt?.metrics?.[split]?.[model] ?? null;
  const isRandom = LEAKY_SPLITS.includes(split);

  async function onPredict() {
    setError(null);
    setResult(null);
    const nums = FIELDS.map((f) => parseFloat(feat[f.key]));
    if (!food || nums.some((n) => Number.isNaN(n))) {
      setError("Lengkapi semua nilai sensor dan pilih tipe makanan.");
      return;
    }
    setBusy(true);
    try {
      const r = await predict({
        split,
        model,
        food_type: food,
        mq2: nums[0],
        mq135: nums[1],
        mq4: nums[2],
      });
      setResult(r);
      addHistory({
        split,
        model,
        food_type: food,
        features: { mq2: nums[0], mq135: nums[1], mq4: nums[2] },
        label: r.label,
        prob_spoiled: r.prob_spoiled,
        source: mode === "sensor" ? "sensor" : "manual",
      });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function onReadSensor() {
    setError(null);
    setSensorInfo(null);
    setReading(true);
    try {
      const s = await readSensor(port, 10);
      setFeat({
        mq2: s.mq2.toFixed(1),
        mq135: s.mq135.toFixed(1),
        mq4: s.mq4.toFixed(1),
      });
      setSensorInfo(`Rata-rata dari ${s.samples} sampel (${s.seconds} dtk; humidity/temp diabaikan).`);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setReading(false);
    }
  }

  const spoiled = result?.label === "spoiled";
  const confidence = result ? (spoiled ? result.prob_spoiled : result.prob_fresh) * 100 : 0;

  return (
    <main className="p-6">
      <div className="mx-auto max-w-2xl space-y-6">
        <header>
          <h1 className="text-2xl font-bold">Uji Model</h1>
          <p className="text-sm text-muted-foreground">
            Klasifikasi <b>Fresh</b> vs <b>Spoiled</b> dari sensor gas (MQ2/MQ135/MQ4).
          </p>
        </header>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Konfigurasi</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
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
              <Field label="Tipe Makanan">
                <Select value={food} onValueChange={(v) => setFood(v ?? "")}>
                  <SelectTrigger className="w-full"><SelectValue placeholder="Makanan" /></SelectTrigger>
                  <SelectContent>
                    {opt?.food_types.map((f) => (
                      <SelectItem key={f} value={f}>{f}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </Field>
            </div>

            {isRandom && (
              <p className="text-xs text-amber-600">
                ⚠️ Split ini berbasis baris acak (rawan kebocoran) — bandingkan dengan
                Grouped untuk melihat performa jujur pada data baru.
              </p>
            )}

            <Tabs value={mode} onValueChange={setMode}>
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="manual">Input Manual</TabsTrigger>
                <TabsTrigger value="sensor">Baca Sensor</TabsTrigger>
              </TabsList>

              <TabsContent value="manual" className="pt-3">
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                  {FIELDS.map((f) => (
                    <Field key={f.key} label={f.label}>
                      <Input
                        type="number"
                        step="any"
                        value={feat[f.key]}
                        onChange={(e) => setFeat({ ...feat, [f.key]: e.target.value })}
                        placeholder="0"
                      />
                    </Field>
                  ))}
                </div>
              </TabsContent>

              <TabsContent value="sensor" className="pt-3 space-y-3">
                <div className="flex items-end gap-3">
                  <Field label="Port serial">
                    <Input value={port} onChange={(e) => setPort(e.target.value)} className="w-32" />
                  </Field>
                  <Button onClick={onReadSensor} disabled={reading} variant="secondary">
                    {reading ? "Membaca… (10 dtk)" : "Baca Sensor (10 dtk)"}
                  </Button>
                </div>
                {sensorInfo && <p className="text-xs text-muted-foreground">{sensorInfo}</p>}
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                  {FIELDS.map((f) => (
                    <Field key={f.key} label={f.label}>
                      <Input value={feat[f.key]} readOnly className="bg-muted" placeholder="—" />
                    </Field>
                  ))}
                </div>
              </TabsContent>
            </Tabs>

            <Button onClick={onPredict} disabled={busy} className="w-full bg-emerald-600 hover:bg-emerald-700">
              {busy ? "Memprediksi…" : "Prediksi"}
            </Button>

            {error && <p className="text-sm text-red-600">{error}</p>}
          </CardContent>
        </Card>

        {result && (
          <Card className={spoiled ? "border-red-300" : "border-green-300"}>
            <CardContent className="pt-6 text-center space-y-3">
              <div
                className={`inline-block rounded-lg px-6 py-3 text-2xl font-bold ${
                  spoiled ? "bg-red-100 text-red-700" : "bg-green-100 text-green-700"
                }`}
              >
                {spoiled ? "SPOILED (Busuk)" : "FRESH (Segar)"}
              </div>
              <p className="text-sm text-muted-foreground">
                Keyakinan: <b>{confidence.toFixed(1)}%</b>
                {"  ·  "}P(spoiled) = {(result.prob_spoiled * 100).toFixed(1)}%
              </p>
              <MetricRow title="Metrik validasi model ini" m={result.model_metrics} />
            </CardContent>
          </Card>
        )}

        {!result && metrics && (
          <Card>
            <CardContent className="pt-6">
              <MetricRow
                title={`Metrik validasi — ${model} / ${SPLIT_LABEL[split] ?? split}`}
                m={metrics}
              />
            </CardContent>
          </Card>
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

function MetricRow({ title, m }: { title: string; m: Metrics }) {
  const items: [string, number][] = [
    ["Accuracy", m.accuracy],
    ["Precision", m.precision_macro],
    ["Recall", m.recall_macro],
    ["F1", m.f1_macro],
    ["ROC-AUC", m.roc_auc],
  ];
  return (
    <div>
      <p className="text-xs text-muted-foreground mb-2">{title}</p>
      <div className="grid grid-cols-5 gap-2">
        {items.map(([k, v]) => (
          <div key={k} className="rounded-md border p-2">
            <div className="text-[10px] uppercase text-muted-foreground">{k}</div>
            <div className="text-sm font-semibold">{v.toFixed(3)}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
