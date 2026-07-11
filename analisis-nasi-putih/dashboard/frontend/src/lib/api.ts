const BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type Metrics = {
  accuracy: number;
  precision_macro: number;
  recall_macro: number;
  f1_macro: number;
  roc_auc: number;
};

export type MetricMap = Record<string, Record<string, Metrics>>;

export type Options = {
  splits: string[];
  models: string[];
  food_types: string[];
  metrics: MetricMap;        // validation metrics per split -> model
  train_metrics: MetricMap;  // training/internal metrics per split -> model
};

export type PredictResult = {
  label: "fresh" | "spoiled";
  prob_spoiled: number;
  prob_fresh: number;
  model_metrics: Metrics;
};

export type SensorResult = {
  samples: number;
  seconds: number;
  mq2: number;
  mq135: number;
  mq4: number;
  humidity: number;
  tempC: number;
};

export type Features = {
  mq2: number;
  mq135: number;
  mq4: number;
};

export type EvalMetrics = {
  n_rows: number;
  n_fresh: number;
  n_spoiled: number;
  confusion: number[][]; // [[TN, FP], [FN, TP]] for [fresh, spoiled]
  accuracy: number;
  precision_macro: number;
  recall_macro: number;
  f1_macro: number;
  roc_auc: number | null;
};

export type FileEval = EvalMetrics & { filename: string };

export type EvalResult = {
  split: string;
  model: string;
  n_files: number;
  overall: EvalMetrics;
  per_file: FileEval[];
  warnings: string[];
  reference_metrics: Metrics;
};

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json();
}

export async function getOptions(): Promise<Options> {
  return handle(await fetch(`${BASE}/api/options`));
}

export async function predict(body: {
  split: string;
  model: string;
  food_type: string;
} & Features): Promise<PredictResult> {
  return handle(
    await fetch(`${BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
  );
}

export async function evaluateCsv(body: {
  split: string;
  model: string;
  files: { name: string; content: string }[];
}): Promise<EvalResult> {
  return handle(
    await fetch(`${BASE}/api/evaluate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
  );
}

export async function readSensor(port = "COM4", seconds = 10): Promise<SensorResult> {
  return handle(
    await fetch(`${BASE}/api/sensor/read`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ port, seconds }),
    })
  );
}
