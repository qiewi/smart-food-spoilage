// Local test-history log, persisted in localStorage (no backend).

export type TestEntry = {
  id: string;
  ts: number; // epoch ms
  split: string;
  model: string;
  food_type: string;
  features: { mq2: number; mq135: number; mq4: number };
  label: "fresh" | "spoiled";
  prob_spoiled: number;
  source: "manual" | "sensor";
};

const KEY = "freshness_test_history";
const MAX = 200;

export function getHistory(): TestEntry[] {
  if (typeof window === "undefined") return [];
  try {
    return JSON.parse(localStorage.getItem(KEY) || "[]") as TestEntry[];
  } catch {
    return [];
  }
}

export function addHistory(e: Omit<TestEntry, "id" | "ts">): TestEntry[] {
  const entry: TestEntry = { ...e, id: crypto.randomUUID(), ts: Date.now() };
  const all = [entry, ...getHistory()].slice(0, MAX);
  localStorage.setItem(KEY, JSON.stringify(all));
  return all;
}

export function clearHistory(): void {
  localStorage.removeItem(KEY);
}
