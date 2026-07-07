"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Trash2 } from "lucide-react";
import { clearHistory, getHistory, type TestEntry } from "@/lib/history";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

const SPLIT_SHORT: Record<string, string> = {
  Grouped: "Grouped",
  "Random Split": "Random",
};
const shortModel = (m: string) =>
  m.replace("Logistic Regression", "Log. Reg.").replace("Random Forest", "R. Forest").replace("Decision Tree", "Dec. Tree");

export default function Riwayat() {
  const [items, setItems] = useState<TestEntry[]>([]);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    setItems(getHistory());
    setReady(true);
  }, []);

  const total = items.length;
  const spoiled = items.filter((i) => i.label === "spoiled").length;
  const fresh = total - spoiled;

  function onClear() {
    clearHistory();
    setItems([]);
  }

  return (
    <main className="p-6 space-y-6">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold">Riwayat Pengujian</h1>
          <p className="text-sm text-muted-foreground">
            Log hasil uji model (tersimpan lokal di browser ini).
          </p>
        </div>
        {total > 0 && (
          <Button variant="destructive" size="sm" onClick={onClear}>
            <Trash2 className="h-4 w-4" /> Hapus Riwayat
          </Button>
        )}
      </header>

      <div className="grid grid-cols-3 gap-4">
        <Stat label="Total Uji" value={total} />
        <Stat label="Fresh" value={fresh} tone="green" />
        <Stat label="Spoiled" value={spoiled} tone="red" />
      </div>

      <Card>
        <CardContent className="p-0">
          {!ready ? null : total === 0 ? (
            <div className="p-10 text-center text-sm text-muted-foreground">
              Belum ada riwayat. Lakukan pengujian di{" "}
              <Link href="/uji-model" className="text-emerald-600 underline">
                Uji Model
              </Link>
              .
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b text-left text-xs text-muted-foreground">
                    <Th>Waktu</Th>
                    <Th>Makanan</Th>
                    <Th>Model</Th>
                    <Th>Split</Th>
                    <Th>Input (mq2/135/4)</Th>
                    <Th>Hasil</Th>
                    <Th>P(spoiled)</Th>
                    <Th>Sumber</Th>
                  </tr>
                </thead>
                <tbody>
                  {items.map((it) => {
                    const f = it.features;
                    const sp = it.label === "spoiled";
                    return (
                      <tr key={it.id} className="border-b last:border-0 hover:bg-muted/40">
                        <Td className="whitespace-nowrap text-muted-foreground">
                          {new Date(it.ts).toLocaleString("id-ID", {
                            dateStyle: "short",
                            timeStyle: "short",
                          })}
                        </Td>
                        <Td className="font-medium">{it.food_type}</Td>
                        <Td>{shortModel(it.model)}</Td>
                        <Td>{SPLIT_SHORT[it.split] ?? it.split}</Td>
                        <Td className="whitespace-nowrap text-xs text-muted-foreground">
                          {f.mq2}/{f.mq135}/{f.mq4}
                        </Td>
                        <Td>
                          <span
                            className={`rounded-md px-2 py-0.5 text-xs font-semibold ${
                              sp ? "bg-red-100 text-red-700" : "bg-green-100 text-green-700"
                            }`}
                          >
                            {sp ? "Spoiled" : "Fresh"}
                          </span>
                        </Td>
                        <Td>{(it.prob_spoiled * 100).toFixed(1)}%</Td>
                        <Td className="capitalize text-muted-foreground">{it.source}</Td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </main>
  );
}

function Stat({ label, value, tone }: { label: string; value: number; tone?: "green" | "red" }) {
  const color =
    tone === "green" ? "text-emerald-600" : tone === "red" ? "text-red-600" : "text-foreground";
  return (
    <Card>
      <CardContent className="p-4">
        <div className="text-xs text-muted-foreground">{label}</div>
        <div className={`text-2xl font-bold ${color}`}>{value}</div>
      </CardContent>
    </Card>
  );
}

function Th({ children }: { children: React.ReactNode }) {
  return <th className="px-4 py-2.5 font-medium">{children}</th>;
}
function Td({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <td className={`px-4 py-2.5 ${className}`}>{children}</td>;
}
