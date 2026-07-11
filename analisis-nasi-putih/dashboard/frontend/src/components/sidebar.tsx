"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { FileSpreadsheet, FlaskConical, History, LayoutDashboard, Leaf, Settings } from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

const NAV = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/uji-model", label: "Uji Model", icon: FlaskConical },
  { href: "/evaluasi", label: "Evaluasi CSV", icon: FileSpreadsheet },
  { href: "/riwayat", label: "Riwayat", icon: History },
];

// Komoditas yang tersedia. Untuk sekarang hanya Nasi Putih; tambah makanan lain di sini
// (beserta ikon emoji-nya, mis. { name: "Ayam Goreng", icon: "🍗" }) ketika modelnya siap.
const FOODS = [
  { name: "Nasi Putih", icon: "🍚" },
];

export default function Sidebar() {
  const path = usePathname();
  const [food, setFood] = useState(FOODS[0].name);
  return (
    <aside className="hidden md:flex w-60 shrink-0 flex-col border-r bg-white dark:bg-zinc-950">
      <div className="flex items-center gap-2.5 px-5 h-16 border-b">
        <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-emerald-500 text-white">
          <Leaf className="h-5 w-5" />
        </div>
        <div className="leading-tight">
          <div className="font-bold text-sm">Smart Canteen</div>
          <div className="text-[10px] tracking-wider text-muted-foreground">SPOILAGE DETECTOR</div>
        </div>
      </div>

      <div className="px-3 pt-4">
        <p className="px-3 pb-1 text-[10px] font-semibold tracking-wider text-muted-foreground">
          KOMODITAS
        </p>
        <Select value={food} onValueChange={(v) => setFood(v ?? FOODS[0].name)}>
          <SelectTrigger className="w-full">
            <SelectValue placeholder="Pilih makanan" />
          </SelectTrigger>
          <SelectContent>
            {FOODS.map((f) => (
              <SelectItem key={f.name} value={f.name}>
                <span className="mr-1.5">{f.icon}</span>
                {f.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <nav className="flex-1 px-3 py-4 space-y-1">
        <p className="px-3 pb-1 text-[10px] font-semibold tracking-wider text-muted-foreground">
          MENU
        </p>
        {NAV.map(({ href, label, icon: Icon }) => {
          const active = path === href;
          return (
            <Link
              key={href}
              href={href}
              className={`flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors ${
                active
                  ? "bg-emerald-50 text-emerald-700 font-medium dark:bg-emerald-950 dark:text-emerald-300"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground"
              }`}
            >
              <Icon className="h-4 w-4" />
              {label}
            </Link>
          );
        })}
      </nav>

      <div className="border-t p-3 space-y-2">
        <div className="flex items-center gap-2 rounded-lg px-3 py-2 text-sm text-muted-foreground">
          <Settings className="h-4 w-4" /> Pengaturan
        </div>
        <div className="flex items-center gap-2.5 rounded-lg px-3 py-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-emerald-100 text-emerald-700 text-xs font-bold">
            RA
          </div>
          <div className="leading-tight">
            <div className="text-sm font-medium">Rizqi Andhika</div>
            <div className="text-[10px] text-muted-foreground">Peneliti</div>
          </div>
        </div>
      </div>
    </aside>
  );
}
