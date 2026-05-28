"""
============================================================
Smart Canteen — Relabel Dataset (Buffer scheme → Fresh/Spoiled)
============================================================
Mengubah label dataset CSV yang sudah ada ke skema BARU:

  Makanan          Fresh         Spoiled
  ──────────────────────────────────────────
  Nasi Putih       0 – 4 jam     ≥ 4 jam
  Ayam Goreng      0 – 4 jam     ≥ 4 jam
  Ikan Goreng      0 – 10 jam    ≥ 10 jam
  Telur Rebus      0 – 24 jam    ≥ 24 jam

Label dihitung ULANG dari kolom 'elapsed' (HH:MM:SS),
jadi label lama (fresh/spoiled/buffer) akan ditimpa.

Cara pakai:
  1. python relabel.py
  2. Paste path file CSV satu per satu, tekan Enter tiap baris
  3. Kosongkan (Enter langsung) saat selesai → proses dimulai
  4. Hasil disimpan ke folder "relabeled/"
     (file asli TIDAK diubah / tetap aman)
============================================================
"""

import csv
import os

# ── THRESHOLD BARU (batas atas FRESH, dalam JAM) ────────────
#   elapsed < threshold  → "fresh"
#   elapsed >= threshold → "spoiled"
FRESH_THRESHOLD_HOURS = {
    "Nasi Putih":   4,
    "Ayam Goreng":  4,
    "Ikan Goreng":  10,
    "Telur Rebus":  24,
}

# Alias supaya nama yang sedikit beda tetap kebaca
# (mis. CSV berisi "Ikan Goreng Nila" → dipetakan ke "Ikan Goreng")
ALIASES = {
    "ikan goreng nila": "Ikan Goreng",
    "ikan goreng":      "Ikan Goreng",
    "nasi putih":       "Nasi Putih",
    "nasi":             "Nasi Putih",
    "ayam goreng":      "Ayam Goreng",
    "telur rebus":      "Telur Rebus",
}


def normalize_food(name: str) -> str:
    """Cocokkan nama makanan ke key threshold standar."""
    key = name.strip().lower()
    return ALIASES.get(key, name.strip())


def elapsed_to_hours(elapsed_str: str) -> float:
    """Konversi 'HH:MM:SS' → jam (float). Dukung juga format ms (angka)."""
    s = elapsed_str.strip()
    if ":" in s:
        h, m, sec = (int(x) for x in s.split(":"))
        return h + m / 60 + sec / 3600
    else:
        # fallback kalau dataset lama masih pakai ms (angka mentah)
        return int(s) / 3_600_000


def get_label(elapsed_hours: float, food_type: str) -> str:
    norm = normalize_food(food_type)
    threshold = FRESH_THRESHOLD_HOURS.get(norm)
    if threshold is None:
        return "UNKNOWN_FOOD"
    return "spoiled" if elapsed_hours >= threshold else "fresh"


def relabel_file(path: str, out_dir: str = "relabeled") -> dict:
    """Relabel satu file CSV, simpan ke folder 'relabeled/'."""
    os.makedirs(out_dir, exist_ok=True)

    base     = os.path.basename(path)
    name     = base.replace(".csv", "_relabeled.csv")
    if name == base:  # jaga-jaga kalau tak berakhiran .csv
        name = base + "_relabeled.csv"
    out_path = os.path.join(out_dir, name)

    stats = {"fresh": 0, "spoiled": 0, "unknown": 0, "total": 0}

    with open(path, "r", newline="", encoding="utf-8") as fin, \
         open(out_path, "w", newline="", encoding="utf-8") as fout:

        reader = csv.reader(fin)
        writer = csv.writer(fout)

        header = next(reader)
        writer.writerow(header)

        # Cari index kolom 'elapsed', 'food_type', 'label'
        try:
            i_elapsed = header.index("elapsed")
        except ValueError:
            # dataset lama mungkin pakai 'ms'
            i_elapsed = header.index("ms")
        i_food  = header.index("food_type")
        i_label = header.index("label")

        for row in reader:
            if len(row) <= max(i_elapsed, i_food, i_label):
                continue  # skip baris rusak

            hours     = elapsed_to_hours(row[i_elapsed])
            new_label = get_label(hours, row[i_food])

            row[i_label] = new_label
            writer.writerow(row)

            stats["total"] += 1
            if new_label == "fresh":
                stats["fresh"] += 1
            elif new_label == "spoiled":
                stats["spoiled"] += 1
            else:
                stats["unknown"] += 1

    stats["out_path"] = out_path
    return stats


def main():
    print("=" * 60)
    print("  Smart Canteen — Relabel Dataset")
    print("=" * 60)
    print("Paste path file CSV satu per satu, lalu tekan Enter.")
    print("Kalau sudah selesai, kosongkan saja (Enter langsung) untuk mulai.")
    print("Tip: bisa drag-and-drop file ke terminal untuk dapat path-nya.\n")

    paths = []
    while True:
        raw = input(f"  [{len(paths)+1}] Path CSV (atau Enter untuk selesai): ").strip()
        if raw == "":
            break

        # Bersihkan tanda kutip kalau ikut ke-paste (umum di Windows/drag-drop)
        raw = raw.strip().strip('"').strip("'")

        if not os.path.isfile(raw):
            print(f"      ⚠️  File tidak ditemukan, dilewati: {raw}")
            continue
        if not raw.lower().endswith(".csv"):
            print(f"      ⚠️  Bukan file .csv, dilewati: {raw}")
            continue

        paths.append(raw)
        print(f"      ✅  ditambahkan")

    if not paths:
        print("\n❌  Tidak ada file yang dimasukkan. Keluar.")
        return

    out_dir = "relabeled"
    print("\n" + "=" * 60)
    print(f"  Memproses {len(paths)} file → folder '{out_dir}/'")
    print("=" * 60)

    grand_fresh = grand_spoiled = grand_unknown = 0

    for path in paths:
        s = relabel_file(path, out_dir)
        print(f"\n📄  {os.path.basename(path)}")
        print(f"    → {out_dir}/{os.path.basename(s['out_path'])}")
        print(f"    fresh: {s['fresh']}  |  spoiled: {s['spoiled']}"
              + (f"  |  ⚠️ unknown: {s['unknown']}" if s['unknown'] else ""))
        grand_fresh   += s["fresh"]
        grand_spoiled += s["spoiled"]
        grand_unknown += s["unknown"]

    print("\n" + "=" * 60)
    print("  TOTAL SEMUA FILE")
    print("=" * 60)
    print(f"  Fresh   : {grand_fresh}")
    print(f"  Spoiled : {grand_spoiled}")
    if grand_unknown:
        print(f"  ⚠️  Unknown food (tidak ke-mapping): {grand_unknown}")
        print(f"      Cek kolom food_type & tambahkan ke ALIASES.")
    print("=" * 60)
    print(f"  File asli tetap aman. Hasil ada di folder '{out_dir}/'")


if __name__ == "__main__":
    main()