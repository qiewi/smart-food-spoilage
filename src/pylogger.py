"""
============================================================
Smart Canteen — Data Logger + Auto-Labeling (with Buffer Zone)
============================================================
Jalankan : python logger.py
Kebutuhan: pip install pyserial

Skema labeling per makanan (suhu ruang ~28-32°C):

  Makanan          Fresh        Buffer (skip)  Spoiled     Total Eksperimen
  ─────────────────────────────────────────────────────────────────────────
  Nasi Putih       0 – 4 jam    4 – 8 jam      8 – 16 jam  16 jam
  Ayam Goreng      0 – 2 jam    2 – 5 jam      5 – 10 jam  10 jam
  Ikan Goreng      0 – 6 jam    6 – 12 jam    12 – 24 jam  24 jam
  Telur Rebus      0 – 8 jam    8 – 20 jam    20 – 36 jam  36 jam

Row yang jatuh di zona BUFFER tidak ditulis ke CSV sama sekali
(dibuang agar model tidak belajar dari data ambigus).

Output CSV kolom:
  ms, mq2, mq135, mq4, humidity, tempC, food_type, label
  label hanya "fresh" atau "spoiled"
============================================================
"""

import serial
import csv
import sys
from datetime import datetime, timedelta

# ── ⚙️  KONFIGURASI PORT ─────────────────────────────────────
# Cek di: Device Manager → Ports (COM & LPT)
PORT = "COM4"
BAUD = 115200


# ── ⏱️  THRESHOLD PER MAKANAN (dalam JAM) ───────────────────
# Format nilai: (fresh_end_jam, buffer_end_jam, exp_end_jam)
#
#   0             → fresh_end   : label "fresh"
#   fresh_end     → buffer_end  : BUFFER — baris di-SKIP, tidak ditulis
#   buffer_end    → exp_end     : label "spoiled"
#   > exp_end                   : logging berhenti otomatis
#
FOOD_ZONES_HOURS = {
    "Nasi Putih":   ( 4,  8, 16),
    "Ayam Goreng":  ( 4,  8, 20),
    "Ikan Goreng":  ( 6, 12, 24),
    "Telur Rebus":  ( 8, 24, 36),
    "Nasi Goreng":  ( 6, 12, 16),
}

def _to_ms(hours: float) -> int:
    return int(hours * 3600 * 1000)

FOOD_ZONES_MS = {
    k: tuple(_to_ms(h) for h in v)
    for k, v in FOOD_ZONES_HOURS.items()
}

FOOD_OPTIONS = list(FOOD_ZONES_HOURS.keys())


# ── Fungsi Labeling ──────────────────────────────────────────

def get_label(elapsed_ms: int, food_type: str) -> str:
    """
    Kembalikan:
      "fresh"   — tulis ke CSV
      "buffer"  — SKIP, tidak ditulis
      "spoiled" — tulis ke CSV
      "done"    — eksperimen selesai, loop berhenti
    """
    fresh_end, buffer_end, exp_end = FOOD_ZONES_MS[food_type]

    if elapsed_ms > exp_end:
        return "done"
    elif elapsed_ms >= buffer_end:
        return "spoiled"
    elif elapsed_ms >= fresh_end:
        return "buffer"
    else:
        return "fresh"


# ── Fungsi UI ────────────────────────────────────────────────

def choose_offset() -> int:
    """Tanya berapa menit sejak makanan selesai dimasak. Kembalikan offset dalam ms."""
    print("Berapa menit sejak makanan selesai dimasak?")
    print("(Isi 0 kalau langsung mulai logging setelah masak)\n")
    while True:
        try:
            raw = input("Offset waktu (menit, contoh: 60): ").strip()
            offset_min = float(raw)
            if offset_min < 0:
                print("Tidak boleh negatif. Coba lagi.")
                continue
            offset_ms = int(offset_min * 60 * 1000)
            if offset_min > 0:
                print(f"\n✅  Offset      : {offset_min:.0f} menit "
                      f"= {offset_min/60:.2f} jam ({offset_ms:,} ms)")
                print(f"    Artinya ms dari ESP32 akan ditambah {offset_ms:,} ms "
                      f"sebelum dicek labelnya.\n")
            else:
                print(f"\n✅  Offset      : 0 menit (logging mulai dari awal)\n")
            return offset_ms
        except ValueError:
            print("Input tidak valid. Masukkan angka, contoh: 60 atau 90")


def choose_food() -> str:
    print("\n" + "=" * 62)
    print("   Smart Canteen — Food Freshness Logger")
    print("=" * 62)
    print("Pilih jenis makanan yang sedang di-logging:\n")
    print(f"  {'No':<5} {'Makanan':<16} {'Fresh':<13} {'Buffer (skip)':<16} {'Spoiled'}")
    print(f"  {'──':<5} {'──────':<16} {'─────':<13} {'─────────────':<16} {'───────'}")
    for i, name in enumerate(FOOD_OPTIONS, 1):
        fh, bh, sh = FOOD_ZONES_HOURS[name]
        print(f"  [{i}]   {name:<16} 0–{fh} jam      {fh}–{bh} jam          {bh}–{sh} jam")
    print()

    while True:
        try:
            choice = int(input("Masukkan nomor pilihan: ").strip())
            if 1 <= choice <= len(FOOD_OPTIONS):
                selected = FOOD_OPTIONS[choice - 1]
                fh, bh, sh = FOOD_ZONES_HOURS[selected]
                est_end = datetime.now() + timedelta(hours=sh)
                print(f"\n✅  Makanan     : {selected}")
                print(f"    Fresh       : 0 – {fh} jam")
                print(f"    Buffer skip : {fh} – {bh} jam")
                print(f"    Spoiled     : {bh} – {sh} jam")
                print(f"    Selesai est.: {est_end.strftime('%H:%M')} "
                      f"(+{sh} jam dari sekarang)\n")
                return selected
            else:
                print(f"Masukkan angka 1–{len(FOOD_OPTIONS)}.")
        except ValueError:
            print("Input tidak valid. Coba lagi.")


def make_filename(food_type: str) -> str:
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = food_type.replace(" ", "_")
    return f"dataset_{safe_name}_{ts}.csv"


# ── Main ──────────────────────────────────────────────────────

def main():
    food_type  = choose_food()
    offset_ms  = choose_offset()
    filename   = make_filename(food_type)

    print(f"Membuka port {PORT} @ {BAUD} baud...")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=5)
    except serial.SerialException as e:
        print(f"\n❌  Gagal membuka port: {e}")
        print("Pastikan port benar & ESP32 sudah terhubung.")
        sys.exit(1)

    print(f"Menyimpan ke : {filename}")
    print("Tekan CTRL+C untuk berhenti lebih awal.\n")

    count_fresh   = 0
    count_spoiled = 0
    count_skipped = 0

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # ── Tunggu header dari ESP32 ──────────────────────────
        print("Menunggu header dari ESP32", end="", flush=True)
        while True:
            raw = ser.readline().decode(errors="ignore").strip()
            if raw == "ms,mq2,mq135,mq4,humidity,tempC":
                writer.writerow(["elapsed", "mq2", "mq135", "mq4",
                                  "humidity", "tempC", "food_type", "label"])
                f.flush()
                print(f"\n✅  Header diterima.\n")
                break
            print(".", end="", flush=True)

        # ── Header kolom di terminal ──────────────────────────
        print(f"  {'Waktu':>10}  {'elapsed':>10}  {'mq2':>5}  {'mq135':>6}  "
              f"{'mq4':>5}  {'hum%':>6}  {'°C':>5}  Status")
        print("  " + "─" * 72)

        # ── Loop logging ──────────────────────────────────────
        try:
            while True:
                raw = ser.readline().decode(errors="ignore").strip()
                if not raw:
                    continue

                parts = raw.split(",")
                if len(parts) != 6:
                    continue

                try:
                    elapsed_ms = int(parts[0]) + offset_ms  # tambah offset sejak masak
                except ValueError:
                    continue

                # Format HH:MM:SS
                total_sec   = elapsed_ms // 1000
                elapsed_hms = (f"{total_sec // 3600:02d}:"
                               f"{(total_sec % 3600) // 60:02d}:"
                               f"{total_sec % 60:02d}")

                label     = get_label(elapsed_ms, food_type)
                elapsed_h = elapsed_ms / 3_600_000
                now_str   = datetime.now().strftime("%H:%M:%S")
                base      = (f"  {now_str:>10}  {elapsed_hms:>10}  {parts[1]:>5}  "
                             f"{parts[2]:>6}  {parts[3]:>5}  {parts[4]:>6}  {parts[5]:>5}")

                if label == "done":
                    print(f"\n🏁  Eksperimen selesai ({elapsed_h:.1f} jam). "
                          f"Logging dihentikan otomatis.")
                    break

                elif label == "buffer":
                    count_skipped += 1
                    writer.writerow([elapsed_hms] + parts[1:] + [food_type, "buffer"])
                    f.flush()
                    print(f"{base}  🟡 buffer")

                else:
                    writer.writerow([elapsed_hms] + parts[1:] + [food_type, label])
                    f.flush()
                    if label == "fresh":
                        count_fresh += 1
                        print(f"{base}  🟢 fresh")
                    else:
                        count_spoiled += 1
                        print(f"{base}  🔴 spoiled")

        except KeyboardInterrupt:
            print("\n\nLogging dihentikan manual.")

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  SUMMARY")
    print("=" * 50)
    print(f"  File         : {filename}")
    print(f"  Makanan      : {food_type}")
    print(f"  Offset awal  : {offset_ms / 60_000:.0f} menit ({offset_ms / 3_600_000:.2f} jam)")
    print(f"  Fresh rows   : {count_fresh}")
    print(f"  Spoiled rows : {count_spoiled}")
    print(f"  Buffer rows  : {count_skipped}")
    print(f"  Total saved  : {count_fresh + count_spoiled + count_skipped}")
    print("=" * 50)


if __name__ == "__main__":
    main()