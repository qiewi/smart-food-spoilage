"""Baca sensor live via serial selama N detik dan kembalikan pembacaan rata-rata.

Mengikuti format dari src/pylogger.py: baris koma `ms,mq2,mq135,mq4,humidity,tempC` (6 field).
Baris header & baris rusak dilewati. Rata-rata mq2/mq135/mq4 dipakai sebagai input model
(humidity/tempC tetap dikembalikan untuk info, tapi tidak dipakai model gas-only).
"""

import time

import serial

BAUD = 115200


def list_ports():
    from serial.tools import list_ports as lp
    return [p.device for p in lp.comports()]


def read_average(port="COM4", baud=BAUD, seconds=10):
    try:
        ser = serial.Serial(port, baud, timeout=2)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Tidak bisa membuka {port}: {e}. Port tersedia: {list_ports()}")

    sums = {"mq2": 0.0, "mq135": 0.0, "mq4": 0.0, "humidity": 0.0, "tempC": 0.0}
    n = 0
    t0 = time.time()
    try:
        while time.time() - t0 < seconds:
            raw = ser.readline().decode(errors="ignore").strip()
            if not raw:
                continue
            parts = raw.split(",")
            if len(parts) != 6:
                continue
            try:                                   # parts[0]=ms ignored; rest are sensors
                _, mq2, mq135, mq4, hum, temp = (float(p) for p in parts)
            except ValueError:
                continue                           # header ("ms,mq2,...") or junk
            sums["mq2"] += mq2
            sums["mq135"] += mq135
            sums["mq4"] += mq4
            sums["humidity"] += hum
            sums["tempC"] += temp
            n += 1
    finally:
        ser.close()

    if n == 0:
        raise RuntimeError(f"Tidak ada data valid dari {port} dalam {seconds} detik.")
    return {"samples": n, "seconds": seconds, **{k: v / n for k, v in sums.items()}}
