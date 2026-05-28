import serial
import csv
from datetime import datetime

PORT = "COM3"
BAUD = 115200

filename = f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

ser = serial.Serial(PORT, BAUD, timeout=2)

with open(filename, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    # Tunggu header dari ESP32
    while True:
        line = ser.readline().decode(errors="ignore").strip()
        if line == "ms,mq2,mq135,mq4,humidity,tempC":
            writer.writerow(line.split(","))
            f.flush()
            print("Header detected:", line)
            break

    print("Logging started...")
    print("Press CTRL+C to stop.\n")

    while True:
        line = ser.readline().decode(errors="ignore").strip()
        if not line:
            continue

        parts = line.split(",")
        if len(parts) == 6:
            writer.writerow(parts)
            f.flush()
            print(line)