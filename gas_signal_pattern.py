"""Gas-sensor signal pattern vs storage duration — reference figure for the document.

Plots the three MQ sensors' smoothed readings (rolling mean = general trend, no noise/
spikes) over storage time for one Nasi Putih training trial. The leading warm-up region
(where the freshly-heated sensors settle and the signal declines) is trimmed at the
signal trough. Pure sensor data — no fresh/spoiled labelling.

Output: results/figures/gas_signal_<trial>.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src.load_data import elapsed_to_seconds

CSV = config.PROJECT_ROOT / "training" / "dataset_Nasi_Putih_20260413_125941_relabeled.csv"
SMOOTH_WINDOW = 120        # rolling-mean window (points) -> tren umum, buang derau/anomali
SENSORS = [("mq2", "MQ2", "#10b981"), ("mq135", "MQ135", "#0ea5e9"), ("mq4", "MQ4", "#f59e0b")]

df = pd.read_csv(CSV)
df["hours"] = df["elapsed"].map(elapsed_to_seconds) / 3600
# the raw log is not time-ordered (logger restarts) -> sort so lines read left-to-right
df = df.sort_values("hours").reset_index(drop=True)
# tren umum: rolling mean membuang derau + spike anomali
for col, _, _ in SENSORS:
    df[col] = df[col].rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean()

# buang region warm-up di awal (sinyal menurun saat sensor stabil) -> mulai dari titik terendah
norm = sum((df[c] - df[c].min()) / (df[c].max() - df[c].min()) for c, _, _ in SENSORS) / 3
cut = df["hours"].iloc[norm.idxmin()]
df = df[df["hours"] >= cut].reset_index(drop=True)
print(f"{CSV.name}: warm-up dibuang < {cut:.1f} jam -> plot {df['hours'].min():.1f}-{df['hours'].max():.1f} jam")

fig, ax = plt.subplots(figsize=(10, 5))
for col, name, color in SENSORS:
    ax.plot(df["hours"], df[col], lw=2.2, color=color, label=name)
    ax.annotate(name, xy=(df["hours"].iloc[-1], df[col].iloc[-1]),
                xytext=(4, 0), textcoords="offset points",
                fontsize=9, fontweight="bold", color=color, va="center")

ymax = df[[s for s, _, _ in SENSORS]].max().max()
ax.set_xlabel("Durasi penyimpanan (jam)")
ax.set_ylabel("Nilai sensor (ADC)")
ax.set_xlim(df["hours"].min(), df["hours"].max() * 1.05)
ax.set_ylim(0, ymax * 1.10)
ax.grid(True, ls=":", lw=0.6, color="#ddd")
ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
ax.set_title("Pola Sinyal Sensor Gas terhadap Durasi Penyimpanan — Nasi Putih",
             fontsize=12, fontweight="bold", pad=12)

fig.tight_layout()
out = "gas_signal_" + CSV.stem.replace("dataset_", "").replace("_relabeled", "") + ".png"
fig.savefig(config.FIGURES_DIR / out, dpi=150)
print(f"Figure: results/figures/{out}")
