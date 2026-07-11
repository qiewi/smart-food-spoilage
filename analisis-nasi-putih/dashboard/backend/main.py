"""FastAPI backend untuk dashboard Nasi Putih (gas-only, lokal saja).

Endpoints:
  GET  /api/options       -> data dropdown (splits, models, food_types=[Nasi Putih]) + metrik
  POST /api/predict       -> klasifikasi satu pembacaan dgn (split, model) terpilih
  POST /api/sensor/read   -> baca sensor live N detik, kembalikan nilai rata-rata
  POST /api/evaluate      -> skor (split, model) pada CSV berlabel yang diunggah

Jalankan:  uvicorn main:app --port 8000   (dari analisis-nasi-putih/dashboard/backend/)
"""

from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import batch_eval
import inference
import sensor

app = FastAPI(title="Smart Food Spoilage Dashboard API — Nasi Putih")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictReq(BaseModel):
    split: str
    model: str
    food_type: str
    mq2: float
    mq135: float
    mq4: float


class SensorReq(BaseModel):
    port: str = "COM4"
    seconds: int = 10


class EvalFile(BaseModel):
    name: str
    content: str


class EvalReq(BaseModel):
    split: str
    model: str
    files: List[EvalFile]


@app.get("/api/options")
def options():
    m = inference.metadata()
    return {"splits": m["splits"], "models": m["models"],
            "food_types": m["food_types"], "metrics": m["metrics"],
            "train_metrics": m.get("train_metrics", {})}


@app.post("/api/predict")
def predict(req: PredictReq):
    try:
        return inference.predict(req.split, req.model, req.food_type,
                                 req.mq2, req.mq135, req.mq4)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/sensor/read")
def sensor_read(req: SensorReq):
    try:
        return sensor.read_average(req.port, seconds=req.seconds)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/api/evaluate")
def evaluate(req: EvalReq):
    try:
        files = [{"name": f.name, "content": f.content} for f in req.files]
        return batch_eval.evaluate(req.split, req.model, files)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(e))
