"""FastAPI backend for the Smart Food Spoilage dashboard (local only).

Endpoints:
  GET  /api/options       -> dropdown data (splits, models, food_types) + per-model metrics
  POST /api/predict       -> classify one reading with the chosen (split, model)
  POST /api/sensor/read   -> read the live sensor for N seconds, return averaged values
  POST /api/evaluate      -> score a (split, model) on uploaded labelled CSV(s)

Run:  uvicorn main:app --port 8000   (from dashboard/backend/)
"""

from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import batch_eval
import inference
import sensor

app = FastAPI(title="Smart Food Spoilage Dashboard API")
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
