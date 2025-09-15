from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load dwell time model
dwell_model = joblib.load("rf_model.pkl")

app = FastAPI()

# Feature list
dwell_features = [
    'TrainNo','TrainType', 'HaltStation', 'PFNo', 'BlockNo',
    'BlockLen', 'ApproachingBlockNo',
    'CurrentSpeed', 'CurrentDelay', 'DFNS', 'RunningStatus'
]

class DwellRequest(BaseModel):
    TrainNo: str
    TrainType: str
    HaltStation: str
    PFNo: str               # was string in training
    BlockNo: str
    BlockLen: float
    ApproachingBlockNo: str # was string in training
    CurrentSpeed: float
    CurrentDelay: float
    DFNS: float
    RunningStatus: str      # was string in training


@app.post("/predict_dwell")
def predict_dwell(request: DwellRequest):
    data = pd.DataFrame([request.dict()])
    dwell_sec = dwell_model.predict(data)[0]

    return {
        "Dwell_sec": float(dwell_sec),
        "Dwell_min": round(float(dwell_sec) / 60, 2)
    }
