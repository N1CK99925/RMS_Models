from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import datetime 
import timedelta
# Load model
model = joblib.load("ETADS.pkl")

app = FastAPI()

features = [
    'CurrentDelay', 'CurrentSpeed', 'DFNS', 'DwellTime', 
    'BlockLen', 'IB', 'RunningStatus', 'TrainType'
]

class ETADSRequest(BaseModel):
      # numeric
    CurrentDelay: float
    CurrentSpeed: float
    DFNS: float
    DwellTime: float
    BlockLen: float

    # categorical (use int encoding or string)
    TrainType: str

    # boolean
    IB: bool
    RunningStatus: bool

   

@app.post("/predict")
def predict(request: ETADSRequest):
    # Convert request to DataFrame
    data = pd.DataFrame([request.dict()])

    # Predict in seconds
    etads_sec = model.predict(data)[0]

    # Convert back to minutes
    etads_min = etads_sec / 60
     # Convert to datetime relative to scheduled arrival
    # Suppose you have ScheduledArrival as a field in request (string)
    # scheduled_arrival = datetime.strptime(request.ScheduledArrival, "%Y-%m-%d %H:%M:%S")
    # predicted_arrival = scheduled_arrival + timedelta(seconds=etads_sec)

    return {
        "ETADS_sec": float(etads_sec),
        "ETADS_min": round(float(etads_min), 2),  # optional rounding
        "ETADS_DateTIme": pd.to_datetime(etads_sec)
    }
