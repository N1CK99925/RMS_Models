# 🚆 RMS_Models

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![ML](https://img.shields.io/badge/Machine%20Learning-Dwell%20Time%20%7C%20ETA-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

---

## 📌 Overview
The **RMS_Models** repository provides machine learning models and services for **railway operations analytics**, focusing on:

- ⏱️ **Dwell Time Prediction** – estimating how long a train spends at a station.  
- 🕒 **ETA/ETD Modeling (ETADS)** – predicting expected arrival and departure times.  
- 🧩 **Ensemble Learning** – combining multiple models to improve accuracy.  
- 🔄 **Data preprocessing pipelines** – cleaning and preparing raw railway datasets.  

This project is built for both **research experiments** (Jupyter notebooks) and **production deployment** (Python services + model artifacts).

---

## ✨ Features
- Machine learning models (Random Forest, Ensemble) for dwell time prediction.  
- ETA/ETD model for real-time train arrival/departure estimation.  
- REST API services (`Flask` or `FastAPI`) for model deployment.  
- Modular code structure for easy experimentation.  
- Preprocessed datasets (`Live_Trains_Fixed_Clean.csv`) for reliable training.  
- Training logs for reproducibility.  

---

## 📂 Repository Structure
RMS_models/
│── .gitattributes
│── README.md
│── pycache/
│
│── models/
│ ├── ensemble_dwell_time_model.pkl
│ ├── ensemble_dwell_time_model(Self).pkl
│ ├── rf_model.pkl
│ ├── ETADS.pkl
│
│── datasets/
│ ├── Block_Relation.csv
│ ├── Blocks.csv
│ ├── Stations.csv
│ ├── Trains.csv
│ ├── Live_Trains.csv
│ ├── Live_Trains_Fixed_Clean.csv
│
│── notebooks/
│ ├── DwellTime(2).ipynb
│ ├── DwellTime(Draft).ipynb
│ ├── ETA.ipynb
│
│── scripts/
│ ├── dwellTime.py
│ ├── dwell_time_ml_service.py
│ ├── ml_service.py
│
│── logs/
│ ├── train_model.log

yaml
Copy code

---

## ⚙️ Models

### 🔹 Dwell Time Model
- **Goal**: Predict how long a train will stay at a station.  
- **Techniques**: Random Forest + Ensemble.  
- **Inputs**: Train ID, Station, Time, Block data.  
- **Output**: Predicted dwell time (in minutes).  

Artifacts:
- `rf_model.pkl` → Random Forest model.  
- `ensemble_dwell_time_model.pkl` → Final ensemble model.  

---

### 🔹 ETADS Model
- **Goal**: Predict Expected Time of Arrival (ETA) & Departure (ETD).  
- **Inputs**: Station sequences, block relations, live train status.  
- **Output**: ETA and ETD predictions.  

Artifact:
- `ETADS.pkl`  

---

## 📊 Datasets
- `Stations.csv` → Station metadata.  
- `Blocks.csv`, `Block_Relation.csv` → Railway block definitions and connectivity.  
- `Trains.csv` → Train metadata.  
- `Live_Trains.csv` → Raw live data.  
- `Live_Trains_Fixed_Clean.csv` → Preprocessed dataset used in modeling.  

---

## 🚀 Installation & Setup

### 1️⃣ Clone Repository
```bash
git clone https://github.com/<your-username>/RMS_models.git
cd RMS_models
2️⃣ Install Requirements
bash
Copy code
pip install -r requirements.txt
3️⃣ Train Models
bash
Copy code
python scripts/dwellTime.py
4️⃣ Run Dwell Time API Service
bash
Copy code
python scripts/dwell_time_ml_service.py
🔧 API Usage
📌 Endpoint: /predict_dwell_time
Request (JSON):

json
Copy code
{
  "train_id": 101,
  "station_id": "NDLS",
  "arrival_time": "2025-09-25T10:30:00"
}
Response (JSON):

json
Copy code
{
  "predicted_dwell_time": 7.5
}
Example with curl
bash
Copy code
curl -X POST http://127.0.0.1:5000/predict_dwell_time \
     -H "Content-Type: application/json" \
     -d '{"train_id":101,"station_id":"NDLS","arrival_time":"2025-09-25T10:30:00"}'
📈 Experiments
DwellTime(Draft).ipynb → Feature engineering + baseline experiments.

DwellTime(2).ipynb → Final tuned dwell time model.

ETA.ipynb → ETADS prototype (arrival/departure estimation).

📝 Logs
train_model.log contains:

Training progress.

Hyperparameters used.

Evaluation metrics (MAE, RMSE, R²).
