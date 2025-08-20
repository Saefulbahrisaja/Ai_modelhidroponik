
from pydantic import BaseModel
import pandas as pd
from pipeline_realtime import train_and_save_models, train_from_mysql, RealtimePredictor
from fastapi import FastAPI, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import traceback

app = FastAPI()

# --- FIX: tambahkan middleware sebelum route ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8080", "http://localhost:8080"],          # izinkan semua origin
    allow_methods=["*"],          # izinkan semua method (GET, POST, OPTIONS, dll)
    allow_headers=["*"],          # izinkan semua headers
)

# ====== TRAIN ULANG MODEL DARI FILE ======
@app.post("/train")
async def train_model(file: UploadFile):
    df = pd.read_csv(file.file)
    result = train_and_save_models(df)
    return result

# ====== TRAIN ULANG MODEL DARI MYSQL (JSON body) ======
class MySQLConfig(BaseModel):
    user: str
    password: str
    host: str = "srv1982.hstgr.io"
    port: int = 3306
    db: str = "u797687691_hijau"
    table: str = "sensor_data"

@app.post("/train-mysql")
async def train_mysql(config: MySQLConfig):
    result = train_from_mysql(
        user=config.user,
        password=config.password,
        host=config.host,
        port=config.port,
        db=config.db,
        table=config.table
    )
    return result

# ====== PREDIKSI REALTIME ======
@app.post("/predict")
async def predict(data: dict = Body(...)):
    try:
        predictor = RealtimePredictor()

        # Ambil semua feature yang dipakai model
        features = predictor.feature_cols

        # Isi default 0 dan cast numeric
        safe_data = {}
        for col in features:
            val = data.get(col, 0)
            try:
                safe_data[col] = float(val)
            except (TypeError, ValueError):
                safe_data[col] = 0.0

        # Debug: print data yang dipakai prediksi
        print("DEBUG /predict safe_data:", safe_data)

        # Prediksi
        result = predictor.predict_next(**safe_data)

        return JSONResponse(content=result)

    except Exception as e:
        # Print full stacktrace ke console uvicorn
        traceback.print_exc()
        # Return JSON error ke client
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=200  # tetap 200 supaya client JS bisa baca JSON
        )