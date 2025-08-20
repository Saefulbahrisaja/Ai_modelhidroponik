from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pipeline_realtime import train_and_save_models, train_from_mysql, RealtimePredictor
app = FastAPI()

# --- FIX: tambahkan middleware sebelum route ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # izinkan semua origin
    allow_methods=["*"],          # izinkan semua method (GET, POST, OPTIONS, dll)
    allow_headers=["*"],          # izinkan semua headers
)

@app.get("/")
async def root():
    return {"message": "FastAPI dengan CORS aktif"}

@app.post("/predict")
async def predict(data: dict):
    predictor = RealtimePredictor()
    result = predictor.predict_next(**data)
    return result