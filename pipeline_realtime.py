import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

MODEL_SUHU = "model_suhu.pkl"
MODEL_KELEMBABAN = "model_kelembaban.pkl"
FEATURE_COLS = "feature_cols.pkl"

# ================= TRAINING ==================
def train_and_save_models(realtime_df: pd.DataFrame):
    df = realtime_df.copy()
    df['created_at'] = pd.to_datetime(df['created_at'])
    df = df.sort_values('created_at')

    # Bersihkan data
    df[['suhu','kelembaban','suhuudara','ph','tds','level_air','air_min']] = (
        df[['suhu','kelembaban','suhuudara','ph','tds','level_air','air_min']]
        .replace({-127.0: np.nan, 0.0: np.nan})
        .ffill().bfill()
    )

    features = ["suhuudara", "ph", "tds", "pompa_air", "pompa_nutrisi", "level_air", "air_min"]
    X = df[features]
    y_suhu = df["suhu"]
    y_kelembaban = df["kelembaban"]

    # Train model suhu
    X_train, X_test, y_train, y_test = train_test_split(X, y_suhu, test_size=0.2, random_state=42)
    model_suhu = RandomForestRegressor(n_estimators=100, random_state=42)
    model_suhu.fit(X_train, y_train)

    # Train model kelembaban
    X_train, X_test, y_train, y_test = train_test_split(X, y_kelembaban, test_size=0.2, random_state=42)
    model_kelembaban = RandomForestRegressor(n_estimators=100, random_state=42)
    model_kelembaban.fit(X_train, y_train)

    # Simpan model
    joblib.dump(model_suhu, MODEL_SUHU)
    joblib.dump(model_kelembaban, MODEL_KELEMBABAN)
    joblib.dump(features, FEATURE_COLS)

    return {"status": "trained", "features": features, "n_data": len(df)}


def train_from_mysql(user, password, host, port, db, table="sensor_data"):
    """
    Latih ulang model dengan data dari MySQL remote.
    """
    from sqlalchemy import create_engine
    import urllib.parse
    import traceback

    try:
        # Encode password biar @ dan + tidak rusak
        safe_password = urllib.parse.quote_plus(password)

        conn_str = f"mysql+pymysql://{user}:{safe_password}@{host}:{port}/{db}"
        print("🔗 Connecting to:", conn_str)  # Debug
        engine = create_engine(conn_str)

        query = f"SELECT * FROM {table}"
        df = pd.read_sql(query, engine)

        return train_and_save_models(df)
    except Exception as e:
        print("🔥 ERROR MYSQL:", e)
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

# ================= PREDIKTOR ==================
class RealtimePredictor:
    def __init__(self):
        self.model_suhu = joblib.load(MODEL_SUHU)
        self.model_kelembaban = joblib.load(MODEL_KELEMBABAN)
        self.feature_cols = joblib.load(FEATURE_COLS)

    def predict_next(self, **kwargs):
        # Ambil fitur input
        x = np.array([kwargs.get(col, 0) for col in self.feature_cols]).reshape(1, -1)

        # Prediksi suhu & kelembaban
        suhu_pred = self.model_suhu.predict(x)[0]
        kelembaban_pred = self.model_kelembaban.predict(x)[0]

        # Deteksi anomali sederhana (contoh rule based)
        anomaly = 0
        if suhu_pred < 10 or suhu_pred > 40 or kelembaban_pred < 30 or kelembaban_pred > 90:
            anomaly = 1

        # Estimasi penggunaan air & nutrisi per hari (dummy rule, bisa diganti)
        est_use_air_per_day_units = round(max(0, kelembaban_pred / 6), 2)
        est_use_nutrisi_per_day_L = round(max(0, suhu_pred / 40 * 1.0), 2)

        # Estimasi panen (misalnya fix 18 hari ke depan)
        est_days_to_harvest = 18
        est_harvest_date = (datetime.today() + timedelta(days=est_days_to_harvest)).strftime("%Y-%m-%d")

        return {
            "pred_suhu": float(round(suhu_pred, 2)),
            "pred_kelembaban": float(round(kelembaban_pred, 2)),
            "anomaly": anomaly,
            "est_use_air_per_day_units": est_use_air_per_day_units,
            "est_use_nutrisi_per_day_L": est_use_nutrisi_per_day_L,
            "est_days_to_harvest": est_days_to_harvest,
            "est_harvest_date": est_harvest_date,
            "input": kwargs
        }