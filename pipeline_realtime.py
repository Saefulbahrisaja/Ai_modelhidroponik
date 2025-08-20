import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta

MODEL_SUHU = "model_suhu.pkl"
MODEL_KELEMBABAN = "model_kelembaban.pkl"
FEATURE_COLS = "feature_cols.pkl"

# ================= TRAINING DENGAN VALIDASI & PLOT ==================
def train_and_save_models(realtime_df: pd.DataFrame, plot=True):
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

    results = {}

    # ===== Model Suhu =====
    X_train, X_test, y_train, y_test = train_test_split(X, y_suhu, test_size=0.2, random_state=42)
    model_suhu = RandomForestRegressor(n_estimators=100, random_state=42)
    model_suhu.fit(X_train, y_train)
    y_pred_suhu = model_suhu.predict(X_test)
    mse_suhu = mean_squared_error(y_test, y_pred_suhu)
    rmse_suhu = np.sqrt(mse_suhu)
    r2_suhu = r2_score(y_test, y_pred_suhu)
    results['suhu'] = {"MSE": mse_suhu, "RMSE": rmse_suhu, "R2": r2_suhu}

    # ===== Model Kelembaban =====
    X_train, X_test, y_train, y_test = train_test_split(X, y_kelembaban, test_size=0.2, random_state=42)
    model_kelembaban = RandomForestRegressor(n_estimators=100, random_state=42)
    model_kelembaban.fit(X_train, y_train)
    y_pred_kelembaban = model_kelembaban.predict(X_test)
    mse_kelembaban = mean_squared_error(y_test, y_pred_kelembaban)
    rmse_kelembaban = np.sqrt(mse_kelembaban)
    r2_kelembaban = r2_score(y_test, y_pred_kelembaban)
    results['kelembaban'] = {"MSE": mse_kelembaban, "RMSE": rmse_kelembaban, "R2": r2_kelembaban}

    # Simpan model & fitur
    joblib.dump(model_suhu, MODEL_SUHU)
    joblib.dump(model_kelembaban, MODEL_KELEMBABAN)
    joblib.dump(features, FEATURE_COLS)

    results.update({
        "status": "trained",
        "features": features,
        "n_data": len(df)
    })

    # ===== Plot Prediksi vs Aktual =====
    if plot:
        plt.figure(figsize=(12, 5))

        # Suhu
        plt.subplot(1, 2, 1)
        plt.scatter(y_test, y_pred_suhu, alpha=0.6, color='blue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel("Suhu Aktual")
        plt.ylabel("Suhu Prediksi")
        plt.title(f"Suhu: R2={r2_suhu:.2f}, RMSE={rmse_suhu:.2f}")

        # Kelembaban
        plt.subplot(1, 2, 2)
        plt.scatter(y_test, y_pred_kelembaban, alpha=0.6, color='green')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel("Kelembaban Aktual")
        plt.ylabel("Kelembaban Prediksi")
        plt.title(f"Kelembaban: R2={r2_kelembaban:.2f}, RMSE={rmse_kelembaban:.2f}")

        plt.tight_layout()
        plt.show()

    return results

# ===== Contoh penggunaan =====
if __name__ == "__main__":
    df = pd.read_csv("data_sensor.csv")
    metrics = train_and_save_models(df, plot=True)
    print("✅ Training selesai, validasi:")
    print(metrics)
