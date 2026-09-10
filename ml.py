import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas
from joblib import dump, load
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


cols = [
    "Time_stamp",
    "Location",
    "latitude",
    "longitude",
    "temperature",
    "Humidity_p",
    "Rain_mm",
    "wind_speed",
    "pressure",
    "Cloud_cover",
    "precipitation",
]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_historic_dataframe():
    preferred_path = os.path.join(BASE_DIR, "kolkata_west_bengal_weather_historic_template.csv")
    legacy_path = "/home/jam/Desktop/SIH/bros-ciphers/kolkata_west_bengal_weather_historic_template.csv"
    data_path = preferred_path if os.path.exists(preferred_path) else (
        legacy_path if os.path.exists(legacy_path) else preferred_path
    )
    return pandas.read_csv(data_path, names=cols)


def build_live_weather_payload(weather_input):
    return {
        "Humidity_p": float(weather_input.get("humidity_percent", weather_input.get("Humidity_p", 0.0)) or 0.0),
        "Rain_mm": float(weather_input.get("rain_mm", weather_input.get("Rain_mm", 0.0)) or 0.0),
        "wind_speed": float(weather_input.get("wind_speed_kmh", weather_input.get("wind_speed", 0.0)) or 0.0),
        "pressure": float(weather_input.get("pressure_hpa", weather_input.get("pressure", 0.0)) or 0.0),
        "Cloud_cover": float(weather_input.get("cloud_cover_percent", weather_input.get("Cloud_cover", 0.0)) or 0.0),
        "precipitation": float(weather_input.get("precipitation_mm", weather_input.get("precipitation", 0.0)) or 0.0),
    }


def get_historic_mean_std():
    df = load_historic_dataframe()
    return {
        "mean_humidity": float(df["Humidity_p"].mean()),
        "mean_rain": float(df["Rain_mm"].mean()),
        "mean_wind_speed": float(df["wind_speed"].mean()),
        "mean_pressure": float(df["pressure"].mean()),
        "mean_cloud": float(df["Cloud_cover"].mean()),
        "mean_precip": float(df["precipitation"].mean()),
        "std_humidity": float(df["Humidity_p"].std()),
        "std_rain": float(df["Rain_mm"].std()),
        "std_wind_speed": float(df["wind_speed"].std()),
        "std_pressure": float(df["pressure"].std()),
        "std_cloud": float(df["Cloud_cover"].std()),
        "std_precip": float(df["precipitation"].std()),
    }


def prediction(value, mean, sd):
    if sd == 0:
        return 0.0
    return (value - mean) / sd


def train_and_save_models(force_train=False):
    data = load_historic_dataframe()
    numeric_cols = ["temperature", "Humidity_p", "Rain_mm", "wind_speed", "pressure", "Cloud_cover", "precipitation"]
    df_ml = data.copy()
    for column in numeric_cols:
        df_ml[column] = pandas.to_numeric(df_ml[column], errors="coerce")
    df_ml = df_ml.dropna(subset=numeric_cols)

    features = ["Humidity_p", "Rain_mm", "wind_speed", "pressure", "Cloud_cover", "precipitation"]
    target = "temperature"
    X = df_ml[features].values
    y = df_ml[target].values

    model_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path_temp = os.path.join(model_dir, "temperature_model.joblib")
    anom_model_path = os.path.join(model_dir, "anomaly_model.joblib")
    scaler_path = os.path.join(model_dir, "scaler_X.joblib")

    if os.path.exists(model_path_temp) and os.path.exists(anom_model_path) and os.path.exists(scaler_path) and not force_train:
        return load(model_path_temp), load(scaler_path), load(anom_model_path)

    if len(X) <= 10:
        raise ValueError("Not enough data to train the model.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"[ML PIPELINE] Temperature model MSE: {mse:.3f}, R2: {r2:.3f}")

    dump(model, model_path_temp)
    dump(scaler_x, scaler_path)

    feats = df_ml[features].values
    feats_scaled = scaler_x.transform(feats)
    anom_clf = IsolationForest(contamination="auto", random_state=42)
    anom_clf.fit(feats_scaled)
    anomaly_scores = anom_clf.decision_function(feats_scaled)
    print(f"[ANOMALY PIPELINE] Learned anomaly score range: min={anomaly_scores.min():.4f}, max={anomaly_scores.max():.4f}")
    dump(anom_clf, anom_model_path)
    return model, scaler_x, anom_clf


def predict_from_map_weather(weather_input, model=None, scaler=None, anomaly_model=None):
    if model is None or scaler is None or anomaly_model is None:
        model = load(os.path.join(BASE_DIR, "models", "temperature_model.joblib"))
        scaler = load(os.path.join(BASE_DIR, "models", "scaler_X.joblib"))
        anomaly_model = load(os.path.join(BASE_DIR, "models", "anomaly_model.joblib"))

    live_weather = build_live_weather_payload(weather_input)
    feature_array = np.array(
        [[
            live_weather["Humidity_p"],
            live_weather["Rain_mm"],
            live_weather["wind_speed"],
            live_weather["pressure"],
            live_weather["Cloud_cover"],
            live_weather["precipitation"],
        ]],
        dtype=float,
    )

    scaled_input = scaler.transform(feature_array)
    temperature_prediction = float(model.predict(scaled_input)[0])
    anomaly_pred_raw = anomaly_model.predict(scaled_input)[0]
    anomaly_prediction = 1 if anomaly_pred_raw == -1 else 0

    stats = get_historic_mean_std()
    z_inputs = [
        prediction(live_weather["Humidity_p"], stats["mean_humidity"], stats["std_humidity"]),
        prediction(live_weather["Rain_mm"], stats["mean_rain"], stats["std_rain"]),
        prediction(live_weather["wind_speed"], stats["mean_wind_speed"], stats["std_wind_speed"]),
        prediction(live_weather["pressure"], stats["mean_pressure"], stats["std_pressure"]),
        prediction(live_weather["Cloud_cover"], stats["mean_cloud"], stats["std_cloud"]),
        prediction(live_weather["precipitation"], stats["mean_precip"], stats["std_precip"]),
    ]
    z_score = float(np.mean(np.abs(np.array(z_inputs, dtype=float))))

    return {
        "new_data": live_weather,
        "temperature_prediction": temperature_prediction,
        "anomaly_prediction": anomaly_prediction,
        "z_score": z_score,
    }


def run_cli():
    parser = argparse.ArgumentParser(description="Run ml.py (default) or predict using saved model")
    parser.add_argument("--predict", nargs=6, type=float, metavar=("Humidity_p", "Rain_mm", "wind_speed", "pressure", "Cloud_cover", "precipitation"), help="Predict temperature from six features")
    parser.add_argument("--predict-anom", nargs=6, type=float, metavar=("Humidity_p", "Rain_mm", "wind_speed", "pressure", "Cloud_cover", "precipitation"), help="Predict anomaly from six features")
    parser.add_argument("--force-train", action="store_true", help="Force retraining even if saved models exist")
    args = parser.parse_args()

    data = load_historic_dataframe()
    if args.predict:
        try:
            scaler_x = load(os.path.join(BASE_DIR, "models", "scaler_X.joblib"))
            model = load(os.path.join(BASE_DIR, "models", "temperature_model.joblib"))
        except FileNotFoundError:
            print("Saved model not found. Train the model first by running without --predict.")
            raise SystemExit(1)

        feat = np.array(args.predict).reshape(1, -1)
        pred = model.predict(scaler_x.transform(feat))[0]
        print(f"Predicted temperature: {pred:.3f}")
        raise SystemExit(0)

    if args.predict_anom:
        try:
            scaler_x = load(os.path.join(BASE_DIR, "models", "scaler_X.joblib"))
            anom_model = load(os.path.join(BASE_DIR, "models", "anomaly_model.joblib"))
        except FileNotFoundError:
            print("Saved model not found. Train the model first by running without --predict-anom.")
            raise SystemExit(1)

        feat = np.array(args.predict_anom).reshape(1, -1)
        pred_raw = anom_model.predict(scaler_x.transform(feat))[0]
        pred = 1 if pred_raw == -1 else 0
        print(f"Predicted anomaly: {pred} ({'anomaly' if pred == 1 else 'normal'})")
        raise SystemExit(0)

    print(data.head())

    temp = data["temperature"]
    humidity = data["Humidity_p"]
    rain = data["Rain_mm"]
    wind_speed = data["wind_speed"]
    pressure = data["pressure"]
    cloud = data["Cloud_cover"]
    precip = data["precipitation"]

    plt.scatter(temp, pressure, color="hotpink")
    plt.title("temperature VS pressure")
    plt.xlabel("temperature")
    plt.ylabel("pressure")
    plt.show()

    plt.scatter(temp, humidity, color="orange")
    plt.title("temperature VS humidity")
    plt.xlabel("temperature")
    plt.ylabel("humidity")
    plt.show()

    plt.scatter(pressure, wind_speed, color="green")
    plt.title("pressure VS windspeed")
    plt.xlabel("pressure")
    plt.ylabel("windspeed")
    plt.show()

    plt.scatter(cloud, precip, color="blue")
    plt.title("cloud VS precipitation")
    plt.xlabel("cloud")
    plt.ylabel("precipitation")
    plt.show()

    plt.scatter(humidity, precip, color="#8E27F5")
    plt.title("humidity VS precipitation")
    plt.xlabel("humidity")
    plt.ylabel("precipitation")
    plt.show()

    train_and_save_models(force_train=args.force_train)


if __name__ == "__main__":
    run_cli()
