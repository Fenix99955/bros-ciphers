import os
from typing import Dict

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from joblib import load
from pydantic import BaseModel, Field

from ml import predict_from_map_weather


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
TEMPERATURE_MODEL_PATH = os.path.join(MODEL_DIR, "temperature_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_X.joblib")
ANOMALY_MODEL_PATH = os.path.join(MODEL_DIR, "anomaly_model.joblib")

app = FastAPI(
    title="MOSDAC Weather & Anomaly Predictor",
    description="A map-driven weather API that resolves city names, fetches current weather, and predicts temperature/anomaly values for the pinned location.",
)


class PinPayload(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0, example=22.5726)
    longitude: float = Field(..., ge=-180.0, le=180.0, example=88.3639)


class CityWeatherResult(BaseModel):
    status: str
    latitude: float
    longitude: float
    city_name: str
    temperature_c: float | None = None
    humidity_percent: float | None = None
    rain_mm: float | None = None
    wind_speed_kmh: float | None = None
    pressure_hpa: float | None = None
    cloud_cover_percent: float | None = None
    precipitation_mm: float | None = None
    new_data: Dict[str, float]
    temperature_prediction: float | None = None
    anomaly_prediction: int | None = None
    anomaly_label: str | None = None
    anomaly_reason: str | None = None
    operation_message: str
    user_function_output: str


def _load_ml_models():
    required = [TEMPERATURE_MODEL_PATH, SCALER_PATH, ANOMALY_MODEL_PATH]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=(
                "Model artifacts are missing. Train the ML model first using the script in "
                "this project before calling the API."
            ),
        )

    model = load(TEMPERATURE_MODEL_PATH)
    scaler = load(SCALER_PATH)
    anomaly_model = load(ANOMALY_MODEL_PATH)
    return model, scaler, anomaly_model


def _resolve_city_name(lat: float, lon: float) -> str:
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {"lat": lat, "lon": lon, "format": "json"}
    headers = {
        "User-Agent": "MosdacWeatherPredictor/1.0 (Educational)",
        "Referer": "http://localhost",
    }

    response = requests.get(url, params=params, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()
    address = data.get("address", {})

    city_name = (
        address.get("city")
        or address.get("town")
        or address.get("village")
        or address.get("municipality")
        or address.get("county")
        or data.get("display_name")
        or "Unknown Location"
    )
    return city_name


def _fetch_weather(lat: float, lon: float) -> Dict[str, float | None]:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,precipitation,rain,pressure_msl,surface_pressure,cloud_cover,wind_speed_10m",
        "timezone": "auto",
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    payload = response.json()
    current = payload.get("current", {})

    weather_data = {
        "temperature_c": current.get("temperature_2m"),
        "humidity_percent": current.get("relative_humidity_2m"),
        "rain_mm": current.get("rain"),
        "wind_speed_kmh": current.get("wind_speed_10m"),
        "pressure_hpa": current.get("pressure_msl") or current.get("surface_pressure"),
        "cloud_cover_percent": current.get("cloud_cover"),
        "precipitation_mm": current.get("precipitation"),
    }
    return weather_data


def _build_feature_vector(weather: Dict[str, float | None]) -> np.ndarray:
    feature_values = [
        float(weather.get("humidity_percent") or 0.0),
        float(weather.get("rain_mm") or 0.0),
        float(weather.get("wind_speed_kmh") or 0.0),
        float(weather.get("pressure_hpa") or 0.0),
        float(weather.get("cloud_cover_percent") or 0.0),
        float(weather.get("precipitation_mm") or 0.0),
    ]
    return np.array([feature_values], dtype=float)


def _find_major_anomaly_metric(weather_data: Dict[str, float | None]):
    csv_candidates = [
        os.path.join(BASE_DIR, "kolkata_west_bengal_weather_historic_template.csv"),
        "/home/jam/Desktop/SIH/bros-ciphers/kolkata_west_bengal_weather_historic_template.csv",
    ]
    csv_path = next((p for p in csv_candidates if os.path.exists(p)), None)
    if csv_path is None:
        return None

    df = pd.read_csv(csv_path, names=[
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
    ])
    feature_map = {
        "temperature_c": "temperature",
        "humidity_percent": "Humidity_p",
        "rain_mm": "Rain_mm",
        "wind_speed_kmh": "wind_speed",
        "pressure_hpa": "pressure",
        "cloud_cover_percent": "Cloud_cover",
        "precipitation_mm": "precipitation",
    }
    display_names = {
        "temperature_c": "Temperature Anomaly",
        "humidity_percent": "Humidity Anomaly",
        "rain_mm": "Rain Anomaly",
        "wind_speed_kmh": "Wind Speed Anomaly",
        "pressure_hpa": "Pressure Anomaly",
        "cloud_cover_percent": "Cloud Cover Anomaly",
        "precipitation_mm": "Precipitation Anomaly",
    }

    best_metric = None
    best_score = -1.0
    for weather_key, column_name in feature_map.items():
        if column_name not in df.columns:
            continue
        value = float(weather_data.get(weather_key) or 0.0)
        series = pd.to_numeric(df[column_name], errors='coerce').dropna()
        if series.empty:
            continue
        median = float(series.median())
        if median == 0:
            continue
        pct_diff = abs(value - median) / abs(median) * 100.0
        if pct_diff > best_score:
            best_score = pct_diff
            best_metric = (weather_key, median, value, pct_diff)

    if best_metric is None:
        return None

    weather_key, median, value, pct_diff = best_metric
    direction = "higher than usual" if value > median else "lower than usual"
    label = display_names.get(weather_key, weather_key)
    return label, direction, value, median, pct_diff


def _build_anomaly_reason(weather_data: Dict[str, float | None], anomaly_prediction: int) -> str:
    if anomaly_prediction != 1:
        return "Weather values are within the normal range based on the historical data."

    metric = _find_major_anomaly_metric(weather_data)
    if metric is None:
        return "This location shows a weather anomaly because the live conditions are far from the learned historical pattern."

    label, direction, value, median, pct_diff = metric
    return f"{label}: the current value is {direction} by about {pct_diff:.1f}% ({value:.1f} vs typical {median:.1f})."


def _build_anomaly_label(weather_data: Dict[str, float | None], anomaly_prediction: int) -> str:
    if anomaly_prediction != 1:
        return "Weather Normal"

    metric = _find_major_anomaly_metric(weather_data)
    if metric is None:
        return "Weather Anomaly"

    label, _, _, _, _ = metric
    return label


def process_mosdac_data_for_city(city_val: str) -> str:
    print(f"[Operation] Starting data analysis workflow for: {city_val}")
    return f"Successfully initialized downstream calculations for {city_val} region."


def user_defined_function_city(city_val: str) -> str:
    print(f"[UserFn] Received city value: {city_val}")
    return "scanning-complete"


def user_defined_function_coords(lat: float, lon: float) -> str:
    print(f"[UserFn] Received coords: {lat}, {lon}")
    return "helloworld"


@app.post("/api/pin-location", response_model=CityWeatherResult)
def process_pinned_location(data: PinPayload):
    try:
        city_name = _resolve_city_name(data.latitude, data.longitude)
        weather_data = _fetch_weather(data.latitude, data.longitude)

        model, scaler, anomaly_model = _load_ml_models()
        prediction_result = predict_from_map_weather(weather_data, model=model, scaler=scaler, anomaly_model=anomaly_model)
        temperature_prediction = prediction_result["temperature_prediction"]
        anomaly_prediction = prediction_result["anomaly_prediction"]
        new_data = prediction_result["new_data"]
        anomaly_label = _build_anomaly_label(weather_data, anomaly_prediction)
        anomaly_reason = _build_anomaly_reason(weather_data, anomaly_prediction)

        operation_message = process_mosdac_data_for_city(city_name)
        user_output = user_defined_function_city(city_name)
        _ = user_defined_function_coords(data.latitude, data.longitude)

        return CityWeatherResult(
            status="Success",
            latitude=data.latitude,
            longitude=data.longitude,
            city_name=city_name,
            temperature_c=weather_data.get("temperature_c"),
            humidity_percent=weather_data.get("humidity_percent"),
            rain_mm=weather_data.get("rain_mm"),
            wind_speed_kmh=weather_data.get("wind_speed_kmh"),
            pressure_hpa=weather_data.get("pressure_hpa"),
            cloud_cover_percent=weather_data.get("cloud_cover_percent"),
            precipitation_mm=weather_data.get("precipitation_mm"),
            new_data=new_data,
            temperature_prediction=temperature_prediction,
            anomaly_prediction=anomaly_prediction,
            anomaly_label=anomaly_label,
            anomaly_reason=anomaly_reason,
            operation_message=operation_message,
            user_function_output=user_output,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Weather or geocoding service failed: {str(exc)}") from exc


@app.get("/", response_class=HTMLResponse)
def serve_map_interface():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>MOSDAC Weather Intelligence</title>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <style>
            :root {
                --bg: #06131b;
                --bg-soft: #0d1d2b;
                --panel: rgba(12, 22, 31, 0.84);
                --panel-strong: rgba(9, 16, 23, 0.96);
                --line: rgba(255,255,255,0.08);
                --text: #ebf6ff;
                --muted: #a9c1d3;
                --cyan: #7dd3fc;
                --blue: #60a5fa;
                --purple: #a78bfa;
                --teal: #34d399;
                --amber: #fbbf24;
                --red: #f87171;
                --shadow: 0 20px 48px rgba(2, 6, 12, 0.5);
            }

            * { box-sizing: border-box; }

            html, body {
                margin: 0;
                height: 100%;
                font-family: Inter, "Segoe UI", sans-serif;
                background:
                    radial-gradient(circle at top left, rgba(96,165,250,0.18), transparent 28%),
                    radial-gradient(circle at bottom right, rgba(167,139,250,0.18), transparent 24%),
                    var(--bg);
                color: var(--text);
            }

            body {
                min-height: 100vh;
                overflow: hidden;
            }

            .app-shell {
                display: flex;
                min-height: 100vh;
                width: 100%;
            }

            .map-pane {
                position: relative;
                flex: 1.65;
                min-width: 0;
                background: #0b1a22;
            }

            .map-pane::before {
                content: "";
                position: absolute;
                inset: 0;
                background: linear-gradient(180deg, rgba(2,6,12,0.10), rgba(2,6,12,0.30));
                pointer-events: none;
                z-index: 500;
            }

            .map-overlay {
                position: absolute;
                top: 20px;
                left: 20px;
                z-index: 700;
                display: flex;
                align-items: center;
                gap: 12px;
                background: rgba(8, 17, 25, 0.7);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 14px;
                padding: 12px 14px;
                backdrop-filter: blur(10px);
                box-shadow: var(--shadow);
            }

            .brand-mark {
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background: linear-gradient(135deg, var(--cyan), var(--teal));
                box-shadow: 0 0 16px rgba(125, 211, 252, 0.9);
            }

            .map-overlay .title {
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                font-size: 0.75rem;
                color: var(--muted);
            }

            .map-overlay .live {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 6px 10px;
                border-radius: 999px;
                background: rgba(52, 211, 153, 0.12);
                border: 1px solid rgba(52, 211, 153, 0.2);
                color: #d9fff0;
                font-weight: 600;
                font-size: 0.72rem;
            }

            .map-overlay .live::before {
                content: "";
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: var(--teal);
                box-shadow: 0 0 12px rgba(52, 211, 153, 0.9);
            }

            #map {
                width: 100%;
                height: 100vh;
                background: #07141b;
            }

            #crosshair {
                position: absolute;
                inset: 0;
                pointer-events: none;
                z-index: 600;
            }

            #crosshair::before,
            #crosshair::after {
                content: "";
                position: absolute;
                left: 50%;
                top: 50%;
                transform: translate(-50%, -50%);
                pointer-events: none;
                background: rgba(255,255,255,0.9);
                box-shadow: 0 0 14px rgba(125,211,252,0.5);
            }

            #crosshair::before {
                width: 2px;
                height: 140px;
            }

            #crosshair::after {
                width: 140px;
                height: 2px;
            }

            .dashboard {
                width: 440px;
                min-width: 440px;
                background: linear-gradient(180deg, rgba(8,16,23,0.97), rgba(5,11,16,0.94));
                border-left: 1px solid var(--line);
                box-shadow: var(--shadow);
                padding: 22px 18px 18px;
                overflow-y: auto;
            }

            .header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 12px;
                margin-bottom: 18px;
                padding-bottom: 14px;
                border-bottom: 1px solid rgba(255,255,255,0.06);
            }

            .header h1 {
                margin: 0;
                font-size: 1.4rem;
                letter-spacing: 0.03em;
            }

            .header .pill {
                border: 1px solid rgba(125,211,252,0.2);
                background: rgba(125,211,252,0.08);
                color: var(--cyan);
                border-radius: 999px;
                padding: 7px 10px;
                font-size: 0.7rem;
                letter-spacing: 0.12em;
                text-transform: uppercase;
            }

            .summary-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 12px;
                margin-bottom: 18px;
            }

            .card {
                background: rgba(15, 24, 35, 0.82);
                border: 1px solid var(--line);
                border-radius: 16px;
                padding: 14px 12px;
                box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
                animation: fadeUp 0.65s ease both;
            }

            .card.small {
                min-height: 108px;
            }

            .label {
                display: block;
                font-size: 0.68rem;
                text-transform: uppercase;
                letter-spacing: 0.12em;
                color: var(--muted);
                margin-bottom: 8px;
            }

            .value {
                font-size: 1.55rem;
                font-weight: 700;
                letter-spacing: -0.04em;
            }

            .value .unit {
                font-size: 0.75rem;
                color: var(--muted);
                margin-left: 4px;
            }

            .trend {
                display: flex;
                align-items: center;
                justify-content: center;
                height: 42px;
                margin-top: 8px;
            }

            .trend svg {
                width: 100%;
                height: 42px;
                overflow: visible;
            }

            .stat-panel {
                margin-bottom: 18px;
                border-radius: 18px;
                padding: 14px 14px 10px;
                background: rgba(14, 23, 33, 0.8);
                border: 1px solid var(--line);
            }

            .row {
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 12px;
                padding: 11px 0;
                border-bottom: 1px solid rgba(255,255,255,0.04);
                font-size: 0.95rem;
            }

            .row:last-child {
                border-bottom: none;
            }

            .row span:first-child {
                color: var(--muted);
            }

            .coord {
                font-family: "SFMono-Regular", Consolas, monospace;
                font-weight: 700;
                color: #dfefff;
            }

            .badge {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                padding: 5px 9px;
                border-radius: 999px;
                font-size: 0.72rem;
                font-weight: 700;
                letter-spacing: 0.05em;
                text-transform: uppercase;
                border: 1px solid rgba(255,255,255,0.08);
                background: rgba(255,255,255,0.03);
            }

            .badge.critical {
                background: rgba(248, 113, 113, 0.11);
                border-color: rgba(248,113,113,0.2);
                color: #ffd4d4;
            }

            .badge.normal {
                background: rgba(52, 211, 153, 0.12);
                border-color: rgba(52,211,153,0.22);
                color: #d9fff0;
            }

            .flow-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 10px;
                margin: 18px 0;
            }

            .flow-step {
                position: relative;
                background: rgba(10, 18, 27, 0.8);
                border: 1px solid var(--line);
                border-radius: 14px;
                padding: 12px 10px 10px;
                text-align: center;
                font-size: 0.72rem;
                color: var(--muted);
                min-height: 88px;
            }

            .flow-step:not(:last-child)::after {
                content: "→";
                position: absolute;
                right: -10px;
                top: 50%;
                transform: translateY(-50%);
                color: rgba(255,255,255,0.38);
            }

            .flow-step .num {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 26px;
                height: 26px;
                border-radius: 50%;
                background: linear-gradient(135deg, rgba(125,211,252,0.17), rgba(96,165,250,0.06));
                border: 1px solid rgba(125,211,252,0.22);
                color: var(--cyan);
                font-size: 0.72rem;
                font-weight: 700;
                margin-bottom: 8px;
            }

            .detection-window {
                margin-top: 18px;
                background: linear-gradient(180deg, rgba(12, 20, 29, 0.96), rgba(6, 11, 16, 0.95));
                border: 1px solid rgba(125, 211, 252, 0.2);
                border-radius: 20px;
                padding: 14px;
                box-shadow: 0 24px 40px rgba(2, 8, 15, 0.55), inset 0 1px 0 rgba(255,255,255,0.06);
                position: relative;
                overflow: hidden;
            }

            .detection-window::before {
                content: "";
                position: absolute;
                inset: 0 0 auto 0;
                height: 3px;
                background: linear-gradient(90deg, #38bdf8, #60a5fa, #a78bfa);
            }

            .analysis-box {
                background: rgba(8, 15, 23, 0.72);
                border: 1px solid rgba(255,255,255,0.06);
                border-radius: 16px;
                padding: 12px 12px 10px;
                box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
            }

            .analysis-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 12px;
                margin-bottom: 10px;
            }

            .analysis-header h3 {
                margin: 0;
                font-size: 0.76rem;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                color: var(--muted);
            }

            .reason {
                margin-top: 12px;
                padding: 12px 13px;
                border-radius: 12px;
                background: rgba(11, 18, 25, 0.7);
                border: 1px solid rgba(255,255,255,0.06);
                font-size: 0.95rem;
                line-height: 1.7;
                color: #e9f9ff;
            }

            .button-toolbar {
                display: flex;
                gap: 10px;
                margin-top: 18px;
            }

            button {
                flex: 1;
                border: none;
                border-radius: 14px;
                background: linear-gradient(135deg, #1d9bf0, #2563eb, #7c3aed);
                color: white;
                font-size: 0.96rem;
                font-weight: 700;
                padding: 14px 18px;
                cursor: pointer;
                box-shadow: 0 16px 28px rgba(59,130,246,0.35);
                transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease;
                animation: floatPulse 3s ease-in-out infinite;
            }

            button:hover:not(:disabled) {
                transform: translateY(-2px);
                box-shadow: 0 20px 30px rgba(59,130,246,0.42);
                filter: brightness(1.08);
            }

            button:disabled {
                opacity: 0.55;
                cursor: wait;
            }

            .mini-output {
                margin-top: 18px;
                background: rgba(9, 15, 23, 0.7);
                border: 1px solid var(--line);
                border-radius: 16px;
                padding: 12px 14px;
                color: #dfeeff;
                font-size: 12px;
                line-height: 1.7;
                max-height: 200px;
                overflow: auto;
            }

            .leaflet-control-zoom {
                border: 1px solid rgba(255,255,255,0.12) !important;
                border-radius: 12px !important;
                overflow: hidden;
                box-shadow: var(--shadow);
            }

            .leaflet-control-zoom a {
                background: rgba(8, 18, 28, 0.82) !important;
                color: white !important;
                border-bottom: 1px solid rgba(255,255,255,0.08) !important;
            }

            .leaflet-container {
                background: #08151d;
                filter: saturate(1.15) contrast(1.08);
            }

            .leaflet-popup-content-wrapper, .leaflet-popup-tip {
                background: rgba(8, 17, 25, 0.96);
                color: white;
                border: 1px solid rgba(255,255,255,0.08);
            }

            @keyframes fadeUp {
                from {
                    opacity: 0;
                    transform: translateY(10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            @keyframes floatPulse {
                0%, 100% { transform: translateY(0); }
                50% { transform: translateY(-2px); }
            }

            @media (max-width: 1200px) {
                .app-shell {
                    flex-direction: column;
                }
                .map-pane {
                    flex: 1;
                    min-height: 55vh;
                }
                .dashboard {
                    width: 100%;
                    min-width: 0;
                    max-height: 45vh;
                }
                body { overflow: auto; }
            }
        </style>
    </head>
    <body>
        <div class="app-shell">
            <div class="map-pane">
                <div class="map-overlay">
                    <div class="brand-mark"></div>
                    <div class="title">Live Weather Intelligence</div>
                    <div class="live">Live</div>
                </div>
                <div id="map"></div>
                <div id="crosshair"></div>
            </div>

            <aside class="dashboard">
                <div class="header">
                    <h1>Weather Insights</h1>
                    <div class="pill">MOSDAC</div>
                </div>

                <div class="summary-grid">
                    <div class="card small">
                        <span class="label">Temp</span>
                        <div class="value"><span id="temp_display">-</span><span class="unit">°C</span></div>
                        <div class="trend">
                            <svg viewBox="0 0 80 42" preserveAspectRatio="none">
                                <path d="M0,30 C15,25, 22,14, 38,18 S60,28, 80,10" fill="none" stroke="#7dd3fc" stroke-width="2.2" stroke-linecap="round"></path>
                            </svg>
                        </div>
                    </div>
                    <div class="card small">
                        <span class="label">Humidity</span>
                        <div class="value"><span id="humidity_display">-</span><span class="unit">%</span></div>
                        <div class="trend">
                            <svg viewBox="0 0 80 42" preserveAspectRatio="none">
                                <path d="M0,20 C16,12, 26,30, 40,19 S60,8, 80,24" fill="none" stroke="#60a5fa" stroke-width="2.2" stroke-linecap="round"></path>
                            </svg>
                        </div>
                    </div>
                    <div class="card small">
                        <span class="label">Risk</span>
                        <div class="value"><span id="risk_display">-</span></div>
                        <div class="trend">
                            <svg viewBox="0 0 80 42" preserveAspectRatio="none">
                                <path d="M0,35 C15,18, 28,10, 40,18 S62,24, 80,12" fill="none" stroke="#fbbf24" stroke-width="2.2" stroke-linecap="round"></path>
                            </svg>
                        </div>
                    </div>
                </div>

                <div class="stat-panel">
                    <div class="row"><span>City</span><span id="city_display">-</span></div>
                    <div class="row"><span>Latitude</span><span id="lat_display" class="coord">-</span></div>
                    <div class="row"><span>Longitude</span><span id="lon_display" class="coord">-</span></div>
                    <div class="row"><span>Wind</span><span id="wind_display">-</span> km/h</div>
                    <div class="row"><span>Pressure</span><span id="pressure_display">-</span> hPa</div>
                    <div class="row"><span>Output</span><span id="user_output">-</span></div>
                </div>

                <div class="flow-grid">
                    <div class="flow-step">
                        <div class="num">1</div>
                        Select point
                    </div>
                    <div class="flow-step">
                        <div class="num">2</div>
                        Pull weather
                    </div>
                    <div class="flow-step">
                        <div class="num">3</div>
                        Predict model
                    </div>
                    <div class="flow-step">
                        <div class="num">4</div>
                        Explain anomaly
                    </div>
                </div>

                <div class="detection-window">
                    <div class="analysis-header">
                        <h3>Detection Pane</h3>
                        <span id="status_badge" class="badge normal">Stable</span>
                    </div>
                    <div class="analysis-box">
                        <div class="row"><span>Model Temp</span><span id="temp_pred_display">-</span> °C</div>
                        <div class="row"><span>Prediction</span><span id="anomaly_value_display">-</span></div>
                        <div class="row"><span>Condition</span><span id="anomaly_label_display" class="badge normal">-</span></div>
                        <div class="reason" id="anomaly_reason_display">Select a location to begin analysis.</div>
                    </div>
                </div>

                <div class="button-toolbar">
                    <button id="process_btn" disabled onclick="executeBackendPipeline()">Run Forecast</button>
                </div>

                <pre class="mini-output" id="json_output">Click any point on the map to fetch live weather, forecast temperature, and detect anomalies.</pre>
            </aside>
        </div>

        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <script>
            const defaultCenter = [22.5726, 88.3639];
            const map = L.map('map', {
                center: defaultCenter,
                zoom: 11,
                zoomControl: true,
                attributionControl: true,
                scrollWheelZoom: true,
                preferCanvas: true
            });

            L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
                maxZoom: 19,
                attribution: 'Tiles &copy; Esri'
            }).addTo(map);

            L.tileLayer('https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places_Alternate/MapServer/tile/{z}/{y}/{x}', {
                maxZoom: 19,
                opacity: 0.55
            }).addTo(map);

            L.control.scale({ imperial: false, metric: true, position: 'bottomleft' }).addTo(map);

            let activeMarker = null;
            let selectedLat = null;
            let selectedLon = null;

            function setBadge(mode) {
                const badge = document.getElementById('status_badge');
                if (!badge) return;
                if (mode === 'critical') {
                    badge.textContent = 'Anomaly';
                    badge.className = 'badge critical';
                } else {
                    badge.textContent = 'Stable';
                    badge.className = 'badge normal';
                }
            }

            function updateSelection(lat, lng) {
                selectedLat = lat;
                selectedLon = lng;
                document.getElementById('lat_display').innerText = lat.toFixed(6);
                document.getElementById('lon_display').innerText = lng.toFixed(6);
                document.getElementById('process_btn').disabled = false;

                if (activeMarker) {
                    activeMarker.setLatLng([lat, lng]);
                } else {
                    activeMarker = L.marker([lat, lng], {
                        draggable: true,
                        riseOnHover: true,
                        title: 'Selected location'
                    }).addTo(map);

                    activeMarker.on('dragend', function () {
                        const pos = activeMarker.getLatLng();
                        selectedLat = pos.lat;
                        selectedLon = pos.lng;
                        document.getElementById('lat_display').innerText = selectedLat.toFixed(6);
                        document.getElementById('lon_display').innerText = selectedLon.toFixed(6);
                    });
                }

                executeBackendPipeline();
            }

            map.on('click', function (event) {
                const lat = event.latlng.lat;
                const lng = event.latlng.lng;
                map.flyTo([lat, lng], Math.max(map.getZoom(), 12), { duration: 1.2 });
                updateSelection(lat, lng);
            });

            activeMarker = L.marker(defaultCenter, { draggable: true, riseOnHover: true }).addTo(map);
            activeMarker.on('dragend', function () {
                const pos = activeMarker.getLatLng();
                selectedLat = pos.lat;
                selectedLon = pos.lng;
                document.getElementById('lat_display').innerText = selectedLat.toFixed(6);
                document.getElementById('lon_display').innerText = selectedLon.toFixed(6);
            });
            updateSelection(defaultCenter[0], defaultCenter[1]);
            document.getElementById('process_btn').disabled = false;

            async function executeBackendPipeline() {
                const output = document.getElementById('json_output');
                output.textContent = 'Analyzing weather conditions...';
                document.getElementById('process_btn').disabled = true;

                try {
                    const response = await fetch('/api/pin-location', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ latitude: selectedLat, longitude: selectedLon }),
                    });

                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}`);
                    }

                    const payload = await response.json();
                    output.textContent = JSON.stringify(payload, null, 2);

                    document.getElementById('city_display').innerText = payload.city_name || '-';
                    document.getElementById('user_output').innerText = payload.user_function_output || '-';
                    document.getElementById('temp_display').innerText = payload.temperature_c != null ? Number(payload.temperature_c).toFixed(1) : '-';
                    document.getElementById('humidity_display').innerText = payload.humidity_percent != null ? Number(payload.humidity_percent).toFixed(1) : '-';
                    document.getElementById('wind_display').innerText = payload.wind_speed_kmh != null ? Number(payload.wind_speed_kmh).toFixed(1) : '-';
                    document.getElementById('pressure_display').innerText = payload.pressure_hpa != null ? Number(payload.pressure_hpa).toFixed(1) : '-';
                    document.getElementById('temp_pred_display').innerText = payload.temperature_prediction != null ? Number(payload.temperature_prediction).toFixed(2) : '-';
                    const anomalyValue = Number(payload.anomaly_prediction ?? 0);
                    document.getElementById('risk_display').innerText = anomalyValue === 1 ? 'High' : 'Low';
                    document.getElementById('anomaly_value_display').innerText = anomalyValue === 1 ? 'Detected' : 'Normal';
                    document.getElementById('anomaly_label_display').textContent = payload.anomaly_label || '-';
                    document.getElementById('anomaly_reason_display').textContent = payload.anomaly_reason || 'No anomaly reason available.';

                    const labelEl = document.getElementById('anomaly_label_display');
                    if (anomalyValue === 1) {
                        labelEl.className = 'badge critical';
                        setBadge('critical');
                    } else {
                        labelEl.className = 'badge normal';
                        setBadge('normal');
                    }

                } catch (error) {
                    output.textContent = 'Error communicating with API server: ' + error;
                    document.getElementById('anomaly_reason_display').textContent = 'Unable to reach the live weather backend.';
                } finally {
                    document.getElementById('process_btn').disabled = false;
                }
            }
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("map_weather_api:app", host="0.0.0.0", port=8000, reload=False)
