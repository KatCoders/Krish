import os
import requests
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load API keys from env
from dotenv import load_dotenv
load_dotenv()
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "").strip()
DATA_GOV_API_KEY = os.getenv("DATA_GOV_API_KEY", "").strip()

# ---- Market Data (data.gov.in) ----
def fetch_market_data(state: str, commodity: str) -> Optional[dict]:
    try:
        RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"
        url = f"https://api.data.gov.in/resource/{RESOURCE_ID}"
        params = {
            "api-key": DATA_GOV_API_KEY,
            "format": "json",
            "limit": 10,
            "filters[state]": state,
            "filters[commodity]": commodity
        }

        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        records = resp.json().get("records", [])

        if not records:
            return None

        latest = sorted(records, key=lambda x: x["arrival_date"], reverse=True)[0]
        return {
            "commodity": latest["commodity"],
            "state": latest["state"],
            "market": latest["market"],
            "modal_price": latest["modal_price"],
            "arrival_date": latest["arrival_date"]
        }
    except Exception as e:
        return {"error": str(e)}

def format_price_response(data: dict) -> str:
    if not data or data.get("error"):
        return "क्षमा करें, इस फसल के लिए ताज़ा मंडी भाव उपलब्ध नहीं है।"
    
    return (
        f"📊 **{data['commodity']}** का भाव ({data['market']}, {data['state']}):\n\n"
        f"💰 औसत (मोडल) भाव: ₹{data['modal_price']} प्रति क्विंटल\n"
        f"📅 तारीख: {data['arrival_date']}"
    )

# ---- Weather Data (weatherapi.com with fallback) ----
def fetch_weather(location: str) -> Dict[str, Any]:
    if not WEATHER_API_KEY:
        return {"temperature": 25, "humidity": 70, "condition": "साफ़"}
    
    try:
        url = "http://api.weatherapi.com/v1/current.json"
        params = {"key": WEATHER_API_KEY, "q": location}
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        current = data.get("current", {})
        return {
            "temperature": current.get("temp_c", 25),
            "humidity": current.get("humidity", 70),
            "condition": current.get("condition", {}).get("text", "साफ़")
        }
    except:
        return {"temperature": 25, "humidity": 70, "condition": "साफ़"}

# ---- Soil Data (simulated) ----
def fetch_soil() -> Dict[str, float]:
    return {"ph": 6.5, "nitrogen": 50, "organic_carbon": 10}

# ---- Simple ML Model ----
def get_crop_prediction(soil: dict, weather: dict) -> tuple:
    try:
        clf, scaler = get_trained_model()
        X = np.array([[weather["temperature"], weather["humidity"], soil["ph"], soil["nitrogen"]]])
        Xs = scaler.transform(X)
        probs = clf.predict_proba(Xs)[0]
        pred = int(clf.predict(Xs)[0])
        crop_map = {0: "🌾 गेहूँ", 1: "🌱 धान", 2: "🌽 मक्का"}
        return crop_map.get(pred, "🔄 अज्ञात"), float(max(probs) * 100)
    except Exception:
        return "🌾 गेहूँ", 75.0


def get_trained_model():
    np.random.seed(42)
    n = 2000
    feats, labels = [], []
    for _ in range(n):
        t = np.random.normal(25, 10)
        h = np.random.normal(70, 20)
        ph = np.random.normal(6.5, 1.2)
        nval = np.random.normal(50, 25)
        feats.append([t, h, ph, nval])
        if t < 22 and h > 55 and ph > 6.0:
            labels.append(0)
        elif t > 28 and h > 75 and ph < 7.5:
            labels.append(1)
        elif 20 < t < 35 and h < 80:
            labels.append(2)
        else:
            labels.append(np.random.choice([0, 1, 2]))
    X = np.array(feats)
    y = np.array(labels)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_split=5, min_samples_leaf=2)
    clf.fit(Xs, y)
    return clf, scaler
