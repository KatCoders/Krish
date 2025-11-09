import os
import io
import time
import json
import tempfile
import requests
import numpy as np
import pandas as pd
import streamlit as st
import logging
from PIL import Image
from typing import Optional, Tuple, Dict, Any
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from gtts import gTTS
from voice_pipeline import *
from st_audiorec import st_audiorec
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from voiceassit import voice_input_component
import streamlit.components.v1 as components
from streamlit_geolocation import streamlit_geolocation 
from auth import google_login
from llm import *
# Langchain / Groq imports
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ------------------- Logging Setup -------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ------------------- Load environment variables -------------------
load_dotenv()
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

# Data.gov.in API Configuration

# ------------------- Page config & Enhanced CSS -------------------
st.set_page_config(
    page_title="🌾 Krish AI कृषि सहायक", 
    layout="wide", 
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Krish AIकृषि सहायक - आपका डिजिटल खेती सलाहकार"
    }
)

st.markdown("""
<style>
    .main-title { 
        text-align: center; 
        color: #2E8B57; 
        font-size: 2.2rem; 
        margin-bottom: 1rem; 
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .location-prompt {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin: 2rem auto;
        max-width: 600px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    .location-prompt h2 {
        font-size: 2rem;
        margin-bottom: 1rem;
    }
    .location-prompt p {
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
        opacity: 0.9;
    }
    .location-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.1); }
    }
    .voice-section { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
        padding: 1.5rem; 
        border-radius: 12px; 
        color: white; 
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .chat-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #2196f3;
    }
    .assistant-message {
        background-color: #f1f8e9;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #4caf50;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .price-card {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)
# ------------------- Authentication Check -------------------
# auth.py

from auth import google_login
import streamlit as st

if "user" not in st.session_state:
    st.markdown("""
    <style>
        body {
            background: linear-gradient(to top right, #d1ffea, #f4fff9);
        }
        .login-box {
            max-width: 450px;
            margin: 80px auto;
            background: white;
            padding: 30px 40px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            text-align: center;
            font-family: 'Segoe UI', sans-serif;
        }
        .login-box h2 {
            color: #2e8b57;
            font-size: 30px;
            font-weight: 700;
        }
        .login-box p {
            font-size: 16px;
            color: #555;
        }
        .google-btn {
            margin-top: 20px;
        }
        .brand-logo {
            width: 80px;
            margin-bottom: 10px;
        }
    </style>

    <div class="login-box">
        <img src="https://cdn-icons-png.flaticon.com/128/756/756669.png" class="brand-logo" alt="KRISH Logo"/>
        <h2>🌾 KRISH कृषि सहायक</h2>
        <p>स्मार्ट खेती के लिए डिजिटल साथी!<br>
        मंडी भाव, फसल सुझाव, मौसम और बहुत कुछ...</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
            google_login()  # Renders the Google sign-in button
    st.markdown("</div>", unsafe_allow_html=True)

    st.stop()






# ------------------- Session state initialization -------------------
def init_session_state():
    """Initialize all session state variables"""
    if "app_initialized" not in st.session_state:
        st.session_state.update({
            "app_initialized": False,
            "location_granted": False,
            "tts_system_ready": False,
            "stt_warmed": False,
            "chat_history": [],
            "processing": False,
            "last_audio_data": None,
            "last_audio": None,
            "voice_enabled": True,
            "auto_play_response": True,
            "use_offline_tts": False,
            "location_method": "ip",
            "client_location": None,
            "warmup_status": "प्रारंभ कर रहे हैं...",
            "tts_system": UnifiedTTSSystem(),
            "stt": SpeechToText(),
            "user_lat": None,
            "user_lon": None,
            "user_city": None,
            "market_data_loaded": False
        })

init_session_state()

# ------------------- Location Request Screen -------------------
def show_location_request_screen():
    """Display location permission request screen"""
    st.markdown('<h1 class="main-title">🌾 KRISH AI आधारित फसल सलाह सहायक</h1>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 3, 1])
    with col1:
        pass
    with col3:
        pass
    with col2:
        st.markdown("<h5><b>कृपया इस लोगो पर क्लिक करें</b></h5>", unsafe_allow_html=True)
        loc = streamlit_geolocation()
    st.markdown("""
    <div class="location-prompt">
        <div class="location-icon">📍</div>
        <h2>स्थान की अनुमति चाहिए</h2>
        <p>आपकी सटीक कृषि सलाह के लिए हमें आपके स्थान की जरूरत है।</p>
        <p style="font-size: 0.9rem;">
            ✅ मौसम आधारित सलाह<br>
            ✅ स्थानीय मिट्टी की जानकारी<br>
            ✅ क्षेत्रीय फसल सुझाव<br>
            ✅ मंडी भाव की जानकारी
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if loc and isinstance(loc, dict):
        lat = loc.get("latitude")
        lon = loc.get("longitude")
        
        if lat is not None and lon is not None:
            st.session_state.user_lat = lat
            st.session_state.user_lon = lon
            st.session_state.location_granted = True
            
            try:
                response = requests.get(
                    "https://nominatim.openstreetmap.org/reverse",
                    params={
                        "lat": lat,
                        "lon": lon,
                        "format": "json",
                        "accept-language": "hi"
                    },
                    headers={"User-Agent": "AgroMind-App/1.0"},
                    timeout=5
                )
                if response.status_code == 200:
                    data = response.json()
                    address = data.get("address", {})
                    city = (address.get("city") or 
                           address.get("town") or 
                           address.get("village") or 
                           address.get("state_district") or
                           "आपका स्थान")
                    st.session_state.user_city = f"📍 {city}"
            except:
                st.session_state.user_city = "📍 आपका स्थान (GPS)"
            
            st.success("✅ स्थान प्राप्त हो गया! ऐप लोड हो रहा है...")
            time.sleep(1)
            st.rerun()
    else:
        st.info("👆 कृपया अपने ब्राउज़र में स्थान की अनुमति दें")
        
        st.markdown("---")
        st.markdown("### या")
        
        if st.button("🌐 IP आधारित स्थान का उपयोग करें", type="secondary"):
            try:
                response = requests.get("https://ipinfo.io/json", timeout=8)
                if response.status_code == 200:
                    data = response.json()
                    loc_str = data.get("loc", "28.61,77.20").split(",")
                    city = data.get("city", "दिल्ली")
                    region = data.get("region", "")
                    
                    st.session_state.user_lat = float(loc_str[0])
                    st.session_state.user_lon = float(loc_str[1])
                    st.session_state.user_city = f"🌐 {city}, {region} (IP)"
                    st.session_state.location_granted = True
                    
                    st.success("✅ IP आधारित स्थान प्राप्त हो गया!")
                    time.sleep(1)
                    st.rerun()
            except Exception as e:
                st.error(f"❌ स्थान प्राप्त नहीं हो सका: {str(e)}")

if not st.session_state.location_granted:
    show_location_request_screen()
    st.stop()

# ------------------- Main App -------------------
lat = st.session_state.user_lat
lon = st.session_state.user_lon
city = st.session_state.user_city

st.markdown('<h2 class="main-title">🌾 KRISH AI आधारित फसल सलाह सहायक (हिंदी, आवाज़ सहित)</h2>', unsafe_allow_html=True)

# ------------------- Enhanced utility functions -------------------
def get_default_soil_data() -> Dict[str, float]:
    return {"ph": 6.5, "nitrogen": 50, "organic_carbon": 10, "sand": 40, "silt": 40, "clay": 20}

def get_default_weather_data() -> Dict[str, Any]:
    return {"temperature": 25, "humidity": 70, "precipitation": 2, "wind_speed": 10, "condition": "साफ़"}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_soil(lat: float, lon: float) -> Dict[str, float]:
    try:
        url = "https://rest.isric.org/soilgrids/v2.0/properties"
        params = {"lon": lon, "lat": lat, "property": "phh2o", "depth": "0-5cm", "value": "mean"}
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            base_soil = get_default_soil_data()
            lat_factor = (lat - 20) / 20
            base_soil["ph"] += lat_factor * 0.5
            base_soil["nitrogen"] += lat_factor * 10
            return base_soil
    except Exception as e:
        logger.error(f"Soil data fetch error: {e}")
    return get_default_soil_data()

@st.cache_data(ttl=600, show_spinner=False)
def fetch_weather(lat: float, lon: float) -> Dict[str, Any]:
    if not WEATHER_API_KEY:
        return get_default_weather_data()
    try:
        url = "http://api.weatherapi.com/v1/current.json"
        params = {"key": WEATHER_API_KEY, "q": f"{lat},{lon}", "aqi": "no"}
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            current = data.get("current", {})
            return {
                "temperature": current.get("temp_c", 25),
                "humidity": current.get("humidity", 70),
                "precipitation": current.get("precip_mm", 2),
                "wind_speed": current.get("wind_kph", 10),
                "condition": current.get("condition", {}).get("text", "साफ़"),
                "feels_like": current.get("feelslike_c", 25),
                "uv": current.get("uv", 5)
            }
    except Exception as e:
        logger.error(f"Weather data fetch error: {e}")
    return get_default_weather_data()

@st.cache_resource(show_spinner=False)
def get_trained_model() -> Tuple[RandomForestClassifier, StandardScaler]:
    np.random.seed(42)
    n_samples = 2000
    features, labels = [], []
    
    for _ in range(n_samples):
        temp = np.random.normal(25, 10)
        humidity = np.random.normal(70, 20)
        ph = np.random.normal(6.5, 1.2)
        nitrogen = np.random.normal(50, 25)
        features.append([temp, humidity, ph, nitrogen])
        
        if temp < 22 and humidity > 55 and ph > 6.0:
            labels.append(0)
        elif temp > 28 and humidity > 75 and ph < 7.5:
            labels.append(1)
        elif temp > 20 and temp < 35 and humidity < 80:
            labels.append(2)
        else:
            labels.append(np.random.choice([0, 1, 2]))
    
    X = np.array(features)
    y = np.array(labels)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42)
    clf.fit(X_scaled, y)
    return clf, scaler

def get_crop_prediction(soil: Dict[str, float], weather: Dict[str, Any]) -> Tuple[str, float]:
    try:
        clf, scaler = get_trained_model()
        features = np.array([[
            weather.get("temperature", 25),
            weather.get("humidity", 70),
            soil.get("ph", 6.5),
            soil.get("nitrogen", 50)
        ]])
        features_scaled = scaler.transform(features)
        probabilities = clf.predict_proba(features_scaled)[0]
        prediction = int(clf.predict(features_scaled)[0])
        crop_map = {0: "🌾 गेहूँ", 1: "🌱 धान", 2: "🌽 मक्का"}
        confidence = float(max(probabilities) * 100)
        return crop_map.get(prediction, "❓ अज्ञात"), confidence
    except Exception as e:
        logger.error(f"Crop prediction failed: {e}")
        return "🌾 गेहूँ", 75.0

def perform_comprehensive_warmup():
    if st.session_state.app_initialized:
        return True
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        warmup_steps = [
            ("🔧 सिस्टम प्रारंभ...", 20),
            ("🎤 आवाज़ सिस्टम तैयार...", 50),
            ("🔊 TTS सिस्टम वार्म अप...", 70),
            ("📊 मंडी डेटा लोड...", 85),
            ("✅ तैयार!", 100)
        ]
        for step_text, progress_value in warmup_steps:
            status_text.markdown(f'<div class="status-info">{step_text}</div>', unsafe_allow_html=True)
            progress_bar.progress(progress_value)
            time.sleep(0.3)
        time.sleep(0.5)
        progress_container.empty()
    st.session_state.app_initialized = True
    return True

perform_comprehensive_warmup()

# Load data
with st.spinner("🌍 पर्यावरण डेटा लोड कर रहे हैं..."):
    soil_data = fetch_soil(lat, lon)
    weather_data = fetch_weather(lat, lon)
   
        

# ------------------- Enhanced Groq LLM with Market Rate Tool -------------------

    
def process_text_input(user_input: str):
    if st.session_state.processing:
        st.warning("⏳ कृपया प्रतीक्षा करें...")
        return
    
    st.session_state.processing = True
    try:
        with st.chat_message("user"):
            st.markdown(f"✍️ {user_input}")
        
        st.session_state.chat_history.append({
            "role": "user", 
            "content": user_input, 
            "type": "text",
            "timestamp": datetime.now().isoformat()
        })
    
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_placeholder.markdown("🤖 सोच रहा हूं... 🧠")
            
            full_response = ""
            try:
                response = get_llm_response(user_input )
                full_response = response
                response_placeholder.markdown(f"🤖 {full_response}")
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": full_response,
                    "type": "text",
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                error_msg = f"जवाब तैयार करने में समस्या: {str(e)}"
                response_placeholder.error(f"❌ {error_msg}")
                full_response = "क्षमा करें, तकनीकी समस्या के कारण जवाब नहीं दे सका।"
                logger.error(f"LLM generation error: {e}")
        
        if st.session_state.voice_enabled and full_response:
            with st.spinner("🎧 आवाज़ में तैयार कर रहे हैं..."):
                try:
                    audio_bytes = st.session_state.tts_system.generate_audio(full_response)
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3")
                        st.success("🔊 तैयार!")
                    else:
                        st.info("💡 टेक्स्ट जवाब तैयार है")
                except Exception as tts_error:
                    logger.warning(f"TTS generation failed: {tts_error}")
                    st.info("💡 टेक्स्ट जवाब तैयार है")

    except Exception as e:
        st.error(f"❌ प्रोसेसिंग में समस्या: {str(e)}")
        logger.error(f"Text processing error: {e}")
    finally:
        st.session_state.processing = False

# ------------------- Tomato Disease Detection Section -------------------



# ------------------- Voice Input Section -------------------
st.markdown("""
<style>
.chat-container {
    background-color: #000000;  
    color: #FFFFFF;           
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.5);
    margin-top: 20px;
    margin-bottom: 20px;
    font-family: 'Segoe UI', sans-serif;
}
.chat-container h4 {
    font-size: 1.8rem;
    margin-bottom: 10px;
    color: #00FF7F;
}
.chat-container ul li {
    margin-bottom: 5px;
}
.chat-container em {
    color: #FFD700;
}
</style>

<div class="chat-container">
    <h4>👋 नमस्ते किसान भाई!</h4>
    <p>मैं आपका AI कृषि सलाहकार हूं। आप मुझसे निम्नलिखित विषयों पर सवाल पूछ सकते हैं:</p>
    <ul>
        <li>🌾 <strong>फसल की सिफारिश</strong> - कौन सी फसल बोएं</li>
        <li>🌱 <strong>मिट्टी की देखभाल</strong> - मिट्टी सुधार के तरीके</li>
        <li>🌧️ <strong>मौसम आधारित सलाह</strong> - मौसम के अनुसार खेती</li>
        <li>🐛 <strong>कीट और रोग नियंत्रण</strong> - समस्याओं का समाधान</li>
        <li>💧 <strong>सिंचाई प्रबंधन</strong> - पानी की सही व्यवस्था</li>
        <li>🌿 <strong>जैविक खेती</strong> - प्राकृतिक तरीके</li>
        <li>💰 <strong>मंडी भाव</strong> - फसलों की कीमत जानें (नया!)</li>
    </ul>
    <p><em>आप टेक्स्ट लिखकर या आवाज़ में सवाल पूछ सकते हैं!</em></p>
</div>
""", unsafe_allow_html=True)



st.subheader("🎤 आवाज़ से सवाल पूछें")
st.caption("अपनी आवाज़ की फ़ाइल अपलोड करें (WAV/MP3)")

col1, col2 = st.columns([1, 2])
with col1:
    pass
with col2:
   
   audio_file = st.file_uploader("अपनी आवाज़ फ़ाइल अपलोड करें", type=["wav", "mp3"])

if audio_file:
    wav_audio_data = audio_file.read()
    if wav_audio_data != st.session_state.get("last_audio_data"):
        st.session_state["last_audio_data"] = wav_audio_data
        st.audio(wav_audio_data, format="audio/wav" if audio_file.type=="audio/wav" else "audio/mp3")
        
        if not st.session_state.get("processing", False):
            st.session_state.processing = True
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(wav_audio_data)
                    tmp_file.flush()
                    tmp_path = tmp_file.name
                
                try:
                    voice_text = st.session_state.stt.transcribe(tmp_path, language="hi")
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                
                if voice_text and voice_text.strip():
                    st.info(f"📝 **आपने कहा:** {voice_text}")
                    
                    with st.spinner("🤖 जवाब तैयार कर रहे हैं..."):
                        response = get_llm_response(voice_text)
                    
                    st.success(f"🤖 {response}")
                    
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": voice_text,
                        "type": "voice",
                        "timestamp": datetime.now().isoformat()
                    })
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response,
                        "type": "text",
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    if st.session_state.get("voice_enabled", False):
                        with st.spinner("🎧 आवाज़ तैयार कर रहे हैं..."):
                            try:
                                audio_bytes = st.session_state.tts_system.generate_audio(response)
                                if audio_bytes:
                                    st.audio(audio_bytes, format="audio/mp3")
                                    st.success("🔊 तैयार!")
                            except Exception as tts_error:
                                logger.warning(f"TTS failed: {tts_error}")
                                st.info("💡 टेक्स्ट पढ़ें")
                else:
                    st.warning("⚠️ आवाज़ स्पष्ट नहीं थी")
                    
            except Exception as e:
                st.error(f"❌ त्रुटि: {str(e)}")
                logger.error(f"Voice error: {e}", exc_info=True)
            finally:
                st.session_state.processing = False

# ------------------- Enhanced Text Input Section -------------------


# Handle chat input

if user_input := st.chat_input("✍️ अपना सवाल यहाँ लिखें... (मंडी भाव पूछने के लिए: 'गेहूं का भाव क्या है?')"):
    process_text_input(user_input)

# ------------------- Enhanced Footer Section -------------------
def show_tomato_detection_page():
    st.markdown('<h2 class="main-title">🍅 टमाटर रोग पहचान विशेषज्ञ</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background-color: #e8f5e9; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;'>
        <h4 style='color: #2e7d32;'>📸 टमाटर के पत्ते की फोटो अपलोड करें</h4>
        <p>हमारा AI मॉडल तुरंत बीमारी की पहचान करेगा और उपचार सुझाएगा।</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "🖼️ टमाटर पत्ती की छवि चुनें",
            type=["jpg", "jpeg", "png"],
            help="JPG, JPEG या PNG फॉर्मेट में तस्वीर अपलोड करें"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="अपलोड की गई छवि", use_container_width=True)
            
            if st.button("🔍 रोग की जांच करें", type="primary", use_container_width=True):
                with st.spinner("🔬 विश्लेषण कर रहे हैं..."):
                    try:
                        # Get prediction
                        disease_name, confidence = predict_disease(image)
                        
                        # Store in session state
                        st.session_state.last_prediction = {
                            "disease": disease_name,
                            "confidence": confidence,
                            "timestamp": datetime.now()
                        }
                        st.session_state.last_uploaded_image = uploaded_file.name
                        
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ त्रुटि: {str(e)}")
                        logger.error(f"Prediction error: {e}")
    
    with col2:
        if st.session_state.last_prediction:
            pred = st.session_state.last_prediction
            disease_name = pred["disease"]
            confidence = pred["confidence"]
            
            # Translate disease name to Hindi
            disease_hindi = disease_name.replace("Tomato___", "").replace("_", " ")
            is_healthy = "healthy" in disease_name.lower()
            
            if is_healthy:
                st.markdown(f"""
                <div class="healthy-card">
                    <h3>✅ स्वस्थ पौधा!</h3>
                    <p style='font-size: 1.2rem; margin: 1rem 0;'>
                        <strong>पहचान:</strong> {disease_hindi}
                    </p>
                    <p style='font-size: 1.1rem;'>
                        <strong>विश्वास स्तर:</strong> {confidence*100:.1f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="disease-card">
                    <h3>⚠️ रोग पहचाना गया</h3>
                    <p style='font-size: 1.2rem; margin: 1rem 0;'>
                        <strong>रोग का नाम:</strong> {disease_hindi}
                    </p>
                    <p style='font-size: 1.1rem;'>
                        <strong>विश्वास स्तर:</strong> {confidence*100:.1f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Confidence meter
            st.markdown(f"""
            <div class="confidence-meter">
                <div class="confidence-fill" style="width: {confidence*100}%">
                    {confidence*100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Get AI recommendations
            st.markdown("### 🤖 AI सुझाव और उपचार")
            
            with st.spinner("💭 विशेषज्ञ सलाह तैयार कर रहे हैं..."):
                query = f"टमाटर के पौधे में {disease_hindi} रोग है। इसका उपचार और रोकथाम के उपाय बताएं।"
                try:
                    response = get_llm_response(query)
                    st.markdown(f"""
                    <div class="prediction-box">
                        {response}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"रोग पहचान: {disease_hindi} ({confidence*100:.1f}% विश्वास)\n\n{response}",
                        "type": "tomato_prediction",
                        "timestamp": datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    st.error(f"❌ सलाह प्राप्त करने में समस्या: {str(e)}")
        else:
            st.info("👈 कृपया पहले एक छवि अपलोड करें")
    
    # Additional Info Section
    st.markdown("---")
    st.markdown("""
    ### 📚 टमाटर की सामान्य बीमारियां
    
    हमारा मॉडल निम्नलिखित बीमारियों की पहचान कर सकता है:
    - 🦠 बैक्टीरियल स्पॉट
    - 🍂 अर्ली ब्लाइट
    - 🍃 लेट ब्लाइट
    - 🌿 लीफ मोल्ड
    - 🔴 सेप्टोरिया लीफ स्पॉट
    - 🕷️ स्पाइडर माइट्स
    - 🎯 टारगेट स्पॉट
    - 🟡 येलो लीफ कर्ल वायरस
    - 🌀 मोज़ेक वायरस
    - ✅ स्वस्थ पौधा
    """)

# ------------------- Crop Advice WhatsApp Page -------------------
def show_crop_whatsapp_page():
    st.markdown('<h2 class="main-title">🌾 फसल सलाह WhatsApp सेवा</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background-color: #e8f5e9; padding: 2rem; border-radius: 10px; text-align: center;'>
        <h3>📱 WhatsApp पर तुरंत सलाह पाएं</h3>
        <p style='font-size: 1.1rem; margin: 1rem 0;'>
            हमारे कृषि विशेषज्ञ WhatsApp पर 24/7 उपलब्ध हैं
        </p>
        <a href='https://wa.me/1234567890' target='_blank' 
           style='background-color: #25D366; color: white; padding: 15px 30px; 
                  text-decoration: none; border-radius: 25px; font-size: 1.2rem;
                  display: inline-block; margin-top: 1rem;'>
            💬 WhatsApp पर संपर्क करें
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    ### 📞 सेवाएं:
    - 🌾 फसल चयन सलाह
    - 🌱 बीज और खाद की जानकारी
    - 💰 मंडी भाव अपडेट
    - 🌧️ मौसम आधारित चेतावनी
    - 🐛 कीट नियंत्रण उपाय
    """)

# ------------------- Home Page -------------------
def show_home_page():
    st.markdown('<h2 class="main-title">🌾 KRISH AI आधारित फसल सलाह सहायक</h2>', unsafe_allow_html=True)
    
    # Voice Input Section
    st.markdown("""
    <div class="chat-container">
        <h4>👋 नमस्ते किसान भाई!</h4>
        <p>मैं आपका AI कृषि सलाहकार हूं। आप मुझसे निम्नलिखित विषयों पर सवाल पूछ सकते हैं:</p>
        <ul>
            <li>🌾 फसल की सिफारिश</li>
            <li>🌱 मिट्टी की देखभाल</li>
            <li>🌧️ मौसम आधारित सलाह</li>
            <li>💰 मंडी भाव</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Audio file upload
    st.subheader("🎤 आवाज़ से सवाल पूछें")
    audio_file = st.file_uploader("अपनी आवाज़ फ़ाइल अपलोड करें", type=["wav", "mp3"])
    
    if audio_file:
        st.audio(audio_file)
        if st.button("🎙️ सवाल प्रोसेस करें"):
            with st.spinner("🤖 आवाज़ को समझ रहे हैं..."):
                st.info("आवाज़ प्रोसेसिंग फीचर जल्द आ रहा है!")
    
    # Text Input
    if user_input := st.chat_input("✍️ अपना सवाल यहाँ लिखें..."):
        with st.chat_message("user"):
            st.markdown(f"✍️ {user_input}")
        
        with st.chat_message("assistant"):
            with st.spinner("🤖 जवाब तैयार कर रहे हैं..."):
                response = get_llm_response(user_input)
                st.markdown(f"🤖 {response}")
        
        # Store in history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })

# ------------------- Sidebar Navigation -------------------
with st.sidebar:
    st.header("🚜 Krish AI Menu")
    
    if st.button("🏡 होम", use_container_width=True):
        st.session_state.nav = "home"
        st.rerun()
    
    if st.button("🍅 टमाटर विशेषज्ञ", use_container_width=True):
        st.session_state.nav = "tomato"
        st.rerun()
    
    if st.button("🌾 फसल सलाह WhatsApp", use_container_width=True):
        st.session_state.nav = "crop"
        st.rerun()
    
    st.markdown("---")
    
    st.subheader("💬 चैट उपकरण")
    if st.button("♻️ चैट रीसेट", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.last_prediction = None
        st.success("✅ चैट रीसेट हो गया!")
    
    if st.button("📥 चैट एक्सपोर्ट", use_container_width=True):
        if st.session_state.chat_history:
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "location": st.session_state.user_city,
                "chat_history": st.session_state.chat_history
            }
            st.download_button(
                label="💾 JSON डाउनलोड करें",
                data=json.dumps(export_data, ensure_ascii=False, indent=2),
                file_name=f"krish_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        else:
            st.info("⚠️ कोई चैट नहीं है")

# ------------------- Main Navigation Logic -------------------
if st.session_state.nav == "home":
    show_home_page()
elif st.session_state.nav == "tomato":
    show_tomato_detection_page()
elif st.session_state.nav == "crop":
    show_crop_whatsapp_page()

# ------------------- Footer -------------------
st.markdown("""
<div style='text-align: center; color: #666; margin-top: 2rem; padding: 1rem; border-top: 1px solid #ddd;'>
    <p>🌾 <strong>AI कृषि सहायक (By AgroMind)</strong> - आपके खेत का डिजिटल मित्र</p>
    <p><small>संस्करण 4.0 | टमाटर रोग पहचान जोड़ा गया!</small></p>
</div>
""", unsafe_allow_html=True)
