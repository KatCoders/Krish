import os
from typing import Dict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool

from data_sources import (
    fetch_market_data,
    format_price_response,
    fetch_weather,
    fetch_soil,
    get_crop_prediction,
)
from utils import extract_crop_query, translate_hindi_to_english

# Load environment
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
import json


import numpy as np

# Loads a list of dicts: [{title, content, embedding}, ...]
TOMATO_KB = np.load("tomato_embeddings.npy", allow_pickle=True).tolist()

def keyword_search(query, knowledge_base, top_k=2):
    results = []
    query = query.lower()
    query_words = query.split()

    for item in knowledge_base:
        text = f"{item.get('title', '')} {item.get('content', '')}".lower()
        score = sum(word in text for word in query_words)
        if score > 0:
            results.append((score, item))

    results.sort(reverse=True, key=lambda x: x[0])
    return [item for score, item in results][:top_k]

@tool("search_tomato_kb")
def tool_search_tomato_kb(query: str) -> str:
    """Search scientific tomato farming data using semantic similarity."""
    results = keyword_search(query, TOMATO_KB)
    if not results:
        return "क्षमा करें, इस विषय पर कोई जानकारी नहीं मिली।"
    return "\n\n".join(f"📘 {item['title']}:\n{item['content']}" for item in results)

# ----- Tool Definitions with Proper Docstrings -----
@tool("market_price")
def tool_market_price(query: str) -> str:
    """Fetch current mandi price for a given crop and state in India."""
    qp = extract_crop_query(query)
    if not qp.get("crop") or not qp.get("state"):
        return "कृपया फसल और राज्य बताएं। उदाहरण: 'मध्य प्रदेश में टमाटर का भाव'"
    crop_en = translate_hindi_to_english(qp["crop"])
    data = fetch_market_data(qp["state"], crop_en)
    return format_price_response(data)


@tool("weather_info")
def tool_weather_info(query: str) -> str:
    """Get live weather information for the detected or default location."""
    qp = extract_crop_query(query)
    data = fetch_weather(qp.get("state") or "Indore, India")
    return (
        f"🌤️ तापमान: {data['temperature']}°C\n"
        f"💧 आर्द्रता: {data['humidity']}%\n"
        f"☁️ मौसम: {data['condition']}"
    )


@tool("soil_info")
def tool_soil_info(query: str) -> str:
    """Provide soil information like pH and nitrogen for the region."""
    soil = fetch_soil()
    return f"🧪 pH: {soil['ph']}\n🟢 नाइट्रोजन: {soil['nitrogen']}"


@tool("predict_crop")
def tool_predict_crop(query: str) -> str:
    """Recommend the best crop to grow based on weather and soil conditions."""
    soil = fetch_soil()
    weather = fetch_weather("Indore, India")
    crop, conf = get_crop_prediction(soil, weather)
    return f"सुझाई गई फसल: {crop} (विश्वास: {conf:.1f}%)"


# ----- Agent Setup -----
def get_agent():
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing. Please set it in the .env file.")
    
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=GROQ_API_KEY
    )
    
    tools = [
        tool_market_price, 
        tool_weather_info, 
        tool_soil_info, 
        tool_predict_crop,
        
    ]
    
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
"""
आपका नाम कृष है। आप एक अनुभवी भारतीय कृषि विशेषज्ञ AI हैं।
if tomato image uploade use function  tool_predict_crop then based on its output answer in hindi cure and precaution
निम्न टूल्स का उपयोग करें:
- मंडी, मौसम, मिट्टी संबंधित के लिए उनके उचित टूल्स
 से खोज करके जवाब दें।
सभी जवाब सरल और हिंदी में दें।
"""
        )
    )


# ----- Main LLM Response Handler -----
def get_llm_response(query: str, tool_name=None) -> str:
    try:
        agent = get_agent()
        result = agent.invoke({"messages": [{"role": "user", "content": query}]})

        # ✅ Case 1: LangChain-like dict response
        if isinstance(result, dict) and "messages" in result:
            messages = result["messages"]
            if isinstance(messages, list) and len(messages) > 0:
                last_msg = messages[-1]
                if hasattr(last_msg, "content"):
                    return str(last_msg.content)
                elif isinstance(last_msg, dict) and last_msg.get("content"):
                    return last_msg["content"]

        # ✅ Case 2: result is list (rare in Groq agents)
        if isinstance(result, list):
            # Flatten to string
            combined = " ".join(
                [
                    msg.content if hasattr(msg, "content") else str(msg)
                    for msg in result
                ]
            )
            return combined

        # ✅ Case 3: it's already a clean string
        if isinstance(result, str):
            return result

        # ✅ Catch-all for unexpected result types
        return str(result)

    except Exception as e:
        return f"🤖 क्षमा करें! तकनीकी समस्या के कारण जवाब तैयार नहीं हो सका। ({str(e)})"


