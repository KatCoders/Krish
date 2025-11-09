from dotenv import load_dotenv
import os

from openai import embeddings

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")




def extract_crop_query(text: str) -> dict:
    out = {"crop": None, "state": None}
    crops = ["टमाटर", "प्याज", "गेहूं", "धान", "मक्का", "सोयाबीन"]
    states = ["Madhya Pradesh", "Gujarat", "Karnataka", "Rajasthan"]

    for c in crops:
        if c in text:
            out["crop"] = c
            break
    for s in states:
        if s.lower() in text.lower():
            out["state"] = s
            break
    return out

def translate_hindi_to_english(crop: str) -> str:
    mapping = {
        "गेहूं": "Wheat",
        "धान": "Rice",
        "मक्का": "Maize",
        "टमाटर": "Tomato",
        "प्याज": "Onion"
    }
    return mapping.get(crop, crop)
