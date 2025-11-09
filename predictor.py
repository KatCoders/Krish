# predictor.py

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load labels matching your model
CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_(Two-spotted_spider_mite)",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# Cache the model so it's loaded once
_loaded_model = None

def load_tomato_model():
    global _loaded_model
    if _loaded_model is None:
        _loaded_model = load_model("tomato_model.h5", compile=False)
    return _loaded_model

def predict_disease(img: Image.Image):
    """Predict disease from a PIL image."""
    model = load_tomato_model()
    img = img.resize((224, 224))  # Adjust size if needed
    img_array = preprocess_input(image.img_to_array(img))
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_idx = np.argmax(predictions[0])
    return CLASS_NAMES[predicted_idx], float(np.max(predictions[0]))
