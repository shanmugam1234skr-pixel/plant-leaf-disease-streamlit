import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Plant Disease Detection",
    page_icon="🌱",
    layout="centered"
)

MODEL_PATH = "model_v2_with_non_leaf.keras"
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 60  # %

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ---------------- CLASSES ----------------
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "NON_LEAF",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# ---------------- TREATMENTS ----------------
TREATMENTS = {
    "Apple___Apple_scab": "Apply fungicide and remove fallen leaves.",
    "Apple___Black_rot": "Prune infected branches and use copper spray.",
    "Corn_(maize)___Northern_Leaf_Blight": "Crop rotation and resistant varieties.",
    "Potato___Early_blight": "Avoid overhead irrigation and use fungicide.",
    "Potato___Late_blight": "Destroy infected plants and apply fungicide.",
    "Tomato___Early_blight": "Remove infected leaves and spray fungicide.",
    "Tomato___Late_blight": "Avoid moisture and spray immediately.",
    "Tomato___Bacterial_spot": "Use disease-free seeds and copper spray.",
    "Tomato___Leaf_Mold": "Improve ventilation and apply fungicide.",
    "Tomato___Septoria_leaf_spot": "Remove infected leaves and avoid wet foliage.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Use neem oil or insecticidal soap.",
    "Tomato___Target_Spot": "Crop rotation and fungicide spray.",
    "Tomato___Tomato_mosaic_virus": "Remove infected plants and disinfect tools.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control whiteflies and remove infected plants."
}

# ---------------- LANGUAGE UI ----------------
LANG = st.selectbox(
    "🌐 Select Language / மொழி / भाषा / భాష",
    ["English", "Tamil", "Hindi", "Telugu"]
)

TEXT = {
    "title": {
        "English": "🌱 AI Plant Leaf Disease Detection",
        "Tamil": "🌱 தாவர இலை நோய் கண்டறிதல்",
        "Hindi": "🌱 पौधों की पत्ती रोग पहचान",
        "Telugu": "🌱 మొక్కల ఆకు వ్యాధి గుర్తింపు"
    },
    "upload": {
        "English": "Upload a leaf image",
        "Tamil": "இலைப் படத்தை பதிவேற்றவும்",
        "Hindi": "पत्ती की छवि अपलोड करें",
        "Telugu": "ఆకు చిత్రాన్ని అప్లోడ్ చేయండి"
    },
    "not_leaf": {
        "English": "🚫 This image is NOT a plant leaf",
        "Tamil": "🚫 இது தாவர இலை அல்ல",
        "Hindi": "🚫 यह पौधे की पत्ती नहीं है",
        "Telugu": "🚫 ఇది మొక్క ఆకు కాదు"
    },
    "healthy": {
        "English": "✅ Leaf is HEALTHY",
        "Tamil": "✅ இலை ஆரோக்கியமாக உள்ளது",
        "Hindi": "✅ पत्ती स्वस्थ है",
        "Telugu": "✅ ఆకు ఆరోగ్యంగా ఉంది"
    }
}

# ---------------- UI ----------------
st.title(TEXT["title"][LANG])
st.write("🍎 Apple | 🌽 Corn | 🥔 Potato | 🍅 Tomato")

file = st.file_uploader(TEXT["upload"][LANG], type=["jpg", "jpeg", "png"])

# ---------------- PREDICTION ----------------
if file:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    idx = np.argmax(preds)
    confidence = np.max(preds) * 100
    predicted = CLASS_NAMES[idx]

    if confidence < CONFIDENCE_THRESHOLD:
        st.warning("⚠️ Low confidence. Please upload a clear leaf image.")

    elif predicted == "NON_LEAF":
        st.error(TEXT["not_leaf"][LANG])

    else:
        crop = predicted.split("___")[0]
        st.success(f"🌿 Crop: {crop}")

        if "healthy" in predicted.lower():
            st.success(TEXT["healthy"][LANG])
        else:
            disease = predicted.split("___")[1]
            st.error(f"🦠 Disease: {disease}")
            st.info(f"💊 Treatment: {TREATMENTS.get(predicted, 'General care recommended')}")

        st.info(f"📊 Confidence: {confidence:.2f}%")

st.markdown("---")
st.caption("🔬 AI-powered commercial plant disease prediction system")

st.caption(t["disclaimer"])
st.caption("Commercial AI Demo • Streamlit + TensorFlow")

