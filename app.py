import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="AI Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# ================== CONSTANTS ==================
MODEL_PATH = "model.keras"
IMG_SIZE = 224

# ================== LOAD MODEL ==================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# ================== CLASS NAMES ==================
CLASS_NAMES = [
    "Apple Black Rot",
    "Apple Healthy",
    "Corn Cercospora Leaf Spot",
    "Corn Healthy",
    "Potato Early Blight",
    "Potato Late Blight",
    "Potato Healthy",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Septoria Leaf Spot",
    "Tomato Spider Mites",
    "Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato Mosaic Virus",
    "Tomato Healthy"
]

# ================== TREATMENTS ==================
TREATMENTS = {
    "Apple Black Rot": "Remove infected fruits and branches. Apply fungicide.",
    "Corn Cercospora Leaf Spot": "Use resistant varieties and fungicides.",
    "Potato Early Blight": "Apply mancozeb or chlorothalonil.",
    "Potato Late Blight": "Remove infected plants immediately.",
    "Tomato Early Blight": "Remove affected leaves and apply fungicide.",
    "Tomato Late Blight": "Destroy infected plants and apply fungicide.",
    "Tomato Leaf Mold": "Reduce humidity and improve ventilation.",
    "Tomato Septoria Leaf Spot": "Remove infected leaves and apply fungicide.",
    "Tomato Spider Mites": "Use neem oil or insecticidal soap.",
    "Tomato Target Spot": "Remove debris and apply fungicide.",
    "Tomato Yellow Leaf Curl Virus": "Control whiteflies and remove infected plants.",
    "Tomato Mosaic Virus": "Remove infected plants and disinfect tools."
}

# ================== MULTI-LANGUAGE UI ==================
TEXT = {
    "English": {
        "title": "AI Plant Leaf Disease Detection",
        "warning": "Supports only Apple, Corn, Potato, and Tomato leaves",
        "upload": "Upload a leaf image",
        "status": "Select leaf condition",
        "healthy": "Healthy leaf",
        "diseased": "Diseased leaf",
        "analyzing": "AI is analyzing the image...",
        "confidence": "Confidence",
        "treatment": "Treatment & Prevention",
        "healthy_msg": "Leaf is healthy. No disease detected.",
        "disclaimer": "AI predictions are advisory and based on trained crops only."
    },
    "Tamil": {
        "title": "ஏ.ஐ. தாவர இலை நோய் கண்டறிதல்",
        "warning": "ஆப்பிள், சோளம், உருளைக்கிழங்கு மற்றும் தக்காளி இலைகளுக்கு மட்டும்",
        "upload": "இலை படத்தை பதிவேற்றவும்",
        "status": "இலை நிலையை தேர்வு செய்யவும்",
        "healthy": "ஆரோக்கியமான இலை",
        "diseased": "நோயுற்ற இலை",
        "analyzing": "ஏ.ஐ. படம் பகுப்பாய்வு செய்கிறது...",
        "confidence": "நம்பகத்தன்மை",
        "treatment": "சிகிச்சை மற்றும் தடுப்பு",
        "healthy_msg": "இலை ஆரோக்கியமாக உள்ளது.",
        "disclaimer": "ஏ.ஐ. கணிப்புகள் வழிகாட்டுதலுக்காக மட்டுமே."
    },
    "Hindi": {
        "title": "एआई पौधा पत्ती रोग पहचान",
        "warning": "केवल सेब, मक्का, आलू और टमाटर पत्तियाँ समर्थित हैं",
        "upload": "पत्ती की छवि अपलोड करें",
        "status": "पत्ती की स्थिति चुनें",
        "healthy": "स्वस्थ पत्ती",
        "diseased": "बीमार पत्ती",
        "analyzing": "एआई छवि का विश्लेषण कर रहा है...",
        "confidence": "विश्वसनीयता",
        "treatment": "उपचार और रोकथाम",
        "healthy_msg": "पत्ती स्वस्थ है।",
        "disclaimer": "एआई परिणाम केवल सलाह के लिए हैं।"
    }
}

# ================== LANGUAGE SELECT ==================
language = st.selectbox("🌐 Language / மொழி / भाषा", ["English", "Tamil", "Hindi"])
t = TEXT[language]

# ================== UI ==================
st.title(f"🌿 {t['title']}")
st.warning(f"⚠️ {t['warning']}")

leaf_status = st.radio(
    t["status"],
    [t["healthy"], t["diseased"]]
)

uploaded_file = st.file_uploader(
    t["upload"],
    type=["jpg", "jpeg", "png"]
)

# ================== LOGIC ==================
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ✅ HEALTHY LEAF (HUMAN-IN-LOOP)
    if leaf_status == t["healthy"]:
        st.success(t["healthy_msg"])
        st.caption("✔ Verified by user input")
        st.stop()

    # 🔍 DISEASE PREDICTION
    with st.spinner(t["analyzing"]):
        img = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)
        index = int(np.argmax(preds))
        confidence = float(np.max(preds))

    disease = CLASS_NAMES[index]

    st.success(f"🦠 {disease}")
    st.progress(confidence)
    st.metric(t["confidence"], f"{confidence*100:.2f}%")

    st.markdown(f"### 💊 {t['treatment']}")
    st.info(TREATMENTS.get(disease, "Consult agricultural expert."))

# ================== FOOTER ==================
st.markdown("---")
st.caption(t["disclaimer"])
st.caption("Commercial AI Demo • Streamlit + TensorFlow")
