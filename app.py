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
CONFIDENCE_THRESHOLD = 0.60

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
    "Apple Black Rot": "Remove infected fruits and branches. Apply recommended fungicides.",
    "Apple Healthy": "No treatment required. Maintain good orchard hygiene.",

    "Corn Cercospora Leaf Spot": "Use resistant varieties and apply fungicides if required.",
    "Corn Healthy": "Healthy crop. Maintain proper nutrient management.",

    "Potato Early Blight": "Apply mancozeb or chlorothalonil. Avoid overhead irrigation.",
    "Potato Late Blight": "Remove infected plants immediately and apply fungicides.",
    "Potato Healthy": "Healthy crop. Regular monitoring is sufficient.",

    "Tomato Early Blight": "Remove affected leaves and apply fungicide.",
    "Tomato Late Blight": "Destroy infected plants and apply fungicides early.",
    "Tomato Leaf Mold": "Reduce humidity and improve air circulation.",
    "Tomato Septoria Leaf Spot": "Remove infected leaves and apply fungicide.",
    "Tomato Spider Mites": "Use neem oil or insecticidal soap.",
    "Tomato Target Spot": "Remove plant debris and apply fungicide.",
    "Tomato Yellow Leaf Curl Virus": "Control whiteflies and remove infected plants.",
    "Tomato Mosaic Virus": "Remove infected plants and disinfect tools.",
    "Tomato Healthy": "Healthy leaf. Maintain proper irrigation."
}

# ================== MULTI-LANGUAGE TEXT ==================
TEXT = {
    "English": {
        "title": "AI Plant Leaf Disease Detection",
        "upload": "Upload a leaf image",
        "warning": "Supports only Apple, Corn, Potato, and Tomato leaves",
        "analyzing": "AI is analyzing the image...",
        "confidence": "Confidence",
        "treatment": "Treatment & Prevention",
        "low_conf": "Low confidence prediction. Image may be unclear or outside dataset.",
        "disclaimer": "This AI system predicts diseases only from trained crops. Results are advisory."
    },
    "Tamil": {
        "title": "ஏ.ஐ. தாவர இலை நோய் கண்டறிதல்",
        "upload": "இலை படத்தை பதிவேற்றவும்",
        "warning": "ஆப்பிள், சோளம், உருளைக்கிழங்கு மற்றும் தக்காளி இலைகளுக்கு மட்டும்",
        "analyzing": "ஏ.ஐ. படம் பகுப்பாய்வு செய்கிறது...",
        "confidence": "நம்பகத்தன்மை",
        "treatment": "சிகிச்சை மற்றும் தடுப்பு",
        "low_conf": "குறைந்த நம்பகத்தன்மை. தெளிவான படத்தை பதிவேற்றவும்.",
        "disclaimer": "இந்த ஏ.ஐ. பயிற்சி பெற்ற பயிர்களுக்கு மட்டும் பொருந்தும்."
    },
    "Hindi": {
        "title": "एआई पौधा पत्ती रोग पहचान",
        "upload": "पत्ती की छवि अपलोड करें",
        "warning": "केवल सेब, मक्का, आलू और टमाटर पत्तियाँ समर्थित हैं",
        "analyzing": "एआई छवि का विश्लेषण कर रहा है...",
        "confidence": "विश्वसनीयता",
        "treatment": "उपचार और रोकथाम",
        "low_conf": "कम विश्वसनीयता। कृपया स्पष्ट छवि अपलोड करें।",
        "disclaimer": "यह एआई केवल प्रशिक्षित फसलों के लिए मान्य है।"
    }
}

# ================== LANGUAGE SELECT ==================
language = st.selectbox("🌐 Select Language / மொழி / भाषा", ["English", "Tamil", "Hindi"])
t = TEXT[language]

# ================== UI ==================
st.title(f"🌿 {t['title']}")

st.warning(f"⚠️ {t['warning']}")

uploaded_file = st.file_uploader(
    t["upload"],
    type=["jpg", "jpeg", "png"]
)

# ================== PREDICTION ==================
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner(t["analyzing"]):
        img = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)
        confidence = float(np.max(preds))
        index = int(np.argmax(preds))

    disease = CLASS_NAMES[index]
    crop = disease.split()[0]

    st.success(f"🦠 {disease}")
    st.progress(confidence)
    st.metric(t["confidence"], f"{confidence*100:.2f}%")

    if confidence < CONFIDENCE_THRESHOLD:
        st.warning(t["low_conf"])

    st.markdown(f"### 💊 {t['treatment']}")
    st.info(TREATMENTS[disease])

# ================== FOOTER ==================
st.markdown("---")
st.caption(t["disclaimer"])
st.caption("Commercial AI Demo • Streamlit + TensorFlow")
