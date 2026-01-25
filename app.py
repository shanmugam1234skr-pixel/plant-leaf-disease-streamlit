import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Plant Leaf Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# ---------------- CONSTANTS ----------------
MODEL_PATH = "model.keras"
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.60

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# ---------------- CLASS NAMES ----------------
# MUST match training order
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

# ---------------- TREATMENTS ----------------
TREATMENTS = {
    "Apple Black Rot": "Remove infected fruits and branches. Apply recommended fungicides.",
    "Apple Healthy": "No treatment required. Maintain good orchard hygiene.",

    "Corn Cercospora Leaf Spot": "Use resistant varieties and apply fungicides if needed.",
    "Corn Healthy": "No treatment required. Maintain proper nutrition.",

    "Potato Early Blight": "Apply fungicides like mancozeb. Avoid overhead irrigation.",
    "Potato Late Blight": "Remove infected plants immediately and apply fungicides.",
    "Potato Healthy": "No treatment required. Monitor crop regularly.",

    "Tomato Early Blight": "Remove affected leaves and apply fungicide.",
    "Tomato Late Blight": "Destroy infected plants and apply fungicide early.",
    "Tomato Leaf Mold": "Reduce humidity and improve air circulation.",
    "Tomato Septoria Leaf Spot": "Remove infected leaves and apply fungicide.",
    "Tomato Spider Mites": "Use neem oil or insecticidal soap.",
    "Tomato Target Spot": "Remove plant debris and apply fungicide.",
    "Tomato Yellow Leaf Curl Virus": "Control whiteflies and remove infected plants.",
    "Tomato Mosaic Virus": "Remove infected plants and disinfect tools.",
    "Tomato Healthy": "No treatment required. Maintain proper watering."
}

# ---------------- UI ----------------
st.title("🌿 Plant Leaf Disease Detection")

st.warning(
    "⚠️ This application supports ONLY Apple, Corn, Potato, and Tomato leaf images."
)

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- PREDICTION ----------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    confidence = float(np.max(preds))
    index = int(np.argmax(preds))

    predicted_disease = CLASS_NAMES[index]
    crop = predicted_disease.split()[0]

    st.success(f"🦠 Predicted Disease: **{predicted_disease}**")
    st.metric("Confidence", f"{confidence * 100:.2f}%")

    if confidence < CONFIDENCE_THRESHOLD:
        st.warning(
            "⚠️ Low confidence prediction. "
            "The image may not belong to the trained dataset."
        )

    st.markdown("### 💊 Treatment & Prevention")
    st.info(TREATMENTS[predicted_disease])

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption(
    "Note: This system performs closed-set classification using the PlantVillage dataset. "
    "Predictions for images outside the training distribution may be inaccurate."
)

