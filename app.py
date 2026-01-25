import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Plant Leaf Disease Detection",
    page_icon="🌿",
    layout="centered"
)

IMG_SIZE = 224
MODEL_PATH = "model.keras"
CONFIDENCE_THRESHOLD = 0.6

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# ---------------- CLASS NAMES ----------------
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
    "Apple Black Rot": "Remove infected fruits and branches. Apply fungicide.",
    "Apple Healthy": "No treatment required.",
    "Corn Cercospora Leaf Spot": "Use resistant varieties and fungicides.",
    "Corn Healthy": "No treatment required.",
    "Potato Early Blight": "Apply mancozeb or chlorothalonil.",
    "Potato Late Blight": "Remove infected plants and apply fungicides.",
    "Potato Healthy": "No treatment required.",
    "Tomato Early Blight": "Remove infected leaves and apply fungicide.",
    "Tomato Late Blight": "Destroy infected plants and apply fungicide.",
    "Tomato Leaf Mold": "Reduce humidity and apply fungicide.",
    "Tomato Septoria Leaf Spot": "Remove infected leaves and apply fungicide.",
    "Tomato Spider Mites": "Use neem oil or insecticidal soap.",
    "Tomato Target Spot": "Remove debris and apply fungicide.",
    "Tomato Yellow Leaf Curl Virus": "Control whiteflies and remove infected plants.",
    "Tomato Mosaic Virus": "Remove infected plants and disinfect tools.",
    "Tomato Healthy": "No treatment required."
}

# ---------------- UI ----------------
st.title("🌿 Plant Leaf Disease Detection")
st.caption("Upload a plant leaf image for disease prediction")

uploaded_file = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    confidence = float(np.max(preds))
    index = int(np.argmax(preds))

    disease = CLASS_NAMES[index]

    st.success(f"🦠 Predicted Disease: **{disease}**")
    st.metric("Confidence", f"{confidence*100:.2f}%")

    if confidence < CONFIDENCE_THRESHOLD:
        st.warning("⚠️ Low confidence prediction. Image may not belong to trained dataset.")

    st.markdown("### 💊 Treatment & Prevention")
    st.info(TREATMENTS[disease])

    st.caption(
        "Note: The model predicts only among trained classes. "
        "Predictions may be inaccurate for images outside the dataset."
    )
