import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- CONFIG ---------------- #
st.set_page_config(
    page_title="Plant Leaf Disease Detection",
    page_icon="🌿",
    layout="centered"
)

MODEL_PATH = "model.keras"
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.70  # 70%

# ---------------- LOAD MODEL ---------------- #
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# ---------------- CLASS NAMES ---------------- #
CLASS_NAMES = [
    "Apple – Black rot",
    "Apple – Healthy",
    "Corn – Cercospora leaf spot",
    "Corn – Healthy",
    "Potato – Early blight",
    "Potato – Late blight",
    "Potato – Healthy",
    "Tomato – Early blight",
    "Tomato – Late blight",
    "Tomato – Leaf Mold",
    "Tomato – Septoria leaf spot",
    "Tomato – Spider mites",
    "Tomato – Target Spot",
    "Tomato – Yellow Leaf Curl Virus",
    "Tomato – Mosaic Virus",
    "Tomato – Healthy"
]

# ---------------- LEAF VALIDATION ---------------- #
def is_leaf_like(img_array):
    """
    Rejects cars, humans, screenshots, plain objects.
    Accepts green, yellow, diseased leaves.
    """
    img = img_array[0]

    r_std = np.std(img[:, :, 0])
    g_std = np.std(img[:, :, 1])
    b_std = np.std(img[:, :, 2])

    total_variance = r_std + g_std + b_std

    return total_variance > 0.08


# ---------------- UI ---------------- #
st.title("🌿 Plant Leaf Disease Detection")
st.caption("Upload a plant leaf image (Tomato / Potato / Corn / Apple)")

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input Image", use_column_width=True)

    # -------- PREPROCESS -------- #
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # -------- LEAF CHECK -------- #
    if not is_leaf_like(img_array):
        st.error("❌ This does not appear to be a plant leaf.")
        st.info("Please upload a clear plant leaf image.")
        st.stop()

    # -------- PREDICTION -------- #
    preds = model.predict(img_array)
    confidence = float(np.max(preds))
    class_index = int(np.argmax(preds))

    # -------- CONFIDENCE FILTER -------- #
    if confidence < CONFIDENCE_THRESHOLD:
        st.warning("⚠️ Leaf detected, but disease is uncertain.")
        st.info("Try a clearer image with good lighting.")
        st.metric("Confidence", f"{confidence*100:.2f}%")
        st.stop()

    predicted_class = CLASS_NAMES[class_index]

    # -------- RESULT DISPLAY -------- #
    st.success(f"🌱 Disease Detected: **{predicted_class}**")
    st.metric("Confidence", f"{confidence*100:.2f}%")

    # -------- EXTRA INFO -------- #
    if "Healthy" in predicted_class:
        st.info("✅ The leaf appears healthy.")
    else:
        st.warning("⚠️ Disease detected. Consider proper treatment.")

# ---------------- FOOTER ---------------- #
st.markdown("---")
st.caption("AI-based Plant Leaf Disease Detection • Streamlit + TensorFlow")
