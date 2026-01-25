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

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust",
    "Corn_(maize)___healthy",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus"
]

ALLOWED_CROPS = ["Apple", "Corn", "Potato", "Tomato"]
CONFIDENCE_THRESHOLD = 70.0

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# ---------------- UI ----------------
st.title("🌿 Plant Leaf Disease Detection")
st.write("Upload a **leaf image** from Apple, Corn, Potato, or Tomato crops.")

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- PREDICTION ----------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input Image", use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ---------- BASIC LEAF CHECK ----------
    green_ratio = np.mean(img_array[..., 1])
    brightness = np.mean(img_array)

    if green_ratio < 0.25 or brightness < 0.15 or brightness > 0.9:
        st.error("❌ This does not appear to be a plant leaf.")
        st.info("Please upload a clear green leaf image.")
        st.stop()

    preds = model.predict(img_array)
    index = int(np.argmax(preds))
    confidence = float(np.max(preds)) * 100

    label = CLASS_NAMES[index]
    crop, disease = label.split("___")

    crop = crop.replace("(maize)", "").replace("_", " ").strip()
    disease = disease.replace("_", " ").strip()

    # ---------- UNSUPPORTED CROP ----------
    if crop not in ALLOWED_CROPS:
        st.error("❌ Unsupported crop detected")
        st.info(
            "This model supports only Apple, Corn, Potato, and Tomato leaves.\n\n"
            "The uploaded image seems to be from an unknown crop."
        )
        st.stop()

    # ---------- LOW CONFIDENCE ----------
    if confidence < CONFIDENCE_THRESHOLD:
        st.warning("⚠️ Low confidence prediction")
        st.info("Please upload a clearer leaf image.")
        st.stop()

    # ---------- RESULT ----------
    st.success(f"🌱 Crop: **{crop}**")
    st.success(f"🦠 Disease: **{disease}**")
    st.info(f"📊 Confidence: **{confidence:.2f}%**")

    if disease.lower() == "healthy":
        st.success("✅ The leaf appears healthy.")
    else:
        st.warning(
            "⚠️ This is an AI-based prediction.\n\n"
            "For accurate treatment, consult an agricultural expert."
        )

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption(
    "This application uses a deep learning model trained on the PlantVillage dataset. "
    "Predictions may be unreliable for images outside the training distribution."
)
