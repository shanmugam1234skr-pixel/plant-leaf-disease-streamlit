import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Plant Leaf Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# ---------------- CONSTANTS ---------------- #
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.70
MODEL_PATH = "model.keras"

# ---------------- LOAD MODEL ---------------- #
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# ---------------- CLASS NAMES (ORDER MUST MATCH TRAINING) ---------------- #
CLASS_NAMES = [
    "Apple_Black_rot",
    "Apple_healthy",
    "Corn_Cercospora_leaf_spot",
    "Corn_healthy",
    "Potato_Early_blight",
    "Potato_Late_blight",
    "Potato_healthy",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites",
    "Tomato_Target_Spot",
    "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_Mosaic_Virus",
    "Tomato_healthy"
]

# ---------------- TREATMENT DICTIONARY ---------------- #
TREATMENTS = {
    "Apple_Black_rot": "Remove infected fruits and branches. Apply fungicides like captan. Maintain orchard hygiene.",
    "Apple_healthy": "No disease detected. Maintain proper irrigation and pruning.",
    
    "Corn_Cercospora_leaf_spot": "Use resistant varieties. Apply fungicides. Practice crop rotation.",
    "Corn_healthy": "Healthy leaf. Continue proper nutrient and water management.",
    
    "Potato_Early_blight": "Apply chlorothalonil or mancozeb fungicides. Avoid overhead irrigation.",
    "Potato_Late_blight": "Use certified seed potatoes. Apply systemic fungicides. Remove infected plants immediately.",
    "Potato_healthy": "Healthy crop. Maintain balanced fertilization.",
    
    "Tomato_Early_blight": "Apply fungicides regularly. Remove affected leaves. Use mulch to prevent soil splash.",
    "Tomato_Late_blight": "Destroy infected plants. Use resistant varieties. Apply fungicides early.",
    "Tomato_Leaf_Mold": "Improve ventilation. Reduce humidity. Apply copper-based fungicides.",
    "Tomato_Septoria_leaf_spot": "Remove infected leaves. Avoid wet foliage. Use fungicides.",
    "Tomato_Spider_mites": "Spray neem oil or insecticidal soap. Increase humidity.",
    "Tomato_Target_Spot": "Remove infected plant debris. Apply fungicides.",
    "Tomato_Yellow_Leaf_Curl_Virus": "Control whiteflies. Remove infected plants. Use resistant varieties.",
    "Tomato_Mosaic_Virus": "Remove infected plants. Disinfect tools. Avoid handling wet plants.",
    "Tomato_healthy": "Healthy leaf. Maintain proper watering and pest control."
}

# ---------------- LEAF VALIDATION ---------------- #
def is_leaf_like(img_array):
    img = img_array[0]
    variance = np.std(img[:, :, 0]) + np.std(img[:, :, 1]) + np.std(img[:, :, 2])
    return variance > 0.08

# ---------------- UI ---------------- #
st.title("🌿 Plant Leaf Disease Detection")
st.caption("Upload a clear plant leaf image")

uploaded_file = st.file_uploader(
    "Upload leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input Image", use_column_width=True)

    # -------- PREPROCESS -------- #
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # -------- CHECK IF LEAF -------- #
    if not is_leaf_like(img_array):
        st.error("❌ This image does not appear to be a plant leaf.")
        st.stop()

    # -------- PREDICTION -------- #
    predictions = model.predict(img_array)
    confidence = float(np.max(predictions))
    index = int(np.argmax(predictions))

    if confidence < CONFIDENCE_THRESHOLD:
        st.warning("⚠️ Unable to confidently identify the disease.")
        st.info("Please upload a clearer leaf image.")
        st.metric("Confidence", f"{confidence*100:.2f}%")
        st.stop()

    disease = CLASS_NAMES[index]
    crop = disease.split("_")[0]
    disease_name = disease.replace("_", " ")

    # -------- OUTPUT -------- #
    st.success(f"🌱 Crop: **{crop}**")
    st.warning(f"🦠 Disease: **{disease_name}**")
    st.metric("Confidence", f"{confidence*100:.2f}%")

    st.markdown("### 💊 Treatment & Prevention")
    st.info(TREATMENTS[disease])

# ---------------- FOOTER ---------------- #
st.markdown("---")
st.caption("AI-Based Plant Leaf Disease Detection | Streamlit + TensorFlow")

# ---------------- FOOTER ---------------- #
st.markdown("---")
st.caption("AI-based Plant Leaf Disease Detection • Streamlit + TensorFlow")

