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

CONFIDENCE_THRESHOLD = 75.0
ENTROPY_THRESHOLD = 2.2   # higher = more confusion

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# ---------------- HELPER FUNCTIONS ----------------
def softmax_entropy(probs):
    return -np.sum(probs * np.log(probs + 1e-10))

def is_leaf_like(img_array):
    """
    Simple heuristic:
    - leaf images have a strong green channel
    - reject cars, humans, animals, screens
    """
    red = np.mean(img_array[..., 0])
    green = np.mean(img_array[..., 1])
    blue = np.mean(img_array[..., 2])

    green_ratio = green / (red + blue + 1e-6)

    return green_ratio > 0.9

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

    # ---------- REJECTION 1: LEAF CHECK ----------
    if not is_leaf_like(img_array):
        st.error("❌ This image does not appear to be a plant leaf.")
        st.info("Please upload a clear green leaf image.")
        st.stop()

    # ---------- MODEL PREDICTION ----------
    preds = model.predict(img_array)[0]
    index = int(np.argmax(preds))
    confidence = float(np.max(preds)) * 100
    entropy = softmax_entropy(preds)

    label = CLASS_NAMES[index]
    crop, disease = label.split("___")

    crop = crop.replace("(maize)", "").replace("_", " ").strip()
    disease = disease.replace("_", " ").strip()

    # ---------- REJECTION 2: CONFIDENCE ----------
    if confidence < CONFIDENCE_THRESHOLD:
        st.error("❌ Low confidence prediction")
        st.info("This image may not belong to the trained plant classes.")
        st.stop()

    # ---------- REJECTION 3: ENTROPY ----------
    if entropy > ENTROPY_THRESHOLD:
        st.error("❌ The model is unsure about this image")
        st.info("This image may not be a valid plant leaf.")
        st.stop()

    # ---------- REJECTION 4: UNSUPPORTED CROP ----------
    if crop not in ALLOWED_CROPS:
        st.error("❌ Unsupported crop detected")
        st.info("Only Apple, Corn, Potato, and Tomato leaves are supported.")
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
    "Predictions are limited to the PlantVillage dataset. "
    "Images outside this domain are automatically rejected."
)
