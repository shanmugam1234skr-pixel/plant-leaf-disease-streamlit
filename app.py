import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Plant Leaf Disease Detection",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Plant Leaf Disease Detection")
st.write("Upload, drag-drop, paste, or capture a leaf image")

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras", compile=False)

model = load_model()

# -------------------------------------------------
# CLASS NAMES (PlantVillage – exact training order)
# -------------------------------------------------
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",

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

# -------------------------------------------------
# THRESHOLDS (IMPORTANT)
# -------------------------------------------------
CONFIDENCE_THRESHOLD = 0.80      # reject weak predictions
GAP_THRESHOLD = 0.15             # top-1 vs top-2 confidence gap

# -------------------------------------------------
# TREATMENT SUGGESTIONS
# -------------------------------------------------
TREATMENTS = {
    "Apple": "Prune infected leaves, apply recommended fungicide, ensure good air circulation.",
    "Corn": "Use resistant varieties, rotate crops, avoid overhead irrigation.",
    "Potato": "Apply fungicide early, remove infected plants, avoid excess moisture.",
    "Tomato": "Remove affected leaves, use neem oil or fungicide, maintain proper spacing."
}

# -------------------------------------------------
# INPUT METHODS
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload / Drag-Drop / Paste image",
    type=["jpg", "jpeg", "png"]
)

camera_image = st.camera_input("Or capture image using camera")

image = None
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
elif camera_image:
    image = Image.open(camera_image).convert("RGB")

# -------------------------------------------------
# PREDICTION LOGIC
# -------------------------------------------------
if image is not None:
    st.image(image, caption="Input Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Simple image sanity check
    mean_pixel = np.mean(img_array)
    if mean_pixel < 0.2 or mean_pixel > 0.9:
        st.error("❌ Image quality too low or not a leaf")
        st.stop()

    # Model prediction
    preds = model.predict(img_array)[0]

    top1 = float(np.max(preds))
    top2 = float(np.sort(preds)[-2])
    gap = top1 - top2
    index = int(np.argmax(preds))

    # -------------------------------------------------
    # UNKNOWN / WRONG IMAGE HANDLING
    # -------------------------------------------------
    if top1 < CONFIDENCE_THRESHOLD or gap < GAP_THRESHOLD:
        st.error("❌ Image not confidently recognized")
        st.info("Please upload a clear leaf image of Apple, Corn, Potato, or Tomato.")
    else:
        label = CLASS_NAMES[index]
        crop, disease = label.split("___")

        crop_clean = crop.replace("_", " ")
        disease_clean = disease.replace("_", " ")

        st.success(f"🌱 Crop: **{crop_clean}**")
        st.warning(f"🦠 Disease: **{disease_clean}**")

        st.info(f"📊 Confidence: **{top1 * 100:.2f}%**")
        st.progress(top1)

        if "healthy" in disease.lower():
            st.success("✅ The plant is healthy. No treatment required.")
        else:
            st.write("💊 **Suggested Treatment & Prevention:**")
            st.write(TREATMENTS.get(crop_clean.split()[0], "Consult an agricultural expert."))

        # -------------------------------------------------
        # TRANSPARENCY: TOP-3 PREDICTIONS
        # -------------------------------------------------
        st.markdown("### 🔍 Top-3 Predictions")
        top3_idx = preds.argsort()[-3:][::-1]
        for i in top3_idx:
            name = CLASS_NAMES[i].replace("___", " → ").replace("_", " ")
            st.write(f"- {name} : {preds[i]*100:.2f}%")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")
st.caption("AI-Based Plant Leaf Disease Detection | Streamlit + TensorFlow")

