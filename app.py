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

st.title("🌿 Plant Leaf Disease Detection")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras", compile=False)

model = load_model()

# ---------------- CLASS NAMES ----------------
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

CONFIDENCE_THRESHOLD = 0.60

# ---------------- INPUT METHODS ----------------
st.subheader("Upload or Paste a Leaf Image")

uploaded_file = st.file_uploader(
    "Upload / Drag & Drop / Paste image here",
    type=["jpg", "jpeg", "png"]
)

camera_image = st.camera_input("Or capture image using camera")

# Decide input source
image = None
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
elif camera_image:
    image = Image.open(camera_image).convert("RGB")

# ---------------- PREDICTION ----------------
if image is not None:
    st.image(image, caption="Input Image", use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    confidence = float(np.max(preds))
    index = int(np.argmax(preds))

    if confidence < CONFIDENCE_THRESHOLD:
        st.error("❌ Image not recognized as a plant leaf")
        st.info("Please upload a clear leaf image (Apple, Corn, Potato, Tomato).")
    else:
        label = CLASS_NAMES[index]
        crop, disease = label.split("___")

        st.success(f"🌱 Crop: **{crop.replace('_',' ')}**")
        st.warning(f"🦠 Disease: **{disease.replace('_',' ')}**")
        st.info(f"📊 Confidence: **{confidence*100:.2f}%**")
        st.progress(confidence)

        if "healthy" in disease.lower():
            st.success("✅ Plant is healthy")
        else:
            st.write("💊 Suggested actions:")
            st.write("- Remove infected leaves")
            st.write("- Apply suitable fungicide")
            st.write("- Avoid overwatering")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("AI-Based Plant Leaf Disease Detection | Streamlit")
