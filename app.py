import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Plant Leaf Disease Detection",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Plant Leaf Disease Detection")

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras", compile=False)

model = load_model()

# -------------------- CLASS NAMES --------------------
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

# -------------------- CONFIDENCE THRESHOLD --------------------
CONFIDENCE_THRESHOLD = 0.60  # 60%

# -------------------- FILE UPLOADER --------------------
uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

# -------------------- PREDICTION --------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    confidence = float(np.max(predictions))
    index = int(np.argmax(predictions))

    # -------------------- UNKNOWN IMAGE HANDLING --------------------
    if confidence < CONFIDENCE_THRESHOLD:
        st.error("❌ Image not recognized as a valid plant leaf.")
        st.info("Please upload a clear leaf image from Apple, Corn, Potato, or Tomato plants.")

    else:
        label = CLASS_NAMES[index]
        crop, disease = label.split("___")

        crop = crop.replace("_", " ")
        disease = disease.replace("_", " ")

        st.success(f"🌱 Crop: **{crop}**")
        st.warning(f"🦠 Disease: **{disease}**")

        st.info(f"📊 Confidence: **{confidence * 100:.2f}%**")
        st.progress(confidence)

        # -------------------- HEALTH CHECK --------------------
        if "healthy" in disease.lower():
            st.success("✅ The plant is healthy. No treatment required.")
        else:
            st.write("💊 **Suggested Action:**")
            st.write("• Remove affected leaves")
            st.write("• Use recommended fungicide")
            st.write("• Avoid overwatering")
            st.write("• Consult an agricultural expert if symptoms persist")

# -------------------- FOOTER --------------------
st.markdown("---")
st.caption("AI-Based Plant Leaf Disease Detection | Streamlit + TensorFlow")
