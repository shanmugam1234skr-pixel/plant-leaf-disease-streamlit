import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Plant Leaf Disease Detection", layout="centered")

st.title("🌿 Plant Leaf Disease Detection")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras", compile=False)

model = load_model()

class_names = [
    "Apple___Black_rot",
    "Apple___healthy",
    "Corn___Cercospora_leaf_spot",
    "Corn___healthy",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Early_blight",
    "Tomato___healthy"
]

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.success(f"🌱 Disease: **{class_names[index]}**")
    st.info(f"📊 Confidence: **{confidence:.2f}%**")
