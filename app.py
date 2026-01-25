import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras", compile=False)

model = load_model()

st.title("🌿 Plant Leaf Disease Detection")

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    class_index = np.argmax(preds)
    confidence = np.max(preds) * 100

    st.success(f"Prediction index: {class_index}")
    st.info(f"Confidence: {confidence:.2f}%")

