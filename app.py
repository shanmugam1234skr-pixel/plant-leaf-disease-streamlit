import os
import uuid
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import requests
from datetime import datetime
from supabase import create_client

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Plant Leaf Disease Detection", layout="wide")

st.markdown("""
<style>
html, body { overflow-x: hidden; }
.block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "model_v2_with_non_leaf.keras"
IMG_SIZE = (224, 224)
CONF_THRESHOLD = 0.70

# ---------------- SUPABASE ----------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------- AUTH ----------------
st.sidebar.title("🔐 Login / Signup")

mode = st.sidebar.selectbox("Mode", ["Login", "Signup"])
email = st.sidebar.text_input("Email")
password = st.sidebar.text_input("Password", type="password")

if st.sidebar.button("Submit"):
    try:
        if mode == "Signup":
            supabase.auth.sign_up({
                "email": email,
                "password": password
            })
            st.sidebar.success("Signup successful. Please login.")
        else:
            response = supabase.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            st.session_state["access_token"] = response.session.access_token
            st.session_state["user_email"] = response.user.email
            st.sidebar.success("Login successful.")
    except Exception as e:
        st.sidebar.error(f"Auth Error: {e}")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found.")
    st.stop()

model = load_model()

# ---------------- CLASS NAMES ----------------
CLASS_NAMES = [
    "Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust",
    "Apple___healthy","Corn_(maize)___Cercospora_leaf_spot",
    "Corn_(maize)___Common_rust","Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy","NON_LEAF","Potato___Early_blight",
    "Potato___Late_blight","Potato___healthy","Tomato___Bacterial_spot",
    "Tomato___Early_blight","Tomato___Late_blight","Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot","Tomato___Spider_mites",
    "Tomato___Target_Spot","Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus","Tomato___healthy"
]

# ---------------- MAIN UI ----------------
st.title("🌿 Plant Leaf Disease Detection")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg","jpeg","png"])

def preprocess(img):
    img = img.resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

if uploaded_file:

    if "access_token" not in st.session_state:
        st.warning("Please login first.")
        st.stop()

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width=400)

    preds = model.predict(preprocess(image))
    confidence = float(np.max(preds))
    predicted_class = CLASS_NAMES[int(np.argmax(preds))]

    if predicted_class == "NON_LEAF" or confidence < CONF_THRESHOLD:
        st.warning("Not a valid plant leaf.")
    else:
        crop, disease = predicted_class.split("___")

        st.write("Crop:", crop)
        st.write("Disease:", disease.replace("_"," "))
        st.write("Confidence:", f"{confidence*100:.2f}%")

        try:
            # ---------- STORAGE ----------
            file_name = f"{uuid.uuid4()}.png"
            image_bytes = uploaded_file.getvalue()

            supabase.storage.from_("leaf-images").upload(
                file_name,
                image_bytes,
                {"content-type": "image/png"}
            )

            image_url = f"{SUPABASE_URL}/storage/v1/object/public/leaf-images/{file_name}"

            # ---------- DIRECT REST INSERT ----------
            headers = {
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {st.session_state['access_token']}",
                "Content-Type": "application/json",
                "Prefer": "return=representation"
            }

            data = {
                "user_email": st.session_state["user_email"],
                "image_url": image_url,
                "disease": disease,
                "confidence": confidence,
                "created_at": datetime.utcnow().isoformat()
            }

            response = requests.post(
                f"{SUPABASE_URL}/rest/v1/predictions",
                json=data,
                headers=headers
            )

            st.write("Insert status:", response.status_code)
            st.write("Insert response:", response.text)

            if response.status_code == 201:
                st.success("Saved to database.")
            else:
                st.error("Insert failed.")

        except Exception as e:
            st.error(f"Insert Error: {e}")

# ---------------- ADMIN DASHBOARD ----------------
st.sidebar.markdown("---")
if st.sidebar.checkbox("Admin Dashboard"):
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {st.session_state.get('access_token','')}"
    }
    response = requests.get(
        f"{SUPABASE_URL}/rest/v1/predictions?select=*",
        headers=headers
    )
    if response.status_code == 200:
        records = response.json()
        st.dataframe(records)
        st.write("Total Records:", len(records))
    else:
        st.error("Failed to fetch records.")
