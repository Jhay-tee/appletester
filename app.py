import pathlib

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

IMAGE_SIZE = (224, 224)
MODEL_PATH = pathlib.Path(__file__).parent / "model.keras"
CLASS_NAMES = ["Fresh Apple", "Rotten Apple"]

ACCEPTED_TYPES = ["jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif", "gif"]

APPLE_LABELS = {"granny_smith"}
FRUIT_LABELS = {
    "granny_smith", "fig", "pomegranate", "lemon", "orange",
    "strawberry", "pineapple", "banana", "custard_apple", "jackfruit",
}
GATEKEEPER_TOP_K = 5

st.set_page_config(page_title="Fresh vs Rotten Apple Classifier", page_icon="🍎", layout="centered")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"model.keras not found at {MODEL_PATH}. Place it next to app.py.")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_gatekeeper_model():
    return tf.keras.applications.MobileNetV2(weights="imagenet")


def check_is_apple_like(gatekeeper_model, pil_image):
    img = pil_image.convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)

    preds = gatekeeper_model.predict(arr, verbose=0)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=GATEKEEPER_TOP_K)[0]

    top_label, top_conf = decoded[0][1], float(decoded[0][2])
    labels_seen = {label.lower() for (_id, label, _conf) in decoded}

    if labels_seen & APPLE_LABELS:
        return "apple", top_label, top_conf
    if labels_seen & FRUIT_LABELS:
        return "fruit", top_label, top_conf
    return "other", top_label, top_conf


def predict(model, pil_image):
    img = pil_image.convert("RGB").resize(IMAGE_SIZE)
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)

    prob_rotten = float(model.predict(arr, verbose=0)[0][0])
    prob_fresh = 1.0 - prob_rotten

    label = CLASS_NAMES[1] if prob_rotten >= 0.5 else CLASS_NAMES[0]
    return label, prob_fresh * 100, prob_rotten * 100


st.title("🍎 Fresh vs Rotten Apple Classifier")
st.write("Upload a photo of an apple to check whether it's fresh or rotten.")

model = load_model()
gatekeeper_model = load_gatekeeper_model()

uploaded_file = st.file_uploader("Upload an apple image", type=ACCEPTED_TYPES)

if uploaded_file:
    try:
        img = Image.open(uploaded_file)
        img.load()
    except UnidentifiedImageError:
        st.error("Couldn't read that file as an image. Try a different photo.")
        st.stop()

    st.image(img, width=300, caption="Uploaded image")

    with st.spinner("Checking image..."):
        gate_status, gate_label, gate_conf = check_is_apple_like(gatekeeper_model, img)

    with st.spinner("Classifying..."):
        label, fresh_pct, rotten_pct = predict(model, img)

    if gate_status == "other":
        st.caption(
            f"Note: this doesn't look like a typical apple photo (closest guess: "
            f"{gate_label.replace('_', ' ')}, {gate_conf * 100:.1f}%). If it's a heavily "
            f"decayed apple, ignore this note — the result below still applies."
        )
    elif gate_status == "fruit":
        st.caption(
            f"Note: this looks like it might be {gate_label.replace('_', ' ')} "
            f"rather than an apple specifically."
        )

    st.write(f"### Prediction: **{label}**")
    st.progress(int(round(fresh_pct)), text=f"Fresh: {fresh_pct:.1f}%")
    st.progress(int(round(rotten_pct)), text=f"Rotten: {rotten_pct:.1f}%")

    if label == CLASS_NAMES[0]:
        st.success("This apple looks fresh.")
    else:
        st.warning("This apple looks rotten.")
else:
    st.info("Upload a .jpg, .jpeg, or .png image of an apple to get a prediction.")

st.divider()
st.caption("GET 324 | Group CV10 | MobileNetV2 transfer learning")