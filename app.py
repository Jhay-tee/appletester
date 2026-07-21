# =============================================================================
# GET 324 — Laboratory Exercise 10 (Mini-Project)
# Cloud Computing and AI Model Deployment for Engineering Applications
# Assigned Task (Group CV10): Fresh Apple vs Rotten Apple — Binary Image
# Classification, deployed as an interactive Streamlit web application.
#
# Run locally with:   streamlit run app.py
# =============================================================================

# STEP 1: Import libraries for the web application, data handling, and model loading
import pathlib

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# -----------------------------------------------------------------------------
# STEP 2: Configuration
# -----------------------------------------------------------------------------
# Must match tl_model input size used during training (IMAGE_HEIGHT/IMAGE_WIDTH
# in the training notebook — 224x224 for MobileNetV2).
IMAGE_SIZE = (224, 224)

# Path to the trained model file. By default this looks for model.keras sitting
# in the SAME folder as this app.py file, which is where you should place the
MODEL_PATH = pathlib.Path(__file__).parent / "model.keras"

# Class names IN INDEX ORDER as printed by STEP 10 ("Classes: [...]") in the
# training notebook. image_dataset_from_directory sorts class folders
# alphabetically, so if your dataset's subfolders are named "fresh" and
# "rotten", index 0 = fresh, index 1 = rotten. UPDATE THIS if your folder
# names differ (e.g. "freshapples" / "rottenapples").
CLASS_NAMES = ["Fresh Apple", "Rotten Apple"]

# --- "Is this even an apple?" gatekeeper settings -----------------------------
# Your fresh/rotten model was ONLY ever trained on apple images, so it will
# happily (and wrongly) label a photo of a dog or a car as "Fresh"/"Rotten"
# with high confidence. To catch this, we first run the uploaded image through
# a general-purpose, 1000-class ImageNet MobileNetV2 and check whether an
# apple- or fruit-like object was actually recognised before trusting the
# fresh/rotten prediction.
APPLE_LABELS = {"granny_smith"}
FRUIT_LABELS = {
    "granny_smith", "fig", "pomegranate", "lemon", "orange",
    "strawberry", "pineapple", "banana", "custard_apple", "jackfruit",
}
GATEKEEPER_TOP_K = 5

# -----------------------------------------------------------------------------
# STEP 3: Configure the Streamlit page (title, icon, layout)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Fresh vs Rotten Apple Classifier",
    page_icon="🍎",
    layout="centered",
)

# -----------------------------------------------------------------------------
# STEP 4: Load the saved models (cached so they only load once per session)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(
            f"Model file not found at '{MODEL_PATH}'. "
            "Copy the model.keras file you downloaded from Kaggle into the "
            "same folder as app.py, or update MODEL_PATH at the top of app.py."
        )
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_gatekeeper_model():
    """General-purpose ImageNet model used only to sanity-check that the
    uploaded image actually contains a fruit/apple before we trust the
    fresh-vs-rotten prediction. Downloads its own ImageNet weights on first
    run (needs internet — fine for a deployed Streamlit app)."""
    return tf.keras.applications.MobileNetV2(weights="imagenet")


# -----------------------------------------------------------------------------
# STEP 5: Prediction functions
# -----------------------------------------------------------------------------
def check_is_apple_like(gatekeeper_model, pil_image):
    """Return (status, top_label, confidence) where status is one of:
    'apple'  -> confidently an apple (Granny Smith recognised)
    'fruit'  -> some other fruit recognised, not confidently an apple
    'other'  -> nothing fruit-like recognised in the top-K guesses
    """
    img = pil_image.convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)

    preds = gatekeeper_model.predict(arr, verbose=0)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(
        preds, top=GATEKEEPER_TOP_K
    )[0]  # list of (class_id, label, confidence)

    top_label, top_conf = decoded[0][1], float(decoded[0][2])
    labels_seen = {label.lower() for (_id, label, _conf) in decoded}

    if labels_seen & APPLE_LABELS:
        return "apple", top_label, top_conf
    if labels_seen & FRUIT_LABELS:
        return "fruit", top_label, top_conf
    return "other", top_label, top_conf


def predict(model, pil_image):
    """Run the fresh/rotten model on a PIL image and return
    (label, fresh_pct, rotten_pct)."""
    img = pil_image.convert("RGB").resize(IMAGE_SIZE)
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)  # (1, H, W, 3)

    # Single sigmoid output -> probability of class index 1 (Rotten Apple)
    prob_rotten = float(model.predict(arr, verbose=0)[0][0])
    prob_fresh = 1.0 - prob_rotten

    label = CLASS_NAMES[1] if prob_rotten >= 0.5 else CLASS_NAMES[0]
    return label, prob_fresh * 100, prob_rotten * 100


# -----------------------------------------------------------------------------
# STEP 6: Build the user interface
# -----------------------------------------------------------------------------
st.title("🍎 Fresh vs Rotten Apple Classifier")
st.write(
    "Upload a photo of an apple and this AI model — a MobileNetV2 transfer "
    "learning model trained for GET 324 (Group CV10) — will predict whether "
    "it looks **fresh** or **rotten**."
)

model = load_model()
gatekeeper_model = load_gatekeeper_model()

uploaded_file = st.file_uploader(
    "Upload an apple image", type=["jpg", "jpeg", "png", "webp"]
)

# -----------------------------------------------------------------------------
# STEP 7: Make predictions and display the results
# -----------------------------------------------------------------------------
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, width=300, caption="Uploaded image")

    with st.spinner("Checking image..."):
        gate_status, gate_label, gate_conf = check_is_apple_like(gatekeeper_model, img)

    if gate_status == "other":
        st.error(
            f"⚠️ This doesn't look like an apple. The model's best guess for "
            f"what's actually in the image is **{gate_label.replace('_', ' ')}** "
            f"({gate_conf * 100:.1f}% confidence). Please upload a clear photo "
            f"of an apple — the fresh/rotten classifier below is only "
            f"trained to recognise apples, so its result would not be reliable here."
        )
    else:
        if gate_status == "fruit":
            st.warning(
                f"This looks like it might be **{gate_label.replace('_', ' ')}** "
                f"rather than an apple specifically — the prediction below may "
                f"be less reliable."
            )

        with st.spinner("Classifying..."):
            label, fresh_pct, rotten_pct = predict(model, img)

        st.write(f"### Prediction: **{label}**")

        st.progress(int(round(fresh_pct)), text=f"Fresh: {fresh_pct:.1f}%")
        st.progress(int(round(rotten_pct)), text=f"Rotten: {rotten_pct:.1f}%")

        if label == CLASS_NAMES[0]:
            st.success("This apple looks fresh! 🍏")
        else:
            st.warning("This apple looks rotten. 🍎⚠️")
else:
    st.info("Upload a .jpg, .webp, .jpeg, or .png image of an apple to get a prediction.")

st.divider()
st.caption(
    "GET 324 — Cloud Computing and AI Model Deployment for Engineering "
    "Applications | Group CV10 | Model: MobileNetV2 (Transfer Learning)"
)