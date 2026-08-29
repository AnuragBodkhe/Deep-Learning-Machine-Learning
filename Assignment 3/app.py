# ============================================================
# CAT vs DOG CNN CLASSIFICATION GUI
# ============================================================

import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐱",
    layout="centered"
)


# ============================================================
# 2. MODEL PATH
# ============================================================

MODEL_PATH = r"D:\Deep Learning\Assignment 3\cat_dog_cnn_best.keras"


# ============================================================
# 3. LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_trained_model():

    if not os.path.exists(MODEL_PATH):
        st.error(
            "Model file not found!\n\n"
            f"Expected location:\n{MODEL_PATH}"
        )
        st.stop()

    model = tf.keras.models.load_model(MODEL_PATH)

    return model


model = load_trained_model()


# ============================================================
# 4. PREDICTION FUNCTION
# ============================================================

def predict_image(image):

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize to model input size
    image = image.resize((64, 64))

    # Convert image to NumPy array
    image_array = np.array(image)

    # Normalize pixel values
    image_array = image_array.astype("float32") / 255.0

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Make prediction
    prediction = model.predict(
        image_array,
        verbose=0
    )[0][0]

    # Determine class
    if prediction >= 0.5:

        label = "DOG"
        confidence = prediction

    else:

        label = "CAT"
        confidence = 1 - prediction

    return label, confidence


# ============================================================
# 5. APPLICATION TITLE
# ============================================================

st.title("🐱 CAT vs DOG PREDICTION 🐶")

st.write(
    "Upload an image of a cat or dog "
    "to classify it using a Convolutional Neural Network (CNN)."
)


# ============================================================
# 6. MODEL INFORMATION
# ============================================================

st.write("---")

st.subheader("🧠 CNN Model")

st.write(
    "The uploaded image is resized to 64 × 64 pixels "
    "and normalized before classification."
)


# ============================================================
# 7. IMAGE UPLOADER
# ============================================================

st.write("---")

uploaded_file = st.file_uploader(
    "📁 Choose an image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# 8. IMAGE CLASSIFICATION
# ============================================================

if uploaded_file is not None:

    # Open uploaded image
    image = Image.open(uploaded_file)

    # Display uploaded image
    st.subheader("📷 Uploaded Image")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Classify button
    if st.button(
        "🔍 Classify Image",
        use_container_width=True
    ):

        with st.spinner("Classifying image..."):

            label, confidence = predict_image(image)

        st.write("---")

        # ====================================================
        # 9. DISPLAY RESULT
        # ====================================================

        st.subheader("📊 Classification Result")

        if label == "DOG":

            st.success(
                f"🐶 Prediction: {label}"
            )

        else:

            st.success(
                f"🐱 Prediction: {label}"
            )

        st.info(
            f"Confidence: {confidence * 100:.2f}%"
        )

        # ====================================================
        # 10. CONFIDENCE PROGRESS BAR
        # ====================================================

        st.write("### Confidence")

        st.progress(
            float(confidence)
        )

        # ====================================================
        # 11. RESULT INTERPRETATION
        # ====================================================

        if confidence >= 0.90:

            st.write(
                "✅ The model is highly confident in this prediction."
            )

        elif confidence >= 0.70:

            st.write(
                "👍 The model has good confidence in this prediction."
            )

        elif confidence >= 0.50:

            st.write(
                "⚠️ The model has moderate confidence. "
                "The image may be difficult to classify."
            )

        else:

            st.write(
                "⚠️ The model has low confidence in this prediction."
            )


# ============================================================
# 12. FOOTER
# ============================================================

st.write("---")

st.caption(
    "Cat vs Dog Image Classification using CNN | "
    "Deep Learning Assignment"
)