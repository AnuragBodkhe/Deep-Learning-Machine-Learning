import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🖼️",
    layout="centered"
)

# --------------------------------------------------
# Load saved model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cifar10_mlp_model.keras")


model = load_model()

# --------------------------------------------------
# CIFAR-10 class names
# --------------------------------------------------

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🖼️ CIFAR-10 Image Classification")

st.write(
    "Upload an image and the trained neural network "
    "will predict its CIFAR-10 class."
)

st.divider()

# --------------------------------------------------
# Upload image
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # Display original image
    st.subheader("Uploaded Image")

    st.image(
        image,
        caption="Input Image",
        width=300
    )

    # --------------------------------------------------
    # Resize to CIFAR-10 size
    # --------------------------------------------------

    image_resized = image.resize((32, 32))

    # --------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------

    image_array = np.array(image_resized)

    # --------------------------------------------------
    # Normalize
    # --------------------------------------------------

    image_array = image_array.astype("float32") / 255.0

    # --------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    prediction = model.predict(
        image_array,
        verbose=0
    )

    # --------------------------------------------------
    # Predicted class
    # --------------------------------------------------

    predicted_class = np.argmax(
        prediction[0]
    )

    confidence = np.max(
        prediction[0]
    ) * 100

    # --------------------------------------------------
    # Display result
    # --------------------------------------------------

    st.divider()

    st.subheader("Prediction")

    st.success(
        f"Predicted Class: {class_names[predicted_class]}"
    )

    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )

    # --------------------------------------------------
    # Display all probabilities
    # --------------------------------------------------

    st.subheader("Class Probabilities")

    for i in range(10):

        probability = float(prediction[0][i])

        st.write(
            f"{class_names[i]}: "
            f"{probability * 100:.2f}%"
        )

        st.progress(probability)