import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image, ImageOps

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="centered"
)

# --------------------------------------------------
# Load saved model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_model.keras")


model = load_model()

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🔢 MNIST Handwritten Digit Classification")

st.write(
    "Upload an image of a handwritten digit "
    "and the trained neural network will predict the digit."
)

st.divider()

# --------------------------------------------------
# Upload image
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a handwritten digit image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if uploaded_file is not None:

    # Open image
    original_image = Image.open(uploaded_file)

    # Convert to grayscale
    image = original_image.convert("L")

    # Display uploaded image
    st.subheader("Uploaded Image")

    st.image(
        image,
        caption="Input Image",
        width=250
    )

    # --------------------------------------------------
    # Resize image to MNIST dimensions
    # --------------------------------------------------

    image = image.resize((28, 28))

    # --------------------------------------------------
    # Convert image to numpy array
    # --------------------------------------------------

    image_array = np.array(image)

    # --------------------------------------------------
    # Invert image
    #
    # MNIST normally uses:
    # black background + white digit
    # --------------------------------------------------

    image_array = ImageOps.invert(image)

    image_array = np.array(image_array)

    # --------------------------------------------------
    # Normalize
    # Same preprocessing as training
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
    # Get predicted digit
    # --------------------------------------------------

    predicted_digit = np.argmax(
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
        f"Predicted Digit: {predicted_digit}"
    )

    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )

    # --------------------------------------------------
    # Display probabilities
    # --------------------------------------------------

    st.subheader("Prediction Probabilities")

    probabilities = prediction[0]

    for digit in range(10):

        st.write(
            f"Digit {digit}: "
            f"{probabilities[digit] * 100:.2f}%"
        )

        st.progress(
            float(probabilities[digit])
        )