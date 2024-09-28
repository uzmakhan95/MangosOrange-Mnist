import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image

# Load your trained model
model = load_model('bestmodel.keras')

# Function to preprocess the image
def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Streamlit app
st.title("MNIST Digit Classification")
uploaded_file = st.file_uploader("Choose a PNG or JPG image...", type=["png", "jpg"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file)
    processed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)

    # Display the image and prediction
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write(f"Predicted Digit: {predicted_class}")
