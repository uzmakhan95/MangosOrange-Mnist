import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

model = tf.keras.models.load_model('hdrcnn_model.h5')

def preprocess_image(image):

    image = ImageOps.grayscale(image)
    image = ImageOps.fit(image, (28, 28), Image.Resampling.LANCZOS)
    image_array = np.array(image)
    image_array = image_array.astype('float32') / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)
    
    return image_array

st.title("Handwritten Digit Recognition")
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:

        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        predicted_digit = np.argmax(predictions)
        st.write(f"Predicted Digit: {predicted_digit}")
        
    except Exception as e:

        st.error(f"Error processing image: {e}")
