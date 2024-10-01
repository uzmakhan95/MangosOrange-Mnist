from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Initialize the Flask app
app = Flask(__name__)

# Load the saved model
try:
    model = tf.keras.models.load_model('model/mnist_cnn_model.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Define home route
@app.route('/')
def home():
    return render_template('index.html')

# Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image from the request
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400

        # Read and preprocess the image
        img = Image.open(io.BytesIO(file.read())).convert('L')  # Convert to grayscale
        img = img.resize((28, 28))  # Resize to 28x28 pixels
        img = np.array(img)  # Convert to numpy array

        # Display the image to verify preprocessing
        print(f"Image shape before reshaping: {img.shape}")
        img = img / 255.0  # Normalize pixel values to [0, 1]
        img = img.reshape(1, 28, 28, 1)  # Reshape to match model input shape
        print(f"Image shape after reshaping: {img.shape}")

        # Make prediction
        prediction = model.predict(img)
        print(f"Raw prediction output: {prediction}")  # Debug: Raw prediction values
        predicted_class = np.argmax(prediction, axis=1)[0]
        print(f"Predicted class: {predicted_class}")

        # Return the prediction
        return jsonify({'prediction': int(predicted_class)})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Prediction failed. Check server logs for details.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
