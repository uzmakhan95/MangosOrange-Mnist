import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_test = x_test / 255.0  # Normalize pixel values to [0, 1]
x_test = x_test[..., np.newaxis]  # Reshape to match model input shape (N, 28, 28, 1)

# Load the trained model
try:
    model = tf.keras.models.load_model('model/mnist_cnn_model.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Make predictions on the first 10 test images
predictions = model.predict(x_test[:10])
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(tf.keras.utils.to_categorical(y_test[:10], 10), axis=1)

print("Predicted classes:", predicted_classes)
print("True classes:", true_classes)

# Visualize the first 10 test images and their predictions
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Pred: {predicted_classes[i]}, True: {true_classes[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
