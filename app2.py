import pygame
import sys
import numpy as np
from keras.models import load_model
import cv2

# Constants
WINDOW_SIZE_X = 640
WINDOW_SIZE_Y = 480
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Load the trained model
MODEL = load_model("bestmodel.keras")

# Labels for predictions
LABELS = {
    0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four",
    5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"
}

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE_X, WINDOW_SIZE_Y))
pygame.display.set_caption("Digit Board")

# Create a blank surface for drawing
draw_surface = pygame.Surface((WINDOW_SIZE_X, WINDOW_SIZE_Y))
draw_surface.fill(WHITE)

# Function to preprocess the drawn image for prediction
def preprocess_image(image):
    # Check if the image has three channels
    if image.shape[2] == 3:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image[:, :, 0]  # Use the single channel directly if already grayscale

    # Resize to 28x28 pixels
    resized = cv2.resize(gray, (28, 28))
    
    # Invert colors: Make background black and digit white
    inverted = cv2.bitwise_not(resized)

    # Normalize the image data
    normalized = inverted / 255.0

    # Reshape for the model input (1, 28, 28, 1)
    reshaped = normalized.reshape(1, 28, 28, 1)
    
    return reshaped

# Initialize writing state
is_writing = False
predicted_digits = []  # Store all predicted digits

# Main loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.MOUSEMOTION and is_writing:
            xcord, ycord = event.pos
            pygame.draw.circle(draw_surface, BLACK, (xcord, ycord), 4)

        if event.type == pygame.MOUSEBUTTONDOWN:
            is_writing = True

        if event.type == pygame.MOUSEBUTTONUP:
            is_writing = False
            
            # Convert Pygame surface to a NumPy array
            image = pygame.surfarray.array3d(draw_surface)
            image = np.moveaxis(image, 0, -1)  # Change axis for (H, W, C)
            
            # Preprocess the image
            processed_image = preprocess_image(image)

            # Make prediction
            prediction = MODEL.predict(processed_image)
            predicted_digit = np.argmax(prediction)  # Get the predicted digit
            predicted_digits.append(predicted_digit)  # Append to list of predictions
            
            # Print the predicted digit
            print(f"The predicted digit is: {predicted_digit}")

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:  # Clear the drawing
                draw_surface.fill(WHITE)
                predicted_digits.clear()  # Reset the predicted digits

    # Fill the screen with white and blit the draw surface
    screen.fill(WHITE)
    screen.blit(draw_surface, (0, 0))

    # Optional: Display predicted digits
    if predicted_digits:  # Only render text if there are predictions
        font = pygame.font.SysFont(None, 55)
        for i, digit in enumerate(predicted_digits):
            text = font.render(f"Predicted Digit {i + 1}: {digit}", True, BLACK)
            screen.blit(text, (20, 20 + i * 30))  # Adjust y position for each digit

    pygame.display.flip()
