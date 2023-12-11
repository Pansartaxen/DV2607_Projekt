import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Airport: 0
# Avenue: 1
# Bridge: 2
# Building: 3
# Dense Residential: 4
# FGSM: 5
# Highway: 6
# Marina: 7
# Medium Residential: 8
# Parking Lot: 9
# Residents: 10
# Storeroom: 11

# Function to load and preprocess images
def load_and_preprocess_image(img_path, target_size=(128, 128)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Rescale pixel values to be between 0 and 1
    return img_array


# Load the trained model
model = load_model('model.h5')

# Directory where the attacked images are stored
attacked_images_dir = '../Images/fgsm/airport'

correct = 0
incorrect = 0

# Loop over each image in the directory
for img_name in os.listdir(attacked_images_dir):

    # Full path to the image
    img_path = os.path.join(attacked_images_dir, img_name)

    # Load and preprocess the image
    img_tensor = load_and_preprocess_image(img_path)

    # Predict the class of the image
    predictions = model.predict(img_tensor)

    # Decode the predictions
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    if predicted_class == 0:
        correct += 1
    else:
        incorrect += 1


print(f"Correctly classified: {correct} out of {correct + incorrect}, or {correct / (correct + incorrect) * 100}%")
