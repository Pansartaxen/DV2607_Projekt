import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Function to load and preprocess images
def load_and_preprocess_image(img_path, target_size=(128, 128)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Rescale pixel values to be between 0 and 1
    return img_array

# Load the trained model
model = load_model('../Models/cnn_V2.h5')

# Parent directory where the attacked images are stored
parent_dir = '../Images/clean'

# Mapping of directory names to class labels
class_labels = {
    "airport": 0,
    "avenue": 1,
    "bridge": 2,
    "building": 3,
    "denseresidential": 4,
    "highway": 5,
    "marina": 6,
    "mediumresidential": 7,
    "parkinglot": 8,
    "residents": 9,
    "storeroom": 10
}
total_correct = 0
total_incorrect = 0
# Loop over each subdirectory in the parent directory
for subdir in os.listdir(parent_dir):
    if subdir.lower() in class_labels:
        subdir_path = os.path.join(parent_dir, subdir)
        correct = 0
        incorrect = 0

        # Loop over each image in the subdirectory
        for img_name in os.listdir(subdir_path):
            # Full path to the image
            img_path = os.path.join(subdir_path, img_name)

            # Load and preprocess the image
            img_tensor = load_and_preprocess_image(img_path)

            # Predict the class of the image
            predictions = model.predict(img_tensor, verbose=0)

            # Decode the predictions
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])

            # Compare predicted class with actual class
            if predicted_class == class_labels[subdir.lower()]:
                correct += 1
                total_correct += 1
            else:
                incorrect += 1
                total_incorrect += 1

        # Print the accuracy for the current subdirectory
        total = correct + incorrect
        if total > 0:
            print(f"In {subdir}: Correctly classified {correct} out of {total}, or {round((correct / total * 100),2)}%")

print(f"Total: Correctly classified: {total_correct} out of {total_correct + total_incorrect}, or {round((total_correct / (total_correct + total_incorrect) * 100),2)}%")